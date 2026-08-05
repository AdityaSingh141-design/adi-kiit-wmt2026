"""
=============================================================================
Script 06-BASE — DPO directly on the BASE model (NO SFT)
=============================================================================
Tests the hypothesis (from the SFT-only diagnostic) that SFT causes the
degradation. Here we SKIP SFT entirely: a fresh LoRA adapter is initialized
on the base model and trained with DPO, anchored to the frozen BASE model
as the reference.

  policy    = base + fresh LoRA (trainable, zero-init B => starts == base)
  reference = base (frozen, zero-shot)

If DPO is corrective and SFT is harmful, this should beat both SFT+DPO and
potentially approach/exceed the zero-shot baseline.

Reads the base-model scored candidates (rl_scored_base.csv).

Usage:
    python3 06_dpo_on_base.py --config ../data2/config_v7.yaml
=============================================================================
"""
import os, sys, ast, logging, argparse, yaml, torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--scored_file", type=str, default=None,
                   help="Default: cfg[scored_file] with rl_scored_base.csv")
    return p.parse_args()


def setup_logger(lp):
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lp), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)


class PairDataset(Dataset):
    def __init__(self, csv, tok, ip, cfg, logger):
        df = pd.read_csv(csv)
        for c in ["candidates", "rewards"]:
            if isinstance(df[c].iloc[0], str):
                df[c] = df[c].apply(ast.literal_eval)
        prompts, chos, rej = [], [], []
        skip = 0
        for _, r in df.iterrows():
            cs, rw = r["candidates"], r["rewards"]
            if len(cs) < 2 or len(rw) < 2:
                skip += 1; continue
            c0, c1 = str(cs[0]).strip(), str(cs[1]).strip()
            if not c0 or not c1 or c0 == c1 or c0 == "nan" or c1 == "nan":
                skip += 1; continue
            r0, r1 = float(rw[0]), float(rw[1])
            prompts.append(r["src"])
            chos.append(c0 if r0 >= r1 else c1)
            rej.append(c1 if r0 >= r1 else c0)
        logger.info(f"Valid pairs: {len(prompts)} (skipped {skip})")

        mp, mt = cfg["dpo_max_prompt_length"], cfg["dpo_max_target_length"]
        sp = ip.preprocess_batch(prompts, src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"])
        pe = tok(sp, max_length=mp, truncation=True, padding="max_length", return_tensors="pt")
        cp = ip.preprocess_batch(chos, src_lang=cfg["tgt_lang"], tgt_lang=cfg["tgt_lang"])
        ce = tok(text_target=cp, max_length=mt, truncation=True, padding="max_length", return_tensors="pt")
        rp = ip.preprocess_batch(rej, src_lang=cfg["tgt_lang"], tgt_lang=cfg["tgt_lang"])
        re = tok(text_target=rp, max_length=mt, truncation=True, padding="max_length", return_tensors="pt")
        self.pid, self.pm = pe["input_ids"], pe["attention_mask"]
        self.cid, self.cm = ce["input_ids"], ce["attention_mask"]
        self.rid, self.rm = re["input_ids"], re["attention_mask"]

    def __len__(self): return len(self.pid)
    def __getitem__(self, i):
        return {"pid": self.pid[i], "pm": self.pm[i],
                "cid": self.cid[i], "cm": self.cm[i],
                "rid": self.rid[i], "rm": self.rm[i]}


def logp(model, pid, pm, did, dm, grad=False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        o = model(input_ids=pid, attention_mask=pm,
                  decoder_input_ids=did, decoder_attention_mask=dm)
    sl = o.logits[:, :-1, :].contiguous()
    lab = did[:, 1:].contiguous()
    msk = dm[:, 1:].contiguous().float()
    lp = F.log_softmax(sl, -1).gather(2, lab.unsqueeze(-1)).squeeze(-1) * msk
    return lp.sum(-1) / msk.sum(-1).clamp(min=1.0)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    scored = args.scored_file or cfg["scored_file"].replace(
        "rl_scored.csv", "rl_scored_base.csv")

    exp = cfg["experiment_name"]
    run = f"{exp}-dpoonbase"
    out_dir = cfg["dpo_adapter_dir"].replace(exp, run)
    log_dir = cfg["log_dir"].replace(exp, run)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "06_dpo_on_base.log"))

    logger.info("=" * 60)
    logger.info("DPO ON BASE (no SFT) — fresh LoRA, reference = base")
    logger.info(f"Run: {run}")
    logger.info(f"Scored: {scored}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tok.padding_side = "left"
    ip = IndicProcessor(inference=True)

    ds = PairDataset(scored, tok, ip, cfg, logger)
    loader = DataLoader(ds, batch_size=cfg["dpo_batch_size"], shuffle=True)

    # Policy: base + FRESH LoRA
    logger.info("Loading base + fresh LoRA (policy)...")
    b1 = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"],
        torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    if not hasattr(b1.config, "hidden_size"):
        b1.config.hidden_size = b1.config.decoder_embed_dim
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        task_type=TaskType.SEQ_2_SEQ_LM)
    policy = get_peft_model(b1, lora).to(DEVICE)
    policy.train()

    # Reference: frozen base (zero-shot)
    logger.info("Loading frozen base (reference = zero-shot)...")
    b2 = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"],
        torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    if not hasattr(b2.config, "hidden_size"):
        b2.config.hidden_size = b2.config.decoder_embed_dim
    ref = b2.to(DEVICE)
    for p in ref.parameters(): p.requires_grad = False
    ref.eval()

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=float(cfg["dpo_learning_rate"]))
    beta = cfg["dpo_beta"]

    logger.info("Training DPO on base...")
    for ep in range(cfg["dpo_epochs"]):
        L, A = [], []
        for st, b in enumerate(loader):
            pid, pm = b["pid"].to(DEVICE), b["pm"].to(DEVICE)
            cid, cm = b["cid"].to(DEVICE), b["cm"].to(DEVICE)
            rid, rm = b["rid"].to(DEVICE), b["rm"].to(DEVICE)
            pc = logp(policy, pid, pm, cid, cm, grad=True)
            pr = logp(policy, pid, pm, rid, rm, grad=True)
            rc = logp(ref, pid, pm, cid, cm, grad=False)
            rr = logp(ref, pid, pm, rid, rm, grad=False)
            loss = -F.logsigmoid(beta*((pc-rc)-(pr-rr))).mean()
            acc = ((pc-rc) > (pr-rr)).float().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()
            L.append(loss.item()); A.append(acc.item())
            if (st+1) % 50 == 0 or (st+1) == len(loader):
                logger.info(f"  E{ep+1} S{st+1}/{len(loader)} "
                            f"loss={loss.item():.4f} acc={acc.item():.3f}")
        logger.info(f"Epoch {ep+1} done | loss={np.mean(L):.4f} acc={np.mean(A):.3f}")

    policy.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    logger.info(f"Saved -> {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()