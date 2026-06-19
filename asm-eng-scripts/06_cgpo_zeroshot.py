"""
=============================================================================
Script 06-CG — Competence-Gated Preference Optimization (CGPO)
English -> Assamese RLFT Pipeline | IIT Patna
=============================================================================
NOVEL METHOD (publishable core).

Motivation
----------
Standard DPO applies a uniform-strength preference update to every example.
We observe (across Hin->Mai, Eng->Asm, Asm->Eng) that this DEGRADES
directions where the base model is already competent: optimizing preference
on an already-good example pushes the policy off its good optimum.

CGPO conditions the update on per-example *base-model competence*:
    weight_i = (1 - competence_i) ** gamma
    loss     = - mean_i [ weight_i * log sigmoid( beta * (r_chosen - r_rejected) ) ]

where competence_i in [0,1] is the reward model's assessment of how good the
model's output already is for example i. High-competence examples contribute
near-zero gradient; the policy is left alone where it is already strong.

Design choices that make this a CLEAN contribution
--------------------------------------------------
1. Weights are normalized to mean 1 over the dataset, so total gradient
   magnitude matches standard DPO. This DECOUPLES the method from a change
   in effective learning rate (a confound reviewers will check).
2. gamma = 0  ==>  all weights = 1  ==>  EXACTLY standard DPO (ablation base).
   gamma = 1  ==>  linear down-weighting by competence.
   gamma > 1  ==>  sharper gating (only the weakest examples get signal).
3. competence source is configurable (max | mean of candidate rewards).
4. Per-example weights and competence are saved to CSV for analysis/plots.

Usage
-----
    # Standard DPO (sanity check — should match 06_dpo_training.py)
    python3 06_competence_gated_dpo.py --config ../data2/config_v7.yaml --gamma 0

    # Competence-gated (recommended starting point)
    python3 06_competence_gated_dpo.py --config ../data2/config_v7.yaml --gamma 1.0

    # Sharper gating
    python3 06_competence_gated_dpo.py --config ../data2/config_v7.yaml --gamma 2.0
=============================================================================
"""

import os
import sys
import ast
import logging
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gating sharpness. 0=standard DPO, 1=linear, >1=sharper.")
    p.add_argument("--competence", type=str, default="zeroshot",
                   choices=["max", "mean", "zeroshot"],
                   help="Competence source. zeroshot = reward of base-model "
                        "output (read from zeroshot_competence column).")
    p.add_argument("--scored_file", type=str, default=None,
                   help="Override scored file (default: cfg[scored_file]; "
                        "use rl_scored_zscomp.csv for zeroshot competence).")
    p.add_argument("--min_gap", type=float, default=0.0,
                   help="Drop pairs whose reward gap < min_gap (0 keeps all).")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for output dir. Default: cgpo-g{gamma}.")
    return p.parse_args()


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


class CGPODataset(Dataset):
    """DPO dataset that also carries a per-example training weight."""
    def __init__(self, scored_csv, tokenizer, ip, cfg, logger,
                 gamma=1.0, competence="max", min_gap=0.0, weight_dump=None):
        df = pd.read_csv(scored_csv)
        for col in ["candidates", "rewards"]:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(ast.literal_eval)
        logger.info(f"Loaded {len(df)} scored rows")

        rows = []
        skipped = 0
        for _, row in df.iterrows():
            cands, rewards = row["candidates"], row["rewards"]
            if len(cands) < 2 or len(rewards) < 2:
                skipped += 1; continue
            c0, c1 = str(cands[0]).strip(), str(cands[1]).strip()
            r0, r1 = float(rewards[0]), float(rewards[1])
            if not c0 or not c1 or c0 == "nan" or c1 == "nan" or c0 == c1:
                skipped += 1; continue
            gap = abs(r0 - r1)
            if gap < min_gap:
                skipped += 1; continue
            if competence == "zeroshot":
                comp = float(row["zeroshot_competence"])
            elif competence == "max":
                comp = max(r0, r1)
            else:
                comp = 0.5 * (r0 + r1)
            rows.append({
                "src": row["src"],
                "chosen":   c0 if r0 >= r1 else c1,
                "rejected": c1 if r0 >= r1 else c0,
                "competence": float(np.clip(comp, 0.0, 1.0)),
                "gap": gap,
            })
        logger.info(f"Valid pairs: {len(rows)} (skipped {skipped})")
        if not rows:
            logger.error("No valid pairs. Aborting."); sys.exit(1)

        # ---- competence -> raw weight ------------------------------------
        comps = np.array([r["competence"] for r in rows])
        raw_w = np.power(1.0 - comps, gamma)        # gamma=0 -> all ones

        # ---- normalize weights to mean 1 (decouple from effective LR) ----
        mean_w = raw_w.mean()
        if mean_w <= 1e-8:
            logger.warning("All weights ~0 (every example highly competent). "
                           "Falling back to uniform weights.")
            norm_w = np.ones_like(raw_w)
        else:
            norm_w = raw_w / mean_w
        for r, w in zip(rows, norm_w):
            r["weight"] = float(w)

        logger.info("=" * 55)
        logger.info(f"CGPO gating | gamma={gamma} | competence={competence}")
        logger.info(f"  competence : mean={comps.mean():.4f} "
                    f"median={np.median(comps):.4f} "
                    f"min={comps.min():.4f} max={comps.max():.4f}")
        logger.info(f"  weight(norm): mean={norm_w.mean():.4f} "
                    f"median={np.median(norm_w):.4f} "
                    f"min={norm_w.min():.4f} max={norm_w.max():.4f}")
        eff = (norm_w > 0.1).sum()
        logger.info(f"  examples with weight>0.1 : {eff}/{len(norm_w)} "
                    f"({100*eff/len(norm_w):.1f}%)")
        logger.info("=" * 55)

        # ---- dump weights for analysis/plots -----------------------------
        if weight_dump:
            pd.DataFrame({
                "src":        [r["src"] for r in rows],
                "competence": comps,
                "weight":     norm_w,
                "gap":        [r["gap"] for r in rows],
            }).to_csv(weight_dump, index=False)
            logger.info(f"  Per-example weights saved: {weight_dump}")

        # ---- tokenize ----------------------------------------------------
        max_src = cfg["dpo_max_prompt_length"]
        max_tgt = cfg["dpo_max_target_length"]
        prompts   = [r["src"] for r in rows]
        chosens   = [r["chosen"] for r in rows]
        rejecteds = [r["rejected"] for r in rows]

        logger.info("Tokenizing source / chosen / rejected ...")
        src_p = ip.preprocess_batch(prompts, src_lang=cfg["src_lang"],
                                    tgt_lang=cfg["tgt_lang"])
        prm = tokenizer(src_p, max_length=max_src, truncation=True,
                        padding="max_length", return_tensors="pt")
        ch_p = ip.preprocess_batch(chosens, src_lang=cfg["tgt_lang"],
                                   tgt_lang=cfg["tgt_lang"])
        ch = tokenizer(text_target=ch_p, max_length=max_tgt, truncation=True,
                       padding="max_length", return_tensors="pt")
        rj_p = ip.preprocess_batch(rejecteds, src_lang=cfg["tgt_lang"],
                                   tgt_lang=cfg["tgt_lang"])
        rj = tokenizer(text_target=rj_p, max_length=max_tgt, truncation=True,
                       padding="max_length", return_tensors="pt")

        self.prompt_input_ids        = prm["input_ids"]
        self.prompt_attention_mask   = prm["attention_mask"]
        self.chosen_input_ids        = ch["input_ids"]
        self.chosen_attention_mask   = ch["attention_mask"]
        self.rejected_input_ids      = rj["input_ids"]
        self.rejected_attention_mask = rj["attention_mask"]
        self.weights = torch.tensor([r["weight"] for r in rows],
                                    dtype=torch.float32)
        logger.info(f"Dataset built: {len(self.weights)} weighted pairs")

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, i):
        return {
            "prompt_input_ids":        self.prompt_input_ids[i],
            "prompt_attention_mask":   self.prompt_attention_mask[i],
            "chosen_input_ids":        self.chosen_input_ids[i],
            "chosen_attention_mask":   self.chosen_attention_mask[i],
            "rejected_input_ids":      self.rejected_input_ids[i],
            "rejected_attention_mask": self.rejected_attention_mask[i],
            "weight":                  self.weights[i],
        }


def get_logprobs(model, input_ids, attention_mask,
                 dec_ids, dec_mask, require_grad=False):
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=dec_ids,
                    decoder_attention_mask=dec_mask)
    logits = out.logits
    sl = logits[:, :-1, :].contiguous()
    lab = dec_ids[:, 1:].contiguous()
    msk = dec_mask[:, 1:].contiguous().float()
    lp  = F.log_softmax(sl, dim=-1)
    tok = lp.gather(2, lab.unsqueeze(-1)).squeeze(-1) * msk
    return tok.sum(-1) / msk.sum(-1).clamp(min=1.0)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.scored_file:
        cfg["scored_file"] = args.scored_file
    elif args.competence == "zeroshot":
        cfg["scored_file"] = cfg["scored_file"].replace(
            "rl_scored.csv", "rl_scored_zscomp.csv")

    exp = cfg["experiment_name"]
    tag = args.tag or f"cgpozs-g{args.gamma}"
    run_name = f"{exp}-{tag}"

    dpo_adapter_dir = cfg["dpo_adapter_dir"].replace(exp, run_name)
    log_dir = cfg["log_dir"].replace(exp, run_name)
    os.makedirs(dpo_adapter_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    weight_dump = os.path.join(log_dir, "cgpo_weights.csv")

    logger = setup_logger(os.path.join(log_dir, "06_cgpo.log"))
    logger.info("=" * 60)
    logger.info("Phase 5-CG: Competence-Gated DPO (CGPO)")
    logger.info(f"Run name    : {run_name}")
    logger.info(f"gamma       : {args.gamma}  (0 = standard DPO)")
    logger.info(f"competence  : {args.competence}")
    logger.info(f"Scored file : {cfg['scored_file']}")
    logger.info(f"SFT adapter : {cfg['sft_adapter_dir']}")
    logger.info(f"Output      : {dpo_adapter_dir}")
    logger.info(f"beta        : {cfg['dpo_beta']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    logger.info(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],
                                              trust_remote_code=True)
    tokenizer.padding_side = "left"
    ip = IndicProcessor(inference=True)

    ds = CGPODataset(cfg["scored_file"], tokenizer, ip, cfg, logger,
                     gamma=args.gamma, competence=args.competence,
                     min_gap=args.min_gap, weight_dump=weight_dump)
    loader = DataLoader(ds, batch_size=cfg["dpo_batch_size"],
                        shuffle=True, drop_last=False)

    logger.info("Loading policy (SFT LoRA) ...")
    b1 = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True)
    if not hasattr(b1.config, "hidden_size"):
        b1.config.hidden_size = b1.config.decoder_embed_dim
    policy = PeftModel.from_pretrained(b1, cfg["sft_adapter_dir"]).to(DEVICE)
    for n, p in policy.named_parameters():
        p.requires_grad = "lora" in n.lower()
    policy.train()

    logger.info("Loading frozen reference (SFT) ...")
    b2 = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True)
    if not hasattr(b2.config, "hidden_size"):
        b2.config.hidden_size = b2.config.decoder_embed_dim
    ref = PeftModel.from_pretrained(b2, cfg["sft_adapter_dir"]).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=float(cfg["dpo_learning_rate"]))
    beta = cfg["dpo_beta"]

    logger.info("Starting CGPO training ...")
    for epoch in range(cfg["dpo_epochs"]):
        losses, accs = [], []
        for step, batch in enumerate(loader):
            pid  = batch["prompt_input_ids"].to(DEVICE)
            pmsk = batch["prompt_attention_mask"].to(DEVICE)
            cid  = batch["chosen_input_ids"].to(DEVICE)
            cmsk = batch["chosen_attention_mask"].to(DEVICE)
            rid  = batch["rejected_input_ids"].to(DEVICE)
            rmsk = batch["rejected_attention_mask"].to(DEVICE)
            w    = batch["weight"].to(DEVICE)

            pc = get_logprobs(policy, pid, pmsk, cid, cmsk, require_grad=True)
            pr = get_logprobs(policy, pid, pmsk, rid, rmsk, require_grad=True)
            rc = get_logprobs(ref,    pid, pmsk, cid, cmsk, require_grad=False)
            rr = get_logprobs(ref,    pid, pmsk, rid, rmsk, require_grad=False)

            chosen_rw   = beta * (pc - rc)
            rejected_rw = beta * (pr - rr)
            # per-example weighted DPO loss
            per_ex = -F.logsigmoid(chosen_rw - rejected_rw)
            loss   = (w * per_ex).mean()
            acc    = (chosen_rw > rejected_rw).float().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()

            losses.append(loss.item()); accs.append(acc.item())
            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                logger.info(f"  E{epoch+1}/{cfg['dpo_epochs']} "
                            f"S{step+1}/{len(loader)} | "
                            f"loss={loss.item():.4f} acc={acc.item():.3f} "
                            f"w_mean={w.mean().item():.3f}")
        logger.info(f"Epoch {epoch+1} done | loss={np.mean(losses):.4f} "
                    f"acc={np.mean(accs):.3f}")

    logger.info(f"Saving adapter -> {dpo_adapter_dir}")
    policy.save_pretrained(dpo_adapter_dir)
    tokenizer.save_pretrained(dpo_adapter_dir)
    logger.info("CGPO complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()