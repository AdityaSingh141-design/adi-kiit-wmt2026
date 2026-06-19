"""
=============================================================================
Script 04-BASE — Diverse Candidate Generation from the BASE model (no SFT)
=============================================================================
For the DPO-on-base experiment. Generates greedy + sampled candidates
DIRECTLY from the zero-shot base model (NO adapter), for the same RL
training sentences, so DPO can be run on the base model without SFT.

Reuses the exact RL sentences from rl_scored.csv for perfect alignment.

Output: candidates_file -> rl_candidates_base.csv

Usage:
    python3 04_generate_candidates_base.py --config ../data2/config_v7.yaml
=============================================================================
"""
import os, sys, logging, argparse, yaml, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)


def gen(model, tok, ip, batch, src_lang, tgt_lang, max_len, max_new,
        device, sample=False, temperature=0.8, top_p=0.9, seed=None):
    proc = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    enc = tok(proc, return_tensors="pt", padding=True,
              truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    if sample and seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        if sample:
            out = model.generate(**enc, max_new_tokens=max_new, num_beams=1,
                                 num_return_sequences=1, do_sample=True,
                                 temperature=temperature, top_p=top_p, top_k=0)
        else:
            out = model.generate(**enc, max_new_tokens=max_new, num_beams=1,
                                 num_return_sequences=1, do_sample=False)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(dec, lang=tgt_lang)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    logger = setup_logger(os.path.join(cfg["log_dir"], "04_base.log"))

    logger.info("=" * 60)
    logger.info("Phase 3-BASE: Diverse candidates from BASE model (no SFT)")
    logger.info(f"Model: {cfg['model_name']}  (NO adapter)")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Reuse exact RL sentences from the existing scored file (alignment)
    scored = pd.read_csv(cfg["scored_file"])
    srcs = scored["src"].astype(str).tolist()
    tgts = scored["tgt"].astype(str).tolist()
    logger.info(f"Loaded {len(srcs)} RL sentences from rl_scored.csv")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True)
    if not hasattr(model.config, "hidden_size"):
        model.config.hidden_size = model.config.decoder_embed_dim
    model = model.to(DEVICE).eval()
    ip = IndicProcessor(inference=True)

    rows, bs = [], 8
    for start in range(0, len(srcs), bs):
        b = srcs[start:start+bs]
        c1 = gen(model, tok, ip, b, cfg["src_lang"], cfg["tgt_lang"],
                 cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE,
                 sample=False)
        c2 = gen(model, tok, ip, b, cfg["src_lang"], cfg["tgt_lang"],
                 cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE,
                 sample=True, temperature=0.8, top_p=0.9, seed=42+start)
        for i in range(len(b)):
            rows.append({"src": b[i], "tgt": tgts[start+i],
                        "candidates": [c1[i], c2[i]]})
        if (start+bs) % 200 < bs:
            logger.info(f"  {min(start+bs,len(srcs))}/{len(srcs)}")

    out = cfg["candidates_file"].replace("rl_candidates.csv",
                                         "rl_candidates_base.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Base candidates saved: {out}  ({len(rows)} rows)")
    logger.info("Next: score with 05 (override paths), then 06_dpo_on_base.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()