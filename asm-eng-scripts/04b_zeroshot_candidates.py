"""
=============================================================================
Script 04b — Zero-Shot Candidate Generation for Competence Signal
=============================================================================
Generates GREEDY zero-shot translations (base model, NO adapter) for the
exact RL training sentences used in DPO, and writes them in the candidates
CSV format so the UNCHANGED 05_reward_scoring.py can score them.

The resulting reward of the zero-shot output = base-model competence per
example, which CGPO-zs uses as its gating signal.

Reads the src/tgt from the existing rl_scored.csv (guarantees identical
sentence alignment with the DPO training set).

Output: a candidates file with candidates=[zeroshot, zeroshot] per row
        (duplicated so k=2 format matches; both identical, scored once).

Usage:
    python3 04b_zeroshot_candidates.py --config ../data2/config_v7.yaml
=============================================================================
"""

import os
import sys
import ast
import logging
import argparse
import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger(os.path.join(cfg["log_dir"], "04b_zeroshot.log"))
    logger.info("=" * 60)
    logger.info("Phase 4b: Zero-Shot Candidate Generation (competence signal)")
    logger.info(f"Model    : {cfg['model_name']}  (NO adapter — zero-shot)")
    logger.info(f"src_lang : {cfg['src_lang']}  tgt_lang : {cfg['tgt_lang']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Read the EXACT RL training sentences from the scored file (alignment!)
    scored = pd.read_csv(cfg["scored_file"])
    srcs = scored["src"].astype(str).tolist()
    tgts = scored["tgt"].astype(str).tolist()
    logger.info(f"Loaded {len(srcs)} RL training sentences from rl_scored.csv")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"],
                                              trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True)
    if not hasattr(model.config, "hidden_size"):
        model.config.hidden_size = model.config.decoder_embed_dim
    model = model.to(DEVICE).eval()
    ip = IndicProcessor(inference=True)

    logger.info("Generating zero-shot (greedy) translations...")
    zs = []
    bs = cfg.get("eval_batch_size", 16)
    for start in range(0, len(srcs), bs):
        batch = srcs[start:start+bs]
        proc = ip.preprocess_batch(batch, src_lang=cfg["src_lang"],
                                   tgt_lang=cfg["tgt_lang"])
        enc = tokenizer(proc, return_tensors="pt", padding=True,
                        truncation=True, max_length=cfg["max_length"])
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc,
                                 max_new_tokens=cfg["gen_max_new_tokens"],
                                 num_beams=1, do_sample=False)
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        zs.extend(ip.postprocess_batch(dec, lang=cfg["tgt_lang"]))
        if (start + bs) % 200 < bs:
            logger.info(f"  {min(start+bs,len(srcs))}/{len(srcs)}")

    # Write in candidates format (duplicate so k=2 matches pipeline)
    out_df = pd.DataFrame({
        "src": srcs,
        "tgt": tgts,
        "candidates": [[z, z] for z in zs],
    })
    out_path = cfg["candidates_file"].replace(
        "rl_candidates.csv", "rl_zeroshot_candidates.csv")
    out_df.to_csv(out_path, index=False)
    logger.info(f"Zero-shot candidates saved: {out_path}")
    logger.info(f"  Rows: {len(out_df)}")
    logger.info("Next: score this file with 05_reward_scoring.py (override "
                "candidates_file + scored_file), then run 04c_merge_competence.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()