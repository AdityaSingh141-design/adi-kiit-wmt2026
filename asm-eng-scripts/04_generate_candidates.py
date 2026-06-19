"""
=============================================================================
Script 04 — Candidate Generation (Phase 3)
English → Assamese RLFT Pipeline | IIT Patna
=============================================================================
Generates k=2 translation candidates per source sentence using the SFT model.
For iterative DPO (V9): uses generation_adapter_dir if specified in config.

Critical constraints (same as Hindi-Maithili):
  - Greedy decoding ONLY (beam search freezes IndicTrans2 CUDA kernels)
  - postprocess_batch() called ONCE per batch (threading hang bug)
  - IndicProcessor pre-processes source ONCE before generation loop

Usage:
    python3 04_generate_candidates.py --config ../configs/config_v7.yaml
=============================================================================
"""

import os
import sys
import gc
import logging
import argparse
import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def generate_batch(model, tokenizer, ip, src_batch, tgt_lang,
                   src_lang, max_length, max_new_tokens, device,
                   sample=False, temperature=0.8, top_p=0.9, seed=None):
    """
    Generate translations for a batch of source sentences.
    Calls postprocess_batch() ONCE per batch to avoid threading deadlock.
    """

    # Preprocess source batch
    processed = ip.preprocess_batch(
        src_batch, src_lang=src_lang, tgt_lang=tgt_lang
    )
    enc = tokenizer(
        processed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    if sample and seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        if sample:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0
            )
        else:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                num_return_sequences=1,
                do_sample=False
            )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    # Single postprocess_batch call per batch (threading queue bug)
    postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)
    return postprocessed


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_path = os.path.join(cfg["log_dir"], "04_generate_candidates.log")
    logger = setup_logger(log_path)

    # Check if iterative DPO (V9) — use DPO adapter as generation model
    gen_adapter = cfg.get("generation_adapter_dir", cfg["sft_adapter_dir"])
    is_iterative = gen_adapter != cfg["sft_adapter_dir"]

    logger.info("=" * 60)
    logger.info("Phase 3: Candidate Generation")
    logger.info(f"Model          : {cfg['model_name']}")
    logger.info(f"src_lang       : {cfg['src_lang']}")
    logger.info(f"tgt_lang       : {cfg['tgt_lang']}")
    logger.info(f"Generation adapter: {'DPO (iterative)' if is_iterative else 'SFT'}")
    logger.info(f"Adapter path   : {gen_adapter}")
    logger.info(f"k_candidates   : {cfg['k_candidates']}")
    logger.info(f"rl_samples     : {cfg['rl_samples']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Resolve data paths
    data_dir = os.path.dirname(args.config)
    project_dir = os.path.dirname(data_dir)
    train_src_path = os.path.join(project_dir, "data2", cfg["train_src"])
    train_tgt_path = os.path.join(project_dir, "data2", cfg["train_tgt"])

    # Load RL training samples (fixed seed for reproducibility)
    logger.info(f"Sampling {cfg['rl_samples']} RL training sentences...")
    with open(train_src_path, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(train_tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]

    import random
    random.seed(42)
    indices = random.sample(range(len(src_lines)), cfg["rl_samples"])
    src_sample = [src_lines[i] for i in indices]
    tgt_sample = [tgt_lines[i] for i in indices]
    logger.info(f"  Sampled {len(src_sample)} sentences (seed=42)")

    # Load tokenizer and model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )

    logger.info(f"Loading generation model from {gen_adapter}...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    if not hasattr(base.config, "hidden_size"):
        base.config.hidden_size = base.config.decoder_embed_dim

    model = PeftModel.from_pretrained(base, gen_adapter)
    model = model.to(DEVICE)
    model.eval()

    # Load IndicProcessor
    ip = IndicProcessor(inference=True)

    # Generate candidates
    all_rows = []
    batch_size = 8
    total = len(src_sample)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        src_batch = src_sample[start:end]
        tgt_batch = tgt_sample[start:end]

        # Generate candidate 1 (standard length)
        cands_1 = generate_batch(
            model, tokenizer, ip, src_batch,
            cfg["tgt_lang"], cfg["src_lang"],
            cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE
        )

        # Generate candidate 2 (slightly longer)
        cands_2 = generate_batch(
            model, tokenizer, ip, src_batch,
            cfg["tgt_lang"], cfg["src_lang"],
            cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE,
            sample=True, temperature=0.8, top_p=0.9, seed=42 + start
        )

        for i in range(len(src_batch)):
            all_rows.append({
                "src": src_batch[i],
                "tgt": tgt_batch[i],
                "candidates": [cands_1[i], cands_2[i]]
            })

        if (end) % 200 == 0 or end == total:
            logger.info(f"  Progress: {end}/{total}")
            logger.info(f"    SRC : {src_batch[0][:80]}")
            logger.info(f"    Model: {cands_1[0][:80]}")
            logger.info(f"    Ref  : {tgt_batch[0][:80]}")

            # Save checkpoint
            ckpt_df = pd.DataFrame(all_rows)
            ckpt_path = cfg["candidates_file"].replace(
                ".csv", f"_ckpt_{end}.csv"
            )
            ckpt_df.to_csv(ckpt_path, index=False)

        if end % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save final candidates file
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(cfg["candidates_file"]), exist_ok=True)
    df.to_csv(cfg["candidates_file"], index=False)
    logger.info(f"Candidates saved to: {cfg['candidates_file']}")
    logger.info(f"Total rows: {len(df)}")
    logger.info("Phase 3 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()