"""
=============================================================================
Script 01 — Data Preparation (Phase 1)
RLFT Pipeline | WMT 2026
=============================================================================
Reads pre-provided train/valid/test files from data2/ folder.
Files are already split — this script just verifies and logs them.

Data format: plain text, one sentence per line
  train.eng_Latn / train.asm_Beng  (54,000 pairs)
  valid.eng_Latn / valid.asm_Beng  (1,024 pairs)
  test.eng_Latn  / test.asm_Beng   (1,503 pairs)

Usage:
    python3 01_data_preparation.py --config ../configs/config_v7.yaml
=============================================================================
"""

import os
import sys
import logging
import argparse
import yaml


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


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def verify_file(path, name, logger):
    if not os.path.exists(path):
        logger.error(f"MISSING: {name} at {path}")
        sys.exit(1)
    n = count_lines(path)
    logger.info(f"  {name}: {n} lines — {path}")
    return n


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_path = os.path.join(cfg["log_dir"], "01_data_preparation.log")
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Phase 1: Data Preparation — English → Assamese")
    logger.info(f"src_lang : {cfg['src_lang']}")
    logger.info(f"tgt_lang : {cfg['tgt_lang']}")
    logger.info("=" * 60)

    data_dir = os.path.dirname(args.config)
    # data_dir is configs/ — go up one level to reach data2/
    project_dir = os.path.dirname(data_dir)

    splits = {
        "train_src": os.path.join(project_dir, "data2", cfg["train_src"]),
        "train_tgt": os.path.join(project_dir, "data2", cfg["train_tgt"]),
        "valid_src": os.path.join(project_dir, "data2", cfg["valid_src"]),
        "valid_tgt": os.path.join(project_dir, "data2", cfg["valid_tgt"]),
        "test_src":  os.path.join(project_dir, "data2", cfg["test_src"]),
        "test_tgt":  os.path.join(project_dir, "data2", cfg["test_tgt"]),
    }

    logger.info("Verifying data files...")
    counts = {}
    for name, path in splits.items():
        counts[name] = verify_file(path, name, logger)

    # Sanity checks
    assert counts["train_src"] == counts["train_tgt"], "Train src/tgt line count mismatch!"
    assert counts["valid_src"] == counts["valid_tgt"], "Valid src/tgt line count mismatch!"
    assert counts["test_src"]  == counts["test_tgt"],  "Test src/tgt line count mismatch!"

    logger.info("")
    logger.info("Data verification summary:")
    logger.info(f"  Train : {counts['train_src']} pairs")
    logger.info(f"  Valid : {counts['valid_src']} pairs")
    logger.info(f"  Test  : {counts['test_src']} pairs")
    logger.info(f"  Total : {counts['train_src'] + counts['valid_src'] + counts['test_src']} pairs")

    # Show sample sentences
    logger.info("")
    logger.info("Sample training sentences:")
    with open(splits["train_src"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            logger.info(f"  ENG: {line.strip()}")
    with open(splits["train_tgt"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            logger.info(f"  ASM: {line.strip()}")

    logger.info("")
    logger.info("Phase 1 complete. Data is ready.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
