"""
=============================================================================
Script 00 — Zero-Shot Baseline Evaluation
English → Assamese | IIT Patna
=============================================================================
Evaluates the BASE model WITHOUT any fine-tuning.

OUTPUTS:
  baseline_results.json       — BLEU, chrF++, COMET scores
  baseline_hypotheses.txt     — raw model output, one line per test sentence
  baseline_evaluation_full.tsv — src | reference | hypothesis

Usage:
    python3 00_baseline_eval.py --config ../data2/config_v7.yaml
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import sacrebleu
from comet import download_model, load_from_checkpoint


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


def translate_batch(model, tokenizer, ip, src_batch,
                    src_lang, tgt_lang, max_length,
                    max_new_tokens, device):
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
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False
        )
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)
    return postprocessed


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["log_dir"].replace("eng-asm-v7", "eng-asm-baseline")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "00_baseline_eval.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Zero-Shot Baseline Evaluation (No Fine-Tuning)")
    logger.info(f"Model    : {cfg['model_name']}")
    logger.info(f"src_lang : {cfg['src_lang']}")
    logger.info(f"tgt_lang : {cfg['tgt_lang']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Load test data
    with open(cfg["test_src"], "r", encoding="utf-8") as f:
        test_src = [l.strip() for l in f if l.strip()]
    with open(cfg["test_tgt"], "r", encoding="utf-8") as f:
        test_tgt = [l.strip() for l in f if l.strip()]
    logger.info(f"Test pairs: {len(test_src)}")

    # Load BASE model only — no LoRA adapter
    logger.info("Loading BASE model (zero-shot, no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    if not hasattr(model.config, "hidden_size"):
        model.config.hidden_size = model.config.decoder_embed_dim
    model = model.to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)

    # Generate translations
    logger.info("Generating zero-shot translations...")
    hypotheses = []
    batch_size  = cfg.get("eval_batch_size", 16)

    for start in range(0, len(test_src), batch_size):
        end   = min(start + batch_size, len(test_src))
        batch = test_src[start:end]
        hyps  = translate_batch(
            model, tokenizer, ip, batch,
            cfg["src_lang"], cfg["tgt_lang"],
            cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE
        )
        hypotheses.extend(hyps)
        if end % 200 == 0 or end == len(test_src):
            logger.info(f"  Translated {end}/{len(test_src)}")

    # ── Save raw hypotheses ───────────────────────────────────────────────
    hyp_path = os.path.join(log_dir, "baseline_hypotheses.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        for hyp in hypotheses:
            f.write(hyp.strip() + "\n")
    logger.info(f"Raw hypotheses saved: {hyp_path}")
    logger.info(f"  Total lines: {len(hypotheses)}")

    # ── Save full TSV ─────────────────────────────────────────────────────
    tsv_path = os.path.join(log_dir, "baseline_evaluation_full.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("source\treference\thypothesis\n")
        for src, ref, hyp in zip(test_src, test_tgt, hypotheses):
            f.write(f"{src.strip()}\t{ref.strip()}\t{hyp.strip()}\n")
    logger.info(f"Full TSV saved: {tsv_path}")

    # Sample outputs
    logger.info("\nSample zero-shot translations:")
    for i in range(min(10, len(test_src))):
        logger.info(f"\n[{i+1}]")
        logger.info(f"  SRC : {test_src[i]}")
        logger.info(f"  REF : {test_tgt[i]}")
        logger.info(f"  HYP : {hypotheses[i]}")

    # BLEU
    logger.info("\nComputing BLEU...")
    bleu = sacrebleu.corpus_bleu(
        hypotheses, [test_tgt], use_effective_order=True
    )

    # chrF++
    logger.info("Computing chrF++...")
    chrf = sacrebleu.corpus_chrf(
        hypotheses, [test_tgt], word_order=2
    )

    # COMET
    logger.info("Computing COMET...")
    comet_path  = download_model(cfg["comet_model"], saving_directory=None)
    comet_model = load_from_checkpoint(comet_path)
    comet_data  = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(test_src, hypotheses, test_tgt)
    ]
    comet_results = comet_model.predict(
        comet_data, batch_size=8,
        gpus=0, num_workers=0, progress_bar=True
    )
    comet_score = float(np.mean(comet_results.scores))

    # Save results
    results = {
        "experiment":      "eng-asm-baseline-zeroshot",
        "model":           cfg["model_name"],
        "src_lang":        cfg["src_lang"],
        "tgt_lang":        cfg["tgt_lang"],
        "bleu":            round(bleu.score, 4),
        "chrf":            round(chrf.score, 4),
        "comet":           round(comet_score, 4),
        "n_test":          len(test_src),
        "note":            "Zero-shot baseline, no fine-tuning",
        "hypotheses_file": hyp_path,
        "full_tsv_file":   tsv_path,
    }
    results_path = os.path.join(log_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("ZERO-SHOT BASELINE SUMMARY")
    logger.info(f"  BLEU   : {bleu.score:.4f}")
    logger.info(f"  chrF++ : {chrf.score:.4f}")
    logger.info(f"  COMET  : {comet_score:.4f}")
    logger.info(f"  Hypotheses : {hyp_path}")
    logger.info(f"  Full TSV   : {tsv_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()