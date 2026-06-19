"""
=============================================================================
Script 07 — Final Evaluation (Phase 6)
English → Assamese RLFT Pipeline | IIT Patna
=============================================================================
Evaluates the DPO-trained model on the held-out test set.
Metrics: BLEU (SacreBLEU), chrF++, COMET (wmt22-comet-da)

OUTPUTS:
  eval_results.json        — BLEU, chrF++, COMET scores
  hypotheses.txt           — raw model output, one line per test sentence
  evaluation_full.tsv      — src | reference | hypothesis (tab-separated)

Usage:
    python3 07_evaluate_final.py --config ../data2/config_v7.yaml
    # Evaluate a specific adapter (e.g. CGPO output) with a custom run name:
    python3 07_evaluate_final.py --config ../data2/config_v7.yaml \
        --adapter_override ../models2/eng-asm-v7-cgpo-g1.0/dpo_lora_adapter \
        --run_name eng-asm-v7-cgpo-g1.0
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
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from comet import download_model, load_from_checkpoint
for _pkg in ['wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter_override", type=str, default=None,
                        help="Path to a DPO adapter to evaluate instead of "
                             "cfg['dpo_adapter_dir'] (e.g. a CGPO output).")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Evaluate the BASE model with NO adapter "
                             "(zero-shot baseline).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override experiment name for output files "
                             "(keeps results from overwriting each other).")
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
    sig = "unavailable"
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve adapter + run name (supports CGPO / arbitrary adapters)
    adapter_dir = args.adapter_override or cfg.get("dpo_adapter_dir")
    run_name    = args.run_name or cfg["experiment_name"]

    # Output files go in a run-specific log dir so nothing is overwritten
    if args.run_name:
        log_dir = cfg["log_dir"].replace(cfg["experiment_name"], run_name)
        results_file = cfg["results_file"].replace(
            cfg["experiment_name"], run_name)
    else:
        log_dir = cfg["log_dir"]
        results_file = cfg["results_file"]
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "07_evaluate_final.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Phase 6: Final Evaluation")
    logger.info(f"Run name   : {run_name}")
    logger.info(f"Model      : {cfg['model_name']}")
    logger.info(f"src_lang   : {cfg['src_lang']}")
    logger.info(f"tgt_lang   : {cfg['tgt_lang']}")
    logger.info(f"DPO adapter: {adapter_dir}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Load test data
    with open(cfg["test_src"], "r", encoding="utf-8") as f:
        test_src = [l.strip() for l in f if l.strip()]
    with open(cfg["test_tgt"], "r", encoding="utf-8") as f:
        test_tgt = [l.strip() for l in f if l.strip()]
    logger.info(f"Test pairs: {len(test_src)}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )

    if args.zero_shot:
        logger.info("ZERO-SHOT: loading BASE model with NO adapter...")
    else:
        logger.info(f"Loading DPO model from {adapter_dir}...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    if not hasattr(base.config, "hidden_size"):
        base.config.hidden_size = base.config.decoder_embed_dim

    if args.zero_shot:
        model = base.to(DEVICE)
    else:
        model = PeftModel.from_pretrained(base, adapter_dir)
        model = model.to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)

    logger.info("Generating translations on test set...")
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

    hyp_path = os.path.join(log_dir, f"{run_name}_hypotheses.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        for hyp in hypotheses:
            f.write(hyp.strip() + "\n")
    logger.info(f"Raw hypotheses saved: {hyp_path}")
    logger.info(f"  Total lines: {len(hypotheses)}")

    tsv_path = os.path.join(log_dir, f"{run_name}_evaluation_full.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("source\treference\thypothesis\n")
        for src, ref, hyp in zip(test_src, test_tgt, hypotheses):
            f.write(f"{src.strip()}\t{ref.strip()}\t{hyp.strip()}\n")
    logger.info(f"Full TSV saved: {tsv_path}")

    logger.info("\nSample translations:")
    for i in range(min(10, len(test_src))):
        logger.info(f"\n[{i+1}]")
        logger.info(f"  SRC : {test_src[i]}")
        logger.info(f"  REF : {test_tgt[i]}")
        logger.info(f"  HYP : {hypotheses[i]}")

    logger.info("\nComputing BLEU...")
    bleu = sacrebleu.corpus_bleu(
        hypotheses, [test_tgt], use_effective_order=True
    )
    logger.info("Computing chrF++ (word_order=2)...")
    chrf = sacrebleu.corpus_chrf(
        hypotheses, [test_tgt], word_order=2
    )
    logger.info("Computing chrF (word_order=0, WMT-style)...")
    chrf0 = sacrebleu.corpus_chrf(
        hypotheses, [test_tgt], word_order=0
    )
    logger.info("Computing METEOR (NLTK)...")
    _meteors = []
    for hyp, ref in zip(hypotheses, test_tgt):
        try:
            _meteors.append(meteor_score([ref.split()], hyp.split()))
        except Exception:
            _meteors.append(0.0)
    meteor = float(np.mean(_meteors)) if _meteors else 0.0
    logger.info("Computing COMET...")
    comet_model_path = download_model(cfg["comet_model"], saving_directory=None)
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data  = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(test_src, hypotheses, test_tgt)
    ]
    comet_results = comet_model.predict(
        comet_data, batch_size=8,
        gpus=0, num_workers=0, progress_bar=True
    )
    comet_score = float(np.mean(comet_results.scores))

    results = {
        "experiment": run_name,
        "adapter":    adapter_dir,
        "model":      cfg["model_name"],
        "src_lang":   cfg["src_lang"],
        "tgt_lang":   cfg["tgt_lang"],
        "bleu":       round(bleu.score, 4),
        "chrf++":     round(chrf.score, 4),
        "chrf":       round(chrf0.score, 4),
        "meteor":     round(meteor, 4),
        "comet":      round(comet_score, 4),
        "sacrebleu_signature": sig,
        "n_test":     len(test_src),
        "hypotheses_file": hyp_path,
        "full_tsv_file":   tsv_path,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL EVALUATION SUMMARY — {run_name}")
    logger.info(f"  BLEU   : {bleu.score:.4f}")
    logger.info(f"  chrF++ : {chrf.score:.4f}")
    logger.info(f"  chrF   : {chrf0.score:.4f}  (WMT-style, word_order=0)")
    logger.info(f"  METEOR : {meteor:.4f}")
    logger.info(f"  COMET  : {comet_score:.4f}")
    # SacreBLEU signature (robust across versions)
    sig = "unavailable"
    try:
        from sacrebleu.metrics import BLEU as _BLEU
        sig = str(_BLEU().get_signature())
    except Exception:
        try:
            sig = bleu.get_signature().format()
        except Exception:
            sig = "unavailable (check sacrebleu version manually)"
    logger.info(f"  SacreBLEU signature: {sig}")
    logger.info(f"  Results    : {results_file}")
    logger.info(f"  Hypotheses : {hyp_path}")
    logger.info(f"  Full TSV   : {tsv_path}")
    logger.info("Phase 6 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()