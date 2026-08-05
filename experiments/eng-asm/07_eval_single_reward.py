"""
=============================================================================
Script 07 — Evaluation for Single Reward Ablation
English → Assamese | IIT Patna
=============================================================================
Evaluates DPO adapter trained with a single reward component.
Saves raw hypotheses and full TSV for comparison.

OUTPUTS:
  results_{reward}_only.json          — BLEU, chrF++, COMET scores
  {experiment}_{reward}_hypotheses.txt — raw model output, one line per sentence
  {experiment}_{reward}_evaluation_full.tsv — src | reference | hypothesis

Usage:
    python3 07_eval_single_reward.py --config ../data2/config_v7.yaml --reward gemba
    python3 07_eval_single_reward.py --config ../data2/config_v7.yaml --reward mqm
    python3 07_eval_single_reward.py --config ../data2/config_v7.yaml --reward comet
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
from comet import download_model, load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True,
                        choices=["gemba", "mqm", "comet"])
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
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    processed = ip.preprocess_batch(
        src_batch, src_lang=src_lang, tgt_lang=tgt_lang
    )
    enc = tokenizer(
        processed, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=bos_id,
            max_new_tokens=max_new_tokens,
            num_beams=1, do_sample=False
        )
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt_lang)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    reward = args.reward
    exp_name = cfg["experiment_name"]

    # Reward-specific directories
    dpo_adapter_dir = cfg["dpo_adapter_dir"].replace(
        exp_name, f"{exp_name}-{reward}"
    )
    log_dir = cfg["log_dir"].replace(
        exp_name, f"{exp_name}-{reward}"
    )
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"07_eval_{reward}_only.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info(f"Evaluation — {reward.upper()} only DPO")
    logger.info(f"Experiment : {exp_name}-{reward}")
    logger.info(f"DPO adapter: {dpo_adapter_dir}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Load test data
    with open(cfg["test_src"], "r", encoding="utf-8") as f:
        test_src = [l.strip() for l in f if l.strip()]
    with open(cfg["test_tgt"], "r", encoding="utf-8") as f:
        test_tgt = [l.strip() for l in f if l.strip()]
    logger.info(f"Test pairs: {len(test_src)}")

    # Load model with single-reward DPO adapter
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True
    )
    if not hasattr(base_model.config, "hidden_size"):
        base_model.config.hidden_size = base_model.config.decoder_embed_dim

    model = PeftModel.from_pretrained(base_model, dpo_adapter_dir)
    model = model.to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)

    # Generate translations
    logger.info(f"Generating translations with {reward.upper()} reward model...")
    hypotheses = []
    batch_size  = cfg.get("eval_batch_size", 16)

    for start in range(0, len(test_src), batch_size):
        end  = min(start + batch_size, len(test_src))
        hyps = translate_batch(
            model, tokenizer, ip, test_src[start:end],
            cfg["src_lang"], cfg["tgt_lang"],
            cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE
        )
        hypotheses.extend(hyps)
        if end % 300 == 0 or end == len(test_src):
            logger.info(f"  Translated {end}/{len(test_src)}")

    # ── Save raw hypotheses ───────────────────────────────────────────────
    hyp_path = os.path.join(
        log_dir, f"{exp_name}-{reward}_hypotheses.txt"
    )
    with open(hyp_path, "w", encoding="utf-8") as f:
        for hyp in hypotheses:
            f.write(hyp.strip() + "\n")
    logger.info(f"Raw hypotheses saved: {hyp_path}")
    logger.info(f"  Total lines: {len(hypotheses)}")

    # ── Save full TSV ─────────────────────────────────────────────────────
    tsv_path = os.path.join(
        log_dir, f"{exp_name}-{reward}_evaluation_full.tsv"
    )
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("source\treference\thypothesis\n")
        for src, ref, hyp in zip(test_src, test_tgt, hypotheses):
            f.write(f"{src.strip()}\t{ref.strip()}\t{hyp.strip()}\n")
    logger.info(f"Full TSV saved: {tsv_path}")

    # Count টা artifact
    ta_count = sum(1 for h in hypotheses if h.strip().startswith("টা"))
    logger.info(f"  'টা' prefix count: {ta_count}/{len(hypotheses)} "
                f"({100*ta_count/len(hypotheses):.1f}%)")

    # Sample outputs
    logger.info("\nSample translations:")
    for i in range(min(5, len(test_src))):
        logger.info(f"  SRC: {test_src[i]}")
        logger.info(f"  REF: {test_tgt[i]}")
        logger.info(f"  HYP: {hypotheses[i]}")

    # BLEU
    bleu  = sacrebleu.corpus_bleu(
        hypotheses, [test_tgt], use_effective_order=True
    )
    # chrF++
    chrf  = sacrebleu.corpus_chrf(
        hypotheses, [test_tgt], word_order=2
    )
    # COMET
    comet_path    = download_model(cfg["comet_model"], saving_directory=None)
    comet_model   = load_from_checkpoint(comet_path)
    comet_data    = [{"src": s, "mt": h, "ref": r}
                     for s, h, r in zip(test_src, hypotheses, test_tgt)]
    comet_results = comet_model.predict(
        comet_data, batch_size=8,
        gpus=0, num_workers=0, progress_bar=True
    )
    comet_score = float(np.mean(comet_results.scores))

    results = {
        "experiment":       f"{exp_name}-{reward}-only",
        "reward_component": reward.upper(),
        "bleu":             round(bleu.score, 4),
        "chrf":             round(chrf.score, 4),
        "comet":            round(comet_score, 4),
        "ta_prefix_count":  ta_count,
        "ta_prefix_pct":    round(100*ta_count/len(hypotheses), 1),
        "n_test":           len(test_src),
        "hypotheses_file":  hyp_path,
        "full_tsv_file":    tsv_path,
    }
    results_path = os.path.join(log_dir, f"results_{reward}_only.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"EVALUATION SUMMARY — {reward.upper()} only")
    logger.info(f"  BLEU        : {bleu.score:.4f}")
    logger.info(f"  chrF++      : {chrf.score:.4f}")
    logger.info(f"  COMET       : {comet_score:.4f}")
    logger.info(f"  টা artifact : {ta_count}/{len(hypotheses)} "
                f"({100*ta_count/len(hypotheses):.1f}%)")
    logger.info(f"  Hypotheses  : {hyp_path}")
    logger.info(f"  Full TSV    : {tsv_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()