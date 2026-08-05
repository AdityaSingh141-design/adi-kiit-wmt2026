"""
=============================================================================
Script 07 — Evaluation for Single Reward Ablation
English → Assamese | IIT Patna
=============================================================================
Evaluates DPO adapter trained with a single reward component.

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
    dpo_adapter_dir = cfg["dpo_adapter_dir"].replace(
        "asm-eng-v7", f"asm-eng-v7-{reward}"
    )
    log_dir = cfg["log_dir"].replace(
        "asm-eng-v7", f"asm-eng-v7-{reward}"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"07_eval_{reward}_only.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info(f"Evaluation — {reward.upper()} only DPO")
    logger.info(f"DPO adapter: {dpo_adapter_dir}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    base = "/mnt/storage/boynao/aditya/exp2/rlft-asm-eng/data2"
    test_src_path = cfg["test_src"] if cfg["test_src"].startswith("/") \
                    else f"{base}/{cfg['test_src']}"
    test_tgt_path = cfg["test_tgt"] if cfg["test_tgt"].startswith("/") \
                    else f"{base}/{cfg['test_tgt']}"

    with open(test_src_path, "r", encoding="utf-8") as f:
        test_src = [l.strip() for l in f if l.strip()]
    with open(test_tgt_path, "r", encoding="utf-8") as f:
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
    hypotheses = []
    batch_size  = cfg.get("eval_batch_size", 16)
    for start in range(0, len(test_src), batch_size):
        end   = min(start + batch_size, len(test_src))
        hyps  = translate_batch(
            model, tokenizer, ip, test_src[start:end],
            cfg["src_lang"], cfg["tgt_lang"],
            cfg["max_length"], cfg["gen_max_new_tokens"], DEVICE
        )
        hypotheses.extend(hyps)
        if end % 300 == 0 or end == len(test_src):
            logger.info(f"  Translated {end}/{len(test_src)}")

    # Sample outputs
    logger.info("\nSample translations:")
    for i in range(min(5, len(test_src))):
        logger.info(f"  SRC: {test_src[i]}")
        logger.info(f"  REF: {test_tgt[i]}")
        logger.info(f"  HYP: {hypotheses[i]}")

    # Metrics
    bleu  = sacrebleu.corpus_bleu(
        hypotheses, [test_tgt], use_effective_order=True
    )
    chrf  = sacrebleu.corpus_chrf(
        hypotheses, [test_tgt], word_order=2
    )
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
        "experiment": f"asm-eng-v7-{reward}-only",
        "reward_component": reward.upper(),
        "bleu":  round(bleu.score, 4),
        "chrf":  round(chrf.score, 4),
        "comet": round(comet_score, 4),
    }
    results_path = os.path.join(log_dir, f"results_{reward}_only.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"EVALUATION SUMMARY — {reward.upper()} only")
    logger.info(f"  BLEU   : {bleu.score:.4f}")
    logger.info(f"  chrF++ : {chrf.score:.4f}")
    logger.info(f"  COMET  : {comet_score:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()