"""
=============================================================================
Script 05 — Reward Scoring (Phase 4)
English → Assamese RLFT Pipeline | IIT Patna
=============================================================================
Composite reward: R = 0.4*GEMBA + 0.4*MQM + 0.2*COMET-poly

Components:
  GEMBA     : Qwen2.5-7B-Instruct holistic score (GPU)
  MQM       : wmt22-cometkiwi-da reference-free (CPU, num_workers=0)
  COMET-poly: XCOMET-XL reference-based (CPU, num_workers=0)

Bug fix: num_workers=0 in all COMET predict() calls to prevent
         multiprocessing deadlock inside training loop.

Usage:
    python3 05_reward_scoring.py --config ../configs/config_v7.yaml
=============================================================================
"""

import os
import sys
import ast
import gc
import logging
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet import download_model, load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--gemba_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size",  type=int, default=8)
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


class GEMBAScorer:
    PROMPT_TEMPLATE = """You are an expert translator and translation quality evaluator.

Your task is to evaluate the quality of a machine translation from English to Assamese.

Source (English): {src}
Translation (Assamese): {mt}
Reference (Assamese): {ref}

Rate the translation quality on a scale from 0 to 100, where:
- 0-20  : Very poor, major errors, unintelligible
- 21-40 : Poor, many errors, meaning partially lost
- 41-60 : Acceptable, some errors, meaning mostly preserved
- 61-80 : Good, minor errors, meaning well preserved
- 81-100: Excellent, near perfect translation

Respond with ONLY a single integer between 0 and 100. No explanation."""

    def __init__(self, model_id, device, logger):
        self.device = device
        self.logger = logger
        logger.info(f"  Loading GEMBA judge: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("  GEMBA judge loaded.")

    def score(self, src, mt, ref):
        prompt = self.PROMPT_TEMPLATE.format(src=src, mt=mt, ref=ref)
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        try:
            score = float("".join(filter(
                lambda c: c.isdigit() or c == ".",
                response.split()[0]
            )))
            score = max(0.0, min(100.0, score))
        except Exception:
            self.logger.warning(f"  GEMBA parse failed: '{response}'. Default 50.")
            score = 50.0
        return score / 100.0


class MQMScorer:
    MODEL_ID = "Unbabel/wmt22-cometkiwi-da"

    def __init__(self, logger):
        logger.info(f"  Loading MQM scorer: {self.MODEL_ID}")
        model_path = download_model(self.MODEL_ID, saving_directory=None)
        self.model = load_from_checkpoint(model_path)
        logger.info("  MQM scorer loaded.")

    def score_batch(self, sources, translations, batch_size=8):
        data = [{"src": s, "mt": t}
                for s, t in zip(sources, translations)]
        results = self.model.predict(
            data, batch_size=batch_size,
            gpus=0, num_workers=0, progress_bar=False
        )
        return [max(0.0, min(1.0, float(s))) for s in results.scores]


class COMETPolyScorer:
    MODEL_ID = "Unbabel/XCOMET-XL"

    def __init__(self, logger):
        logger.info(f"  Loading COMET-poly: {self.MODEL_ID}")
        model_path = download_model(self.MODEL_ID, saving_directory=None)
        self.model = load_from_checkpoint(model_path)
        logger.info("  COMET-poly loaded.")

    def score_batch(self, sources, translations, references, batch_size=8):
        data = [{"src": s, "mt": t, "ref": r}
                for s, t, r in zip(sources, translations, references)]
        results = self.model.predict(
            data, batch_size=batch_size,
            gpus=0, num_workers=0, progress_bar=False
        )
        return [max(0.0, min(1.0, float(s))) for s in results.scores]


def compute_reward(gemba, mqm, comet,
                   w1=0.4, w2=0.4, w3=0.2):
    return w1 * gemba + w2 * mqm + w3 * comet


def main():
    from huggingface_hub import login
    token_path = "/mnt/storage/boynao/aditya/cache/huggingface/token"
    if os.path.exists(token_path):
        login(token=open(token_path).read().strip())

    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_path = os.path.join(cfg["log_dir"], "05_reward_scoring.log")
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Phase 4: LLM-as-Judge Reward Scoring")
    logger.info(f"GEMBA model  : {args.gemba_model}")
    logger.info(f"MQM model    : {MQMScorer.MODEL_ID}")
    logger.info(f"COMET-poly   : {COMETPolyScorer.MODEL_ID}")
    logger.info("Reward       : R = 0.4*GEMBA + 0.4*MQM + 0.2*COMET")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    # Load candidates
    logger.info(f"Loading candidates from {cfg['candidates_file']}...")
    df = pd.read_csv(cfg["candidates_file"])
    if isinstance(df["candidates"].iloc[0], str):
        df["candidates"] = df["candidates"].apply(ast.literal_eval)
    df = df[df["candidates"].apply(
        lambda x: len(x) > 0)].reset_index(drop=True)
    logger.info(f"Rows: {len(df)}, Candidates/row: {len(df['candidates'].iloc[0])}")

    # Load all scorers
    logger.info("Loading scoring models...")
    gemba_scorer = GEMBAScorer(args.gemba_model, DEVICE, logger)
    mqm_scorer   = MQMScorer(logger)
    comet_scorer = COMETPolyScorer(logger)
    logger.info("All scorers ready.")

    all_rewards = []
    all_best_idx = []
    all_best_trans = []
    all_gemba = []
    all_mqm = []
    all_comet = []
    total = len(df)

    for i, row in df.iterrows():
        src   = row["src"]
        ref   = row["tgt"]
        cands = row["candidates"]

        row_rewards = []
        row_gemba = []
        row_mqm = []
        row_comet = []

        for cand in cands:
            g = gemba_scorer.score(src, cand, ref)
            m = mqm_scorer.score_batch([src], [cand], batch_size=1)[0]
            c = comet_scorer.score_batch([src], [cand], [ref], batch_size=1)[0]
            R = compute_reward(g, m, c)
            row_gemba.append(round(g, 6))
            row_mqm.append(round(m, 6))
            row_comet.append(round(c, 6))
            row_rewards.append(round(R, 6))

        best_idx   = int(np.argmax(row_rewards))
        best_trans = cands[best_idx]

        all_rewards.append(row_rewards)
        all_best_idx.append(best_idx)
        all_best_trans.append(best_trans)
        all_gemba.append(row_gemba)
        all_mqm.append(row_mqm)
        all_comet.append(row_comet)

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or (i + 1) == total:
            avg_r = np.mean([max(r) for r in all_rewards])
            logger.info(
                f"  Progress: {i+1}/{total} | "
                f"Avg best R: {avg_r:.4f} | "
                f"GEMBA: {row_gemba[best_idx]:.3f} | "
                f"MQM: {row_mqm[best_idx]:.3f} | "
                f"COMET: {row_comet[best_idx]:.3f}"
            )

        if (i + 1) % 200 == 0 or (i + 1) == total:
            ckpt_df = df.iloc[:i+1].copy()
            ckpt_df["rewards"] = all_rewards
            ckpt_df["gemba_scores"] = all_gemba
            ckpt_df["mqm_scores"] = all_mqm
            ckpt_df["comet_scores"] = all_comet
            ckpt_df["best_idx"] = all_best_idx
            ckpt_df["best_translation"] = all_best_trans
            ckpt_path = cfg["scored_file"].replace(
                ".csv", f"_ckpt_{i+1}.csv"
            )
            ckpt_df.to_csv(ckpt_path, index=False)
            logger.info(f"  Checkpoint: {ckpt_path}")

    df["rewards"] = all_rewards
    df["gemba_scores"] = all_gemba
    df["mqm_scores"] = all_mqm
    df["comet_scores"] = all_comet
    df["best_idx"] = all_best_idx
    df["best_translation"] = all_best_trans
    df.to_csv(cfg["scored_file"], index=False)
    logger.info(f"Scored data saved: {cfg['scored_file']}")

    best_rewards = [max(r) for r in all_rewards]
    logger.info("\n" + "=" * 60)
    logger.info("REWARD SCORING SUMMARY")
    logger.info(f"  Composite R — Mean: {np.mean(best_rewards):.4f} | "
                f"Min: {np.min(best_rewards):.4f} | "
                f"Max: {np.max(best_rewards):.4f}")
    logger.info("Phase 4 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()