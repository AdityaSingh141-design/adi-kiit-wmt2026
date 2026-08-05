"""
=============================================================================
Script 06 — DPO Training with Single Reward Component (Ablation)
English → Assamese RLFT Pipeline | IIT Patna
=============================================================================
Ablation study: instead of composite reward R = 0.4*GEMBA + 0.4*MQM + 0.2*COMET
use only ONE component at a time to understand each component's contribution.

Usage:
    # GEMBA only
    python3 06_dpo_single_reward.py --config ../data2/config_v7.yaml --reward gemba

    # MQM (COMETKiwi) only
    python3 06_dpo_single_reward.py --config ../data2/config_v7.yaml --reward mqm

    # COMET-poly (XCOMET-XL) only
    python3 06_dpo_single_reward.py --config ../data2/config_v7.yaml --reward comet
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True,
                        choices=["gemba", "mqm", "comet"],
                        help="Which reward component to use alone")
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


class DPODatasetSingleReward(Dataset):
    """
    DPO Dataset using only one reward component for preference selection.
    Reuses the already-scored CSV from V7 — no re-scoring needed.
    """
    def __init__(self, scored_csv, reward_component,
                 tokenizer, ip, cfg, logger):
        df = pd.read_csv(scored_csv)

        # Parse list columns
        for col in ["candidates", "rewards", "gemba_scores",
                    "mqm_scores", "comet_scores"]:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(ast.literal_eval)

        # Select which score column to use
        score_col_map = {
            "gemba": "gemba_scores",
            "mqm":   "mqm_scores",
            "comet": "comet_scores"
        }
        score_col = score_col_map[reward_component]
        logger.info(f"Using reward component: {reward_component.upper()} "
                    f"(column: {score_col})")

        prompts, chosens, rejecteds = [], [], []
        skipped = 0

        for _, row in df.iterrows():
            cands  = row["candidates"]
            scores = row[score_col]   # single component scores

            if len(cands) < 2 or len(scores) < 2:
                skipped += 1
                continue
            c0, c1 = str(cands[0]).strip(), str(cands[1]).strip()
            s0, s1 = float(scores[0]), float(scores[1])
            if not c0 or not c1 or c0 == "nan" or c1 == "nan":
                skipped += 1
                continue
            if c0 == c1:
                skipped += 1
                continue
            if abs(s0 - s1) < 0.01:   # tighter threshold for single component
                skipped += 1
                continue
            chosen, rejected = (c0, c1) if s0 >= s1 else (c1, c0)
            prompts.append(row["src"])
            chosens.append(chosen)
            rejecteds.append(rejected)

        logger.info(f"Valid DPO pairs: {len(prompts)} (skipped {skipped})")
        if len(prompts) == 0:
            logger.error("No valid DPO pairs!")
            sys.exit(1)

        max_src = cfg["dpo_max_prompt_length"]
        max_tgt = cfg["dpo_max_target_length"]

        logger.info("Pre-tokenizing source (English)...")
        src_processed = ip.preprocess_batch(
            prompts, src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"]
        )
        prompt_enc = tokenizer(
            src_processed, max_length=max_src,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        logger.info("Pre-tokenizing chosen (Assamese)...")
        chosen_processed = ip.preprocess_batch(
            chosens, src_lang=cfg["tgt_lang"], tgt_lang=cfg["tgt_lang"]
        )
        chosen_enc = tokenizer(
            text_target=chosen_processed, max_length=max_tgt,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        logger.info("Pre-tokenizing rejected (Assamese)...")
        rejected_processed = ip.preprocess_batch(
            rejecteds, src_lang=cfg["tgt_lang"], tgt_lang=cfg["tgt_lang"]
        )
        rejected_enc = tokenizer(
            text_target=rejected_processed, max_length=max_tgt,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        self.prompt_input_ids        = prompt_enc["input_ids"]
        self.prompt_attention_mask   = prompt_enc["attention_mask"]
        self.chosen_input_ids        = chosen_enc["input_ids"]
        self.chosen_attention_mask   = chosen_enc["attention_mask"]
        self.rejected_input_ids      = rejected_enc["input_ids"]
        self.rejected_attention_mask = rejected_enc["attention_mask"]
        logger.info(f"Dataset built: {len(self.prompt_input_ids)} pairs")

    def __len__(self):
        return len(self.prompt_input_ids)

    def __getitem__(self, idx):
        return {
            "prompt_input_ids":        self.prompt_input_ids[idx],
            "prompt_attention_mask":   self.prompt_attention_mask[idx],
            "chosen_input_ids":        self.chosen_input_ids[idx],
            "chosen_attention_mask":   self.chosen_attention_mask[idx],
            "rejected_input_ids":      self.rejected_input_ids[idx],
            "rejected_attention_mask": self.rejected_attention_mask[idx],
        }


def get_logprobs(model, input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 require_grad=False):
    if require_grad:
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
    else:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )
    logits          = outputs.logits
    shift_logits    = logits[:, :-1, :].contiguous()
    shift_labels    = decoder_input_ids[:, 1:].contiguous()
    shift_mask      = decoder_attention_mask[:, 1:].contiguous().float()
    log_probs       = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask
    mean_log_probs  = token_log_probs.sum(-1) / \
                      shift_mask.sum(-1).clamp(min=1.0)
    return mean_log_probs


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    reward = args.reward  # gemba / mqm / comet

    # Output to a reward-specific adapter directory
    dpo_adapter_dir = cfg["dpo_adapter_dir"].replace(
        "eng-asm-v7", f"eng-asm-v7-{reward}"
    )
    log_dir = cfg["log_dir"].replace(
        "eng-asm-v7", f"eng-asm-v7-{reward}"
    )
    os.makedirs(dpo_adapter_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"06_dpo_{reward}_only.log")
    logger   = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info(f"DPO Training — Single Reward Ablation: {reward.upper()} only")
    logger.info(f"Scored file  : {cfg['scored_file']}")
    logger.info(f"SFT adapter  : {cfg['sft_adapter_dir']}")
    logger.info(f"DPO output   : {dpo_adapter_dir}")
    logger.info(f"DPO beta     : {cfg['dpo_beta']}")
    logger.info(f"DPO epochs   : {cfg['dpo_epochs']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    ip = IndicProcessor(inference=True)

    # Build dataset using single reward component
    logger.info("Building DPO dataset with single reward component...")
    dpo_dataset = DPODatasetSingleReward(
        cfg["scored_file"], reward,
        tokenizer, ip, cfg, logger
    )
    loader = DataLoader(
        dpo_dataset,
        batch_size=cfg["dpo_batch_size"],
        shuffle=True,
        drop_last=False
    )

    # Load policy model
    logger.info("Loading policy model (SFT LoRA)...")
    base1 = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True
    )
    if not hasattr(base1.config, "hidden_size"):
        base1.config.hidden_size = base1.config.decoder_embed_dim

    policy_model = PeftModel.from_pretrained(base1, cfg["sft_adapter_dir"])
    policy_model = policy_model.to(DEVICE)

    trainable = 0
    for name, param in policy_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    logger.info(f"  Trainable LoRA parameters: {trainable:,}")
    policy_model.train()

    # Load frozen reference model
    logger.info("Loading reference model (frozen)...")
    base2 = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32,
        device_map=None, trust_remote_code=True
    )
    if not hasattr(base2.config, "hidden_size"):
        base2.config.hidden_size = base2.config.decoder_embed_dim

    ref_model = PeftModel.from_pretrained(base2, cfg["sft_adapter_dir"])
    ref_model  = ref_model.to(DEVICE)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=float(cfg["dpo_learning_rate"])
    )

    beta     = cfg["dpo_beta"]
    best_acc = 0.0

    logger.info(f"Starting DPO training with {reward.upper()} reward only...")
    for epoch in range(cfg["dpo_epochs"]):
        epoch_losses = []
        epoch_accs   = []

        for step, batch in enumerate(loader):
            prompt_ids    = batch["prompt_input_ids"].to(DEVICE)
            prompt_mask   = batch["prompt_attention_mask"].to(DEVICE)
            chosen_ids    = batch["chosen_input_ids"].to(DEVICE)
            chosen_mask   = batch["chosen_attention_mask"].to(DEVICE)
            rejected_ids  = batch["rejected_input_ids"].to(DEVICE)
            rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

            policy_chosen_logp = get_logprobs(
                policy_model, prompt_ids, prompt_mask,
                chosen_ids, chosen_mask, require_grad=True
            )
            policy_rejected_logp = get_logprobs(
                policy_model, prompt_ids, prompt_mask,
                rejected_ids, rejected_mask, require_grad=True
            )
            ref_chosen_logp = get_logprobs(
                ref_model, prompt_ids, prompt_mask,
                chosen_ids, chosen_mask, require_grad=False
            )
            ref_rejected_logp = get_logprobs(
                ref_model, prompt_ids, prompt_mask,
                rejected_ids, rejected_mask, require_grad=False
            )

            chosen_reward   = beta * (policy_chosen_logp  - ref_chosen_logp)
            rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            acc  = (chosen_reward > rejected_reward).float().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy_model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc.item())

            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                logger.info(
                    f"  Epoch {epoch+1}/{cfg['dpo_epochs']} | "
                    f"Step {step+1}/{len(loader)} | "
                    f"loss={loss.item():.4f} | acc={acc.item():.3f}"
                )

        avg_loss = np.mean(epoch_losses)
        avg_acc  = np.mean(epoch_accs)
        logger.info(
            f"Epoch {epoch+1} complete | "
            f"avg_loss={avg_loss:.4f} | avg_acc={avg_acc:.3f}"
        )
        if avg_acc > best_acc:
            best_acc = avg_acc

    logger.info(f"Training complete. Best accuracy: {best_acc:.3f}")

    logger.info(f"Saving {reward.upper()}-only DPO adapter to {dpo_adapter_dir}...")
    policy_model.save_pretrained(dpo_adapter_dir)
    tokenizer.save_pretrained(dpo_adapter_dir)
    logger.info("Adapter saved.")
    logger.info(f"Phase complete — {reward.upper()} only DPO done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()