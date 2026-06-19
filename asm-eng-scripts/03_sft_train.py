"""
=============================================================================
Script 03 — Supervised Fine-Tuning (Phase 2)
English → Assamese RLFT Pipeline | IIT Patna
=============================================================================
Fine-tunes ai4bharat/indictrans2-en-indic-dist-200M using LoRA.

Key differences from indic-indic model:
  - src_lang = eng_Latn (English, Latin script)
  - tgt_lang = asm_Beng (Assamese, Bengali script)
  - model_name = indictrans2-en-indic-dist-200M
  - decoder_embed_dim = 512 (same as 320M indic-indic)

Bugs fixed (same as Hindi-Maithili):
  - hidden_size attribute missing → set from decoder_embed_dim
  - LoRA gradients not enabled → explicit requires_grad=True
  - IndicProcessor used ONCE before training loop

Usage:
    python3 03_sft_train.py --config ../configs/config_v7.yaml
=============================================================================
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
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


def load_data(src_path, tgt_path, logger):
    logger.info(f"Loading {src_path}")
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    assert len(src_lines) == len(tgt_lines), "Line count mismatch!"
    logger.info(f"  Loaded {len(src_lines)} pairs")
    return src_lines, tgt_lines


def tokenize_dataset(src_lines, tgt_lines, tokenizer, ip,
                     src_lang, tgt_lang, max_length, logger, split_name):
    logger.info(f"Tokenizing {split_name} ({len(src_lines)} pairs)...")

    # Preprocess with IndicProcessor ONCE before tokenization
    src_processed = ip.preprocess_batch(
        src_lines, src_lang=src_lang, tgt_lang=tgt_lang
    )
    tgt_processed = ip.preprocess_batch(
        tgt_lines, src_lang=tgt_lang, tgt_lang=tgt_lang
    )

    model_inputs = tokenizer(
        src_processed,
        text_target=tgt_processed,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    logger.info(f"  Tokenized {split_name}: {len(model_inputs['input_ids'])}")
    return model_inputs


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_path = os.path.join(cfg["log_dir"], "03_sft_train.log")
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("Phase 2: Supervised Fine-Tuning (SFT)")
    logger.info(f"Model    : {cfg['model_name']}")
    logger.info(f"src_lang : {cfg['src_lang']}")
    logger.info(f"tgt_lang : {cfg['tgt_lang']}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.backends.cuda.matmul.allow_tf32 = False

    # Resolve data paths
    data_dir = os.path.dirname(args.config)
    project_dir = os.path.dirname(data_dir)

    train_src_path = os.path.join(project_dir, "data2", cfg["train_src"])
    train_tgt_path = os.path.join(project_dir, "data2", cfg["train_tgt"])
    valid_src_path = os.path.join(project_dir, "data2", cfg["valid_src"])
    valid_tgt_path = os.path.join(project_dir, "data2", cfg["valid_tgt"])

    # Load data
    logger.info("Loading training data...")
    train_src, train_tgt = load_data(train_src_path, train_tgt_path, logger)
    valid_src, valid_tgt = load_data(valid_src_path, valid_tgt_path, logger)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True
    )

    # Load IndicProcessor ONCE — do all preprocessing before training loop
    logger.info("Loading IndicProcessor...")
    ip = IndicProcessor(inference=True)

    # Tokenize datasets
    train_encodings = tokenize_dataset(
        train_src, train_tgt, tokenizer, ip,
        cfg["src_lang"], cfg["tgt_lang"],
        cfg["max_length"], logger, "train"
    )
    valid_encodings = tokenize_dataset(
        valid_src, valid_tgt, tokenizer, ip,
        cfg["src_lang"], cfg["tgt_lang"],
        cfg["max_length"], logger, "valid"
    )

    train_dataset = HFDataset.from_dict(train_encodings)
    valid_dataset = HFDataset.from_dict(valid_encodings)

    logger.info(f"Tokenized train : {len(train_dataset)}")
    logger.info(f"Tokenized valid : {len(valid_dataset)}")

    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )

    # Fix missing hidden_size attribute (Bug 4)
    if not hasattr(model.config, "hidden_size"):
        model.config.hidden_size = model.config.decoder_embed_dim
        logger.info(f"  [Fix] hidden_size = {model.config.decoder_embed_dim}")

    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Explicitly enable LoRA gradients (Bug 3)
    trainable = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    logger.info(f"  Trainable LoRA parameters: {trainable:,}")
    model.print_trainable_parameters()

    # Training arguments
    os.makedirs(cfg["sft_adapter_dir"], exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["sft_adapter_dir"],
        num_train_epochs=cfg["sft_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=cfg["warmup_steps"],
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        predict_with_generate=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info(f"Saving SFT adapter to {cfg['sft_adapter_dir']}...")
    model.save_pretrained(cfg["sft_adapter_dir"])
    tokenizer.save_pretrained(cfg["sft_adapter_dir"])
    logger.info("SFT adapter saved.")
    logger.info("Phase 2 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()