"""
=============================================================================
Script 03-ES — SFT with Validation-Based Early Stopping (chrF++ selection)
English<->Assamese RLFT Pipeline | IIT Patna
=============================================================================
Upgrade of 03_sft_train.py. Identical data handling and model setup, but:
  - Evaluates chrF++ on the validation set each epoch (predict_with_generate)
  - Keeps the checkpoint with the BEST validation chrF++ (not best loss)
  - EarlyStoppingCallback halts when validation chrF++ stops improving
  - LR and MAX-epochs are CLI flags, so we can sweep cleanly
  - Tags output dir per run so sweep runs don't overwrite each other

This selects on translation quality, tuned on VALIDATION (never test).

Usage:
    # single run
    python3 03_sft_train_earlystop.py --config ../data2/config_v7.yaml \
        --learning_rate 5e-5 --epochs 6 --run_tag lr5e5

    # the value printed at the end ("BEST VALID chrF++") is what to compare
=============================================================================
"""

import os, sys, shutil, logging, argparse, yaml, torch
import numpy as np
import sacrebleu
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from IndicTransToolkit.processor import IndicProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Override cfg[learning_rate]")
    p.add_argument("--epochs", type=int, default=None,
                   help="MAX epochs (early stopping may halt sooner)")
    p.add_argument("--patience", type=int, default=2,
                   help="Early-stopping patience (epochs w/o chrF++ gain)")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Suffix for output dir, e.g. lr5e5")
    return p.parse_args()


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)


def load_data(src_path, tgt_path, logger):
    with open(src_path, encoding="utf-8") as f:
        s = [l.strip() for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        t = [l.strip() for l in f if l.strip()]
    assert len(s) == len(t), "Line count mismatch!"
    logger.info(f"  Loaded {len(s)} pairs")
    return s, t


def tokenize_dataset(src, tgt, tokenizer, ip, src_lang, tgt_lang, max_len, logger, name):
    logger.info(f"Tokenizing {name} ({len(src)} pairs)...")
    sp = ip.preprocess_batch(src, src_lang=src_lang, tgt_lang=tgt_lang)
    tp = ip.preprocess_batch(tgt, src_lang=tgt_lang, tgt_lang=tgt_lang)
    enc = tokenizer(sp, text_target=tp, max_length=max_len,
                    truncation=True, padding=False)
    return enc


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    lr = args.learning_rate if args.learning_rate is not None else float(cfg["learning_rate"])
    max_epochs = args.epochs if args.epochs is not None else cfg["sft_epochs"]

    # Tagged FINAL output dir on /mnt/storage (network mount)
    out_dir = cfg["sft_adapter_dir"]
    if args.run_tag:
        out_dir = out_dir.rstrip("/") + "_" + args.run_tag
    os.makedirs(out_dir, exist_ok=True)

    # Trainer writes per-epoch checkpoints to FAST LOCAL disk to avoid
    # network-mount stalls (/mnt/storage has hung mid-save). Only the final
    # best adapter is copied to out_dir at the end.
    local_work = os.path.join("/tmp", "sft_work_" + (args.run_tag or "default"))
    if os.path.exists(local_work):
        shutil.rmtree(local_work)
    os.makedirs(local_work, exist_ok=True)

    tag = args.run_tag or "default"
    logger = setup_logger(os.path.join(cfg["log_dir"], f"03_es_{tag}.log"))

    logger.info("=" * 60)
    logger.info("Phase 2-ES: SFT with chrF++ early stopping")
    logger.info(f"Run tag      : {tag}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Max epochs   : {max_epochs}  (patience {args.patience})")
    logger.info(f"Output       : {out_dir}")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    data_dir = os.path.dirname(args.config)
    project_dir = os.path.dirname(data_dir)
    tr_s = os.path.join(project_dir, "data2", cfg["train_src"])
    tr_t = os.path.join(project_dir, "data2", cfg["train_tgt"])
    va_s = os.path.join(project_dir, "data2", cfg["valid_src"])
    va_t = os.path.join(project_dir, "data2", cfg["valid_tgt"])

    logger.info("Loading data...")
    train_src, train_tgt = load_data(tr_s, tr_t, logger)
    valid_src, valid_tgt = load_data(va_s, va_t, logger)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    train_enc = tokenize_dataset(train_src, train_tgt, tokenizer, ip,
        cfg["src_lang"], cfg["tgt_lang"], cfg["max_length"], logger, "train")
    valid_enc = tokenize_dataset(valid_src, valid_tgt, tokenizer, ip,
        cfg["src_lang"], cfg["tgt_lang"], cfg["max_length"], logger, "valid")
    train_ds = HFDataset.from_dict(train_enc)
    valid_ds = HFDataset.from_dict(valid_enc)

    logger.info("Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"],
        torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
    if not hasattr(model.config, "hidden_size"):
        model.config.hidden_size = model.config.decoder_embed_dim

    lora = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"], target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"], bias="none")
    model = get_peft_model(model, lora)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.print_trainable_parameters()

    # chrF++ on validation, compared against RAW references (in eval order)
    def compute_metrics(eval_preds):
        # NOTE: chrF++ computed on RAW decoded text (no IndicProcessor
        # postprocess) — IndicProcessor.postprocess_batch deadlocks inside
        # the Trainer eval loop. For early-stopping we only need to RANK
        # epochs, so the absolute chrF++ value is unimportant; skipping
        # postprocess removes the deadlock and preserves the ranking.
        preds, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        dec = tokenizer.batch_decode(preds, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(dec, [valid_tgt], word_order=2)
        return {"chrf": round(chrf.score, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_work,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        warmup_steps=cfg["warmup_steps"],
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        predict_with_generate=True,
        generation_max_length=cfg["gen_max_new_tokens"],
        generation_num_beams=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=8)

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=valid_ds,
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    logger.info("Training (chrF++ early stopping)...")
    trainer.train()

    # Best model already loaded (load_best_model_at_end). Save it to the
    # real network location once, then clean up local scratch.
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    try:
        shutil.rmtree(local_work)
    except Exception:
        pass

    best = trainer.state.best_metric
    logger.info("=" * 60)
    logger.info(f"RUN {tag} | LR {lr} | BEST VALID chrF++: {best}")
    logger.info(f"Best adapter saved: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()