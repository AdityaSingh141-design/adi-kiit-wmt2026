# ADI-KIIT: English-Assamese MT for WMT 2026

This is our submission code for the WMT 2026 Shared Task on Low-Resource Indic Language Translation, English-Assamese pair.

## Approach

We started from IndicTrans2's distilled 200M checkpoints and fine-tuned with LoRA rather than full fine-tuning, mostly because of compute constraints on our end and because Assamese parallel data is limited enough that full fine-tuning risked overfitting. On top of the SFT models we ran a preference optimization stage: generate candidates, score them with a composite reward, then train with DPO using the scored pairs as preference data.

The reward signal itself is a mix of three things — GEMBA (an LLM-as-judge score), a reference-free MQM-style error signal, and COMET. We combined these rather than relying on any single metric because each one seemed to catch different failure modes during early experiments (COMET was decent at fluency but missed some adequacy issues that the MQM signal picked up, for instance).

Three systems went into the final submission:

- Primary As→En — SFT only, no DPO stage
- Contrastive As→En — SFT followed by DPO, submitted as a contrast system
- Primary En→As — SFT followed by DPO

We didn't submit a DPO'd As→En as primary since in our dev-set evaluations the SFT-only model actually scored competitively and we wanted a cleaner comparison point; the DPO'd version went in as contrastive instead.

## Repository layout

```
asm-eng-scripts/   Assamese -> English pipeline
eng-asm-scripts/   English -> Assamese pipeline
```

Both folders mirror the same script structure, just pointed at different base checkpoints and data directions.

## Scripts

| Script | What it does |
|---|---|
| `01_data_preparation.py` | Cleaning, dedup, train/valid/test splits |
| `03_sft_train.py` | LoRA supervised fine-tuning |
| `03_sft_train_earlystop.py` | Same as above, with early stopping on the dev set |
| `04_generate_candidates.py` | Generates translation candidates for reward scoring |
| `05_reward_scoring.py` | Computes the composite reward (GEMBA + MQM + COMET) |
| `06_dpo_training.py` | DPO training on the scored candidate pairs |
| `07_evaluate_final_wmt.py` | Final inference + evaluation |

(Numbering starts at 01 and skips 02 — that step was an earlier data-augmentation experiment we ended up not using, but didn't bother renumbering around.)

## Model configuration

- Base models: `ai4bharat/indictrans2-indic-en-dist-200M` (As→En), `ai4bharat/indictrans2-en-indic-dist-200M` (En→As)
- LoRA: rank 8, alpha 16, dropout 0.1, applied to `q_proj`
- SFT: learning rate 2e-5
- DPO: beta 0.1, learning rate 1e-5, 3 epochs

## Citation

System description paper to appear in the WMT 2026 proceedings; citation details will be added here once available.
