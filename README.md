# ADI-IITP: English-Assamese MT for WMT 2026

This is our submission code for the WMT 2026 Shared Task on Low-Resource Indic Language Translation, English-Assamese pair.

📄 **Paper**: ADI-IITP: Reward-Guided Preference Optimization for Low-Resource English–Assamese Machine Translation (WMT 2026)

**Institution**: School of Computer Engineering, KIIT Deemed to be University & Department of Computer Science and Engineering, Indian Institute of Technology Patna, India

**Contact**: adityazn141@gmail.com

---

## Approach

We started from IndicTrans2's distilled 200M checkpoints and fine-tuned with LoRA rather than full fine-tuning, mostly because of compute constraints on our end and because Assamese parallel data is limited enough that full fine-tuning risked overfitting. On top of the SFT models we ran a preference optimization stage: generate candidates, score them with a composite reward, then train with DPO using the scored pairs as preference data.

The reward signal itself is a mix of three things — GEMBA (an LLM-as-judge score), a reference-free quality-estimation signal from CometKiwi, and a reference-based score from xCOMET. We combined these rather than relying on any single metric because each one seemed to catch different failure modes during early experiments.

Three systems went into the final submission:

- **Primary As→En** — SFT only, no DPO stage
- **Contrastive As→En** — SFT followed by DPO
- **Primary En→As** — SFT followed by DPO

We didn't submit an SFT+DPO As→En as primary since in our dev-set evaluations the SFT-only model actually scored competitively and we wanted a cleaner comparison point; the SFT+DPO version went in as contrastive instead.

## Systems Submitted

| System | Direction | Method | BLEU |
|---|---|---|---|
| Primary | Assamese → English | SFT only | 24.08 |
| Contrastive | Assamese → English | SFT + DPO | 25.11 |
| Primary | English → Assamese | SFT + DPO | 15.57 |

## Pipeline

1. **Data Preparation** — Cleaning, dedup, train/valid/test splits
2. **SFT Training** — LoRA fine-tuning of IndicTrans2 (200M distilled)
3. **Candidate Generation** — Greedy + nucleus sampling for 2,000 sentences
4. **Reward Scoring** — Composite reward: 0.4×GEMBA + 0.4×CometKiwi + 0.2×xCOMET
5. **DPO Training** — Direct Preference Optimization on scored pairs
6. **Evaluation** — BLEU, METEOR, TER, chrF++, BERTScore, COMET

## Model Configuration

- Base models: `ai4bharat/indictrans2-indic-en-dist-200M` (As→En), `ai4bharat/indictrans2-en-indic-dist-200M` (En→As)
- LoRA: rank 8, alpha 16, dropout 0.1, applied to `q_proj`
- SFT: learning rate 2e-5, early stopping on dev set
- DPO: beta 0.1, learning rate 1e-5, 3 epochs
- Final decoding: greedy (num_beams=1)

## Reward Components

| Component | Model | Paper |
|---|---|---|
| GEMBA | Qwen2.5-7B-Instruct (LLM judge) | Kocmi & Federmann (2023) |
| CometKiwi | Unbabel/wmt22-cometkiwi-da | Rei et al. (2022) |
| xCOMET | Unbabel/XCOMET-XL | Guerreiro et al. (2024) |

## Repository Layout

- `asm-eng-scripts/` — Assamese → English pipeline
- `eng-asm-scripts/` — English → Assamese pipeline
- `experiments/` — Exploratory and experimental scripts

## Scripts

| Script | What it does |
|---|---|
| `01_data_preparation.py` | Cleaning, dedup, train/valid/test splits |
| `03_sft_train.py` | LoRA supervised fine-tuning |
| `03_sft_train_earlystop.py` | SFT with early stopping on dev set |
| `04_generate_candidates.py` | Generates translation candidates for reward scoring |
| `05_reward_scoring.py` | Computes composite reward (GEMBA + CometKiwi + xCOMET) |
| `06_dpo_training.py` | DPO training on scored candidate pairs |
| `07_evaluate_final_wmt.py` | Final inference + evaluation |

*(Numbering skips 02 — an earlier data-augmentation experiment we didn't use.)*

## Setup Notes

- Update all file paths in `config_*.yaml` to match your local setup
- Set your HuggingFace token or use `huggingface-cli login`
- The `experiments/` folder contains exploratory scripts with hardcoded paths — provided as-is for reference

## Data

We use the official parallel data provided by the organizers of the WMT 2026 Low-Resource Indic Language Translation shared task for the English-Assamese pair. The training set contains 54,001 sentence pairs. Prior to fine-tuning, we applied a preprocessing pipeline consisting of duplicate removal, whitespace normalization, and additional cleaning and filtering; the retained pairs were then ordered by increasing sentence length. After preprocessing, 50,078 sentence pairs remained. The Assamese text is written in the Eastern Nagari (Bengali-Assamese) script.

## Citation

System description paper to appear in the WMT 2026 proceedings; citation details will be added here once available.
