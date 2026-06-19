# ADI-KIIT — WMT 2026 Low-Resource Indic MT (English-Assamese)

Submission for the WMT 2026 Shared Task on Low-Resource Indic Language Translation.
Systems built on the 200M distilled IndicTrans2 model, adapted with LoRA-based
supervised fine-tuning (SFT) and a preference-optimization (RLHF) pipeline using a
composite reward (GEMBA-style LLM-as-judge + reference-free MQM + COMET) and DPO.

## Pipeline (script_v7)
- 01_data_preparation.py : data preparation
- 03_sft_train.py / 03_sft_train_earlystop.py : LoRA supervised fine-tuning
- 04_generate_candidates.py : candidate generation
- 05_reward_scoring.py : composite reward scoring
- 06_dpo_training.py : Direct Preference Optimization
- 07_evaluate_final_wmt.py : evaluation / inference

asm-eng-scripts/ = Assamese to English
eng-asm-scripts/ = English to Assamese

Base model: ai4bharat/indictrans2-*-dist-200M. LoRA r=8, alpha=16, dropout=0.1 (q_proj).
