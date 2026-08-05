import yaml, sys
import numpy as np
from transformers import AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

cfg = yaml.safe_load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
ip = IndicProcessor(inference=True)

with open(cfg["test_tgt"], encoding="utf-8") as f:
    refs = [l.strip() for l in f if l.strip()]

# Preprocess Assamese refs through IndicProcessor (treats asm as both src/tgt for tokenizing)
processed = ip.preprocess_batch(refs, src_lang=cfg["tgt_lang"], tgt_lang=cfg["tgt_lang"])

lengths = []
for p in processed:
    ids = tok(p, add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))

lengths = np.array(lengths)
cur = cfg["gen_max_new_tokens"]

print("=" * 55)
print(f"Reference token-length distribution (n={len(refs)})")
print("=" * 55)
print(f"  current gen_max_new_tokens : {cur}")
print(f"  mean   : {lengths.mean():.1f}")
print(f"  median : {np.median(lengths):.0f}")
print(f"  p90    : {np.percentile(lengths,90):.0f}")
print(f"  p95    : {np.percentile(lengths,95):.0f}")
print(f"  max    : {lengths.max()}")
print("-" * 55)
over = (lengths > cur).sum()
print(f"  refs LONGER than {cur} tokens : {over} ({100*over/len(refs):.1f}%)")
print(f"  refs longer than 64  : {(lengths>64).sum()}")
print(f"  refs longer than 128 : {(lengths>128).sum()}")
print("=" * 55)
safe = int(np.percentile(lengths, 99)) + 10
print(f"RECOMMENDED gen_max_new_tokens >= {safe}")
