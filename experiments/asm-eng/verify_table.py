from sacrebleu.metrics import BLEU, CHRF

ASM_ENG = "/mnt/storage/boynao/aditya/exp2/rlft-asm-eng"
ENG_ASM = "/mnt/storage/boynao/aditya/exp2/rlft-eng-asm"

def lines(path):
    xs = open(path, encoding="utf-8").read().split("\n")
    return xs[:-1] if xs and xs[-1] == "" else xs

def score(hyp, ref):
    return (round(BLEU().corpus_score(hyp, [ref]).score, 2),
            round(CHRF(word_order=0).corpus_score(hyp, [ref]).score, 2))

# As->En : correct half = LAST 1000 (Assamese-origin)
ref = lines(f"{ASM_ENG}/data2/test2/test.eng_Latn")[1000:]
for tag in ["asm-eng-zeroshot-test2", "asm-eng-bestsft-dpo-test2"]:
    h = lines(f"{ASM_ENG}/models2/{tag}/logs/{tag}_hypotheses.txt")[1000:]
    print(f"As->En {tag:30s} BLEU/chrF =", score(h, ref))

# En->As : correct half = FIRST 1000 (English-origin)
ref = lines(f"{ENG_ASM}/data2/test2/test.asm_Beng")[:1000]
for tag in ["eng-asm-zeroshot-test2", "eng-asm-bestsft-dpo-test2"]:
    h = lines(f"{ENG_ASM}/models2/{tag}/logs/{tag}_hypotheses.txt")[:1000]
    print(f"En->As {tag:30s} BLEU/chrF =", score(h, ref))
