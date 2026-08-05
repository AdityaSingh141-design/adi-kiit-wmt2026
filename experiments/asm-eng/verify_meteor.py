import nltk
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

ASM_ENG = "/mnt/storage/boynao/aditya/exp2/rlft-asm-eng"
ENG_ASM = "/mnt/storage/boynao/aditya/exp2/rlft-eng-asm"

def lines(p):
    xs = open(p, encoding="utf-8").read().split("\n")
    return xs[:-1] if xs and xs[-1]=="" else xs

def meteor(hyp, ref):
    s = [meteor_score([r.split()], h.split()) for h, r in zip(hyp, ref)]
    return round(sum(s)/len(s), 4)

# As->En : Assamese-origin half = LAST 1000
ref = lines(f"{ASM_ENG}/data2/test2/test.eng_Latn")[1000:]
for tag in ["asm-eng-zeroshot-test2", "asm-eng-bestsft-dpo-test2"]:
    h = lines(f"{ASM_ENG}/models2/{tag}/logs/{tag}_hypotheses.txt")[1000:]
    print(f"As->En {tag:28s} METEOR =", meteor(h, ref))

# En->As : English-origin half = FIRST 1000
ref = lines(f"{ENG_ASM}/data2/test2/test.asm_Beng")[:1000]
for tag in ["eng-asm-zeroshot-test2", "eng-asm-bestsft-dpo-test2"]:
    h = lines(f"{ENG_ASM}/models2/{tag}/logs/{tag}_hypotheses.txt")[:1000]
    print(f"En->As {tag:28s} METEOR =", meteor(h, ref))
