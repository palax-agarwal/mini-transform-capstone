"""Toy translation task: English number words -> digit string (0-9999).
Train / evaluate the NumPy transformer.  Usage: python train.py <name> [key=value ...]
"""

import sys, json, time
import numpy as np
from transformer import *

ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split()
)
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def to_words(n):
    if n == 0:
        return ["zero"]
    w = []
    if n >= 1000:
        w += [ONES[n // 1000], "thousand"]
        n %= 1000
    if n >= 100:
        w += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        w.append(TENS[n // 10])
        n %= 10
        if n:
            w.append(ONES[n])
    elif n:
        w.append(ONES[n])
    return w


SRC_VOCAB = ["<pad>"] + sorted(set(ONES + TENS[2:] + ["hundred", "thousand"]))
TGT_VOCAB = ["<pad>", "<bos>", "<eos>"] + list("0123456789")
S2I = {w: i for i, w in enumerate(SRC_VOCAB)}
T2I = {w: i for i, w in enumerate(TGT_VOCAB)}
BOS, EOS = T2I["<bos>"], T2I["<eos>"]


def encode_pair(n):
    return [S2I[w] for w in to_words(n)], [BOS] + [T2I[c] for c in str(n)] + [EOS]


def pad(seqs):
    L = max(len(s) for s in seqs)
    return np.array([s + [0] * (L - len(s)) for s in seqs])


def make_splits(seed=0, n_max=10000):
    nums = np.random.default_rng(seed).permutation(n_max)
    a, b = int(0.8 * n_max), int(0.9 * n_max)
    return nums[:a], nums[a:b], nums[b:]


def evaluate(model, nums, bs=500):
    em, tok, tot, loss_sum, n_tok = 0, 0, 0, 0.0, 0
    preds = []
    for i in range(0, len(nums), bs):
        chunk = nums[i : i + bs]
        pairs = [encode_pair(int(n)) for n in chunk]
        src, tgt = pad([p[0] for p in pairs]), pad([p[1] for p in pairs])
        CTX["train"] = False
        logits = model.forward(src, tgt[:, :-1])
        l, _ = cross_entropy(logits, tgt[:, 1:])
        m = (tgt[:, 1:] != 0).sum()
        loss_sum += l * m
        n_tok += m
        out = greedy_decode(model, src, BOS, EOS, max_len=8)
        for n, o in zip(chunk, out):
            gold = [T2I[c] for c in str(int(n))]
            em += o == gold
            tok += sum(a == b for a, b in zip(o, gold)) / max(len(gold), len(o))
            preds.append(o)
        tot += len(chunk)
    CTX["train"] = True
    return dict(loss=float(loss_sum / n_tok), exact=em / tot, token=tok / tot), preds


def run(
    name, steps=6000, bs=64, peak_lr=1.5e-3, warmup=400, seed=0, log_every=200, **mk
):
    CTX["rng"] = np.random.default_rng(seed + 100)
    train, val, test = make_splits(0)
    model = Transformer(len(SRC_VOCAB), len(TGT_VOCAB), seed=seed, **mk)
    opt = Adam(model.parameters())
    rng = np.random.default_rng(seed)
    hist = dict(step=[], train_loss=[], val_loss=[], val_exact=[])
    t0, run_loss = time.time(), []
    for step in range(1, steps + 1):
        idx = rng.choice(len(train), bs, replace=False)
        pairs = [encode_pair(int(train[i])) for i in idx]
        src, tgt = pad([p[0] for p in pairs]), pad([p[1] for p in pairs])
        model.zero_grad()
        loss, d = cross_entropy(model.forward(src, tgt[:, :-1]), tgt[:, 1:])
        model.backward(d)
        opt.step(noam_lr(step, peak_lr, warmup))
        run_loss.append(loss)
        if step % log_every == 0 or step == steps:
            v, _ = evaluate(model, val[:500])
            hist["step"].append(step)
            hist["train_loss"].append(float(np.mean(run_loss)))
            hist["val_loss"].append(v["loss"])
            hist["val_exact"].append(v["exact"])
            print(
                f"[{name}] step {step:5d} train {np.mean(run_loss):.4f} val {v['loss']:.4f} "
                f"exact {v['exact']:.3f}  ({time.time()-t0:.0f}s)",
                flush=True,
            )
            run_loss = []
    final, preds = evaluate(model, test)
    res = dict(
        name=name,
        params=model.n_params(),
        test=final,
        hist=hist,
        seconds=time.time() - t0,
        cfg=dict(steps=steps, bs=bs, peak_lr=peak_lr, seed=seed, **mk),
    )
    print(
        f"[{name}] TEST loss {final['loss']:.4f} exact {final['exact']:.4f} token {final['token']:.4f}",
        flush=True,
    )
    return model, res, test, preds


if __name__ == "__main__":
    name = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("=")
        kw[k] = (
            (v == "True")
            if v in ("True", "False")
            else (float(v) if "." in v or "e" in v else int(v))
        )
    model, res, test, preds = run(name, **kw)
    json.dump(res, open(f"results_{name}.json", "w"))
    np.savez(
        f"model_{name}.npz",
        **{f"p{i}": p.data for i, p in enumerate(model.parameters())},
    )
