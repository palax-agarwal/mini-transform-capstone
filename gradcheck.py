"""Finite-difference check of every hand-written backward pass."""

import numpy as np
from transformer import *

CTX["train"] = False  # dropout off for a deterministic check
rng = np.random.default_rng(1)
m = Transformer(11, 9, d=8, h=2, dff=16, n_enc=2, n_dec=2, p=0.0, seed=3)
src = rng.integers(1, 11, (3, 5))
src[0, 3:] = 0
src[1, 4:] = 0
tgt = rng.integers(1, 9, (3, 6))
tgt[0, 4:] = 0
tin, tout = tgt[:, :-1], tgt[:, 1:]


def loss():
    return cross_entropy(m.forward(src, tin), tout)[0]


m.zero_grad()
l, d = cross_entropy(m.forward(src, tin), tout)
m.backward(d)
worst, eps, rows = 0, 1e-6, []
names = {}
for name, mod in [
    ("src_emb", m.src_emb),
    ("tgt_emb", m.tgt_emb),
    ("enc_layer0", m.enc_layers[0]),
    ("dec_layer1", m.dec_layers[1]),
    ("enc_ln", m.enc_ln),
    ("out", m.out),
]:
    w = 0
    wa = 0
    for p in mod.parameters():
        flat, g = p.data.reshape(-1), p.grad.reshape(-1)
        for i in rng.choice(flat.size, min(12, flat.size), replace=False):
            o = flat[i]
            flat[i] = o + eps
            lp = loss()
            flat[i] = o - eps
            lm = loss()
            flat[i] = o
            num = (lp - lm) / (2 * eps)
            ad = abs(num - g[i])
            wa = max(wa, ad)
            # relative error only where the gradient is non-negligible (|g|>1e-6); exactly-zero
            # grads (e.g. key bias: softmax is shift-invariant) are float round-off of the difference
            rel = (
                ad / (abs(num) + abs(g[i])) if max(abs(num), abs(g[i])) > 1e-6 else 0.0
            )
            w = max(w, rel)
    rows.append((name, w, wa))
    worst = max(worst, w)
for n, w, wa in rows:
    print(f"{n:12s} max relative error = {w:.2e}   max absolute error = {wa:.2e}")
print("OVERALL max relative error = %.2e" % worst)
print("PASS" if worst < 1e-5 else "FAIL")
