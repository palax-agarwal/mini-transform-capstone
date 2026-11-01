"""Unit / behavioural tests for the NumPy transformer (run: python tests.py)."""
import numpy as np
from transformer import *
from train import *

res = []
def check(name, ok, detail=""):
    res.append((name, ok)); print(("PASS" if ok else "FAIL"), name, detail)

CTX["train"] = False
m = Transformer(len(SRC_VOCAB), len(TGT_VOCAB), p=0.0, seed=5)

# T1 shapes
src = pad([encode_pair(n)[0] for n in (7, 1234, 90)]); tgt = pad([encode_pair(n)[1] for n in (7, 1234, 90)])
lg = m.forward(src, tgt[:, :-1])
check("T1 output shape (B, T_tgt, vocab)", lg.shape == (3, tgt.shape[1] - 1, len(TGT_VOCAB)), str(lg.shape))

# T2 causal mask: changing a later target token must not change earlier logits
t2 = tgt[:, :-1].copy(); t2[:, 3] = (t2[:, 3] % 10) + 3
lg2 = m.forward(src, t2)
check("T2 causal mask (no look-ahead)", np.allclose(lg[:, :3], lg2[:, :3], atol=1e-12) and not np.allclose(lg[:, 3:], lg2[:, 3:]))

# T3 padding invariance: sequence alone vs padded inside a longer batch
a = m.forward(pad([encode_pair(7)[0]]), pad([encode_pair(7)[1]])[:, :-1])
b = m.forward(src, tgt[:, :-1])[0:1, : a.shape[1]]
check("T3 padding invariance (src + tgt pads masked)", np.allclose(a, b, atol=1e-10), "max diff %.1e" % np.abs(a - b).max())

# T4 batching invariance of greedy decoding
nums = [3, 45, 678, 9012, 1000]
bs = pad([encode_pair(n)[0] for n in nums])
batch = greedy_decode(m, bs, BOS, EOS)
single = [greedy_decode(m, pad([encode_pair(n)[0]]), BOS, EOS)[0] for n in nums]
check("T4 greedy decoding identical batched vs single", batch == single)

# T5 attention rows sum to 1 and masked keys get zero weight
m.forward(src, tgt[:, :-1])
A = m.enc_layers[0].att.attn
check("T5 attention rows sum to 1", np.allclose(A.sum(-1), 1.0))
k0 = (src[0] == 0).sum(); check("T5c  exact zero weight on pad keys", A[0, :, :, src.shape[1] - k0:].max() == 0.0 if k0 else True)

# T6 data integrity
def words_to_num(ws):
    vals = {w: i for i, w in enumerate(ONES)}; vals.update({w: 10 * i for i, w in enumerate(TENS) if w != "_"})
    tot, cur = 0, 0
    for w in ws:
        if w == "hundred": cur *= 100
        elif w == "thousand": tot += cur * 1000; cur = 0
        else: cur += vals[w]
    return tot + cur
check("T6 words -> number round trip for all 0..9999", all(words_to_num(to_words(n)) == n for n in range(10000)))
tr, va, te = make_splits(0)
check("T6b train/val/test disjoint & cover 10,000 numbers", len(set(tr) | set(va) | set(te)) == 10000 and not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te)), f"{len(tr)}/{len(va)}/{len(te)}")

# T7 overfit a single batch
CTX["train"] = True
mo = Transformer(len(SRC_VOCAB), len(TGT_VOCAB), p=0.0, seed=1); opt = Adam(mo.parameters())
pr = [encode_pair(int(n)) for n in tr[:16]]; s_, t_ = pad([p[0] for p in pr]), pad([p[1] for p in pr])
for i in range(1, 301):
    mo.zero_grad(); l, d = cross_entropy(mo.forward(s_, t_[:, :-1]), t_[:, 1:]); mo.backward(d); opt.step(noam_lr(i, 2e-3, 50))
check("T7 overfits a single batch (loss < 0.01 in 300 steps)", l < 0.01, "final loss %.5f" % l)

# T8 reproducibility
def short(seed):
    _, r, _, _ = run("rep", steps=100, seed=seed, log_every=100); return r["test"]["loss"]
a1, a2 = short(4), short(4)
check("T8 same seed -> bit-identical results", a1 == a2, f"{a1:.8f} vs {a2:.8f}")
print(f"\n{sum(o for _, o in res)}/{len(res)} tests passed")
