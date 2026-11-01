"""Mini Transformer (encoder-decoder) built from scratch in NumPy.

Every layer implements forward() and a hand-derived backward(); no autograd
and no deep-learning library is used.  Arrays are batch-first (B, T, D).
"""

import numpy as np

CTX = {"train": True, "rng": np.random.default_rng(0)}  # global train/eval flag


class Parameter:
    def __init__(self, data):
        self.data = data.astype(np.float64)
        self.grad = np.zeros_like(self.data)


class Module:
    def parameters(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                out.append(v)
            elif isinstance(v, Module):
                out += v.parameters()
            elif isinstance(v, list):
                for m in v:
                    if isinstance(m, Module):
                        out += m.parameters()
        return out


def xavier(rng, n_in, n_out):
    lim = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-lim, lim, (n_in, n_out))


class Linear(Module):
    def __init__(self, rng, n_in, n_out):
        self.W = Parameter(xavier(rng, n_in, n_out))
        self.b = Parameter(np.zeros(n_out))

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.W.grad += x2.T @ dy2
        self.b.grad += dy2.sum(0)
        return dy @ self.W.data.T


class LayerNorm(Module):
    def __init__(self, d, eps=1e-5):
        self.g = Parameter(np.ones(d))
        self.b = Parameter(np.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        self.inv = 1.0 / np.sqrt(var + self.eps)
        self.xh = (x - mu) * self.inv
        return self.g.data * self.xh + self.b.data

    def backward(self, dy):
        self.g.grad += (dy * self.xh).reshape(-1, dy.shape[-1]).sum(0)
        self.b.grad += dy.reshape(-1, dy.shape[-1]).sum(0)
        dxh = dy * self.g.data
        return self.inv * (
            dxh
            - dxh.mean(-1, keepdims=True)
            - self.xh * (dxh * self.xh).mean(-1, keepdims=True)
        )


class Dropout(Module):
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        if not CTX["train"] or self.p == 0:
            self.mask = None
            return x
        self.mask = (CTX["rng"].random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask


class Embedding(Module):
    def __init__(self, rng, vocab, d):
        self.E = Parameter(rng.normal(0, d**-0.5, (vocab, d)))
        self.scale = np.sqrt(d)

    def forward(self, ids):
        self.ids = ids
        return self.E.data[ids] * self.scale

    def backward(self, dy):
        np.add.at(self.E.grad, self.ids, dy * self.scale)


def positional_encoding(max_len, d):
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / np.power(10000.0, i / d)
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


class MultiHeadAttention(Module):
    """Scaled dot-product attention with h heads.  mask: True = may attend."""

    def __init__(self, rng, d, h):
        assert d % h == 0
        self.h, self.d, self.dk = h, d, d // h
        self.wq, self.wk = Linear(rng, d, d), Linear(rng, d, d)
        self.wv, self.wo = Linear(rng, d, d), Linear(rng, d, d)

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dk).transpose(0, 2, 1, 3)

    def forward(self, xq, xkv, mask):
        q, k, v = (
            self._split(self.wq.forward(xq)),
            self._split(self.wk.forward(xkv)),
            self._split(self.wv.forward(xkv)),
        )
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        s = np.where(mask, s, -1e9)
        a = softmax(s)
        ctx = a @ v
        B, _, Tq, _ = ctx.shape
        self.q, self.k, self.v, self.a = q, k, v, a
        self.attn = a  # kept for visualisation
        return self.wo.forward(ctx.transpose(0, 2, 1, 3).reshape(B, Tq, self.d))

    def backward(self, dout):
        B, Tq, _ = dout.shape
        dctx = (
            self.wo.backward(dout).reshape(B, Tq, self.h, self.dk).transpose(0, 2, 1, 3)
        )
        a, q, k, v = self.a, self.q, self.k, self.v
        da = dctx @ v.transpose(0, 1, 3, 2)
        dv = a.transpose(0, 1, 3, 2) @ dctx
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        merge = lambda t: t.transpose(0, 2, 1, 3).reshape(
            t.shape[0], t.shape[2], self.d
        )
        dxq = self.wq.backward(merge(dq))
        dxkv = self.wk.backward(merge(dk)) + self.wv.backward(merge(dv))
        return dxq, dxkv


class FeedForward(Module):
    def __init__(self, rng, d, dff):
        self.l1, self.l2 = Linear(rng, d, dff), Linear(rng, dff, d)

    def forward(self, x):
        self.h = np.maximum(self.l1.forward(x), 0)
        return self.l2.forward(self.h)

    def backward(self, dy):
        dh = self.l2.backward(dy) * (self.h > 0)
        return self.l1.backward(dh)


class EncoderLayer(Module):
    def __init__(self, rng, d, h, dff, p):
        self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
        self.att, self.ff = MultiHeadAttention(rng, d, h), FeedForward(rng, d, dff)
        self.dr1, self.dr2 = Dropout(p), Dropout(p)

    def forward(self, x, mask):
        n = self.ln1.forward(x)
        x = x + self.dr1.forward(self.att.forward(n, n, mask))
        return x + self.dr2.forward(self.ff.forward(self.ln2.forward(x)))

    def backward(self, dx):
        d2 = self.ln2.backward(self.ff.backward(self.dr2.backward(dx)))
        dx = dx + d2
        dq, dkv = self.att.backward(self.dr1.backward(dx))
        return dx + self.ln1.backward(dq + dkv)


class DecoderLayer(Module):
    def __init__(self, rng, d, h, dff, p):
        self.ln1, self.ln2, self.ln3 = LayerNorm(d), LayerNorm(d), LayerNorm(d)
        self.self_att, self.cross_att = MultiHeadAttention(
            rng, d, h
        ), MultiHeadAttention(rng, d, h)
        self.ff = FeedForward(rng, d, dff)
        self.dr1, self.dr2, self.dr3 = Dropout(p), Dropout(p), Dropout(p)

    def forward(self, x, enc, self_mask, cross_mask):
        n = self.ln1.forward(x)
        x = x + self.dr1.forward(self.self_att.forward(n, n, self_mask))
        x = x + self.dr2.forward(
            self.cross_att.forward(self.ln2.forward(x), enc, cross_mask)
        )
        return x + self.dr3.forward(self.ff.forward(self.ln3.forward(x)))

    def backward(self, dx):
        dx = dx + self.ln3.backward(self.ff.backward(self.dr3.backward(dx)))
        dq, denc = self.cross_att.backward(self.dr2.backward(dx))
        dx = dx + self.ln2.backward(dq)
        dq, dkv = self.self_att.backward(self.dr1.backward(dx))
        return dx + self.ln1.backward(dq + dkv), denc


class Transformer(Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d=64,
        h=4,
        dff=128,
        n_enc=2,
        n_dec=2,
        p=0.1,
        max_len=64,
        use_pe=True,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.d, self.use_pe = d, use_pe
        self.pe = positional_encoding(max_len, d)
        self.src_emb, self.tgt_emb = Embedding(rng, src_vocab, d), Embedding(
            rng, tgt_vocab, d
        )
        self.enc_layers = [EncoderLayer(rng, d, h, dff, p) for _ in range(n_enc)]
        self.dec_layers = [DecoderLayer(rng, d, h, dff, p) for _ in range(n_dec)]
        self.enc_ln, self.dec_ln = LayerNorm(d), LayerNorm(d)
        self.out = Linear(rng, d, tgt_vocab)
        self.drop_in_s, self.drop_in_t = Dropout(p), Dropout(p)

    # ---- masks (True = attention allowed) ----
    @staticmethod
    def masks(src, tgt_in, pad_s=0, pad_t=0):
        src_mask = (src != pad_s)[:, None, None, :]
        T = tgt_in.shape[1]
        causal = np.tril(np.ones((T, T), bool))[None, None]
        self_mask = causal & (tgt_in != pad_t)[:, None, None, :]
        return src_mask, self_mask

    def encode(self, src, src_mask):
        x = self.src_emb.forward(src)
        if self.use_pe:
            x = x + self.pe[: src.shape[1]]
        x = self.drop_in_s.forward(x)
        for l in self.enc_layers:
            x = l.forward(x, src_mask)
        return self.enc_ln.forward(x)

    def decode(self, tgt_in, enc, src_mask, self_mask):
        x = self.tgt_emb.forward(tgt_in)
        if self.use_pe:
            x = x + self.pe[: tgt_in.shape[1]]
        x = self.drop_in_t.forward(x)
        for l in self.dec_layers:
            x = l.forward(x, enc, self_mask, src_mask)
        return self.out.forward(self.dec_ln.forward(x))

    def forward(self, src, tgt_in):
        src_mask, self_mask = self.masks(src, tgt_in)
        enc = self.encode(src, src_mask)
        return self.decode(tgt_in, enc, src_mask, self_mask)

    def backward(self, dlogits):
        dx = self.dec_ln.backward(self.out.backward(dlogits))
        denc = 0
        for l in reversed(self.dec_layers):
            dx, de = l.backward(dx)
            denc = denc + de
        self.tgt_emb.backward(self.drop_in_t.backward(dx))
        dx = self.enc_ln.backward(denc)
        for l in reversed(self.enc_layers):
            dx = l.backward(dx)
        self.src_emb.backward(self.drop_in_s.backward(dx))

    def zero_grad(self):
        for p in self.parameters():
            p.grad[...] = 0

    def n_params(self):
        return sum(p.data.size for p in self.parameters())


def cross_entropy(logits, target, pad=0):
    """Mean token cross-entropy ignoring pads; returns loss and dlogits."""
    p = softmax(logits)
    m = target != pad
    n = m.sum()
    idx = np.where(m, target, 0)
    ll = -np.log(np.take_along_axis(p, idx[..., None], -1)[..., 0] + 1e-12)
    loss = (ll * m).sum() / n
    d = p.copy()
    np.put_along_axis(
        d, idx[..., None], np.take_along_axis(d, idx[..., None], -1) - 1, -1
    )
    return loss, d * m[..., None] / n


class Adam:
    def __init__(self, params, b1=0.9, b2=0.98, eps=1e-9, clip=1.0):
        self.p, self.b1, self.b2, self.eps, self.clip, self.t = (
            params,
            b1,
            b2,
            eps,
            clip,
            0,
        )
        self.m = [np.zeros_like(q.data) for q in params]
        self.v = [np.zeros_like(q.data) for q in params]

    def step(self, lr):
        self.t += 1
        norm = np.sqrt(sum((q.grad**2).sum() for q in self.p))
        sc = min(1.0, self.clip / (norm + 1e-12))
        for i, q in enumerate(self.p):
            g = q.grad * sc
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            q.data -= lr * mh / (np.sqrt(vh) + self.eps)
        return norm


def noam_lr(step, peak, warmup):
    return peak * min(step / warmup, np.sqrt(warmup / step))


def greedy_decode(model, src, bos, eos, max_len=10, pad=0):
    """Autoregressive greedy decoding for a batch; returns list of id lists."""
    CTX["train"] = False
    B = src.shape[0]
    src_mask = (src != 0)[:, None, None, :]
    enc = model.encode(src, src_mask)
    ys = np.full((B, 1), bos)
    done = np.zeros(B, bool)
    for _ in range(max_len):
        T = ys.shape[1]
        self_mask = np.tril(np.ones((T, T), bool))[None, None]
        nxt = model.decode(ys, enc, src_mask, self_mask)[:, -1].argmax(-1)
        nxt = np.where(done, pad, nxt)
        ys = np.concatenate([ys, nxt[:, None]], 1)
        done |= nxt == eos
        if done.all():
            break
    CTX["train"] = True
    out = []
    for row in ys[:, 1:]:
        r = []
        for t in row:
            if t in (eos, pad):
                break
            r.append(int(t))
        out.append(r)
    return out
