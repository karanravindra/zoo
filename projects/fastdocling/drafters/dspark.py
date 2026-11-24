"""DSpark (arXiv 2607.05147): DFlash's parallel block backbone plus a light sequential head and a confidence head.

Differences from `dflash.py`:
- The anchor is itself a prediction position: a block of k inputs (anchor + k-1 masks) gives k outputs, output j predicting
  x[t+2+j], the same offsets as every other drafter here.
- Low-rank Markov head: logits_j = U_j + W1[prev_j] @ W2, where U_j is the frozen LM head on the backbone output and prev_j is the
  previous token of the block (x[t+1+j]; teacher-forced in training, the sampled draft at inference). The paper's W1 and W2 are
  V x r and r x V (r = 256 by default), the dominant parameter cost with a 100k vocabulary: `--markov-rank` shrinks it.
- Confidence head: c_j = sigmoid(w . [h_j; W1[prev_j]]), trained to the soft label 1 - TV(p_draft, p_target) / 2, which is the
  chance the draft token survives verification. At inference it is left in `last_conf` for a scheduler to choose how many tokens
  to verify. The scheduler and the paper's calibration step are not implemented here yet; they belong with `bench`.
- Loss = 0.1 CE + 0.9 TV + 1.0 confidence BCE, all weighted by exp(-j / gamma) over the block. TV is against the target's own
  next-token distribution at the same position (its final hidden state through the frozen LM head).

Not in the paper, so chosen here: teacher forcing for the Markov bias in training; the `loss_gamma` default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from drafters.base import DraftOut, Drafter, Features, register, shifted_targets
from drafters.dflash import Backbone, extend_kv, position_weights, sample_anchors


@register("dspark")
class DSpark(Drafter):
    def __init__(self, cfg, hidden_size: int, n_feature_layers: int, vocab_size: int = 100352):
        super().__init__()
        H, r = hidden_size, cfg.markov_rank
        self.k, self.ce_chunk, self.anchors = cfg.k, cfg.ce_chunk, cfg.anchors
        self.gamma = cfg.loss_gamma or max(2.0, (cfg.k + 1) / 2)
        self.bb = Backbone(H, n_feature_layers, cfg.n_layers)
        self.w1 = nn.Embedding(vocab_size, r)  # previous token -> r
        self.w2 = nn.Parameter(torch.zeros(r, vocab_size))  # r -> a bias over the vocabulary (zero: starts as plain DFlash)
        nn.init.normal_(self.w1.weight, std=0.02)
        self.conf = nn.Linear(H + r, 1)
        self.last_conf: torch.Tensor | None = None

    def _chunk(self, weight, h, left, tg, th, w):
        """CE and TV of one chunk of rows (logits are rebuilt in backward, never stored for the whole batch)."""
        logits = (h @ weight.t().to(h.dtype)).float() + (left @ self.w2.to(left.dtype)).float()
        with torch.no_grad():
            pt = F.softmax((th @ weight.t().to(th.dtype)).float(), dim=-1)
        pd = F.softmax(logits, dim=-1)
        tv = (pd - pt).abs().sum(-1)
        ce = F.cross_entropy(logits, tg, reduction="none")
        return (ce * w).sum(), (tv * w).sum(), logits.argmax(-1) == tg, tv.detach()

    def forward(self, feats: Features) -> DraftOut:
        ids = feats.input_ids
        B, T = ids.shape
        k = self.k
        dev = ids.device
        targets, valid = shifted_targets(ids, feats.mask, k)
        A = self.anchors if self.training else 4 * self.anchors
        t = sample_anchors(valid.any(-1), A, None if self.training else 0)  # [B, A]
        nxt = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
        out = self.bb.run(self.bb.context(feats.hidden), nxt.gather(1, t), t, k, feats.embed)  # [B, A, k, H]
        b = torch.arange(B, device=dev)[:, None]
        v = valid[b, t]  # [B, A, k]
        pos = (t[:, :, None] + 1 + torch.arange(k, device=dev)).clamp(max=T - 1)  # position of x[t+1+j]
        prev_tok = ids[b[:, :, None], pos]  # the token before each prediction
        tgt_h = feats.last[b[:, :, None], pos]  # the target's state there, whose distribution is p^t for x[t+2+j]
        w = position_weights(k, self.gamma, dev).expand(B, A, k)

        h, prev, tg, th, wr = out[v], prev_tok[v], targets[b, t][v], tgt_h[v], w[v]
        left = self.w1(prev)  # [N, r]
        ce = tvs = h.new_zeros((), dtype=torch.float32)
        matches, tv_rows = [], []
        for i in range(0, len(h), self.ce_chunk):
            sl = slice(i, i + self.ce_chunk)
            c, tvw, m, tv = checkpoint(self._chunk, feats.embed.weight, h[sl], left[sl], tg[sl], th[sl], wr[sl], use_reentrant=False)
            ce, tvs = ce + c, tvs + tvw
            matches.append(m)
            tv_rows.append(tv)
        tv_rows = torch.cat(tv_rows) if tv_rows else h.new_zeros(0)
        c_logit = self.conf(torch.cat([h, left], dim=-1)).squeeze(-1)
        label = (1 - 0.5 * tv_rows).clamp(0, 1)  # chance this token is accepted given the ones before it were
        bce = (F.binary_cross_entropy_with_logits(c_logit.float(), label, reduction="none") * wr).sum()
        den = wr.sum().clamp(min=1e-6)
        loss = (0.1 * ce + 0.9 * tvs + 1.0 * bce) / den

        match = torch.zeros(B, A, k, dtype=torch.bool, device=dev)
        match[v] = torch.cat(matches) if matches else match.new_zeros(0)
        match, v2 = match.view(B * A, k), v.view(B * A, k)
        rows = v2.any(-1)
        logs = {"ce": (ce / den).detach(), "tv": (tvs / den).detach(), "conf_bce": (bce / den).detach(),
                "conf_mean": torch.sigmoid(c_logit).mean().detach(), "conf_label": label.mean()}
        return DraftOut(loss, match[rows], v2[rows], logs)

    def _draft(self, h, anchor_tok, embed, k):
        """The sequential part: block outputs h [B, k, H] -> tokens, each biased by the one before (a lookup and a matmul)."""
        weight = embed.weight.t().to(h.dtype)
        prev, drafts, confs = anchor_tok, [], []
        for j in range(min(k, self.k)):
            left = self.w1(prev)
            prev = (h[:, j] @ weight + left @ self.w2.to(left.dtype)).argmax(-1)
            drafts.append(prev)
            confs.append(torch.sigmoid(self.conf(torch.cat([h[:, j], left], dim=-1))).squeeze(-1))
        self.last_conf = torch.stack(confs, dim=1)
        return torch.stack(drafts, dim=1)

    @torch.no_grad()
    def propose(self, hidden, last, next_tokens, embed, k):
        B, T = next_tokens.shape
        t = torch.full((B, 1), T - 1, device=next_tokens.device)
        h = self.bb.run(self.bb.context(hidden), next_tokens[:, -1:], t, self.k, embed)[:, 0]  # [B, k, H]
        return self._draft(h, next_tokens[:, -1], embed, k)

    @torch.no_grad()
    def propose_cached(self, state, hidden, last, next_tokens, embed, k):
        kv, n = extend_kv(state, self.bb, hidden)
        h = self.bb.run_cached(kv, next_tokens[:, -1], n - 1, self.k, embed)
        return self._draft(h, next_tokens[:, -1], embed, k), (kv, n)
