"""EAGLE-3: one decoder layer over the target's fused low/mid/high hidden states plus the next token's embedding, trained on its own
drafts ("training-time test").

Step 0 sees the fused real features. Step j >= 1 sees the drafter's own hidden state from step j-1, with the true token as input
(teacher forcing, which is what the accepted prefix looks like). Attention at step j covers the real prefix keys (step 0, causal)
plus this chain's own keys from steps 1..j at the same position, so training matches how drafting runs at inference. Query and
key positions advance with the step (position t + j), as the draft tokens sit further along the sequence.

The LM head is the target's frozen tied embedding (full 100k vocabulary, no reduced draft vocab). `n_layers` is unused: EAGLE-3 is
one layer by design. Head sizes follow the target's (head_dim 64, kv heads = heads / 3, MLP 8/3 H), which is its Llama config."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drafters.base import HEAD_DIM, DraftOut, Drafter, Features, chunked_ce, register, rope, shifted_targets

@register("eagle3")
class Eagle3(Drafter):
    def __init__(self, cfg, hidden_size: int, n_feature_layers: int):
        super().__init__()
        H = hidden_size
        self.k, self.ce_chunk = cfg.k, cfg.ce_chunk
        self.nh = H // HEAD_DIM
        self.nkv = max(1, self.nh // 3)
        inter = round(8 * H / 3 / 64) * 64
        self.fc = nn.Linear(n_feature_layers * H, H, bias=False)  # fuse low/mid/high target states
        self.norm_e, self.norm_h = nn.RMSNorm(H), nn.RMSNorm(H)
        self.q = nn.Linear(2 * H, self.nh * HEAD_DIM, bias=False)
        self.kv = nn.Linear(2 * H, 2 * self.nkv * HEAD_DIM, bias=False)
        self.o = nn.Linear(self.nh * HEAD_DIM, H, bias=False)
        self.norm_post = nn.RMSNorm(H)
        self.gate_up = nn.Linear(H, 2 * inter, bias=False)
        self.down = nn.Linear(inter, H, bias=False)
        self.norm_out = nn.RMSNorm(H)

    def _steps(self, fused, embed, tokens, k):
        """Yield (j, normed hidden [B, T, H]) for each unrolled step. `tokens(j, out)` gives the [B, T] input ids of step j
        (out = the previous step's hidden, None at step 0)."""
        B, T, _ = fused.shape
        dev = fused.device
        pos = torch.arange(T, device=dev)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        eye = torch.eye(T, dtype=torch.bool, device=dev)
        prev, out, Ks, Vs = fused, None, [], []
        for j in range(k):
            x = torch.cat([self.norm_e(embed(tokens(j, out))), self.norm_h(prev)], dim=-1)
            q = self.q(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
            kk, vv = self.kv(x).view(B, T, 2, self.nkv, HEAD_DIM).permute(2, 0, 3, 1, 4)
            Ks.append(rope(kk, pos + j))
            Vs.append(vv)
            mask = torch.cat([causal] + [eye] * j, dim=1)  # step-0 keys causally, later steps' keys only at the same position
            rep = self.nh // self.nkv
            keys, vals = torch.cat(Ks, 2).repeat_interleave(rep, 1), torch.cat(Vs, 2).repeat_interleave(rep, 1)
            a = F.scaled_dot_product_attention(rope(q, pos + j), keys, vals, attn_mask=mask)
            prev = prev + self.o(a.transpose(1, 2).reshape(B, T, -1))
            g, u = self.gate_up(self.norm_post(prev)).chunk(2, dim=-1)
            prev = prev + self.down(F.silu(g) * u)
            out = self.norm_out(prev)
            yield j, out

    def forward(self, feats: Features) -> DraftOut:
        ids = feats.input_ids
        B, T = ids.shape
        targets, valid = shifted_targets(ids, feats.mask, self.k)

        def tokens(j, _):  # step j reads x[t+1+j] (padded with the last token past the end; those positions are never valid)
            return torch.cat([ids[:, 1 + j :], ids[:, -1:].expand(B, min(1 + j, T))], dim=1)[:, :T]

        weight = feats.embed.weight
        loss, matches = feats.last.new_zeros((), dtype=torch.float32), []
        for j, out in self._steps(self.fc(feats.hidden), feats.embed, tokens, self.k):
            v = valid[:, :, j]
            lj, mj = chunked_ce(out[v], weight, targets[:, :, j][v], self.ce_chunk)
            loss = loss + lj
            full = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
            full[v] = mj
            matches.append(full)
        any_valid = valid.any(-1)
        v = valid[any_valid]
        return DraftOut(loss / v.sum().clamp(min=1), torch.stack(matches, dim=-1)[any_valid], v)

    @torch.no_grad()
    def propose(self, hidden, last, next_tokens, embed, k):
        ids, drafts = next_tokens, []

        def tokens(j, out):
            nonlocal ids
            if j > 0:  # feed back the previous draft at the final position (others only feed their own, unused, chains)
                ids = ids.clone()
                ids[:, -1] = drafts[-1]
            return ids

        for j, out in self._steps(self.fc(hidden), embed, tokens, min(k, self.k)):
            drafts.append((out[:, -1] @ embed.weight.t().to(out.dtype)).argmax(-1))
        return torch.stack(drafts, dim=1)

    def _qkv(self, x, pos):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.nh, HEAD_DIM).transpose(1, 2)
        k, v = self.kv(x).view(B, N, 2, self.nkv, HEAD_DIM).permute(2, 0, 3, 1, 4)
        return rope(q, pos), rope(k, pos), v

    @torch.no_grad()
    def propose_cached(self, state, hidden, last, next_tokens, embed, k):
        """Step-0 keys/values of every position are cached, so a cycle costs k one-token steps instead of k passes over the prefix.
        Step j >= 1 attends to all step-0 keys plus its own chain's keys from steps 1..j, as in training."""
        B, m, _ = hidden.shape
        fused = self.fc(hidden)
        n0 = 0 if state is None else state["n"]
        pos = torch.arange(m, device=hidden.device) + n0
        q0, k0, v0 = self._qkv(torch.cat([self.norm_e(embed(next_tokens)), self.norm_h(fused)], dim=-1), pos)
        K0 = k0 if state is None else torch.cat([state["K"], k0], 2)
        V0 = v0 if state is None else torch.cat([state["V"], v0], 2)
        n, t = n0 + m, n0 + m - 1
        rep = self.nh // self.nkv
        prev, q, drafts, ck, cv = fused[:, -1:], q0[:, :, -1:], [], [], []
        for j in range(min(k, self.k)):
            if j > 0:
                x = torch.cat([self.norm_e(embed(drafts[-1][:, None])), self.norm_h(prev)], dim=-1)
                q, kj, vj = self._qkv(x, torch.full((1,), t + j, device=hidden.device))
                ck.append(kj)
                cv.append(vj)
            keys = torch.cat([K0] + ck, 2).repeat_interleave(rep, 1)
            vals = torch.cat([V0] + cv, 2).repeat_interleave(rep, 1)
            a = F.scaled_dot_product_attention(q, keys, vals)
            prev = prev + self.o(a.transpose(1, 2).reshape(B, 1, -1))
            g, u = self.gate_up(self.norm_post(prev)).chunk(2, dim=-1)
            prev = prev + self.down(F.silu(g) * u)
            drafts.append((self.norm_out(prev)[:, -1] @ embed.weight.t().to(prev.dtype)).argmax(-1))
        return torch.stack(drafts, dim=1), {"K": K0, "V": V0, "n": n}
