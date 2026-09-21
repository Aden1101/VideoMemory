"""Attention over the context plus a few extra key/value tokens, without evicting anything.

Mixed into CausalWanSelfAttention. Installed by pipeline/switch/additive_kv.py as `self._add_kv`;
without it, none of this runs and the attention is upstream's.
"""
import copy
import math
import os

import torch

from wan.modules.attention import attention


class AdditiveKVAttentionMixin:
    def _attend(self, roped_query, k, v, grid_sizes, freqs, frame_seqlen, v_ref, *,
                window_frame, n_sink=0):
        """The block's attention over its context, plus the anchor tokens when an arm installed
        some. `window_frame` is the RoPE frame where the local window starts: the anchor goes
        one frame before it by default, in the band the sink and the window leave free, so
        nothing in the cache moves. With no anchor this is upstream's attention call."""
        if getattr(self, "_add_kv", None) is None:
            return attention(roped_query, k, v)
        return self._additive_attention(roped_query, k, v, grid_sizes, freqs, frame_seqlen,
                                        max(0, window_frame - 1), v_ref, n_sink=n_sink)

    # Recall an anchor frame by ADDING key/value tokens rather than evicting a true frame for
    # it: they sit in the free RoPE band between the sink and the window. Installed by the
    # pipeline as `self._add_kv`; absent, nothing below runs.
    #
    # The gain g is an exact softmax reweighting, not an interpolation. With Z the softmax
    # denominators, the merged output is
    #     out = (Z_ctx * out_ctx + g * Z_anc * out_anc) / (Z_ctx + g * Z_anc)
    # which is attention over [context ; anchor] with the anchor logits shifted by log g, so
    # g = 1 is plain concatenation. alpha = g*Z_anc/(Z_ctx + g*Z_anc) is also the attention mass
    # the anchor actually takes, so the check comes out of the same computation.
    @staticmethod
    def _sdpa_lse(q, k, v):
        """(B,H,L,D) half tensors -> (out, logsumexp) from the flash kernel (LSE natural log,
        fp32, shape (B,H,L)). PyTorch >= 2.2 returns the true LSE of the scaled scores."""
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), 0.0, False, False)[:2]
        return out, lse

    def _additive_attention(self, roped_query, k_ctx, v_ctx, grid_sizes, freqs, frame_seqlen,
                            auto_pos, v_ref, n_sink=0):
        """Attention of `roped_query` over the context [k_ctx, v_ctx] plus the installed anchor
        tokens, merged exactly through the two log-sum-exps. The anchor keys are roped at
        self._add_pos, or at `auto_pos`, the frame just before the window, when that is -1.
        Records the attention mass the anchor took, and per source when MEM_ADD_MASS_SRC=1."""
        # imported here and not at the top: causal_model imports this module
        from wan.modules.causal_model import causal_rope_apply
        k_raw, v_add = self._add_kv
        pos = int(getattr(self, "_add_pos", -1))
        pos = int(auto_pos) if pos < 0 else pos
        gain = float(getattr(self, "_add_gain", 1.0))
        g_add = copy.deepcopy(grid_sizes)
        g_add[:, 0] = k_raw.shape[1] // frame_seqlen
        k_add = causal_rope_apply(k_raw, g_add, freqs, start_frame=pos).type_as(v_ref)
        v_add = v_add.type_as(v_ref)
        q = roped_query.transpose(1, 2)
        out_ctx, lse_ctx = self._sdpa_lse(q, k_ctx.transpose(1, 2), v_ctx.transpose(1, 2))
        out_add, lse_add = self._sdpa_lse(q, k_add.transpose(1, 2), v_add.transpose(1, 2))
        alpha = torch.sigmoid(math.log(max(gain, 1e-12)) + (lse_add - lse_ctx).float())  # (B,H,L)
        out = ((1.0 - alpha).unsqueeze(-1) * out_ctx.float()
               + alpha.unsqueeze(-1) * out_add.float())
        self._add_mass = float(alpha.mean())
        self._add_last_pos = pos
        # attention mass per source [sink | window | anchor], exact rather than approximate:
        # the log-sum-exp of a concatenation is the logaddexp of the parts, so the three
        # fractions sum to 1. Costs two more attention calls, only when the flag is set.
        if int(os.environ.get("MEM_ADD_MASS_SRC", "0") or 0) and 0 < n_sink < k_ctx.shape[1]:
            ks, vs = k_ctx[:, :n_sink], v_ctx[:, :n_sink]
            kw, vw = k_ctx[:, n_sink:], v_ctx[:, n_sink:]
            _, lse_s = self._sdpa_lse(q, ks.transpose(1, 2), vs.transpose(1, 2))
            _, lse_w = self._sdpa_lse(q, kw.transpose(1, 2), vw.transpose(1, 2))
            lg = math.log(max(gain, 1e-12))
            tot = torch.logaddexp(lse_ctx.float(), lg + lse_add.float())
            self._add_mass_src = (float(torch.exp(lse_s.float() - tot).mean()),
                                  float(torch.exp(lse_w.float() - tot).mean()),
                                  float(torch.exp(lg + lse_add.float() - tot).mean()))
            self._add_ntok_src = (int(n_sink), int(k_ctx.shape[1] - n_sink), int(k_add.shape[1]))
        return out.to(roped_query.dtype).transpose(1, 2)
