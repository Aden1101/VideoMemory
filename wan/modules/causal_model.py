# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import os
from wan.modules.attention import attention
from wan.modules.model import rope_freqs_to, rope_phase_dtype
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from wan.modules.video_memory import VideoMemoryLayer, TTTBlockCache
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

from utils.debug_option import DEBUG, MY_VISUAL_DEBUG
from typing import List, Dict, Tuple
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(rope_phase_dtype(freqs[0])).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        # c2 cache-dropout: all sink reads in forward() go through this runtime-overridable
        # attribute (CausalWanModel.set_runtime_sink_size). Defaults to the config sink_size
        # => byte-identical unless a dropout rollout overrides it.
        self._sink_runtime = int(sink_size)
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


    # =========================================================================================
    # MEM_ADDITIVE_KV (arm oracle1add, gate 1 of 2026-08-02), the ADDITIVE recall channel.
    #
    # the idea. Every recall mechanism measured so far (oracle1, memdec20, framebank) splices:
    # the recalled frame takes a slot of the 12-frame local window, so one true frame of the
    # current context is evicted and the model's "immediate declared past" is partly a foreign
    # world (hence bg_cont 0.46 vs 0.90 and cut2 20.0 vs 7.7 on the same-scene tier). Here the
    # anchor is instead injected as 1560 extra k/v tokens, concatenated at attention time
    # (template: the existing prompt_cache path right above), living in the free RoPE band
    # between the sink and the local window. Nothing is evicted, the 12 true frames stay intact.
    #
    # The tokens are installed by the pipeline (see SwitchCausalInferencePipeline._install_add_kv)
    # as `self._add_kv = (k_raw, v)`, `self._add_pos` (RoPE band start, -1 = auto = one frame
    # before the window), `self._add_gain` (attention-mass gain g). `_add_kv` absent/None =>
    # nothing below runs and the legacy path is bitwise.
    #
    # The gain is an exact softmax reweighting, not a heuristic: writing Z for the softmax
    # denominators, the merged output is
    #     out = (Z_ctx * out_ctx + g * Z_anc * out_anc) / (Z_ctx + g * Z_anc)
    # which is the attention over [context ; anchor] with the anchor logits shifted by log g
    # (g = 1 => plain concatenation, exactly). Implemented with the two log-sum-exps returned by
    # the flash kernel, so alpha = g*Z_anc/(Z_ctx + g*Z_anc) is also the anchor's attention MASS
    # -> the tripwire of the design plan comes out for free (self._add_mass).
    @staticmethod
    def _sdpa_lse(q, k, v):
        """(B,H,L,D) half tensors -> (out, logsumexp) from the flash kernel (LSE natural log,
        fp32, shape (B,H,L)). PyTorch >= 2.2 returns the true LSE of the scaled scores."""
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), 0.0, False, False)[:2]
        return out, lse

    def _additive_attention(self, roped_query, k_ctx, v_ctx, grid_sizes, freqs, frame_seqlen,
                            auto_pos, v_ref, n_sink=0):
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
        # MEM_ADD_MASS_SRC (T1, lot offensif 2026-08-04): la masse d'attention par source
        # [sink | fenetre | ancre]. Exact, pas une approximation: le log-sum-exp d'une
        # concatenation est le logaddexp des deux, donc Z_ctx = Z_sink + Z_win et les trois
        # fractions somment a 1. Deux flash-attn de plus, uniquement quand le drapeau est mis
        # (defaut OFF -> aucun cout, sortie inchangee dans tous les cas).
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

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        rope_enhance=False,
        prompt_cache=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # MEM_ADDITIVE_KV phantom pass (arm oracle1add): store the raw (pre-RoPE) k/v this block
        # produces for the anchor frame. Attribute absent by default => zero cost, zero change.
        if getattr(self, "_add_capture", False):
            self._add_captured = (k.detach().clone(), v.detach().clone())

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            # 这里不会更新cache，更新cache的部分在_apply_cache_updates函数
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            gen_frame = q.shape[1] //frame_seqlen
            if not rope_enhance:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            # current_end = current_start + roped_query.shape[1]
            current_end = current_start + q.shape[1]

            sink_tokens = self._sink_runtime * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            # num_new_tokens = roped_query.shape[1]
            num_new_tokens = q.shape[1]
            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print("***********before attention***********")
            #     print(f"kv_cache_size = {kv_cache_size / frame_seqlen}")
            #     print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
            #     print(f"current_end = {current_end / frame_seqlen}")
            #     print(f"current_start = {current_start / frame_seqlen}")
            #     print(f"kv_cache['global_end_index'] = {kv_cache['global_end_index']}")
            #     print(f"kv_cache['local_end_index'] = {kv_cache['local_end_index']}")
            #     print(f"num_new_tokens = {num_new_tokens}")

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            # 这里会出现等于的情况：也就是在四步迭代中，后续的3步都会出现recomput的情况
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"need roll")
                #     print(f"num_rolled_tokens: {num_rolled_tokens / frame_seqlen}")
                #     print(f"num_evicted_tokens: {num_evicted_tokens / frame_seqlen}")
                #     print(f"sink_tokens: {sink_tokens / frame_seqlen}")

                # Compute updated local indices
                # global_end_index: 表示上一次运行的current_end（是一个总体的forward长度 ，一直增长）
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                
                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    if rope_enhance:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    else:
                        temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",  # cache满了以后的操作
                    "sink_tokens": sink_tokens,  # sink token的长度
                    "num_rolled_tokens": num_rolled_tokens,  # 滚动cache的长度，总长-sink长-pop token的长度
                    "num_evicted_tokens": num_evicted_tokens,  # pop的token数量
                    "local_start_index": local_start_index,  # 表示现在kv cache的起始位置
                    "local_end_index": local_end_index,  # 表示现在kv cache的截止位置
                    "write_start_index": write_start_index,  # 重写kv cache的起始位置
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len] if rope_enhance else roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,  # 表示现在q的截止位置
                    "is_recompute": is_recompute
                }

                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"used kv cache size: local_end_index - local_start_index = {local_end_index - local_start_index}")

            # 这个分支会计算recache，is_recompute和cache未满的三个部分
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    if rope_enhance:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    else:
                        temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "sink_tokens": sink_tokens,  # sink token的长度
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len] if rope_enhance else roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }

            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")

            # Use temporary k, v to compute attention
            if sink_tokens > 0:
                # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
                # 最大的能取到除sink以外的token长度
                local_budget = self.max_attention_size - sink_tokens
                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]

                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"local_budget: {local_budget}")
                if local_budget > 0:
                    # local_end_index最大就是cache长度，根据设定的attention size，计算取cache的起点
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]

                    if rope_enhance:  # wan 默认rope 21最大
                        rope_start_f = min(21 - (q.shape[1] // frame_seqlen), current_start_frame)  # roll update, insert模式下的取值
                        rope_start_f_k_local = max(rope_start_f - (local_budget // frame_seqlen) + gen_frame, self._sink_runtime)  # roll update, insert模式下的取值
                        kv_f = (local_end_index - local_start_for_window) // frame_seqlen
                        
                        grid_sizes_k_local = copy.deepcopy(grid_sizes)
                        grid_sizes_sink = copy.deepcopy(grid_sizes)

                        grid_sizes_sink[:, 0] = self._sink_runtime
                        if local_end_index - local_start_for_window > 0:
                            grid_sizes_k_local[:, 0] = kv_f
                            k_local = causal_rope_apply(k_local, grid_sizes_k_local, freqs, start_frame=rope_start_f_k_local).type_as(v)

                        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=rope_start_f).type_as(v)
                        k_sink = causal_rope_apply(k_sink, grid_sizes_sink, freqs, start_frame=0).type_as(v)

                        # if dist.get_rank() == 0:
                        #     print("[rope] current frame {}, q rope idx {}->{}, k local idx {}->{}, k_local shape is {}, q shape is {}, k_sink shape is {}".format(current_start_frame, rope_start_f, rope_start_f+grid_sizes[0,0], rope_start_f_k_local, rope_start_f_k_local + grid_sizes_k_local[0, 0], k_local.shape, q.shape, k_sink.shape))
                        #     # print("[rope] current frame {}, q rope idx {}->{}, k local idx {}->{}, is_recompute={}, update_mode={}".format(current_start_frame, rope_start_f, rope_start_f+grid_sizes[0,0], rope_start_f_k_local, rope_start_f_k_local + grid_sizes_k_local[0, 0], is_recompute, cache_update_info["action"]))

                    if prompt_cache is not None and prompt_cache["switch_frame"] + self.local_attn_size - self._sink_runtime < current_start_frame:
                        k_prompt_cache = prompt_cache["k"].clone().detach()
                        v_prompt_cache = prompt_cache["v"].clone().detach()

                        prompt_cache_f = k_prompt_cache.shape[1] // frame_seqlen
                        grid_sizes_k_prompt = copy.deepcopy(grid_sizes)
                        grid_sizes_k_prompt[:, 0] = prompt_cache_f
                        if rope_enhance:
                            k_prompt_cache = causal_rope_apply(k_prompt_cache, grid_sizes_k_prompt, freqs, start_frame=prompt_cache["rope_start_frame"]).type_as(v)
                        else:
                            k_prompt_cache = causal_rope_apply(k_prompt_cache, grid_sizes_k_prompt, freqs, start_frame=prompt_cache["switch_frame"]).type_as(v)

                        k_cat = torch.cat([k_sink, k_prompt_cache, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_prompt_cache, v_local], dim=1)
                        
                        # if dist.get_rank() == 0:
                        #     print("[prompt cache rope] current frame {}, switch_frame {}, k_prompt_cache idx {}->{},  k_prompt_cache.shape is {}".format(current_start_frame, prompt_cache["switch_frame"].item(), prompt_cache["rope_start_frame"].item(), prompt_cache["rope_start_frame"].item()+grid_sizes_k_prompt[0,0], k_prompt_cache.shape))
                        #     # print("[prompt cache rope] current frame {}, k_prompt_cache idx {}->{} is_recompute={}, update_mode={}, k sum is {} v sum is {}".format(current_start_frame, prompt_cache["rope_start_frame"].item(), prompt_cache["rope_start_frame"].item()+grid_sizes_k_prompt[0,0], is_recompute, cache_update_info["action"], prompt_cache["k"].sum(), prompt_cache["v"].sum()))

                    elif prompt_cache is not None and prompt_cache.get("prompt_idx", 0) > 1:
                        k_prompt_cache = prompt_cache["k_prior"].clone().detach()
                        v_prompt_cache = prompt_cache["v_prior"].clone().detach()

                        prompt_cache_f = k_prompt_cache.shape[1] // frame_seqlen
                        grid_sizes_k_prompt = copy.deepcopy(grid_sizes)
                        grid_sizes_k_prompt[:, 0] = prompt_cache_f
                        if rope_enhance:
                            k_prompt_cache = causal_rope_apply(k_prompt_cache, grid_sizes_k_prompt, freqs, start_frame=prompt_cache["rope_start_frame"]).type_as(v)
                        else:
                            k_prompt_cache = causal_rope_apply(k_prompt_cache, grid_sizes_k_prompt, freqs, start_frame=prompt_cache["switch_frame"]).type_as(v)

                        k_cat = torch.cat([k_sink, k_prompt_cache, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_prompt_cache, v_local], dim=1)

                        # if dist.get_rank() == 0:
                        #     print("[prompt cache rope] current frame {}, switch_frame {}, k_prompt_cache idx {}->{},  k_prompt_cache.shape is {}".format(current_start_frame, prompt_cache["switch_frame"].item(), prompt_cache["rope_start_frame"].item(), prompt_cache["rope_start_frame"].item()+grid_sizes_k_prompt[0,0], k_prompt_cache.shape))
                        #     # print("[prompt cache rope] current frame {}, k_prompt_cache idx {}->{} is_recompute={}, update_mode={}, k sum is {} v sum is {}".format(current_start_frame, prompt_cache["rope_start_frame"].item(), prompt_cache["rope_start_frame"].item()+grid_sizes_k_prompt[0,0], is_recompute, cache_update_info["action"], prompt_cache["k"].sum(), prompt_cache["v"].sum()))

                    else:
                        k_cat = torch.cat([k_sink, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_local], dim=1)
                    
                    
                else:
                    k_cat = k_sink
                    v_cat = v_sink

                if getattr(self, "_add_kv", None) is None:
                    x = attention(
                        roped_query,
                        k_cat,
                        v_cat
                    )
                else:
                    # MEM_ADDITIVE_KV: the anchor tokens live in the free RoPE band just before
                    # the local window (sink = [0, sink_runtime), window = [rope_start_f_k_local,
                    # 21) under rope_enhance), so the default "auto" position is one frame before
                    # the window and nothing in the cache moves.
                    _auto = (max(0, rope_start_f_k_local - 1) if (rope_enhance and local_budget > 0)
                             else max(0, current_start_frame - max(self.local_attn_size, 1) - 1))
                    x = self._additive_attention(roped_query, k_cat, v_cat, grid_sizes, freqs,
                                                 frame_seqlen, _auto, v, n_sink=k_sink.shape[1])
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                k_win = temp_k[:, window_start:local_end_index]
                v_win = temp_v[:, window_start:local_end_index]
                if rope_enhance:
                    # c2 cache-dropout: sink_tokens == 0 never ran with rope_enhance before
                    # (config sink was always > 0; this path crashed on an unbound
                    # roped_query). Mirror the sink>0 windowed rope with a zero-length sink:
                    # temp_k holds UNroped keys under rope_enhance, so rope the window here.
                    rope_start_f = min(21 - (q.shape[1] // frame_seqlen), current_start_frame)
                    rope_start_f_k_win = max(
                        rope_start_f - (self.max_attention_size // frame_seqlen) + gen_frame, 0)
                    if local_end_index - window_start > 0:
                        grid_sizes_k_win = copy.deepcopy(grid_sizes)
                        grid_sizes_k_win[:, 0] = (local_end_index - window_start) // frame_seqlen
                        k_win = causal_rope_apply(
                            k_win, grid_sizes_k_win, freqs,
                            start_frame=rope_start_f_k_win).type_as(v)
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=rope_start_f).type_as(v)
                if getattr(self, "_add_kv", None) is None:
                    x = attention(
                        roped_query,
                        k_win,
                        v_win
                    )
                else:
                    _auto = (max(0, rope_start_f_k_win - 1) if rope_enhance
                             else max(0, current_start_frame - max(self.local_attn_size, 1) - 1))
                    x = self._additive_attention(roped_query, k_win, v_win, grid_sizes, freqs,
                                                 frame_seqlen, _auto, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_memory=False,
                 gate_init=0.1,
                 ckpt_write=False,
                 gate_extra=None,
                 use_decay=False,
                 decay_init=0.95,
                 bounded_gate=False,
                 single_norm=False,
                 bounded_mod=False,
                 rel_norm=False,
                 dc_remove=False,
                 muon_update_steps=5,
                 mem_stream_scale=0.0,
                 ema_alpha=0.0,
                 eviction_feed=False,
                 mem_state_type="swiglu",
                 mem_linear_beta=0.5):
        super().__init__()
        self.dim = dim
        self.mem_stream_scale = mem_stream_scale
        # e4 anti-flicker: causal temporal EMA on the injection (0.0 = OFF, byte-identical baseline).
        self.ema_alpha = float(ema_alpha or 0.0)
        # c2 eviction-feed: memory writes the evicted chunk instead of the current one
        # (False = byte-identical baseline).
        self.eviction_feed = bool(eviction_feed)
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # parallel memory branch (None = unchanged baseline block)
        # mem_state_type/mem_linear_beta (P0, ADDITIVE): "swiglu" (default) = the historical
        # cell, strictly unchanged; "linear_delta" = matrix-state W with the sealed delta
        # rule (plan 2026-08-07 bras M2), read/write core swapped inside the cell only.
        self.memory = VideoMemoryLayer(dim=dim, num_heads=1, inter_multi=4,
                                       g_init=gate_init, checkpoint_write=ckpt_write,
                                       gate_extra=gate_extra, use_decay=use_decay,
                                       decay_init=decay_init, bounded_gate=bounded_gate,
                                       single_norm=single_norm, bounded_mod=bounded_mod, rel_norm=rel_norm, dc_remove=dc_remove,
                                       muon_update_steps=muon_update_steps,
                                       state_type=mem_state_type,
                                       linear_beta=mem_linear_beta) if use_memory else None

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        rope_enhance=False,
        prompt_cache=None,
        ttt_cache=None,
        ttt_update=False,
        ttt_state=None,
        denoising_step=None,
        switch_global_frame=None,
        mem_ema_key=None,
        mem_ema_prev=None,
        mem_x_write=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        x_in = x
        h = self.norm1(x_in)  # cache normalized input; reused by the memory branch
        self_attn_result = self.self_attn(
            (h.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, rope_enhance, prompt_cache)

        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x_in + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # parallel memory branch
        if self.memory is not None and ttt_cache is not None:
            # ttt_state = state snapshot taken outside the gradient-checkpoint region:
            # checkpoint saves these tensors, so the backward recompute reads the same state
            state_in = ttt_state if ttt_state is not None else ttt_cache.state
            # build the optional per-token gate_extra signals: denoising_step from the timestep t,
            # frames_since_switch from the global frame index vs the switch.
            mem_ds, mem_fss = None, None
            gx = getattr(self.memory, "gate_extra", None)
            if gx:
                B = x.shape[0]
                if "denoising_step" in gx and denoising_step is not None:
                    ds = denoising_step.float()
                    if ds.dim() == 1:                       # [B] -> [B, F]
                        ds = ds[:, None].expand(B, num_frames)
                    elif ds.shape[1] != num_frames:
                        ds = ds[:, :1].expand(B, num_frames)
                    mem_ds = ds.repeat_interleave(frame_seqlen, dim=1)          # (B, L)
                if "frames_since_switch" in gx:
                    csf = current_start // frame_seqlen                          # chunk start frame
                    gf = csf + torch.arange(num_frames, device=x.device)         # global frame idx
                    if switch_global_frame is None:
                        fss = gf.float()                                         # no switch -> since start
                    else:
                        fss = torch.where(gf >= switch_global_frame,
                                          gf - switch_global_frame, gf).float()  # resets to 0 at switch
                    mem_fss = fss[None, :].expand(B, num_frames).repeat_interleave(frame_seqlen, dim=1)
            if self.eviction_feed and ttt_update:
                # c2 eviction-feed: the memory writes the chunk that just fell out of the
                # attention window (mem_x_write, popped at model level so it is an explicit
                # checkpoint input like ttt_state), not the current chunk whose k/v the KV
                # cache still holds. The read query stays the current chunk's h. Until
                # something has actually been evicted (fifo warm-up, or the popped chunk is
                # still inside the pinned sink) the write is skipped entirely, so the memory
                # accumulates only cache-complement content. The append below is the only
                # fifo mutation inside the (possibly checkpointed) block; backward recompute
                # re-appends, which the trainer's ttt snapshot/restore undoes (same
                # mechanism that already protects ttt_cache.state and .ema).
                _cur_chunk = int(current_start) // (frame_seqlen * num_frames)
                ttt_cache.fifo.append((_cur_chunk, h.detach()))
                _mem_do_update, _mem_xw = mem_x_write is not None, mem_x_write
            else:
                _mem_do_update, _mem_xw = ttt_update, None
            mem_out, new_state = self.memory.step(
                h, state=state_in, do_update=_mem_do_update, x_write=_mem_xw,  # h = norm1(x_in)
                prompt_emb=context,
                denoising_step=mem_ds, frames_since_switch=mem_fss,
                num_frames=num_frames)
            # on the write pass (ttt_update=True) mem_out only feeds the detached clean cache, not
            # the loss -> detach it, else every frozen downstream block builds a wasted ~15GB backward
            # graph; the write gradient reaches the memory via new_state, not x. On reads keep the graph.
            # MEM_ZERO_ALL ablation (env-gated, default off): zero mem_out on both passes.
            _mz_all = int(os.environ.get("MEM_ZERO_ALL", "0"))
            if _mz_all != 0:                    # any nonzero value zeroes the injection on every frame
                if not globals().get("_MZ_ANNOUNCED"):   # print once/process so mem-OFF arms are auditable
                    print(f"[MEM_ZERO_ALL={_mz_all}] memory injection ZEROED, mem-OFF ablation arm", flush=True)
                    globals()["_MZ_ANNOUNCED"] = True
                mem_out = mem_out * 0.0
            # stream-RMS envelope: bound the injection to a fraction of the local residual stream so an
            # active injection cannot accumulate into an artifact over a long OOD rollout (h=norm1(x) is
            # magnitude-normalized, so x here is the only informative local scale). detach = a bound, not
            # a magnitude the memory can game. Env MEM_STREAM_SCALE or config mem_stream_scale; 0 = off.
            _ss = os.environ.get("MEM_STREAM_SCALE", "")
            _ss = float(_ss) if _ss != "" else float(getattr(self, "mem_stream_scale", 0.0) or 0.0)
            if _ss > 0.0:
                _srms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                self.memory.dbg_inj_stream = (
                    (mem_out.detach().float().pow(2).mean(-1, keepdim=True).sqrt() * _ss).mean()
                    / _srms.mean().clamp_min(1e-8))
                mem_out = mem_out * (_ss * _srms).to(mem_out.dtype)
            # e4 anti-flicker: causal temporal EMA on the read injection (config ema_alpha; 0=OFF,
            # byte-identical). mem_ema_prev is read outside the gradient-checkpoint region and passed
            # in as an explicit input (like ttt_state) so backward recompute reuses the same historical
            # buffer value -> the injection is bit-reproducible under checkpointing. The buffer is
            # updated only on read passes (the injection that appears in the output); the write pass
            # (ttt_update) feeds only the detached clean cache and is left untouched. Under the mem-OFF
            # twin (MEM_ZERO_ALL) the injection must stay exactly zero, so EMA is skipped there.
            _alpha = float(getattr(self, "ema_alpha", 0.0) or 0.0)
            if (_alpha > 0.0 and not ttt_update and _mz_all == 0
                    and mem_ema_key is not None and ttt_cache is not None):
                if mem_ema_prev is not None and mem_ema_prev.shape == mem_out.shape:
                    mem_out = _alpha * mem_out + (1.0 - _alpha) * mem_ema_prev.to(device=mem_out.device, dtype=mem_out.dtype)
                # first observation at this key: mem_out passes through unchanged (EMA seed)
                # e4: CPU-offload the persistent EMA buffer (~165MB/GPU) so the seed-step fits 4 GPUs;
                # bf16 CPU<->GPU round-trip is bit-identical (twin selftest re-validated).
                ttt_cache.ema[mem_ema_key] = mem_out.detach().to("cpu", non_blocking=True)
            # R1 probe (env MEM_REJECT_STREAM, default off = byte-identical): per-token vector rejection
            # of the component of the injection parallel to the current residual stream x. The parallel
            # part is exactly "inject more of what is already there" (self-similar -> damps motion); we
            # keep only the orthogonal (novel/corrective) remainder. Rule: no renorm after subtracting
            # (renorm would re-create the e2 fixed-dose dissolution). Self-similar read -> ~0 remainder
            # -> graceful abstain. Applied to the read injection about to be added.
            if int(os.environ.get("MEM_REJECT_STREAM", "0")) != 0:
                _xh = x.detach().float()
                _xh = _xh / _xh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                _m = mem_out.float()
                mem_out = (_m - (_m * _xh).sum(-1, keepdim=True) * _xh).to(mem_out.dtype)
            x = x + (mem_out.detach() if ttt_update else mem_out)
            if ttt_update:
                # attribute write on a TTTBlockCache object (not a dict key!):this mutation survives the call stack.
                ttt_cache.state = new_state
        # ===================================================================================

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 memory_layers=(5,),
                 gate_init=0.1,
                 ckpt_write=False,
                 gate_extra=None,
                 use_decay=False,
                 decay_init=0.95,
                 bounded_gate=False,
                 single_norm=False,
                 bounded_mod=False,
                 rel_norm=False,
                 dc_remove=False,
                 muon_update_steps=5,
                 mem_stream_scale=0.0,
                 ema_alpha=0.0,
                 eviction_feed=False,
                 mem_state_type="swiglu",
                 mem_linear_beta=0.5):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        # which blocks get a memory branch
        MEMORY_LAYERS = {int(x) for x in memory_layers}
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    use_memory=(i in MEMORY_LAYERS),
                                    gate_init=gate_init, ckpt_write=ckpt_write,
                                    gate_extra=gate_extra, use_decay=use_decay,
                                    decay_init=decay_init, bounded_gate=bounded_gate,
                                    single_norm=single_norm, bounded_mod=bounded_mod, rel_norm=rel_norm, dc_remove=dc_remove,
                                    muon_update_steps=muon_update_steps, mem_stream_scale=mem_stream_scale,
                                    ema_alpha=ema_alpha, eviction_feed=eviction_feed,
                                    mem_state_type=mem_state_type,
                                    mem_linear_beta=mem_linear_beta)
            for i in range(num_layers)
        ])
        # e4 anti-flicker: gate the (host-syncing) EMA-key computation in _forward_inference on
        # whether the injection low-pass is actually active, so the default path stays cost-free.
        self._mem_ema_on = float(ema_alpha or 0.0) > 0.0
        # c2 eviction-feed (False = byte-identical): the model-level side pops the evicted
        # chunk from the fifo outside the checkpointed block. Requires local attention
        # (something must actually evict); enforced lazily in _forward_inference.
        self.eviction_feed = bool(eviction_feed)

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        # init_weights() xavier-inits every nn.Linear, which breaks the memory branch's inert
        # gate init (proj=0).
        for block in self.blocks:
            if getattr(block, "memory", None) is not None:
                block.memory.reset_parameters()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def set_runtime_sink_size(self, sink_size=None):
        """c2 cache-dropout: override the attention sink at runtime on every self-attn
        module. None restores the config sink_size (byte-identical default). Walks
        self.modules() so FSDP-wrapped children are reached too."""
        for m in self.modules():
            if hasattr(m, "_sink_runtime") and hasattr(m, "sink_size"):
                m._sink_runtime = int(m.sink_size if sink_size is None else sink_size)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # # debug
        # DEBUG = False
        # if DEBUG:
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos, prompt_cache, new_prompt_chunk_flag):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                    # if dist.get_rank() == 0:
                    #     print("[update cache]cache grad is {}".format(cache["k"].requires_grad))

                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    sink_tokens = update_info["sink_tokens"]
                        
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v


            ###### prompt cache part #####  
            if prompt_cache is not None and new_prompt_chunk_flag:
                # test_only
                if prompt_cache[block_index].get("prompt_idx", None) is not None:
                    prompt_cache[block_index]["k_prior"].copy_(prompt_cache[block_index]["k"])
                    prompt_cache[block_index]["v_prior"].copy_(prompt_cache[block_index]["v"])
                    # if dist.get_rank() == 0:
                    #     print("[update prompt_cache] is acctivate. action is {}. global_end_index is {}. sum k is {}, sum v is {}. recomupte flag is {}. rope_start_frame={}, prompt_idx={}".format(update_info["action"], kv_cache[block_index]["global_end_index"] // 1560, prompt_cache[block_index]["k"].sum(), prompt_cache[block_index]["v"].sum(), update_info.get("is_recompute", False), prompt_cache[block_index]["rope_start_frame"], prompt_cache[block_index]["prompt_idx"]))

                prompt_cache[block_index]["k"].copy_(new_k)
                prompt_cache[block_index]["v"].copy_(new_v)
                prompt_cache[block_index]["rope_start_frame"].fill_(21 - self.local_attn_size - new_k.shape[1] // 1560)
                
                # if dist.get_rank() == 0:
                #     print("[update prompt_cache] is acctivate. action is {}. global_end_index is {}. sum k is {}, sum v is {}. recomupte flag is {}. rope_start_frame={}".format(update_info["action"], kv_cache[block_index]["global_end_index"] // 1560, prompt_cache[block_index]["k"].sum(), prompt_cache[block_index]["v"].sum(), update_info.get("is_recompute", False), prompt_cache[block_index]["rope_start_frame"]))

            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
        flow_cache=None,
        rope_enhance=False,
        prompt_cache=None,
        new_prompt_chunk_flag=False,
        ttt_cache=None,
        ttt_update=False,
        switch_global_frame=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = rope_freqs_to(self.freqs, device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch,
            rope_enhance=rope_enhance,
        )
        # print("kwargs done")
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        # e4 anti-flicker: key the per-block injection EMA buffers by the denoising timestep so
        # consecutive updates at a matched buffer are consecutive sub-blocks (a clean temporal axis).
        # Read the previous buffer outside the checkpoint and pass it in as an explicit input so the
        # backward recompute is bit-reproducible (mirrors the ttt_state mechanism). Only pay the host
        # sync when the low-pass is actually active.
        _mem_ema_key = None
        if getattr(self, "_mem_ema_on", False) and ttt_cache is not None:
            _mem_ema_key = int(round(float(t.detach().flatten()[0].item())))
        def _ema_prev_for(bi):
            if _mem_ema_key is None or ttt_cache is None or ttt_cache[bi] is None:
                return None
            return ttt_cache[bi].ema.get(_mem_ema_key)
        # ---- c2 eviction-feed (default off = byte-identical) --------------------------------
        # On the write pass, pop the just-evicted chunk from each memory block's fifo here,
        # outside the gradient-checkpointed block, and hand it to the block as an explicit
        # kwarg (mem_x_write): checkpoint saves it as an input, so backward recompute reuses
        # the same tensor (mirrors the ttt_state mechanism). Simple robust FIFO-delay, not
        # token-exact KV eviction: a chunk is "evicted" window_chunks = ceil(local_attn_size /
        # num_frames) chunks after it was written; the first sink_chunks chunks are pinned in
        # the KV sink forever and are therefore never fed to the memory.
        _mem_x_write = {}
        if self.eviction_feed and ttt_update and ttt_cache is not None:
            _F = int(grid_sizes[0][0].item())
            _fsl = int(math.prod(grid_sizes[0][1:]).item())
            _law = self.local_attn_size
            if not isinstance(_law, int):
                _vals = ([int(v) for v in _law if int(v) != -1]
                         if hasattr(_law, "__iter__") else [int(_law)])
                _law = max(_vals) if _vals else -1
            assert _law > 0, "eviction_feed requires local attention (local_attn_size > 0)"
            if _F > 0:
                _wc = max(1, math.ceil(_law / _F))                              # window_chunks
                _sk = int(getattr(self.blocks[0].self_attn, "_sink_runtime", 0))
                _sc = math.ceil(_sk / _F)                                       # sink_chunks
                _cur = int(current_start) // (_fsl * _F)                        # global chunk idx
                _dbg = int(os.environ.get("MEM_EVICT_DEBUG", "0"))
                for _bi, _blk in enumerate(self.blocks):
                    if getattr(_blk, "memory", None) is None or ttt_cache[_bi] is None:
                        continue
                    _fifo = ttt_cache[_bi].fifo
                    _popped = None
                    if len(_fifo) >= _wc:      # pre-append pop == append-then-pop(len > wc)
                        _popped, _hev = _fifo.pop(0)
                        if _popped >= _sc:     # past the sink -> genuinely evicted from KV
                            _mem_x_write[_bi] = _hev
                        # else: sink chunk, pinned in the KV cache forever -> never written
                    if _dbg:
                        if not hasattr(self, "_evict_dbg"):
                            self._evict_dbg = []
                        self._evict_dbg.append(
                            {"block": _bi, "chunk": _cur, "popped": _popped,
                             "wrote": _bi in _mem_x_write,
                             "window_chunks": _wc, "sink_chunks": _sc})
                        print(f"[evict-feed] chunk={_cur} block={_bi} popped={_popped} "
                              f"wrote={_bi in _mem_x_write} window_chunks={_wc} sink_chunks={_sc}",
                              flush=True)
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:  # 计算梯度的step
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        # "crossattn_cache": crossattn_cache[block_index],  # 这里不加，使用文本的cross-attention再算一遍，加上它的梯度
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prompt_cache": prompt_cache[block_index] if prompt_cache is not None else None,
                        "ttt_cache": ttt_cache[block_index] if ttt_cache is not None else None,
                        "ttt_update": ttt_update,
                        # state snapshot: read the fast-weight state outside the checkpointed
                        # function and pass it as explicit tensors.
                        "ttt_state": (
                            (ttt_cache[block_index].state
                             if ttt_cache[block_index].state is not None
                             else block.memory.init_state(x.shape[0], x.device, x.dtype))
                            if (ttt_cache is not None and block.memory is not None) else None),
                        "denoising_step": t,
                        "switch_global_frame": switch_global_frame,
                        "mem_ema_key": _mem_ema_key,
                        "mem_ema_prev": _ema_prev_for(block_index),
                        "mem_x_write": _mem_x_write.get(block_index),
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:  # 不计算梯度的step
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prompt_cache": prompt_cache[block_index] if prompt_cache is not None else None,
                        "ttt_cache": ttt_cache[block_index] if ttt_cache is not None else None,
                        "ttt_update": ttt_update,
                        "ttt_state": None,  # read the live object state
                        "denoising_step": t,
                        "switch_global_frame": switch_global_frame,
                        "mem_ema_key": _mem_ema_key,
                        "mem_ema_prev": _ema_prev_for(block_index),
                        "mem_x_write": _mem_x_write.get(block_index),
                    }
                )
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            # 更新cache
            self._apply_cache_updates(kv_cache, cache_update_infos, prompt_cache, new_prompt_chunk_flag)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = rope_freqs_to(self.freqs, device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
