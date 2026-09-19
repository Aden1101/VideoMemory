#!/usr/bin/env python
"""Memory 2.0, OFFLINE WRITER trainer by WRITE-replay (J3 deliverable D).

Premise (memory2_design.md 1.1, selftest_replay.py S1 green, selftest_s1b.py F5.2 green):
under MEM_ZERO_ALL=1 the eviction-feed hidden states h are data (independent of the writer
parameters), so the whole write chain can be re-run OFFLINE, differentiably, from collected
h sequences, no video model in the loop. This trainer does exactly that:

  state_0 = cell.init_state  ->  for i in writes:  state = write_theta(state, h_i)
  at sampled positions:      frame_hat = AnchorFrameDecoder(states, cond)
  loss = subject-weighted norm_mse(frame_hat, target)  +  lambda * InfoNCE(frame_hat, target)

The die-ON patches of reference/memory2_redteam.md are implemented here, each tagged:
  F2.3  warm-start from l1@2000-EMA + assert writer/delta_l1 == 0 at step 0 (vs the ckpt).
  F3.3  AnchorFrameDecoder(detach_states=False) + assert qkv.weight.grad != 0 after step 1.
  F2.1  phase A unfreezes the decoder's addressing only (queries / pos_row / pos_col /
        q_norms); content (fuse/out_norm/head/read_norms) and the cond FiLM stay frozen, so
        a rotation of the key space cannot masquerade as "the state has nothing new".
  F1.6  targets are aged: age sampled in {1,2,5,10,20} writes, metrics split by age and by
        mode (single/switch/anchor); single videos are down-weighted (--single_w).
  F1.1  InfoNCE negatives = the whole split-side bank (which contains the same-prompt twin
        and the k=8 cond-nearest videos, both also logged separately) + cond-dropout 0.5;
        causal margin acc(true state) - acc(shuffled state) logged from step 0.
  F1.3  same-video negatives are masked only within +-mask_chunks of the target (the other
        segment's frames stay in the softmax: the anti-contamination pressure).
  F3.2  mem/influence_decay@k (replay with h_{T-k} neutralized) + loss_by_age.
  F4.1  per-video split (>= 20 val videos, twins on their sibling's side) + proximal
        ||theta_w - theta_l1||^2 (--prox) + train/val gap logged.
  F5.4  peak VRAM of the full backward measured and logged before anything is scaled up
        (--vram_probe sweeps K / n_bptt and prints the table).
  F3.1  --toy = the redefined gate: >= 8 videos, targets >= 10 writes old, pass only if the
        loss collapses and n_bptt_writes=1 does clearly worse and h-shuffle degrades.
no VAE in the loss (J3 mandate): the phase-A instance term lives in latent crop60 space
(mem2_data.latent_embed); the VAE is only used for the viz grid, exactly like train_decoder.py.

---------------------------------- phase B (2026-07-28) -------------------------------------
J3 left two defects to convert (memory2_j3_report verdicts + Aden's eye):
  (a) rendering: the per-token decoder emits a low-pass frame with a visible 2x2 checkerboard
     , nothing in v1 ever mixes neighbouring patches.  Fix = AnchorFrameDecoder(mixer=True)
      (2 residual depthwise-3x3 + pointwise blocks on the 30x52 grid, zero-init => identity at
      init => a v1.2/A2 checkpoint warm-starts bitwise, the only missing keys are `mixer.*`).
  (b) instance: with no instance pressure the stored content is the right one ~1 time in 2.
      Fix = --lam_dino: differentiable VAE decode of one frame -> crop60 -> DINOv2 ViT-B/14
      (the harness re-ID model, imported from run_reid_probe.get_dino) -> InfoNCE against the
      PRECOMPUTED bank of true-anchor DINO embeddings (build_bank_dino.py), same-prompt twin
      and cond-kNN forced in the softmax.  The J3 latent InfoNCE started 100% on the wrong
      side (pos_gap -0.11..-0.36); --dino_probe measures the same quantity in DINO space and
      is a hard gate: no B2 run if pos_gap <= 0 at step 0.
  The phase-B writer is frozen at A2 (--writer_ckpt ... --freeze_writer): only the decoder
  moves, so `writer/delta_ref` must stay exactly 0.0 from step 0 to the final checkpoint
  (asserted at every val and at the end of the run).

------------------------- phase T0 / MEMOIRE 2.1 (2026-08-04) --------------------------------
La nuit du 04/08 a ferme le canal (T0 : le splice est lu, exces apparie +0.83, x20 la
prediction), l'instant (levier 4 : 0/2) et l'ADDITIF (levier 3 : 2/4). Ce qui reste debout est
l'ECRITURE : l'ancre decodee est une image de l'ambiance du rollout (vs A* +0.042, vs B +0.298,
vs X +0.344 ; B > A sur 21/24, p = 3e-4), pas du sujet, parce que le DECODEUR a ete entraine
a reconstruire la frame EVINCEE, et qu'au retour ce qui vient d'etre evince est l'interlude.
2.1 change l'OBJECTIF, pas le mecanisme : --lam_t0 ajoute un terme contrastif dans la SEULE
metrique du dossier qui (a) voit ce defaut et (b) tolere le flou du rendu (la reco reste, avec
--rec_w reduit, comme regulariseur) :
    ancre decodee a la position de retour simulee (etat de fin de sequence + positions
    post-switch echantillonnees)  --T0-->  HAUTE correlation bande-basse avec le segment A,
    BASSE avec l'interlude B, les autres videos et le JUMEAU (meme prompt, autre graine).
Metrique/perte : t0_metric.py (import unique, identique bitwise a prescreen_offensif.lowband).
References : build_bank_t0.py (pre-calculees hors ligne dans les deux espaces, le cote
reference n'a jamais besoin de gradient, seule l'ancre paie un decode VAE si --t0_space pixel).
Gates scelles AVANT tout run dans memory2_probes_registry.md (sections D0 / G1 / G2a / G2b / G3).
`--lam_t0 0` est BYTE-safe : le terme n'est meme pas construit (selftest_t0 test 9).

Usage (GPU pre-check in the same command, house rule):
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
  CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/train_writer.py --toy       # gate F3.1
  ... --smoke                       # 5 steps, batch 1: shapes/asserts/VRAM
  ... --vram_probe                  # F5.4 sweep, no training
  ... --phase A --steps 2500        # the real phase A
  ... --phase B --writer_ckpt A2 --freeze_writer --mixer --dino_probe   # phase-B gate
wandb is mandatory (house rule): never WANDB_MODE=disabled.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
import t0_metric as T0M  # noqa: E402
from build_bank import BANK_PT  # noqa: E402
from build_bank_dino import (BANK_DINO, latent_to_pixels,  # noqa: E402
                             pixels_to_dino_input)
from build_bank_t0 import BANK_T0  # noqa: E402
from t0_bank import T0Bank, force_end_pos  # noqa: E402
from wan.modules.video_memory import (AnchorFrameDecoder, VideoMemoryLayer,  # noqa: E402
                                      replay_writes)
from train_decoder import norm_mse, make_scheduler  # noqa: E402  (v1 idioms, reused verbatim)

DEFAULT_CONFIG = os.path.join(PROBE_DIR, "probe_l1.yaml")
DEC_V12 = os.path.join(FRAME_DEC, "frame_dec_v12/ckpt/dec_step4000.pt")
OUT_ROOT = os.path.join(D.RUNS_DIR, "writer")
WRITER_A2 = os.path.join(OUT_ROOT, "mem2_writer_A2_final.pt")

# theta_w (design 3.2): every parameter the WRITE path consumes. W0/w1/w2 (the state prior)
# stay frozen in phase A/B; the read path (q_norm/proj/o_norm) is out of scope (injection dead).
WRITE_PARAMS = ("qkv.weight", "qkv.bias", "lrs.weight", "lrs.bias",
                "k_norm.weight", "decay_logit")
# F2.1: the decoder's addressing (what it asks the state), never its content or its FiLM.
ADDRESS_PREFIXES = ("queries", "pos_row", "pos_col", "q_norms.")


# ----------------------------- loading ---------------------------------------------------------
def _clean_ema_key(name):
    for pfx in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
        name = name.replace(pfx, "")
    return name


def ref_theta_l1(config, blocks, device, dtype=torch.bfloat16):
    """theta_l1 read straight from the CHECKPOINT (F2.3): the EMA shadow when use_ema is on
    (that is what the harness deploys and what the v3 collection ran), else generator_memory.
    Returned in the deployment dtype, the exact bytes the cells must hold at step 0."""
    ck = torch.load(config.lora_ckpt, map_location="cpu", weights_only=False, mmap=True)
    use_ema = bool(getattr(config, "use_ema", False))
    src = "generator_ema" if use_ema else "generator_memory"
    assert src in ck, f"{config.lora_ckpt}: no {src}"
    sd = {_clean_ema_key(k): v for k, v in ck[src].items()}
    ref = {}
    for bi in blocks:
        pfx = f"model.base_model.model.blocks.{bi}.memory."
        got = {}
        for name in WRITE_PARAMS:
            k = pfx + name
            assert k in sd, f"{src} has no {k}, wrong ckpt layout?"
            got[name] = sd[k].detach().to(device=device, dtype=dtype).clone()
        ref[bi] = got
    del ck, sd
    return ref, src


def build_cells(config, device, mode="pipeline"):
    """The l1@2000-EMA memory cells, in the deployment dtype (bf16) on `device`.
    mode='pipeline' = the canonical harness path (run_reid_probe.build_pipeline: base -> LoRA ->
    generator_memory -> EMA overlay -> .to(bf16)), then the cells are deep-copied out and the
    1.3B generator + T5 are released. Mode='direct' rebuilds the cell from the config kwargs and
    loads the same EMA tensors (cheap, used by the CPU selftests); both are asserted identical
    to theta_l1 at step 0, so 'direct' can never silently diverge."""
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)
    vae = None
    if mode == "pipeline":
        from run_reid_probe import build_pipeline, release_vram
        from collect_pairs import _find_causal_model
        pipeline = build_pipeline(config, device)
        cm = _find_causal_model(pipeline.generator.model)
        assert cm is not None, "CausalWanModel not found under the peft wrappers"
        cells = {bi: copy.deepcopy(cm.blocks[bi].memory) for bi in blocks}
        vae = pipeline.vae
        del cm, pipeline
        release_vram()
    else:
        mk = config.model_kwargs
        ck = torch.load(config.lora_ckpt, map_location="cpu", weights_only=False, mmap=True)
        sd_all = {_clean_ema_key(k): v for k, v in
                  ck["generator_ema" if bool(getattr(config, "use_ema", False))
                     else "generator_memory"].items()}
        cells = {}
        for bi in blocks:
            cell = VideoMemoryLayer(
                dim=1536, num_heads=1, inter_multi=4, g_init=float(mk.gate_init),
                checkpoint_write=bool(mk.ckpt_write), gate_extra=list(mk.gate_extra or []),
                use_decay=bool(mk.use_decay), decay_init=float(getattr(mk, "decay_init", 0.95)),
                bounded_gate=bool(mk.bounded_gate), single_norm=bool(mk.single_norm),
                bounded_mod=bool(mk.bounded_mod), rel_norm=bool(mk.rel_norm),
                dc_remove=bool(mk.dc_remove), muon_update_steps=int(mk.muon_update_steps))
            pfx = f"model.base_model.model.blocks.{bi}.memory."
            sd = {k[len(pfx):]: v for k, v in sd_all.items() if k.startswith(pfx)}
            missing, unexpected = cell.load_state_dict(sd, strict=False)
            assert not unexpected, f"block {bi}: unexpected memory keys {unexpected[:4]}"
            cells[bi] = cell.to(device=device, dtype=torch.bfloat16)
        del ck, sd_all
    for bi, cell in cells.items():
        cell.detach_state = False        # replay owns the graph (design 1.3)
        cell.train()
        assert cell.qkv.weight.dtype == torch.bfloat16, f"block {bi} not in bf16"
    return cells, blocks, vae


def delta_l1(cells, ref):
    """||theta - theta_l1||_1 / ||theta_l1||_1 over the write parameters (F2.3 tripwire)."""
    num = den = 0.0
    for bi, got in ref.items():
        for name, r in got.items():
            p = cells[bi].get_parameter(name)
            num += float((p.detach().float() - r.float()).abs().sum())
            den += float(r.float().abs().sum())
    return num / max(den, 1e-12)


def get_write_params(cell):
    return {name: cell.get_parameter(name) for name in WRITE_PARAMS}


# ----------------------------- losses -----------------------------------------------------------
def norm_mse_per(pred, target, subject_weight=1.0, eps=1e-8):
    """Per-sample norm_mse (train_decoder.norm_mse without the final .mean()), so the loss can be
    split by age / mode. Asserted equal to norm_mse(...).mean() by selftest_trainer S-E1."""
    pf, tf = pred.float(), target.float()
    if subject_weight != 1.0:
        H, W_ = pf.shape[-2], pf.shape[-1]
        ch, cw = int(H * 0.6), int(W_ * 0.6)
        top, left = (H - ch) // 2, (W_ - cw) // 2
        w = torch.ones_like(tf)
        w[..., top:top + ch, left:left + cw] = subject_weight
        w = w / w.mean(dim=tuple(range(1, w.dim())), keepdim=True)
        se = (pf - tf).pow(2) * w
    else:
        se = (pf - tf).pow(2)
    se = se.flatten(1).mean(dim=1)
    denom = tf.flatten(1).pow(2).mean(dim=1).clamp_min(eps)
    return se / denom


def infonce(z, pos, neg, tau, neg_mask=None):
    """z (K,D) anchors (decoded frames), pos (K,D) their true targets, neg (M,D) the bank.
    neg_mask (K,M) True = masked OUT (same video, within +-mask_chunks of the target: F1.3).
    Returns (loss, top1 accuracy)."""
    l_pos = (z * pos).sum(-1, keepdim=True) / tau                  # (K,1)
    l_neg = (z @ neg.t()) / tau                                    # (K,M)
    if neg_mask is not None:
        l_neg = l_neg.masked_fill(neg_mask, float("-inf"))
    logits = torch.cat([l_pos, l_neg], dim=1)
    tgt = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    loss = Fn.cross_entropy(logits, tgt)
    acc = (logits.argmax(dim=1) == 0).float().mean()
    return loss, acc


# ----------------------------- bank ---------------------------------------------------------------
class Bank:
    """The precomputed InfoNCE negatives (build_bank.py) resident on the GPU."""

    def __init__(self, path, device, split_side="train"):
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.entries = b["entries"]
        self.emb_all = b["emb"].to(device=device, dtype=torch.float32)
        self.videos = b["videos"]
        self.id2i = {v["video_id"]: i for i, v in enumerate(self.videos)}
        self.split = b["split"]
        self.twin_index = b["twin_index"]
        self.cond_knn = b["cond_knn"].to(device)
        self.cond = b["cond"].to(device=device, dtype=torch.float32)
        self.device = device
        self.e_video = torch.tensor([e["video_i"] for e in self.entries], device=device)
        self.e_chunk = torch.tensor([e["chunk"] for e in self.entries], device=device)
        self.e_split = [e["split"] for e in self.entries]
        self.sides = {}
        for side in ("train", "val"):
            idx = torch.tensor([i for i, s in enumerate(self.e_split) if s == side],
                               device=device)
            self.sides[side] = idx
        self.dim = self.emb_all.shape[1]
        print(f"[writer] bank: {len(self.entries)} entries "
              f"({len(self.sides['train'])} train / {len(self.sides['val'])} val), "
              f"dim {self.dim}, {self.emb_all.numel() * 4 / 1e6:.0f} MB on {device}", flush=True)

    def negatives(self, side, meta, target_chunks, n_neg=0, mask_chunks=2, rng=None):
        """(neg_emb (M,D), mask (K,M), diag index sets). Full split-side bank by default, it
        contains the same-prompt twin and the cond-kNN videos (both reported separately). With
        n_neg > 0 a random subset is drawn and twin + kNN entries are force-added (mandate)."""
        pool = self.sides[side]
        if n_neg and n_neg < pool.numel():
            sel = pool[torch.randperm(pool.numel(), device=pool.device)[:n_neg]]
            side_mask = torch.zeros(self.emb_all.shape[0], dtype=torch.bool,
                                    device=self.device)
            side_mask[self.sides[side]] = True          # forced entries stay on the same side
            forced = [self.entries_of_video(v)[side_mask[self.entries_of_video(v)]]
                      for v in self.twin_and_knn(meta)]
            forced = [f for f in forced if f.numel()]
            if forced:
                sel = torch.unique(torch.cat([sel] + forced))
            pool = sel
        neg = self.emb_all[pool]
        vid_i = self.id2i[meta["video_id"]]
        same = (self.e_video[pool] == vid_i)
        tc = torch.tensor(target_chunks, device=self.device).view(-1, 1)
        near = (self.e_chunk[pool].view(1, -1) - tc).abs() <= mask_chunks
        mask = same.view(1, -1) & near                      # F1.3: same video and near = masked
        return neg, mask, pool

    def entries_of_video(self, video_i):
        return (self.e_video == video_i).nonzero(as_tuple=True)[0]

    def twin_and_knn(self, meta):
        vid_i = self.id2i[meta["video_id"]]
        out = list(self.cond_knn[vid_i].tolist())
        tw = D.twin_of(meta["video_id"], self.twin_index,
                       {v["video_id"]: v for v in self.videos})
        if tw is not None:
            out.append(self.id2i[tw])
        return out

    def twin_entries(self, meta):
        tw = D.twin_of(meta["video_id"], self.twin_index,
                       {v["video_id"]: v for v in self.videos})
        if tw is None:
            return None
        return self.entries_of_video(self.id2i[tw])


# ----------------------------- trainer -----------------------------------------------------------
class WriterTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.rng = random.Random(args.seed)
        torch.manual_seed(args.seed)

        os.chdir(REPO)                       # WanVAEWrapper reads repo-relative paths
        self.config = OmegaConf.load(args.config)
        self.config.distributed = False
        self.cells, self.blocks, self.vae = build_cells(self.config, self.device, args.load)
        self.ref, ref_src = ref_theta_l1(self.config, self.blocks, self.device)

        # ---- F2.3: warm start is l1@2000-EMA, bit for bit -----------------------------------
        d0 = delta_l1(self.cells, self.ref)
        print(f"[writer] warm start from {ref_src} of {self.config.lora_ckpt}: "
              f"writer/delta_l1 = {d0:.3e}", flush=True)
        assert d0 == 0.0, (f"F2.3 VIOLATION: theta_w != theta_l1 at step 0 (delta_l1={d0:.3e}). "
                           f"The step-0 baseline would not be the v1.2 regime.")

        # trainable write params (bf16 leaves) + fp32 masters (design 3.5: master fp32, forward
        # in the deployment dtype -> the trained function is the deployed function)
        self.wparams, self.masters = {}, {}
        for bi, cell in self.cells.items():
            for p in cell.parameters():
                p.requires_grad_(False)
            for name, p in get_write_params(cell).items():
                p.requires_grad_(True)
                self.wparams[(bi, name)] = p
                m = p.detach().float().clone().requires_grad_(True)
                self.masters[(bi, name)] = m
        n_w = sum(p.numel() for p in self.wparams.values())
        print(f"[writer] theta_w: {len(self.wparams)} tensors, {n_w / 1e6:.2f}M params "
              f"over blocks {self.blocks}", flush=True)

        # ---- phase B: load a trained writer (A2) and freeze it -------------------------------
        # theta_ref (the tripwire reference) becomes whatever the run starts from: l1 in phase A,
        # A2 in phase B. `writer/delta_ref` must stay exactly 0.0 for the whole frozen run.
        self.ref_start = {bi: {k: v.clone() for k, v in g.items()} for bi, g in self.ref.items()}
        self.writer_src = f"{ref_src}@{os.path.basename(str(self.config.lora_ckpt))}"
        if args.writer_ckpt:
            wck = torch.load(args.writer_ckpt, map_location="cpu", weights_only=False)
            assert "writer" in wck, f"{args.writer_ckpt}: no 'writer' entry (MEM_WRITER format)"
            n_loaded = 0
            with torch.no_grad():
                for bi in self.blocks:
                    sd = wck["writer"][str(bi)]
                    for name in WRITE_PARAMS:
                        assert name in sd, f"writer ckpt block {bi}: no {name}"
                        p = self.wparams[(bi, name)]
                        t = sd[name].to(device=p.device, dtype=p.dtype)
                        assert t.shape == p.shape, f"{bi}.{name}: {tuple(t.shape)} vs {tuple(p.shape)}"
                        p.copy_(t)
                        self.masters[(bi, name)].copy_(p.detach().float())
                        self.ref_start[bi][name] = p.detach().clone()
                        n_loaded += 1
            self.writer_src = f"{args.writer_ckpt} (step {wck.get('step', '?')})"
            print(f"[writer] WRITER warm-started from {self.writer_src}: {n_loaded} tensors, "
                  f"drift vs l1 = {delta_l1(self.cells, self.ref):.4e}", flush=True)
            del wck
        self.freeze_writer = bool(args.freeze_writer)
        if self.freeze_writer:
            for p in self.wparams.values():
                p.requires_grad_(False)
            for m in self.masters.values():
                m.requires_grad_(False)
            d_ref = delta_l1(self.cells, self.ref_start)
            assert d_ref == 0.0, f"frozen writer already off its reference: {d_ref:.3e}"
            print(f"[writer] theta_w FROZEN (phase-B contract): delta_ref = {d_ref:.3e}, "
                  f"prox disabled, n_bptt -> 0 (the FULL chain is still replayed, only the "
                  f"graph is cut: replay_writes' forward is unchanged, so the states are "
                  f"bitwise those of a full-BPTT replay, selftest_phaseB test 5)", flush=True)
            args.prox = 0.0
            args.n_bptt = 0

        # ---- decoder (F3.3 detach_states=False, F2.1 addressing-only in phase A) ------------
        self.decoder = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512,
                                          detach_states=False, mixer=bool(args.mixer),
                                          mixer_blocks=args.mixer_blocks,
                                          mixer_hidden=args.mixer_hidden).to(self.device)
        assert self.decoder.detach_states is False, \
            "F3.3: detach_states=True would silently cut the loss off from the writer"
        if args.dec_ckpt:
            ck = torch.load(args.dec_ckpt, map_location="cpu", weights_only=False)
            sd = ck["decoder"] if isinstance(ck, dict) and "decoder" in ck else ck
            missing, unexpected = self.decoder.load_state_dict(sd, strict=False)
            assert not unexpected, f"decoder warm-start unexpected keys: {list(unexpected)[:4]}"
            # strict=False is tolerated for the v2 mixer keys only (they init to the identity):
            # any other missing key would mean a silently re-initialized decoder.
            bad = [k for k in missing if not k.startswith("mixer.")]
            assert not bad, f"decoder warm-start missing NON-mixer keys: {bad[:4]}"
            if missing:
                print(f"[writer] decoder v2: {len(missing)} mixer keys at their IDENTITY init "
                      f"(the rest is bitwise the warm start)", flush=True)
            print(f"[writer] decoder warm-started from {args.dec_ckpt} "
                  f"(step {ck.get('step', '?') if isinstance(ck, dict) else '?'})", flush=True)
        dec_train = []
        for name, p in self.decoder.named_parameters():
            train = (args.phase == "B") or name.startswith(ADDRESS_PREFIXES)
            p.requires_grad_(bool(train))
            if train:
                dec_train.append(p)
        print(f"[writer] decoder phase {args.phase}: "
              f"{sum(p.numel() for p in dec_train) / 1e6:.2f}M trainable / "
              f"{self.decoder.num_params() / 1e6:.1f}M "
              f"({'ALL' if args.phase == 'B' else 'ADDRESSING ONLY (F2.1)'}"
              f"{' + MIXER' if args.mixer else ''})", flush=True)

        groups = [{"params": dec_train, "lr": args.lr_d, "weight_decay": args.wd_d}]
        if not self.freeze_writer:      # a frozen writer is not even in the optimizer state
            groups.insert(0, {"params": list(self.masters.values()), "lr": args.lr_w,
                              "weight_decay": 0.0})
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        self.sched = make_scheduler(self.opt, args.steps, args.warmup, args.min_lr_ratio)

        # ---- data ---------------------------------------------------------------------------
        self.vids = D.load_index(verbose=False)
        self.by_id = {v["video_id"]: v for v in self.vids}
        self.split = D.assign_split(self.vids, args.n_val_switch, args.n_val_single)
        self.train_vids = [v for v in self.vids if self.split[v["video_id"]] == "train"]
        self.val_vids = [v for v in self.vids if self.split[v["video_id"]] == "val"]
        print(f"[writer] data: {len(self.train_vids)} train / {len(self.val_vids)} val videos "
              f"(per-VIDEO split, F4.1)", flush=True)
        self.cache = D.PartCache(size=args.part_cache)
        self.hcache = {}                     # video_id -> {block: [cpu fp16 h]} (toy/smoke)
        self.bank = Bank(args.bank, self.device)

        # ---- phase B: the DINO instance space (bank precomputed by build_bank_dino.py) ------
        self.dino_emb, self.dino_key2row, self.dino = None, {}, None
        self.last_dino_gap = None            # gate: lambda_dino is 0 while pos_gap <= 0
        if args.lam_dino > 0 or args.dino_probe or args.dino_log:
            db = torch.load(args.dino_bank, map_location="cpu", weights_only=False)
            assert db["emb"].shape[0] == len(self.bank.entries), \
                (f"{args.dino_bank}: {db['emb'].shape[0]} rows vs {len(self.bank.entries)} bank "
                 f"entries, rebuild build_bank_dino.py against THIS bank")
            self.dino_emb = db["emb"].to(device=self.device, dtype=torch.float32)
            for r, e in enumerate(self.bank.entries):
                self.dino_key2row[(e["video_id"], int(e["frame"]))] = r
            if self.vae is None:             # --load direct: the pipeline never built one
                from train_decoder import build_vae
                self.vae = build_vae(self.device)
            from run_reid_probe import get_dino
            self.dino = get_dino(self.device)          # the harness model, frozen, fp32
            print(f"[writer] DINO bank: {tuple(self.dino_emb.shape)} "
                  f"({db['meta']['model']}, crop {db['meta']['crop_frac']}), "
                  f"{len(self.dino_key2row)} (video, frame) keys", flush=True)

        # ---- MEMOIRE 2.1: the T0 low-band contrastive term ---------------------------------
        # `--lam_t0 0` and no `--t0_log` => the bank is not even loaded and the term is never
        # built: the run is BYTE-identical to the phase-A/A2 recipe (selftest_t0 test 9).
        self.t0bank = None
        if args.lam_t0 > 0 or args.t0_log or args.t0_probe:
            self.t0bank = T0Bank(args.t0_bank, self.device, space=args.t0_space,
                                 lat_mode=args.t0_lat_mode)
            assert self.t0bank.n == args.t0_n, \
                (f"bank T0 construite a n={self.t0bank.n}, run demande n={args.t0_n} : "
                 f"la bande DOIT etre la meme des deux cotes de la correlation")
            if args.t0_space == "pixel" and self.vae is None:
                from train_decoder import build_vae
                self.vae = build_vae(self.device)

        self.prev_states = None              # for the causal-margin control (F1.1.4)
        self._fixed_pos = {}                 # --fixed_positions (toy/overfit gate)
        self.shuffle_pool = None             # videos the h-shuffle control draws from
        self.peak_vram = 0.0
        self.step = 0

    # ---------- h loading -------------------------------------------------------------------
    def get_h(self, meta, upto):
        """h_seq[block][0..upto] on the GPU in the deployment dtype (bf16). --cache_videos N
        keeps the first N videos' full fp16 h in CPU RAM (1.5 GB/video): the toy/smoke runs
        then never touch the shards again."""
        vid = meta["video_id"]
        if self.args.cache_videos:
            if vid not in self.hcache and len(self.hcache) < self.args.cache_videos:
                full = D.load_h(meta, self.cache, self.blocks, upto=None, device=None,
                                dtype=torch.float16)
                self.hcache[vid] = {b: [t.clone() for t in full[b]] for b in self.blocks}
            src = self.hcache.get(vid)
            if src is not None:
                return {b: [t.to(self.device, torch.bfloat16) for t in src[b][:upto + 1]]
                        for b in self.blocks}
        return D.load_h(meta, self.cache, self.blocks, upto=upto, device=self.device)

    # ---------- one video forward ------------------------------------------------------------
    def forward_video(self, meta, positions, h_override=None, grad=True, n_bptt=None,
                      force_cond=False):
        """Replay the write chain and decode at the sampled positions.
        Returns (pred (K,1,16,60,104), target (K,1,...), states_detached, cond_used)."""
        i_max = max(p["i"] for p in positions)
        h = h_override if h_override is not None else self.get_h(meta, i_max)
        states = {}
        for bi in self.blocks:
            _, all_st = replay_writes(
                self.cells[bi], h[bi][:i_max + 1], detach_state=False,
                n_bptt_writes=(self.args.n_bptt if n_bptt is None else n_bptt),
                use_checkpoint=True, return_all=True)
            states[bi] = all_st
        idxs = [p["i"] for p in positions]
        dec_states = [tuple(torch.cat([states[bi][i][k] for i in idxs], 0) for k in range(3))
                      for bi in self.blocks]
        ca, cb = D.load_conds(meta, self.cache)
        cond = None
        if force_cond or self.rng.random() >= self.args.cond_dropout:   # F1.1.2 cond-dropout
            cond = torch.stack([(ca if p["cond"] == "a" else (cb if cb is not None else ca))
                                for p in positions]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.args.no_amp):
            pred = self.decoder(dec_states, cond=cond)
        target = D.load_frames(meta, self.cache, [p["frame"] for p in positions]) \
            .to(self.device).unsqueeze(1)
        return pred, target, dec_states, cond

    # ---------- DINO instance space (phase B) ---------------------------------------------------
    def dino_z(self, lat):
        """Decoded latent frames (n, 1, 16, 60, 104) -> L2-normalized DINOv2 ViT-B/14 CLS
        (n, 768), differentiable end to end: frozen VAE (bf16) -> [0,1] -> centre crop60 ->
        bilinear 224 -> ImageNet norm -> frozen DINO. Exactly build_bank_dino's pipeline (same
        two functions), so the anchor and the precomputed bank live in the same space."""
        assert self.dino is not None, "dino_z called without a DINO bank (--lam_dino/--dino_log)"
        px = latent_to_pixels(self.vae, lat.to(torch.bfloat16),
                              use_checkpoint=bool(self.args.dino_vae_ckpt))
        x = pixels_to_dino_input(px, crop_frac=self.args.dino_crop, size=self.args.dino_size)
        return Fn.normalize(self.dino(x), dim=-1)

    def dino_positions(self, meta, positions, k=None):
        """The (index-in-positions) subset the DINO term supervises: positions whose target is a
        bank entry (so its DINO embedding is precomputed), anchors first (the revisit task), at
        most k of them, one VAE backward per position is the expensive part of a phase-B step."""
        vid = meta["video_id"]
        cand = [t for t, p in enumerate(positions)
                if (vid, int(p["frame"])) in self.dino_key2row]
        cand.sort(key=lambda t: (0 if positions[t]["kind"] == "anchor" else 1, t))
        return cand[:(self.args.dino_k if k is None else k)]

    def dino_term(self, meta, positions, pred, side, pool=None, mask=None, k=None):
        """InfoNCE in DINO space: anchor = the decoded frame, positive = the precomputed
        embedding of its true target, negatives = the same split-side bank rows the latent term
        uses (twin + cond-kNN included / forced, F1.3 mask applied). Returns (loss, logs)."""
        if self.dino_emb is None:
            return None, {}
        sel = self.dino_positions(meta, positions, k=k)
        if not sel:
            return None, {}
        vid = meta["video_id"]
        rows = torch.tensor([self.dino_key2row[(vid, int(positions[t]["frame"]))] for t in sel],
                            device=self.device)
        if pool is None:
            _, mask, pool = self.bank.negatives(
                side, meta, [p["chunk_i"] for p in positions], n_neg=self.args.n_neg,
                mask_chunks=self.args.mask_chunks)
        z = self.dino_z(pred[sel])
        pos_e = self.dino_emb[rows]
        neg = self.dino_emb[pool]
        m = mask[sel] if mask is not None else None
        loss, acc = infonce(z, pos_e, neg, self.args.dino_tau, m)
        with torch.no_grad():
            cp = (z * pos_e).sum(-1)
            sn = z @ neg.t()
            cb = (sn.masked_fill(m, -2.0) if m is not None else sn).max(dim=1).values
            logs = {"dino": float(loss), "dino_acc": float(acc),
                    "dino_gap": float((cp - cb).mean()), "dino_cos_pos": float(cp.mean()),
                    "dino_cos_best_neg": float(cb.mean()), "dino_n": float(len(sel)),
                    "dino_n_anchor": float(sum(1 for t in sel
                                               if positions[t]["kind"] == "anchor"))}
        return loss, logs

    # ---------- T0 term (memoire 2.1) -----------------------------------------------------------
    def t0_term(self, meta, positions, pred, side, grad=True):
        """Terme contrastif T0 sur l'ancre decodee a la position de retour simulee.

        Support : les positions de kind 'anchor', et elles seules. Ce sont, par construction de
        `mem2_data.sample_positions`, exactement les etats post-switch avec cond = A : la
        situation du revisit. `--t0_end_pos` force la premiere d'entre elles a l'etat de fin de
        sequence (la position de retour la plus profonde disponible dans le corpus).
        POSITIFS = segment A de CETTE video ; NEGATIFS = son interlude B + toutes les autres
        videos du bon cote du split (jumeau inclus, loggue a part).
        Retourne (loss | None, logs)."""
        a = self.args
        if self.t0bank is None or not self.t0bank.has(meta):
            return None, {}
        sel = [t for t, p in enumerate(positions) if p["kind"] == "anchor"]
        if not sel:
            return None, {}
        sel = sel[:a.t0_k]
        E_a, E_b, E_o, E_tw = self.t0bank.refs(meta, side)
        x = pred[sel]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z, band = T0M.t0_embed(x, a.t0_space, n=a.t0_n, lat_mode=a.t0_lat_mode,
                                   vae=self.vae, use_checkpoint=bool(a.t0_vae_ckpt),
                                   return_scale=True)
            loss, logs = T0M.t0_loss(z, E_a, E_b, E_o, tau=a.t0_tau, m0=a.t0_m0,
                                     kind=a.t0_loss, w_margin=a.t0_w_margin, w_b=a.t0_w_b)
        with torch.no_grad():
            if E_tw is not None:
                logs["t0_corr_twin"] = float(T0M.corr_set(z.detach(), E_tw).mean())
            # tripwire anti-Goodhart (G1) : une ancre qui s'aplatit a une correlation mal
            # definie et peut faire monter la marge sans rien apprendre.
            # audit 2026-08-04 : mesure dans l'ESPACE de la METRIQUE (contraste de la bande
            # basse decodee), pas sur le latent, un latent bruite decode parfois plat.
            logs["t0_anchor_std"] = float(band.detach().mean())
            logs["t0_anchor_std_lat"] = T0M.anchor_spatial_std(x.detach())
            logs["t0_k"] = float(len(sel))
        return loss, logs

    # ---------- losses on one video -----------------------------------------------------------
    def video_losses(self, meta, positions, pred, target, side, lam, lam_d=0.0, lam_t0=0.0):
        a = self.args
        per = norm_mse_per(pred, target, subject_weight=a.subject_weight)
        w = a.single_w if meta["mode"] == "single" else 1.0
        loss = per.mean() * w * a.rec_w
        logs = {"rec": float(per.mean()), "w": w}
        z = D.latent_embed(pred, frac=a.crop_frac, pool=a.pool)
        pos_e = D.latent_embed(target, frac=a.crop_frac, pool=a.pool).detach()
        neg, mask, pool = self.bank.negatives(
            side, meta, [p["chunk_i"] for p in positions], n_neg=a.n_neg,
            mask_chunks=a.mask_chunks)
        li, acc = infonce(z, pos_e, neg, a.tau, mask)
        if lam > 0:
            loss = loss + lam * li * w
        logs.update({"inst": float(li), "acc": float(acc), "n_neg": int(neg.shape[0])})
        # phase-B instance term (DINO). lam_d == 0 with --dino_log still measures it (that is
        # the gate quantity) without ever letting it move a weight.
        if self.dino_emb is not None and (lam_d > 0 or a.dino_log):
            if lam_d > 0:
                dl, dlogs = self.dino_term(meta, positions, pred, side, pool=pool, mask=mask)
                if dl is not None:
                    loss = loss + lam_d * dl * w
            else:                      # pure measurement (--dino_log): never build a VAE graph
                with torch.no_grad():
                    _, dlogs = self.dino_term(meta, positions, pred.detach(), side,
                                              pool=pool, mask=mask)
            logs.update(dlogs)
        # memoire 2.1: the T0 low-band term. Lam_t0 == 0 with --t0_log still measures the gate
        # quantity (the sealed margin) without ever letting it move a weight.
        if self.t0bank is not None and (lam_t0 > 0 or a.t0_log):
            if lam_t0 > 0:
                tl, tlogs = self.t0_term(meta, positions, pred, side, grad=True)
                if tl is not None:
                    loss = loss + lam_t0 * tl * w
            else:
                with torch.no_grad():
                    tl, tlogs = self.t0_term(meta, positions, pred.detach(), side, grad=False)
            if tlogs:
                logs.update(tlogs)
                if tl is not None:
                    logs["t0"] = float(tl)
        # diagnostics: the same-prompt twin restricted softmax (the F1.1b hard negative)
        tw = self.bank.twin_entries(meta)
        if tw is not None and tw.numel():
            _, acc_tw = infonce(z.detach(), pos_e, self.bank.emb_all[tw], a.tau)
            logs["acc_twin"] = float(acc_tw)
        return loss, per.detach(), logs, z.detach()

    # ---------- T0 per-video evaluation (the sealed gate quantity: G1 / G2a) --------------------
    @torch.no_grad()
    def t0_eval(self, videos, side, k=None, tag=""):
        """Marge T0 video par video (les gates comptent des videos, pas seulement une moyenne).
        Renvoie (resume, lignes). Etats rejoues (n_bptt=0, bitwise le replay complet, selftest
        phase B test 5), cond FORCEE (comme au deploiement : au revisit le prompt A est donne)."""
        assert self.t0bank is not None, "t0_eval sans banque T0 (--t0_bank/--lam_t0/--t0_log)"
        self.sync_masters_to_cells()
        rows = []
        for meta in videos:
            if not self.t0bank.has(meta):
                continue
            # RNG DEDIE et fixe : le gate doit etre reproductible et ne doit pas perturber
            # la sequence d'entrainement (self.rng reste intact).
            positions = self.positions_for(meta, k=(k or self.args.K), anchor_frac=1.0,
                                           rng=random.Random(20260804))
            if not any(p["kind"] == "anchor" for p in positions):
                continue
            pred, _, _, _ = self.forward_video(meta, positions, n_bptt=0, force_cond=True)
            _, lg = self.t0_term(meta, positions, pred, side, grad=False)
            del pred
            if not lg:
                continue
            rows.append({"video_id": meta["video_id"], "mode": meta["mode"], **lg})
        assert rows, "t0_eval: aucune video n'a de reference T0 (banque/split incoherents ?)"
        n = len(rows)
        summ = {f"{tag}margin": sum(r["t0_margin"] for r in rows) / n,
                f"{tag}margin_o": sum(r["t0_margin_o"] for r in rows) / n,
                f"{tag}corr_a": sum(r["t0_corr_a"] for r in rows) / n,
                f"{tag}corr_b": sum(r["t0_corr_b"] for r in rows) / n,
                f"{tag}corr_o": sum(r["t0_corr_o"] for r in rows) / n,
                f"{tag}anchor_std": sum(r["t0_anchor_std"] for r in rows) / n,
                f"{tag}frac_pos": sum(1 for r in rows if r["t0_margin"] > 0) / n,
                f"{tag}n_pos": float(sum(1 for r in rows if r["t0_margin"] > 0)),
                f"{tag}n": float(n)}
        if all("t0_corr_twin" in r for r in rows):
            summ[f"{tag}corr_twin"] = sum(r["t0_corr_twin"] for r in rows) / n
        return summ, rows

    # ---------- causal margin (F1.1.4) ---------------------------------------------------------
    @torch.no_grad()
    def causal_margin(self, meta, positions, target, acc_true, side):
        """acc(true state) - acc(state of another video, same cond): if ~0 the phase is inert
        whatever the absolute accuracy. Uses the previous step's states (always available,
        works at batch size 1)."""
        if self.prev_states is None:
            return None
        prev, prev_k = self.prev_states
        k = min(prev_k, len(positions))
        st = [tuple(t[:k] for t in b) for b in prev]
        ca, cb = D.load_conds(meta, self.cache)
        cond = torch.stack([(ca if p["cond"] == "a" else (cb if cb is not None else ca))
                            for p in positions[:k]]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.args.no_amp):
            pred = self.decoder(st, cond=cond)
        z = D.latent_embed(pred, frac=self.args.crop_frac, pool=self.args.pool)
        pos_e = D.latent_embed(target[:k], frac=self.args.crop_frac, pool=self.args.pool)
        neg, mask, _ = self.bank.negatives(side, meta, [p["chunk_i"] for p in positions[:k]],
                                           n_neg=self.args.n_neg,
                                           mask_chunks=self.args.mask_chunks)
        _, acc_sh = infonce(z, pos_e, neg, self.args.tau, mask)
        return float(acc_true) - float(acc_sh)

    # ---------- influence decay (F3.2) ----------------------------------------------------------
    @torch.no_grad()
    def influence_probe(self, meta, ks=(1, 2, 5, 10, 20)):
        """Replay the chain, then replay it again with h_{T-k} replaced by its per-token mean
        (an 'uninformative write' of the same magnitude) and report how much the final state and
        the decoded loss move. A 'cheat-short' writer (F3.2) collapses to ~0 past k=2."""
        positions = D.sample_positions(meta, random.Random(0), k=1, anchor_frac=1.0,
                                       min_age_writes=1)
        for p in positions:
            p["chunk_i"] = meta["write_chunks"][p["j"]]
        i_max = max(p["i"] for p in positions)
        h = self.get_h(meta, i_max)
        base = {}
        for bi in self.blocks:
            base[bi] = replay_writes(self.cells[bi], h[bi][:i_max + 1], detach_state=True,
                                     n_bptt_writes=0, use_checkpoint=False)
        dec = [tuple(base[bi][k] for k in range(3)) for bi in self.blocks]
        ca, _ = D.load_conds(meta, self.cache)
        cond = ca[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.args.no_amp):
            pred = self.decoder(dec, cond=cond)
        tgt = D.load_frames(meta, self.cache, [p["frame"] for p in positions]) \
            .to(self.device).unsqueeze(1)
        loss_ref = float(norm_mse(pred, tgt))
        out = {}
        T = i_max + 1
        for k in ks:
            if k > T:
                continue
            j = T - k
            pert = {}
            for bi in self.blocks:
                hs = list(h[bi][:T])
                hs[j] = hs[j].mean(dim=1, keepdim=True).expand_as(hs[j]).contiguous()
                pert[bi] = replay_writes(self.cells[bi], hs, detach_state=True,
                                         n_bptt_writes=0, use_checkpoint=False)
            rel = sum(float((pert[bi][t] - base[bi][t]).float().norm()
                            / base[bi][t].float().norm().clamp_min(1e-9))
                      for bi in self.blocks for t in range(3)) / (3 * len(self.blocks))
            dec_k = [tuple(pert[bi][t] for t in range(3)) for bi in self.blocks]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.args.no_amp):
                pk = self.decoder(dec_k, cond=cond)
            out[k] = (rel, float(norm_mse(pk, tgt)) - loss_ref)
        return loss_ref, out

    # ---------- state health (F3.2/F3.5 tripwires) ------------------------------------------------
    @torch.no_grad()
    def state_health(self, states):
        rms, capped, n = 0.0, 0.0, 0
        for bi, st in zip(self.blocks, states):
            cell = self.cells[bi]
            inits = (cell.w0, cell.w1, cell.w2)
            for t, w in zip(st, inits):
                r = t.float()
                rms += float(r.pow(2).mean().sqrt())
                ratio = r.norm(dim=1) / w.detach().float().norm(dim=1).clamp_min(1e-9)
                capped += float((ratio >= 1.0 - 1e-3).float().mean())
                n += 1
        return rms / n, capped / n

    # ---------- optimizer plumbing ------------------------------------------------------------
    def sync_masters_to_cells(self):
        with torch.no_grad():
            for k, m in self.masters.items():
                self.wparams[k].copy_(m.to(self.wparams[k].dtype))

    def gather_grads(self):
        gn = 0.0
        for k, m in self.masters.items():
            g = self.wparams[k].grad
            m.grad = None if g is None else g.detach().float().clone()
            if m.grad is not None:
                gn += float(m.grad.pow(2).sum())
            self.wparams[k].grad = None
        return gn ** 0.5

    def prox_loss(self):
        """||theta_w - theta_l1||^2 on the fp32 masters, F4.1(ii) makes it obligatory in
        phase A (n_eff ~ 135 videos vs 21M writer params)."""
        if self.args.prox <= 0:
            return None
        s = 0.0
        for (bi, name), m in self.masters.items():
            s = s + (m - self.ref[bi][name].float()).pow(2).sum()
        return self.args.prox * s

    # ---------- the loop -----------------------------------------------------------------------
    def positions_for(self, meta, k=None, min_age=None, anchor_frac=None, rng=None):
        if self.args.fixed_positions and k is None and anchor_frac is None:
            # toy/overfit mode: one fixed (state, target) set per video (drawn once, seeded),
            # instead of a fresh draw every step, an overfit gate must see the same points.
            key = meta["video_id"]
            if key not in self._fixed_pos:
                self._fixed_pos[key] = self.positions_for(
                    meta, k=self.args.K, min_age=min_age,
                    anchor_frac=self.args.anchor_frac)
            return self._fixed_pos[key]
        pos = D.sample_positions(
            meta, (rng if rng is not None else self.rng), k=(k or self.args.K),
            ages=tuple(self.args.ages),
            anchor_frac=self.args.anchor_frac if anchor_frac is None else anchor_frac,
            min_age_writes=self.args.min_age if min_age is None else min_age)
        for p in pos:
            p["chunk_i"] = meta["write_chunks"][p["j"]]
        return self._force_end_pos(meta, pos)

    def _force_end_pos(self, meta, pos):
        """Delegue a t0_bank.force_end_pos (module sans CUDA, donc selftestable sur
        CPU : `import wan` initialise un contexte CUDA des l'import).
        audit 2026-08-04 : conditionne AUSSI a la presence du terme T0. Sans cela le defaut
        `--t0_end_pos 1` modifiait le tirage de positions meme a `--lam_t0 0`, ce qui CASSAIT
        la byte-safety promise (le run n'etait plus la recette A2)."""
        enabled = bool(self.args.t0_end_pos) and self.t0bank is not None
        return force_end_pos(meta, pos, enabled=enabled)

    def train_step(self, videos, lam, tag="train", lam_d=0.0, lam_t0=0.0):
        a = self.args
        self.sync_masters_to_cells()
        self.opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "rec": 0.0, "inst": 0.0, "acc": 0.0, "n": 0}
        dino_agg, t0_agg = {}, {}
        by_age, by_kind, margins, twin_accs = {}, {}, [], []
        last_states = None
        for meta in videos:
            positions = self.positions_for(meta)
            h_over = None
            if a.h_shuffle:
                # toy control (c): the h sequence comes from another video, re-drawn at every
                # step (a fixed pairing would just be memorized by the decoder). The state can
                # then carry nothing about the target -> the loss must stall.
                i_max = max(p["i"] for p in positions)
                # the pool is the run's video pool (self.shuffle_pool), not the current batch:
                # at bs=1 a batch has no other video and the control would silently no-op
                # (caught at J3: the hshuffle curve came out bit-identical to main).
                pool = self.shuffle_pool or videos
                cands = [v for v in pool if v["video_id"] != meta["video_id"]
                         and len(v["write_chunks"]) > i_max]
                assert cands, "h_shuffle: no candidate video to swap h with"
                h_over = self.get_h(self.rng.choice(cands), i_max)
            pred, target, dec_states, _ = self.forward_video(meta, positions, h_override=h_over)
            loss, per, logs, _z = self.video_losses(meta, positions, pred, target, "train", lam,
                                                    lam_d=lam_d, lam_t0=lam_t0)
            (loss / len(videos)).backward()
            agg["loss"] += float(loss)
            agg["rec"] += logs["rec"]
            agg["inst"] += logs.get("inst", 0.0)
            agg["acc"] += logs.get("acc", 0.0)
            agg["n"] += 1
            for k in ("dino", "dino_acc", "dino_gap", "dino_cos_pos", "dino_cos_best_neg",
                      "dino_n", "dino_n_anchor"):
                if k in logs:
                    dino_agg.setdefault(k, []).append(logs[k])
            for k, v in logs.items():
                if k == "t0" or k.startswith("t0_"):
                    t0_agg.setdefault(k, []).append(v)
            if "acc_twin" in logs:
                twin_accs.append(logs["acc_twin"])
            for p, v in zip(positions, per.tolist()):
                by_age.setdefault(D.age_bucket(p["age"]), []).append(v)
                by_kind.setdefault(f"{meta['mode']}/{p['kind']}", []).append(v)
            m = self.causal_margin(meta, positions, target, logs.get("acc", 0.0), "train")
            if m is not None:
                margins.append(m)
            last_states = ([tuple(t.detach() for t in b) for b in dec_states], len(positions))
            del pred, target, dec_states, loss
        self.prev_states = last_states
        # ORDER matters: the replay grads live on the bf16 cell params -> copy them onto the
        # fp32 masters first, then let the proximal term accumulate on top (backward() adds).
        wgn = 0.0
        if not self.freeze_writer:
            wgn = self.gather_grads()
            prox = self.prox_loss()
            if prox is not None:
                prox.backward()
                wgn = float(sum(float(m.grad.pow(2).sum()) for m in self.masters.values()
                                if m.grad is not None) ** 0.5)
            torch.nn.utils.clip_grad_norm_(list(self.masters.values()), a.clip)
        dgn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.decoder.parameters() if p.requires_grad], a.clip)
        self.opt.step()
        self.sched.step()
        self.peak_vram = max(self.peak_vram,
                             torch.cuda.max_memory_allocated() / 1e9
                             if torch.cuda.is_available() else 0.0)
        n = max(1, agg["n"])
        out = {f"{tag}/loss": agg["loss"] / n, f"{tag}/rec": agg["rec"] / n,
               f"{tag}/inst": agg["inst"] / n, f"{tag}/acc": agg["acc"] / n,
               "writer/grad_norm": wgn, "dec/grad_norm": float(dgn),
               "train/lr": self.sched.get_last_lr()[-1], "inst/lambda": lam,
               "dino/lambda": lam_d, "t0/lambda": lam_t0, "vram/peak_gb": self.peak_vram}
        for k, v in dino_agg.items():
            out[f"dino/{k[5:] if k.startswith('dino_') else 'loss'}"] = sum(v) / len(v)
        for k, v in t0_agg.items():
            out[f"t0/{k[3:] if k.startswith('t0_') else 'loss'}"] = sum(v) / len(v)
        for k, v in by_age.items():
            out[f"{tag}/loss_by_age/{k}"] = sum(v) / len(v)
        for k, v in by_kind.items():
            out[f"{tag}/loss_by_kind/{k}"] = sum(v) / len(v)
        if margins:
            out["ctrl/causal_margin"] = sum(margins) / len(margins)
        if twin_accs:
            out["inst/acc_twin"] = sum(twin_accs) / len(twin_accs)
        return out

    @torch.no_grad()
    def evaluate(self, n_videos=8):
        a = self.args
        self.sync_masters_to_cells()
        vids = self.val_vids[:n_videos]
        tot, by_age, by_kind, accs, gaps, hs = [], {}, {}, [], [], []
        dino_agg, t0_agg, t0_pos = {}, {}, []
        for meta in vids:
            positions = self.positions_for(meta)
            pred, target, dec_states, _ = self.forward_video(meta, positions, n_bptt=0,
                                                             force_cond=True)
            per = norm_mse_per(pred, target)     # val loss unweighted + cond always on, so it
            # is comparable with the v1.x decoder vals (0.867 v1.1 / 0.812 v1.2)
            tot += per.tolist()
            for p, v in zip(positions, per.tolist()):
                by_age.setdefault(D.age_bucket(p["age"]), []).append(v)
                by_kind.setdefault(f"{meta['mode']}/{p['kind']}", []).append(v)
            z = D.latent_embed(pred, frac=a.crop_frac, pool=a.pool)
            pe = D.latent_embed(target, frac=a.crop_frac, pool=a.pool)
            neg, mask, _ = self.bank.negatives("val", meta, [p["chunk_i"] for p in positions],
                                               n_neg=a.n_neg, mask_chunks=a.mask_chunks)
            _, acc = infonce(z, pe, neg, a.tau, mask)
            accs.append(float(acc))
            # health of the instance term (measured at J3: it starts negative, a decoded
            # anchor is closer to some other video's frame than to its own target, at every
            # pooling factor, centered or not). Ramp lambda_inst only once this turns positive.
            gaps.append(float(((z * pe).sum(-1)
                               - (z @ neg.t()).masked_fill(mask, -2.0).max(dim=1).values).mean()))
            # phase B: the same quantities in DINO space (the gate + the reported val sim)
            if self.dino_emb is not None:
                _, dl = self.dino_term(meta, positions, pred, "val", pool=None, mask=None,
                                       k=self.args.dino_k_val)
                for k, v in dl.items():
                    dino_agg.setdefault(k, []).append(v)
            # MEMOIRE 2.1 : la quantite des gates, mesuree sur des videos HELD-OUT (G2a), video
            # par video (le gate est un compte de videos, pas seulement une moyenne).
            if self.t0bank is not None and self.t0bank.has(meta):
                _, tl = self.t0_term(meta, positions, pred, "val", grad=False)
                if tl:
                    for k, v in tl.items():
                        t0_agg.setdefault(k, []).append(v)
                    if "t0_margin" in tl:
                        t0_pos.append(1.0 if tl["t0_margin"] > 0 else 0.0)
            hs.append(self.state_health(dec_states))     # (rms, rowcap_frac) tripwires
            del pred, target, dec_states
        out = {"val/loss": sum(tot) / len(tot), "val/acc": sum(accs) / len(accs),
               "inst/pos_gap": sum(gaps) / len(gaps),
               "state/rms": sum(h[0] for h in hs) / len(hs),
               "state/rowcap_frac": sum(h[1] for h in hs) / len(hs)}
        for k, v in dino_agg.items():
            out[f"val/dino_{k[5:] if k.startswith('dino_') else 'loss'}"] = sum(v) / len(v)
        for k, v in t0_agg.items():
            out[f"val/t0_{k[3:] if k.startswith('t0_') else 'loss'}"] = sum(v) / len(v)
        if t0_pos:
            out["val/t0_frac_videos_pos"] = sum(t0_pos) / len(t0_pos)
            out["val/t0_n_videos"] = float(len(t0_pos))
        if "val/dino_gap" in out:
            self.last_dino_gap = out["val/dino_gap"]     # drives the lambda_dino gate
        if self.freeze_writer:                            # phase-B contract, checked at every val
            d = delta_l1(self.cells, self.ref_start)
            out["writer/delta_ref"] = d
            assert d == 0.0, f"PHASE-B VIOLATION: the frozen writer moved (delta_ref={d:.3e})"
        for k, v in by_age.items():
            out[f"val/loss_by_age/{k}"] = sum(v) / len(v)
        for k, v in by_kind.items():
            out[f"val/loss_by_kind/{k}"] = sum(v) / len(v)
        return out

    def writer_diag(self):
        cell = self.cells[self.blocks[0]]
        out = {"writer/delta_l1": delta_l1(self.cells, self.ref),
               "writer/delta_ref": delta_l1(self.cells, self.ref_start)}
        if hasattr(cell, "decay_logit"):
            out["writer/decay"] = float(torch.sigmoid(cell.decay_logit.detach().float()))
        out["writer/lr_softplus_bias"] = float(
            Fn.softplus(cell.lrs.bias.detach().float() + cell.inv_base_lr).mean())
        return out

    def save(self, path):
        writer = {}
        for bi in self.blocks:
            writer[str(bi)] = {name: self.wparams[(bi, name)].detach().to(torch.bfloat16).cpu()
                               for (b, name) in self.wparams if b == bi}
        torch.save({"writer": writer, "decoder": self.decoder.state_dict(),
                    "step": self.step, "args": vars(self.args),
                    "theta_l1_ckpt": str(self.config.lora_ckpt)}, path)
        print(f"[writer] ckpt -> {path}", flush=True)


# ----------------------------- retrieval-space probe ----------------------------------------------
@torch.no_grad()
def acc_probe(tr, n_videos=10, pools=(2, 4, 8, 16)):
    """How retrievable is a v1.2-decoded anchor in latent crop space, per pooling factor?
    The smoke showed inst/acc = 0 against the full 3.4k bank at pool=2: a decoded frame is
    low-pass by nature, so a fine-grained latent cosine can be dominated by the sharpness gap
    rather than by identity. This measures the effect before the long run picks a default
    (and reports cos(pred, its own target) vs the best competing bank entry)."""
    print("\n[writer][acc_probe] retrieval of the DECODED anchor in latent crop space", flush=True)
    from build_bank import BANK_LAT
    bl = torch.load(BANK_LAT, map_location="cpu", weights_only=False)["latents"]
    neg_idx = tr.bank.sides["val"].cpu()
    neg_lat = bl[neg_idx].to(tr.device).float()
    vids = tr.val_vids[:n_videos]
    out = {}
    preds, tgts = [], []
    for meta in vids:
        pos = tr.positions_for(meta, k=2, anchor_frac=1.0)
        pred, target, _, _ = tr.forward_video(meta, pos, n_bptt=0, force_cond=True)
        preds.append(pred.float())
        tgts.append(target.float())
    for pool in pools:
        for center in (False, True):
            # centering = subtract the bank mean embedding before the cosine. A decoded frame
            # is low-pass by construction, so the raw latent cosine is dominated by the common
            # "smooth frame" direction shared by every entry; removing it is the standard fix
            # (and it is free: the bank never moves).
            neg = D.latent_embed(neg_lat, frac=tr.args.crop_frac, pool=pool)
            mu = neg.mean(0, keepdim=True) if center else None
            if center:
                neg = Fn.normalize(neg - mu, dim=1)
            accs, cos_pos, cos_best = [], [], []
            for pred, target in zip(preds, tgts):
                z = D.latent_embed(pred, frac=tr.args.crop_frac, pool=pool)
                pe = D.latent_embed(target, frac=tr.args.crop_frac, pool=pool)
                if center:
                    z, pe = Fn.normalize(z - mu, dim=1), Fn.normalize(pe - mu, dim=1)
                _, acc = infonce(z, pe, neg, tr.args.tau)
                accs.append(float(acc))
                cos_pos.append(float((z * pe).sum(-1).mean()))
                cos_best.append(float((z @ neg.t()).max(dim=1).values.mean()))
            out[(pool, center)] = (sum(accs) / len(accs), sum(cos_pos) / len(cos_pos),
                                   sum(cos_best) / len(cos_best))
            r = out[(pool, center)]
            print(f"[writer][acc_probe] pool={pool:>2} center={int(center)}: "
                  f"top-1 acc {r[0]:.3f} | cos(pred,target) {r[1]:.3f} vs best bank "
                  f"negative {r[2]:.3f}", flush=True)
    return out


# ----------------------------- DINO retrieval probe (phase-B gate) --------------------------------
@torch.no_grad()
def dino_probe(tr, n_videos=10, k=2):
    """The phase-B gate. Same question as J3's --acc_probe, but in the space the verdict is
    measured in: decode val anchors from replayed states with the current writer+decoder,
    embed with the harness DINOv2 ViT-B/14 through the VAE, and retrieve against the val side
    of the precomputed DINO bank.

    In latent space J3 measured top-1 = 0.000 and pos_gap = -0.11..-0.36 at every pooling: the
    instance term would have pushed on the low-pass artefact, not on identity. If pos_gap is
    negative here too, --lam_dino must not be ramped (B2 would train on a wrong-signed signal)."""
    print("\n[writer][dino_probe] retrieval of the DECODED anchor in DINO ViT-B/14 space",
          flush=True)
    assert tr.dino_emb is not None, "dino_probe needs --dino_bank (build_bank_dino.py)"
    accs, gaps, cps, cbs, ns = [], [], [], [], 0
    per_video = []
    for meta in tr.val_vids[:n_videos]:
        pos = tr.positions_for(meta, k=k, anchor_frac=1.0)
        pred, _, _, _ = tr.forward_video(meta, pos, n_bptt=0, force_cond=True)
        _, dl = tr.dino_term(meta, pos, pred, "val", k=k)
        if not dl:
            continue
        accs.append(dl["dino_acc"])
        gaps.append(dl["dino_gap"])
        cps.append(dl["dino_cos_pos"])
        cbs.append(dl["dino_cos_best_neg"])
        ns += int(dl["dino_n"])
        per_video.append((meta["video_id"], meta["mode"], dl["dino_gap"], dl["dino_cos_pos"],
                          dl["dino_cos_best_neg"], dl["dino_acc"]))
        del pred
    assert accs, "dino_probe: no val position had a bank entry, bank/split mismatch"
    out = {"dino_probe/top1": sum(accs) / len(accs),
           "dino_probe/pos_gap": sum(gaps) / len(gaps),
           "dino_probe/cos_pos": sum(cps) / len(cps),
           "dino_probe/cos_best_neg": sum(cbs) / len(cbs),
           "dino_probe/n_pos": float(ns), "dino_probe/n_videos": float(len(accs))}
    for vid, mode, g, cp, cb, ac in per_video:
        print(f"[writer][dino_probe]   {vid:>12} {mode:>6}: gap {g:+.4f} "
              f"(cos_pos {cp:.3f} vs best neg {cb:.3f}), top-1 {ac:.2f}", flush=True)
    print(f"[writer][dino_probe] {len(accs)} val videos / {ns} anchors: "
          f"top-1 {out['dino_probe/top1']:.3f} | cos(pred,target) "
          f"{out['dino_probe/cos_pos']:.3f} vs best negative "
          f"{out['dino_probe/cos_best_neg']:.3f} | POS_GAP "
          f"{out['dino_probe/pos_gap']:+.4f}", flush=True)
    print(f"[writer][dino_probe] GATE: pos_gap > 0 ? "
          f"{'PASS, lambda_dino may be ramped' if out['dino_probe/pos_gap'] > 0 else 'FAIL, do NOT ramp lambda_dino'}",
          flush=True)
    return out


# ----------------------------- T0 probe (MEMOIRE 2.1, gate G2a) ----------------------------------
@torch.no_grad()
def t0_probe(tr, n_videos=26, k=2):
    """Gate G2a (registre) : la marge T0 de l'ancre decodee, sur les videos val held-out.
    rule SCELLEE : marge moyenne >= +0.05 ET >= 60 % des videos val a marge > 0.
    Reference publiee de depart (24 cellules S2, 16.1 quater) : corr vs A* +0.042, vs B +0.298,
    soit une marge de -0.256, c'est ce chiffre que 2.1 doit retourner."""
    print("\n[writer][t0_probe] marge bande-basse de l'ancre decodee, videos VAL held-out",
          flush=True)
    summ, rows = tr.t0_eval(tr.val_vids[:n_videos], "val", k=k)
    for r in rows:
        print(f"[writer][t0_probe]   {r['video_id']:>12} {r['mode']:>6}: marge "
              f"{r['t0_margin']:+.4f}  (A {r['t0_corr_a']:+.4f} | B {r['t0_corr_b']:+.4f} | "
              f"autres {r['t0_corr_o']:+.4f}"
              + (f" | jumeau {r['t0_corr_twin']:+.4f}" if "t0_corr_twin" in r else "")
              + f")  std {r['t0_anchor_std']:.4f}", flush=True)
    ok = (summ["margin"] >= tr.args.t0_gate_margin
          and summ["frac_pos"] >= tr.args.t0_gate_frac)
    print(f"[writer][t0_probe] {int(summ['n'])} videos val : MARGE {summ['margin']:+.4f} "
          f"(A {summ['corr_a']:+.4f} | B {summ['corr_b']:+.4f} | autres {summ['corr_o']:+.4f}"
          + (f" | jumeau {summ['corr_twin']:+.4f}" if "corr_twin" in summ else "")
          + f") | videos a marge > 0 : {int(summ['n_pos'])}/{int(summ['n'])} "
            f"({100 * summ['frac_pos']:.0f} %)", flush=True)
    print(f"[writer][t0_probe] GATE G2a (marge >= {tr.args.t0_gate_margin:+.2f} ET "
          f">= {100 * tr.args.t0_gate_frac:.0f} % de videos positives) : "
          f"{'PASS' if ok else 'FAIL, STOP avant tout rollout'}", flush=True)
    out = {f"t0_probe/{k_}": v for k_, v in summ.items()}
    out["t0_probe/gate"] = float(ok)
    return out, rows, ok


# ----------------------------- VRAM probe (F5.4) --------------------------------------------------
def vram_probe(tr, lam_d=0.0):
    print(f"\n[writer][F5.4] VRAM of the FULL backward (replay + decode"
          f"{' + VAE 1 frame + DINO' if lam_d > 0 else ', no VAE in the loss'})", flush=True)
    rows = []
    for K in tr.args.probe_K:
        for nb in tr.args.probe_bptt:
            torch.cuda.reset_peak_memory_stats()
            meta = [v for v in tr.train_vids if v["mode"] == "switch"][0]
            pos = tr.positions_for(meta, k=K)
            t0 = time.time()
            pred = target = st = None
            try:
                tr.sync_masters_to_cells()
                tr.opt.zero_grad(set_to_none=True)
                pred, target, st, _ = tr.forward_video(meta, pos, n_bptt=(None if nb < 0 else nb))
                loss, _, logs, _ = tr.video_losses(meta, pos, pred, target, "train", 0.3,
                                                   lam_d=lam_d, lam_t0=tr.args.lam_t0)
                loss.backward()
                if not tr.freeze_writer:
                    tr.gather_grads()
                peak = torch.cuda.max_memory_allocated() / 1e9
                rows.append((K, nb, peak, time.time() - t0, float(loss)))
                print(f"[writer]   K={K} n_bptt={'full' if nb < 0 else nb}: "
                      f"peak {peak:.1f} GB, {time.time() - t0:.1f} s/step, loss {float(loss):.4f}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append((K, nb, float("nan"), float("nan"), float("nan")))
                print(f"[writer]   K={K} n_bptt={nb}: OOM", flush=True)
            finally:
                del pred, target, st
                torch.cuda.empty_cache()
    return rows


# ----------------------------- toy gate (F3.1) -----------------------------------------------------
def load_toy_results(name="toy_result.json"):
    """Previous variants' summaries (so one variant can be re-run without redoing the others).
    MEMOIRE 2.1 : les toys T0 ecrivent dans `toy_t0_result.json`, une reprise partielle doit
    relire LE MEME fichier, sinon le controle h-shuffle de G1 serait declare manquant."""
    p = os.path.join(OUT_ROOT, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if isinstance(v, dict) and "last" in v}


def run_toy(args, wandb):
    """The redefined toy gate: >= 8 videos, targets >= 10 writes old.
    pass iff (a) the loss collapses, (b) n_bptt_writes=1 does clearly worse, (c) h-shuffle
    degrades. Anything else = the gradient does not carry long-range write information."""
    results = {}
    want = [v.strip() for v in args.toy_variants.split(",") if v.strip()]
    _res_name = "toy_t0_result.json" if args.lam_t0 > 0 else "toy_result.json"
    results = load_toy_results(_res_name) if len(want) < 3 else {}
    for variant in want:
        a = copy.deepcopy(args)
        a.toy = True
        if variant == "bptt1":
            a.n_bptt = 1
        if variant == "hshuffle":
            a.h_shuffle = True
        tr = WriterTrainer(a)
        sw = [v for v in tr.train_vids if v["mode"] == "switch"][:a.toy_videos]
        assert len(sw) >= 8, f"toy needs >= 8 switch videos, got {len(sw)}"
        tr.shuffle_pool = sw                       # the h-shuffle control draws from the toy pool
        # MEMOIRE 2.1 / gate G1 : la marge T0 AVANT tout pas (le point de depart publie est
        # -0.256 sur les 24 cellules S2 ; on le re-mesure ici, sur le corpus, avant de bouger).
        t0_before = None
        if tr.t0bank is not None:
            t0_before, _ = tr.t0_eval(sw, "train", tag="")
            print(f"[toy:{variant}] T0 AVANT : marge {t0_before['margin']:+.4f} "
                  f"(A {t0_before['corr_a']:+.4f} | B {t0_before['corr_b']:+.4f}), "
                  f"{int(t0_before['n_pos'])}/{int(t0_before['n'])} positives, "
                  f"std {t0_before['anchor_std']:.4f}", flush=True)
        hist = []
        t0 = time.time()
        for s in range(a.toy_steps):
            tr.step = s + 1
            batch = [sw[(s * a.bs + i) % len(sw)] for i in range(a.bs)]
            lam_t0 = a.lam_t0 * min(1.0, max(0.0, (s - a.lam_t0_ramp[0])
                                             / max(1, a.lam_t0_ramp[1] - a.lam_t0_ramp[0])))
            logs = tr.train_step(batch, lam=a.lam_inst if a.toy_inst else 0.0, tag="train",
                                 lam_t0=lam_t0)
            hist.append(logs["train/rec"])
            wandb.log({f"toy/{variant}/" + k.split("/", 1)[-1]: v for k, v in logs.items()},
                      step=s + 1 + {"main": 0, "bptt1": 10000, "hshuffle": 20000}[variant])
            if (s + 1) % 10 == 0:
                print(f"[toy:{variant}] step {s + 1}/{a.toy_steps} rec {logs['train/rec']:.4f} "
                      f"loss {logs['train/loss']:.4f} "
                      f"t0m {logs.get('t0/margin', float('nan')):+.4f} "
                      f"vram {logs['vram/peak_gb']:.1f} GB "
                      f"({(time.time() - t0) / (s + 1):.1f} s/step)", flush=True)
        n0 = max(3, len(hist) // 10)
        n1 = max(3, len(hist) // 5)
        results[variant] = {"first": sum(hist[:n0]) / n0, "last": sum(hist[-n1:]) / n1,
                            "hist": hist, "vram": tr.peak_vram}
        if tr.t0bank is not None:
            t0_after, t0_rows = tr.t0_eval(sw, "train", tag="")
            results[variant]["t0_before"] = t0_before
            results[variant]["t0_after"] = t0_after
            results[variant]["t0_rows"] = t0_rows
            print(f"[toy:{variant}] T0 APRES : marge {t0_after['margin']:+.4f} "
                  f"(A {t0_after['corr_a']:+.4f} | B {t0_after['corr_b']:+.4f} | "
                  f"autres {t0_after['corr_o']:+.4f}), "
                  f"{int(t0_after['n_pos'])}/{int(t0_after['n'])} positives, "
                  f"std {t0_after['anchor_std']:.4f} "
                  f"({100 * t0_after['anchor_std'] / max(t0_before['anchor_std'], 1e-9):.0f} % "
                  f"du depart)", flush=True)
        tr.step = a.toy_steps
        tr.save(os.path.join(OUT_ROOT, f"toy_{variant}.pt"))   # kept: the eye-check needs the
        # post-toy writer + decoder to decode an anchor from a trained state
        print(f"[toy:{variant}] first {results[variant]['first']:.4f} -> "
              f"last {results[variant]['last']:.4f}", flush=True)
        del tr
        torch.cuda.empty_cache()

    # ---- gate G1 (MEMOIRE 2.1, registre) : la marge T0 devient-elle franchement positive ? ----
    g1 = None
    if args.lam_t0 > 0 and "t0_after" in results.get("main", {}):
        mt = results["main"]
        aft, bef = mt["t0_after"], mt["t0_before"]
        g1_margin = aft["margin"] >= args.t0_g1_margin
        g1_count = aft["n_pos"] >= args.t0_g1_min_pos
        g1_corr_a = aft["corr_a"] >= args.t0_g1_corr_a
        # controle scelle 1 : h-shuffle (la sequence h d'une AUTRE video) doit rester <= 0,
        # sinon la marge vient du decodeur/cond seul et G1 est void.
        sh = results.get("hshuffle", {}).get("t0_after")
        g1_shuffle = (sh is not None) and (sh["margin"] <= 0.0)
        # controle scelle 2 (anti-Goodhart) : l'ancre ne doit pas s'aplatir (>= 25 % de l'ecart
        # -type spatial du depart), sinon la correlation est mal definie et G1 est void.
        std_ratio = aft["anchor_std"] / max(bef["anchor_std"], 1e-9)
        g1_std = std_ratio >= args.t0_g1_std_ratio
        g1 = bool(g1_margin and g1_count and g1_corr_a and g1_shuffle and g1_std)
        print("\n[toy] ================= GATE G1 (marge T0) =================")
        print(f"[toy] depart mesure : marge {bef['margin']:+.4f} "
              f"(A {bef['corr_a']:+.4f} | B {bef['corr_b']:+.4f})  "
              f"[reference publiee 24 cellules S2 : -0.256]")
        print(f"[toy] (1) marge moyenne >= {args.t0_g1_margin:+.2f} : "
              f"{aft['margin']:+.4f} : {'PASS' if g1_margin else 'FAIL'}")
        print(f"[toy] (2) >= {args.t0_g1_min_pos:.0f}/{int(aft['n'])} videos a marge > 0 : "
              f"{int(aft['n_pos'])}/{int(aft['n'])} : {'PASS' if g1_count else 'FAIL'}")
        print(f"[toy] (3) corr(ancre, A) >= {args.t0_g1_corr_a:+.2f} : {aft['corr_a']:+.4f} : "
              f"{'PASS' if g1_corr_a else 'FAIL'}")
        print(f"[toy] (4) CONTROLE h-shuffle : marge <= 0 : "
              + (f"{sh['margin']:+.4f} : {'PASS' if g1_shuffle else 'FAIL -> G1 VOID'}"
                 if sh is not None else "NON MESURE -> G1 VOID"))
        print(f"[toy] (5) CONTROLE anti-Goodhart : std spatiale >= "
              f"{100 * args.t0_g1_std_ratio:.0f} % du depart : {100 * std_ratio:.0f} % : "
              f"{'PASS' if g1_std else 'FAIL -> G1 VOID'}")
        print(f"[toy] VERDICT G1 : {'PASS, on paie l entrainement complet' if g1 else 'FAIL'}")
        print("[toy] ======================================================\n", flush=True)
        wandb.log({"G1/margin": aft["margin"], "G1/margin_before": bef["margin"],
                   "G1/corr_a": aft["corr_a"], "G1/corr_b": aft["corr_b"],
                   "G1/n_pos": aft["n_pos"], "G1/n": aft["n"],
                   "G1/std_ratio": std_ratio, "G1/verdict": int(g1),
                   "G1/hshuffle_margin": (sh["margin"] if sh else float("nan"))}, step=30001)

    if not all(k in results for k in ("main", "bptt1", "hshuffle")):
        print("[toy] F3.1 non evaluable (variantes manquantes), seul G1 est rapporte",
              flush=True)
        with open(os.path.join(OUT_ROOT, _res_name), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "hist"}
                       for k, v in results.items()} | {"G1": g1}, f, indent=2)
        return bool(g1)
    m, b, s = results["main"], results["bptt1"], results["hshuffle"]
    a_ok = m["last"] < args.toy_conv_ratio * m["first"]
    b_ok = b["last"] > m["last"] * (1.0 + args.toy_margin)
    c_ok = s["last"] > m["last"] * (1.0 + args.toy_margin)
    print("\n[toy] ===================== GATE F3.1 =====================")
    print(f"[toy] (a) loss collapses on >= {args.toy_min_age}-write-old targets: "
          f"{m['first']:.4f} -> {m['last']:.4f} "
          f"(ratio {m['last'] / m['first']:.3f} < {args.toy_conv_ratio}) : "
          f"{'PASS' if a_ok else 'FAIL'}")
    print(f"[toy] (b) n_bptt_writes=1 does clearly worse: {b['last']:.4f} vs {m['last']:.4f} "
          f"(+{100 * (b['last'] / m['last'] - 1):.1f}%, need +{100 * args.toy_margin:.0f}%) : "
          f"{'PASS' if b_ok else 'FAIL'}")
    print(f"[toy] (c) h-shuffle degrades: {s['last']:.4f} vs {m['last']:.4f} "
          f"(+{100 * (s['last'] / m['last'] - 1):.1f}%) : {'PASS' if c_ok else 'FAIL'}")
    verdict = a_ok and b_ok and c_ok
    print(f"[toy] VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"(peak VRAM {m['vram']:.1f} GB)")
    print("[toy] =====================================================\n", flush=True)
    wandb.log({"toy/verdict": int(verdict), "toy/a_conv": int(a_ok), "toy/b_bptt1": int(b_ok),
               "toy/c_hshuffle": int(c_ok), "toy/main_last": m["last"],
               "toy/bptt1_last": b["last"], "toy/hshuffle_last": s["last"]}, step=30000)
    with open(os.path.join(OUT_ROOT, _res_name), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "hist"}
                   for k, v in results.items()}
                  | {"verdict": verdict, "a": a_ok, "b": b_ok, "c": c_ok, "G1": g1}, f, indent=2)
    # MEMOIRE 2.1 : quand le terme T0 est actif, LE gate est G1 (F3.1 reste rapporte comme
    # diagnostic de chaine, il a deja ete tranche au J3 : a pass, b +8.9 %, c pass).
    return bool(g1) if args.lam_t0 > 0 else verdict


# ----------------------------- main ------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="memory 2.0 writer trainer (write-replay, J3)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--load", choices=["pipeline", "direct"], default="pipeline",
                    help="pipeline = the canonical harness load (run_reid_probe.build_pipeline)")
    ap.add_argument("--dec_ckpt", default=DEC_V12, help="decoder warm start (v1.2)")
    ap.add_argument("--bank", default=BANK_PT)
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    # ---- phase B: frozen trained writer + v2 decoder + DINO instance term --------------------
    ap.add_argument("--writer_ckpt", default=None,
                    help="MEM_WRITER ckpt whose theta_w replaces l1 (phase B: the A2 writer)")
    ap.add_argument("--freeze_writer", action="store_true",
                    help="theta_w out of the optimizer entirely; delta_ref asserted == 0")
    ap.add_argument("--mixer", action="store_true",
                    help="decoder v2: inter-token spatial mixing before the head (identity init)")
    ap.add_argument("--mixer_blocks", type=int, default=2)
    ap.add_argument("--mixer_hidden", type=int, default=768)
    ap.add_argument("--dino_bank", default=BANK_DINO)
    ap.add_argument("--lam_dino", type=float, default=0.0,
                    help="weight of the DINO InfoNCE instance term (0 = reconstruction only)")
    ap.add_argument("--lam_dino_ramp", type=int, nargs=2, default=[300, 600])
    ap.add_argument("--dino_gate", type=int, default=1,
                    help="1 = lambda_dino stays 0 while the measured val pos_gap <= 0")
    ap.add_argument("--dino_k", type=int, default=1,
                    help="positions per video that pay a differentiable VAE decode")
    ap.add_argument("--dino_k_val", type=int, default=1)
    ap.add_argument("--dino_tau", type=float, default=0.07)
    ap.add_argument("--dino_crop", type=float, default=0.6)
    ap.add_argument("--dino_size", type=int, default=224)
    ap.add_argument("--dino_vae_ckpt", type=int, default=1,
                    help="gradient-checkpoint the VAE decode (memory for compute)")
    ap.add_argument("--dino_log", action="store_true",
                    help="measure the DINO term every step even at lambda 0 (B1 diagnostics)")
    ap.add_argument("--dino_probe", action="store_true",
                    help="THE phase-B gate: DINO-space retrieval of the decoded anchor, then exit")
    ap.add_argument("--dino_probe_videos", type=int, default=10)
    # ---- MEMOIRE 2.1: the T0 low-band contrastive term ---------------------------------------
    ap.add_argument("--lam_t0", type=float, default=0.0,
                    help="poids du terme contrastif T0 (0 = terme JAMAIS construit = recette A2)")
    ap.add_argument("--lam_t0_ramp", type=int, nargs=2, default=[100, 400],
                    help="rampe lineaire (step_debut, step_plein) de lam_t0")
    ap.add_argument("--rec_w", type=float, default=1.0,
                    help="poids de la reco (mandat 2.1: REDUIT, elle devient un regulariseur)")
    ap.add_argument("--t0_bank", default=BANK_T0)
    ap.add_argument("--t0_space", choices=["latent", "pixel"], default="pixel",
                    help="espace du terme T0. DEFAUT = pixel : la regle D0 a REJETE le latent "
                         "(D0.a, CPU : Spearman 0.51/0.45 < 0.70 et plus-proche-voisin 21/18 % "
                         "< 50 %, le latent poole ne reproduit PAS la geometrie bande-basse "
                         "pixel sur des ancres decodees). Ne repasser a 'latent' qu'en notant "
                         "la derogation au registre.")
    ap.add_argument("--t0_lat_mode", choices=["chan", "gray"], default="chan")
    ap.add_argument("--t0_n", type=int, default=T0M.DEFAULT_N, help="bande n x n (8 = le dossier)")
    ap.add_argument("--t0_loss", choices=["infonce", "margin", "both"], default="both")
    ap.add_argument("--t0_tau", type=float, default=0.10)
    ap.add_argument("--t0_m0", type=float, default=0.30, help="marge visee du hinge")
    ap.add_argument("--t0_w_margin", type=float, default=1.0)
    ap.add_argument("--t0_w_b", type=float, default=2.0,
                    help="sur-poids de l'interlude B (LE negatif dur : B > A sur 21/24 aujourd'hui)")
    ap.add_argument("--t0_k", type=int, default=2,
                    help="positions 'anchor' supervisees par T0 et par video")
    ap.add_argument("--t0_end_pos", type=int, default=1,
                    help="1 = forcer la 1re position anchor a l'etat de FIN de sequence")
    ap.add_argument("--t0_vae_ckpt", type=int, default=1,
                    help="gradient-checkpoint du decode VAE (--t0_space pixel uniquement)")
    ap.add_argument("--t0_log", action="store_true",
                    help="MESURER la marge T0 a chaque step meme a lambda 0 (diagnostic)")
    ap.add_argument("--t0_probe", action="store_true",
                    help="mesure la marge T0 sur les videos val puis sort (gate G2a)")
    ap.add_argument("--t0_probe_videos", type=int, default=26)
    # seuils des gates SCELLES au registre (memory2_probes_registry.md, G1 / G2a). Ils sont des
    # flags pour etre lisibles dans wandb.config, pas pour etre bouges : tout changement doit
    # etre note au registre avec sa date et sa raison.
    ap.add_argument("--t0_g1_margin", type=float, default=0.10)
    ap.add_argument("--t0_g1_min_pos", type=float, default=6.0)
    ap.add_argument("--t0_g1_corr_a", type=float, default=0.15)
    ap.add_argument("--t0_g1_std_ratio", type=float, default=0.25)
    ap.add_argument("--t0_gate_margin", type=float, default=0.05)
    ap.add_argument("--t0_gate_frac", type=float, default=0.60)
    ap.add_argument("--toy_anchor_frac", type=float, default=-1.0,
                    help=">= 0 : force anchor_frac dans le toy (G1 : 1.0 = tache de revisit pure)")
    ap.add_argument("--run_name", default="mem2_writer_A")
    ap.add_argument("--wandb_project", default="anchor-forcing-memory")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=1, help="videos per step (grad accumulated)")
    ap.add_argument("--K", type=int, default=3, help="supervised positions per video")
    ap.add_argument("--lr_w", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--wd_d", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--prox", type=float, default=1e-4, help="||theta_w - theta_l1||^2 (F4.1)")
    ap.add_argument("--n_bptt", type=int, default=None,
                    help="truncated BPTT depth in writes (None = full)")
    ap.add_argument("--subject_weight", type=float, default=6.0, help="v1.2 loss")
    ap.add_argument("--single_w", type=float, default=0.3, help="weight of single-mode videos")
    ap.add_argument("--ages", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--min_age", type=int, default=1)
    ap.add_argument("--anchor_frac", type=float, default=0.5)
    ap.add_argument("--lam_inst", type=float, default=0.3)
    ap.add_argument("--lam_ramp", type=int, nargs=2, default=[300, 600])
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--n_neg", type=int, default=0, help="0 = the whole split-side bank")
    ap.add_argument("--mask_chunks", type=int, default=2, help="F1.3 same-video mask radius")
    ap.add_argument("--cond_dropout", type=float, default=0.5)
    ap.add_argument("--crop_frac", type=float, default=0.6)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--n_val_switch", type=int, default=18)
    ap.add_argument("--n_val_single", type=int, default=4)
    ap.add_argument("--part_cache", type=int, default=2)
    ap.add_argument("--cache_videos", type=int, default=0, help="keep N videos' h in CPU RAM")
    ap.add_argument("--h_shuffle", action="store_true", help="toy control (c)")
    ap.add_argument("--fixed_positions", action="store_true",
                    help="draw the supervised positions ONCE per video (overfit gate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--probe_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    # modes
    ap.add_argument("--smoke", action="store_true", help="5 steps + all asserts + VRAM")
    ap.add_argument("--vram_probe", action="store_true")
    ap.add_argument("--acc_probe", action="store_true",
                    help="retrieval accuracy of the v1.2-decoded anchor per pooling factor")
    ap.add_argument("--probe_K", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--probe_bptt", type=int, nargs="+", default=[-1, 12, 4])
    ap.add_argument("--toy", action="store_true", help="gate F3.1")
    ap.add_argument("--toy_videos", type=int, default=8)
    ap.add_argument("--toy_variants", default="main,bptt1,hshuffle",
                    help="re-run a subset; the others are read back from toy_result.json")
    ap.add_argument("--toy_steps", type=int, default=200)
    ap.add_argument("--toy_min_age", type=int, default=10)
    ap.add_argument("--toy_inst", action="store_true", help="keep the InfoNCE term in the toy")
    ap.add_argument("--toy_conv_ratio", type=float, default=0.75)
    ap.add_argument("--toy_margin", type=float, default=0.10)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    assert os.environ.get("WANDB_MODE", "") != "disabled", \
        "house rule: WANDB_MODE=disabled is forbidden, every run logs live to wandb"
    os.makedirs(OUT_ROOT, exist_ok=True)
    import wandb
    run_name = args.run_name + ("_toy" if args.toy else ("_smoke" if args.smoke else ""))
    wandb.init(project=args.wandb_project, name=run_name, dir=OUT_ROOT, config=vars(args))

    if args.toy:
        args.ages = [args.toy_min_age, 20]
        args.min_age = args.toy_min_age
        args.cond_dropout = 1.0        # the state is the only route to the target in the toy
        args.cache_videos = max(args.cache_videos, args.toy_videos)
        args.steps = args.toy_steps
        if args.toy_anchor_frac >= 0:
            args.anchor_frac = args.toy_anchor_frac
        ok = run_toy(args, wandb)
        wandb.finish()
        sys.exit(0 if ok else 1)

    tr = WriterTrainer(args)
    if args.t0_probe:
        res, rows, ok_t0 = t0_probe(tr, n_videos=args.t0_probe_videos, k=args.t0_k)
        wandb.log(res, step=0)
        with open(os.path.join(OUT_ROOT, f"{args.run_name}_t0_probe.json"), "w") as f:
            json.dump({"summary": {k.split("/", 1)[-1]: v for k, v in res.items()},
                       "rows": rows, "gate_G2a": bool(ok_t0),
                       "thresholds": {"margin": args.t0_gate_margin,
                                      "frac": args.t0_gate_frac},
                       "space": args.t0_space, "lat_mode": args.t0_lat_mode,
                       "writer_ckpt": args.writer_ckpt, "dec_ckpt": args.dec_ckpt}, f, indent=2)
        wandb.finish()
        sys.exit(0 if ok_t0 else 2)          # le code de sortie EST le verdict du gate
    if args.acc_probe:
        res = acc_probe(tr)
        wandb.log({f"acc_probe/pool{p}_center{int(c)}": v[0] for (p, c), v in res.items()},
                  step=0)
    if args.dino_probe:
        res = dino_probe(tr, n_videos=args.dino_probe_videos)
        wandb.log(res, step=0)
    if args.vram_probe:
        rows = vram_probe(tr, lam_d=args.lam_dino)
        wandb.log({f"vram/K{K}_bptt{nb}": pk for K, nb, pk, _, _ in rows}, step=0)
    if args.acc_probe or args.vram_probe or args.dino_probe:
        wandb.finish()
        if args.dino_probe:      # exit code is the gate verdict
            sys.exit(0 if res["dino_probe/pos_gap"] > 0 else 2)
        return

    steps = 5 if args.smoke else args.steps
    t0 = time.time()
    for s in range(steps):
        tr.step = s + 1
        lam = args.lam_inst * min(1.0, max(0.0, (s - args.lam_ramp[0])
                                           / max(1, args.lam_ramp[1] - args.lam_ramp[0])))
        lam_d = args.lam_dino * min(1.0, max(0.0, (s - args.lam_dino_ramp[0])
                                             / max(1, args.lam_dino_ramp[1]
                                                   - args.lam_dino_ramp[0])))
        lam_t0 = args.lam_t0 * min(1.0, max(0.0, (s - args.lam_t0_ramp[0])
                                            / max(1, args.lam_t0_ramp[1] - args.lam_t0_ramp[0])))
        if args.dino_gate and lam_d > 0 and tr.last_dino_gap is not None \
                and tr.last_dino_gap <= 0:
            lam_d = 0.0              # gate: never train on a wrong-signed instance signal
        batch = [tr.train_vids[(s * args.bs + i) % len(tr.train_vids)] for i in range(args.bs)]
        logs = tr.train_step(batch, lam, lam_d=lam_d, lam_t0=lam_t0)
        logs.update(tr.writer_diag())
        if s == 0:                       # F3.3: the loss must reach the writer
            if tr.freeze_writer:
                print(f"[writer][F3.3] SKIPPED: the writer is FROZEN this run "
                      f"(delta_ref {logs['writer/delta_ref']:.3e}, only the decoder learns)",
                      flush=True)
            else:
                g = tr.masters[(tr.blocks[0], "qkv.weight")].grad
                assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, \
                    "F3.3 VIOLATION: qkv.weight got NO gradient after the first backward"
                print(f"[writer][F3.3] qkv.weight grad |sum| = {float(g.abs().sum()):.3e} "
                      f"(non-zero, finite), the loss reaches the writer", flush=True)
            dg = [p for n_, p in tr.decoder.named_parameters() if n_.startswith("mixer.")
                  and p.grad is not None and float(p.grad.abs().sum()) > 0]
            if args.mixer:
                assert dg, "MIXER VIOLATION: no mixer parameter received a gradient at step 1"
                print(f"[writer] mixer: {len(dg)} tensors with a non-zero gradient at step 1",
                      flush=True)
            print(f"[writer][F5.4] peak VRAM of the full backward: {tr.peak_vram:.1f} GB "
                  f"(K={args.K}, bs={args.bs}, n_bptt={args.n_bptt}, "
                  f"lam_dino={args.lam_dino}, lam_t0={args.lam_t0})", flush=True)
            if tr.t0bank is not None:
                # MEMOIRE 2.1 : la quantite des gates, sur les videos val, AVANT tout pas utile
                # (reference publiee, 24 cellules S2 : A +0.042 / B +0.298 => marge -0.256).
                b0, _ = tr.t0_eval(tr.val_vids[:args.t0_probe_videos], "val", k=args.t0_k,
                                   tag="")
                print(f"[writer][T0] BASELINE step 0 (val held-out) : marge {b0['margin']:+.4f} "
                      f"(A {b0['corr_a']:+.4f} | B {b0['corr_b']:+.4f} | "
                      f"autres {b0['corr_o']:+.4f}), {int(b0['n_pos'])}/{int(b0['n'])} "
                      f"videos positives, std {b0['anchor_std']:.4f}", flush=True)
                wandb.log({f"t0_base/{k_}": v for k_, v in b0.items()}, step=1)
        if (s + 1) % args.log_every == 0 or s == 0 or args.smoke:
            wandb.log(logs, step=s + 1)
            print(f"[writer] step {s + 1}/{steps} loss {logs['train/loss']:.4f} "
                  f"rec {logs['train/rec']:.4f} acc {logs['train/acc']:.3f} "
                  f"margin {logs.get('ctrl/causal_margin', float('nan')):.3f} "
                  f"vram {tr.peak_vram:.1f} GB ({(time.time() - t0) / (s + 1):.1f} s/step)",
                  flush=True)
        if (s + 1) % args.val_every == 0 or (args.smoke and s == steps - 1):
            v = tr.evaluate(n_videos=(2 if args.smoke else 8))
            wandb.log(v, step=s + 1)
            print(f"[writer] step {s + 1} " + " ".join(f"{k}={x:.4f}" for k, x in v.items()
                                                       if isinstance(x, float)), flush=True)
        if (s + 1) % args.probe_every == 0 or (args.smoke and s == steps - 1):
            ref, dec = tr.influence_probe(tr.val_vids[0])
            for k, (rel, dl) in dec.items():
                wandb.log({f"mem/influence_decay@{k}": rel, f"mem/dloss@{k}": dl}, step=s + 1)
            print(f"[writer] influence_decay (loss_ref {ref:.4f}): "
                  + " ".join(f"k={k}:{rel:.3f}/{dl:+.3f}" for k, (rel, dl) in dec.items()),
                  flush=True)
        if (s + 1) % args.ckpt_every == 0:
            tr.save(os.path.join(OUT_ROOT, f"{args.run_name}_step{s + 1}.pt"))
    if tr.freeze_writer:                 # phase-B contract, end of run
        d = delta_l1(tr.cells, tr.ref_start)
        print(f"[writer] PHASE-B FREEZE CHECK at step {steps}: writer/delta_ref = {d:.3e}",
              flush=True)
        assert d == 0.0, f"PHASE-B VIOLATION: the frozen writer moved (delta_ref={d:.3e})"
    tr.save(os.path.join(OUT_ROOT, f"{args.run_name}_final.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()
