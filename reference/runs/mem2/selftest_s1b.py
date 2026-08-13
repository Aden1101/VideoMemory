#!/usr/bin/env python
# ============================================================================================
# SELFTEST S1b -- "MEM_ZERO_ALL is a MULTIPLICATION, not a skip" (memory 2.0 gate F5.2 of
# reference/memory2_redteam.md, DIE-ON list of the RESOLUTION DU DEBAT).
#
# The founding premise of the whole write-replay chantier (design 1.1) is: under
# MEM_ZERO_ALL=1 the eviction-feed h are DATA -- pure functions of the frozen backbone,
# INDEPENDENT of the writer parameters. S1 (selftest_replay.py, green) proved
# "replay == in-situ with the SAME writer"; it did NOT prove the invariance premise.
# The premise has one documented failure mode (causal_model.py L653-658):
#       mem_out = mem_out * 0.0        # a MULTIPLICATION, not a structural skip
#       x = x + mem_out
# so a single non-finite value in the READ branch (state saturation -> bf16 overflow in
# swiglu_apply, exactly what a re-scaled 2.0 writer could cause) gives 0 * NaN = NaN, which
# contaminates x, the latents, AND h -- and h would then depend on theta_w after all.
#
# This selftest does the two halves the red-team asked for:
#   PART 1 (GPU, ~3 min): prove the DANGER IS REAL. Two identical short rollouts under
#     MEM_ZERO_ALL=1 through collect_pairs.rollout_collect(v3_capture=True), the second with an
#     artificial NaN injected into the READ output of the block-5 memory cell (monkeypatched
#     _project_output, restored afterwards; nothing in the repo is modified). norm1 forward
#     hooks record the finiteness of the h of every memory block on every pass.
#       clean run  : every h finite, latents finite            (control)
#       poisoned   : h of the DOWNSTREAM memory blocks non-finite + latents non-finite
#                    => the zeroed injection DOES propagate. (The collector's own fp16-range
#                    assert also fires on such a run: reported as defense-in-depth.)
#   PART 2 (I/O, ~5-15 min): prove IT DID NOT HAPPEN on the collected campaign: scan EVERY h
#     tensor of the 18 v3 parts (+ the 4 twin parts) for NaN/Inf, per memory block, plus the
#     latents; report absmax / RMS per block.
#
# VERDICT: PASS only if part 1 shows the contamination path is real AND part 2 finds zero
# non-finite value in the shards actually on disk (the h used to train the writer).
#
# Usage (GPU pre-check in the SAME command, house rule):
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && CUDA_DEVICE_ORDER=PCI_BUS_ID \
#   nvidia-smi --query-gpu=index,memory.used --format=csv && CUDA_VISIBLE_DEVICES=1 \
#   python reference/runs/mem2/selftest_s1b.py
#   python reference/runs/mem2/selftest_s1b.py --scan-only          # part 2 alone (no GPU model)
#   python reference/runs/mem2/selftest_s1b.py --no-scan            # part 1 alone
# Exit 0 = PASS, 1 = FAIL.
# ============================================================================================
import argparse
import os
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
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402

PROMPT = "a brown bear with one white ear fishing in a shallow mountain river"
_FAILS = []


def check(name, ok, detail=""):
    print(f"[s1b][{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


# ----------------------------- part 1: the contamination path --------------------------------
class NormHooks:
    """Forward hooks on blocks[bi].norm1: its output IS the h the eviction-feed stores
    (causal_model.py L586 `h = self.norm1(x_in)`), so this observes exactly the tensor at
    stake without touching any core code."""

    def __init__(self, cm, blocks):
        self.stats = {bi: {"calls": 0, "nonfinite_calls": 0} for bi in blocks}
        self.handles = []
        for bi in blocks:
            self.handles.append(cm.blocks[bi].norm1.register_forward_hook(self._mk(bi)))

    def _mk(self, bi):
        def hook(mod, inp, out):
            s = self.stats[bi]
            s["calls"] += 1
            if not torch.isfinite(out.detach()).all():
                s["nonfinite_calls"] += 1
        return hook

    def reset(self):
        for s in self.stats.values():
            s["calls"] = 0
            s["nonfinite_calls"] = 0

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def part1(args):
    from collect_pairs import rollout_collect, _find_causal_model, DEFAULT_CONFIG
    from run_reid_probe import build_pipeline

    assert torch.cuda.is_available(), "part 1 needs a GPU (`import wan` is CUDA-bound)"
    device = torch.device("cuda")
    config = OmegaConf.load(args.config or DEFAULT_CONFIG)
    config.distributed = False
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    F = int(config.num_frame_per_block)
    assert args.frames % F == 0

    print(f"[s1b] building pipeline ({args.config or DEFAULT_CONFIG}, l1@2000-EMA)", flush=True)
    pipeline = build_pipeline(config, device)
    cm = _find_causal_model(pipeline.generator.model)
    assert cm is not None, "CausalWanModel not found under the peft wrappers"
    hooks = NormHooks(cm, mem_layers)

    # ---- clean control rollout -------------------------------------------------------------
    print(f"[s1b] clean rollout ({args.frames}f, MEM_ZERO_ALL=1)", flush=True)
    res = rollout_collect(pipeline, config, PROMPT, 0, args.frames, 0, device,
                          v3_capture=True, v3_fp16_roundtrip=False)
    clean_nf = {bi: hooks.stats[bi]["nonfinite_calls"] for bi in mem_layers}
    clean_calls = {bi: hooks.stats[bi]["calls"] for bi in mem_layers}
    h_finite = all(torch.isfinite(t).all().item()
                   for bi in mem_layers for t in res["h_seq"][bi])
    check("part1 control: clean rollout h all finite",
          h_finite and sum(clean_nf.values()) == 0,
          f"norm1 calls {clean_calls}, non-finite {clean_nf}, "
          f"{len(res['write_chunks'])} writes x {len(mem_layers)} blocks")
    check("part1 control: clean rollout latents finite",
          bool(torch.isfinite(res["latents"]).all()))
    lat_clean = res["latents"].clone()

    # ---- poisoned rollout: ONE NaN in the read branch of the first memory block -------------
    poisoned_block = mem_layers[0]
    cell = cm.blocks[poisoned_block].memory
    orig_proj = cell._project_output

    def poisoned_project(x, batch_size, **kw):
        out = orig_proj(x, batch_size, **kw)
        out = out.clone()
        out[..., 0, 0] = float("nan")     # ONE element: the minimal realistic overflow
        return out

    cell._project_output = poisoned_project
    hooks.reset()
    print(f"[s1b] poisoned rollout: NaN injected in block {poisoned_block} read output "
          f"(mem_out), still MEM_ZERO_ALL=1", flush=True)
    collector_assert = None
    res2 = None
    try:
        res2 = rollout_collect(pipeline, config, PROMPT, 0, args.frames, 0, device,
                               v3_capture=True, v3_fp16_roundtrip=False)
    except AssertionError as e:
        collector_assert = str(e)[:160]
    finally:
        cell._project_output = orig_proj
    pois_nf = {bi: hooks.stats[bi]["nonfinite_calls"] for bi in mem_layers}
    downstream = [bi for bi in mem_layers if bi > poisoned_block]
    contaminated = any(pois_nf[bi] > 0 for bi in downstream)
    check("part1 DANGER IS REAL: a zeroed non-finite read contaminates the h of the "
          "downstream memory blocks",
          contaminated,
          f"non-finite norm1 outputs per block {pois_nf}"
          + (f"; collector range-assert also fired: {collector_assert}"
             if collector_assert else ""))
    if res2 is not None:
        lat_nf = not bool(torch.isfinite(res2["latents"]).all())
        check("part1 DANGER IS REAL: latents contaminated too (0 * NaN = NaN reaches x)",
              lat_nf,
              "latents non-finite" if lat_nf else
              "latents STAYED finite -- re-read causal_model L653-658 before trusting this")
    else:
        check("part1 collector aborts on a NaN h (defense in depth)",
              collector_assert is not None, collector_assert or "")
    # the collector would ALSO have crashed, but the point stands: nothing structurally
    # prevents a contaminated h from being written to a shard if the assert is ever relaxed.
    hooks.remove()

    # restore-check: the patch is gone, a third rollout is clean again
    print("[s1b] restored rollout (patch removed)", flush=True)
    res3 = rollout_collect(pipeline, config, PROMPT, 0, args.frames, 0, device,
                           v3_capture=True, v3_fp16_roundtrip=False)
    check("part1 restore: latents byte-identical to the clean control "
          "(the monkeypatch left no trace)",
          torch.equal(lat_clean, res3["latents"]))
    del pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------- part 2: the shards actually on disk ---------------------------
def part2(args):
    device = torch.device("cuda") if (torch.cuda.is_available() and not args.cpu_scan) \
        else torch.device("cpu")
    parts = D.list_parts((D.V3_DIR, D.TWIN_DIR) if not args.no_twins else (D.V3_DIR,))
    if args.max_parts:
        parts = parts[:args.max_parts]
    print(f"[s1b] scanning {len(parts)} parts on {device} "
          f"({sum(os.path.getsize(p) for p in parts) / 1e9:.0f} GB)", flush=True)
    stats = {}
    lat_bad = 0
    n_videos = 0
    t0 = time.time()
    for pi, p in enumerate(parts):
        obj = D.load_part(p)
        for rec in obj["videos"]:
            n_videos += 1
            lat = rec["latents"].to(device)
            if not torch.isfinite(lat).all().item():
                lat_bad += 1
            del lat
            for b, seq in rec["h_seq"].items():
                s = stats.setdefault(int(b), {"tensors": 0, "elems": 0, "nonfinite": 0,
                                              "absmax": 0.0, "sqsum": 0.0})
                for t in seq:
                    x = t.to(device).float()
                    nf = int((~torch.isfinite(x)).sum().item())
                    s["tensors"] += 1
                    s["elems"] += x.numel()
                    s["nonfinite"] += nf
                    if nf == 0:
                        s["absmax"] = max(s["absmax"], float(x.abs().max().item()))
                        s["sqsum"] += float(x.pow(2).sum().item())
                    del x
        del obj
        print(f"[s1b]   part {pi + 1}/{len(parts)} done ({time.time() - t0:.0f}s, "
              f"{n_videos} videos)", flush=True)
    total_nf = sum(s["nonfinite"] for s in stats.values())
    for b in sorted(stats):
        s = stats[b]
        rms = (s["sqsum"] / max(1, s["elems"])) ** 0.5
        print(f"[s1b]   block {b:>2}: {s['tensors']:>5} h tensors, "
              f"{s['elems'] / 1e9:.1f}G elems, non-finite {s['nonfinite']}, "
              f"absmax {s['absmax']:.2f}, rms {rms:.4f}", flush=True)
    check("part2: ZERO non-finite h in the collected v3 shards", total_nf == 0,
          f"{n_videos} videos, {sum(s['tensors'] for s in stats.values())} h tensors, "
          f"{sum(s['elems'] for s in stats.values()) / 1e9:.1f}G elements scanned")
    check("part2: ZERO non-finite latents", lat_bad == 0, f"{lat_bad} bad videos")
    return stats


def main():
    ap = argparse.ArgumentParser(description="S1b: MEM_ZERO_ALL contamination gate (F5.2)")
    ap.add_argument("--config", default="")
    ap.add_argument("--frames", type=int, default=24, help="short rollout (>= (wc+sc+1)*F)")
    ap.add_argument("--scan-only", action="store_true", help="part 2 only (no model build)")
    ap.add_argument("--no-scan", action="store_true", help="part 1 only")
    ap.add_argument("--no-twins", action="store_true")
    ap.add_argument("--max-parts", type=int, default=0)
    ap.add_argument("--cpu-scan", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.scan_only:
        part1(args)
    if not args.no_scan:
        part2(args)
    print(f"\n[s1b] {'ALL GREEN -- gate F5.2 OPEN' if not _FAILS else f'{len(_FAILS)} FAILURE(S): {_FAILS}'}"
          f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    sys.exit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
