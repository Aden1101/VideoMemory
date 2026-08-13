"""VRAM profiler for read-only vs read+write memory training at 1/2/3 layers.

Runs the REAL training step (train.py) for a few iterations per config, in an isolated
subprocess (fresh process = clean memory baseline), and harvests the peak VRAM printed by
the trainer's profile hook ([VRAM_PROFILE] line, gated by config.profile_vram).

For each (MEMORY_LAYERS, train_write) it reports peak allocated / reserved, the write tax
(read+write minus read-only) and whether it fits on one 48GB GPU. OOM is a valid result.

Usage (after freeing a GPU):
    PROFILE_GPU=1 <longlive>/python profiling/profile_vram.py
Knobs (env):
    PROFILE_GPU       physical GPU index to pin            (default 1)
    PROFILE_MAX_LEN   streaming_max_length in LATENT frames (default 21 ~= 5s; secs=(1+4*(L-1))/16)
    PROFILE_LAYERS    comma-list of layer-sets, ';'-sep    (default "5;5,10;5,10,15")
"""
import os
import re
import sys
import subprocess
import tempfile
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(REPO, "configs", "ttt_memory.yaml")
GPU_TOTAL_GIB = 48.0  # RTX 6000 Ada

def latent_to_seconds(L):
    return (1 + 4 * (L - 1)) / 16.0  # Wan VAE: pixel=1+4*(L-1), export 16 fps

GPU = os.environ.get("PROFILE_GPU", "1")
# dmd_switch needs a prompt switch inside the clip (>= 2 chunks), so the minimum is 42 latent
# (~10s, switch at frame 21 = the 5s mark). A single 21-frame chunk (5s) cannot host the switch.
MAX_LEN = int(os.environ.get("PROFILE_MAX_LEN", "42"))  # 42 latent ~= 10.3s (2 chunks)
SWITCH = 21 if MAX_LEN > 21 else ((MAX_LEN - 1) // 3) * 3  # prompt-switch frame, < MAX_LEN, block-aligned
LAYER_SETS = os.environ.get("PROFILE_LAYERS", "5;5,10;5,10,15").split(";")
WRITE_MODES = [False, True]

# overrides applied on top of ttt_memory.yaml for a short, save-free profiling run
OVERRIDES = dict(
    max_iters=12,
    log_iters=100000,         # do not checkpoint within the run
    use_wandb=False,
    profile_vram=3,           # measure peak over 3 steps after warmup
    profile_vram_warmup=4,    # ignore inductor-compile cost of the first steps
    streaming_max_length=MAX_LEN,
    switch_choices=[SWITCH],
    switch_mode="random_choice",
    logdir="/tmp/vram_profile",
)


def run_one(layers, train_write, port):
    cfg = OmegaConf.load(BASE_CONFIG)
    for k, v in OVERRIDES.items():
        cfg[k] = v
    cfg.train_write = train_write
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.path.join(REPO, "configs"))
    OmegaConf.save(cfg, tmp.name)
    tmp.close()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["MAG_MEMORY_LAYERS"] = layers
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_MODE"] = "disabled"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "train.py", "--config_path", os.path.relpath(tmp.name, REPO),
           "--no-auto-resume", "--no-one-logger"]
    try:
        out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=2400)
        text = out.stdout + "\n" + out.stderr
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or "") + "\n" + (e.stderr or "") + "\nTIMEOUT"
    finally:
        os.unlink(tmp.name)

    m = re.search(r"\[VRAM_PROFILE\].*?alloc_GiB=([\d.]+)\s+reserved_GiB=([\d.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2)), "ok"
    if re.search(r"out of memory|CUDA out of memory", text, re.IGNORECASE):
        return None, None, "OOM"
    if "TIMEOUT" in text:
        return None, None, "timeout"
    # surface the tail so failures are debuggable
    sys.stderr.write(f"\n[profile_vram] no result for layers={layers} write={train_write}; tail:\n")
    sys.stderr.write("\n".join(text.strip().splitlines()[-15:]) + "\n")
    return None, None, "fail"


def main():
    results = {}
    port = 29700
    for layers in LAYER_SETS:
        for write in WRITE_MODES:
            n = len(layers.split(","))
            print(f"==> profiling layers={layers} ({n}L) train_write={write} on GPU {GPU} ...", flush=True)
            a, r, status = run_one(layers, write, port)
            port += 1
            results[(layers, write)] = (a, r, status)
            print(f"    {status}: alloc={a} GB  reserved={r} GB", flush=True)

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, float) else "  -- "

    print("\n" + "=" * 78)
    print(f"VRAM profile (1 GPU = {GPU_TOTAL_GIB:.0f} GiB, clip max_len={MAX_LEN} latent ~= {latent_to_seconds(MAX_LEN):.1f}s, switch@{SWITCH}, peak over {OVERRIDES['profile_vram']} steps)")
    print("=" * 78)
    print(f"{'layers':10s} {'read-only alloc/res':>22s} {'read+write alloc/res':>22s} {'write tax':>10s}")
    for layers in LAYER_SETS:
        ro_a, ro_r, ro_s = results[(layers, False)]
        rw_a, rw_r, rw_s = results[(layers, True)]
        ro = f"{fmt(ro_a)}/{fmt(ro_r)} {ro_s}" if ro_s != "ok" else f"{fmt(ro_a)}/{fmt(ro_r)}"
        rw = f"{fmt(rw_a)}/{fmt(rw_r)} {rw_s}" if rw_s != "ok" else f"{fmt(rw_a)}/{fmt(rw_r)}"
        tax = f"{(rw_a - ro_a):+.2f}" if (isinstance(ro_a, float) and isinstance(rw_a, float)) else "  -- "
        print(f"{layers:10s} {ro:>22s} {rw:>22s} {tax:>10s}")

    # per-layer slope of the write tax (informative if at least two layer-sets have both numbers)
    pts = []
    for layers in LAYER_SETS:
        ro_a, _, ro_s = results[(layers, False)]
        rw_a, _, rw_s = results[(layers, True)]
        if ro_s == "ok" and rw_s == "ok":
            pts.append((len(layers.split(",")), rw_a - ro_a))
    if len(pts) >= 2:
        (n0, t0), (n1, t1) = pts[0], pts[-1]
        slope = (t1 - t0) / (n1 - n0) if n1 != n0 else float("nan")
        print(f"\nwrite tax ~ {slope:+.2f} GB per added memory layer")
    print("=" * 78)


if __name__ == "__main__":
    main()
