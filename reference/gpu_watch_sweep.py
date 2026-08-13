#!/usr/bin/env python
"""Fair-play GPU watcher + read+write clip sweep.

Waits until >=2 healthy GPUs are free for >5 min, then sweeps streaming_max_length
in [24,27,30,33] on 2 GPUs (FSDP + checkpoint-write + expandable_segments) to find the
LARGEST clip whose read+write step survives (prints [VRAM_PROFILE]). Stops at first OOM.

Addresses GPUs by UUID (CUDA_DEVICE_ORDER reorders indices; GPU 6 is broken).
Logs to /tmp/vram_sweep.log and writes a summary to /tmp/vram_sweep_results.txt.
"""
import os, re, subprocess, sys, time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BASE_CFG = os.path.join(REPO, "configs/_t2g27.yaml")
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"   # GPU 6, needs reset
CLIPS = [24, 27, 30, 33]                               # streaming_max_length sweep (switch=21)
FREE_MEM_MIB = 1000
FREE_UTIL = 10
FREE_SECS = 300                                        # 5 min fair-play
POLL = 30
RESULTS = "/tmp/vram_sweep_results.txt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        uuid, used, util = [c.strip() for c in line.split(",")]
        try:
            rows.append((uuid, int(used), int(util)))
        except ValueError:
            rows.append((uuid, 10**9, 100))   # util "[N/A]" -> treat busy
    return rows


def free_uuids():
    return {u for (u, m, ut) in gpus()
            if u != BROKEN and m < FREE_MEM_MIB and ut < FREE_UTIL}


def wait_for_two():
    free_since = {}
    while True:
        now = time.time()
        fu = free_uuids()
        for u in list(free_since):
            if u not in fu:
                del free_since[u]
        for u in fu:
            free_since.setdefault(u, now)
        ready = sorted([u for u, t in free_since.items() if now - t >= FREE_SECS],
                       key=lambda u: free_since[u])
        if len(ready) >= 2:
            log(f"claiming {ready[0][:20]} + {ready[1][:20]} (free >{FREE_SECS}s)")
            return ready[0], ready[1]
        held = {u: int(now - t) for u, t in free_since.items()}
        log(f"waiting; free-durations(s)={held}")
        time.sleep(POLL)


def make_cfg(clip):
    txt = open(BASE_CFG).read()
    txt = re.sub(r"streaming_max_length:\s*\d+", f"streaming_max_length: {clip}", txt)
    txt = re.sub(r"streaming_min_new_frame:\s*\d+", f"streaming_min_new_frame: {clip-21}", txt)
    txt = re.sub(r"logdir:\s*\S+", f"logdir: /tmp/vram2gpu_s{clip}", txt)
    p = f"/tmp/_sweep_s{clip}.yaml"
    open(p, "w").write(txt)
    return p


def run(clip, ua, ub, port):
    cfg = make_cfg(clip)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=f"{ua},{ub}",
        NCCL_P2P_DISABLE="1",
        MAG_MEMORY_LAYERS="5",
        MAG_CKPT_WRITE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        WANDB_MODE="disabled",
    )
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2",
           "--max-restarts=0", f"--master_port={port}", "train.py",
           "--config_path", cfg, "--no-auto-resume", "--no-one-logger"]
    log(f"clip={clip}: launching ({' '.join(cmd)})")
    try:
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=1200)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        log(f"clip={clip}: TIMEOUT")
    m = re.search(r"\[VRAM_PROFILE\][^\n]*", out)
    if m:
        return "PASS", m.group(0)
    if "OutOfMemoryError" in out or "out of memory" in out:
        dm = re.search(r"Tried to allocate [^.]*\.[^.]*free", out)
        return "OOM", (dm.group(0) if dm else "OOM")
    # neither -> likely the GPU was stolen / init failure
    tail = "\n".join(out.strip().splitlines()[-3:])
    return "ERROR", tail


def main():
    log(f"watcher up; sweep clips={CLIPS}; excluding broken {BROKEN[:20]}")
    ua, ub = wait_for_two()
    results = []
    port = 29930
    for clip in CLIPS:
        # re-check both still free right before launch
        fu = free_uuids()
        if ua not in fu or ub not in fu:
            log("a claimed GPU was taken; re-waiting")
            ua, ub = wait_for_two()
        status, detail = run(clip, ua, ub, port)
        port += 1
        log(f"clip={clip}: {status} | {detail}")
        results.append((clip, status, detail))
        with open(RESULTS, "w") as f:
            f.write("clip(latent)  status  detail\n")
            for c, s, d in results:
                f.write(f"{c:>4}          {s:<6}  {d}\n")
            passes = [c for c, s, _ in results if s == "PASS"]
            if passes:
                f.write(f"\nMAX clip that fits read+write on 2 GPUs: {max(passes)} latent "
                        f"(~{1+4*(max(passes)-1)}/16 = {(1+4*(max(passes)-1))/16:.1f}s window)\n")
        if status == "OOM":
            log(f"clip={clip} OOM -> max is the previous PASS; stopping sweep")
            break
        if status == "ERROR":
            log(f"clip={clip} ERROR (likely GPU contention); retrying after wait")
            ua, ub = wait_for_two()
    log(f"sweep done; results in {RESULTS}")
    print(open(RESULTS).read(), flush=True)


if __name__ == "__main__":
    main()
