#!/usr/bin/env python
"""Fair-play watcher + read-only 5-layer VRAM feasibility test.

Waits until >=2 healthy GPUs are free for >5 min, then runs the REAL read-only training
config (configs/ttt_memory.yaml, train_write absent = read-only) with 5 memory layers
{5,11,17,23,29}, in VRAM-profile mode, to check whether it fits at the heaviest step.

Addresses GPUs by UUID. Writes /tmp/ro5_result.txt, logs /tmp/ro5_watch.log.
"""
import os, re, subprocess, time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BASE_CFG = os.path.join(REPO, "configs/ttt_memory.yaml")
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"   # GPU 6, needs reset
LAYERS = "5,11,17,23,29"
FREE_MEM_MIB = 1000
FREE_UTIL = 10
FREE_SECS = 300
POLL = 30
RESULTS = "/tmp/ro5_result.txt"


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
            rows.append((uuid, 10**9, 100))
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
            log(f"claiming {ready[0][:20]} + {ready[1][:20]}")
            return ready[0], ready[1]
        held = {u[:20]: int(now - t) for u, t in free_since.items()}
        log(f"waiting; free-durations(s)={held}")
        time.sleep(POLL)


def make_cfg():
    txt = open(BASE_CFG).read()
    txt = re.sub(r"max_iters:\s*\d+", "max_iters: 12", txt)
    txt = re.sub(r"log_iters:\s*\d+", "log_iters: 100000", txt)
    txt = re.sub(r"use_wandb:\s*\w+", "use_wandb: false", txt)
    txt = re.sub(r"logdir:\s*\S+", "logdir: /tmp/ro5_profile", txt)
    txt += "\nprofile_vram: 3\nprofile_vram_warmup: 4\n"
    p = "/tmp/_ro5_profile.yaml"
    open(p, "w").write(txt)
    return p


def main():
    log(f"RO-5-layer feasibility watcher up; layers={LAYERS}; window=60 (real train cfg)")
    ua, ub = wait_for_two()
    fu = free_uuids()
    if ua not in fu or ub not in fu:
        log("a claimed GPU was taken; re-waiting")
        ua, ub = wait_for_two()
    cfg = make_cfg()
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=f"{ua},{ub}",
        NCCL_P2P_DISABLE="1",
        MAG_MEMORY_LAYERS=LAYERS,
        MAG_GATE_INIT="0.1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        WANDB_MODE="disabled",
    )
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2",
           "--max-restarts=0", "--master_port=29940", "train.py",
           "--config_path", cfg, "--no-auto-resume", "--no-one-logger"]
    log(f"launching read-only 5-layer profile: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        log("TIMEOUT")
    m = re.search(r"\[VRAM_PROFILE\][^\n]*", out)
    if m:
        status, detail = "PASS", m.group(0)
    elif "OutOfMemoryError" in out or "out of memory" in out:
        dm = re.search(r"Tried to allocate [^.]*\.[^.]*free", out)
        status, detail = "OOM", (dm.group(0) if dm else "OOM")
    else:
        status, detail = "ERROR", "\n".join(out.strip().splitlines()[-5:])
    log(f"RESULT: {status} | {detail}")
    with open(RESULTS, "w") as f:
        f.write(f"read-only, layers={LAYERS}, window=60\n{status}  {detail}\n")
    print(open(RESULTS).read(), flush=True)


if __name__ == "__main__":
    main()
