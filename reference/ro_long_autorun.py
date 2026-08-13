#!/usr/bin/env python
"""Autonomous read-only LONG-ROLLOUT launcher (1 GPU).

Tests the horizon hypothesis: same read-only config as e1_ro_5L_metrics (5 layers, gate_extra=[],
no write) but streaming_max_length 60 -> 120, so the memory state is trained on a longer horizon.

Waits until the extended 10-video eval is done (/tmp/eval10_result.txt) AND 1 sane GPU is free
(prefers GPU 7), then launches on that single GPU. Watches the first steps: success = a training
step is reached; failure = OOM/crash -> kills and reports. Status -> /tmp/ro_long_result.txt.
"""
import os, time, subprocess, signal

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
CFG = os.path.join(REPO, "reference/ro_long_rollout.yaml")
TRIGGER = "/tmp/eval10_result.txt"          # written by eval_extend when the 10-video eval finishes
RESULT = "/tmp/ro_long_result.txt"
TRAIN_LOG = "/tmp/ro_long_train.log"
PREFER = "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107"   # GPU 7
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"
FREE_MEM, FREE_UTIL, FREE_SECS, POLL = 2000, 15, 60, 15
START_DEADLINE = 2400   # 40 min for model load + first training step


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_uuids():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    s = set()
    for line in out.strip().splitlines():
        try:
            u, m, ut = [c.strip() for c in line.split(",")]
            if u != BROKEN and int(m) < FREE_MEM and int(ut) < FREE_UTIL:
                s.add(u)
        except ValueError:
            pass
    return s


def wait_one_free():
    seen = {}
    while True:
        now = time.time()
        fu = free_uuids()
        for u in list(seen):
            if u not in fu:
                del seen[u]
        for u in fu:
            seen.setdefault(u, now)
        stable = [u for u, t in seen.items() if now - t >= FREE_SECS]
        if stable:
            # prefer GPU 7 if it is among the stable-free ones
            return PREFER if PREFER in stable else sorted(stable, key=lambda u: seen[u])[0]
        log(f"waiting for 1 free GPU; stable={len(stable)} free={len(fu)}")
        time.sleep(POLL)


def launch(gpu):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", "--max-restarts=0",
           "--master_port=29962", "train.py", "--config_path", CFG,
           "--no-auto-resume", "--no-one-logger"]
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(TRAIN_LOG, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def watch(proc):
    deadline = time.time() + START_DEADLINE
    while time.time() < deadline:
        txt = ""
        try:
            txt = open(TRAIN_LOG).read()
        except Exception:
            pass
        low = txt.lower()
        if "out of memory" in low:
            _kill(proc); return "OOM"
        if "mem stats step=" in txt:
            return "running"
        if proc.poll() is not None:
            return "OOM" if "out of memory" in low else "crash"
        time.sleep(20)
    _kill(proc); return "timeout"


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def main():
    res = []
    log("ro-long launcher up; waiting for the 10-video eval to finish (eval10_result.txt)...")
    while not os.path.exists(TRIGGER):
        time.sleep(60)
    res.append("10-video eval finished")
    gpu = wait_one_free()
    res.append(f"GPU free: {gpu[:12]}")
    log(f"launching read-only long-rollout (max_length=120, 1 GPU) on {gpu[:12]}")
    proc = launch(gpu)
    status = watch(proc)
    if status == "running":
        res.append(f"RO-LONG RUNNING pid={proc.pid} (read-only, max_length=120, gpu={gpu[:12]})")
    else:
        res.append(f"RO-LONG FAILED status={status} -- see {TRAIN_LOG} (likely VRAM on 1 GPU; try 2 GPUs)")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
