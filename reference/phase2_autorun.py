#!/usr/bin/env python
"""Autonomous Phase-2 write-training launcher (1 memory layer).

Chain: combo (gating perso) finishes -> combo eval -> THEN this write-training.
Waits until the COMBO eval is done (/tmp/combo_eval_result.txt exists) AND 2 sane GPUs are
free (60s stable, broken GPU excluded), then launches write-training (train_write=True, 1 layer,
the VRAM-proven config) on them. Watches the first steps: success = a training step is reached;
failure = OOM or crash -> kills and reports. Status -> /tmp/phase2_result.txt.
"""
import os, time, subprocess, signal

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
CFG = os.path.join(REPO, "reference/phase2_write.yaml")
EVAL_DONE = "/tmp/combo_eval_result.txt"   # written by combo_eval_pipeline when the combo A/B finishes
RESULT = "/tmp/phase2_result.txt"
TRAIN_LOG = "/tmp/phase2_train.log"
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


def wait_two_free():
    seen = {}
    while True:
        now = time.time()
        fu = free_uuids()
        for u in list(seen):
            if u not in fu:
                del seen[u]
        for u in fu:
            seen.setdefault(u, now)
        stable = sorted([u for u, t in seen.items() if now - t >= FREE_SECS], key=lambda u: seen[u])
        if len(stable) >= 2 and free_uuids() >= {stable[0], stable[1]}:
            return stable[0], stable[1]
        log(f"waiting for 2 free GPUs; stable={len(stable)} free={len(fu)}")
        time.sleep(POLL)


def launch(g1, g2):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=f"{g1},{g2}", NCCL_P2P_DISABLE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--max-restarts=0",
           "--master_port=29958", "train.py", "--config_path", CFG,
           "--no-auto-resume", "--no-one-logger"]
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(TRAIN_LOG, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def watch(proc):
    """Return ('running'|'OOM'|'crash'|'timeout'). Kills proc on failure."""
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
    log("phase-2 launcher up; waiting for the COMBO eval to finish (combo_eval_result.txt)...")
    while not os.path.exists(EVAL_DONE):
        time.sleep(60)
    res.append("read-only eval finished")
    log("eval done; waiting for 2 free GPUs...")
    g1, g2 = wait_two_free()
    res.append(f"2 GPUs free: {g1[:12]},{g2[:12]}")
    log(f"launching phase-2 write (1 layer) on {g1[:12]},{g2[:12]}")
    proc = launch(g1, g2)
    status = watch(proc)
    if status == "running":
        res.append(f"PHASE-2 WRITE RUNNING pid={proc.pid} (1 layer, train_write, gpus={g1[:12]},{g2[:12]})")
    else:
        res.append(f"PHASE-2 FAILED status={status} -- see {TRAIN_LOG} (likely VRAM or config; fix by hand)")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
