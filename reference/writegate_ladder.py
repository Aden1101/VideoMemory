#!/usr/bin/env python
"""Find the max #memory-layers that fit write+gate training on 3 GPUs, then launch the real run.

Tests memory_layers = [5], [5,17], [5,17,29] in order on GPU 3+4+7. Each: short probe run,
watch until step>=6 (PASS = fits VRAM) or OOM/crash. Kill after each PASS, wait for GPUs to free,
try next. On first OOM/crash, STOP and take the last config that PASSED. Then launch the REAL run
at that max (max_iters=3000, train_write+gate 6+5+2, checkpoints every 150).
Result -> /tmp/writegate_ladder_result.txt ; real run log -> /tmp/writegate_final.log
"""
import os, re, time, subprocess, signal
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BASE_CFG = os.path.join(REPO, "reference/phase2_write.yaml")
G1 = "GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57"
G3 = "GPU-07f5cef9-a5d1-052e-6b9c-d7792fd42eb0"
G4 = "GPU-7d78717a-0797-ee9b-9ff7-1af3203ce04e"
G7 = "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107"
DEVS = f"{G7},{G3},{G4},{G1}"   # 4 GPUs: G7 (lily-free) as cuda:0/rank-0; +G3,G4,G1 (lily-only). 4-way FSDP shards optim smaller -> closes the 36MB gap for 2 layers
LADDER = [[5, 17]]   # full window (24); try 2 layers on 4 GPUs (2L was 36MB short on 3); fallback to 1
RESULT = "/tmp/writegate_ladder_result.txt"
PROBE_STEP = 6           # reaching this many steps = a generator update survived = fits
PROBE_DEADLINE = 1800    # 30 min per probe (load + steps on 3 GPU)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def env():
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=DEVS, NCCL_P2P_DISABLE="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def write_cfg(layers, logdir, max_iters):
    cfg = OmegaConf.load(BASE_CFG)
    cfg.model_kwargs.memory_layers = list(layers)
    cfg.logdir = logdir
    cfg.max_iters = max_iters
    p = f"/tmp/_writegate_L{len(layers)}.yaml"
    OmegaConf.save(cfg, p)
    return p


def launch(cfgpath, port, logpath):
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=4", "--max-restarts=0",
           f"--master_port={port}", "train.py", "--config_path", cfgpath,
           "--no-auto-resume", "--no-one-logger"]
    return subprocess.Popen(cmd, cwd=REPO, env=env(),
                            stdout=open(logpath, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def gpus_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used = {}
    for line in out.strip().splitlines():
        try:
            u, m = [c.strip() for c in line.split(",")]
            used[u] = int(m)
        except ValueError:
            pass
    # G1/G3/G4 shared with lilyshi (~6GB) -> "free for us" if < 12GB ; G7 must be ~empty
    return (used.get(G1, 1e9) < 12000 and used.get(G3, 1e9) < 12000
            and used.get(G4, 1e9) < 12000 and used.get(G7, 1e9) < 3000)


def wait_free(timeout=300):
    t = time.time() + timeout
    while time.time() < t:
        if gpus_free():
            return
        time.sleep(10)


def probe(logpath, proc):
    """Return PASS / OOM / CRASH / TIMEOUT. Does NOT kill on PASS (caller decides)."""
    deadline = time.time() + PROBE_DEADLINE
    while time.time() < deadline:
        txt = ""
        try:
            txt = open(logpath).read()
        except Exception:
            pass
        low = txt.lower()
        if "out of memory" in low or "outofmemoryerror" in low:
            return "OOM"
        steps = re.findall(r"step=(\d+)", txt)
        if steps and int(steps[-1]) >= PROBE_STEP:
            return "PASS"
        if proc.poll() is not None:
            return "OOM" if ("out of memory" in low or "outofmemoryerror" in low) else "CRASH"
        time.sleep(20)
    return "TIMEOUT"


def main():
    res = []
    # cleanup any existing write run
    subprocess.run("for p in $(pgrep -f 'phase2_write.yaml\\|_writegate'); do [ \"$(cat /proc/$p/comm 2>/dev/null)\" = python ] && kill -9 $p; done",
                   shell=True)
    wait_free()
    maxL = None
    port = 29956
    for layers in LADDER:
        log(f"probing {len(layers)} layer(s): {layers}")
        cfg = write_cfg(layers, f"/shared/ssd_14T_project/adenelmi/e2_probe_{len(layers)}L", max_iters=20)
        lp = f"/tmp/writegate_probe_{len(layers)}L.log"
        proc = launch(cfg, port, lp); port += 1
        st = probe(lp, proc)
        res.append(f"{len(layers)} layer(s) {layers}: {st}")
        log(f"{len(layers)}L -> {st}")
        kill(proc)
        wait_free()
        if st == "PASS":
            maxL = layers
        else:
            break
    if maxL is None:
        res.append("2 layers did not fit -> falling back to 1 layer [5] (known good)")
        maxL = [5]
    # launch the REAL run at max layers
    log(f"MAX = {len(maxL)} layers {maxL} -> launching REAL run (3000 steps)")
    cfg = write_cfg(maxL, f"/shared/ssd_14T_project/adenelmi/e2_writegate_{len(maxL)}L", max_iters=3000)
    proc = launch(cfg, port, "/tmp/writegate_final.log")
    st = probe("/tmp/writegate_final.log", proc)   # confirm it gets going
    if st == "PASS":
        res.append(f"REAL RUN LIVE: {len(maxL)} layers {maxL}, pid={proc.pid}, train_write+gate(6+5+2), 3000 steps, ckpt/150")
    else:
        res.append(f"REAL RUN failed to start ({st}) -- see /tmp/writegate_final.log")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
