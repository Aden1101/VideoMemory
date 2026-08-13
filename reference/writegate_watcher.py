#!/usr/bin/env python
"""Write-only watcher with adaptive GPU count + layer-ladder.

When >=2 fully-clean GPUs are free, grab as many as available (2/3/4, cap 4) and launch the
write+gate run, trying the MOST memory layers that fit: descend a layer ladder (more GPUs ->
start higher) until a config runs healthy. No read-only, no holders, window unchanged (24).

GPU count -> starting layers:  N==2 -> try [2,1] ;  N>=3 -> try [3,2,1].
Layer sets (spread across depth): 1->[5], 2->[5,17], 3->[5,17,29].

Status -> /tmp/writegate_watcher.txt ; per-attempt logs -> /tmp/writeL{L}.log
"""
import os, re, time, subprocess, signal
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BASE_CFG = os.path.join(REPO, "reference/phase2_write.yaml")
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"   # G6, dead
CLEAN_THRESH = 3000      # MiB of OTHER users => GPU is write-clean below this
MAX_TOTAL = 4
LAYER_SETS = {1: [5], 2: [5, 17], 3: [5, 17, 29]}
STATUS = "/tmp/writegate_watcher.txt"

_IDX = {}


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    open(STATUS, "a").write(s + "\n")


def short(u):
    return f"gpu{_IDX.get(u, '?')}"


def all_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    us = []
    for line in out.strip().splitlines():
        try:
            idx, u = [c.strip() for c in line.split(",")]
            _IDX[u] = idx
            if u != BROKEN:
                us.append(u)
        except ValueError:
            pass
    return us


def is_ours(pid):
    try:
        cl = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore")
    except Exception:
        return False
    return "_writeL" in cl or "phase2_write.yaml" in cl


def other_used():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        try:
            u, p, m = [c.strip() for c in line.split(",")]
            if is_ours(int(p)):
                continue
            d[u] = d.get(u, 0) + int(m)
        except ValueError:
            pass
    return d


def kill_group(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


def cleanup():
    subprocess.run("for p in $(pgrep -f '_writeL[0-9]'); do "
                   "[ \"$(cat /proc/$p/comm 2>/dev/null)\" = python ] && kill -9 $p; done", shell=True)


def make_cfg(L):
    cfg = OmegaConf.load(BASE_CFG)
    cfg.model_kwargs.memory_layers = list(LAYER_SETS[L])
    cfg.logdir = f"/shared/ssd_14T_project/adenelmi/e2_writegate_{L}L"
    p = f"/tmp/_writeL{L}.yaml"
    OmegaConf.save(cfg, p)
    return p


def launch(devs, port, cfgpath, logpath):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=",".join(devs), NCCL_P2P_DISABLE="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-m", "torch.distributed.run", f"--nproc_per_node={len(devs)}", "--max-restarts=0",
           f"--master_port={port}", "train.py", "--config_path", cfgpath,
           "--no-one-logger", "--no-auto-resume"]
    return subprocess.Popen(cmd, cwd=REPO, env=e, stdout=open(logpath, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def confirm(proc, logpath):
    """HEALTHY (passed generator step 6) / SLOW (progressing) / OOM / DIED."""
    deadline = time.time() + 2400
    while time.time() < deadline:
        try:
            txt = open(logpath).read()
        except Exception:
            txt = ""
        low = txt.lower()
        if "out of memory" in low or "outofmemoryerror" in low:
            return "OOM"
        steps = re.findall(r"step=(\d+)", txt)
        if steps and int(steps[-1]) >= 6:
            return "HEALTHY"
        if proc.poll() is not None:
            return "OOM" if ("out of memory" in low or "outofmemory" in low) else "DIED"
        time.sleep(15)
    return "SLOW"


def ladder_for(n):
    return [3, 2, 1] if n >= 3 else [2, 1]


def main():
    open(STATUS, "w").write("")
    cleanup()
    port = 29970
    log(f"watcher up: grab 2-4 clean GPUs, try max layers that fit (ladder), then write+gate. window 24.")
    while True:
        gpus = all_gpus()
        ou = other_used()
        write_clean = [u for u in gpus if ou.get(u, 0) < CLEAN_THRESH]
        if len(write_clean) >= 2:
            n = min(len(write_clean), MAX_TOTAL)
            devs = write_clean[:n]
            log(f"{len(write_clean)} clean -> grabbing {n} GPUs: {[short(x) for x in devs]}; ladder {ladder_for(n)}")
            launched = False
            for L in ladder_for(n):
                lp = f"/tmp/writeL{L}.log"
                log(f"try {L} layer(s) {LAYER_SETS[L]} on {n} GPUs")
                proc = launch(devs, port, make_cfg(L), lp); port += 1
                res = confirm(proc, lp)
                log(f"{L} layer(s) -> {res}")
                if res in ("HEALTHY", "SLOW"):
                    log(f"WRITE+GATE LIVE on {n} GPUs {[short(x) for x in devs]}, {L} layers {LAYER_SETS[L]} -- watcher done.")
                    launched = True
                    break
                kill_group(proc); cleanup(); time.sleep(8)
            if launched:
                return
            log("even 1 layer failed (GPUs likely got snatched mid-ladder) -> keep waiting")
        time.sleep(3)


if __name__ == "__main__":
    main()
