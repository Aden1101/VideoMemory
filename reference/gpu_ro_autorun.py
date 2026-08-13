#!/usr/bin/env python
"""Autonomous read-only memory training launcher (E1, post-fix).

60s stability window (a GPU must be free 60s straight before we grab it, so we don't race a
job launching at the same instant). Opportunistic, 2-GPU run is the priority:
  - 1 GPU free  -> launch an early-signal 1-GPU read-only run, 2 layers {5,17} (fits per VRAM
    table), once. It is a stopgap.
  - 2 GPUs available (free GPUs + the one the 1-GPU run holds) -> CANCEL the 1-GPU run, then
    VRAM-profile 5 layers {5,11,17,23,29}; if OOM fall back to 4 {5,12,19,26}; at the first that
    FITS, launch the REAL 2-GPU read-only training (3000 steps) and exit.
  - both 5L and 4L OOM -> stop and report (window reduction needed -- decide with Aden).

wandb ON for every run and profile. Fully autonomous, no pinging. Logs /tmp/ro_autorun.log,
result /tmp/ro_autorun_result.txt, training stdout /tmp/ro_train.log (2-GPU) and
/tmp/ro_train_1gpu.log (1-GPU early run).
"""
import os, re, signal, subprocess, time
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BASE_CFG = os.path.join(REPO, "configs/ttt_memory.yaml")
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"
ENTITY = "elmi_aden-harvard-university"
PROJECT = "anchor-forcing-memory"
# (tag, layers, fresh logdir basename -> wandb run name)
LAYER_SETS = [("5L", "5,11,17,23,29"), ("4L", "5,12,19,26")]
LAYER_SET_1GPU = ("2L_1gpu", "5,17")   # read-only 2 layers fits on 1 GPU (VRAM table); early-signal run
FREE_MEM_MIB, FREE_UTIL, FREE_SECS, POLL = 1000, 10, 60, 10  # FREE_SECS=60: GPU must be free 60s straight before we grab it (avoids racing a simultaneous launcher)
RESULTS = "/tmp/ro_autorun_result.txt"
TRAIN_LOG = "/tmp/ro_train.log"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        u, m, ut = [c.strip() for c in line.split(",")]
        try:
            rows.append((u, int(m), int(ut)))
        except ValueError:
            rows.append((u, 10**9, 100))
    return rows


def free_uuids():
    return {u for (u, m, ut) in gpus() if u != BROKEN and m < FREE_MEM_MIB and ut < FREE_UTIL}


def alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_group(pid):
    """SIGTERM the whole process group of a detached run (it is a session leader)."""
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def _set_mem(cfg, layers):
    """Override which blocks get a memory branch via model_kwargs (config-driven, no env vars).
    gate_init stays at the base config value (0.1)."""
    cfg.model_kwargs.memory_layers = [int(x) for x in layers.split(",") if x.strip()]


def make_profile_cfg(tag, layers):
    cfg = OmegaConf.load(BASE_CFG)
    cfg.max_iters = 12
    cfg.log_iters = 100000
    cfg.use_wandb = True
    cfg.logdir = f"/tmp/ro_prof_{tag}"
    cfg.profile_vram = 3
    cfg.profile_vram_warmup = 4
    _set_mem(cfg, layers)
    p = f"/tmp/_ro_prof_{tag}.yaml"
    OmegaConf.save(cfg, p)
    return p


def make_train_cfg(tag, layers):
    cfg = OmegaConf.load(BASE_CFG)
    cfg.logdir = f"/shared/ssd_14T_project/adenelmi/e1_ro_{tag}_postfix"
    _set_mem(cfg, layers)
    p = f"/tmp/_ro_train_{tag}.yaml"
    OmegaConf.save(cfg, p)
    return p


def base_env(devs):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=devs, NCCL_P2P_DISABLE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def profile(tag, layers, ua, ub, port):
    cfg = make_profile_cfg(tag, layers)
    env = base_env(f"{ua},{ub}")  # wandb ON for everything, incl. profiles (system GPU-mem curve)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--max-restarts=0",
           f"--master_port={port}", "train.py", "--config_path", cfg,
           "--no-auto-resume", "--no-one-logger"]
    log(f"profiling {tag} ({layers}) ...")
    try:
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
    m = re.search(r"\[VRAM_PROFILE\][^\n]*", out)
    if m:
        return "PASS", m.group(0)
    if "OutOfMemoryError" in out or "out of memory" in out:
        d = re.search(r"Tried to allocate [^.]*\.[^.]*free", out)
        return "OOM", (d.group(0) if d else "OOM")
    return "ERROR", "\n".join(out.strip().splitlines()[-4:])


def launch_training(tag, layers, ua, ub, port):
    cfg = make_train_cfg(tag, layers)
    env = base_env(f"{ua},{ub}")  # wandb ON (no WANDB_MODE)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--max-restarts=0",
           f"--master_port={port}", "train.py", "--config_path", cfg,
           "--no-auto-resume", "--no-one-logger"]
    logf = open(TRAIN_LOG, "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    run_name = f"e1_ro_{tag}_postfix"
    url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_name}"
    return p.pid, run_name, url


def launch_1gpu(uuid, layers, port):
    """Early-signal read-only run on a SINGLE free GPU (2 layers fit per the VRAM table)."""
    tag = LAYER_SET_1GPU[0]
    cfg = make_train_cfg(tag, layers)
    env = base_env(uuid)  # one GPU, wandb ON
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", "--max-restarts=0",
           f"--master_port={port}", "train.py", "--config_path", cfg,
           "--no-auto-resume", "--no-one-logger"]
    logf = open("/tmp/ro_train_1gpu.log", "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    run_name = f"e1_ro_{tag}_postfix"
    url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_name}"
    return p.pid, run_name, url


def write_result(text):
    with open(RESULTS, "w") as f:
        f.write(text + "\n")
    log(text)


def _refresh_stable(seen):
    """Update the per-GPU free-since map and return the uuids free >= FREE_SECS straight."""
    now = time.time()
    fu = free_uuids()
    for u in list(seen):
        if u not in fu:
            del seen[u]
    for u in fu:
        seen.setdefault(u, now)
    stable = sorted([u for u, t in seen.items() if now - t >= FREE_SECS], key=lambda u: seen[u])
    return stable, fu


def main():
    log(f"autonomous RO launcher up ({FREE_SECS}s stability window). 1 GPU stable -> 1-GPU 2L "
        "read-only (early signal, CANCELLED as soon as 2 GPUs become available); 2 available -> "
        "5L (fallback 4L). wandb on.")
    port = 29940
    small_pid, small_uuid = None, None   # the 1-GPU early run, and the GPU it holds
    seen = {}                             # uuid -> first time seen free (stability window)
    while True:
        if small_pid and not alive(small_pid):
            log("[1gpu] early run exited on its own"); small_pid, small_uuid = None, None
        stable, fu = _refresh_stable(seen)
        # available for the real run = stably-free GPUs + the one we can reclaim from the 1-GPU run
        avail = len(stable) + (1 if alive(small_pid) else 0)

        # --- enough for the real 2-GPU run: cancel the 1-GPU run first, then 5L/4L ---
        if avail >= 2:
            reclaimed = None
            if alive(small_pid):
                log(f"[2gpu] 2 GPUs available -> cancelling 1-GPU run (pid {small_pid})")
                kill_group(small_pid); reclaimed = small_uuid
                small_pid, small_uuid = None, None
                for _ in range(30):                 # wait (<=60s) for the reclaimed GPU to free
                    time.sleep(2)
                    if reclaimed in free_uuids():
                        break
            stable, fu = _refresh_stable(seen)
            # the reclaimed GPU is ours (safe to reuse now); fill up with stably-free others
            pick = [reclaimed] if (reclaimed and reclaimed in fu) else []
            for u in stable:
                if u not in pick:
                    pick.append(u)
                if len(pick) >= 2:
                    break
            if len(pick) < 2:
                log(f"[2gpu] not 2 usable after cancel (have {len(pick)}); re-polling")
                time.sleep(POLL); continue
            ua, ub = pick[0], pick[1]
            oom, lost = 0, False
            for tag, layers in LAYER_SETS:
                if not (free_uuids() >= {ua, ub}):
                    lost = True; break
                status, detail = profile(tag, layers, ua, ub, port); port += 1
                log(f"[2gpu] {tag}: {status} | {detail}")
                if status == "PASS":
                    pid, run_name, url = launch_training(tag, layers, ua, ub, port); port += 1
                    write_result(f"FITS at {tag} ({layers}). 2-GPU read-only training LAUNCHED "
                                 f"pid={pid} run={run_name}\nVRAM: {detail}\nwandb: {url}")
                    return
                if status == "ERROR":
                    write_result(f"[2gpu] {tag}: ERROR (not OOM) -- {detail}")
                    return
                oom += 1
            if lost:
                log("[2gpu] a GPU was taken mid-profile; re-polling")
            elif oom == len(LAYER_SETS):
                write_result("Both 5L and 4L OOM at window=60 on 2 GPUs. Reduce window or rethink.")
                return
        # --- only 1 GPU's worth (stably free): fire the early-signal 1-GPU run once ---
        elif len(stable) >= 1 and not alive(small_pid):
            uuid = stable[0]
            small_pid, run_name, url = launch_1gpu(uuid, LAYER_SET_1GPU[1], port); port += 1
            small_uuid = uuid
            log(f"[1gpu] early read-only 2L LAUNCHED on {uuid[:20]} pid={small_pid} "
                f"run={run_name} wandb={url}")
        else:
            log(f"waiting; stable={len(stable)} free={len(fu)} small_running={alive(small_pid)}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
