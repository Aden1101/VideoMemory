"""4-GPU handoff watcher.

Watches for a 4th clean+good GPU. When one appears AND the running 1-layer/3-GPU
write run has saved at least one checkpoint, it:
  1. kills the 1-layer run (checkpoint stays on disk),
  2. launches the 2-layer write run on 4 GPUs, resuming block-5 memory + LoRA from
     that checkpoint (block-17 memory starts fresh, strict=False),
  3. confirms it steps, logs "4-GPU HANDOFF DONE", and exits.

Honors the user's rule: it does NOT stop the 1-layer run until a checkpoint exists
(so we never lose training), and it relaunches immediately (no idle window).

Device map note (this machine): CUDA default order swaps nvidia-smi gpu6<->gpu7.
  nvidia-smi idx -> CVD idx : 0..5 identity, 7 -> 6, and 6 (ab1015a0, DEAD/poison) -> 7 = NEVER use.
"""
import os, re, time, glob, subprocess

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY   = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
POISON_SMI_IDX = 6                       # gpu6 = ab1015a0, dead (poisons torch init)
CUR_SMI = {1, 2, 7}                      # the 1-layer run's GPUs (nvidia-smi indices)
IDX2CVD = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",7:"6"}   # 6 excluded on purpose
CLEAN_THRESH = 3000                      # MiB of OTHER users => GPU is clean below this
ONE_CFG  = "/tmp/_write1L.yaml"
ONE_LOG  = "/tmp/write1L.log"
ONE_LOGDIR = "/shared/ssd_14T_project/adenelmi/write_1L_3gpu"
TWO_CFG  = "/tmp/_write2L.yaml"
TWO_LOG  = "/tmp/handoff_2L_4gpu.log"
TWO_LOGDIR = "/shared/ssd_14T_project/adenelmi/write_2L_4gpu"
STATUS   = "/tmp/handoff_watcher.txt"
ENV_EXTRA = dict(NCCL_P2P_DISABLE="1",
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def smi(query, extra=None):
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    if extra:
        cmd[1:1] = extra
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()


def other_user_mem_by_idx():
    """nvidia-smi idx -> MiB used by processes NOT owned by adenelmi."""
    idx2uuid, uuid2idx = {}, {}
    for ln in smi("gpu=index,uuid"):
        i, u = [x.strip() for x in ln.split(",")]
        idx2uuid[int(i)] = u; uuid2idx[u] = int(i)
    used = {i: 0 for i in idx2uuid}
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for ln in out.strip().splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid, mem = parts[0], parts[1], parts[2]
        if uuid not in uuid2idx or not pid.isdigit():
            continue
        try:
            owner = subprocess.run(["ps", "-o", "user=", "-p", pid],
                                   capture_output=True, text=True).stdout.strip()
        except Exception:
            owner = ""
        if owner != "adenelmi":
            try:
                used[uuid2idx[uuid]] += int(mem)
            except ValueError:
                pass
    return used


def latest_ckpt(logdir):
    cks = glob.glob(os.path.join(logdir, "checkpoint_model_*/model.pt"))
    if not cks:
        return None, -1
    best, beststep = None, -1
    for c in cks:
        m = re.search(r"checkpoint_model_(\d+)", c)
        s = int(m.group(1)) if m else -1
        if s > beststep:
            best, beststep = c, s
    return best, beststep


def kill_one():
    subprocess.run("pkill -9 -u adenelmi -f '_write1L.yaml'", shell=True)
    subprocess.run("pkill -9 -u adenelmi -f 'torch.distributed.run'", shell=True)
    time.sleep(6)


def set_two_cfg(ckpt):
    # point the 2-layer config at the 1-layer checkpoint + a fresh logdir
    from omegaconf import OmegaConf
    c = OmegaConf.load(TWO_CFG)
    c.lora_ckpt = ckpt
    c.logdir = TWO_LOGDIR
    OmegaConf.save(c, TWO_CFG)


def launch_two(cvd):
    e = dict(os.environ); e.update(ENV_EXTRA, CUDA_VISIBLE_DEVICES=cvd)
    open(TWO_LOG, "w").close()
    cmd = [PY, "-m", "torch.distributed.run", f"--nproc_per_node={len(cvd.split(','))}",
           "--max-restarts=0", "--master_port=30021", "train.py",
           "--config_path", TWO_CFG, "--no-one-logger", "--no-auto-resume"]
    return subprocess.Popen(cmd, cwd=REPO, env=e,
                            stdout=open(TWO_LOG, "w"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def main():
    open(STATUS, "w").close()
    log("handoff watcher up: wait for a 4th clean GPU + a saved checkpoint, then 1L->2L on 4 GPUs.")
    warned_wait = False
    while True:
        other = other_user_mem_by_idx()
        # candidate 4th GPUs: clean, not poison, not already ours
        fourth = [i for i, m in other.items()
                  if i != POISON_SMI_IDX and i not in CUR_SMI and i in IDX2CVD and m < CLEAN_THRESH]
        if fourth:
            ckpt, step = latest_ckpt(ONE_LOGDIR)
            if ckpt is None:
                if not warned_wait:
                    log(f"4th GPU free (smi {fourth[0]}) but NO checkpoint yet -> waiting for the 1L run to save (step 150).")
                    warned_wait = True
                time.sleep(20); continue
            new = fourth[0]
            cvd = ",".join([IDX2CVD[i] for i in sorted(CUR_SMI | {new})])
            log(f"HANDOFF: 4th GPU = smi {new}; checkpoint @ step {step} = {ckpt}")
            set_two_cfg(ckpt)
            kill_one()
            log(f"1-layer run stopped. Launching 2-layer on 4 GPUs (CVD={cvd}) resuming from step {step}...")
            proc = launch_two(cvd)
            # confirm it steps
            deadline = time.time() + 1200
            ok = False
            while time.time() < deadline:
                try:
                    txt = open(TWO_LOG).read()
                except Exception:
                    txt = ""
                if "out of memory" in txt.lower():
                    log("2-layer/4-GPU OOM -- handoff FAILED, check manually."); break
                steps = re.findall(r"step=(\d+)", txt)
                if steps and int(steps[-1]) >= 2:
                    ok = True; break
                if proc.poll() is not None:
                    log("2-layer/4-GPU DIED at init -- handoff FAILED, check manually."); break
                time.sleep(15)
            if ok:
                log(f"4-GPU HANDOFF DONE: 2-layer LIVE on 4 GPUs (CVD={cvd}), resumed step {step}. wandb run write_2L_4gpu")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
