"""4-GPU watcher: read-only (3 GPUs) -> 2-layer no-combo write (4 GPUs).

Waits for a 4th clean GPU (one of nvidia-smi 0/3/4/5; gpu6=ab1015a0 is DEAD/poison,
gpu1/2/7 are ours). When one appears:
  1. (optionally) grab the latest read-only checkpoint to resume Phase-1 progress,
  2. kill the 3-GPU read-only hold,
  3. launch 2-layer no-combo write on the 4 GPUs (our gpu1/2/7 + the new one),
  4. confirm it steps past the OOM/gen point, log "4-GPU WRITE LIVE", exit.

Device map note (this machine): CUDA default order swaps nvidia-smi gpu6<->gpu7.
  nvidia-smi idx -> CVD idx : 0..5 identity, 7 -> 6 ; gpu6 (DEAD) would be CVD7 = NEVER.
"""
import os, re, time, glob, subprocess

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY   = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
POISON_SMI_IDX = 6
CUR_SMI = {1, 2, 7}                                  # our read-only GPUs
IDX2CVD = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",7:"6"}
CANDIDATE_SMI = [0, 3, 4, 5]                          # where a 4th could free up
CLEAN_THRESH = 3000
READ_LOGDIR = "/shared/ssd_14T_project/adenelmi/read_1L_3gpu"
WRITE_CFG = "/tmp/_write2Lnc.yaml"                    # 2 layers [5,17], gate_extra [], train_write
WRITE_LOGDIR = "/shared/ssd_14T_project/adenelmi/write_2L_4gpu"
WRITE_LOG = "/tmp/gpu4_write.log"
STATUS = "/tmp/gpu4_watcher.txt"
ENV_EXTRA = dict(NCCL_P2P_DISABLE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        open(STATUS, "a").write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def other_user_mem():
    """nvidia-smi idx -> MiB used by NON-adenelmi processes."""
    uuid2idx = {}
    for ln in subprocess.run(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines():
        i, u = [x.strip() for x in ln.split(",")]
        uuid2idx[u] = int(i)
    used = {i: 0 for i in uuid2idx.values()}
    out = subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3 or p[0] not in uuid2idx or not p[1].isdigit():
            continue
        owner = subprocess.run(["ps","-o","user=","-p",p[1]], capture_output=True, text=True).stdout.strip()
        if owner != "adenelmi":
            try: used[uuid2idx[p[0]]] += int(p[2])
            except ValueError: pass
    return used


def latest_ckpt():
    cks = glob.glob(os.path.join(READ_LOGDIR, "checkpoint_model_*/model.pt"))
    best, beststep = None, -1
    for c in cks:
        m = re.search(r"checkpoint_model_(\d+)", c)
        s = int(m.group(1)) if m else -1
        if s > beststep: best, beststep = c, s
    return best, beststep


def main():
    open(STATUS, "w").close()
    log("gpu4 watcher up: wait for a 4th clean GPU (0/3/4/5), then read-only(3) -> 2L no-combo write(4).")
    while True:
        other = other_user_mem()
        free4 = [i for i in CANDIDATE_SMI if other.get(i, 9e9) < CLEAN_THRESH]
        if free4:
            new = free4[0]
            ckpt, step = latest_ckpt()
            cvd = ",".join(IDX2CVD[i] for i in sorted(CUR_SMI | {new}))
            log(f"4th GPU = smi {new}. ckpt={ckpt} (step {step}). Launching 2L no-combo write on 4 GPUs (CVD={cvd}).")
            # point the write config at the read-only checkpoint if we have one
            from omegaconf import OmegaConf
            c = OmegaConf.load(WRITE_CFG)
            c.logdir = WRITE_LOGDIR
            if ckpt:
                c.lora_ckpt = ckpt
            OmegaConf.save(c, WRITE_CFG)
            # kill read-only
            subprocess.run("pkill -9 -u adenelmi -f '_read1L.yaml'", shell=True)
            subprocess.run("pkill -9 -u adenelmi -f 'master_port=30047'", shell=True)
            time.sleep(8)
            # launch 4-GPU write
            e = dict(os.environ); e.update(ENV_EXTRA, CUDA_VISIBLE_DEVICES=cvd)
            open(WRITE_LOG, "w").close()
            proc = subprocess.Popen(
                [PY,"-m","torch.distributed.run",f"--nproc_per_node=4","--max-restarts=0",
                 "--master_port=30051","train.py","--config_path",WRITE_CFG,
                 "--no-one-logger","--no-auto-resume"],
                cwd=REPO, env=e, stdout=open(WRITE_LOG,"w"), stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            # confirm
            deadline = time.time() + 1800
            while time.time() < deadline:
                try: txt = open(WRITE_LOG).read()
                except Exception: txt = ""
                if "out of memory" in txt.lower():
                    log("4-GPU 2L write OOM -- did not fit even on 4 GPUs, check manually."); return
                steps = re.findall(r"step=(\d+)", txt)
                if steps and int(steps[-1]) >= 7:
                    log(f"4-GPU WRITE LIVE: 2L no-combo on 4 GPUs (CVD={cvd}) passed step 6. wandb run write_2L_4gpu"); return
                if proc.poll() is not None:
                    log("4-GPU 2L write DIED at init, check manually."); return
                time.sleep(15)
            log("4-GPU 2L write SLOW (no step 7 in 30min), check manually."); return
        time.sleep(20)


if __name__ == "__main__":
    main()
