#!/usr/bin/env python
"""Opportunistic gating launcher.

If 2 sane GPUs become free (60s stable, broken GPU excluded), launch the gating
(Idea-6, gate_extra=[agreement]) training on them NOW -- in parallel with the
read-only run on GPU 1+7. Writes /tmp/gating_launched.flag so the post-train
pipeline won't double-launch the gating. Exits as soon as the flag exists
(whether set here or by the pipeline).
"""
import os, subprocess, time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"
CFG = os.path.join(REPO, "reference/next_run_5L_agreement.yaml")
FLAG = "/tmp/gating_launched.flag"
FREE_MEM, FREE_UTIL, FREE_SECS, POLL = 1000, 10, 60, 10


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


def launch(g1, g2):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=f"{g1},{g2}", NCCL_P2P_DISABLE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--max-restarts=0",
           "--master_port=29952", "train.py", "--config_path", CFG,
           "--no-auto-resume", "--no-one-logger"]
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open("/tmp/gating_train.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def main():
    if os.path.exists(FLAG):
        log("gating already launched (flag exists) -> exit")
        return
    log("opportunistic gating launcher up: waiting for 2 free GPUs (60s stable)...")
    seen = {}
    while True:
        if os.path.exists(FLAG):
            log("flag appeared (pipeline launched gating) -> exit")
            return
        now = time.time()
        fu = free_uuids()
        for u in list(seen):
            if u not in fu:
                del seen[u]
        for u in fu:
            seen.setdefault(u, now)
        stable = sorted([u for u, t in seen.items() if now - t >= FREE_SECS], key=lambda u: seen[u])
        if len(stable) >= 2 and free_uuids() >= {stable[0], stable[1]}:
            g1, g2 = stable[0], stable[1]
            pid = launch(g1, g2)
            open(FLAG, "w").write(f"pid={pid} gpus={g1},{g2} run=e1_ro_5L_agreement (opportunistic)\n")
            log(f"GATING launched opportunistically pid={pid} on {g1[:12]},{g2[:12]}")
            return
        log(f"waiting; stable={len(stable)} free={len(fu)}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
