#!/usr/bin/env python
"""Parallel A/B eval: 3L WRITE memory (ckpt 1200) vs AF baseline, at 60/120/240s, across gpu1/2/7.

Worker pool: 3 workers, one bound to each usable GPU (by UUID), pulling from a longest-first job
queue so the heavy 240s jobs start immediately and the pool self-balances. Each job = inference
(inference/inference.py, 1 GPU) then VBench (subject + background consistency, long_custom_input).
Result trend -> is the identity drift horizon-limited? (does mem's gap vs baseline grow with length).
"""
import os, glob, json, subprocess, threading, queue, time
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VBENCH_DIR = "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo"
OUT = "/shared/ssd_14T_project/adenelmi/e2_write3L_eval"
MEM_CKPT = "/shared/ssd_14T_project/adenelmi/e2_write3L/checkpoint_model_001200/model.pt"
MEM_LAYERS = [5, 16, 27]
PROMPTS = os.path.join(REPO, "reference/cmp_prompts_5.txt")
# the 3 usable GPUs by UUID: gpu1, gpu2, gpu7 (avoids dead gpu6 + other users)
GPUS = ["GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57",   # gpu1
        "GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7",   # gpu2
        "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107"]   # gpu7
# latent frames -> seconds: (1+4*(L-1))/16   ==>  240~=60s, 480~=120s, 960~=240s
LENGTHS = [240, 480]   # phase 1: 60s + 120s first results; 960 (240s) deferred to phase 2
LOGDIR = os.path.join(REPO, "reference/e2_eval_logs")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def infer(tag, lora_ckpt, layers, out_dir, nframes, uuid, port):
    cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
    cfg.lora_ckpt = lora_ckpt
    cfg.model_kwargs.memory_layers = list(layers)
    if layers:  # memory jobs: replicate the TRAINING memory architecture (else the ckpt's
        # gate_extra keys gx_po/gx_pc/proj_extra are "unexpected" -> inference asserts and dies
        cfg.model_kwargs.gate_init = 0.1
        cfg.model_kwargs.gate_extra = ["agreement", "frames_since_switch", "denoising_step"]
        cfg.model_kwargs.ckpt_write = True
    cfg.output_folder = out_dir
    cfg.num_output_frames = nframes
    cfg.data_path = PROMPTS
    p = f"/tmp/_e2eval_{tag}.yaml"
    OmegaConf.save(cfg, p)
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=uuid, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "inference/inference.py", "--config_path", p]
    log(f"[{tag}] INFER {nframes} latent on ...{uuid[-12:]} (port {port}) -> {out_dir}")
    with open(f"{LOGDIR}/infer_{tag}.log", "w") as lf:
        subprocess.run(cmd, cwd=REPO, env=e, stdout=lf, stderr=subprocess.STDOUT, timeout=28800)


def vbench(tag, videos_dir, out_dir, uuid):
    os.makedirs(out_dir, exist_ok=True)
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=uuid, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for dim in ["subject_consistency", "background_consistency"]:
        cmd = [PY, "vbench2_beta_long/eval_long.py", "--videos_path", videos_dir,
               "--dimension", dim, "--mode", "long_custom_input", "--output_path", out_dir]
        log(f"[{tag}] VBENCH {dim} on ...{uuid[-12:]}")
        with open(f"{LOGDIR}/vbench_{tag}.log", "a") as lf:
            subprocess.run(cmd, cwd=VBENCH_DIR, env=e, stdout=lf, stderr=subprocess.STDOUT, timeout=14400)


# the AF baseline is deterministic (we only trained the memory), so baselines already computed at
# 240 latent (60s) and 960 latent (240s) on these exact 5 prompts are REUSED (e0_cmp). Only the 3
# memory lengths + the MISSING baseline at 480 latent (120s) are (re)generated here.
EXISTING_BASE = {
    240: "/shared/ssd_14T_project/adenelmi/e0_cmp/eval_baseline_L240",   # subj 0.9597 / bg 0.9686
    960: "/shared/ssd_14T_project/adenelmi/e0_cmp/eval_baseline",        # subj 0.9535 / bg 0.9644
}
JOBS = [
    ("mem_L240", MEM_CKPT, list(MEM_LAYERS), 240),   # 60s -- fastest, first A/B (vs reused base 0.9597)
    ("mem_L480", MEM_CKPT, list(MEM_LAYERS), 480),   # 120s (base_L480 already done: 0.9552/0.9626)
    # base_L480 done; baselines 60s/240s reused; mem_L960 (240s) deferred to phase 2
]

q = queue.Queue()
for j in JOBS:
    q.put(j)


def worker(wid, uuid):
    port = 29950 + wid * 30
    while True:
        try:
            tag, lora, layers, L = q.get_nowait()
        except queue.Empty:
            log(f"[worker{wid} ...{uuid[-12:]}] queue empty, exiting")
            return
        try:
            infer(tag, lora, layers, f"{OUT}/{tag}", L, uuid, port)
            port += 1
            vbench(tag, f"{OUT}/{tag}", f"{OUT}/eval_{tag}", uuid)
        except Exception as ex:
            log(f"[{tag}] FAILED: {ex}")
        finally:
            q.task_done()


def scores(out_dir):
    s = {}
    for f in glob.glob(os.path.join(out_dir, "*eval_results.json")):
        try:
            d = json.load(open(f))
            for k, v in d.items():
                s[k] = round(v[0], 4) if isinstance(v, list) else v
        except Exception:
            pass
    return s


def main():
    t0 = time.time()
    log(f"START parallel eval: {len(JOBS)} jobs on {len(GPUS)} GPUs -> {OUT}")
    threads = [threading.Thread(target=worker, args=(i, GPUS[i]), daemon=False) for i in range(len(GPUS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    res = ["E2 3L WRITE (ckpt 1200, layers [5,16,27]) vs AF BASELINE -- VBench, same 5 prompts",
           "(baseline 60s/240s reused from e0_cmp; only baseline 120s freshly run)", ""]
    for L in sorted(LENGTHS):
        b = scores(EXISTING_BASE[L]) if L in EXISTING_BASE else scores(f"{OUT}/eval_base_L{L}")
        m = scores(f"{OUT}/eval_mem_L{L}")
        sb, sm = b.get("subject_consistency", 0), m.get("subject_consistency", 0)
        gb, gm = b.get("background_consistency", 0), m.get("background_consistency", 0)
        sec = round((1 + 4 * (L - 1)) / 16)
        res.append(f"L={L:4d} (~{sec:3d}s)  subject: base={sb:.4f} mem={sm:.4f} d={sm-sb:+.4f}  "
                   f"| background: base={gb:.4f} mem={gm:.4f} d={gm-gb:+.4f}")
    out = "\n".join(res)
    open(f"{OUT}/RESULT.txt", "w").write(out + "\n")
    log(f"DONE in {round((time.time()-t0)/60,1)} min:\n" + out)


if __name__ == "__main__":
    main()
