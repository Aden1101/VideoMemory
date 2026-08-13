#!/usr/bin/env python
"""Length-sweep A/B eval: does the read-only memory hurt LESS on short videos?

Same 5 prompts (cmp_prompts_5.txt), baseline (pure AF) vs read-only memory, at SHORT lengths
(60 latent ~=15s = our training length; 240 latent ~=60s = upstream training length). VBench
subject + background consistency. Combined with the already-measured 960 (240s) result -> a
length trend. If the memory's gap shrinks/reverses as videos get shorter -> length/horizon is
the cause (#2), not the untrained write (#1). Result -> /tmp/short_eval_result.txt.
"""
import os, glob, json, subprocess
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VBENCH_DIR = "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo"
OUT = "/shared/ssd_14T_project/adenelmi/e0_cmp"
RO_CKPT = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_metrics/checkpoint_model_003000/model.pt"
MEM_LAYERS = [5, 11, 17, 23, 29]
PROMPTS = os.path.join(REPO, "reference/cmp_prompts_5.txt")
GPU = os.environ.get("EVAL_GPU", "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107")  # GPU 7
LENGTHS = [60, 240]   # latent frames: 60~=15s (our train len), 240~=60s (upstream train len)
RESULT = "/tmp/short_eval_result.txt"


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def env_for(dev):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=dev, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def infer(tag, lora_ckpt, layers, out_dir, nframes, port):
    cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
    cfg.lora_ckpt = lora_ckpt
    cfg.model_kwargs.memory_layers = list(layers)
    cfg.output_folder = out_dir
    cfg.num_output_frames = nframes
    cfg.data_path = PROMPTS
    p = f"/tmp/_shorteval_{tag}.yaml"
    OmegaConf.save(cfg, p)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "inference/inference.py", "--config_path", p]
    log(f"inference {tag} ({nframes} latent) -> {out_dir}")
    subprocess.run(cmd, cwd=REPO, env=env_for(GPU),
                   stdout=open("/tmp/short_eval_infer.log", "a"),
                   stderr=subprocess.STDOUT, timeout=14400)


def vbench(videos_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for dim in ["subject_consistency", "background_consistency"]:
        cmd = [PY, "vbench2_beta_long/eval_long.py", "--videos_path", videos_dir,
               "--dimension", dim, "--mode", "long_custom_input", "--output_path", out_dir]
        log(f"vbench {dim} on {videos_dir}")
        subprocess.run(cmd, cwd=VBENCH_DIR, env=env_for(GPU),
                       stdout=open("/tmp/short_eval_eval.log", "a"),
                       stderr=subprocess.STDOUT, timeout=7200)


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
    res = ["LENGTH SWEEP A/B (subject_consistency / background_consistency), same 5 prompts", ""]
    port = 29990
    rows = []
    try:
        for L in LENGTHS:
            infer(f"baseline_L{L}", "ckpt/final.pt", [], f"{OUT}/baseline_L{L}", L, port); port += 1
            infer(f"memory_L{L}", RO_CKPT, MEM_LAYERS, f"{OUT}/memory_L{L}", L, port); port += 1
            vbench(f"{OUT}/baseline_L{L}", f"{OUT}/eval_baseline_L{L}")
            vbench(f"{OUT}/memory_L{L}", f"{OUT}/eval_memory_L{L}")
            b, m = scores(f"{OUT}/eval_baseline_L{L}"), scores(f"{OUT}/eval_memory_L{L}")
            rows.append((L, b, m))
        # reference: the already-measured 960 (240s) on the same 5 prompts (post_train eval dirs)
        b960, m960 = scores(f"{OUT}/eval_baseline"), scores(f"{OUT}/eval_memory")
        if b960 and m960:
            rows.append((960, b960, m960))
        for L, b, m in rows:
            sb, sm = b.get("subject_consistency", 0), m.get("subject_consistency", 0)
            gb, gm = b.get("background_consistency", 0), m.get("background_consistency", 0)
            res.append(f"L={L:4d} (~{round((1+4*(L-1))/16):3d}s)  subject: base={sb:.4f} mem={sm:.4f} Δ={sm-sb:+.4f}  "
                       f"| background: base={gb:.4f} mem={gm:.4f} Δ={gm-gb:+.4f}")
    except Exception as e:
        res.append(f"short_eval FAILED: {e}")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
