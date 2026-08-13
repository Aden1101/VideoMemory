#!/usr/bin/env python
"""Extend the read-only A/B eval from 5 to 10 prompts (adds 5 NEW prompts), on one GPU.

Generates 5 NEW baseline + 5 NEW memory videos (240s), VBench-scores them, then combines with
the existing 5-prompt scores -> robust 10-prompt A/B. Mean of two equal-size 5-sets = the 10-set
mean. Result -> /tmp/eval10_result.txt.
"""
import os, glob, json, subprocess
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VBENCH_DIR = "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo"
OUT = "/shared/ssd_14T_project/adenelmi/e0_cmp"
RO_CKPT = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_metrics/checkpoint_model_003000/model.pt"
MEM_LAYERS = [5, 11, 17, 23, 29]
PROMPTS_B = os.path.join(REPO, "reference/cmp_prompts_5b.txt")
GPU = os.environ.get("EVAL_GPU", "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107")  # GPU 7
RESULT = "/tmp/eval10_result.txt"


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def env_for(dev):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=dev, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def infer(tag, lora_ckpt, layers, out_dir, port):
    cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
    cfg.lora_ckpt = lora_ckpt
    cfg.model_kwargs.memory_layers = list(layers)
    cfg.output_folder = out_dir
    cfg.num_output_frames = 960
    cfg.data_path = PROMPTS_B
    p = f"/tmp/_evalext_{tag}.yaml"
    OmegaConf.save(cfg, p)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "inference/inference.py", "--config_path", p]
    log(f"inference {tag} -> {out_dir}")
    rc = subprocess.run(cmd, cwd=REPO, env=env_for(GPU),
                        stdout=open("/tmp/eval_extend_infer.log", "a"),
                        stderr=subprocess.STDOUT, timeout=28800).returncode
    log(f"inference {tag} rc={rc}")


def vbench(videos_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for dim in ["subject_consistency", "background_consistency"]:
        cmd = [PY, "vbench2_beta_long/eval_long.py", "--videos_path", videos_dir,
               "--dimension", dim, "--mode", "long_custom_input", "--output_path", out_dir]
        log(f"vbench {dim} on {videos_dir}")
        subprocess.run(cmd, cwd=VBENCH_DIR, env=env_for(GPU),
                       stdout=open("/tmp/eval_extend_eval.log", "a"),
                       stderr=subprocess.STDOUT, timeout=10800)


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
    res = []
    try:
        infer("baseline_b", "ckpt/final.pt", [], f"{OUT}/baseline_b", 29970)
        infer("memory_b", RO_CKPT, MEM_LAYERS, f"{OUT}/memory_b", 29971)
        vbench(f"{OUT}/baseline_b", f"{OUT}/eval_baseline_b")
        vbench(f"{OUT}/memory_b", f"{OUT}/eval_memory_b")
        new_b, new_m = scores(f"{OUT}/eval_baseline_b"), scores(f"{OUT}/eval_memory_b")
        old_b, old_m = scores(f"{OUT}/eval_baseline"), scores(f"{OUT}/eval_memory")
        res.append(f"OLD-5  BASELINE: {old_b}")
        res.append(f"OLD-5  MEMORY:   {old_m}")
        res.append(f"NEW-5  BASELINE: {new_b}")
        res.append(f"NEW-5  MEMORY:   {new_m}")

        def comb(a, b):
            ks = set(a) & set(b)
            return {k: round((a[k] + b[k]) / 2, 4) for k in ks}
        res.append(f"COMBINED-10 BASELINE: {comb(old_b, new_b)}")
        res.append(f"COMBINED-10 MEMORY:   {comb(old_m, new_m)}")
    except Exception as e:
        res.append(f"eval_extend FAILED: {e}")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
