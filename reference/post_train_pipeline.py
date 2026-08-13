#!/usr/bin/env python
"""Autonomous post-training pipeline (read-only run e1_ro_5L_metrics).

On training completion:
  1. A/B inference: baseline (pure AF) vs trained memory, same prompts/seed.
  2. VBench eval (subject_consistency + background_consistency) on both sets.
  3. Launch the gating (Idea-6) training on GPU 1+7.

The gating launch is GUARANTEED (try/finally) -> it runs even if A/B/eval fail,
so we never waste the freed GPUs. Result -> /tmp/post_train_result.txt.
NB: A/B inference + VBench eval are UNVALIDATED on the memory checkpoint; if they
fail, the result file says so and the gating still launches (then fix eval by hand).
"""
import os, time, glob, json, subprocess
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
TRAIN_TAG = "_ro_train_5L_metrics.yaml"     # current run's config -> detect end
CKPT_DIR = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_metrics"
VBENCH_DIR = "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo"
OUT = "/shared/ssd_14T_project/adenelmi/e0_cmp"
G1 = "GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57"
G7 = "GPU-df09dc6f-c851-343b-7a73-73f3f34b0107"
MEM_LAYERS = [5, 11, 17, 23, 29]
PROMPTS = os.path.join(REPO, "reference/cmp_prompts_5.txt")   # 5 comparison prompts -> 5 videos / condition
RESULT = "/tmp/post_train_result.txt"
GATING_FLAG = "/tmp/gating_launched.flag"   # set once the gating run is launched (here or opportunistically)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def training_alive():
    out = subprocess.run(["pgrep", "-f", TRAIN_TAG], capture_output=True, text=True).stdout
    for pid in out.split():
        try:
            if open(f"/proc/{pid}/comm").read().strip() == "python":
                return True
        except Exception:
            pass
    return False


def env_for(devs):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=devs, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def infer(tag, lora_ckpt, layers, out_dir, port):
    cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
    cfg.lora_ckpt = lora_ckpt
    cfg.model_kwargs.memory_layers = list(layers)
    cfg.output_folder = out_dir
    cfg.num_output_frames = 960   # ~240s (long video -> identity drift shows); divisible by num_frame_per_block=3
    cfg.data_path = PROMPTS       # 5 prompts -> 5 videos for this condition
    p = f"/tmp/_infer_{tag}.yaml"
    OmegaConf.save(cfg, p)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "inference/inference.py", "--config_path", p]
    log(f"inference {tag} -> {out_dir}")
    rc = subprocess.run(cmd, cwd=REPO, env=env_for(G1),
                        stdout=open("/tmp/post_train_infer.log", "a"),
                        stderr=subprocess.STDOUT, timeout=28800).returncode
    log(f"inference {tag} rc={rc}")
    return rc


def vbench(videos_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for dim in ["subject_consistency", "background_consistency"]:
        cmd = [PY, "vbench2_beta_long/eval_long.py", "--videos_path", videos_dir,
               "--dimension", dim, "--mode", "long_custom_input", "--output_path", out_dir]
        log(f"vbench {dim} on {videos_dir}")
        subprocess.run(cmd, cwd=VBENCH_DIR, env=env_for(G1),
                       stdout=open("/tmp/post_train_eval.log", "a"),
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


def launch_gating():
    if os.path.exists(GATING_FLAG):
        return None   # already launched opportunistically on other GPUs -> don't double-launch
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--max-restarts=0",
           "--master_port=29950", "train.py", "--config_path",
           os.path.join(REPO, "reference/next_run_5L_agreement.yaml"),
           "--no-auto-resume", "--no-one-logger"]
    e = env_for(f"{G1},{G7}"); e["NCCL_P2P_DISABLE"] = "1"
    p = subprocess.Popen(cmd, cwd=REPO, env=e, stdout=open("/tmp/gating_train.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    open(GATING_FLAG, "w").write(f"pid={p.pid} gpus={G1},{G7} run=e1_ro_5L_agreement (post-train, GPU1+7)\n")
    return p.pid


def main():
    res = []
    log("post-train pipeline up; waiting for the read-only run to finish...")
    while training_alive():
        time.sleep(60)
    log("read-only training finished.")
    cks = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint_model_*")))
    ck = cks[-1] if cks else None
    res.append(f"final checkpoint: {ck}")
    try:
        if not ck:
            raise RuntimeError("no checkpoint found")
        model = os.path.join(ck, "model.pt")
        infer("baseline", "ckpt/final.pt", [], f"{OUT}/baseline", 29960)
        infer("memory", model, MEM_LAYERS, f"{OUT}/memory", 29961)
        vbench(f"{OUT}/baseline", f"{OUT}/eval_baseline")
        vbench(f"{OUT}/memory", f"{OUT}/eval_memory")
        sb, sm = scores(f"{OUT}/eval_baseline"), scores(f"{OUT}/eval_memory")
        res.append(f"BASELINE: {sb}")
        res.append(f"MEMORY:   {sm}")
    except Exception as e:
        res.append(f"A/B + eval FAILED ({e}) -- gating still launching, fix eval by hand")
    finally:
        try:
            pid = launch_gating()
            if pid is None:
                res.append("GATING already launched opportunistically on other GPUs -- skipped relaunch")
            else:
                res.append(f"GATING training LAUNCHED pid={pid} run=e1_ro_5L_agreement (gate_extra=[agreement])")
        except Exception as e:
            res.append(f"GATING launch FAILED: {e}")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
