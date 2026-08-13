#!/usr/bin/env python
"""Autonomous A/B eval for the combo (gating perso 6+5+2) run.

Waits until the combo training (next_run_5L_combo652.yaml) finishes, then runs the A/B eval
(baseline pure-AF vs the trained combo memory) on the 5 comparison prompts, 240s each, on one
free GPU. VBench (subject + background consistency). Result -> /tmp/combo_eval_result.txt.

The combo checkpoint carries gate_extra modules, so the inference config sets the matching
gate_extra=[agreement, frames_since_switch, denoising_step] (signals are available at inference:
denoising_step from t, frames_since_switch from the frame index, agreement from the prompt).
"""
import os, time, glob, json, subprocess
from omegaconf import OmegaConf

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VBENCH_DIR = "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo"
OUT = "/shared/ssd_14T_project/adenelmi/e0_cmp_combo"
COMBO_TAG = "next_run_5L_combo652.yaml"
CKPT_DIR = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_combo652"
MEM_LAYERS = [5, 11, 17, 23, 29]
GATE_EXTRA = ["agreement", "frames_since_switch", "denoising_step"]
PROMPTS = os.path.join(REPO, "reference/cmp_prompts_5.txt")
BROKEN = "GPU-ab1015a0-7c1d-77c9-fbdf-738bbf32f3eb"
RESULT = "/tmp/combo_eval_result.txt"
FREE_MEM, FREE_UTIL, FREE_SECS, POLL = 2000, 15, 60, 15


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def combo_alive():
    out = subprocess.run(["pgrep", "-f", COMBO_TAG], capture_output=True, text=True).stdout
    for pid in out.split():
        try:
            if open(f"/proc/{pid}/comm").read().strip() == "python":
                return True
        except Exception:
            pass
    return False


def one_free_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    seen = {}
    while True:
        now = time.time()
        cur = set()
        for line in out.strip().splitlines():
            try:
                u, m, ut = [c.strip() for c in line.split(",")]
                if u != BROKEN and int(m) < FREE_MEM and int(ut) < FREE_UTIL:
                    cur.add(u)
            except ValueError:
                pass
        for u in list(seen):
            if u not in cur:
                del seen[u]
        for u in cur:
            seen.setdefault(u, now)
        stable = [u for u, t in seen.items() if now - t >= FREE_SECS]
        if stable:
            return stable[0]
        time.sleep(POLL)
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout


def env_for(dev):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=dev, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def infer(tag, lora_ckpt, layers, gate_extra, out_dir, gpu, port):
    cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
    cfg.lora_ckpt = lora_ckpt
    cfg.model_kwargs.memory_layers = list(layers)
    cfg.model_kwargs.gate_extra = list(gate_extra)
    cfg.output_folder = out_dir
    cfg.num_output_frames = 960
    cfg.data_path = PROMPTS
    p = f"/tmp/_comboeval_{tag}.yaml"
    OmegaConf.save(cfg, p)
    cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master_port={port}",
           "inference/inference.py", "--config_path", p]
    log(f"inference {tag} -> {out_dir} on {gpu[:12]}")
    rc = subprocess.run(cmd, cwd=REPO, env=env_for(gpu),
                        stdout=open("/tmp/combo_eval_infer.log", "a"),
                        stderr=subprocess.STDOUT, timeout=28800).returncode
    log(f"inference {tag} rc={rc}")


def vbench(videos_dir, out_dir, gpu):
    os.makedirs(out_dir, exist_ok=True)
    for dim in ["subject_consistency", "background_consistency"]:
        cmd = [PY, "vbench2_beta_long/eval_long.py", "--videos_path", videos_dir,
               "--dimension", dim, "--mode", "long_custom_input", "--output_path", out_dir]
        log(f"vbench {dim} on {videos_dir}")
        subprocess.run(cmd, cwd=VBENCH_DIR, env=env_for(gpu),
                       stdout=open("/tmp/combo_eval_eval.log", "a"),
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
    log("combo-eval up; waiting for the combo (gating perso) run to finish...")
    while combo_alive():
        time.sleep(60)
    log("combo training finished.")
    cks = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint_model_*")))
    ck = cks[-1] if cks else None
    res.append(f"combo final checkpoint: {ck}")
    try:
        if not ck:
            raise RuntimeError("no combo checkpoint found")
        gpu = one_free_gpu()
        res.append(f"using GPU {gpu[:12]}")
        infer("baseline", "ckpt/final.pt", [], [], f"{OUT}/baseline", gpu, 29980)
        infer("combo", os.path.join(ck, "model.pt"), MEM_LAYERS, GATE_EXTRA, f"{OUT}/combo", gpu, 29981)
        vbench(f"{OUT}/baseline", f"{OUT}/eval_baseline", gpu)
        vbench(f"{OUT}/combo", f"{OUT}/eval_combo", gpu)
        sb, sc = scores(f"{OUT}/eval_baseline"), scores(f"{OUT}/eval_combo")
        res.append(f"BASELINE: {sb}")
        res.append(f"COMBO:    {sc}")
    except Exception as e:
        res.append(f"combo A/B + eval FAILED: {e}")
    open(RESULT, "w").write("\n".join(res) + "\n")
    log("DONE:\n" + "\n".join(res))


if __name__ == "__main__":
    main()
