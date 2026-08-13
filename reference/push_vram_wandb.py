#!/usr/bin/env python
"""Push the VRAM-profiling results to W&B as a static run (wandb was off at run time).
Creates run 'vram_feasibility' in elmi_aden-harvard-university/anchor-forcing-memory:
a results table + the 2-GPU read+write deficit-convergence line + summary scalars.
Run: <longlive>/python reference/push_vram_wandb.py
"""
import math
import wandb

ENTITY, PROJECT = "elmi_aden-harvard-university", "anchor-forcing-memory"
NA = float("nan")

run = wandb.init(
    entity=ENTITY, project=PROJECT, name="vram_feasibility", id="vram_feasibility",
    resume="allow", job_type="profiling",
    config={
        "gpu_capacity_GiB": 47.37,
        "switch_latent": 21,
        "note": "Static summary of the VRAM profiler runs (wandb was disabled at run time; "
                "only peak alloc/reserved exist, no live memory-time curve).",
    },
)

# --- full results table ---
cols = ["gpus", "mode", "layers", "window_latent", "result",
        "alloc_GiB", "reserved_GiB", "deficit_MiB", "note"]
rows = [
    [1, "read",       1, 42, "FITS", 44.73, 46.39, 0.0, "layer 5"],
    [1, "read",       2, 42, "FITS", 45.42, 46.39, 0.0, "layers 5,10"],
    [1, "read",       3, 42, "OOM",  NA,    NA,    NA,  "layers 5,10,15"],
    [1, "read+write", 1, 42, "OOM",  NA,    NA,    NA,  "no FSDP -> needs >1 GPU"],
    [2, "read+write", 1, 42, "OOM",  NA,    NA,    560.0, "FSDP full-shard only"],
    [2, "read+write", 1, 42, "OOM",  NA,    NA,    166.0, "+ checkpoint-write"],
    [2, "read+write", 1, 33, "OOM",  NA,    NA,    80.0,  "+ window 33"],
    [2, "read+write", 1, 27, "OOM",  NA,    NA,    42.0,  "+ window 27 (fragmentation)"],
    [2, "read+write", 1, 24, "FITS", 45.22, 46.61, 0.0,  "+ expandable_segments -> PASS (~5.8s)"],
]
run.log({"vram_results": wandb.Table(columns=cols, data=rows)})

# --- 2-GPU read+write: deficit shrinking to 0 (plots as a line) ---
conv = [("FSDP only", 560), ("+ckpt-write", 166), ("+window33", 80),
        ("+window27", 42), ("+expand/window24", 0)]
for i, (lbl, mib) in enumerate(conv):
    run.log({"readwrite_2gpu/deficit_MiB": mib, "readwrite_2gpu/reduction_step": i}, step=i)

# --- 1-GPU read-only peak alloc per layer count (bar) ---
try:
    t = wandb.Table(data=[[1, 44.73], [2, 45.42], [3, float("nan")]],
                    columns=["layers", "alloc_GiB"])
    run.log({"read_1gpu_alloc": wandb.plot.bar(t, "layers", "alloc_GiB",
                                                title="1-GPU read-only peak alloc (GiB), 3L=OOM")})
except Exception as e:
    print("bar plot skipped:", e)

run.summary["max_clip_latent_2gpu_readwrite"] = 24
run.summary["max_window_seconds"] = 5.8
run.summary["read_only_1gpu_max_layers"] = 2
run.summary["readwrite_fits_min_gpus"] = 2
run.summary["readwrite_pass_alloc_GiB"] = 45.22
run.summary["readwrite_pass_reserved_GiB"] = 46.61

print("RUN URL:", run.url)
run.finish()
