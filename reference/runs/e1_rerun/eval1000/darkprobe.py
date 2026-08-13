# Fable darkening probe: quantify the MAG injection magnitude vs the residual stream at
# inference, per chunk and per denoising pass, on the step-1000 e1_rerun2 checkpoint.
# Usage: CUDA_VISIBLE_DEVICES=<uuid> python darkprobe.py [--frames 12] [--memoff]
import argparse, os, sys, json
import torch
from omegaconf import OmegaConf

sys.path.append("/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=12)   # latent frames; %3==0
ap.add_argument("--memoff", action="store_true")
ap.add_argument("--out", type=str, default="/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e1_rerun/eval1000/darkprobe_stats.json")
args = ap.parse_args()

ED = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e1_rerun/eval1000"
config = OmegaConf.load(f"{ED}/memon.yaml")
device = torch.device("cuda")
set_seed(config.seed)
config.distributed = False
torch.set_grad_enabled(False)

if args.memoff:
    os.environ["MEM_ZERO_ALL"] = "1"

pipeline = CausalInferencePipeline(config, device=device)

# --- load base ckpt (same as inference.py non-EMA branch) ---
state_dict = torch.load(config.generator_ckpt, map_location="cpu")
raw = state_dict["generator" if "generator" in state_dict else "model"]
missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
assert not [k for k in missing if "memory" not in k]

# --- LoRA + memory ---
from utils.lora_utils import configure_lora_for_model
import peft
pipeline.generator.model = configure_lora_for_model(
    pipeline.generator.model, model_name="generator", lora_config=config.adapter, is_main_process=True)
lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu")
peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
mem_sd = lora_checkpoint["generator_memory"]
result = pipeline.generator.load_state_dict(mem_sd, strict=False)
assert not result.unexpected_keys
assert not [k for k in result.missing_keys if "memory" in k]
print(f"[probe] {len(mem_sd)} trained memory keys loaded")

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.text_encoder.to(device=device)
pipeline.vae.to(device=device)

# ---------------- instrumentation ----------------
from wan.modules.video_memory import VideoMemoryLayer

STATS = []           # one record per memory _project_output call
CTX = {"chunk": -1, "pass": -1, "t": -1}
_orig_proj = VideoMemoryLayer._project_output
_orig_step = VideoMemoryLayer.step

def patched_project(self, x, batch_size, prompt_emb=None, frames_since_switch=None, denoising_step=None):
    # replicate forward to capture internals
    xn = x if self.single_norm else self.o_norm(x)
    from einops import rearrange
    xr = rearrange(xn, '(b h) l d -> b l (h d)', h=self.num_heads, b=batch_size)
    shift, scale, gate = self.proj(xr).chunk(3, dim=-1)
    if self.bounded_gate:
        gate_t = torch.tanh(gate)
    else:
        gate_t = gate
    if self.bounded_mod:
        scale_t, shift_t = torch.tanh(scale), torch.tanh(shift)
    else:
        scale_t, shift_t = scale, shift
    carrier = self.a_norm(xr) * (1 + scale_t) + shift_t
    out = gate_t * carrier
    f = lambda t: t.detach().float()
    o = f(out)
    dc = o.mean(dim=1)                       # (B, d) mean over tokens = the constant component
    rec = dict(
        block=getattr(self, "_probe_block", -1),
        chunk=CTX["chunk"], pass_idx=CTX["pass"], t=CTX["t"],
        read_rms=float(f(x).pow(2).mean().sqrt()),
        gate_pre_absmean=float(f(gate).abs().mean()),
        gate_absmean=float(f(gate_t).abs().mean()),
        gate_mean=float(f(gate_t).mean()),
        gate_sat=float((f(gate_t).abs() > 0.9).float().mean()),
        scale_absmean=float(f(scale_t).abs().mean()),
        shift_rms=float(f(shift_t).pow(2).mean().sqrt()),
        carrier_rms=float(f(carrier).pow(2).mean().sqrt()),
        out_rms=float(o.pow(2).mean().sqrt()),
        out_dc_rms=float(dc.pow(2).mean().sqrt()),   # RMS of the token-mean vector
        dc_frac=float(dc.pow(2).mean().sqrt() / (o.pow(2).mean().sqrt() + 1e-9)),
    )
    STATS.append(rec)
    return out

VideoMemoryLayer._project_output = patched_project

# stream RMS at block entry for the memory blocks
STREAM = []
mem_blocks = [int(x) for x in config.model_kwargs.memory_layers]
def mk_hook(bi):
    def hook(mod, inp, kwargs):
        x = inp[0] if inp else kwargs.get("x")
        if torch.is_tensor(x):
            STREAM.append(dict(block=bi, chunk=CTX["chunk"], pass_idx=CTX["pass"], t=CTX["t"],
                               stream_rms=float(x.detach().float().pow(2).mean().sqrt())))
    return hook

blocks = pipeline.generator.model.base_model.model.blocks
for bi in mem_blocks:
    blocks[bi].register_forward_pre_hook(mk_hook(bi), with_kwargs=True)
    if blocks[bi].memory is not None:
        blocks[bi].memory._probe_block = bi

# tag chunk / denoising-pass via generator wrapper
_orig_gen_fwd = pipeline.generator.forward
def gen_fwd(*a, **kw):
    ts = kw.get("timestep")
    CTX["t"] = int(ts.flatten()[0].item()) if torch.is_tensor(ts) else -1
    CTX["pass"] += 1
    return _orig_gen_fwd(*a, **kw)
pipeline.generator.forward = gen_fwd

# chunk counter: patch the pipeline loop is intrusive; infer chunk from pass count
# (5 generator calls per chunk: 4 denoise + 1 clean write)
class ChunkTick:
    pass

# run
prompts = [l.strip() for l in open(config.data_path) if l.strip()]
prompt = prompts[0].split("|")[0] if "|" in prompts[0] else prompts[0]
print(f"[probe] prompt: {prompt[:80]}")
noise = torch.randn([1, args.frames, 16, 60, 104], device=device, dtype=torch.bfloat16)

# wrap inference to advance chunk index: 5 passes per chunk
_orig_sched_gen = pipeline.generator.forward
def gen_fwd2(*a, **kw):
    CTX["chunk"] = CTX["pass"] // 5   # 4 denoise + 1 write per chunk
    return _orig_sched_gen(*a, **kw)
pipeline.generator.forward = gen_fwd2

video = pipeline.inference(noise=noise, text_prompts=[prompt], low_memory=False)
v = video[0] if isinstance(video, (list, tuple)) else video
print(f"[probe] video shape {v.shape}, mean luma proxy {float(v.float().mean())*255:.1f}")

# luma per pixel-frame (video normalized [0,1], (B?,T,C,H,W) or (T,C,H,W))
vv = v if v.dim() == 4 else v[0]
luma = (0.299*vv[:,0] + 0.587*vv[:,1] + 0.114*vv[:,2]).float().mean(dim=(1,2)) * 255
print("[probe] per-frame luma (first 12):", [round(float(x),1) for x in luma[:12]])
print("[probe] per-frame luma (last 12):", [round(float(x),1) for x in luma[-12:]])

json.dump(dict(stats=STATS, stream=STREAM,
               luma=[float(x) for x in luma],
               memoff=bool(args.memoff)),
          open(args.out if not args.memoff else args.out.replace(".json", "_memoff.json"), "w"), indent=1)

# ---- summary table ----
import collections
print("\nblock chunk pass t | stream_rms | out_rms  gate|.| sat  dcfrac carrier read_rms")
sm = {(r["block"], r["chunk"], r["pass_idx"]): r["stream_rms"] for r in STREAM}
for r in STATS:
    srms = sm.get((r["block"], r["chunk"], r["pass_idx"]), float("nan"))
    print(f'{r["block"]:>3}  {r["chunk"]:>3}  {r["pass_idx"]:>4} {r["t"]:>4} | {srms:9.3f} | {r["out_rms"]:7.4f} {r["gate_absmean"]:6.3f} {r["gate_sat"]:5.2f} {r["dc_frac"]:6.3f} {r["carrier_rms"]:7.3f} {r["read_rms"]:9.2f}')
print("\nDONE")
