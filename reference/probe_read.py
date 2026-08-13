#!/usr/bin/env python
"""Read pre-gate probe (Q3 diagnostic / supervisor 'feature-maps' directive).

Captures the memory READ BEFORE the gate -- o_hat = o_norm(f_W(q)) -- at every VideoMemoryLayer
during inference, reshapes it to per-frame spatial heatmaps, and measures temporal drift.

Why pre-gate: the gate is ~closed (gate~6e-4), so mem_out ~= 0 and looking at the OUTPUT shows
nothing. o_hat is the actual recall, computed regardless of the gate. It answers:
  - o_hat has identity structure (subject lights up, stable in time) -> recall works -> the GATE
    is the right lever (the 6 gating ideas apply).
  - o_hat is noise -> the read/write is broken -> refining the gate is premature.

Run (1 GPU, longlive env, from repo root):
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python reference/probe_read.py \
      --config_path configs/memory.yaml \
      --out reference/probe_out --max_frames 45 --prompt_idx 0

NB1: needs a POST-FIX trained memory checkpoint to be meaningful (pre-fix 450/600/750 are invalid).
NB2: the config's model_kwargs.memory_layers must match the layers the checkpoint was trained with.
NB3: setup mirrors inference/inference.py -- smoke-test on a free GPU before trusting numbers.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


# ----------------------------------------------------------------------------- the reusable core
class ReadProbe:
    """Forward-hook every VideoMemoryLayer.o_norm; its output IS o_hat (pre-gate read)."""

    def __init__(self):
        self.reads = {}      # layer_idx -> list of (L, dim) cpu tensors, one per o_norm fire
        self.names = {}      # layer_idx -> module name
        self.handles = []

    def attach(self, model):
        from wan.modules.video_memory import VideoMemoryLayer
        idx = 0
        for name, m in model.named_modules():
            if isinstance(m, VideoMemoryLayer):
                li = idx
                self.names[li] = name

                def mk(li):
                    def hook(mod, inp, out):
                        # out: (b*h, L, head_dim); num_heads=1, b=1 -> (1, L, dim)
                        self.reads.setdefault(li, []).append(out.detach().float().cpu()[0])
                    return hook

                self.handles.append(m.o_norm.register_forward_hook(mk(li)))
                idx += 1
        print(f"[probe] attached to {idx} memory layer(s): {list(self.names.values())}")
        if idx == 0:
            print("[probe] WARNING: no VideoMemoryLayer found -- is MAG_MEMORY_LAYERS set?")
        return self

    def detach(self):
        for h in self.handles:
            h.remove()

    def summarize(self, outdir, hw=(30, 52)):
        os.makedirs(outdir, exist_ok=True)
        tpf = hw[0] * hw[1]                       # tokens per frame (latent grid, 2x2 patch of 60x104)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            plt = None
            print(f"[probe] matplotlib unavailable ({e}); saving .npy only")

        for li, fires in self.reads.items():
            name = self.names[li]
            # --- temporal drift: per-fire global recall vector, cosine to the first fire ---
            vecs = torch.stack([f.mean(0) for f in fires])          # (n_fires, dim)
            v0 = vecs[0]
            cos = torch.nn.functional.cosine_similarity(vecs, v0[None], dim=1).numpy()
            np.save(os.path.join(outdir, f"drift_layer{li}.npy"), cos)

            # --- spatial heatmaps at a few points across the generation ---
            picks = sorted(set([0, len(fires) // 4, len(fires) // 2,
                                3 * len(fires) // 4, len(fires) - 1]))
            maps = []
            for p in picks:
                f = fires[p]                                         # (L, dim)
                if f.shape[0] % tpf == 0:
                    nf = f.shape[0] // tpf
                    hm = f.norm(dim=1).reshape(nf, hw[0], hw[1])[0]  # first frame of that block
                    maps.append((p, hm.numpy()))

            if plt is not None:
                ncol = max(1, len(maps)) + 1
                fig, ax = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.0))
                ax[0].plot(cos); ax[0].set_title(f"L{li} read drift\n(cos to t0)")
                ax[0].set_xlabel("fire (~time)"); ax[0].set_ylim(0, 1.02)
                for j, (p, hm) in enumerate(maps):
                    a = ax[j + 1]
                    a.imshow(hm, cmap="magma"); a.set_title(f"o_hat |.| @fire {p}")
                    a.axis("off")
                fig.suptitle(f"Read pre-gate probe -- memory layer {li} ({name})", fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, f"probe_layer{li}.png"), dpi=120)
                plt.close(fig)

            print(f"[probe] layer {li}: {len(fires)} fires, "
                  f"drift cos(end,t0)={cos[-1]:.3f} (1=stable, low=drifts)")
        print(f"[probe] saved to {outdir}/  (probe_layer*.png, drift_layer*.npy)")


# ----------------------------------------------------------------------------- runner (mirrors inference.py)
def build_pipeline(config, device):
    from pipeline import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    import peft

    pipeline = CausalInferencePipeline(config, device=device)

    # generator backbone
    if config.generator_ckpt:
        sd = torch.load(config.generator_ckpt, map_location="cpu")
        raw = sd.get("generator_ema" if getattr(config, "use_ema", False) else "generator",
                     sd.get("model"))
        missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
        non_mem = [k for k in missing if "memory" not in k]
        assert not non_mem, f"Missing non-memory keys: {non_mem[:5]}"
        assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"

    # LoRA + MAG memory
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model, model_name="generator",
            lora_config=config.adapter, is_main_process=True)
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            lck = torch.load(lora_ckpt_path, map_location="cpu")
            peft.set_peft_model_state_dict(
                pipeline.generator.model,
                lck["generator_lora"] if isinstance(lck, dict) and "generator_lora" in lck else lck)
            if isinstance(lck, dict) and "generator_memory" in lck:
                res = pipeline.generator.load_state_dict(lck["generator_memory"], strict=False)
                assert not res.unexpected_keys, f"Unexpected memory keys: {list(res.unexpected_keys)[:5]}"
                still = [k for k in res.missing_keys if "memory" in k]
                assert not still, f"Memory NOT loaded (stays at init!): {still[:5]}"
                print(f"[MAG] loaded {len(lck['generator_memory'])} trained memory keys")
            else:
                print("[MAG] WARNING: no 'generator_memory' in lora_ckpt -> memory is at INIT "
                      "(probe will show untrained recall)")
        pipeline.is_lora_enabled = True

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--out", default="reference/probe_out")
    ap.add_argument("--max_frames", type=int, default=45, help="latent frames to generate (short)")
    ap.add_argument("--prompt_idx", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="override prompt text")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from utils.dataset import TextDataset
    from utils.misc import set_seed

    config = OmegaConf.load(args.config_path)
    config.distributed = False
    device = torch.device("cuda")
    set_seed(getattr(config, "seed", 0))
    torch.set_grad_enabled(False)

    pipeline = build_pipeline(config, device)

    # prompt
    if args.prompt is not None:
        prompt = args.prompt
    else:
        ds = TextDataset(prompt_path=config.data_path, extended_prompt_path=config.data_path)
        b = ds[args.prompt_idx]
        prompt = (b.get("extended_prompts") or b.get("prompts"))
        prompt = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
    print(f"[probe] prompt: {prompt}")

    probe = ReadProbe().attach(pipeline.generator.model)

    noise = torch.randn([1, args.max_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                       low_memory=True, profile=False)

    probe.detach()
    probe.summarize(args.out)


if __name__ == "__main__":
    main()
