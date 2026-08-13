import sys, torch
sys.path.insert(0, ".")
from wan.modules.video_memory import VideoMemoryLayer

torch.manual_seed(0)
dim, B, L, Lc = 64, 2, 12, 5

# 1) combo layer with all 3 signals -> proj_extra must be zero-init (do-no-harm)
combo = VideoMemoryLayer(dim=dim, num_heads=1, g_init=0.1,
                         gate_extra=["agreement", "frames_since_switch", "denoising_step"]).eval()
assert combo.proj_extra.weight.abs().sum().item() == 0.0, "proj_extra not zero-init!"
print("proj_extra zero-init OK; extra_dim =", combo.proj_extra.weight.shape[1], "(expect 33=1+16+16)")

x = torch.randn(B, L, dim)
pe = torch.randn(B, Lc, dim)
ds = torch.randint(0, 1000, (B, L)).float()
fss = torch.arange(L).float()[None, :].expand(B, L).contiguous()

out, st = combo.step(x, prompt_emb=pe, denoising_step=ds, frames_since_switch=fss)
print("combo step out:", tuple(out.shape), "agreement mean:", round(float(combo.dbg_agreement), 4))
assert out.shape == (B, L, dim)

# 2) do-no-harm: base layer (no extra) sharing the same weights -> identical output at init
base = VideoMemoryLayer(dim=dim, num_heads=1, g_init=0.1, gate_extra=[]).eval()
missing, unexpected = base.load_state_dict(combo.state_dict(), strict=False)
out_b, _ = base.step(x, prompt_emb=pe)
diff = (out - out_b).abs().max().item()
print("do-no-harm max|combo-base| =", diff)
assert diff < 1e-5, f"NOT do-no-harm: {diff}"

# 3) guard: a declared signal that arrives None must raise (so the block MUST wire it)
for kw in [dict(denoising_step=None, frames_since_switch=fss),
           dict(denoising_step=ds, frames_since_switch=None)]:
    try:
        combo.step(x, prompt_emb=pe, **kw)
        raise SystemExit("guard missing!")
    except ValueError:
        pass
print("missing-signal guards OK")

# 4) frames_since_switch sawtooth (mirror of the block logic) -- resets to 0 at the switch
frame_seqlen, num_frames = 4, 3
exp = {(0, 21): [0, 1, 2], (21, 21): [0, 1, 2], (24, 21): [3, 4, 5]}
for (cs_frame, swg), e in exp.items():
    current_start = cs_frame * frame_seqlen
    csf = current_start // frame_seqlen
    gf = csf + torch.arange(num_frames)
    val = torch.where(gf >= swg, gf - swg, gf).tolist()
    assert val == e, f"{(cs_frame, swg)}: {val} != {e}"
    print(f"chunk_start_frame={cs_frame:2d} switch={swg} -> fss {val}")

print("ALL TESTS PASSED")
