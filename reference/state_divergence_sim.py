# Standalone simulation of the VideoMemoryLayer inference write chain.
# Loads video_memory.py directly (no wan package init -> no CUDA touch), CPU only.
# Question: under the renorm-to-init Muon write, what happens to the fast-weight STATE
# over a 240-frame (80-write) chain vs the ~7-20 writes seen in training?
import importlib.util
import math
import torch

spec = importlib.util.spec_from_file_location(
    "vm", "/shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/video_memory.py")
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)

torch.manual_seed(0)

# Small but faithful geometry: di < dh like the real layer (1536 -> 6144, ratio 4)
di, dh = 96, 384
B = 1
L = 512            # tokens per write (real: 3 frames * 1560 = 4680)
base_lr = 1e-2
muon_steps = 5
N_WRITES = 240

w0 = torch.randn(B, di, dh) / math.sqrt(di)
w1 = torch.randn(B, dh, di) / math.sqrt(dh)
w2 = torch.randn(B, di, dh) / math.sqrt(di)
w0_init, w1_init, w2_init = w0.clone(), w1.clone(), w2.clone()

# fixed probe query to watch the READ drift
q_probe = torch.randn(B, 64, di)
o_init = vm.swiglu_apply(q_probe, w0, w1, w2)

# video-like k/v: slowly drifting latent content (correlated across writes)
kbase = torch.randn(B, L, di)
vbase = torch.randn(B, L, di)

def cos_to_init(w, wi):
    return torch.nn.functional.cosine_similarity(
        w.flatten(1), wi.flatten(1), dim=1).item()

def specstats(w):
    s = torch.linalg.svdvals(w[0].float())
    erank = (s.sum() ** 2 / (s ** 2).sum()).item()   # participation-ratio effective rank
    return erank, (s[0] / s.sum()).item()

print(f"{'write':>5} {'cos(w0,init)':>12} {'cos(w1,init)':>12} {'|w1|col':>8} "
      f"{'erank(w1)':>9} {'top-sv%':>8} {'|read|/|read0|':>14} {'step_rot_deg(w1)':>16}")

prev_w1 = w1.clone()
for t in range(1, N_WRITES + 1):
    # drift the content slowly (adjacent-frame correlation ~0.995)
    kbase = 0.995 * kbase + 0.1 * torch.randn_like(kbase)
    vbase = 0.995 * vbase + 0.1 * torch.randn_like(vbase)
    k = kbase / kbase.norm(dim=-1, keepdim=True) * math.sqrt(di)  # RMS-normed like k_norm
    v = vbase
    lr = torch.full((B, L, 1), base_lr)
    g0, g1, g2 = vm.swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr, lr, lr)
    w0, w1, w2 = vm.weight_norm_update(w0, w1, w2, g0, g1, g2, muon_steps)

    if t in (1, 3, 7, 14, 20, 40, 80, 160, 240):
        # per-step rotation of a column of w1
        c = torch.nn.functional.cosine_similarity(w1[0, :, 0], prev_w1[0, :, 0], dim=0).clamp(-1, 1)
        rot = math.degrees(math.acos(c.item()))
        o = vm.swiglu_apply(q_probe, w0, w1, w2)
        er, tsv = specstats(w1)
        print(f"{t:>5} {cos_to_init(w0, w0_init):>12.3f} {cos_to_init(w1, w1_init):>12.3f} "
              f"{w1[0].norm(dim=0).mean().item():>8.3f} {er:>9.1f} {tsv * 100:>7.1f}% "
              f"{(o.norm() / o_init.norm()).item():>14.3f} {rot:>16.1f}")
    prev_w1 = w1.clone()

# --- key property: is the write step size independent of the gradient magnitude? ---
print("\nMuon erases write magnitude (renorm path): state change per write vs v-scale")
for scale in (1.0, 1e-2, 1e-4, 1e-6):
    wa, wb, wc = w0_init.clone(), w1_init.clone(), w2_init.clone()
    k = torch.randn(B, L, di); k = k / k.norm(dim=-1, keepdim=True) * math.sqrt(di)
    v = torch.randn(B, L, di) * scale
    lr = torch.full((B, L, 1), base_lr)
    g0, g1, g2 = vm.swiglu_fast_weight_gradient(k, v, wa, wb, wc, lr, lr, lr)
    na, nb, nc = vm.weight_norm_update(wa, wb, wc, g0, g1, g2, muon_steps)
    d = (nb - wb).norm() / wb.norm()
    print(f"  |v| scale {scale:>8.0e}  ->  relative state change of w1 per ONE write: {d.item():.3f}")
