"""G0 positive control (Level A, v2): can the fast-weight memory STORE an associative pair
and RETRIEVE it past the KV-eviction gap, in a vacuum?

Proper associative-memory test: write <k_A, v_A> (the write ascends <f_W(k), v>), evict with
N distractor writes, then READ AT THE KEY  r = swiglu_apply(k_A, state)  (bypasses the untrained
query projection so we test the substrate, not qkv training), and measure per-token cos(r, v_A)
vs a shuffled-value chance baseline. Retrieval = align - chance.

If retrieval collapses to chance well before the eviction gap (13 writes), even with favorable
knobs, the Muon+renorm substrate cannot hold content across eviction -> mechanism dead at storage.

Run: CUDA_VISIBLE_DEVICES=0 python reference/e1_positive_control.py
"""
import sys, torch
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
from wan.modules.video_memory import VideoMemoryLayer, swiglu_apply
import torch.nn.functional as F
import statistics as st

DEV = "cuda"; DIM = 1536; L = 390; EVICT = 13; SEEDS = 6
BF16 = torch.bfloat16


def make_mem(base_lr, muon):
    m = VideoMemoryLayer(dim=DIM, num_heads=1, base_lr=base_lr, muon_update_steps=muon,
                         bounded_gate=True, single_norm=False, use_decay=False, g_init=0.0)
    m = m.to(DEV).to(BF16).eval(); m.detach_state = True
    return m


@torch.no_grad()
def trial(seed, base_lr, muon, repeat, evict):
    m = make_mem(base_lr, muon)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rnd = lambda: torch.randn(1, L, DIM, generator=g, device=DEV).to(BF16)
    xA = rnd(); D = [rnd() for _ in range(evict)]
    q, k, v, l0, l1, l2, B = m._qkv_lr(xA)          # k_A, v_A actually written for xA
    v0 = v.float(); perm = torch.randperm(L, device=DEV)

    s = m.init_state(1, DEV, BF16)
    for _ in range(repeat):
        _, s = m.step(xA, state=s, do_update=True)  # write <k_A, v_A> (reinforced x repeat)
    r0 = swiglu_apply(k, *s).float()                # read at the key, fresh
    align0 = F.cosine_similarity(r0, v0, dim=-1).mean().item()
    chance0 = F.cosine_similarity(r0, v0[:, perm], dim=-1).mean().item()

    for d in D:                                     # eviction gap: N distractor writes
        _, s = m.step(d, state=s, do_update=True)
    rE = swiglu_apply(k, *s).float()                # read at the SAME key, post-eviction
    alignE = F.cosine_similarity(rE, v0, dim=-1).mean().item()
    chanceE = F.cosine_similarity(rE, v0[:, perm], dim=-1).mean().item()
    return align0, chance0, alignE, chanceE


def sweep(name, base_lr, muon, repeat):
    print(f"\n[{name}] base_lr={base_lr} muon={muon} repeat={repeat}")
    print(f"  {'gap':>4} {'align':>8} {'chance':>8} {'retrieval':>10}  verdict")
    passed = {}
    for evict in [0, 1, 2, 3, 5, 8, 13, 20]:
        rows = [trial(s, base_lr, muon, repeat, evict) for s in range(SEEDS)]
        aE = st.mean(r[2] for r in rows); cE = st.mean(r[3] for r in rows)
        cEsd = st.pstdev([r[3] for r in rows]) or 1e-6
        retr = aE - cE
        ok = (aE > cE + 3 * cEsd) and (retr > 0.02)
        passed[evict] = ok
        print(f"  {evict:>4} {aE:>8.3f} {cE:>8.3f} {retr:>10.3f}  {'RETRIEVES' if ok else 'chance (gone)'}")
    return passed


if __name__ == "__main__":
    print("=== G0 positive control v2: associative store + retrieve past eviction ===")
    print(f"DIM={DIM} L={L} eviction_gap={EVICT} writes  seeds={SEEDS}")
    # sanity that fresh write is retrievable at all (gap 0/1 should RETRIEVE if the test is valid)
    p_fav = sweep("favorable (repeat4, lr5e-2, muon5)", 5e-2, 5, 4)
    p_def = sweep("default (repeat1, lr1e-2, muon5)", 1e-2, 5, 1)
    p_hot = sweep("very favorable (repeat8, lr1e-1, muon5)", 1e-1, 5, 8)

    print("\n=== SUMMARY (does retrieval survive the 13-write eviction gap?) ===")
    for nm, p in [("favorable", p_fav), ("default", p_def), ("very-fav", p_hot)]:
        horizon = max([g for g, ok in p.items() if ok], default=-1)
        print(f"  {nm:>10}: retrieves up to gap {horizon} writes  (eviction gap = {EVICT})")
    fresh_ok = any(p.get(0) or p.get(1) for p in (p_fav, p_def, p_hot))
    evict_ok = any(p.get(EVICT) for p in (p_fav, p_def, p_hot))
    print()
    if not fresh_ok:
        print("  >>> TEST INVALID: even a fresh write is not retrievable at its key -> metric/qkv issue,")
        print("  >>> not a storage verdict. Do not conclude from this run.")
    elif not evict_ok:
        print("  >>> Fresh writes ARE retrievable, but retrieval decays to chance BEFORE the 13-write")
        print("  >>> eviction gap, even with favorable knobs => the Muon+renorm substrate's memory")
        print("  >>> horizon < eviction gap => MECHANISM DEAD AT STORAGE. No 30h training run needed.")
    else:
        print("  >>> Retrieval survives the eviction gap in >=1 regime => substrate CAN store across")
        print("  >>> eviction => proceed to the full clean re-run (train seed + fix objective + bound read).")
