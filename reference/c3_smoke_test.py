"""C3 smoke test: cross-chunk write-training autograd.

Reproduces the C3 launch blocker at the VideoMemoryLayer level and proves the fix.

Scenario (mirrors training: multi-block chunk, ONE backward per chunk, fast-weight state
carried chunk->chunk by reference, write under enable_grad with detach_state=False):
  - WITHOUT detach between chunks  -> chunk-2 backward walks chunk-1's freed graph -> RuntimeError
  - WITH    detach between chunks  -> each chunk's write-graph starts from a leaf -> OK

The applied fix (model/streaming_training.py::_generate_chunk, before generate_chunk_with_cache)
does exactly the "WITH detach" branch on inference_pipeline.ttt_cache[*].state.
"""
import os, sys
# Force Anchor-Forcing's wan (NOT infinity-rope's) to win, and keep compute on CPU.
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
import torch

from wan.modules.video_memory import VideoMemoryLayer
assert "Anchor-Forcing" in VideoMemoryLayer.__module__ or "Anchor-Forcing" in sys.modules["wan"].__file__, \
    f"wrong wan loaded: {sys.modules['wan'].__file__}"

torch.manual_seed(0)
DEV = "cpu"
DIM, NB = 64, 2          # small dim; 2 blocks per "chunk"


def make_layer():
    m = VideoMemoryLayer(dim=DIM, num_heads=1, inter_multi=4, base_lr=1e-2,
                         muon_update_steps=5, use_modulation=True, g_init=0.1,
                         detach_state=False, gate_extra=[]).to(DEV)
    m.train()                                  # training mode (keeps graph)
    return m


def run_chunk(m, state):
    """One chunk = NB blocks, each does apply-first read + write (do_update). One loss."""
    loss = 0.0
    for _ in range(NB):
        x = torch.randn(1, 3, DIM, device=DEV)         # (B, L, dim) activations
        with torch.enable_grad():
            out, state = m.step(x, state=state, do_update=True)
        loss = loss + out.float().sum()
    return loss, state


def trial(detach_between_chunks):
    m = make_layer()
    state = None
    # chunk 1
    loss1, state = run_chunk(m, state)
    loss1.backward()                                    # frees chunk-1 graph
    # boundary
    if detach_between_chunks:
        state = tuple(s.detach() for s in state)        # <-- the C3 fix
    # chunk 2 (reuses carried state)
    loss2, state = run_chunk(m, state)
    loss2.backward()                                    # walks carried state's graph


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device={DEV}  blocks/chunk={NB}")

    print("\n[1] WITHOUT fix (expect the C3 crash):")
    try:
        trial(detach_between_chunks=False)
        print("    UNEXPECTED: no error -> the test did not reproduce C3 (investigate)")
        bug_repro = False
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"    reproduced RuntimeError: {msg}")
        bug_repro = "second time" in str(e) or "freed" in str(e) or "buffers" in str(e)

    print("\n[2] WITH fix (detach carried state -> expect success):")
    try:
        trial(detach_between_chunks=True)
        print("    OK: two chunks trained, no backward error")
        fix_ok = True
    except RuntimeError as e:
        print(f"    FAILED: {str(e).splitlines()[0]}")
        fix_ok = False

    print("\n=== RESULT ===")
    print(f"  bug reproduced without fix : {bug_repro}")
    print(f"  fix resolves it            : {fix_ok}")
    print("  C3 SMOKE TEST PASSED" if (bug_repro and fix_ok) else "  C3 SMOKE TEST INCONCLUSIVE")
