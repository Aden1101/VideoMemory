#!/usr/bin/env python
# ============================================================================================
# The prompt-embedding cache, verified -- CPU, no real encoder (11.4 GB), ~1 s.
#
# THE test is P3: a prompt read back from the cache must return the tensor BIT FOR BIT. This
# project compares effects the size of its own chain noise (+/-2 pairs out of 19, measured
# 07/08); a rounding error introduced by the cache would be invisible and would falsify
# everything. No quantization is permitted here, not even a "harmless" one.
#
#   python tests/test_prompt_cache.py
# ============================================================================================
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import prompt_cache as pc  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


TMP = tempfile.mkdtemp(prefix="promptcache_")
try:
    A = "a woman in a red coat walks along a canal at dusk"
    B = "a baker kneading dough in a warm bakery"

    # --- P1: the key is stable and distinguishes what it must ---------------------------------
    check("P1 stable key", pc.key_of(A) == pc.key_of(A))
    # THE trap the key has to cover: pass 1 and pass 2 must work in the same precision, because
    # inference.py casts the whole pipeline to bfloat16. A float32 cache served to a bfloat16
    # model gives a wrong conditioning, with no error raised.
    check("P1d dtype is part of the key",
          pc.key_of(A, torch.float32) != pc.key_of(A, torch.bfloat16))
    check("P1b different prompts -> different keys", pc.key_of(A) != pc.key_of(B))
    check("P1c a space changes the key (no silent normalization)",
          pc.key_of(A) != pc.key_of(A + " "))

    # --- P2: absent before, present after ------------------------------------------------------
    check("P2 absent at the start", not pc.has(A, TMP, torch.float32)
          and pc.get(A, TMP, torch.float32) is None)
    emb = torch.randn(512, 4096, dtype=torch.float32)
    pc.put(A, emb, TMP)
    check("P2b present after write", pc.has(A, TMP, torch.float32))

    # --- P3: THE test -- bitwise read-back -----------------------------------------------------
    back = pc.get(A, TMP, torch.float32)
    check("P3 read-back BITWISE identical", torch.equal(back, emb),
          f"max|diff| = {float((back-emb).abs().max())}")
    check("P3b dtype preserved", back.dtype == emb.dtype)
    check("P3c shape preserved", back.shape == emb.shape)
    for dt in (torch.float16, torch.bfloat16):
        e2 = torch.randn(8, 16).to(dt)
        pc.put("x" + str(dt), e2, TMP)
        check(f"P3d {str(dt).split('.')[-1]} bitwise too",
              torch.equal(pc.get("x" + str(dt), TMP, dt), e2))

    # --- P4: get_many preserves input order and reports the misses -----------------------------
    # Catches the error that would raise no exception: pairing a prompt with another one's
    # conditioning because the rows got shuffled.
    # a bf16 embedding must NOT answer a float32 request
    pc.put(A, emb.to(torch.bfloat16), TMP)
    check("P1e bf16 and f32 coexist without being confused",
          pc.get(A, TMP, torch.float32).dtype is torch.float32
          and pc.get(A, TMP, torch.bfloat16).dtype is torch.bfloat16)

    got, missing = pc.get_many([B, A, B], TMP, torch.float32)
    check("P4 misses are located", missing == [0, 2], str(missing))
    check("P4b the hit is in its slot", torch.equal(got[1], emb))
    check("P4c misses are None", got[0] is None and got[2] is None)

    # --- P5: an inconsistent cache is a loud error, not a wrong conditioning --------------------
    p = pc.path_of(B, TMP, torch.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"prompt": "un autre texte", "embed": torch.zeros(4)}, p)
    try:
        pc.get(B, TMP, torch.float32)
        raise SystemExit("FAIL P5: a cache holding a different prompt was accepted")
    except RuntimeError:
        check("P5 inconsistent cache loudly refused", True)

    # --- P6: a damaged file re-encodes instead of crashing --------------------------------------
    with open(p, "wb") as f:
        f.write(b"pas un tenseur")
    check("P6 damaged file -> None (we re-encode)", pc.get(B, TMP, torch.float32) is None)

    # --- P7: the write is atomic (no .tmp survives) ---------------------------------------------
    leftovers = [f for _, _, fs in os.walk(TMP) for f in fs if ".tmp" in f]
    check("P7 no temporary file left behind", not leftovers, str(leftovers))

    st = pc.stats(TMP)
    check("P8 stats count the entries", st["n"] >= 3 and st["bytes"] > 0, str(st))

    print(f"\n{len(OK)}/{len(OK)} prompt-cache tests PASS")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
