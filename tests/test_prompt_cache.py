#!/usr/bin/env python
"""The prompt cache, on CPU, without the real encoder. Two checks matter most: a prompt read
back must be the same tensor bit for bit, and the dtype must be part of the key.

    python tests/test_prompt_cache.py
"""
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

    # --- the key is stable and distinguishes what it must ---------------------------------
    check("stable key", pc.key_of(A) == pc.key_of(A))
    # The trap the key has to cover: pass 1 and pass 2 must work in the same precision, because
    # inference.py casts the whole pipeline to bfloat16. A float32 cache served to a bfloat16
    # model gives a wrong conditioning, with no error raised.
    check("dtype is part of the key",
          pc.key_of(A, torch.float32) != pc.key_of(A, torch.bfloat16))
    check("different prompts -> different keys", pc.key_of(A) != pc.key_of(B))
    check("a space changes the key (no silent normalization)",
          pc.key_of(A) != pc.key_of(A + " "))

    # --- absent before, present after ------------------------------------------------------
    check("absent at the start", not pc.has(A, TMP, torch.float32)
          and pc.get(A, TMP, torch.float32) is None)
    emb = torch.randn(512, 4096, dtype=torch.float32)
    pc.put(A, emb, TMP)
    check("present after write", pc.has(A, TMP, torch.float32))

    # --- the one that matters, bitwise read-back --------------------------------------------
    back = pc.get(A, TMP, torch.float32)
    check("read-back bitwise identical", torch.equal(back, emb),
          f"max|diff| = {float((back-emb).abs().max())}")
    check("dtype preserved", back.dtype == emb.dtype)
    check("shape preserved", back.shape == emb.shape)
    for dt in (torch.float16, torch.bfloat16):
        e2 = torch.randn(8, 16).to(dt)
        pc.put("x" + str(dt), e2, TMP)
        check(f"{str(dt).split('.')[-1]} bitwise too",
              torch.equal(pc.get("x" + str(dt), TMP, dt), e2))

    # --- get_many preserves input order and reports the misses -----------------------------
    # Catches the error that would raise no exception: pairing a prompt with another one's
    # conditioning because the rows got shuffled.
    # a bf16 embedding must not answer a float32 request
    pc.put(A, emb.to(torch.bfloat16), TMP)
    check("bf16 and f32 coexist without being confused",
          pc.get(A, TMP, torch.float32).dtype is torch.float32
          and pc.get(A, TMP, torch.bfloat16).dtype is torch.bfloat16)

    got, missing = pc.get_many([B, A, B], TMP, torch.float32)
    check("misses are located", missing == [0, 2], str(missing))
    check("the hit is in its slot", torch.equal(got[1], emb))
    check("misses are None", got[0] is None and got[2] is None)

    # --- an inconsistent cache is a loud error, not a wrong conditioning --------------------
    p = pc.path_of(B, TMP, torch.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"prompt": "un autre texte", "embed": torch.zeros(4)}, p)
    try:
        pc.get(B, TMP, torch.float32)
        raise SystemExit("FAIL: a cache holding a different prompt was accepted")
    except RuntimeError:
        check("inconsistent cache loudly refused", True)

    # --- a damaged file re-encodes instead of crashing --------------------------------------
    with open(p, "wb") as f:
        f.write(b"pas un tenseur")
    check("damaged file -> None (we re-encode)", pc.get(B, TMP, torch.float32) is None)

    # --- the write is atomic (no .tmp survives) ---------------------------------------------
    leftovers = [f for _, _, fs in os.walk(TMP) for f in fs if ".tmp" in f]
    check("no temporary file left behind", not leftovers, str(leftovers))

    st = pc.stats(TMP)
    check("stats count the entries", st["n"] >= 3 and st["bytes"] > 0, str(st))

    print(f"\n{len(OK)}/{len(OK)} prompt-cache tests PASS")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
