#!/usr/bin/env python
# ============================================================================================
# ORACLE DE LA REGLE D'ECRITURE LINEAIRE -- SELFTESTS OBLIGATOIRES AVANT LE RUN COMPLET
# (registre : section "ORACLE DE LA REGLE D'ECRITURE LINEAIRE", GO 2026-08-06 soir).
#
#   T0  py_compile + les 4 bras pointent sur UN SEUL lecteur (READERS[lin_*] is read_lin) et
#       UN SEUL coeur d'ecriture (lin_chunk_update), masquage par maskwrite_keep (jamais
#       recopie)
#   T1  ANALYTIQUE : mini-cas construit a la main (2 tokens connus, d = 4) -- W (moyenne ET
#       delta rule, mise a jour sequentielle deroulee a la main) puis W @ sondes, verifies
#       contre le calcul numpy explicite, exactement
#   T2  MOYENNE CUMULATIVE : sur une vraie video, W a l'age j == moyenne EXACTE de tous les
#       tokens ecrits jusqu'a j (recalcul INDEPENDANT en fp64, une seule GEMM globale),
#       allclose fp32 strict, ecart max rapporte -- bras masque ET tous-tokens
#   T3  MASQUAGE : memes indices de tokens que l'oracle precedent, chunk par chunk, 2 videos
#       (compare aux keep/info de maskwrite_keep, la fonction que lin_states appelle)
#   T4  DETERMINISME : 2 executions completes (lin_states -> load_payload -> signature ->
#       fp16) -> bit a bit, pour les 4 bras, 1 video
#   Stats loggees : ||W||_F par age et par bras, tokens/chunk, chrono par bras (l'estimation
#   du run complet en depend).
#
# Usage (GPU obligatoire, pre-check DANS la commande, jamais l'index smi 6) :
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/selftest_lin.py
# ============================================================================================
import inspect
import os
import py_compile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mem2_data as D  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failed = []


def check(name, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        _failed.append(name)
    return ok


def first_pair_videos(corpus, k):
    vids, pairs = C.pair_dataset(verbose=False, corpus=corpus)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sel = [v for v in vids if v["video_id"] in pair_ids]
    sel.sort(key=lambda v: (v["part"], v["vi"]))
    return sel[:k]


def main():
    assert torch.cuda.is_available(), "GPU obligatoire"
    C.ensure_dirs()
    C.set_determinism(0)
    import build_signatures as B
    import build_states as BS

    # ---------------- T0 : compilation + unicite des chemins -------------------------------
    print("[T0] compilation + un seul lecteur / un seul coeur d'ecriture")
    for f in ("build_states.py", "build_signatures.py", "train_probe.py", "selftest_lin.py"):
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    check("T0 py_compile 4 fichiers", True)
    check("T0 les 4 bras -> UN lecteur read_lin",
          all(B.READERS[s] is B.read_lin for s in B.LIN_SOURCES)
          and sorted(B.LIN_SOURCES) == sorted(BS.LIN_SPEC))
    src = inspect.getsource(BS.lin_states)
    check("T0 lin_states : masquage par maskwrite_keep + coeur lin_chunk_update (rien recopie)",
          "maskwrite_keep(meta, \"subject\")" in src and "lin_chunk_update(" in src)

    # ---------------- T1 : analytique, 2 tokens connus, d = 4 ------------------------------
    print("[T1] mini-cas a la main (d=4, 2 tokens) vs numpy explicite")
    h_np = np.array([[1.0, 2.0, 2.0, 0.0],       # ||h1|| = 3
                     [0.0, 3.0, 0.0, 4.0]])      # ||h2|| = 5
    k_np = h_np / np.linalg.norm(h_np, axis=1, keepdims=True)
    # moyenne : W = (v1 k1^T + v2 k2^T) / 2
    W_mean_ref = (np.outer(h_np[0], k_np[0]) + np.outer(h_np[1], k_np[1])) / 2.0
    # delta rule deroulee a la main : W0 = 0 ; W1 = W0 + b (v1 - W0 k1) k1^T ;
    #                                 W2 = W1 + b (v2 - W1 k2) k2^T ; b = 0.5
    b = BS.LIN_BETA
    W1 = b * np.outer(h_np[0], k_np[0])
    W_delta_ref = W1 + b * np.outer(h_np[1] - W1 @ k_np[1], k_np[1])
    x = torch.tensor(h_np, dtype=torch.float32)
    Z = torch.zeros(4, 4)
    S, N, _ = BS.lin_chunk_update(x, "mean", Z.clone(), 0, Z.clone())
    _, _, Wd = BS.lin_chunk_update(x, "delta", Z.clone(), 0, Z.clone())
    check("T1 beta scelle = 0.5", b == 0.5)
    check("T1 moyenne : W == numpy exact",
          np.allclose((S / N).numpy(), W_mean_ref, rtol=1e-6, atol=1e-7),
          f"ecart max {np.abs((S / N).numpy() - W_mean_ref).max():.2e}")
    check("T1 delta : W == numpy exact (2 pas sequentiels)",
          np.allclose(Wd.numpy(), W_delta_ref, rtol=1e-6, atol=1e-7),
          f"ecart max {np.abs(Wd.numpy() - W_delta_ref).max():.2e}")
    # l'ORDRE compte pour la delta rule : tokens inverses => W different (test de sequentialite)
    _, _, Wr = BS.lin_chunk_update(x.flip(0), "delta", Z.clone(), 0, Z.clone())
    check("T1 delta : l'ordre des tokens compte (sequentiel, pas une somme)",
          not torch.allclose(Wd, Wr))
    # lecture : W @ sondes contre numpy, pour 3 sondes connues
    q = torch.tensor(np.eye(4)[:3], dtype=torch.float32)
    r = B.read_lin(q, (S / N).to(torch.float16))
    r_ref = (W_mean_ref.astype(np.float16).astype(np.float32) @ np.eye(4)[:3].T).T
    check("T1 lecture read_lin == W @ sondes (numpy, via le fp16 stocke)",
          np.allclose(r.numpy(), r_ref, rtol=1e-6, atol=1e-7),
          f"ecart max {np.abs(r.numpy() - r_ref).max():.2e}")

    # ---------------- T2 : moyenne cumulative == recalcul independant ----------------------
    print("[T2] W(age j) == moyenne exacte de TOUS les tokens ecrits jusqu'a j (fp64 ref)")
    cache = D.PartCache(size=1)
    meta = first_pair_videos("v2", 1)[0]
    ages = C.state_ages(meta)
    keep_s, _ = BS.maskwrite_keep(meta, "subject")
    for masked, label in ((True, "sujet-masque"), (False, "tous-tokens")):
        got, stats = BS.lin_states(meta, cache, [5, 16, 27], ages, "cuda", "mean", masked)
        h = D.load_h(meta, cache, [5, 16, 27], upto=max(int(a["i"]) for a in ages),
                     device="cuda", dtype=torch.float32)
        ok, worst = True, 0.0
        for a in ages:
            i = int(a["i"])
            for bi in (5, 16, 27):
                toks = []
                for j in range(i + 1):
                    xj = h[bi][j][0]
                    if masked:
                        if keep_s[j] is None:
                            continue
                        xj = xj[torch.as_tensor(np.asarray(keep_s[j]), dtype=torch.long,
                                                device=xj.device)]
                    toks.append(xj)
                X = torch.cat(toks, 0).double()          # reference INDEPENDANTE, une GEMM fp64
                K = X / X.norm(dim=-1, keepdim=True)
                Wref = (X.transpose(0, 1) @ K / X.shape[0]).float()
                Wgot = got[i][bi].to("cuda", torch.float32)   # le snapshot fp16 releve
                dev = float((Wgot - Wref).abs().max())
                scale = float(Wref.abs().max())
                worst = max(worst, dev / max(scale, 1e-12))
                # tolerance = l'arrondi fp16 du SNAPSHOT (2^-11), pas celui de l'accumulation
                ok &= dev <= 2 ** -10 * max(scale, 1.0)
        check(f"T2 {label} : allclose strict vs ref fp64 (tol = arrondi fp16 du snapshot)",
              ok, f"ecart relatif max {worst:.3e}")
        del h

    # ---------------- T3 : masquage == oracle precedent, chunk par chunk -------------------
    print("[T3] indices masques == maskwrite_keep (l'alignement de l'oracle precedent)")
    for m3 in first_pair_videos("v2", 2):
        k1, info1 = BS.maskwrite_keep(m3, "subject")
        _, stats = BS.lin_states(m3, cache, [5, 16, 27], C.state_ages(m3), "cuda",
                                 "mean", True)
        same_counts = stats["kept_counts"] == [0 if k is None else int(k.size) for k in k1]
        check(f"T3 {m3['video_id']} comptes par chunk == maskwrite_keep "
              f"(probe1_common.chunk_token_mask:429, min_frac=0.0)", same_counts,
              f"tokens/chunk moy {np.mean([c for c in stats['kept_counts'] if c > 0]):.0f}, "
              f"{stats['kept_counts'].count(0)} sautee(s)")

    # ---------------- T4 : determinisme bout en bout, 4 bras -------------------------------
    print("[T4] 2 executions completes -> signatures bit a bit (4 bras, 1 video)")
    from build_probes import PROBES_PT
    pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    probes, blocks = pk["probes"].to("cuda"), list(pk["blocks"])
    chrono = {}
    for srcname in sorted(B.LIN_SOURCES):
        rule, masked = BS.LIN_SPEC[srcname]
        sigs = []
        t0 = time.time()
        for _run in range(2):
            states, stats = BS.lin_states(meta, cache, blocks, ages, "cuda", rule, masked)
            per_age = []
            for a in ages:
                pay = B.load_payload(srcname, meta, a, cache, blocks, "cuda", states=states)
                per_age.append(B.signature(srcname, probes, pay, blocks).to(torch.float16))
            sigs.append(per_age)
        chrono[srcname] = (time.time() - t0) / 2
        same = all(torch.equal(x, y) for x, y in zip(*sigs))
        nrm = {i: round(v[blocks[0]], 3) for i, v in sorted(stats["norms"].items())}
        check(f"T4 {srcname} bitwise (etats + signatures)", same,
              f"{chrono[srcname]:.1f} s/video ; ||W||_F bloc {blocks[0]} par age {nrm}")

    print()
    est = 60 * sum(chrono.values()) / 60
    print(f"[chrono] par video : " + ", ".join(f"{k}={v:.1f}s" for k, v in chrono.items()))
    print(f"[chrono] extrapolation 60 videos, 4 bras : ~{est:.0f} min de signatures "
          f"(+ ~8.5 min par CV x 4)")
    if _failed:
        print(f"SELFTEST LIN : {len(_failed)} ECHEC(S) : {_failed}")
        sys.exit(1)
    print("SELFTEST LIN : TOUT PASSE (T0-T4). STOP ici : relecture de code avant le run "
          "complet (commandes : run_oracle_lin.sh).")


if __name__ == "__main__":
    main()
