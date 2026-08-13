#!/usr/bin/env python
# ============================================================================================
# P0 -- MEMOIRE LINEAIRE NATIVE : TEST D'EQUIVALENCE GENERATION == REJEU HORS LIGNE.
# (plan : reference/plan_entrainement_video_2026-08-07.md, section 2, bras M2 -- "le seul vrai
#  developpement du programme", et LE test qui autorise la suite : sans lui on entrainerait un
#  decodeur sur un etat qui n'est pas celui du deploiement.)
#
# Ce qui est teste, sur 2 videos du corpus v2 (main:9000000, main:9000001 par defaut) :
#   N0  NON-REGRESSION EN GENERATION (memoire par DEFAUT, SwiGLU) : le rollout regenere
#       (build_pipeline + rollout_collect, MEM_ZERO_ALL=1, poke fp16) reproduit BIT A BIT le
#       shard v2 du disque -- latents fp16 et h fp16, chaque bloc, chaque ecriture. C'est la
#       preuve en conditions reelles que le chemin par defaut n'a pas bouge apres l'ajout du
#       type lineaire (en plus des selftests historiques rejoues a cote).
#   T0  PARITE DES DEUX IMPLEMENTATIONS : video_memory.linear_delta_chunk_update (le coeur
#       DEPLOYE) == build_states.lin_chunk_update rule='delta' (l'oracle scelle), torch.equal
#       sur un chunk REEL (4680 x 1536) et sur un mini-cas synthetique.
#   T1  IDENTITE DE REGENERATION (memoire LINEAIRE activee par config) : meme graine, memes
#       write_chunks, latents et h bit a bit egaux au shard -- sous MEM_ZERO_ALL=1 l'etat ne
#       nourrit que l'injection zeroisee, donc la memoire lineaire est INVISIBLE du backbone ;
#       cette egalite prouve qu'on rejoue bien LES videos du corpus, pas des cousines.
#   T2  EQUIVALENCE fp32, CHAQUE ECRITURE : l'etat W produit EN GENERATION (snapshot fp32
#       apres chacune des 25 ecritures, 3 blocs) == la chaine hors ligne lin_chunk_update sur
#       les h STOCKES (fp16 disque -> fp32, exact), torch.equal (bit a bit).
#   T3  EQUIVALENCE CONTRE LA FONCTION DE PRODUCTION : build_states.lin_states (delta, tous
#       tokens = bras lin_delta_all) aux ~4 ages scelles -> ses snapshots fp16 == fp16(W de
#       generation) aux memes ecritures, bit a bit.
#   Chrono + VRAM : s/video et pic memoire, memoire par defaut vs lineaire, meme GPU.
#
# Pourquoi le bit a bit est ATTEIGNABLE (et pas seulement un allclose) :
#   - le collecteur remplace le h du fifo par son aller-retour fp16 AVANT l'ecriture (poke
#     scelle de collect_pairs) : l'ecriture in-situ consomme EXACTEMENT les octets du shard ;
#   - bf16 -> fp32 (generation) et fp16 -> fp32 (rejeu) sont exacts, memes valeurs ;
#   - _linear_write force TF32 OFF (sauvegarde/restauration) et le rejeu tourne TF32 OFF
#     (asserte ici) : memes kernels fp32, memes formes (T, d), meme ordre sequentiel ;
#   - meme GPU, meme process pour les deux chemins.
#
# Usage (GPU obligatoire ; pre-check DANS la commande, jamais l'index 6, jamais une carte d'un
# autre utilisateur) :
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=<X> python reference/runs/probe1/selftest_lin_native.py
# Options : --skip-baseline (saute N0), --n-videos 2, --config <yaml>.
# Sortie : code 0 = tout vert ; JSON recapitulatif dans $HEAVY/runs/selftest_lin_native.json.
# ============================================================================================
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402  (pose REPO/mem2/reid_probe_l1/frame_dec dans sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
from selftest_lin import first_pair_videos  # noqa: E402  (les MEMES 2 videos que selftest_lin)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failed = []


def check(name, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        _failed.append(name)
    return ok


def max_abs_diff(a, b):
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


def run_rollout(pipeline, config, meta, snap_every):
    """Regeneration d'UNE video v2 par LE code du collecteur (rollout_collect importe, jamais
    recopie) : v3_capture=True => MEM_ZERO_ALL=1 + poke fp16 (les octets consommes par
    l'ecriture sont ceux du shard). Rend (res, secondes, pic_vram_gb)."""
    from collect_pairs import rollout_collect
    assert meta["mode"] == "single" and meta["switch_frame"] is None, \
        f"{meta['video_id']}: le corpus v2 est single-only"
    seed_offset = D.TWIN_SEED_OFFSET if meta["twin"] else 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = rollout_collect(pipeline, config, meta["prompt"], meta["corpus_idx"],
                          meta["frames"], 0, "cuda",
                          v3_capture=True, v3_fp16_roundtrip=True,
                          v3_state_snap_every=snap_every, seed_offset=seed_offset)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert res["seed"] == meta["seed"], \
        f"{meta['video_id']}: graine regeneree {res['seed']} != stockee {meta['seed']}"
    assert res["write_chunks"] == meta["write_chunks"], \
        f"{meta['video_id']}: write_chunks regeneres != stockes"
    return res, dt, peak


def check_identity(tag, meta, res, rec, blocks):
    """T1/N0 : latents fp16 et h fp16 regeneres == shard du disque, bit a bit."""
    lat_ok = torch.equal(res["latents"][0].to(torch.float16), rec["latents"])
    check(f"{tag} {meta['video_id']} latents fp16 bit a bit vs shard", lat_ok,
          "" if lat_ok else f"max|d|={max_abs_diff(res['latents'][0].to(torch.float16), rec['latents']):.3e}")
    h_ok, first_bad = True, None
    for bi in blocks:
        stored = rec["h_seq"][str(bi)]
        got = res["h_seq"][bi]
        if len(stored) != len(got):
            h_ok, first_bad = False, (bi, "count", len(got), len(stored))
            break
        for j, (a, b) in enumerate(zip(got, stored)):
            if not torch.equal(a, b):
                h_ok, first_bad = False, (bi, j, max_abs_diff(a, b))
                break
        if not h_ok:
            break
    check(f"{tag} {meta['video_id']} h fp16 bit a bit vs shard "
          f"({len(meta['write_chunks'])} ecritures x {len(blocks)} blocs)", h_ok,
          "" if h_ok else f"premiere divergence {first_bad}")
    return lat_ok and h_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--n-videos", type=int, default=2)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="saute N0 (rollout memoire par defaut) -- equivalence seulement")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU obligatoire (`import wan` touche CUDA)"
    # parite bit a bit : le rejeu de l'oracle a tourne TF32 OFF (set_determinism) ; les
    # process d'inference ne touchent jamais ces flags (meme assert que selftest_replay).
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 est ON : parite fp32 impossible"
    C.ensure_dirs()
    dev_name = torch.cuda.get_device_name(0)
    print(f"[lin-native] GPU = {dev_name} ; les shards v2 ont ete generes sur la meme "
          f"machine (meme modele de carte requis pour le bit a bit)", flush=True)

    import build_states as BS
    from run_reid_probe import build_pipeline, release_vram
    from wan.modules.video_memory import linear_delta_chunk_update

    config = OmegaConf.load(args.config)
    config.distributed = False
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)
    vids = first_pair_videos("v2", args.n_videos)
    assert len(vids) == args.n_videos, f"corpus v2 : {len(vids)} videos < {args.n_videos}"
    print(f"[lin-native] videos : {[v['video_id'] for v in vids]} ; blocs {blocks}", flush=True)
    cache = D.PartCache(size=1)
    summary = {"gpu": dev_name, "config": os.path.abspath(args.config),
               "videos": [v["video_id"] for v in vids], "blocks": blocks,
               "beta": float(BS.LIN_BETA), "checks": {}, "chrono": {}, "vram_gb": {}}

    # ---------- empreinte memoire de l'etat (chiffree, demandee par la mission) ------------
    d = C.DIM
    lin_fp32 = d * d * 4 / 1e6
    lin_fp16 = d * d * 2 / 1e6
    swiglu = 3 * d * (4 * d) * 2 / 1e6
    print(f"[etat] lineaire : W fp32 {d}x{d} = {lin_fp32:.2f} MB/bloc "
          f"({len(blocks)} blocs = {lin_fp32 * len(blocks):.1f} MB), snapshot fp16 "
          f"{lin_fp16:.2f} MB/bloc ({lin_fp16 * len(blocks):.1f} MB/age) ; "
          f"SwiGLU actuel : {swiglu:.1f} MB/bloc ({swiglu * len(blocks):.1f} MB) "
          f"-> lineaire ~{swiglu / lin_fp32:.1f}x plus petit", flush=True)
    summary["state_mb"] = {"linear_fp32_per_block": lin_fp32, "swiglu_bf16_per_block": swiglu}

    # ================= N0 : non-regression memoire par DEFAUT, en generation ===============
    if not args.skip_baseline:
        print("[N0] pipeline memoire par DEFAUT (SwiGLU l1@2000-EMA), rollout video 1",
              flush=True)
        pipeline = build_pipeline(config, "cuda")
        from collect_pairs import _find_causal_model
        cm = _find_causal_model(pipeline.generator.model)   # peft wrappe le modele
        assert cm is not None, "CausalWanModel introuvable sous les wrappers peft"
        for bi in blocks:
            st = getattr(cm.blocks[bi].memory, "state_type", "swiglu")
            assert st == "swiglu", f"bloc {bi}: state_type {st!r} != swiglu (defaut)"
        meta0 = vids[0]
        res0, dt0, peak0 = run_rollout(pipeline, config, meta0, snap_every=0)
        rec0 = D.get_record(meta0, cache)
        ok0 = check_identity("N0", meta0, res0, rec0, blocks)
        summary["checks"]["N0_default_bitwise"] = bool(ok0)
        summary["chrono"]["default_s_per_video"] = round(dt0, 1)
        summary["vram_gb"]["default_peak"] = round(peak0, 2)
        print(f"[N0] {dt0:.1f} s/video, pic VRAM {peak0:.2f} GB", flush=True)
        del res0, pipeline
        release_vram()

    # ================= T1 : memoire LINEAIRE par CONFIG, identite de regeneration ==========
    print("[T1] pipeline memoire LINEAIRE (model_kwargs.mem_state_type=linear_delta)",
          flush=True)
    config_lin = OmegaConf.load(args.config)
    config_lin.distributed = False
    config_lin.model_kwargs.mem_state_type = "linear_delta"   # LE champ de config P0
    pipeline = build_pipeline(config_lin, "cuda")
    from collect_pairs import _find_causal_model
    cm = _find_causal_model(pipeline.generator.model)
    assert cm is not None, "CausalWanModel introuvable sous les wrappers peft"
    for bi in blocks:
        cell = cm.blocks[bi].memory
        check(f"T1 threading config -> bloc {bi} state_type=linear_delta, beta=0.5",
              getattr(cell, "state_type", None) == "linear_delta"
              and float(cell.linear_beta) == float(BS.LIN_BETA))

    gen_snaps = {}
    for k, meta in enumerate(vids):
        res, dt, peak = run_rollout(pipeline, config_lin, meta, snap_every=1)
        rec = D.get_record(meta, cache)
        check_identity("T1", meta, res, rec, blocks)
        n = len(meta["write_chunks"])
        snaps = {}
        for bi in blocks:
            lst = res["check_states"][str(bi)]
            assert len(lst) == n, f"bloc {bi}: {len(lst)} snapshots != {n} ecritures"
            for j, (chunk, st) in enumerate(lst):
                assert chunk == meta["write_chunks"][j], \
                    f"bloc {bi} ecriture {j}: snapshot au chunk {chunk} != {meta['write_chunks'][j]}"
                assert len(st) == 1 and st[0].dtype == torch.float32, \
                    f"bloc {bi}: etat lineaire attendu (W fp32,), recu {[t.dtype for t in st]}"
            snaps[bi] = [st[0][0] for _, st in lst]      # (1536,1536) fp32 cpu, par ecriture
        gen_snaps[meta["video_id"]] = snaps
        summary["chrono"][f"linear_s_per_video_{k}"] = round(dt, 1)
        summary["vram_gb"][f"linear_peak_{k}"] = round(peak, 2)
        print(f"[T1] {meta['video_id']} : {dt:.1f} s/video, pic VRAM {peak:.2f} GB", flush=True)
        del res
    del pipeline
    release_vram()

    # ================= T0 : parite des deux implementations du coeur ========================
    print("[T0] parite lin_chunk_update (oracle) == linear_delta_chunk_update (deploye)",
          flush=True)
    xs = torch.tensor(np.array([[1.0, 2.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]]),
                      dtype=torch.float32)
    Z = torch.zeros(4, 4)
    _, _, Wo = BS.lin_chunk_update(xs, "delta", Z.clone(), 0, Z.clone())
    Wn = linear_delta_chunk_update(xs, Z.clone(), BS.LIN_BETA)
    check("T0 mini-cas synthetique (cpu, d=4) bit a bit", torch.equal(Wo, Wn))
    meta = vids[0]
    h = D.load_h(meta, cache, [blocks[0]], upto=0, device="cuda", dtype=torch.float32)
    xr = h[blocks[0]][0]
    xr = xr[0] if xr.dim() == 3 else xr
    Zr = torch.zeros(d, d, dtype=torch.float32, device="cuda")
    _, _, Wo = BS.lin_chunk_update(xr, "delta", Zr.clone(), 0, Zr.clone())
    Wn = linear_delta_chunk_update(xr, Zr.clone(), BS.LIN_BETA)
    check(f"T0 chunk reel {tuple(xr.shape)} (gpu) bit a bit", torch.equal(Wo, Wn),
          f"max|d|={max_abs_diff(Wo, Wn):.3e}" if not torch.equal(Wo, Wn) else "")
    del h

    # ================= T2 : generation == chaine hors ligne, fp32, CHAQUE ecriture ==========
    print("[T2] equivalence fp32 par ecriture : generation vs lin_chunk_update sur les h "
          "du disque", flush=True)
    for meta in vids:
        n = len(meta["write_chunks"])
        h = D.load_h(meta, cache, blocks, upto=n - 1, device="cuda", dtype=torch.float32)
        for bi in blocks:
            S = torch.zeros(d, d, dtype=torch.float32, device="cuda")
            W = torch.zeros(d, d, dtype=torch.float32, device="cuda")
            N = 0
            ok, first_bad, worst = True, None, 0.0
            for j in range(n):
                x = h[bi][j]
                x = x[0] if x.dim() == 3 else x
                S, N, W = BS.lin_chunk_update(x, "delta", S, N, W)
                Wg = gen_snaps[meta["video_id"]][bi][j]
                if not torch.equal(W.cpu(), Wg):
                    dmax = max_abs_diff(W, Wg)
                    worst = max(worst, dmax)
                    if ok:
                        ok, first_bad = False, j
            check(f"T2 {meta['video_id']} bloc {bi} : {n} ecritures bit a bit (fp32)", ok,
                  f"||W||_F final {float(W.norm()):.1f}" if ok
                  else f"premiere divergence ecriture {first_bad}, max|d|={worst:.3e}")
        del h

    # ================= T3 : contre lin_states (la fonction de production), ages scelles =====
    print("[T3] equivalence fp16 aux ages scelles : lin_states (bras lin_delta_all) vs "
          "fp16(W de generation)", flush=True)
    for meta in vids:
        ages = C.state_ages(meta)
        got, stats = BS.lin_states(meta, cache, blocks, ages, "cuda", "delta", False)
        for a in ages:
            i = int(a["i"])
            for bi in blocks:
                Wg16 = gen_snaps[meta["video_id"]][bi][i].to(torch.float16)
                eq = torch.equal(got[i][bi], Wg16)
                check(f"T3 {meta['video_id']} age {a['name']} (ecriture {i}) bloc {bi} "
                      f"fp16 bit a bit", eq,
                      "" if eq else f"max|d|={max_abs_diff(got[i][bi], Wg16):.3e}")

    summary["checks"]["failed"] = list(_failed)
    out = C.save_json(summary, os.path.join(C.RUNS_DIR, "selftest_lin_native.json"))
    print(f"[lin-native] recapitulatif -> {out}", flush=True)
    if _failed:
        print(f"SELFTEST LIN NATIVE : {len(_failed)} ECHEC(S) : {_failed}")
        sys.exit(1)
    print("SELFTEST LIN NATIVE : TOUT PASSE (N0, T0-T3). L'etat de generation EST l'etat du "
          "rejeu lin_delta_all, bit a bit ; P1 attend le GO.")


if __name__ == "__main__":
    main()
