#!/usr/bin/env python
# ============================================================================================
# ORACLE D'ECRITURE MASQUEE -- SELFTESTS OBLIGATOIRES AVANT LE RUN COMPLET (registre :
# section "ORACLE D'ECRITURE MASQUEE", GO 2026-08-06).
#
#   S0  py_compile des fichiers touches + le chemin `none` est syntaxiquement l'HISTORIQUE
#       (replay_video keep=None passe par l'appel replay_writes par troncons, inchange)
#   S1  `--write-mask none` == etats du DISQUE, bit a bit (2 videos v1, torch.equal sur
#       CHAQUE tenseur de CHAQUE bloc a CHAQUE age). ECHEC => STOP, rien d'autre ne compte.
#   S2  bras `subject` (2 videos v2) : stats tokens-sujet/chunk (moy, min, max) + fraction
#       d'ecritures sautees ; ALIGNEMENT masque->token = LA MEME fonction que le chemin
#       `hdirect_masked` (probe1_common.chunk_token_mask), verifie (a) sur le source et
#       (b) par equivalence des indices quand le repli de lecture ne joue pas ;
#       DETERMINISME : keep recalcule 2x identique, rejeu masque 2x -> etats bit a bit.
#   S3  bras `bg_matched` (1 video v2) : compte de tokens IDENTIQUE au bras subject chunk
#       par chunk, intersection NULLE avec les tokens sujet, MEMES ecritures sautees,
#       tirage deterministe (2 appels identiques).
#   S4a (condition du mode stream, relecture pre-run 06/08) : rejeu avec keep = TOUS les
#       tokens (le chemin par-ecriture : index_select plein + write par write) == chemin
#       historique keep=None, bit a bit, 1 video x 3 blocs x 4 ages. Isole la PLOMBERIE de
#       la boucle par-ecriture de l'effet du masque.
#   S4b (idem) : signatures produites en mode STREAM (etats en memoire) == signatures
#       produites depuis des ETATS-DISQUE (memes champs que build_states), bras subject,
#       1 video, bit a bit en fp32 ET en fp16 (le format stocke). Echec de l'un des deux
#       => STOP, repli mode disque.
#
# Usage (GPU obligatoire, pre-check DANS la commande, jamais l'index smi 6) :
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/selftest_maskwrite.py
# ============================================================================================
import os
import py_compile
import sys

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
    """Les k premieres videos de paires dans L'ORDRE DE build_states.main (tri (part, vi)) :
    deterministe, et pour v1 ce sont des videos dont l'etat est deja sur le disque."""
    vids, pairs = C.pair_dataset(verbose=False, corpus=corpus)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sel = [v for v in vids if v["video_id"] in pair_ids]
    sel.sort(key=lambda v: (v["part"], v["vi"]))
    return sel[:k]


def states_equal(a, b):
    """Deux sorties de replay_video {age: {bloc: (w0, w1, w2)}} -> bit a bit ?"""
    if sorted(a) != sorted(b):
        return False
    for i in a:
        if sorted(a[i]) != sorted(b[i]):
            return False
        for bi in a[i]:
            if len(a[i][bi]) != len(b[i][bi]):
                return False
            if not all(torch.equal(x, y) for x, y in zip(a[i][bi], b[i][bi])):
                return False
    return True


def main():
    assert torch.cuda.is_available(), "GPU obligatoire (S1/S2 rejouent des ecritures)"
    C.ensure_dirs()
    C.set_determinism(0)
    from omegaconf import OmegaConf

    import build_signatures as B
    import build_states as BS
    from train_writer import build_cells, delta_l1, ref_theta_l1

    # ---------------- S0 : compilation + le chemin `none` est l'historique -----------------
    print("[S0] compilation + chemin `none` intouche")
    for f in ("probe1_common.py", "build_states.py", "build_signatures.py", "train_probe.py",
              "selftest_maskwrite.py"):
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    import inspect
    src = inspect.getsource(BS.replay_video)
    check("S0 py_compile 5 fichiers", True)
    check("S0 keep=None -> appel par troncons historique",
          "if keep is None:" in src and "h[bi][prev:i + 1]" in src)
    check("S0 maskwrite_keep passe par C.chunk_token_mask (JAMAIS recopie)",
          "C.chunk_token_mask(" in inspect.getsource(BS.maskwrite_keep)
          and "C.video_mask_grids(" in inspect.getsource(BS.maskwrite_keep))

    # ---------------- cellules l1@2000-EMA (contrat F2.3) ----------------------------------
    config = OmegaConf.load(C.DEFAULT_CONFIG)
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)
    cells, blocks_c, _ = build_cells(config, "cuda", mode="direct")
    assert blocks_c == blocks
    ref, srcname = ref_theta_l1(config, blocks, "cuda")
    dl1 = delta_l1(cells, ref)
    check("contrat F2.3 delta_l1 == 0", dl1 == 0.0, f"delta_l1={dl1} ({srcname})")

    cache = D.PartCache(size=1)

    # ---------------- S1 : none == disque, bit a bit (2 videos v1) -------------------------
    print("[S1] --write-mask none == etats existants de states/ (bit a bit, 2 videos v1)")
    s1_ok = True
    for meta in first_pair_videos("v1", 2):
        ages = C.state_ages(meta)
        missing = [a["name"] for a in ages
                   if not os.path.exists(BS.state_path(meta["video_id"], a["name"]))]
        assert not missing, f"{meta['video_id']}: etats absents du disque {missing}"
        got = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=None)
        for a in ages:
            ck = torch.load(BS.state_path(meta["video_id"], a["name"]), map_location="cpu",
                            weights_only=False)
            same = all(torch.equal(x, y) for bi in blocks
                       for x, y in zip(got[int(a["i"])][bi], ck["state"][bi]))
            s1_ok &= check(f"S1 {meta['video_id']} {a['name']} bitwise", same)
        del got
    if not s1_ok:
        print("\nS1 FAIL : le rejeu ne reproduit PAS les etats du disque -> STOP (rapporter, "
              "ne PAS lancer les bras masques).")
        sys.exit(1)

    # ---------------- S2 : bras subject (2 videos v2) --------------------------------------
    print("[S2] bras subject : stats, alignement hdirect_masked, determinisme (2 videos v2)")
    v2 = first_pair_videos("v2", 2)
    for meta in v2:
        n_tok = int(meta["num_frame_per_block"]) * C.TOKENS_PER_FRAME
        keep1, info1 = BS.maskwrite_keep(meta, "subject")
        keep2, info2 = BS.maskwrite_keep(meta, "subject")
        det = info1 == info2 and all(
            (a is None and b is None) or (a is not None and b is not None
                                          and np.array_equal(a, b))
            for a, b in zip(keep1, keep2))
        check(f"S2 {meta['video_id']} keep deterministe (2 appels)", det)

        kc = [k for k in info1["kept_counts"] if k > 0]
        n_skip = len(info1["skipped_writes"])
        print(f"       {meta['video_id']}: {info1['n_writes']} ecritures | tokens sujet/chunk "
              f"moy {np.mean(kc) if kc else 0:.1f} min {min(kc) if kc else 0} "
              f"max {max(kc) if kc else 0} (sur {n_tok}) | sautees {n_skip} "
              f"({n_skip / info1['n_writes']:.0%})")

        # alignement = LA MEME fonction que hdirect_masked : quand le repli de LECTURE ne
        # joue pas (fb=False), la selection de lecture et la selection d'ecriture sont LES
        # MEMES indices ; quand il joue, les tokens-seuil sont un SOUS-ENSEMBLE du top-k.
        grids = C.video_mask_grids(meta["video_id"])
        align_ok, n_fb = True, 0
        for j in range(info1["n_writes"]):
            m_read, fb = C.chunk_token_mask(meta, j, grids, n_tokens=n_tok)  # chemin LECTURE
            idx_read = np.flatnonzero(m_read)
            idx_write = keep1[j] if keep1[j] is not None else np.empty(0, dtype=np.int64)
            if fb:
                n_fb += 1
                align_ok &= set(idx_write.tolist()) <= set(idx_read.tolist())
            else:
                align_ok &= np.array_equal(idx_read, idx_write)
        check(f"S2 {meta['video_id']} alignement == chunk_token_mask "
              f"({info1['n_writes'] - n_fb} chunks exacts, {n_fb} chunks repli-lecture "
              f"ou seuil-pur subset)", align_ok)

        ages = C.state_ages(meta)
        got_a = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=keep1)
        got_b = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=keep2)
        check(f"S2 {meta['video_id']} rejeu masque deterministe (2 rejeux bit a bit)",
              states_equal(got_a, got_b))
        # controle de sanite : le bras masque doit DIFFERER du bras none (sinon le masque
        # n'a rien filtre, ce qui est impossible avec un sujet a 5-12 % de l'image)
        got_n = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=None)
        check(f"S2 {meta['video_id']} bras subject != bras none (le filtre agit)",
              not states_equal(got_a, got_n))
        del got_a, got_b, got_n

    # ---------------- S3 : bras bg_matched (1 video nominale + le cas limite) --------------
    print("[S3] bras bg_matched : appariement chunk a chunk, intersection nulle")
    vids_all, _ = C.pair_dataset(verbose=False, corpus="v2")
    byid = C.vids_by_id(vids_all)
    # v2[0] = le cas nominal ; twin:9000000 = le cas limite constate au dry-check (sujet
    # > 50 % d'un chunk : appariement exact impossible, regle du complement-entier + deficit).
    metas3 = [v2[0]] + ([byid["twin:9000000"]]
                        if "twin:9000000" in byid and v2[0]["video_id"] != "twin:9000000"
                        else [])
    for meta in metas3:
        ks, infs = BS.maskwrite_keep(meta, "subject")
        kb1, infb1 = BS.maskwrite_keep(meta, "bg_matched")
        kb2, infb2 = BS.maskwrite_keep(meta, "bg_matched")
        check(f"S3 {meta['video_id']} tirage bg deterministe (2 appels)",
              infb1 == infb2 and all(
                  (a is None and b is None) or (a is not None and b is not None
                                                and np.array_equal(a, b))
                  for a, b in zip(kb1, kb2)))
        check(f"S3 {meta['video_id']} memes ecritures sautees",
              infs["skipped_writes"] == infb1["skipped_writes"])
        deficit = {int(j): int(d) for j, d in infb1["bg_deficit"].items()}
        counts_ok = all(
            (kb == 0 and kcs == 0) or
            (j in deficit and kb == infb1["n_tokens"] - infs["subject_counts"][j]
             and deficit[j] == infs["subject_counts"][j] - kb) or
            (j not in deficit and kb == kcs)
            for j, (kcs, kb) in enumerate(zip(infs["kept_counts"], infb1["kept_counts"])))
        inter_ok = all((a is None and b is None)
                       or (a is not None and b is not None
                           and np.intersect1d(a, b).size == 0)
                       for a, b in zip(ks, kb1))
        lab = (f"deficits declares {deficit}" if deficit
               else f"kept={infb1['kept_counts']}")
        check(f"S3 {meta['video_id']} compte apparie chunk par chunk "
              f"({len(deficit)} chunk(s) en regle-deficit)", counts_ok, lab)
        check(f"S3 {meta['video_id']} intersection sujet/fond NULLE sur chaque chunk",
              inter_ok)

    # ---------------- S4a : keep PLEIN == chemin historique, bit a bit ---------------------
    print("[S4a] keep = TOUS les tokens (chemin par-ecriture) == keep=None (historique)")
    meta = v2[0]
    n_tok = int(meta["num_frame_per_block"]) * C.TOKENS_PER_FRAME
    ages = C.state_ages(meta)
    full = [np.arange(n_tok, dtype=np.int64) for _ in range(len(meta["write_chunks"]))]
    got_full = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=full)
    got_none = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=None)
    s4a = check(f"S4a {meta['video_id']} bit a bit (3 blocs x {len(ages)} ages)",
                states_equal(got_full, got_none))
    del got_full, got_none

    # ---------------- S4b : signatures stream == signatures etats-disque -------------------
    print("[S4b] signatures STREAM (etats en memoire) == signatures depuis ETATS-DISQUE "
          "(bras subject, 1 video)")
    from build_probes import PROBES_PT
    pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    probes = pk["probes"].to("cuda")
    assert list(pk["blocks"]) == blocks, f"batterie {pk['blocks']} != cellules {blocks}"
    keep_s, _ = BS.maskwrite_keep(meta, "subject")
    states_mem = BS.replay_video(cells, blocks, meta, cache, ages, "cuda", keep=keep_s)
    tmpd = os.path.join(C.HEAVY, "oracle_maskwrite", "selftest_s4_tmp")
    os.makedirs(tmpd, exist_ok=True)
    s4b = True
    try:
        for a in ages:   # memes champs que build_states.main (le format du mode disque)
            torch.save({"video_id": meta["video_id"], "age_name": a["name"],
                        "write_index": int(a["i"]), "n_writes": len(meta["write_chunks"]),
                        "blocks": blocks, "state": states_mem[int(a["i"])],
                        "write_mask": "subject"},
                       BS.state_path(meta["video_id"], a["name"], tmpd))
        for a in ages:
            pay_s = B.load_payload("state", meta, a, cache, blocks, "cuda",
                                   states=states_mem)
            pay_d = B.load_payload("state", meta, a, cache, blocks, "cuda", state_dir=tmpd)
            sig_s = B.signature("state", probes, pay_s, blocks)
            sig_d = B.signature("state", probes, pay_d, blocks)
            same = torch.equal(sig_s, sig_d) and torch.equal(
                sig_s.to(torch.float16), sig_d.to(torch.float16))
            s4b &= check(f"S4b {meta['video_id']} {a['name']} bit a bit (fp32 ET fp16 stocke)",
                         same)
            del pay_s, pay_d, sig_s, sig_d
    finally:
        for a in ages:
            p = BS.state_path(meta["video_id"], a["name"], tmpd)
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmpd) and not os.listdir(tmpd):
            os.rmdir(tmpd)
    del states_mem

    print()
    if _failed:
        stop4 = not (s4a and s4b)
        print(f"SELFTEST MASKWRITE : {len(_failed)} ECHEC(S) : {_failed}"
              + ("\nS4 FAIL => STOP : PAS de run stream ; repli = mode disque bras subject "
                 "d'abord, avec purge." if stop4 else ""))
        sys.exit(1)
    print("SELFTEST MASKWRITE : TOUT PASSE (S0-S4). Mode stream VALIDE (S4a plomberie "
          "par-ecriture, S4b stream==disque) ; run complet : run_oracle_maskwrite.sh.")


if __name__ == "__main__":
    main()
