#!/usr/bin/env python
# ============================================================================================
# SELFTESTS DU TOY D'ECRITURE-IDENTITE -- a passer AVANT les deux runs (mandat).
#
#   T1  py_compile du module + import propre.
#   T2  SIGNATURE BIT A BIT : le chemin DIFFERENTIABLE de `ToyTrainer.signature_of` rend
#       EXACTEMENT (torch.equal) ce que rend `build_signatures.signature(...)` suivi de
#       `compress_signature` -- c'est-a-dire le chemin du VERDICT DU SONDAGE. Sans ce test,
#       "on reutilise l'instrument valide" serait une affirmation, pas un fait.
#   T3  WARM-START (F2.3) : delta_l1 == 0.0 au step 0 (theta_w EST l1@2000-EMA).
#   T4  GRADIENT jusqu'a theta_w A TRAVERS LE REJEU : apres UN backward du seul terme
#       d'identite (rec_w = 0), le gradient de qkv.weight est fini et non nul ; le FORWARD est
#       inchange par la troncature de BPTT (meme perte a n_bptt=1) mais le GRADIENT, lui,
#       change -- donc la chaine des 25 ecritures est bien dans le graphe.
#   T5  BYTE-SAFETY --lam_id 0 : ni tete ni sondes ne sont construites, le tirage de la banque
#       de references n'a jamais lieu, et la perte du pas vaut EXACTEMENT rec_w * reco.
#   T6  REFERENCES IMPORTEES BIT A BIT : `train_probe.load_refs` == normalize(dino_subj[valid])
#       des .npz du sondage, sur les 16 videos.
#   T7  SPLIT TRIVIAL (assume) : les 8 paires sont les 8 premieres de la liste FORTE scellee,
#       et entrainement et evaluation portent sur LES MEMES 16 videos (aucun held-out).
#   T8  PARITE DE REJEU : les etats obtenus avec graphe sont bit a bit ceux obtenus sans.
#   T9  MARQUEUR DU SCORE : un signal PARFAIT donne toutes les paires ; le meme signal PERMUTE
#       retombe au hasard.
#   T10 DETERMINISME DE LA RECO : deux mesures consecutives donnent le meme nombre.
#
# Usage : CUDA_VISIBLE_DEVICES=7 python reference/runs/toy_identity/selftest_toy.py
# ============================================================================================
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import toy_identity as T  # noqa: E402  (pose sys.path pour tout le reste : a importer EN PREMIER)

import probe1_common as C  # noqa: E402
from build_signatures import compress_signature, signature  # noqa: E402
from train_probe import load_refs  # noqa: E402
from train_writer import delta_l1  # noqa: E402

OK, KO = [], []


def check(name, cond, detail=""):
    (OK if cond else KO).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""),
          flush=True)


def base_args(**kw):
    a = T.parse_args(["--stage", "run"])
    a.n_pairs, a.steps, a.bs, a.device, a.dropout = 2, 1, 1, "cuda:0", 0.0
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def main():
    import py_compile
    for f in ("toy_identity.py", "selftest_toy.py"):
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    check("T1 py_compile + import", True)

    trA = T.ToyTrainer(base_args(lam_id=1.0, rec_w=0.0))
    data = trA.data
    meta = data.metas[0]

    # ---- T3 --------------------------------------------------------------------------------
    d0 = delta_l1(trA.cells, trA.ref)
    check("T3 warm-start l1@2000-EMA (delta_l1 == 0)", d0 == 0.0, f"delta_l1 = {d0}")

    # ---- T2 / T8 / T10 ---------------------------------------------------------------------
    with torch.no_grad():
        st = trA.states_of(meta, grad=False)
        i = data.ages[meta["video_id"]][-1]
        mine = trA.signature_of(st, i)
        theirs = compress_signature(
            signature("state", trA.probes, {b: st[b][i] for b in trA.blocks},
                      trA.blocks)).reshape(-1)
    check("T2 signature identique au chemin du sondage (bitwise)",
          bool(torch.equal(mine, theirs)),
          f"dim {tuple(mine.shape)}, max|delta| = {float((mine - theirs).abs().max()):.3e}")

    st_grad = trA.states_of(meta, grad=True)
    check("T8 parite de rejeu (avec graphe == sans graphe, bitwise)",
          all(torch.equal(st_grad[b][i][k].detach(), st[b][i][k])
              for b in trA.blocks for k in range(3)))
    del st_grad
    torch.cuda.empty_cache()

    with torch.no_grad():
        r1, r2 = float(trA.reco(meta, st)), float(trA.reco(meta, st))
    check("T10 reco deterministe", r1 == r2, f"{r1:.6f} == {r2:.6f}")
    del st
    torch.cuda.empty_cache()

    # ---- T4 : le gradient traverse le rejeu, et la profondeur de BPTT compte ----------------
    trA.fit_norm()
    trB = T.ToyTrainer(base_args(lam_id=1.0, rec_w=0.0, n_bptt=1), data=data, verbose=False)
    trB.mu, trB.sd = trA.mu, trA.sd
    trB.head.load_state_dict(trA.head.state_dict())
    trA.rng = random.Random(0)
    torch.manual_seed(0)
    lA = trA.train_step([0])
    trB.rng = random.Random(0)
    torch.manual_seed(0)
    lB = trB.train_step([0])
    g = trA.masters[(trA.blocks[0], "qkv.weight")].grad
    gB = trB.masters[(trB.blocks[0], "qkv.weight")].grad
    gsum = float(g.abs().sum()) if g is not None else 0.0
    check("T4a gradient non nul jusqu'a theta_w (terme d'identite SEUL, a travers le rejeu)",
          g is not None and bool(torch.isfinite(g).all()) and gsum > 0,
          f"|sum| qkv.weight = {gsum:.3e}, VRAM {trA.peak_vram:.1f} GB")
    check("T4b forward inchange par la troncature de BPTT (meme perte)",
          lA["train/id"] == lB["train/id"], f"{lA['train/id']:.8f} == {lB['train/id']:.8f}")
    rel = float((g.float() - gB.float()).abs().sum() / g.float().abs().sum().clamp_min(1e-12))
    check("T4c la chaine des ecritures est dans le graphe (BPTT complet != n_bptt 1)",
          rel > 1e-3, f"ecart relatif des gradients = {rel:.3f}")
    del trB
    torch.cuda.empty_cache()

    # ---- T5 : byte-safety --lam_id 0 --------------------------------------------------------
    tr5 = T.ToyTrainer(base_args(lam_id=0.0, rec_w=1.0), data=data, verbose=False)
    seen = {"n": 0}
    orig = T.ToyData.ref_bank
    T.ToyData.ref_bank = lambda self, rng: (seen.__setitem__("n", seen["n"] + 1),
                                            orig(self, rng))[1]
    with torch.no_grad():
        st5 = tr5.states_of(meta, grad=False)
        rec_ref = float(tr5.reco(meta, st5))
        del st5
    l5 = tr5.train_step([0])
    T.ToyData.ref_bank = orig
    check("T5a byte-safety --lam_id 0 (aucune tete, aucune sonde, aucun tirage de banque)",
          tr5.head is None and tr5.probes is None and seen["n"] == 0
          and l5["train/id"] == 0.0)
    check("T5b perte(--lam_id 0) == rec_w * reco (bitwise)",
          abs(l5["train/loss"] - rec_ref) <= 1e-9 * max(1.0, abs(rec_ref)),
          f"loss {l5['train/loss']:.8f} vs reco {rec_ref:.8f}")
    del tr5
    torch.cuda.empty_cache()

    # ---- T6 / T7 : les 8 paires REELLES (refs et index seulement, aucun h charge) -----------
    _byid, pairs = T.select_pairs(T.N_PAIRS)
    board = [p["board_no"] for p in pairs]
    want = sorted(C.ADEN_V2_STRONG_19)[:T.N_PAIRS]
    check("T7a les 8 paires = les 8 premieres de la liste FORTE scellee (corpus v2)",
          board == list(want), f"planche {board} (attendu {list(want)})")
    bit = []
    for p in pairs:
        for role in ("main", "twin"):
            vid = p[role]
            z = np.load(os.path.join(C.MASK_DIR, C.seg_key(vid, "A") + ".npz"),
                        allow_pickle=True)
            e = torch.as_tensor(z["dino_subj"]).float()
            v = torch.as_tensor(z["valid"]).bool()
            bit.append(bool(torch.equal(load_refs(vid, "subject", seg="A"),
                                        Fn.normalize(e[v] if v.any() else e, dim=-1))))
    check("T6 references DINO = BIT A BIT celles du sondage (16 videos)", all(bit),
          f"{sum(bit)}/{len(bit)} videos identiques")
    check("T7b split trivial : entrainement == evaluation, aucun held-out (ASSUME)",
          len({m["video_id"] for m in data.metas}) == data.n_videos()
          and data.n_videos() == 2 * len(data.pairs))

    # ---- T9 : marqueurs du score ------------------------------------------------------------
    P = Fn.normalize(torch.stack([data.refs[v].mean(0) for v in range(data.n_videos())
                                  for _ in range(T.N_AGES_READ)], 0), dim=-1)
    rows = [v for v in range(data.n_videos()) for _ in range(T.N_AGES_READ)]
    s_ok = trA.score(P, rows)
    pm = float(np.mean([trA.score(P, trA.derangement(rows, 1234 + k))["n_pairs_correct"]
                        for k in range(25)]))
    check("T9a marqueur : signal PARFAIT -> toutes les paires justes",
          s_ok["n_pairs_correct"] == s_ok["n_pairs"],
          f"{s_ok['n_pairs_correct']}/{s_ok['n_pairs']} (marge {s_ok['margin_mean']:+.4f})")
    check("T9b marqueur : le MEME signal permute retombe au hasard",
          pm <= 0.5 * s_ok["n_pairs"] + 1e-9,
          f"permute {pm:.2f}/{s_ok['n_pairs']} sur 25 derangements")

    print("\n" + "=" * 80)
    print(f"SELFTESTS : {len(OK)} PASS / {len(KO)} FAIL" + (f" -> {KO}" if KO else ""))
    print("=" * 80, flush=True)
    sys.exit(1 if KO else 0)


if __name__ == "__main__":
    main()
