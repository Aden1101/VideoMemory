#!/usr/bin/env python
# ============================================================================================
# MODE INTER-CORPUS (`--stage cross`) -- SELFTESTS AVANT LE RUN DE VERDICT C2.
#
# Le mode entraine sur UN corpus et teste sur un AUTRE. Ce qu'il faut prouver, c'est
# exactement ce que le protocole exige : ZERO FUITE du corpus de test, et un instrument qui
# repond au signal et pas au bruit.
#
#   X1  ZERO FUITE : saccager entierement le corpus de TEST ne bouge NI le blanchiment, NI la
#       base ACP, NI les hyperparametres choisis, NI les poids appris (bit a bit) -- le seul
#       chemin par lequel le test entre est la derniere evaluation
#   X2  SIGNAL : deux corpus disjoints construits par LA MEME loi (le sujet est dans
#       l'entree) -> le sondeur entraine sur l'un doit reussir sur l'autre
#   X3  BRUIT : meme protocole, entree sans signal -> retour au hasard (le mode ne fabrique
#       pas de succes)
#   X4  PERMUTE : le controle d'effondrement retombe au hasard sur le corpus de test
#   X5  SOUS-ENSEMBLES + p BINOMIAL : les comptes par sous-ensemble sont coherents, et le p
#       binomial rend bien la valeur du registre (12/15 -> 0.0176, 10/12 -> 0.0193)
#   X6  GARDE-FOUS : --stage cross refuse sans --audited ; refuse train==test ; refuse un
#       recouvrement de paires entre les deux corpus
#
# Usage :  python reference/runs/probe1/selftest_cross.py
# ============================================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import train_probe as T  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failed = []


def check(name, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        _failed.append(name)
    return ok


def synth_corpus(signal=True, n_pairs=30, n_ages=4, D=256, seed=0, snr=1.0, cidx0=0, A=None):
    """Meme structure que `load_dataset`, avec une plage de corpus_idx DEDIEE (`cidx0`) : deux
    appels a cidx0 disjoints font deux corpus sans aucune paire commune. `A` (partagee) = la
    loi qui lie l'embedding du sujet a la signature : c'est ce qui rend le transfert possible,
    exactement comme deux corpus de videos partagent la physique du modele."""
    g = torch.Generator().manual_seed(seed)
    if A is None:
        A = torch.randn(C.DINO_DIM, D, generator=g)
    split_of = (["train"] * 18 + ["val"] * 5 + ["test"] * 7) * 4
    videos, X, s_vid, s_age, pairs = [], [], [], [], {}
    for p in range(n_pairs):
        base = len(videos)
        cidx = cidx0 + p
        e = Fn.normalize(torch.randn(2, C.DINO_DIM, generator=g), dim=-1)
        pairs[cidx] = {"corpus_idx": cidx, "pair_id": p, "main": f"m{cidx}",
                       "twin": f"t{cidx}", "board_no": p + 1, "corpus": "synth"}
        for off in range(2):
            refs = Fn.normalize(e[off][None] + 0.05 * torch.randn(5, C.DINO_DIM, generator=g),
                                dim=-1)
            videos.append({"video_id": f"synth:{cidx}:{off}", "refs": refs,
                           "twin_index": base + (1 - off), "split": split_of[p],
                           "pair_id": p, "corpus_idx": cidx, "board_no": p + 1})
            for a in range(n_ages):
                noise = torch.randn(D, generator=g)
                x = (e[off] @ A) * snr + noise if signal else noise
                X.append(x.to(torch.float16))
                s_vid.append(base + off)
                s_age.append(f"a{a}")
    return {"videos": videos, "X": torch.stack(X, 0), "s_vid": torch.tensor(s_vid),
            "s_age": s_age, "source": "synth", "target": "subject", "dropped_pairs": [],
            "features": "pool", "corpus": f"synth{cidx0}", "pairs": pairs}, A


SMALL = {"sig_dim": [0], "hidden": [256], "depth": [2], "lr": [1e-3], "wd": [1e-2],
         "temp": [0.05], "dropout": [0.1]}      # grille d'UN point : les selftests vont vite


def main():
    C.set_determinism(0)
    dev = "cpu"

    # ---------------- X1 : zero fuite ------------------------------------------------------
    print("X1 ZERO FUITE : saccager le corpus de TEST ne change RIEN a l'entrainement")
    tr, A = synth_corpus(signal=True, seed=1, cidx0=0)
    te, _ = synth_corpus(signal=True, seed=2, cidx0=1000, A=A)
    te_bad = dict(te)
    te_bad["X"] = te["X"].clone() * 100 + 7        # on saccage TOUT le corpus de test
    te_bad["pairs"] = te["pairs"]
    r1 = T.cross_run(dict(tr), te, device=dev, seeds=(0,), grid=SMALL, verbose=False)
    tr2, _ = synth_corpus(signal=True, seed=1, cidx0=0)   # meme corpus d'entrainement, neuf
    r2 = T.cross_run(tr2, te_bad, device=dev, seeds=(0,), grid=SMALL, verbose=False)
    check("X1 memes hyperparametres choisis", r1["cfg"] == r2["cfg"], f"{r1['cfg']}")
    check("X1 meme val interne (le val du corpus d'entrainement n'a pas bouge)",
          r1["val_acc_internal"] == r2["val_acc_internal"],
          f"{r1['val_acc_internal']:.4f} vs {r2['val_acc_internal']:.4f}")
    # le blanchiment/ACP : ajuste sur le corpus d'ENTRAINEMENT, donc identique des deux cotes
    p1 = T.Preproc(tr["X"][T.split_rows(tr, "train")])
    p2 = T.Preproc(tr2["X"][T.split_rows(tr2, "train")])
    check("X1 blanchiment + base ACP identiques (jamais ajustes sur le test)",
          torch.equal(p1.mu, p2.mu) and torch.equal(p1.sd, p2.sd)
          and torch.equal(p1.W_full, p2.W_full))
    # les poids appris : meme graine, meme corpus d'entrainement -> bit a bit
    m1, _ = T.train_one(dict(tr, _pre=None, _Z={}), r1["cfg"], seed=0, device=dev)
    m2, _ = T.train_one(dict(tr2, _pre=None, _Z={}), r1["cfg"], seed=0, device=dev)
    same_w = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(),
                                                   m2.state_dict().values()))
    check("X1 poids appris identiques bit a bit", same_w)

    # ---------------- X2 / X3 / X4 : signal, bruit, permute --------------------------------
    print("X2/X3/X4 signal -> transfert ; bruit -> hasard ; permute -> hasard")
    s = r1["summary"]["all"]
    check("X2 SIGNAL : le sondeur entraine sur un corpus reussit sur l'AUTRE",
          s["acc_pair"] >= 0.85,
          f"{s['n_pairs_correct']:.1f}/{s['n_pairs']} (pair={s['acc_pair']:.3f}, "
          f"p_binom={s['binom_pair_p']:.3g})")
    trn, An = synth_corpus(signal=False, seed=3, cidx0=0)
    ten, _ = synth_corpus(signal=False, seed=4, cidx0=1000, A=An)
    rn = T.cross_run(trn, ten, device=dev, seeds=(0, 1), grid=SMALL, verbose=False)
    sn = rn["summary"]["all"]
    check("X3 BRUIT PUR : retour au hasard (0.30-0.70)", 0.30 <= sn["acc_pair"] <= 0.70,
          f"pair={sn['acc_pair']:.3f}")
    check("X4 PERMUTE (25 derangements) : hasard sur le corpus de test",
          0.35 <= r1["summary"]["permuted"]["acc_pair_25perm"] <= 0.65,
          f"{r1['summary']['permuted']['acc_pair_25perm']:.3f}")

    # ---------------- X5 : sous-ensembles + p binomial --------------------------------------
    print("X5 sous-ensembles coherents + p binomial = celui du registre")
    per_row, pair_marg = T._agg_rows([r1["per_seed"][0]["per_row"]])
    keep = set(sorted(pair_marg)[:10])
    sub = T.subset_report(per_row, pair_marg, keep)
    check("X5 un sous-ensemble de 10 paires en rend bien 10", sub["n_pairs"] == 10)
    check("X5 acc_pair == n_pairs_correct / n_pairs",
          abs(sub["acc_pair"] - sub["n_pairs_correct"] / sub["n_pairs"]) < 1e-9)
    from scipy.stats import binomtest
    p12_15 = float(binomtest(12, 15, 0.5, alternative="greater").pvalue)
    p10_12 = float(binomtest(10, 12, 0.5, alternative="greater").pvalue)
    p11_15 = float(binomtest(11, 15, 0.5, alternative="greater").pvalue)
    check("X5 binomial du registre : 12/15 = 0.0176 (< 0.05) et 11/15 = 0.059 (ne passe pas)",
          abs(p12_15 - 0.0176) < 5e-4 and p11_15 > 0.05,
          f"12/15 -> {p12_15:.4f} ; 11/15 -> {p11_15:.4f}")
    check("X5 binomial secondaire : 10/12 = 0.0193", abs(p10_12 - 0.0193) < 5e-4,
          f"{p10_12:.4f}")

    # ---------------- X6 : garde-fous ------------------------------------------------------
    print("X6 garde-fous du mode")
    import subprocess
    q = subprocess.run([sys.executable, os.path.join(HERE, "train_probe.py"), "--stage", "cross",
                        "--train-corpus", "v2", "--test-corpus", "v3"],
                       capture_output=True, text=True)
    check("X6 `--stage cross` REFUSE sans --audited",
          q.returncode != 0 and "REFUS" in (q.stdout + q.stderr))
    try:
        T.cross_run(dict(tr), dict(tr), device=dev, seeds=(0,), grid=SMALL, verbose=False)
        ok = False
    except AssertionError as e:
        ok = "communes" in str(e).lower() or "fuite" in str(e).lower()
    check("X6 recouvrement de paires train/test -> AssertionError", ok)

    print()
    if _failed:
        print(f"SELFTEST CROSS : {len(_failed)} ECHEC(S) : {_failed}")
        sys.exit(1)
    print("SELFTEST CROSS : TOUT PASSE (X1-X6). Le mode inter-corpus est pret pour la "
          "relecture, puis l'etape 3.")


if __name__ == "__main__":
    main()
