#!/usr/bin/env python
"""ETAPE 1, application MECANIQUE des SEUILS SCELLES (version v2, 2026-08-05).

Ce script ne calcule rien : il lit les JSON produits par train_probe.py et applique les
regles. Il n'y a donc aucune place pour une lecture narrative apres coup.

Les regles, telles qu'elles sont scellees au registre (memory2_probes_registry.md) :

  1. VALIDITE DU PROTOCOLE (borne). Texte d'origine : "le meme pipeline sur les activations
     brutes doit atteindre >= 90 % de precision par paire". re-ANCRAGE scelle le 05/08
     (section PLAFOND HUMAIN) apres le verdict d'oeil d'Aden : sur les materiaux v1 les deux
     jumeaux d'une paire sont souvent LE MEME individu (5 paires "quasi identiques", plafond
     humain estime ~74 %), donc 90 % etait inatteignable MEME pour un HUMAIN et ne pouvait
     pas juger l'instrument. La borne se juge desormais sur le sous-ensemble "CLAIREMENT
     differents" {01,02,11,12,19,27} = 6 paires, seuil >= 5/6 (mission 04/08). Le 90 % global
     reste imprime, comme information, jamais comme verdict.
  2. CONTROLE EFFONDREMENT : etats permutes entre videos -> precision <= 55 %.
  3. SUCCES de l'etat : precision-sujet >= 65 % ET Wilcoxon p < 0.05 sur les marges ET
     (precision-sujet - precision-fond) >= +10 points.
  4. zone GRISE (55-75 % sans les 3 conditions) : +30 paires generees AVANT toute conclusion.
     ECHEC : < 55 % ou ecart sujet-fond nul -> ARRET du chantier memoire.

Lecture : "precision par paire" se lit PARTOUT en par PAIRE (reglage final n 1 du 04/08 au
soir : c'etait la lettre du protocole, et l'audit T9b a montre que la lecture par-ligne
plafonne a 0,911 meme a signal parfait). PRIMARY est fige a "pair" ; la lecture par ligne est
imprimee a cote.

Procedure : la primaire est la validation CROISEE par paires sur les 30 paires
(train_probe.py --stage cv). Les fichiers de l'ancien split fixe 18/5/7 (final_*.json du
04/08) ne sont plus lus du tout : en l'absence de `cv_*.json`, ce script sort sans verdict.

Usage :  python reference/runs/probe1/gates.py [--corpus v1|v2] [--features pool|raw]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

T_UPPER = 0.90              # regle 1, lecture GLOBALE (information)
T_UPPER_DECISIVE = 5        # regle 1 re-ancree : >= 5 paires sur 6 "clairement differentes"
T_PERM = 0.55               # regle 2
T_SUCCESS = 0.65            # regle 3
T_GAP = 0.10                # regle 3 (10 points)
T_P = 0.05                  # regle 3
T_GREY_LO, T_GREY_HI = 0.55, 0.75      # regle 4

# La lecture est FIGEE a "pair" (reglage final n 1, scelle avant tout score v2).
PRIMARY = "pair"
SUBSET_LABEL = {"all": "30 paires", "agent": "sous-ens. agent (regle des masques)",
                "aden15": "sous-ens. oeil d'Aden", "decisive": "clairement differents",
                "valid25": "masque valide (PRIMAIRE v2)", "strong19": "clairement differents"}

# La regle 1 par corpus, citee du registre, jamais choisie ici :
#   v1  "hdirect_masked >= 5/6 (v1, decisif)"      -> sous-ensemble {01,02,11,12,19,27}
#   v2  "hdirect_masked >= 16/19 (v2, fortes)"     -> les 19 paires "clairement differentes" v2
# (sections "SCELLEMENT final v2", "labels OEIL D'ADEN, corpus v2" et "CORRECTIF de borne").
RULE1 = {"v1": ("decisive", T_UPPER_DECISIVE), "v2": ("strong19", 16)}
# Le sous-ensemble PRIMAIRE des regles 2/3/4, par corpus (registre : v1 = les 30 paires ;
# v2 = "Primaire = les 25 valides ; rapport separe sur les 19 fortes").
PRIMARY_SUBSET = {"v1": "all", "v2": "valid25"}
# La source de la borne. `hdirect` = pooling uniforme (borne du 05/08 matin, echouee sur les
# deux corpus) ; `hdirect_masked` = pooling sous le masque-sujet, la source scellee au registre
# le 05/08 nuit ("CORRECTIF de borne") et donc le defaut ici.
UPPER_SOURCES = ("hdirect_masked", "hdirect")


def cv_path(source, target, features, pretrain, corpus):
    tag = f"{source}_{target}_{features}" + ("_pre" if pretrain else "") \
        + ("" if corpus == "v1" else f"_{corpus}")
    return os.path.join(C.RUNS_DIR, f"cv_{tag}.json")


def read_cv(source, target, args):
    p = cv_path(source, target, args.features, args.pretrain, args.corpus)
    return (C.load_json(p) if os.path.exists(p) else None), p


def acc(summary, name="all", reading=None):
    r = reading or PRIMARY
    s = summary.get(name)
    return None if s is None else s["acc_pair" if r == "pair" else "acc_row"]


def subset_lines(res, title):
    """Le tableau des sous-ensembles PRE-enregistres pour un bras, rapportes cote a cote.
    Aucun d'eux ne rend de verdict, sauf `decisive` pour la regle 1 (re-ancrage scelle)."""
    out = [f"  {title}"]
    s = res["summary"]
    for name in ("all", "agent", "aden15", "decisive", "valid25", "strong19"):
        if name not in s:
            continue
        d = s[name]
        out.append(f"    {SUBSET_LABEL[name]:34s} n={d['n_pairs']:2d}  "
                   f"pair={d['acc_pair']:.3f} ({d['n_pairs_correct']:.1f}/{d['n_pairs']})  "
                   f"row={d['acc_row']:.3f}  marge={d['margin_mean']:+.4f}  "
                   f"p={d['wilcoxon_pair_p']:.4g}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="v1", choices=("v1", "v2"))
    ap.add_argument("--features", default="pool", choices=("pool", "raw"))
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--upper-source", dest="upper_source", default=UPPER_SOURCES[0],
                    choices=UPPER_SOURCES,
                    help="source de la BORNE de validite (regle 1). Defaut = la source scellee "
                         "au registre le 05/08 nuit.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sub1, thr1 = RULE1[args.corpus]
    prim = PRIMARY_SUBSET[args.corpus]
    up, up_p = read_cv(args.upper_source, "subject", args)
    st, st_p = read_cv("state", "subject", args)
    bg, bg_p = read_cv("state", "background", args)
    lines, verdict = [], None
    lines.append(f"CORPUS = {args.corpus} | FEATURES = {args.features} | "
                 f"PRE-ENTRAINEMENT = {args.pretrain} | LECTURE PRIMAIRE = '{PRIMARY}' | "
                 f"BORNE = {args.upper_source} | REGLE 1 = {sub1} >= {thr1} | "
                 f"PRIMAIRE = {prim}")

    if up is None:
        print(f"Aucun run de validation croisee : {up_p} absent.\n"
              "C'est ATTENDU tant que l'audit externe du code v2 n'a pas eu lieu "
              "(`train_probe.py --stage cv --audited`).")
        return

    u_dec = up["summary"].get(sub1)
    u_all = acc(up["summary"], "all")
    lines += subset_lines(up, f"BORNE DE VALIDITE, source {args.upper_source} (activations "
                              "poolees DIRECTEMENT, sans aucune ecriture dans la boucle)")
    if u_dec is None:
        lines.append(f"  REGLE 1 : sous-ensemble '{sub1}' indisponible sur ce "
                     "corpus -> la borne se juge sur les 30 paires.")
        ok1 = u_all >= T_UPPER
        lines.append(f"  REGLE 1 (lecture globale) : {u_all:.3f} (seuil >= {T_UPPER}) -> "
                     f"{'PASS' if ok1 else 'PROTOCOLE INVALIDE'}")
        n_ok, n_dec = None, None
    else:
        n_ok = int(round(u_dec["n_pairs_correct"]))
        n_dec = int(u_dec["n_pairs"])
        ok1 = n_ok >= thr1
        lines.append(f"  REGLE 1 RE-ANCREE (clairement differents) : {n_ok}/{n_dec} "
                     f"(seuil >= {thr1}/{n_dec}) -> "
                     f"{'PASS' if ok1 else 'PROTOCOLE INVALIDE'}")
        lines.append(f"  [information : lecture globale 30 paires = {u_all:.3f}, "
                     f"ancien seuil {T_UPPER}, inatteignable meme pour un humain sur les "
                     f"materiaux v1 (plafond humain estime ~0.74)]")
    if not ok1:
        verdict = ("PROTOCOLE INVALIDE : la borne de validite n'est pas atteinte. On repare "
                   "l'instrument ou on change de materiaux. AUCUNE conclusion sur la memoire "
                   "n'est autorisee.")

    if st is None or bg is None:
        lines.append(f"REGLE 2/3/4 : runs `state` manquants ({st_p} / {bg_p}).")
        print("\n".join(lines))
        return
    # Garde-fou : un JSON `state` produit AVANT le gel des sous-ensembles de ce corpus ne
    # porte pas le sous-ensemble primaire. On refuse de bricoler une lecture de rechange :
    # ces bras se relancent, ils sont bon marche.
    if prim not in st["summary"] or prim not in bg["summary"]:
        lines.append(f"REGLE 2/3/4 : les runs `state` du corpus {args.corpus} sont ANTERIEURS "
                     f"au gel des sous-ensembles (pas de '{prim}' dans leur resume), "
                     f"relance `train_probe.py --stage cv --audited --source state "
                     f"--target subject|background --corpus {args.corpus}`.")
        print("\n".join(lines))
        return

    lines += subset_lines(st, "ETAT, cible SUJET")
    lines += subset_lines(bg, "ETAT, cible FOND (le controle)")
    s = acc(st["summary"], prim)
    b = acc(bg["summary"], prim)
    # regle 2 : la lecture par paire, MOYENNEE sur les 25 derangements (audit 2 du 05/08).
    # `acc_pair` ne porte que le premier tirage : a 30 paires, un tirage unique a ~29 % de
    # chance de depasser 0,55 alors que la vraie valeur est 0,50, la regle 2 aurait echoue
    # au hasard du tirage. On retombe sur l'ancien champ si le JSON est anterieur au correctif.
    perm1 = acc(st["summary"], "permuted")
    perm = st["summary"]["permuted"].get("acc_pair_25perm", perm1)
    perm25 = st["summary"]["permuted"].get("acc_row_25perm")
    p = st["summary"][prim]["wilcoxon_pair_p"]
    gap = s - b
    lines.append("")
    lines.append(f"REGLE 2 controle etats permutes  : pair={perm:.3f} (25 derangements ; "
                 f"1er tirage {perm1:.3f}, row 25 derangements {perm25:.3f}) "
                 f"(seuil <= {T_PERM}) -> "
                 f"{'PASS' if perm <= T_PERM else 'ECHEC (le sondeur triche)'}")
    lines.append(f"REGLE 3 precision-sujet (etat)   : {s:.3f}  (seuil >= {T_SUCCESS})")
    lines.append(f"        Wilcoxon p (marges/paire): {p:.4g} (seuil < {T_P})")
    lines.append(f"        precision-fond (etat)    : {b:.3f}")
    lines.append(f"        ecart sujet - fond       : {gap:+.3f} (seuil >= +{T_GAP})")
    if gap <= 0.0:
        lines.append("        NOTE : ecart sujet-fond NON POSITIF. La grille (d) du plan "
                     "appelle ce cas 'enregistreur d'ambiance -> ARRET' ; la regle 4 scellee ne "
                     "declare l'echec que si l'ecart est NUL. Arbitrage Aden.")

    # `ok1` est en conjonction : sans la borne de validite, AUCUNE conclusion sur la memoire
    # n'est autorisee, le champ `success` du JSON ne doit donc jamais pouvoir valoir `true`
    # sous un verdict "PROTOCOLE INVALIDE" (audit 2 du 05/08).
    success = ok1 and (s >= T_SUCCESS) and (p < T_P) and (gap >= T_GAP) and (perm <= T_PERM)
    if verdict is None:
        if success:
            verdict = ("(c) SUCCES : l'identite est extractible des poids rapides, TIREE PAR LE "
                       "SUJET -> feu vert etapes 2-3, la tete de sondage devient un outil.")
        elif s < T_GREY_LO or abs(gap) < 1e-9:
            verdict = ("ECHEC : precision < 55 % ou ecart sujet-fond nul -> ARRET du chantier "
                       "memoire, le sondage negatif est la contribution.")
        elif T_GREY_LO <= s <= T_GREY_HI:
            verdict = ("ZONE GRISE (55-75 % sans les 3 conditions) : +30 paires generees (une "
                       "nuit) AVANT toute conclusion.")
        else:
            verdict = ("(b) au-dessus du hasard mais PAS specifiquement par le sujet -> memoire "
                       "holistique de scene : ARRET aussi.")
    print("\n".join(lines))
    print("\nVERDICT SCELLE : " + verdict)
    # la source de la borne entre dans le NOM : le verdict rendu avec l'ancienne borne
    # (`gates_v1_pool_pre.json`, 05/08 02:37) reste sur disque, il n'est jamais ecrase.
    out = args.out or os.path.join(C.RUNS_DIR, f"gates_{args.corpus}_{args.features}"
                                              + ("_pre" if args.pretrain else "")
                                              + f"_{args.upper_source}.json")
    C.save_json({"primary_reading": PRIMARY, "corpus": args.corpus, "features": args.features,
                 "pretrain": bool(args.pretrain), "upper_source": args.upper_source,
                 "rule1_subset": sub1, "rule1_threshold": thr1, "primary_subset": prim,
                 "rule1_decisive_correct": n_ok, "rule1_decisive_n": n_dec,
                 "rule1_pass": bool(ok1), "upper_all_pair": u_all,
                 "state_subject": s, "state_background": b, "permuted": perm,
                 "permuted_pair_1perm": perm1,
                 "permuted_row_25": perm25, "wilcoxon_pair_p": p, "gap": gap,
                 "success": bool(success),
                 "subsets": {"upper_hdirect": up["summary"], "state_subject": st["summary"],
                             "state_background": bg["summary"]},
                 "subsets_corpus_idx": up.get("subsets_corpus_idx"),
                 "verdict": verdict, "lines": lines}, out)


if __name__ == "__main__":
    main()
