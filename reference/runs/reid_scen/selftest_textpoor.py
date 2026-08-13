#!/usr/bin/env python
# ============================================================================================
# SELFTEST du PILOTE "TEXTE INSUFFISANT" (2026-08-03)
#
# Le risque silencieux souleve par Aden : avec un prompt de retour APPAUVRI ("the animal from
# earlier returns to the meadow"), le cosinus de prompt s'effondre, le routeur de la banque ne
# se declenche jamais, et le bras framebank degenere en memzero SANS RIEN DIRE -- la comparaison
# memdec20 vs framebank deviendrait "un splice contre pas de splice" au lieu de "quel contenu".
# FRAMEBANK_GT=1 fait venir le QUAND de la verite du planning, la SELECTION restant celle de la
# methode. Ce selftest verifie les deux moities :
#
#   (a) compile ;
#   (b) prompts du tier held-out, uniques, et permutation wronganchor 17 <-> 18 ;
#   (c) UNITE : _fb_select(force=...) -- force=True rappelle l'argmax MEME SOUS TAU, force=False
#       ne rappelle rien, la decision du ROUTEUR est conservee dans l'evenement, et force=None
#       reproduit exactement le comportement publie (bit a bit) ;
#   (d) AUDIT DES RUNS REELS : dans les 2 cellules, CHAQUE bras memoire a exactement 1 splice,
#       au retour (frame 180), et memzero zero -- lu dans les logs de la campagne, pas simule.
#
#   python selftest_textpoor.py
# ============================================================================================
import glob
import os
import re
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE,
                os.path.join(REPO, "reference/runs/frame_dec")]

import torch                                                            # noqa: E402

FAILS = []
LOGS = os.path.join(HERE, "logs")
# bras memoire du pilote -> (log de la campagne, motif de la ligne de splice)
SPLICE_PAT = {
    "memdec20": (["textpoor_memdec20.log", "tp2_memdec20.log"],
                 r"\[MEM_DECODE\] decoded anchor spliced"),
    "memdec21un": (["tp2_aden.log"], r"\[MEM_DECODE\] decoded anchor spliced"),
    "wronganchor": (["textpoor_wronganchor.log", "tp2_wrong.log"],
                    r"\[MEM_DECODE\] decoded anchor spliced"),
    "noiseanchor": (["tp2_noise.log"], r"\[MEM_DECODE\] decoded anchor spliced"),
    "framebank_gt": (["textpoor_fbgt.log", "tp2_base.log"],
                     r"\[FRAMEBANK\] recalled frame spliced"),
    "framebank_oracle_gt": (["textpoor_fboracle.log"], r"\[FRAMEBANK\] recalled frame spliced"),
}


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def test_a_compile():
    import py_compile
    ok = True
    for p in ("pipeline/switch_causal_inference.py", "reference/runs/reid_scen/scenarios.py",
              "reference/runs/reid_scen/run_reid_scen.py",
              "reference/runs/reid_scen/score_textpoor.py",
              "reference/runs/reid_scen/score_textpoor2.py",
              "reference/runs/reid_scen/make_eyepack_textpoor2.py"):
        try:
            py_compile.compile(os.path.join(REPO, p), doraise=True)
        except Exception as e:                                          # noqa: BLE001
            ok = False
            print(f"   {p}: {e}")
    check("(a) py_compile", ok)


def test_b_tier():
    import scenarios as S
    from collect_pairs import find_l1_corpus_path, _norm_text
    new = [_norm_text(p) for t in S.TEXTPOOR_TRIPLES for p in t] + \
          [_norm_text(p) for p in S.TEXTPOOR_PARAPHRASES]
    # les SEULS doublons tolerES sont les A/B/C d'une reprise declarEe (t19 = t17 corrigE) :
    # c'est ce qui rend les deux cellules comparables. Tout autre doublon est un bug.
    dup_ok = sum(3 for _ in S.TEXTPOOR_RERUN)
    check("(b1) prompts du tier distincts, hors reprises declarees",
          len(new) - len(set(new)) == dup_ok, f"{len(new)} prompts, {len(set(new))} distincts, "
          f"{dup_ok} doublons attendus ({S.TEXTPOOR_RERUN})")
    for k, v in S.TEXTPOOR_RERUN.items():
        check(f"(b1.{k}) t{k} reprend t{v} VERBATIM (A, B, C) et ne change QUE le retour",
              tuple(S.TRIPLES[k]) == tuple(S.TRIPLES[v])
              and S.PARAPHRASES[k] != S.PARAPHRASES[v])
    allp = [_norm_text(p) for p in S.scen_prompts()]
    check("(b2) aucun doublon inattendu dans TOUS les prompts du scenario",
          len(allp) - len(set(allp)) == dup_ok)
    # les prompts de retour, eux, doivent TOUS etre distincts (c'est la variable du tier)
    rets = [_norm_text(S.PARAPHRASES[t]) for t in S.TEXTPOOR_TRIPLE_IDS]
    check("(b2b) les prompts de RETOUR du tier sont tous distincts", len(rets) == len(set(rets)))
    # correctif de design : la CATEGORIE doit rester dans le prompt de retour (la lecon de t17)
    bad = [t for t in S.TEXTPOOR_TRIPLE_IDS if t != 17
           and not any(w in _norm_text(S.PARAPHRASES[t])
                       for w in ("horse", "dog", "cat", "truck", "parrot"))]
    check("(b2c) chaque retour du mini-round NOMME la categorie (la lecon de t17)", not bad,
          str(bad))
    hits = []
    with open(find_l1_corpus_path(), encoding="utf-8") as f:
        for line in f:
            c = _norm_text(line)
            if c and any(p in c or c in p for p in new):
                hits.append(c[:50])
    check("(b3) 0 hit dans le corpus d'entrainement l1/v3 (held-out, 2 sens)", not hits, str(hits[:2]))
    tier = S.TEXTPOOR_TRIPLE_IDS
    img = [S.wrong_anchor_triple(t) for t in tier]
    check("(b4) permutation wronganchor DANS le tier, sans point fixe",
          sorted(img) == sorted(tier) and all(a != b for a, b in zip(tier, img)), str(img))
    check("(b5) les tiers precedents ne bougent pas",
          S.wrong_anchor_triple(0) == 1 and S.wrong_anchor_triple(11) == 0
          and S.wrong_anchor_triple(12) == 13)
    check("(b6) S2p : 1 seul retour, a la frame 180, sujet 0, prompt 3 (paraphrase appauvrie)",
          S.revisit_segments("S2p") == [3] and S.revisit_start_frames("S2p") == [180]
          and S.seg_roles("S2p")[3] == 0 and S.SCENARIOS["S2p"]["schedule"][3][1] == 3)


class _FakePipe:
    """Juste ce qu'il faut pour appeler _fb_select hors GPU : une banque et un log."""
    def __init__(self, keys_subjects):
        from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
        self._P = P
        self._fb_bank = []
        self._fb_events = []
        for seg, (k, subj) in enumerate(keys_subjects):
            v = torch.zeros(8)
            v[k] = 1.0
            self._fb_bank.append({"seg": seg, "prompt_idx": seg, "subject": subj,
                                  "end_frame": 60 * (seg + 1), "key": v,
                                  "frame": torch.full((1, 1, 4, 2, 2), float(seg))})

    def select(self, seg_idx, prompt_idx, key_vec, subject, mode, force=None):
        # _fb_key est le SEUL point qui touche le texte : on l'injecte, tout le reste du
        # routeur (candidats non-adjacents, argmax, tau, tie, force) est le code REEL.
        self._fb_key = lambda cond: key_vec
        return self._P._fb_select(self, seg_idx, prompt_idx, None, subject=subject, mode=mode,
                                  force=force)


def test_c_forced_routing():
    os.environ["FRAMEBANK_TAU"] = "0.68"
    os.environ["FRAMEBANK_TIE"] = "recent"
    # banque : seg0 = l'ouvreur (sujet 2), seg1 = A (sujet 0). Requete = le prompt appauvri :
    # cosinus 0.62 avec l'ouvreur, 0.55 avec A -> sous tau ET argmax sur le MAUVAIS souvenir,
    # exactement la situation mesuree sur t17.
    q = torch.tensor([0.62, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.557])      # norme ~ 1
    q = q / q.norm()
    bank = [(0, 2), (1, 0)]

    p = _FakePipe(bank)
    frame, ev = p.select(3, 3, q, 0, "key", force=None)
    check("(c1) force=None = comportement publie : sous tau -> AUCUN rappel",
          frame is None and ev["recalled"] is False and "forced" not in ev,
          f"s_max = {ev.get('s_max', float('nan')):.4f}")

    p = _FakePipe(bank)
    frame, ev = p.select(3, 3, q, 0, "key", force=True)
    check("(c2) force=True : rappel FORCE malgre s_max < tau",
          frame is not None and ev["recalled"] is True and ev["forced"] is True)
    check("(c3) et la SELECTION reste celle de la methode (argmax = le MAUVAIS souvenir)",
          ev["chosen_seg"] == 0 and ev["correct"] is False)
    check("(c4) la decision du ROUTEUR est conservee a part (le resultat a documenter)",
          ev["router_recalled"] is False and ev["router_chosen_seg"] is None)

    p = _FakePipe(bank)
    frame, ev = p.select(3, 3, q, 0, "oracle", force=True)
    check("(c5) le plafond (oracle) force choisit le BON souvenir",
          frame is not None and ev["chosen_seg"] == 1 and ev["correct"] is True)

    p = _FakePipe(bank)
    frame, ev = p.select(2, 1, q, 1, "key", force=False)
    check("(c6) force=False (switch qui n'est pas un retour) : aucun rappel, mais loguE",
          frame is None and ev["recalled"] is False and ev["forced"] is False
          and len(p._fb_events) == 1)

    # tau haut + force=None : la reference publiee ne bouge pas d'un iota
    os.environ["FRAMEBANK_TAU"] = "0.90"
    p = _FakePipe(bank)
    _, ev = p.select(3, 3, q, 0, "key", force=None)
    check("(c7) rien de nouveau ne fuit dans l'evenement legacy",
          not any(k.startswith("router_") or k == "forced" for k in ev))


def test_d_audit_runs():
    """AUDIT des rollouts REELS : 1 splice par bras memoire par cellule, au retour."""
    import scenarios as S
    cells = [f"S2p t{t} s0" for t in S.TEXTPOOR_TRIPLE_IDS]
    for arm, (lognames, pat) in SPLICE_PAT.items():
        paths = [os.path.join(LOGS, n) for n in lognames if os.path.exists(os.path.join(LOGS, n))]
        if not paths:
            check(f"(d) log de {arm} present", False, str(lognames))
            continue
        txt = "".join(open(p, errors="replace").read() for p in paths)
        # une "cellule" commence a la ligne [scen] S2p tN s0 arm=...
        # un bloc = une (cellule, bras) : on ne garde QUE les blocs du bras audite, sinon un log
        # multi-bras (tp2_base = memzero + framebank_gt) melange les comptes.
        blocks = re.split(r"\[scen\] S2p t(\d+) s0 arm=", txt)[1:]
        seen, nsw = {}, 0
        for i in range(0, len(blocks), 2):
            ti, body = int(blocks[i]), blocks[i + 1]
            if body.split("\n")[0].split()[0] != arm:
                continue
            seen[ti] = len(re.findall(pat, body))
            nsw += len(re.findall(r"switch to segment 3 \(prompt 3\) at frame 180", body))
        # les cellules PRESENTES dans ce(s) log(s) doivent chacune avoir EXACTEMENT 1 splice
        ok = bool(seen) and all(v == 1 for v in seen.values())
        check(f"(d1.{arm}) exactement 1 splice par cellule rendue ({len(seen)} cellules)", ok,
              str(seen))
        check(f"(d2.{arm}) chaque cellule a son retour au segment 3 frame 180",
              nsw == len(seen), f"{nsw} switchs pour {len(seen)} cellules")
    zs = [os.path.join(LOGS, n) for n in ("textpoor_base.log", "tp2_base.log")
          if os.path.exists(os.path.join(LOGS, n))]
    if zs:
        txt = "".join(open(z, errors="replace").read() for z in zs)
        m = re.split(r"\[scen\] S2p t(\d+) s0 arm=", txt)[1:]
        nz = 0
        for i in range(0, len(m), 2):
            head = m[i + 1].split("\n")[0]
            if head.startswith("memzero"):
                nz += len(re.findall(r"spliced", m[i + 1]))
        check("(d3) memzero (jumeau apparie) ne splice JAMAIS", nz == 0, f"{nz} splices")


def main():
    print("=" * 92)
    print("SELFTEST pilote TEXTE INSUFFISANT (FRAMEBANK_GT + tier 17/18)")
    print("=" * 92)
    for t in (test_a_compile, test_b_tier, test_c_forced_routing, test_d_audit_runs):
        print(f"\n--- {t.__name__} ---")
        t()
    print("\n" + "=" * 92)
    if FAILS:
        print(f"{len(FAILS)} FAIL: {FAILS}")
        sys.exit(1)
    print("TOUT PASSE")


if __name__ == "__main__":
    main()
