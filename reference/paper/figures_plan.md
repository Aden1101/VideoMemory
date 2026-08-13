# PLAN DES FIGURES ET TABLES (ICLR 2027)

Pour chaque item : ce qu'elle prouve, sa place, la SOURCE des données (chemins absolus depuis la
racine du repo `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/`), et un croquis textuel.
Budget cible : **4 figures + 2 tables en main text**, le reste en annexe.

Racines de données :
- `R = reference/runs/reid_scen/` (campagne scénarios + gates composite)
- `P = reference/runs/mem2/` (probes, écrivain 2.0, eyepacks)
- `L = reference/runs/reid_probe_l1/results/summary.json` (campagne 24 cellules "harness l1")
- `V = /shared/ssd_14T_project/adenelmi/reid_scen/videos/` (816 + 54 mp4)

## TABLE 1 (main) : matrice scénarios × méthodes

**Ce qu'elle prouve** : le renversement complet entre terrain ouvrant et terrain non-ouvrant.
C'est la table du papier ; tout le reste la sert.

**Source** :
- ligne S2/S2p (scoring propre) : `R/results/clean_rescore/summary_clean_rescore.json`
  (champs `variants.full.paired_vs_memzero.<arm>.{mean_delta,n_pos,n_neg,sign_p,old_mean_delta}`
  et `arm_means`), régénérable par `python R/rescore_clean.py --mode aggregate`
- lignes S1/S3/S3p/S4/S4p : `R/results/*_scores.json` (un JSON par cellule) agrégés par
  `python R/run_reid_scen.py --mode summary --scen S1,S3,S3p,S4,S4p` ; valeurs déjà consolidées
  dans `reference/bilan_2026-07-30.md` §2
- ligne S1 absolus (harness l1) : `L`
- **ATTENTION (vérification à faire avant soumission)** : la ligne S1 vient du harness l1 et les
  lignes S3/S4 de la campagne scen. Il faut soit tout re-agréger avec la même convention de
  référence, soit séparer visuellement les deux blocs et le dire en légende. Le fichier
  `R/results/summary_scen.json` actuellement sur disque est PARTIEL (3 cellules, ancien scoring) :
  ne pas le citer tel quel.

**Croquis** :
```
                         │ LongLive │ Echo bank │  OURS   │ Oracle1 │   n
─── A opens (sink does the work) ──────────────────────────────────────────
 S1  A→B→A               │  +0.16   │  = oracle │  +0.13  │  +0.22  │  24
 S3  A→B→C→A             │  +0.23   │   +0.30   │  +0.12  │  +0.30  │  24
 S3p + paraphrase        │  +0.25   │   +0.29   │  +0.12  │  +0.33  │  24
 S4  A→B→A→C→A           │  +0.22   │   +0.21   │  +0.11  │  +0.24  │  24
 S4p + paraphrase        │  +0.15   │   +0.20   │  −0.01  │  +0.25  │  24
─── A does NOT open (the sink holds X) ────────────────────────────────────
 S2  X→A→B→A             │ −0.10 ** │  +0.03 ns │ +0.43***│ +0.03 ns│  24
 S2p + paraphrase        │  −0.06   │  +0.09 ns │ +0.36***│ +0.09 ns│  24
```
- cellules = Δ apparié vs `memzero`, avec **signes en exposant** (24+/0−) et p du test des signes ;
  la moitié basse en gras / trame de fond pour marquer la bascule.
- colonne supplémentaire "scored against the contaminated in-rollout reference" pour S2/S2p
  uniquement (−0.378 / +0.209 / +0.208 / +0.063), avec un renvoi vers Fig. 2 et §Intégrité.
- pied de table : `memzero` absolus (S2 0.2235, S2p 0.2326) + note "`stockoff` scoré contre son
  propre A\*".

## TABLE 2 (main) : l'échelle causale des bornes

**Ce qu'elle prouve** : où est exactement le plafond, et que ce n'est ni le contenu ni le lecteur
mais le **canal état → frame**. C'est la table qui rend la limite d'instance non négociable.

**Source** : `reference/memory2_probes_report.md` §CAMPAGNE (24 cellules appariées) ; JSON
`L` ; prédictions `reference/memory2_probes_registry.md`.

**Croquis** (5 colonnes : bras / ce qu'il injecte / Δ full / signes, p / prédiction pré-enregistrée) :
```
 memon20   lecture native active            −0.018   8+/16−  p=.15   pred ≈0        ✓
 memdec12  ancre décodée, écrivain stock    +0.119  18+/6−           (repère)
 memdec20  NOUS (écrivain 2.0)              +0.126  21+/3−           (repère)
 thru      VRAIE frame → write → relue      +0.131  18+/6−  p=.023   pred +0.13-15  ✓ PIVOT
 twin      vraie frame nette, MAUVAISE inst +0.135  14+/10− p=.54    pred ≈0        ✗ réfutée
 blur24    vraie frame floutée σ=24         +0.158  23+/1−  p=2e-5   pred +0.04-14  ✗ au-dessus
 blur0     calibration VAE seule            +0.202  23+/1−
 oracle1   vraie frame parfaite             +0.224  24+/0−           (borne)
```
- barre horizontale annotée sous la table : `[canal] ────── +0.13 ─────► [contenu]`, avec la
  bande +0.119 … +0.158 grisée = "tout ce qui passe par une frame, quel que soit son contenu".

## FIG. 1 (main, teaser + mécanisme) : écrivain silencieux → write-replay → splice → α

**Ce qu'elle prouve** : la méthode en un coup d'œil, et surtout **pourquoi l'entraînement offline
est exact**.

**Source** : schéma à dessiner (pas de données), références de code pour la légende :
`wan/modules/causal_model.py` (h = norm1 avant la branche mémoire ; `x = x + mem_out` annulé sous
`MEM_ZERO_ALL`), `wan/modules/video_memory.py` (`write_only`, fast weights w0/w1/w2),
`pipeline/switch_causal_inference.py` (`_switch_recache`, hook `MEM_DECODE`, `_memdec_window`,
`_comp_alpha`). Chiffres à incruster : état 28.3 M/bloc × 3 blocs, θ_w 21.3 M, décodeur 33 M,
fenêtre 12 frames, sink 1.

**Croquis** (3 panneaux horizontaux) :
```
(a) COLLECTE (mémoire SILENCIEUSE)      (b) WRITE-REPLAY OFFLINE        (c) INFÉRENCE
 backbone gelé ──► h (donnée)            h₁ h₂ … h_N  (sur disque)      état ──► décodeur
        │  mem_out × 0  ✗                    ▼  ▼      ▼                       ▼ ancre
        └────► latents == memzero          write write write  (θ_w)       ┌──────────┐
   « h ⟂ θ_write »  ← LE THÉORÈME             ▼                            │ fenêtre  │
   selftest : replay == in-situ, bitwise    état ──► décodeur ──► loss     │ 12 frames│
                                             (VIDEO MODEL ABSENT)          └──────────┘
                                                                      slot récence ← α·ancre
```
Un encart "α" sous (c) : α=1 splice plein, α∈(0,1) composite union, α=0 = pas de rappel
(bit à bit `memdec20`).

## FIG. 2 (main) : la planche de contamination (référence vs A\* vs bras)

**Ce qu'elle prouve** : d'un coup d'œil, que la référence de scoring montre X, que la banque et
l'oracle réimportent X, et que seul notre bras rejoint A\*. C'est la figure la plus persuasive du
dossier, et elle porte la section intégrité.

**Source** : `R/eyepack_s2/S2_board_reference_vs_cleanA_vs_arms.png` (déjà produite ;
générateur `R/make_eyepack_s2.py`). Cellules t2 (chat tigré), t4 (cheval blanc), t11 (théière
cuivre), seed 0. Vidéos correspondantes : `R/eyepack_s2/S2_t{2,4,11}_s0_memzero-framebank-memdec20.mp4`.

**Croquis** (3 lignes × 5 colonnes, annotée) :
```
            scoring REF      CLEAN A*      memzero      framebank     OURS
            (S2 seg1)        (S3 seg0)     (return)     (return)      (return)
 t2  chat  │ camion pompiers│ chat tigré │ camion      │ camion      │ CHAT ✓
 t4  cheval│ potier au tour │ cheval     │ potier      │ potier      │ CHEVAL ✓
 t11 théière│ fanfare       │ théière    │ fanfare     │ fanfare     │ THÉIÈRE ✓
            └── cos(col1, col2) = 0.165 ──┘            └ cos(retour, ouverture) 0.517 vs 0.017 ┘
```
Bandeau sous la figure : "la colonne 1 est ce contre quoi la campagne scorait ; les colonnes 3 et
4 lui ressemblent, la colonne 5 ressemble à la colonne 2".

## FIG. 3 (main) : l'inversion (même 24 cellules, deux références)

**Ce qu'elle prouve** : que le renversement n'est pas un changement d'échelle mais un changement
de **signe**, cellule par cellule, et qu'il est unanime.

**Source** : `R/results/clean_rescore/summary_clean_rescore.json` : pour chaque bras,
`paired_vs_memzero.<arm>.deltas` (24 valeurs, clés `t*_s*`) = colonne "propre" et
`old_mean_delta` = moyenne de la colonne contaminée ; les deltas contaminés par cellule sont dans
`R/results/<scen>_t*_s*_scores.json`.

**Croquis** (deux panneaux côte à côte, mêmes axes) :
```
   Δ vs memzero                    contaminated ref          clean ref A*
   +0.6 │                                                     ● ●●●● ●● (ours, 24/24 > 0)
   +0.3 │        ▲▲▲ (bank +0.21)                            ▲ ▲ ▲ (bank ≈ 0)
    0.0 ├──────────────────────────────────────────────────────────────────
   −0.3 │  ● ●●●●● (ours −0.38)                              ■ ■ ■ (LongLive < 0)
        └── une ligne par cellule, appariée entre les deux panneaux (segments gris)
```
Variante plus compacte si la place manque : **slopegraph** à deux colonnes (contaminée → propre)
avec une ligne par bras et l'épaisseur = nombre de signes positifs.

## FIG. 4 (main) : le front de Pareto sujet / fond

**Ce qu'elle prouve** : qu'il y a **une seule courbe et deux boutons**, que la position est
binaire, que la dose est continue, et que le masque union sort de la droite sur les deux axes.

**Source** : `R/results/comp/comp_SS2*.json` (tous les bras, 3 cellules SS2 seed 0), scoré par
`R/score_comp.py` ; table consolidée dans `reference/composite_gates_report.md` §5.0, §6, §7 ;
plancher de bruit `comp_SS2_noisefloor.json`.

**Croquis** (scatter, x = fond head, y = sujet crop60, taille du point = cut2, couleur = dose) :
```
 sujet
  0.70 │                                      ● memdec20 (dose 1.0, cut 20.0)
  0.55 │                        ★ memdec21u (union, α)        ○ inv (dose .71)
  0.40 │                              ● 21c (.288)
  0.30 │                    ▲ bank Echo / oracle1      ○ noise-anchor control (.288)
  0.26 │                                        ■ memzero ● slot−1  ● additive-kv
       └────────────────────────────────────────────────────────────────────► fond
         0.45         0.60         0.78    0.83   0.90        0.93
   droite pointillée memzero → splice plein = "dose line" ; ★ est AU-DESSUS des deux côtés
   (+0.060 sujet, +0.108 fond) ; les points "slot−1" et "additive-kv" retombent sur memzero.
```
Encart en coin : barre "attention mass of the additive anchor = 2.04 % (uniform share 7.7 %)".

## FIG. 5 (main ou annexe) : la série turquoise / buisson / scooter

**Ce qu'elle prouve** : ce que le prompt peut repeindre (sujet générique animé) et ce qu'il ne
peut pas (objet rigide spécifique) ; que le score peut MONTER pendant que l'instance est FAUSSE.
C'est l'illustration de la limite d'instance, et elle vaut mieux qu'un paragraphe.

**Source** : `R/eyepack_fill/SS2_planche_zoom_fantome.png` (planche zoom fantôme) et les hstacks
`R/eyepack_fill/SS2_t1{2,3,4}_s0_union-vs-prioritaire-vs-plancher.mp4` et
`…_union-vs-unionbruit-vs-banque.mp4` ; générateur `R/make_eyepack_fill.py` ; chiffres
`R/results/comp/comp_SS2_fill{,prio,floor}.json` et `reference/composite_gates_report.md` §10-10.2.

**Croquis** (3 lignes = cellules, 4 colonnes = variantes de remplissage) :
```
          A* (target)   union (21u)      priority (21up)        floor (21uf)
 t12 café  veste ROUGE  fantôme flou     homme VESTE TURQUOISE  BUISSON halluciné
 t13 chien  1 chien     aplat vert       2e animal blanc        meilleur panneau
 t14 scooter scooter bleu  partiel       A NE REVIENT PAS       intermédiaire
```
Légende chiffrée : sujet 0.558 / 0.533 / 0.551, ghost_hf 0.0327 / 0.0354 / 0.0464 (memzero 0.0310),
tripwire sujet par cellule pour `21up` : **t12 +0.220, t13 −0.093, t14 −0.203**.

## FIG. 6 (annexe) : contrôles anti-illusion, deux tiers

**Ce qu'elle prouve** : que la forme fait un vrai travail (P4a, P4c) **et** qu'une moitié du gain
composite sur le tier même-décor est de l'effacement (P4b réfutée), tandis que sur le tier
cross-scène les contrôles s'effondrent (2 % et 9 %).

**Source** : `R/results/comp/comp_SS2gates_g3geom.json`, `comp_S2gates_g3.json` ;
`reference/composite_gates_report.md` §5.0, §5.2, §5.4.

**Croquis** : barres groupées, "% du chemin memzero → oracle1" par contrôle, deux panneaux
(SS2 même décor | S2 cross-scène) :
```
 composite   132 % | 118 %
 unif (dose)  58 % |   2 %      ← la forme vaut 56 % ici, 98 % là
 noise anchor 54 % |   9 %      ← LE contrôle gênant, publié
 shift        27 % |  (n/a)
```

## FIG. 7 (annexe) : test d'instance et sa calibration

**Ce qu'elle prouve** : que le négatif est un **vrai** négatif (la métrique sait faire) et non un
manque de puissance.

**Source** : `R/results/clean_rescore/instance_power.json` +
`summary_clean_rescore.json` (bloc instance) ; `reference/scenarios_design_report.md` §9.3.

**Croquis** : deux distributions appariées, une ligne par cellule :
```
 panneau A (calibration, 12 triples) : same instance 0.9329 ●───────● 0.7994 other seed  (12/12, marge +0.134)
 panneau B (notre retour, 24 cells)  : same seed   0.6577 ●─●  0.6654 other seed  (Δ −0.008 ± 0.024, 11+/13−)
 ligne pointillée : niveau inter-seed 0.799 > notre absolu 0.658
```

## FIG. 8 (annexe) : ablation P1 (ce qu'on lit vient de ce qu'on écrit)

**Source** : `P/eyepack_probes/p1_target_base_abl_ctrl.png` (déjà produite, 4 vidéos, VAE pleine
résolution) ; chiffres `/shared/ssd_14T_project/adenelmi/mem2_runs/logs/p1_block_ablation.json`
et `reference/memory2_probes_report.md` §P1.
**Croquis** : 4 lignes × 4 colonnes [cible vraie | replay base | bloc A ABLATÉ | bloc A ← autre
vidéo], avec sous chaque colonne son `norm_mse` (0.800 → 1.229) et le rappel de la ligne rouge
pré-enregistrée (Δloss < 0.02, ratée ×21).

## FIG. 9 (annexe) : la preuve d'instance-dans-l'état (cheval 3 bras)

**Source** : `P/eyepack20/horse_3way.png` (+ `anchor_memdec{12,20,20d}*.png` pour les variantes).
**Croquis** : 3 panneaux [décodeur v1.2 | contrôle | écrivain 2.0] sur une cellule single-prompt
(état 100 % A) : ancre **brune** / brune / **pâle**, légende "l'instance entre dans l'état ;
elle ne sort pas du canal (cf. Table 2)".

## TABLE A1 (annexe) : selftests et plancher de bruit
**Source** : `R/logs/selftest.log` (52/52), `R/logs/selftest_gating_2026-08-02.log` (67/67),
`P/selftest_probes.py` + `selftest_phaseB.py` (17/17), `R/selftest_gates.py` + `selftest_composite.py`
(18 + 24), `R/results/comp/comp_SS2_noisefloor.json`.
Colonnes : test / ce qu'il garantit / mesure (`max|diff|`) / verdict. Inclure les 0.000e+00
(schedule hérité == `inference()`, `framebank == oracle1` verbatim, α=0 ⇒ `memdec20`, absence de
fuite d'env, gel de l'écrivain en phase B) et le plancher 0.0008.

## TABLE A2 (annexe) : coût et empreinte
**Source** : `reference/memory2_design.md` §2.2-2.3, `reference/memory2_j3_report.md` §2, §4.
Lignes : collecte (165 vidéos, 5 175 writes, 229 GB, 43.1 MB de h par write), entraînement
(15.2 GB, 3.9 s/step à K=3/n_bptt=12/B=2 ; 2 500 steps = 1.6 h), inférence (surcoût du décodage
d'ancre : 1 décodage par retour), état O(1) vs banque O(#scènes).

## Vidéos pour le supplementary (déjà sur disque)
- `R/eyepack_s2/S2_t{2,4,11}_s0_memzero-framebank-memdec20.mp4` (hstack 3 bras, natif 2496×480,
  11.8 / 11.8 / 20.3 MB) : LE matériel qualitatif du claim principal.
- `R/eyepack_fill/*.mp4` (6 clips, union vs variantes) pour Fig. 5.
- `P/eyepack_probes/pair{1,4,7}_s0_oracle1thru.mp4` pour Table 2.
- Réserve : `V` (816 mp4 de la campagne + 54 des gates).
