# BATTERIE ARBITRE — RAPPORT DE CHIFFRES (2026-07-28)

Chiffres bruts face aux prédictions pré-enregistrées de `reference/memory2_probes_registry.md`.
**Aucune conclusion sur le cap ici** : ce document produit les nombres et les visuels, la lecture
appartient à Aden. Tout est additif, rien n'est commité.

| Livrable | Fichier |
|---|---|
| P1 (ablation bloc A, offline) | `reference/runs/mem2/probe_p1_block_ablation.py` |
| hooks d'inférence (4 arms) | `pipeline/switch_causal_inference.py` (additif, env-gated, défaut OFF) |
| arms + driver | `reference/runs/reid_probe_l1/run_reid_probe.py` |
| selftests obligatoires | `reference/runs/mem2/selftest_probes.py` |
| visuels | `reference/runs/mem2/eyepack_probes/` |
| logs / pids | `reference/runs/mem2/logs/` |
| P2/P3 (grille dosage, offline) | `reference/runs/mem2/probe_p2p3_gain_steps.py` |
| eye-pack | `reference/runs/mem2/make_eyepack_probes.py` |
| lanceur de shard (pré-check GPU inclus) | `reference/runs/mem2/launch_probes.sh` |
| JSON | `mem2_runs/logs/{p1_block_ablation,p2p3_gain_steps}.json`, `reid_probe_l1/results/summary.json` |
| wandb | `mem2_probe_p1`, `mem2_probe_p2p3` (projet `anchor-forcing-memory`) |

**Nouveaux hooks d'inférence** (tous env-gated, `_OFF` par défaut, aucun bras existant modifié —
selftest (d) le vérifie bit à bit) : `MEM_DECODE_THRU` (décodage à l'éviction),
`ORACLE_ANCHOR_DUMP` / `ORACLE_ANCHOR_SRC` (persistance et rejeu de l'ancre origA d'une autre
cellule), `ORACLE_BLUR_SIGMA` (aller-retour VAE + flou). Nouveaux bras du harness :
`oracle1thru`, `oracle1twin`, `oracle1blur`, `oracle1blur0`, `memon20`, `anchordump`.

---

## P1 — ABLATION DU BLOC CONTIGU A (offline, write-replay, 12 vidéos switch val)

Protocole exact : pour chaque vidéo, `base` = replay normal ; `abl` = replay où **TOUTES** les
écritures du segment A (indices 0..`anchor_write_index`, c.-à-d. toute écriture dont le chunk
évincé précède `switch_chunk`) sont remplacées par leur **moyenne par token** (l'écriture
non-informative de même magnitude, idiome `influence_probe`) ; `ctrl` = ces mêmes indices
remplacés par les h d'une **AUTRE vidéo** (4 vidéos). 3 positions de revisit par vidéo (ancre,
âges 5→17 écritures), `norm_mse` en convention val (non pondérée, cond ON).

### Chiffres

| régime | ‖Δétat‖/‖état‖ (moy) | par bloc b5 / b16 / b27 | norm_mse ancre base → abl | **Δloss** | signe | contrôle "autre vidéo" |
|---|---|---|---|---|---|---|
| **A2** (écrivain 2.0 + décodeur A2 = arm `memdec20`) | **0.803** [0.763, 0.826] | 0.780 / 0.781 / 0.849 | 0.800 → 1.229 | **+0.429** (méd. +0.351, [+0.058, +1.053]) | **12/12 pires** | Δloss +0.184, rel 0.712 |
| **l1** (écrivain stock l1@2000-EMA + décodeur v1.2 = arm `memdec12`) | **0.996** [0.965, 1.025] | 0.988 / 0.976 / 1.024 | 0.875 → 1.004 | **+0.128** (méd. +0.121, [−0.148, +0.599]) | **10/12 pires** | Δloss +0.137, rel 1.006 |

### Face au registre
- Ligne rouge dure n°1 de l'avocat TTT-pur (**Δloss recon < 0.02** ⇒ le +0.127 ne vient pas des
  fast weights) : **non atteinte**, de très loin — +0.429 (A2) et +0.128 (l1), soit ×21 et ×6.4
  le seuil. Le scénario tueur (P = 30 % selon l'avocat du diable) **ne s'est pas produit**.
- Fourchette de l'avocat du diable : `‖Δétat‖/‖état‖` 0.35–0.60 → **mesuré 0.80 (A2) et 1.00
  (l1)**, au-dessus de la fourchette. `Δnorm_mse` +0.10 à +0.20 → **mesuré +0.128 (l1, dans la
  fourchette) et +0.429 (A2, très au-dessus)**.
- Note de lecture (brute, non interprétative) : le régime **l1 est en régime de saturation**
  (`rel_state ≈ 1.0` : neutraliser les écritures A déplace l'état d'autant que sa propre norme)
  alors que **A2 est à 0.80 avec un Δloss 3.3× plus grand** — même déplacement d'état, effet sur
  l'ancre décodée bien plus fort chez A2.
- Contrôle : remplacer les h de A par ceux d'une autre vidéo dégrade **moins** que la moyenne
  par token chez A2 (+0.184 vs +0.429), **autant** chez l1 (+0.137 vs +0.128).

### Œil
`reference/runs/mem2/eyepack_probes/p1_target_base_abl_ctrl.png` — 4 vidéos, colonnes
[cible vraie | replay base | bloc A ABLATÉ | bloc A ← autre vidéo], VAE pleine résolution.
Le replay base garde la teinte et la disposition générale de sa cible ; l'ablaté part sur une
autre image (et exhibe le pavage) ; le contrôle donne encore une autre scène.

wandb : `mem2_probe_p1` (projet `anchor-forcing-memory`).

---

## SELFTESTS (obligatoires avant campagne) — **ALL PASS**

`reference/runs/mem2/selftest_probes.py`, un rollout court A→B→A (36 frames, switch 12, retour
27, GPU 0). Log : `/tmp/.../selftest` (rejouable en 10 min).

| test | mesure | verdict |
|---|---|---|
| (a) `oracle1twin` avec décalage de seed **0** == `oracle1` | max\|diff\| latents **0.000e+00** | **PASS bitwise** |
| (b) `oracle1blur` σ=0 ≈ `oracle1` | round-trip VAE sur le LATENT d'ancre : **rel-L2 0.492**, max\|diff\| 3.20 ; divergence du rollout vs oracle1 : σ=0 **1.15e−01** vs σ=24 **1.78e−01** | **PASS** (σ=24 diverge plus) mais **tolérance NON négligeable, voir ci-dessous** |
| (c) `oracle1thru` décode à l'éviction | `anchor_frame 11, anchor_chunk 3, evict_chunk 7, window_chunks 4` ; décodage à l'éviction = **2 écritures avant le retour** dans le selftest, **24 écritures** dans la campagne (ancre frame 80 → chunk 26, évincée/écrite au chunk 30, retour au chunk 54) ; le splice au retour réutilise la frame cachée (aucun décodage au retour) | **PASS** |
| (d) aucune fuite d'env d'un bras au suivant | `memzero` rejoué APRÈS tous les bras probes : max\|diff\| **0.000e+00** ; `MEM_WRITER` overlay restauré (`_memwriter_applied = None`) | **PASS bitwise** |
| bonus continu | le hook `ORACLE_ANCHOR_DUMP` **re-vérifie bit à bit** le fichier d'ancre déjà écrit à chaque bras MEM_ZERO_ALL de la même cellule : `max|diff| = 0.000e+00` à chaque fois (invariant « le segment A ne dépend pas du bras ») | **PASS** |

**CAVEAT MESURÉ sur le bras blur** : décoder UNE frame latente isolée (hors contexte temporel)
puis la ré-encoder coûte déjà **rel-L2 0.49** — le VAE causal 3D voit cette frame comme « la
première d'une vidéo », pas comme la frame 80 d'une séquence. Le plancher σ=0 vaut donc **64 %
de la divergence du rollout à σ=24** (0.115 vs 0.178). Le bras `oracle1blur` mesure donc
« vraie frame floutée **ET** passée par un aller-retour VAE » ; un bras de calibration σ=0 est
listé en fin de rapport (à lancer si une GPU se libère) pour attribuer les deux parts.

---

## EYE-PACK (obligatoire — `reference/runs/mem2/eyepack_probes/`)

`anchors_true_thru_twin_blur.png` — paires **1, 4, 7** (seed 0), une ligne par paire, colonnes
**[frame VRAIE (ce que splice oracle1) | thruwriter relue (décodée à l'éviction, âge 1 écriture) |
twin (ancre de la seed 1, même prompt) | blur σ=24 (aller-retour VAE inclus)]**, décodage VAE
natif 480×832. Les mp4 des cellules `oracle1thru` des 3 paires sont copiés à côté
(`pair{1,4,7}_s0_oracle1thru.mp4`).

Ce que la planche montre (description, pas interprétation) : la colonne **thruwriter à l'âge 1**
reste une tache basse fréquence avec le pavage 2×2 — la teinte de scène est là (plage, prairie
verte, mur rouge) mais le sujet n'est pas identifiable ; la colonne **twin** est parfaitement
nette, même scène, instance/pose différentes ; la colonne **blur σ=24** conserve la
disposition des couleurs du vrai sujet.

---

## CAMPAGNE — 4 ARMS × 24 CELLULES (harness l1, delta apparié vs `memzero`)

Lancée en 2 shards (GPU 0 pairs 0-3, GPU 1 pairs 4-7, pré-check >25 GB dans la commande,
`reference/runs/mem2/launch_probes.sh`), 96 rollouts de 240 frames, 20:44 → 00:01.
Logs `reference/runs/mem2/logs/probes_shard{A,B}.log`, JSON
`reference/runs/reid_probe_l1/results/summary.json`.

### Deltas appariés (24 cellules, moyenne ± sem, test des signes exact)

| bras | **full** Δ vs memzero | signes | sign-p | **crop60** Δ | signes | sign-p |
|---|---|---|---|---|---|---|
| `oracle1thru` (vraie frame → vraie écriture → relue à l'âge 1) | **+0.1312** ± 0.0382 | 18+/6− | 0.023 | **+0.1360** ± 0.0442 | 18+/6− | 0.023 |
| `oracle1twin` (vraie frame NETTE, autre instance) | **+0.1347** ± 0.0477 | 14+/10− | 0.54 | **+0.1470** ± 0.0513 | 15+/9− | 0.31 |
| `oracle1blur` (vraie frame floutée σ=24 px) | **+0.1575** ± 0.0395 | 23+/1− | 0.00002 | **+0.1724** ± 0.0434 | 20+/4− | 0.0015 |
| `oracle1blur0` (**calibration** : même aller-retour VAE, σ=0, AUCUN flou) | **+0.2019** ± 0.0428 | 23+/1− | 0.00002 | **+0.2185** ± 0.0444 | 24+/0− | 0.0000 |
| `memon20` (lecture active + écrivain A2) | **−0.0179** ± 0.0317 | 8+/16− | 0.15 | **−0.0503** ± 0.0448 | 6+/18− | 0.023 |

**Décomposition du bras blur** (le caveat du selftest (b) est levé) : `oracle1` +0.224 →
`oracle1blur0` +0.202 (**l'aller-retour VAE ne coûte que −0.022**, malgré son rel-L2 latent de
0.49) → `oracle1blur` σ=24 **+0.158** (**le flou lui-même coûte −0.044**). Le bras blur mesure
donc bien le flou, pas un artefact d'encodage. Le bras `oracle1blur0` a demandé 3 lancements
(2 OOM causés par les jobs d'un autre utilisateur atterrissant sur la GPU en cours de route).

### Repères de la même table (mêmes 24 cellules, campagnes précédentes)

| repère | full | crop60 |
|---|---|---|
| `oracle1` (borne haute, vraie frame parfaite) | +0.2243 (24+/0−) | +0.2411 (24+/0−) |
| `memdec20` (mémoire 2.0 complète : écrivain A2 + décodeur A2) | +0.1264 (21+/3−) | +0.1292 |
| `memdec12` (décodeur v1.2 sur écrivain stock) | +0.1191 (18+/6−) | +0.1271 |
| `memon` (lecture active, écrivain stock) | −0.0206 | −0.0318 |
| niveaux absolus `reid_resA_origA` | memzero 0.6858 · memdec20 0.8122 · **thru 0.8170 · twin 0.8205 · blur 0.8433** · oracle1 0.9101 · stockoff 0.8475 | memzero 0.6533 · memdec20 0.7825 · **thru 0.7893 · twin 0.8002 · blur 0.8257** · oracle1 0.8944 |

### Face aux prédictions pré-enregistrées

| probe | prédiction du registre | mesuré | verdict factuel |
|---|---|---|---|
| **THRUWRITER** | avocat TTT-pur +0.13 à +0.15 ; avocat du diable +0.10 à +0.15 avec P(>+0.18)=25 % | **+0.131 / +0.136** | **dans la fourchette des DEUX avocats**, en plein sur la valeur pivot « ≈ +0.13 » de la règle de lecture ; +0.005 seulement au-dessus de `memdec20` (+0.126) |
| **TWIN/CROSS-SEED** | modèle « contenu = tout » strict : **≈ 0** ; règle : > +0.05 ⇒ ce modèle est faux par construction | **+0.135 / +0.147** (mais 14+/10−, p=0.54 : porté par une minorité de cellules, dont pair1 à +0.54) | **> +0.05**, très au-dessus de la prédiction ≈ 0 ; à la fois **non significatif au test des signes** et **numériquement égal au thru et à memdec20** |
| **BLUR σ≈24** | avocat du diable +0.04 à +0.14 selon calibration ; P(explique tout le plateau) ≈ 55 % | **+0.158 / +0.172**, 23+/1− (calibration σ=0 : +0.202 / +0.219) | **au-dessus de la fourchette haute** ; le plus **régulier** des trois (seul signe quasi unanime) et le plus grand ; la calibration montre que le flou à σ=24 ne fait perdre que **0.044** des 0.224 de l'oracle |
| **MEMON20** | les deux avocats : ≈ 0 (+0.01 [−0.03, +0.04]) | **−0.018 / −0.050** | **≈ 0, voire négatif** (crop60 : 6+/18−, p=0.023) ; l'objection « jamais testé avec un écrivain non-aveugle » est fermée |

Note de lecture strictement factuelle : les trois bras à contenu (thru, twin, blur) tombent tous
dans la bande **+0.13 à +0.17**, c.-à-d. la même bande que `memdec12`/`memdec20` (+0.119 à +0.129),
alors que la borne `oracle1` reste à +0.224. Aucun des trois n'atteint la borne. Le seul bras qui
la rejoint est `oracle1blur0` (+0.202), c.-à-d. la **vraie frame nette de la bonne instance**.

### Échelle complète mesurée aujourd'hui (delta full, 24 cellules appariées)

```
memon20  -0.018 | memon  -0.021 || memdec12 +0.119 | memdec20 +0.126 | thru +0.131 | twin +0.135
         | blur24 +0.158 || blur0 +0.202 | oracle1 +0.224
```

### Écarts de protocole à connaître
- **3 shards prévus, 2 utilisés** pour la campagne principale : au moment du lancement seules les
  GPU 0 et 1 passaient le pré-check >25 GB (2, 3, 4, 5, 6, 7 occupées par un autre utilisateur),
  d'où un découpage pairs 0-3 / 4-7 au lieu de 0-2 / 3-5 / 6-7. Aucun impact sur les chiffres
  (les cellules sont indépendantes), seulement sur le temps de mur (3 h 15).
- `oracle1blur0` **n'est pas au registre** : je l'ai ajouté parce que le selftest (b) a montré que
  l'aller-retour VAE d'une frame isolée coûte rel-L2 0.49 en latent, ce qui rendait le bras blur
  ininterprétable sans plancher. Le plancher est mesuré : −0.022 seulement.

---

## P2/P3 — GRILLE DE DOSAGE DE L'ÉCRITURE (offline, 10 vidéos switch val, écrivain A2 + décodeur A2)

`reference/runs/mem2/probe_p2p3_gain_steps.py`, GPU 5, ~8 min. La règle d'écriture est
**monkeypatchée localement** (`VideoMemoryLayer.write_only` remplacée dans le script,
`wan/modules/video_memory.py` INTACT). Gain α appliqué **après** Muon (Newton-Schulz normalise
son entrée : scaler le gradient brut est un no-op strict à `muon_update_steps=5`), puis cap de
norme par ligne (gated-delta) inchangé. `n` = nombre de pas de gradient internes par chunk évincé.

**Selftest S0 (parité)** : le point (α=1, n=1) reproduit la règle déployée **bit pour bit**
(`max|diff| = 0.000e+00`) — sinon la grille serait nulle et non avenue.

### norm_mse de l'ancre décodée (moyenne 10 vidéos, 3 positions de revisit chacune)

| α \ n | n=1 | n=2 | n=4 |
|---|---|---|---|
| **1** (déployé) | **0.8082** | 0.8623 (+0.054) | 0.9766 (+0.168) |
| 2 | 0.8282 (+0.020) | 0.9308 (+0.123) | 1.0307 (+0.222) |
| 4 | 0.8729 (+0.065) | 0.9826 (+0.174) | 1.0513 (+0.243) |
| 8 | 0.9310 (+0.123) | 1.0143 (+0.206) | 1.0604 (+0.252) |

### saturation et déplacement d'état

| α \ n | rowcap_frac (n=1 / n=2 / n=4) | ‖Δétat‖/‖état‖ vs (1,1) |
|---|---|---|
| 1 | 0.179 / 0.179 / 0.180 | 0.000 / 0.614 / 0.877 |
| 2 | 0.195 / 0.198 / 0.201 | 0.640 / 0.972 / 1.144 |
| 4 | 0.264 / 0.271 / 0.274 | 1.330 / 1.495 / 1.593 |
| 8 | 0.316 / 0.322 / 0.326 | 1.858 / 1.962 / 2.029 |

**Chiffres bruts** : le meilleur point de la grille est le point **déployé** (α=1, n=1) ;
la dégradation est **monotone** dans les deux axes. Le dosage déplace bel et bien l'état
(rel 0.6 → 2.0) et sature le cap de lignes (0.18 → 0.33), donc les leviers ne sont pas inertes :
ils sont dirigés **du mauvais côté** pour la reconstruction de l'ancre.

**CAVEAT de mesure (à ne pas oublier en lisant)** : le décodeur A2 est **gelé** et il a été
co-entraîné sur la distribution d'états produite par α=1, n=1. Cette grille mesure donc
« écrire plus fort aide-t-il **avec le lecteur actuel** », pas « une écriture plus forte est-elle
intrinsèquement meilleure ». Un point de grille avec ré-entraînement du lecteur n'a PAS été fait.

JSON : `/shared/ssd_14T_project/adenelmi/mem2_runs/logs/p2p3_gain_steps.json`, wandb
`mem2_probe_p2p3`.
