# MÉMOIRE 2.1 « T0 » — LA DERNIÈRE CARTOUCHE MÉMOIRE — RAPPORT D'IMPLÉMENTATION (2026-08-04)

# ⚠️ AUDITÉ LE 2026-08-04 — VERDICT **GO-AVEC-CORRECTIFS** (2 bloquants) : lire le **§7** AVANT le §0

**Les §0-§6 sont le rapport d'implémentation, écrits avant l'audit et laissés INTACTS.** L'auditeur
indépendant a re-mesuré leurs affirmations : le §7 dit lesquelles tiennent (métrique bitwise,
plomberie, gradient, VRAM) et lesquelles ne tiennent pas (le point de départ −0.256, le gate G2b,
la présence du jumeau). **Aucun entraînement n'a été lancé** ; l'audit a consommé ~25 GPU-min
(banque pixel, `--vram_probe`, D0.c, sonde de gradient) sur idx 0. La liste exacte des fichiers
est au **§8**, le plan de run au **§5** — et il demande les correctifs du **§7.6** avant le GO.

---

## 0. RÉSUMÉ EN CINQ LIGNES

1. **Gates scellés au registre AVANT toute exécution** (D0, G1, G2a, G2b, G3) —
   `memory2_probes_registry.md`, section « MÉMOIRE 2.1 T0 ».
2. **Décision d'espace D0, partie CPU : tranchée, et elle dit PIXEL.** Le latent avg-poolé est
   **rejeté** (D0.a : Spearman +0.506/+0.454 < 0.70, plus-proche-voisin 21 %/18 % < 50 %) :
   il n'ordonne **pas** les ancres décodées comme la bande basse rendue. Le défaut du code est
   passé à `--t0_space pixel`.
3. **Terme T0 implémenté** dans `train_writer.py`, additif, **`--lam_t0 0` byte-safe** (le terme
   n'est pas même construit ; vérifié par selftest).
4. **45/45 selftests CPU PASS**, dont l'identité **BITWISE** de la métrique avec
   `prescreen_offensif.lowband` — le juge d'entraînement est *le même objet* que le juge du
   dossier, pas une reimplémentation.
5. **Les 12 mp4 comparateurs de G3 sont déjà sur disque** : la campagne finale ne coûte que
   **6 rollouts** (le bras `memdec_t0` seul).

---

## 1. POURQUOI T0, ET POURQUOI C'EST LA DERNIÈRE CARTOUCHE

La nuit du 04/08 a fermé quatre portes et n'en a laissé qu'une :

| ce qui a été testé | verdict | référence |
|---|---|---|
| le CANAL transmet-il ? | **OUI, ×20 la prédiction** (excès apparié +0.827, 23+/1−, p = 1.2e-06) | §16.1 |
| ce qu'il transmet prédit-il le re-ID ? | **NON** (Pearson du contraste −0.075, Spearman −0.163) | §16.1 bis |
| l'INSTANT du décodage est-il le levier ? | **NON**, 0/2, règle scellée appliquée | §16.5 bis |
| l'ADDITIF / la LÉGENDE élargissent-ils la charge ? | 2/4 et 13/24 (p = 0.84) | §16.6, §16.2 |
| **ce que porte l'ancre** | **vs A\* +0.042 | vs B +0.298 | vs X +0.344**, B > A sur **21/24** (p = 3e-4) | §16.1 quater |

> L'ancre décodée est une image de **l'ambiance du rollout**, pas du sujet. Le décodeur fait
> exactement ce qu'on lui a appris — reconstruire la frame **évincée** — et au retour ce qui vient
> d'être évincé est **l'interlude**. C'est **l'OBJECTIF** qui pose la mauvaise question.

2.1 change l'objectif. Et le choix de la métrique n'est pas cosmétique : **deux termes d'instance
sont déjà morts sur ce point précis**.

| métrique candidate | `pos_gap` au step 0 | pourquoi elle échoue |
|---|---|---|
| cosinus **latent** crop60 (J3) | **−0.19 à −0.36** (tous poolings, centré ou non) | dominée par l'artefact « frame décodée = passe-bas » |
| cosinus **DINO** ViT-B/14 (phase B) | **−0.185**, 10/10 vidéos val négatives | **même la vérité terrain floutée à σ = 8 px échoue le gate** ; DINO mesure « à quel point c'est flou », pas « qui c'est » |
| **T0, bande basse n×n** | **elle VOIT le défaut** (+0.042 contre +0.298, 21/24) | moyenne n×n ⇒ **tolère le flou par construction**, elle vit dans la bande que le décodeur atteint réellement |

T0 est la seule des trois qui remplit les deux conditions : **classer correctement le défaut
d'aujourd'hui** et **être atteignable par le rendu d'aujourd'hui**.

---

## 2. RÈGLE D0 — L'ESPACE : **le latent est REJETÉ, le terme vivra en PIXEL**

La règle a été écrite au registre **avant** d'être exécutée. Partie CPU, 0 GPU :

| sous-test | ce qu'il demande | mesure | seuil | verdict |
|---|---|---|---|---|
| **D0.a** | la géométrie bande-basse **latente** doit ordonner les **ancres décodées** comme la **pixel** (30 objets dont les DEUX représentations sont sur disque : `dec_anchors` + `anchor_px`) | `chan` **ρ = +0.506**, PPV **21 %**<br>`gray` **ρ = +0.454**, PPV **18 %** | ρ ≥ 0.70 **et** PPV ≥ 50 % | **FAIL, franc** |
| **D0.b** | une **vraie** frame de A doit être plus proche de A que de B, en latent (18 vidéos switch val) | `chan` **94 %** (marge +0.363)<br>`gray` **94 %** (+0.361) | ≥ 90 % | PASS |
| **D0.c** | l'**ancre A2** doit reproduire en latent le classement **B > A** du pixel | **TIRÉE PAR L'AUDIT** : `chan` 39 % / accord 72 % ; `gray` 56 % / 56 % | ≥ 70 % / accord de signe ≥ 75 % | **FAIL ⇒ PIXEL confirmé, D0 n'est plus provisoire.** Mais la ligne de référence PIXEL du même run donne **B > A sur 44 % seulement** (marge **+0.036**, pas −0.256) : voir **§7.3** |

**Lecture, et elle est instructive.** D0.b dit que la métrique latente est **saine** : sur des
**vraies** frames elle range A du bon côté, avec une grosse marge. D0.a dit qu'elle **n'ordonne
pas les ANCRES DÉCODÉES** comme le pixel — et le gate se joue exactement là. Le VAE de Wan n'est
pas une fonction spatialement alignée de la luminance : deux ancres proches en latent poolé ne le
sont pas en bande basse rendue. La règle scellée est « latent **SSI les trois** » ⇒ **PIXEL**.

**Conséquence d'ingénierie, et c'est ce qui rend le pixel abordable** : le côté **référence** de la
perte n'a **jamais** besoin de gradient (ce sont des frames collectées). Il se **pré-calcule une
fois hors ligne** (`build_bank_t0.py --with_pixel`, ~2 640 décodes VAE) ; seul le côté **ancre**
paie un décode VAE différentiable, `--t0_k` frames par vidéo. C'est exactement l'économie de la
phase B (10.9 GB mesurés avec 1 décode + DINO ; T0 n'a **pas** de DINO).
Artefact : `mem2_runs/writer/space_decision_t0.json`.

---

## 3. CE QUI EST IMPLÉMENTÉ

### 3.1 La métrique (`t0_metric.py`) — **une seule définition, importée partout**
`lowband_px` est **bitwise** `prescreen_offensif.lowband` (selftest 1, `max|Δ| = 0.000e+00`) :
gris 0.299/0.587/0.114 → `adaptive_avg_pool2d(n)` → aplati → **centré par frame** →
**L2-normalisé**. Deux vecteurs unitaires centrés : le produit scalaire **EST** la corrélation de
Pearson (selftest 2, à 1e-5). `corr_set(z, E)` = moyenne sur les frames de l'ensemble, comme
`corr()` du dossier. Variante latente conservée (`chan` / `gray`) pour que D0.c puisse être
tirée et le rejet documenté, **pas** pour être utilisée.

### 3.2 La perte (`t0_metric.t0_loss`)
- **`infonce`** multi-positifs : `−log( Σ_p e^{s_p/τ} / Σ_{p+n} e^{s/τ} )`. Borné, gradient
  stable, standard. Le sur-poids `w_b` de l'interlude est appliqué **en ajoutant log(w_b) aux
  logits** (la façon exacte de pondérer un groupe dans un softmax) et non en multipliant la
  similarité, ce qui aurait changé la température de ce seul groupe.
- **`margin`** : hinge **directement sur la quantité du gate**,
  `w_b·relu(m0 − (corr_A − corr_B)) + relu(m0 − (corr_A − corr_autres))`. Il **sature** une fois
  la marge atteinte : aucune pression au-delà du gate (anti-Goodhart par construction).
- **`both`** (défaut) : le terme standard porte l'optimisation, le hinge garantit qu'on optimise
  bien **ce qui est mesuré**.

Vérifié : InfoNCE mono-positif **==** `cross_entropy` (3.426438 vs 3.426438) ; le hinge saturé
vaut exactement 0.0 ; le gradient est fini, non nul, et `−grad` **augmente** `corr(ancre, A)` ;
une descente complète sur données synthétiques **retourne la marge** (−0.236 → +0.307).

### 3.3 Les références (`build_bank_t0.py`, `t0_bank.py`)
Par vidéo switch : **A** = 8 frames réparties uniformément sur le segment A, **B** = 8 sur
l'interlude. Deux précautions qui ne sont pas des détails :
- `skip_head = 8` : on **saute l'ouvreur**, qui pollue tout le rollout (X mesuré à +0.344) — sinon
  le « positif » contiendrait en partie X ;
- `min_gap = 2` : la frame **cible de la reconstruction** (`switch_frame−1`) **n'est pas** un
  positif T0 — sans quoi le terme T0 serait une copie du terme de reco et non une pression
  d'instance.

Négatifs : l'interlude B de la vidéo (négatif dur) + **toutes les autres vidéos du bon côté du
split**, la vidéo courante étant retirée du pool et **le jumeau y restant** (même prompt, graine
+100000 : le négatif dur validé empiriquement, 93/145 sondes au J2). Sa corrélation est loggée à
part (`t0/corr_twin`). Selftests 9-10 : exclusion et présence vérifiées, plus une assertion si un
côté de split est incohérent.

### 3.4 Les positions de RETOUR (`t0_bank.force_end_pos`)
Le mandat demande « l'état de **FIN de séquence** + 2-3 positions post-switch échantillonnées ».
Implémenté en **forçant la première position `anchor`** à l'état de fin de séquence, les autres
restant tirées parmi les états post-switch éligibles (règle `min_gap` du collecteur). **Aucune
position n'est ajoutée** : `K`, la VRAM et le temps par pas sont inchangés. `--t0_end_pos 0` est
un **no-op strict** : le tirage de la recette A2 est reproduit à l'identique (selftest 12).

### 3.5 Les gates exécutables
- **G1** dans `run_toy` : marge avant/après, comptage par vidéo, **+ les deux contrôles scellés**
  (h-shuffle ≤ 0 sinon VOID ; écart-type spatial de l'ancre ≥ 25 % du départ sinon VOID).
- **G2a** : `train_writer.py --t0_probe` — **le code de sortie EST le verdict** (0 / 2).
- **G2b** : `reid_scen/t0_prescreen.py` — mesure strictement identique à `snap_prescreen.py`
  (A\* propre = segment 0 du rollout où A ouvre), code de sortie = verdict.
- **G3** : bras `memdec_t0`, ajouté au harness.

### 3.6 Le bras `memdec_t0` (harness)
`SCEN_ARM_ENV["memdec_t0"] == SCEN_ARM_ENV["memdec20"]` — **vérifié par selftest** : même chemin
de code, même convention de fenêtre, même splice. **Seuls les poids changent.** Deux garde-fous :
- ses checkpoints viennent de `--mem2-t0-writer-ckpt` / `--mem2-t0-dec-ckpt`, **vides par défaut**,
  et un ckpt manquant lève une **assertion explicite** — jamais un repli silencieux sur A2 (qui
  ferait des deux bras le même modèle et viderait le contraste par construction) ;
- son dump d'ancre s'écrit sous `_memdec_t0`, **jamais** dans `_decanchor` : écraser ce fichier
  casserait la source que lit `wronganchor`.

---

## 4. SELFTESTS — **45/45 PASS, 100 % CPU** *(47/47 après les correctifs d'audit, §7.6)*

`CUDA_VISIBLE_DEVICES="" python reference/runs/mem2/selftest_t0.py`

| bloc | ce qui est garanti |
|---|---|
| 1-4 | métrique **bitwise** identique au dossier ; produit scalaire = Pearson ; invariance affine positive ; variante latente unitaire/centrée/**différentiable** |
| 5-8 | InfoNCE == cross-entropy ; hinge saturé == 0 ; gradient vers A ; **une descente retourne la marge** |
| 9-10 | vidéo courante **exclue** du pool ; **jumeau conservé** ; split incohérent → assertion |
| 11-12 | `--t0_end_pos` force bien l'état de fin de séquence sans toucher cible/cond/âge ; **inerte** quand non demandé ; RNG du gate déterministe |
| 13a | **byte-safety** vérifiée au niveau **source** (6 gardes) — sans importer `train_writer`, donc sans contexte CUDA |
| 13b | byte-safety **fonctionnelle** (`--with_train_writer`, exige un GPU visible) : **SAUTÉE sous le HOLD**, à jouer par l'auditeur |
| 14-15 | frontières A/B, ouvreur sauté, cible de reco exclue, déterminisme ; sur le **vrai corpus**, une vraie frame de A est plus proche de A (8/8 `chan`, marge +0.352) |
| 17-18 | bras `memdec_t0` ≡ `memdec20` ; dump non destructeur ; **les 12 mp4 comparateurs de G3 et les 6 A\* propres sont sur disque** |
| 16 | Spearman correct sur cas fermés (0.8) et ex aequo |

---

## 5. LE PLAN DE RUN, PRÊT À EXÉCUTER AU GO (rien de tout cela n'a tourné)

Pré-check GPU **dans la même commande** (règle maison, renforcée par la leçon de la nuit :
**exclusivité**, pas seulement mémoire libre — un rollout 1.3B monte à ~23.5 GB et deux ne
tiennent pas), **jamais idx6**, **max 2 cartes**.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.used --format=csv   # aucun voisin >= 20 GB

# 0) byte-safety fonctionnelle (13b) : 0 poids charge, juste le contexte CUDA
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/selftest_t0.py --with_train_writer

# 1) BANQUE de references, cote PIXEL (regle D0). ~2 640 decodes VAE, quelques minutes
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/build_bank_t0.py --with_pixel

# 2) D0.c : la confirmation GPU du rejet du latent (~5 min)
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/space_decision_t0.py

# 3) VRAM du backward COMPLET en espace pixel -- AVANT de fixer K / n_bptt / t0_k
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/train_writer.py --vram_probe \
  --lam_t0 1.0 --t0_space pixel --t0_k 2 --load direct

# 4) BASELINE G2a AVANT entrainement : la marge doit sortir NEGATIVE (~ -0.26 publie)
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/train_writer.py --t0_probe \
  --writer_ckpt .../mem2_writer_A2_final.pt --dec_ckpt .../dec_A2_for_memdec.pt \
  --run_name mem2_T0_baseline --wandb_project mem2_writer_T0

# 5) GATE G1 -- le toy (8 videos, ~1-2 GPU-h). C'est LUI qui decide si on paie la suite.
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/train_writer.py --toy \
  --toy_variants main,hshuffle --toy_steps 600 --toy_anchor_frac 1.0 \
  --lam_t0 1.0 --lam_t0_ramp 50 200 --rec_w 0.3 --lam_inst 0 --t0_log \
  --writer_ckpt .../mem2_writer_A2_final.pt --dec_ckpt .../dec_A2_for_memdec.pt \
  --phase B --run_name mem2_T0_toy --wandb_project mem2_writer_T0
#    -> code de sortie 0 = G1 PASS. 1 = FAIL => la cartouche est tiree, on ecrit le negatif.

# 6) ENTRAINEMENT complet (si et seulement si G1 PASS), ~3 h
CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/train_writer.py \
  --steps 2500 --bs 1 --K 3 --n_bptt <fixe par l'etape 3> \
  --lam_t0 1.0 --lam_t0_ramp 100 400 --rec_w 0.3 --lam_inst 0 --t0_log \
  --t0_space pixel --t0_k 2 --t0_end_pos 1 --t0_w_b 2.0 \
  --writer_ckpt .../mem2_writer_A2_final.pt --dec_ckpt .../dec_A2_for_memdec.pt \
  --prox 1e-4 --clip 1.0 --warmup 200 --val_every 100 --ckpt_every 500 \
  --run_name mem2_writer_T0 --wandb_project mem2_writer_T0

# 7) GATE G2a sur le ckpt final -> code de sortie = verdict
# 8) 6 rollouts du bras memdec_t0 (les comparateurs sont deja sur disque)
#    --mem2-t0-writer-ckpt / --mem2-t0-dec-ckpt OBLIGATOIRES
# 9) GATE G2b : python reference/runs/reid_scen/t0_prescreen.py --arm memdec_t0
# 10) G3 : scoring re-ID contre A* PROPRE, uniquement si G2b PASS
```

**Note d'ordonnancement** : le décodeur est dégelé (`--phase B`) **et** l'écrivain aussi — c'est
le co-entraînement demandé. `--freeze_writer` n'est **pas** utilisé.

---

## 6. CE QUE L'AUDITEUR DEVRAIT ATTAQUER EN PRIORITÉ (mes propres doutes, dits d'avance)

1. **La VRAM en espace pixel n'est pas mesurée.** `--t0_k 2` = deux décodes VAE différentiables
   par vidéo et par pas. La phase B mesurait 10.9 GB avec **un** décode + DINO ; ici il n'y a pas
   de DINO mais il y a `t0_k` décodes **et** le BPTT de l'écrivain (que la phase B n'avait pas,
   écrivain gelé). C'est le risque d'ingénierie n°1 ; l'étape 3 du plan existe pour ça, et le
   repli est `--t0_k 1` puis `--n_bptt 12`.
2. **Le seuil G1 (marge ≥ +0.10, ≥ 6/8) est un jugement, pas une dérivation.** Il part de −0.256 et
   demande un franchissement net. Un auditeur peut légitimement le trouver trop laxiste (le gate
   G2b demande +0.15 en absolu) ou trop sévère. Il est scellé ; le changer se note au registre.
3. **`corr_A ≥ 0.15` est un seuil ABSOLU sur une corrélation** dont la valeur de base dépend du
   contenu. Le contrôle `t0_anchor_std` couvre le mode d'échec le plus évident (une ancre qui
   s'aplatit), mais pas tous : une ancre qui apprendrait la **structure moyenne des segments A du
   corpus** (et non l'instance) monterait aussi. **C'est précisément ce que le jumeau dans les
   négatifs et G2b sur cellules held-out doivent attraper** — et si les deux passent pendant que
   G3 échoue, la conclusion est écrite d'avance au registre.
4. **Le pari sur G3 est écrit et il est prudent** : 16.1 bis a mesuré que la quantité copiée **ne
   prédit pas** le re-ID (Pearson −0.075). Mon pari ponctuel scellé est **+0.02 à +0.06, 2/6** —
   c'est-à-dire **un échec de G3**. Il faut que ce soit clair avant le GO : G1 et G2b peuvent
   passer et G3 échouer sans contradiction, et cette branche est déjà assignée
   (« le colis porte enfin A et le rendu ne le lit pas ⇒ le goulot est le CANAL »).
5. **Puissance de G3, dite d'avance** : le témoin sans contenu (`sdedit_wrong − wronganchor`) fait
   **3/6** au-dessus de +0.08 avec sd 0.197 par cellule. **À 6 cellules et 1 graine, le plancher de
   bruit vaut 2-2.5× le seuil.** Un G3 positif à 4/6 devra être répliqué sur une 2e graine ; un G3
   négatif, lui, est concluant.
6. **D0.c n'a pas tourné.** Le rejet du latent repose sur D0.a seul (règle « SSI les trois », donc
   valide), mais la confirmation directe « l'ancre A2 se classe-t-elle pareil dans les deux
   espaces ? » manque. Elle est scriptée et coûte 5 minutes.

---

## 7. AUDIT INDÉPENDANT (2026-08-04, auditeur externe au code) — **GO-AVEC-CORRECTIFS, mais les correctifs sont sur les GATES, pas sur le code**

Tout ce qui suit a été **re-mesuré**, pas relu. Scripts d'audit :
`scratchpad/audit_{metric_identity,idea,g2b_ceiling,g2b_fix,grad}.py`. GPU utilisé : idx 0
(pré-check d'exclusivité dans chaque commande, jamais idx6), ~25 GPU-min au total.

### 7.1 La MÉTRIQUE — **identité bitwise CONFIRMÉE, mais ce n'est pas la question**

Test indépendant sur **10 ancres réelles** de `anchor_px/` (pas des tenseurs aléatoires), les deux
chemins exécutés côte à côte :

| ce qui est comparé | résultat |
|---|---|
| `t0_metric.lowband_px` vs `prescreen_offensif.lowband` | **10/10 bitwise**, max\|Δ\| = 0.000e+00 |
| `t0_metric.corr_set` vs `prescreen_offensif.corr` (sur de vraies frames de rollout) | **10/10 égalités exactes**, max\|Δ\| = 0.000e+00 |
| `t0_metric.t0_margin` vs `corr_A − corr_B` du dossier | exact (−0.105672732 vs −0.105672732) |

**La FONCTION est la même. La MESURE ne l'est pas**, et c'est là que tout se joue (§7.3).
Nit corrigé : l'en-tête de `t0_prescreen.py` disait « identique à `snap_prescreen.py` » ; c'est
faux (snap échantillonne 8 frames/segment, `t0_prescreen` 16). Le défaut `--n_seg 16` est en
revanche **le bon** : c'est celui d'`anchor_what.py`, qui a produit 16.1 quater. En-tête corrigé.

### 7.2 L'IDÉE DANS LE CODE — **la plomberie est propre, sauf le négatif d'instance**

Vérifié moi-même, sur les **105 vidéos switch** du corpus (pas sur une banque synthétique) :

- **Aucune fuite du segment B dans les positifs** : `max(A) < switch_frame ≤ min(B)` sur 105/105.
  L'arithmétique tient : `switch_frame` est la **première** frame de B, la cible de reco vaut
  exactement `switch_frame − 1`, `min_gap = 2` l'exclut. **Pas d'off-by-one.** ✔
- **Position de retour** : `force_end_pos` met bien la 1re ancre à `max(elig) = dernier write`,
  sans toucher `j`/`frame`/`cond`. ✔
- **VAE-decode différentiable → θ_w** : mesuré, terme T0 **seul** (ni reco ni InfoNCE latent) :
  **‖grad θ_w‖ = 4.80e+01**, ‖grad décodeur‖ = 5.74e+00, contrôle `no_grad` à
  `requires_grad = False`. **Le gradient traverse le VAE, le décodeur et le replay.** ✔
- ⚠️ **Contrainte non écrite** : le write qui **encode** l'ancre est à `age = i_end − aj + 1`
  (mesuré : **12** sur la 1re vidéo). Avec `--t0_end_pos 1`, ce write n'est dans la fenêtre BPTT
  que si **`n_bptt ≥ 12`**. Le repli annoncé au §6 (`--n_bptt 12`, puis moins) est donc **le
  plancher, pas un repli** : à `n_bptt 4` le gradient n'atteindrait plus jamais l'écriture de A —
  le mécanisme serait débranché en silence. À écrire dans le plan.
- ❌ **LE JUMEAU N'EST PAS DANS LES NÉGATIFS.** `build_bank_t0.py --with_pixel` **sort en code 1**
  sur son propre selftest : `[bank][FAIL] jumeaux du meme cote du split (F1.1b) -- 0 paires`.
  Cause : les **30 jumeaux du corpus sont tous des vidéos `single`**, et la banque T0 ne contient
  que des `switch`. Le selftest 10 qui « vérifie » la présence du jumeau tourne sur une banque
  **synthétique** où des jumeaux switch ont été fabriqués. Au runtime la bannière l'affiche :
  `[T0] bank bank_t0.pt: ... 0 paires de jumeaux`, `t0/corr_twin` ne sera **jamais** loggé.
  Pire : les **105 vidéos switch ont 105 prompts A distincts** (0 prompt partagé). Il n'existe
  **aucun** négatif même-prompt dans le corpus switch. ⇒ **le terme T0 n'a aucune pression
  d'INSTANCE** : tous ses négatifs sont d'autres scènes. C'est exactement le mode d'échec que
  v1.2 a documenté (« régression de catégorie »), et la §3.3 du rapport affirme le contraire.

### 7.3 LES GATES — **les seuils correspondent au code ; les BASELINES sur lesquelles ils ont été calibrés sont fausses**

Correspondance registre ↔ code : **exacte** (G1 0.10 / 6 / 0.15 / 0.25 ; G2a 0.05 / 0.60 ;
G2b 0.15 / 4 cellules). Le problème est ailleurs, et il est arithmétique.

**(a) D0.c, que j'ai tirée (elle confirme PIXEL, et elle démolit le point de départ).**
Latent rejeté deux fois (`chan` 39 % / accord 72 % ; `gray` 56 % / 56 %) ⇒ **PIXEL confirmé, D0
n'est plus provisoire**. Mais la ligne de référence du même run dit :

> `[D0.c] PIXEL (référence) : B > A sur **44 %** des 18 vidéos val` — soit **marge = +0.036**
> (corr_A **+0.391**, corr_B **+0.355**), **10/18 vidéos déjà positives**.

Le registre scelle le départ de G1 à **−0.256** et « ≥ 6/8 vidéos positives ». **Mesuré sur la
quantité que G1 mesure réellement, le modèle A2 non entraîné est déjà à +0.036 et 56 % de vidéos
positives.** Le critère `corr(ancre, A) ≥ +0.15` est déjà franchi **×2.6** (0.391) au step 0.
**G2a (≥ +0.05 et ≥ 60 %) est à quelques centièmes d'être passé par le modèle qu'on déclare
cassé.** Les gates ne testent pas ce qu'ils croient tester.

**(b) Pourquoi −0.256 et +0.036 ne sont pas le même nombre : les références ne sont pas appariées.**
Dans 16.1 quater, le positif **A\*** vient d'un **AUTRE rollout** (S3/SS1 propre) et le négatif
**B** du **MÊME rollout** que l'ancre. Deux segments d'un même rollout partagent le fond, le
cadrage, la luminance : leur corrélation bande-basse est énorme. Mesuré :

> **ORACLE** (une **vraie frame** du segment A du rollout lui-même, jouée comme ancre, 6 cellules
> de G3) : vs A\* **+0.374** | vs B **+0.894** ⇒ **B > A sur 6/6**.

**La vérité terrain produit exactement la signature publiée** (« B > A sur 21/24, p = 3e-4 »)
qui sert de motivation à 2.1. Ce chiffre ne prouve donc **pas** que l'ancre porte l'ambiance
plutôt que le sujet : il prouve que le négatif est du même rollout et le positif d'un autre.

**(c) Conséquence directe : G2b est INFRANCHISSABLE tel qu'il est scellé.**
La règle est `corr(ancre, A*) ≥ +0.15` **ET** `> corr(ancre, B)`. L'**ORACLE** la passe sur
**0/6** cellules (0/6 même avec sa meilleure frame). Un gate que la vérité terrain échoue 6/6
n'arrêtera pas la campagne pour la bonne raison — il l'arrêtera **quoi qu'il arrive**.

**(d) Et la bande basse ne voit pas le sujet à travers les rollouts.** Test apparié propre — les
**deux** références tirées du **même** rollout propre, seul le SUJET change (attention : `SS1` a
`[(0,0),(81,1),(162,0)]`, son segment 2 est **encore** le sujet A ; je l'ai exclu) :

| « ancre » | marge SUJET = corr(A\*) − corr(autres sujets, même rollout) |
|---|---|
| **ORACLE** (vraie frame de A) | **+0.038**, 3/6 |
| ancre courante (A2) | −0.008, 2/6 |
| ancre étrangère (autre cellule) | −0.063, 3/6 |

**La vérité terrain est au plancher de bruit.** La bande basse 8×8 gris mesure l'ambiance du
rollout, pas l'identité. Aucun entraînement sur cette quantité ne peut faire bouger un juge
d'identité — ce qui rejoint, par un autre chemin, le 16.1 bis (Pearson −0.075) déjà au dossier.

### 7.4 LES TROIS TROUS AVOUÉS — deux fermés, un tranché

1. **VRAM du terme pixel : MESURÉE, et le risque n°1 n'existe pas.** `--vram_probe --lam_t0 1.0
   --t0_space pixel --t0_k 2`, K = 3 : **`n_bptt` complet 35.4 GB / 9.1 s** · **`n_bptt 12`
   21.5 GB / 4.4 s** · **`n_bptt 4` 16.7 GB / 3.0 s**. Tout tient sur **une** carte de 49 GB.
   `n_bptt 12` est le bon point (≈ 3 h pour 2 500 steps) **et** le plancher imposé par §7.2.
2. **D0.c : TIRÉE** (§7.3a). Latent rejeté une seconde fois, PIXEL confirmé, D0 n'est plus
   provisoire. `space_decision_t0.json` mis à jour.
3. **Seuil G1 — mon avis : il est mal ancré, pas trop laxiste ni trop sévère.** Plafonds mesurés
   dans l'espace **pixel** sur les 105 vidéos de la banque : ancre = moyenne des A de la vidéo →
   marge **+0.393** (100 % positives) ; ancre = une **vraie frame** de A → **+0.367** (98 %) ;
   ancre **générique** (moyenne des A des *autres* vidéos, zéro information sur celle-ci) →
   **+0.051** (54 %). Donc **+0.10 vaut 27 % de l'oracle et seulement 2× le modèle nul**, en
   partant non pas de −0.256 mais de **+0.036**. Recommandation : re-sceller G1 en **relatif** —
   marge ≥ **40 % de l'oracle** (≈ +0.15) **et** ≥ 3× la marge de l'ancre générique, cette
   dernière mesurée **au même run** comme troisième contrôle VOID (coût GPU : zéro). Et
   supprimer `corr(ancre, A) ≥ 0.15`, déjà franchi au step 0.

### 7.5 PIÈGES NON AVOUÉS (les trois que je retiens)

1. **Goodhart — testé, et le terme y résiste** (bonne nouvelle) : une ancre générique « A moyen du
   corpus » n'obtient que +0.051 / 54 %. La marge exige bien de l'information **propre à la
   vidéo**. Mais elle n'exige rien de propre au **sujet** (§7.3d) : la loss est satisfaite par
   « reproduire le layout basse fréquence de CE rollout au moment A ». Ce n'est pas rien — ce
   n'est pas l'identité.
2. **Le contraste que l'InfoNCE optimise n'est pas celui du gate.** Denominateur réel mesuré :
   8 positifs, 8 négatifs durs B (poids `w_b = 2`), **1 376** négatifs « autres vidéos » ⇒ le
   groupe B pèse **1.1 %** du softmax. `kind='both'` fait donc porter ~99 % de la pression au
   discriminateur « cette vidéo vs les autres » (= l'ambiance) ; **seul le hinge porte la quantité
   mesurée par G1**. Si le terme est lancé, monter `--t0_w_margin` (≥ 5) ou sous-échantillonner
   `E_o`, sinon on n'optimise pas ce qu'on mesure.
3. **Tripwire anti-effondrement dans le mauvais espace** (corrigé) : `anchor_spatial_std(x, space)`
   **n'a jamais lu son argument `space`** — en `--t0_space pixel` le contrôle VOID n° 2 de G1
   mesurait l'écart-type du **latent**, pas de l'image sur laquelle vit la métrique. Un latent
   bruité peut décoder plat : le mode d'échec visé n'était pas couvert.
4. **Byte-safety cassée par un défaut** (corrigé) : `positions_for` appelait `_force_end_pos` sans
   regarder `lam_t0`, et `--t0_end_pos` vaut **1** par défaut ⇒ à `--lam_t0 0` le tirage de
   positions **changeait** (vérifié : **20/20 vidéos**). La règle transverse n° 5 du registre
   (« `lam_t0 = 0` doit être byte-safe ») était violée par le défaut du code.

### 7.6 CORRECTIFS — appliqués par l'auditeur (code) / à trancher par Aden (gates)

**Appliqués** (selftests re-passés : **47/47**, identité bitwise re-vérifiée, chemin pixel
re-mesuré : même loss 7.7261, même marge −0.1088, ‖grad θ_w‖ 48.0 — comportement inchangé) :

| fichier | correctif |
|---|---|
| `t0_metric.py` | `_finish/lowband_px/lowband_lat/t0_embed` acceptent `return_scale` (contraste de la bande basse **avant** normalisation, dans l'espace de la métrique) ; docstring d'`anchor_spatial_std` rendue honnête (elle ignore `space`) |
| `train_writer.py` | `t0_anchor_std` = contraste **de la métrique** (le latent reste loggé sous `t0_anchor_std_lat`) ; `_force_end_pos` conditionné à `self.t0bank is not None` ⇒ **byte-safety rétablie** |
| `selftest_t0.py` | 13a.5 mis à jour + **13a.7** (byte-safety du forçage) + **13a.8** (tripwire dans le bon espace) |
| `reid_scen/t0_prescreen.py` | en-tête : la mesure est celle d'`anchor_what` (16 frames), pas de `snap_prescreen` (8) |

**À trancher AVANT le GO** (aucun n'est du code, tous sont des règles scellées) :

1. **G2b doit être re-cadré** : la règle actuelle est échouée 6/6 par la vérité terrain. Deux
   options propres : (i) références **appariées** (A\* et « autres sujets » du **même** rollout
   propre) ; (ii) le design d'**excès apparié** de `prescreen_offensif` (ancre propre − ancre
   étrangère), qui annule l'ambiance par construction et qui est la mesure qui a donné le +0.83
   valide de 16.1. Sans cela, la campagne s'arrête à G2b quoi qu'il arrive.
2. **G1/G2a doivent être re-scellés sur la baseline mesurée** (+0.036, 10/18) et non sur −0.256,
   avec le contrôle « ancre générique » ajouté et `corr_A ≥ 0.15` supprimé (§7.4.3).
3. **Le jumeau n'existe pas dans la banque** : soit on collecte des jumeaux `switch` (coût :
   une passe de collecte), soit on **retire la promesse** du rapport, du registre et du header de
   `t0_bank.py`, et on assume que T0 est un terme de **catégorie/scène**, pas d'instance.
   `build_bank_t0.py` sort en **code 1** ; ne pas lancer un entraînement sur un artefact dont le
   selftest échoue.
4. **Plan §5** : ajouter `--fixed_positions` au toy (le protocole G1 scellé le demande, `--toy`
   ne le met pas) et **figer `--n_bptt 12`** (plancher, §7.2), pas comme repli.

### 7.7 VERDICT DE L'AUDITEUR

**Le code fait ce qu'il dit** (métrique bitwise, pas de fuite A/B, gradient jusqu'à θ_w, VRAM
tenable, D0 confirmé) — sur ce plan c'est du travail propre et je n'ai trouvé que des défauts
réparables, tous réparés. **Ce que le code mesure ne fait pas ce que le projet croit.** Le
« défaut » qui motive 2.1 (B > A sur 21/24) est reproduit à l'identique par la vérité terrain,
la marge de départ réelle est +0.036 et non −0.256, G2b est infranchissable par l'oracle, et la
bande basse 8×8 est aveugle au sujet à travers les rollouts (oracle +0.038, 3/6).
⇒ **GO-AVEC-CORRECTIFS sur les quatre points du §7.6, dont le n° 1 et le n° 3 sont bloquants.**
Sans eux la cartouche est tirée à blanc : elle produira un négatif qui ne prouvera rien sur
l'écriture, seulement sur la métrique.

---

## 8. FICHIERS — LISTE EXACTE

**Créés** (tous sous `reference/`, donc gitignorés, aucun commit) :

| fichier | rôle | lignes |
|---|---|---|
| `reference/runs/mem2/t0_metric.py` | LA métrique T0 + la perte contrastive (import unique, partagé par tout le reste) | 163 |
| `reference/runs/mem2/t0_bank.py` | `T0Bank` (références/négatifs) + `force_end_pos` — **module sans CUDA**, donc selftestable sur CPU | 105 |
| `reference/runs/mem2/build_bank_t0.py` | construit la banque de références (latent **et** pixel) + 11 selftests | 222 |
| `reference/runs/mem2/space_decision_t0.py` | **règle D0** exécutable (D0.a / D0.b CPU, D0.c GPU) | 280 |
| `reference/runs/mem2/selftest_t0.py` | **45 selftests CPU** | 449 |
| `reference/runs/reid_scen/t0_prescreen.py` | **gate G2b** en situ (code de sortie = verdict) | 139 |
| `reference/memory2_t0_report.md` | ce rapport | — |

**Modifiés** (tous additifs ; aucun fichier suivi par git n'a été touché ; les lignes
« AUDIT 2026-08-04 » sont les correctifs de l'auditeur, cf. §7.6) :

| fichier | ce qui change |
|---|---|
| `reference/runs/mem2/train_writer.py` | terme T0 (`t0_term`), éval par vidéo (`t0_eval`), sonde G2a (`t0_probe`), gate G1 dans `run_toy`, 22 flags `--lam_t0/--rec_w/--t0_*`, `positions_for(rng=…)`. **`--lam_t0 0` ⇒ recette A2 bit pour bit.** |
| `reference/runs/reid_scen/scenarios.py` | `T0_ARMS` + `SCEN_ARM_ENV["memdec_t0"]` (≡ `memdec20`) + entrée dans `MAG_SCEN_ARMS` |
| `reference/runs/reid_scen/run_reid_scen.py` | checkpoints **propres** aux bras T0 (`--mem2-t0-{writer,dec}-ckpt`, assertion si absents) + dump d'ancre sous un nom qui n'écrase pas `_decanchor` |
| `reference/memory2_probes_registry.md` | **gates scellés D0 / G1 / G2a / G2b / G3** (écrits avant toute exécution) + **résultat D0 partie CPU** |

**Artefact produit** (CPU) : `/shared/ssd_14T_project/adenelmi/mem2_runs/writer/space_decision_t0.json`

**Commandes de vérification pour l'auditeur** :
```bash
CUDA_VISIBLE_DEVICES="" python reference/runs/mem2/selftest_t0.py            # 45/45 attendu
CUDA_VISIBLE_DEVICES="" python reference/runs/mem2/space_decision_t0.py --cpu_only
```

---

## 9. REGISTRE DES RUNS (à tenir dès le GO)

| # | date | run | changement d'hyperparams | raison | verdict |
|---|---|---|---|---|---|
| — | — | *(aucun run lancé : HOLD audit)* | — | — | — |

Budget annoncé et scellé : **≤ 3 allers-retours d'hyperparams**, **kill le 15/08** quoi qu'il
arrive.
