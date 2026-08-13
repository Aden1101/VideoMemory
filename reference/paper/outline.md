# SQUELETTE PAPIER ICLR 2027 (deadline fin sept 2026)

Document de travail, écrit le 2026-08-03. Sections et bullets en français, phrases de
l'abstract et des 3 claims rédigées en anglais (réutilisables telles quelles).
Chaque bullet porte **son chiffre exact** et **son rapport source**. Tout chiffre non sourcé
ici est à considérer comme non vérifié.

## 0. Décisions de framing (à valider par Aden avant rédaction)

| # | Décision | Ce que j'ai retenu | L'alternative rejetée |
|---|---|---|---|
| D1 | Nature du papier | **Papier de MÉTHODE** (mémoire à état + write-replay) dont l'ossature est une **contribution d'évaluation** (le régime non-ouvrant + la contamination de référence). Le résultat S2 est le titre, le write-replay est ce qui l'a rendu possible. | Papier d'évaluation pur ("les mémoires streaming sont de la chance positionnelle") : plus honnête vis-à-vis du n, mais on jette la seule contribution technique originale. |
| D2 | La limite d'instance | **Dans l'abstract**, phrase 4. On annonce "catégorie + mise en scène", pas "individu". | La cacher en §Limites : c'est l'attaque n°1 des reviewers, et elle est imparable si elle arrive de l'extérieur. |
| D3 | Deux points de fonctionnement | **UNE méthode, un curseur α** (pure splice α=1 → composite union) et non deux méthodes. Le composite est présenté en §Analyse comme un point de Pareto, PAS dans la table principale (n=3, seed 0). | Table 1 à deux lignes "nous" : le reviewer verra le n=3 et disqualifiera aussi le n=24. |
| D4 | La colonne contaminée | **Publiée à côté de la colonne propre** dans Table 1, et une sous-section "Experimental integrity" dédiée. On raconte qu'on a publié un verdict inversé en interne et qu'on l'a cassé nous-mêmes. | Ne montrer que la colonne propre : on perd la pièce la plus crédibilisante du dossier. |
| D5 | Statut de la baseline Echo | **Port explicite**, annoncé comme tel, systématiquement doublé de son plafond (`framebank_oracle`) et d'`oracle1` pour qu'aucun verdict ne dépende de la qualité du port. | Se présenter comme "Echo-Forcing" tout court : attaque de fidélité gratuite. |

Titre proposé (placeholder) : **"Positional Luck Is Not Memory: Subject Resurrection When the
Subject Does Not Open the Video"**. Nom de méthode placeholder : **StateAnchor** (écrivain TTT +
décodeur d'ancre + splice à la récence) ; nom de la méthode d'entraînement : **write-replay**.

## 1. Abstract (5 phrases, anglais, à raffiner)

1. Streaming video generators are said to "remember" past subjects, but their memory is either
   *positional luck* (an attention sink that pins the opening frames) or *photocopying* (a bank
   of previously rendered frames or KV): both fail exactly when memory is needed.
2. We isolate that regime with a subject-resurrection protocol in which the subject to be brought
   back does **not** open the video (`X → A → B → A`), and we show that when the render itself is
   contaminated by the pinned opening, every render-storing method collapses: LongLive −0.096,
   an Echo-Forcing-style frame bank +0.027 (n.s.), and even splicing the *real* past frame +0.027
   (n.s.), all against a clean reference.
3. A fast-weight memory whose state is decoded into a single recall anchor recovers the subject in
   **24 of 24 paired cells (+0.434 DINO, p = 1.2e-7)**, because it does not live in the render:
   it re-anchors generation from an O(1) internal state.
4. We make this trainable under a frozen backbone by an exact **offline write-replay**: because the
   memory is silent during collection, the per-block activations it consumes are independent of
   the writer's parameters, so the writer can be trained without the video model in the loop
   (bitwise replay/in-situ parity as a founding self-test), in GPU-hours instead of GPU-weeks.
5. We are explicit about the ceiling: a calibrated instance test is **negative** (−0.008 ± 0.024
   against a metric that resolves instances at +0.134, 12/12), so what returns is the right
   category correctly staged, not yet the right individual, and we trace that ceiling to the
   state → frame channel (a *perfect* frame written and immediately re-read is worth only +0.131).

## 2. Introduction

### 2.1 Le décor (1 paragraphe)
- Génération vidéo en streaming causale : fenêtre d'attention locale de **12 frames latentes**
  (~3 s) + **sink** de 1 frame globale ; vidéos de **240-300 frames latentes = 60-75 s**.
  Substrat : LongLive-1.3B sur base Anchor-Forcing. Source : `scenarios_design_report.md` §8.2,
  `bilan_explique_2026-07-30.md` §1.
- Tout ce qui sort de la fenêtre est perdu, sauf ce que le sink épingle. D'où la course aux
  "mémoires" : banques KV (Echo-Forcing), mémoires co-entraînées e2e (VideoSSM, MemGen).

### 2.2 Le constat qui organise le papier
- Les benchmarks de "scene recall" publiés mesurent presque tous des situations où le sujet à
  ressusciter **ouvre** la vidéo (Echo-Forcing évalue plans 4-6 contre plans 1-3 :
  `scenarios_design_report.md` §1). Dans ce cas le sink fait le travail gratuitement.
- Conséquence : "avoir une mémoire" et "être bien placé" sont confondus par le protocole.

### 2.3 Les 3 claims exacts (avec chiffres, à mettre en gras dans l'intro)

**Claim 1 (régime + renversement).** Sur le terrain non-ouvrant (S2 : `X → A → B → A`, 24 cellules
appariées, DINO full, référence = rendu propre A\*) : `memzero` 0.2235 ;
**`memdec20` (nous) 0.6577, Δ +0.4343, 24+/0−, sign-p 1.2e-7** ; `framebank` (port Echo)
0.2500, +0.0266, 13+/11−, p = 0.84 ; `oracle1` (vraie frame splicée) 0.2505, +0.0270, 13+/11−,
p = 0.84 ; `stockoff` (LongLive) 0.1272, **−0.0962**, 7+/17−, p = 0.064. En paraphrase (S2p) :
nous +0.3569 (23+/1−, p = 3.0e-6), banque +0.0925 (p = 0.064), LongLive −0.0572.
Source : `scenarios_design_report.md` §9.2, `results/clean_rescore/summary_clean_rescore.json`.
*La phrase à écrire* : le seul mécanisme qui survit à la dégradation du rendu est celui qui n'y vit
pas ; une banque stocke ce qui a été rendu, donc elle stocke la contamination (poubelle entrante,
poubelle sortante), et c'est pourquoi la **vraie** frame échoue exactement comme la banque.

**Claim 2 (méthode).** L'écrivain est entraînable **hors-ligne et exactement** : sous injection
neutralisée, `h = norm1(x)` est calculé avant la branche mémoire et l'unique rétroaction
(`x = x + mem_out`) est annulée, donc h est une **donnée** indépendante de θ_write ; le
write-replay n'est pas une approximation. Vérifié par lecture de code op-par-op
(`memory2_design.md` §1.1-1.3) et gaté par des selftests de parité bitwise
(`max|diff| = 0.000e+00`). Coût : corpus de **165 vidéos / 5 175 writes / 229 GB** collecté une
fois, puis entraînement à **15.2 GB de VRAM et 3.9 s/step** (K=3, n_bptt=12, B=2), 2 500 steps
en 1.6 h sur une GPU. Sources : `memory2_j3_report.md` §1 (S1b, h-oracle, toy), §2 (VRAM),
`memory2_design.md` §2-3.

**Claim 3 (limite + intégrité).** Ce qui revient est la bonne catégorie mise en scène, pas
l'individu : test d'instance apparié `cos(retour, A* même seed)` vs `cos(retour, A* autre seed)`
= **−0.0076 ± 0.0235, 11+/13−, p = 0.84**, alors que la métrique résout l'instance
(**0.9329 vs 0.7994, marge +0.1335, 12/12**). La cause est un plafond de canal, mesuré :
une frame **parfaite** écrite puis relue immédiatement ne vaut que **+0.131**, contre +0.126 pour
notre système complet et +0.224 pour l'oracle. Sources : `scenarios_design_report.md` §9.3,
`memory2_probes_report.md` (campagne 4 arms).

### 2.4 Contributions listées (4 puces)
1. Un **protocole** qui sépare mémoire et chance positionnelle (matrice 6 scénarios × 5 méthodes,
   sujet ouvrant / non-ouvrant × verbatim / paraphrase), plus la mise en évidence d'un artefact de
   référence qui **inverse** les verdicts.
2. Un **résultat** : sur le régime non-ouvrant, les mémoires publiées sont nulles (y compris avec
   accès à la vraie frame) et une mémoire à état apprise ramène le sujet 24/24.
3. Une **méthode d'entraînement** : write-replay exact sous mémoire silencieuse, backbone gelé,
   état O(1), composants entraînés offline.
4. Une **analyse causale complète** et une limite caractérisée (ablation, plafond de canal,
   contrôles anti-illusion, test d'instance calibré négatif).

## 3. Related work (positionnement explicite, 4 blocs)

- **Générateurs streaming / substrat.** Self-Forcing (2506.08009) → LongLive (2509.22622) →
  Anchor-Forcing (2603.13405, "Anchor Memory" = réorganisation KV + RoPE tri-région).
  Ce que nous en prenons : le substrat gelé. Ce que nous en disons : leur mémoire est
  **positionnelle** ; le sink de 1 frame est ce qui produit leurs +0.16 à +0.25 quand A ouvre,
  et **−0.096 quand A n'ouvre pas**. Source : `bilan_2026-07-30.md` §2.
- **Banques de rappel : Echo-Forcing (2605.16003).** Décrire le mécanisme réel (KV compressés par
  softmax positionnel, 12 ancres roulantes + 3 frames compressées + fenêtre 3/L=21, banque qui
  **croît** avec le nombre de scènes, routage par **cosinus de prompt**, training-free).
  Notre position : (a) même famille "training-free + backbone gelé", (b) mais mémoire **O(#scènes)
  de contenus rendus** contre notre **O(1) d'état appris**, (c) leur routage par chaîne de
  caractères casse à la reformulation (**rappel 83.3 % en S2p/S3p, 6 faux rappels ; 91.7 % en
  S4p** ; aucun seuil ne sépare paraphrase 0.665 de distracteur 0.7218), (d) et surtout leur
  contenu **hérite de la contamination du rendu**. Source : `scenarios_design_report.md` §1, §3.
- **Mémoires co-entraînées e2e : VideoSSM (2512.04519), MemGen (2512.18741).** Même backbone
  Wan-1.3B, même rollout DMD, mémoire gated-delta / perte de reconstruction, co-entraînées de
  bout en bout. Notre position : nous **ne prétendons pas** battre une co-adaptation e2e ; nous
  montrons qu'on peut entraîner les composants mémoire **sans le modèle vidéo dans la boucle**
  quand la mémoire est silencieuse, ce qui change le coût d'un ordre de grandeur et rend
  l'attribution causale possible (backbone strictement gelé, bitwise). À citer honnêtement comme
  la famille qui nous préempte sur le mécanisme.
- **Fast weights / TTT.** Titans, LaCT (2505.23884), TTT-Video (2504.05298), Scal3R (2604.08542,
  dont notre cellule est adaptée), gated-delta. Notre cellule n'est pas nouvelle : ce qui est
  nouveau est **où on l'entraîne** (offline, par replay) et **comment on la lit** (décodage en
  UNE frame d'ancre splicée à la récence, pas d'injection additive dans le flux).
- **Sinks.** Lignée StreamingLLM/attention-sink pour nommer le phénomène "chance positionnelle".

## 4. Setup et protocole (la section qui doit être blindée)

### 4.1 Substrat et mémoire
- Backbone gelé LongLive-1.3B, `local_attn_size 12`, `sink_size 1` + `global_sink`,
  chunk = 3 frames latentes, 60-75 s par vidéo.
- Mémoire à 3 blocs (5, 16, 27). État = fast weights `w0/w1/w2`, **28.3 M valeurs par bloc**,
  taille FIXE (O(1) en longueur de vidéo). θ écrivain entraîné = `qkv` (7.08 M/bloc) + `k_norm` +
  `lrs` (4.6 K) + `decay_logit` ≈ **21.3 M** au total ; les `w*` init restent gelés (prior d'état).
- Décodeur d'ancre : 26-33 M paramètres, conditionné FiLM sur l'embedding de prompt poolé UMT5.
- Lecture : la frame décodée est **splicée au slot de récence** au recache du retour. Le canal de
  lecture natif (injection additive dans le flux) est **fermé** : `memon20` = −0.018 full /
  **−0.050 crop60 (6+/18−, p = 0.023)**, il NUIT. Source : `memory2_probes_report.md`.

### 4.2 Scénarios (Table 1 = ce tableau)
| nom | schedule | ce qu'il teste |
|---|---|---|
| S1 | A → B → A (A ouvre) | terrain historique |
| S3 | A → B → C → A (A ouvre) | 150 frames d'interférence |
| S4 | A → B → A → C → A (A ouvre) | double résurrection (dérive en chaîne) |
| **S2** | **X → A → B → A (A n'ouvre PAS)** | **le sink tient X, pas A** |
| S2p/S3p/S4p | retour demandé en **paraphrase** | punit le routage par chaîne de caractères |
- **12 triples (A, B, C) × 2 seeds = 24 cellules appariées** par scénario ; couverture
  véhicule / personne / animal / objet rigide.
- Pré-enregistrement anti-fuite : **0 des 48 prompts** n'apparaît dans le corpus d'entraînement
  vidprom ni dans les 165 vidéos collectées (selftest (f), sous-chaîne dans les deux sens).
- Source : `scenarios_design_report.md` §2.3, §4.

### 4.3 Les 5 méthodes comparées
`stockoff` (LongLive officiel) | `memzero` (notre modèle, mémoire coupée = référence des deltas)
| `memdec20` (nous) | `framebank` (port Echo-Forcing : banque croissante, une entrée par segment,
clé = embedding UMT5 L2-normé, routage par cosinus, contrainte de non-adjacence, seuil, injection
à la récence) | `oracle1` (triche : la vraie frame passée, borne haute).
Plus `framebank_oracle` (sélection vérité-terrain = plafond de la banque) et `memdec20r`
(notre bras routé par le MÊME routeur que la banque, pour tuer l'asymétrie oracle-de-planning).
Source : `scenarios_design_report.md` §2.2, §8.4.

### 4.4 Métrique et statistique
- Similarité DINOv2-ViT-B/14, deux variantes (frame pleine et crop60 centré sujet), fenêtres de
  scoring `settle=20` / `resettle=12`, 3 frames coupées en fin de segment.
- **Delta APPARIÉ vs `memzero` + test des signes exact** sur les 24 cellules (pas de test
  paramétrique : la distribution des deltas est un mélange à queue de jackpots, mesuré :
  7 cellules ~0, 12 continues 0.03-0.18, 5 jackpots ≥ 0.35 dont 0.57 ;
  `memory2_probes_registry.md`).
- **Plancher de bruit mesuré** (rejeu de `memzero` sur un autre process/GPU) : **0.0008 sur le
  sujet**, deux ordres de grandeur sous les effets. Le rollout n'est PAS bit-à-bit reproductible
  (divergence flash-attn/cuBLAS dès la frame latente ~3, avant tout hook) : c'est dit, mesuré,
  et l'appariement est justifié statistiquement, pas par bitwise. Source :
  `composite_gates_report.md` §1.1.
- **Point de méthode central (§4.5)** : la référence de re-ID.

### 4.5 La référence de scoring, et pourquoi elle décide de tout (sous-section obligatoire)
- En S2, le sink épingle la frame 0 = **X**, donc le segment "première apparition de A" (référence
  naturelle du scoring) **montre X**. Mesuré : ce segment ressemble au rendu propre de A à
  **0.165** (full) / 0.180 (crop60), et il ressemble à l'ouverture à **0.670**.
- Conséquence : un bras qui splice une frame RÉELLE de ce segment réimporte X et est **récompensé**
  (`framebank` : cos(retour, ouverture) = 0.517, score publié 0.775) ; le seul bras qui produit
  réellement A s'en échappe (`memdec20` : 0.017) et est **puni** (0.188, soit −0.378).
- Correctif : **A\*** = le rendu PROPRE de A = segment 0 du rollout S3 de la MÊME cellule (même
  triple, même seed, A ouvre, rien ne le précède). `stockoff` est scoré contre **son propre** A\*
  (checkpoint différent).
- Source : `scenarios_design_report.md` §8.2, §9.1.

## 5. Méthode : write-replay (la contribution méthodo)

### 5.1 L'insight, énoncé comme un théorème d'ingénierie
- Sous mémoire silencieuse (`mem_out * 0`), (i) les latents du rollout sont exactement ceux du
  modèle sans mémoire, (ii) les activations `h = norm1(x_in)` consommées par l'écriture sont
  calculées **avant** la branche mémoire, et (iii) l'état écrit ne nourrit rien d'autre que
  `mem_out`. Donc **h ⟂ θ_write** : h est une donnée.
- Conséquence : la séquence de writes peut être **rejouée hors-ligne**, sans le modèle vidéo,
  et le gradient de l'écrivain est **exact**, pas approché. C'est la contribution qu'on vend.
- Deux corollaires pratiques : (a) toutes les positions de revisit d'une vidéo sont gratuites au
  replay (contre 8 états figés par vidéo dans notre pipeline v1), (b) train et éval voient la même
  distribution d'états (le collecteur v1 tournait injection ACTIVE : mismatch corrigé).
- Source : `memory2_design.md` §1.1-1.2.

### 5.2 Ce qu'il faut pour que ce soit vrai (les conditions, à publier, elles font la crédibilité)
- Parité bitwise replay vs in-situ : réutiliser le `qkv` **fusionné** (3 matmuls séparés changent
  l'arrondi), module en bf16 comme à l'inférence, round-trip explicite fp32→bf16→fp32 pour `lrs`,
  `detach_state=False` (sinon le graphe est coupé silencieusement à chaque write), même modèle de
  GPU pour le selftest. Source : `memory2_design.md` §1.4.
- Selftest fondateur S1b : `MEM_ZERO_ALL` est une **multiplication**, pas un skip structurel
  (démontré en injectant un NaN : il contamine les h aval et les latents) ; scan du corpus :
  **22 parts / 229 GB / 165 vidéos / 15 525 tenseurs h / 111.6 G éléments → 0 non-fini**.
  Source : `memory2_j3_report.md` §1.

### 5.3 Objectif et curriculum
- Loss principale : `norm_mse(dec(state, cond), target)` pondérée sujet (×3), sur K=3 positions
  switch + 1 position single par passage. BPTT tronqué à 12 writes, checkpoint par write,
  B=2 + grad-accum.
- Terme d'instance InfoNCE en espace DINO contre une banque de cibles précalculée : **spécifié,
  gaté, et NON activé** (voir §5.4). À écrire comme un choix motivé, pas comme un oubli.
- Curriculum : toy-overfit (gate) → écrivain seul décodeur gelé (test causal propre : tout gain
  = information NOUVELLE dans un format déjà lisible) → joint.
- Source : `memory2_design.md` §3, `memory2_j3_report.md` §1.

### 5.4 Les gates pré-enregistrés, y compris ceux qui ont dit NON (honnêteté = argument)
- **h-oracle** (borne : que contient h ?) : val `norm_mse` **0.0004** (contre 0.867 / 0.812 pour
  nos décodeurs, mêmes unités, 1.0 = prédire zéro), contrôle h mélangé **1.87**. Lecture :
  l'information EST dans h ; le goulot est **l'état**, pas le lecteur. Verdict œil : "indiscernable".
- **Toy-overfit** : (a) la loss s'écrase sur des cibles vieilles de ≥ 10 writes,
  0.9829 → 0.4704 (PASS) ; (c) rejouer la séquence h d'une AUTRE vidéo laisse la loss à 0.9395
  (+99.7 %, PASS : l'état porte de l'information spécifique à SA vidéo) ; (b) couper le BPTT au
  dernier write ne coûte que **+8.9 %** pour un seuil pré-enregistré à +10 % → **FAIL déclaré**,
  à lire comme "inconclusif sur la profondeur". À publier tel quel.
- **Gate DINO de la phase B** : `pos_gap = −0.1848`, top-1 0.000, **10/10 vidéos négatives** ⇒
  le terme d'instance en espace DINO est **inutilisable tant que le rendu n'est pas réparé** ; et
  le diagnostic innocente le contenu : **même la vérité terrain floutée à σ=32 échoue au gate**
  (gap −0.127 contre −0.190 pour le décodé). B2 non lancé, conformément au mandat.
- **Phase B1 (mixer spatial)** : NÉGATIVE, mesure appariée (mêmes états, mêmes positions) :
  `norm_mse` 0.7672 → 0.7909 (5/20 meilleures), `cos_DINO` 0.1027 → 0.0919 (8/20) ; contrat de gel
  de l'écrivain tenu bit pour bit (`delta_ref = 0.000e+00` au step 2500). Le goulot du rendu n'est
  pas l'absence de mélange spatial.
- Sources : `memory2_j3_report.md` §1, `memory2_phaseB_report.md` §0-2, §7.

## 6. Expériences (Table 1 + Table 2)

### 6.1 Le résultat principal : régime non-ouvrant (Claim 1, chiffres en §2.3)
- Écrire aussi le crop60 : `memdec20` +0.4232 (24+/0−) en S2, +0.3129 (24+/0−) en S2p ;
  `framebank` +0.034 / +0.091 non significatif ; `stockoff` −0.106 / −0.078 (p = 0.023).
- Le paragraphe "pourquoi tout le monde échoue sauf nous" (mécanisme, pas anecdote).

### 6.2 Le terrain facile : nous ne gagnons pas, et c'est le point (Claim 1, versant honnête)
Table 1, moitié haute (Δ apparié vs `memzero`, 24 cellules) :
| Scénario | LongLive | Banque Echo | NOUS | Oracle1 |
|---|---|---|---|---|
| S1 (A ouvre) | +0.16 (abs 0.847) | ≈ oracle (verbatim) | **+0.13** (abs 0.812) | +0.22 (abs 0.910) |
| S3 | +0.23 | +0.30 | +0.12 (p = 0.06) | +0.30 |
| S3p | +0.25 | +0.29 | +0.12 | +0.33 |
| S4 | +0.22 | +0.21 | +0.11 (seg2 +0.077, seg4 +0.163) | +0.24 |
| S4p | +0.15 | +0.20 | **−0.01 n.s.** (seule régression réelle) | +0.25 |
- Interprétation à écrire : quand A ouvre, le sink porte l'identité gratuitement, et une banque en
  verbatim **EST** l'oracle (routage cos = 1.0000 contre distracteur 0.7218 : il ne peut pas se
  tromper ; prouvé bitwise `framebank == oracle1` sur A→B→A). Notre gain y est plafonné par le
  canal (§7.2). **Nous ne revendiquons pas ce terrain.**
- S4p −0.01 : conditionnement FiLM hors-distribution (le décodeur n'a jamais vu de paraphrase).
  À dire, pas à cacher.
- Source : `bilan_2026-07-30.md` §2, `scenarios_design_report.md` §3, §8.3.

### 6.3 Le coût de la banque : le routage par prompt (mini-résultat, à garder court)
- Verbatim : rappel 100 %. Paraphrase : **83.3 %** en S2p/S3p (**+6 faux rappels**), 91.7 % en S4p.
  Aucun seuil unique ne sépare "paraphrase du même sujet" (min 0.665) de "autre scène du même
  triple" (max 0.7218). Nous : 0 % de raté (et `memdec20r` mesure ce que coûterait le même routeur
  chez nous). Source : `scenarios_design_report.md` §3.

### 6.4 Tripwires de qualité (la table qui prouve qu'on ne casse pas la vidéo)
- Fond (DINO sur le complément du crop, retour vs queue de B), `cut2_ratio` (brutalité du raccord),
  **DD-ellipse** (mouvement DANS le crop : s'effondrer sous `memzero` = autocollant = rejet),
  adhérence **CLIP** au prompt du retour.
- Chiffres du point pur (3 cellules SS2, seed 0) : `memdec20` sujet 0.689 / fond 0.459 / cut 20.0 /
  DD 0.0053 (> memzero 0.0040 : pas d'autocollant) / CLIP 0.3455 (> memzero 0.2485).
  Le prix est **le fond et le raccord**, pas le mouvement ni le texte. Source :
  `composite_gates_report.md` §0, §5.0.

## 7. Analyse causale (la section qui transforme un chiffre en mécanisme)

### 7.1 P1, ablation : ce qu'on lit vient bien de ce que la mémoire écrit
- Protocole : toutes les écritures du segment A neutralisées (remplacées par leur moyenne par
  token = écriture non informative de même magnitude), offline par replay, 12 vidéos held-out.
- Écrivain 2.0 : `‖Δétat‖/‖état‖` **0.803**, `norm_mse` de l'ancre **0.800 → 1.229**,
  **Δloss +0.429, 12/12 pires**. Écrivain stock l1 : rel 0.996, +0.128, 10/12.
- La ligne rouge pré-enregistrée était **Δloss < 0.02** ("alors le +0.127 ne vient pas des fast
  weights") : ratée d'un facteur **×21**. Le scénario tueur (P = 30 % selon l'avocat du diable)
  ne s'est pas produit.
- Contrôle "h d'une autre vidéo" : dégrade **moins** que la moyenne par token chez 2.0
  (+0.184 vs +0.429) : l'écrivain 2.0 dépend **3× plus du contenu réel**.
- Source : `memory2_probes_report.md` §P1.

### 7.2 L'échelle des bornes : où est le plafond, exactement (Table 2)
Delta apparié full, 24 cellules, harness l1 :
```
memon20 −0.018 | memon −0.021 || memdec12 +0.119 | memdec20 +0.126 | thru +0.131 | twin +0.135
        | blur24 +0.158 || blur0 +0.202 | oracle1 +0.224
```
- **thruwriter** (vraie frame → vraie écriture → relecture jeune) = **+0.1312 (18+/6−, p = 0.023)**,
  soit +0.005 seulement au-dessus de notre système complet : **le canal écriture→lecture plafonne
  l'instance, quel que soit le contenu écrit**. C'est le chiffre le plus important de la section.
- **twin** (vraie frame nette, mauvaise instance) = +0.1347 (14+/10−, p = 0.54) : la netteté seule
  paie autant que notre système en terrain facile ⇒ le modèle "contenu = tout" est faux par
  construction. À assumer.
- **blur σ=24** = +0.1575 (23+/1−) et **blur0** (aller-retour VAE seul) = +0.2019 : décomposition
  propre, l'encodage coûte −0.022, le flou coûte −0.044. Le contenu correct survit très bien au flou.
- `memon20` −0.018 / −0.050 : la lecture native NUIT, la porte est fermée.
- Source : `memory2_probes_report.md` (campagne 4 arms + repères).

### 7.3 La preuve que l'instance ENTRE dans l'état (et le négatif de sortie)
- **Cheval 3-bras** (planche `horse_3way.png`) : sur une cellule single-prompt (état 100 % A, aucun
  B), l'ancre décodée est **pâle SEULEMENT avec l'écrivain entraîné** (v1.2 et contrôle : brunes).
  L'instance entre donc bien dans l'état. Source : `memory2_j3_report.md` (verdicts œil 2).
- **Mais le test de sortie est négatif et calibré** : §2.3 Claim 3. Écrire les deux dans le MÊME
  paragraphe : *l'instance entre dans l'état, elle ne sort pas du canal*.
- Diagnostic de rendu (phase B) : au niveau de fidélité atteint, **la vérité terrain floutée échoue
  elle-même** au gate d'instance DINO ⇒ ce n'est pas le terme qui est mal conçu, c'est le rendu qui
  n'est pas prêt. Source : `memory2_phaseB_report.md` §0, §2.

### 7.4 Position : la résurrection est une propriété du présent déclaré, pas de ce qui est lisible
- L'ancre ne compte que si elle occupe **le slot de récence** :
  slot 0 → sujet 0.7249 / fond 0.4553 ; **reculée d'UN slot → 0.2723 / 0.9405** (98 % du gain
  détruit) ; deux slots → 0.2664.
- Canal **additif** (l'ancre en tokens k/v supplémentaires, rien n'est évincé, repondération exacte
  du softmax par log-sum-exp) : masse d'attention **2.04 %** au gain 1 (contre 7.7 % si elle était
  lue uniformément : le modèle la **sous-pondère d'un facteur 4**) ; gain sujet **+0.003 = 3 %**
  du chemin (tier S2), et même à **12 % de masse forcée** on ne récupère que **35 %**.
  Les pixels bougent (3.98/255 après le retour contre 1.20 de bruit ambiant), pas les métriques.
- Conclusion à écrire : ce n'est pas un problème de lecture, c'est un problème de ce que le modèle
  **déclare être son présent immédiat**. Source : `composite_gates_report.md` §3, §4, §7.

### 7.5 Le curseur α : deux points de fonctionnement d'une seule méthode
- Bouton 1 (**position**) : binaire, pas de réglage fin (§7.4).
- Bouton 2 (**dose**, = quelle fraction de la frame de récence vient du passé) : continu, c'est le
  seul curseur réel. Courbe mesurée (3 cellules SS2) :
  | dose | 0 | 0.238 | 0.288 | ~0.55 | 0.712 | 1.0 |
  |---|---|---|---|---|---|---|
  | sujet | 0.265 | 0.295 | 0.412 | **0.558** | 0.535 | 0.689 |
  | fond | 0.934 | 0.908 | 0.827 | **0.781** | 0.598 | 0.459 |
  Le fond est quasi **linéaire en dose et indifférent à la forme** ; le sujet dépend de la forme
  (centrer vaut ~+0.05).
- Le point composite union (`memdec21u`, masque = union des ellipses d'énergie ancre/courant,
  crop 0.75, α porté par le cosinus de prompt, α = 0 ⇒ `memdec20` **bit à bit**) **sort de la
  droite sur les deux axes** : +0.060 de sujet et +0.108 de fond contre l'interpolation ; et il
  bat la banque Echo de **+0.18 de sujet (0.558 vs 0.375)** avec un fond à 0.78, DD 0.0058 >
  memzero, CLIP 0.287 > memzero.
- **Statut à annoncer** : n = 3 cellules, seed 0, une seule graine ⇒ **étude secondaire**, pas
  Table 1 (décision D3). Source : `composite_gates_report.md` §6, §7.

## 8. Intégrité expérimentale (section dédiée, ~0.75 page : c'est elle qui achète la confiance)

1. **La contamination de référence, découverte deux fois.** (a) En S2, le verdict publié en
   interne était `memdec20` −0.378 : la référence elle-même montrait l'ouverture (0.165 de
   similarité avec un rendu propre de A). Prouvé sur 24 cellules par trois mesures indépendantes
   (comptage des splices dans 142/144 rollouts, position dans les logs, scores pré-retour
   identiques entre bras à la troisième décimale), corrigé par la référence A\*, verdict inversé
   de −0.378 à +0.434. (b) Le même piège attendait le tier même-décor de l'étude composite : la
   règle "**jamais scorer contre la première apparition du même rollout**" y est appliquée
   d'emblée. Les deux colonnes (contaminée / propre) sont publiées.
2. **Un faux diagnostic refusé.** La mission initiale demandait de re-lancer le bras en accusant
   le déclenchement du hook : le déclenchement était correct (splices comptés : exactement 1 en
   S2/S2p/S3/S3p, exactement 2 en S4/S4p, sans exception), la relance a été refusée, ~11 GPU-h
   économisées, et l'asymétrie méthodologique visée (notre bras reçoit un signal oracle "ceci est
   un retour") a été corrigée autrement (`memdec20r`, routé comme la banque).
3. **Les prédictions pré-enregistrées, y compris celles qui ont été réfutées.** Registre écrit
   AVANT les runs (`memory2_probes_registry.md` : lignes rouges des deux avocats) ; prédictions du
   débat composite (P1/P1'/P2/P3/P4a/P4b/P4c). Réfutations publiées : **P1 (le mal était
   l'éviction) réfutée**, **P4b (ancre bruitée ⇒ effondrement) réfutée sur le tier même-décor**,
   **P4c inversé** réfutée pour cause de confond de dose (`1−m` = 2.5× plus d'ancre) que la
   prédiction avait raté.
4. **Le contrôle anti-illusion qui fait mal, et qu'on publie.** Ancre **100 % bruit**, masque
   gardé, dose appariée : **+0.0792 = 54 % du gain composite** sur le tier même-décor
   (mécanisme : effacer l'intrus + laisser le prompt repeindre) ; **mais 9 % seulement sur le tier
   cross-scène** (+0.0084), où la géométrie vaut 98 % (masque uniforme +0.0018 = 2 %). Règle de
   méthode qu'on propose à la communauté : **tout chiffre de composite doit être publié avec son
   contrôle "ancre bruitée" sur le même tier**.
5. **La série qui montre ce que le prompt peut et ne peut pas repeindre** (`memdec21up`, priorité
   au bruit) : cellule "homme au café" → un homme crédible revient mais **en veste turquoise
   alors que A porte une veste rouge vif** (le score MONTE, l'instance est FAUSSE) ; cellule
   "golden retriever" → un **second animal blanc** apparaît là où on a effacé ; cellule
   "scooter bleu" → **A ne revient pas du tout** (un objet rigide spécifique n'est pas repeignable
   depuis le texte). Variante `memdec21uf` : **buisson halluciné** en plein cadre. Lecture : le
   levier "effacer + laisser repeindre" porte les sujets génériques/animés et lâche sur l'objet
   rigide spécifique, c'est-à-dire **précisément là où une mémoire d'instance est censée servir**.
6. **Le plancher de bruit et la non-reproductibilité bit-à-bit**, mesurés et publiés (§4.4).
7. **Les selftests comme partie du protocole** : 52/52 puis 67/67 (scénarios), 17/17 (phase B),
   18 + 24 (gates), parités bitwise `max|diff| = 0.000e+00` (schedule hérité == `inference()`,
   `framebank == oracle1` en verbatim, α=0 ⇒ `memdec20`, aucune fuite d'env entre bras).

## 9. Limites (à écrire sèchement, 5 puces)

1. **Catégorie, pas individu** (chiffres §2.3 Claim 3) et sa cause mesurée (plafond de canal).
2. **Un seul substrat** : tout est mesuré sur LongLive-1.3B ; la généralisation est une hypothèse.
3. **Puissance** : 24 cellules par scénario ; le 24+/0− est solide, les deltas ±0.03 ne sont pas
   interprétables individuellement ; l'étude composite est à n = 3, seed 0.
4. **Le point pur paie le fond et le raccord** (fond 0.459 vs 0.934, cut2 20.0 vs 0.71) ; le point
   composite les rachète partiellement mais laisse un fantôme flou à l'ancienne position du sujet
   d'interlude, et les trois correctifs testés échangent ce défaut MOU contre des défauts DURS.
5. **S4p** : régression réelle (−0.013, 9+/15−) sur revisit paraphrasé, cause identifiée
   (conditionnement FiLM hors-distribution).

## 10. Conclusion (3 phrases)
Le régime où une mémoire est nécessaire est celui que les protocoles publiés ne mesurent pas ;
dans ce régime, stocker du rendu ne sert à rien (même le rendu parfait) et seule une mémoire
d'état ramène le sujet ; l'instance reste ouverte, et le goulot est nommé.

## 11. Annexes prévues
- A. Protocole complet des scénarios, prompts, triples, paraphrases, seeds, vérification de
  non-fuite (48 prompts).
- B. Registre des prédictions pré-enregistrées, verbatim, avec la date, et la table
  prédiction/mesure/verdict.
- C. Le write-replay en détail : conditions de parité bitwise, format de shard v3, arithmétique de
  collecte (43.1 MB de h par write, 1.51 GB par vidéo switch), VRAM mesurée par (K, n_bptt).
- D. Table complète des selftests avec leurs `max|diff|`.
- E. Grille de dosage de l'écriture (α × n) : le meilleur point de la grille est le point déployé
  (α=1, n=1), dégradation monotone dans les deux axes, avec le caveat "le décodeur gelé a été
  co-entraîné sur α=1, n=1" (donc la grille mesure "écrire plus fort aide-t-il AVEC LE LECTEUR
  ACTUEL"). Source : `memory2_probes_report.md` §P2/P3.
- F. Toutes les figures qualitatives non retenues + les vidéos côte-à-côte (supplementary).
- G. La colonne contaminée complète, cellule par cellule.

## 12. Ce qui doit encore tourner avant soumission (ordre de priorité)
1. **Le contrôle "ancre bruit" et "ancre d'une autre cellule" sur S2, 24 cellules** (voir
   `risks.md` attaque 3) : c'est le seul trou qui peut coûter le papier.
2. **2e et 3e seed sur S2/S2p** (48-72 cellules) pour blinder le chiffre-titre.
3. `memdec20r` sur la matrice (déjà codé, ~11 GPU-h) : tue l'asymétrie de routage.
4. Étendre l'étude composite à 2 seeds ou l'assumer comme qualitative.
5. Un scénario non-ouvrant supplémentaire (par ex. `X → A → B → C → A`) pour montrer que S2
   n'est pas une forme particulière.
