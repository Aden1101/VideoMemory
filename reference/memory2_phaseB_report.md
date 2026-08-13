# MÉMOIRE 2.0 — PHASE B (rendu + instance) — RAPPORT — 2026-07-28

Mandat : convertir les deux défauts laissés par J3 (`memory2_j3_report.md`, verdicts œil Aden) :
**(a) le rendu** (passe-bas + damier 2×2 que le juge DINO ne récompense pas) et
**(b) l'instance** (sans pression d'instance le contenu stocké n'est le bon qu'une fois sur deux).
Écrivain **A2 FIGÉ** dans tout le périmètre ; seul le décodeur bouge. Tout est additif, rien de
commité, aucun hook d'inférence ni harness touché.

| Fichier | Rôle |
|---|---|
| `wan/modules/video_memory.py` | **+`FrameMixerBlock`** et **`AnchorFrameDecoder(mixer=…)`** (défaut `False` = v1 bit pour bit) |
| `reference/runs/mem2/build_bank_dino.py` | banque DINO des vraies cibles + préproc **différentiable** partagée |
| `reference/runs/mem2/train_writer.py` | phase B : `--writer_ckpt/--freeze_writer/--mixer/--lam_dino/--dino_probe/--dino_log` |
| `reference/runs/mem2/selftest_phaseB.py` | 17 selftests (identité du mixer, warm-start A2, VAE différentiable, banque) |
| `reference/runs/mem2/probe_dino_render.py` | diagnostic rendu-vs-contenu (dose-réponse de flou, centrage, structure) |

Artefacts lourds : `/shared/ssd_14T_project/adenelmi/mem2_runs/` (`bank/bank_dino.pt`,
`writer/`, `logs/`).

---

## 0. RÉSUMÉ EXÉCUTIF

- **B2 NON LANCÉ — le gate DINO du step 0 ÉCHOUE** : `pos_gap = −0.185`, top-1 `0.000`,
  **10/10 vidéos val négatives**. Le mandat est explicite : « si pos_gap<0 au step 0, STOP ».
- **Le diagnostic dit POURQUOI, et il innocente le contenu** : au niveau de fidélité que le
  décodeur atteint aujourd'hui, **même la VÉRITÉ TERRAIN échoue au gate** (une cible réelle
  floutée à σ=32 px : top-1 0.000, gap −0.127, à comparer au décodé −0.190). Le terme
  d'instance DINO n'est pas mal conçu : il est **inutilisable tant que le rendu n'est pas
  réparé**. D'où l'ordre B1 → re-gate → B2, et non B1 ∥ B2.
- **B1 TERMINÉ (2500 steps, 1.6 h) et NÉGATIF** : le mixer ne répare pas le rendu, il le dégrade
  légèrement. Mesure **APPARIÉE** (20 ancres val, MÊMES états, MÊMES positions, seul le décodeur
  change) : `norm_mse` **0.7672 → 0.7909** (5/20 meilleures), `cos_DINO(décodé, sa cible)`
  **0.1027 → 0.0919** (8/20). Re-gate sur le décodeur B1 : `pos_gap −0.1772` (vs −0.1848), donc
  **toujours FAIL**. Contrat de gel tenu bit pour bit : `writer/delta_ref = 0.000e+00` au step
  2500.
- **Conclusion de la phase B** : le goulot du rendu **n'est PAS l'absence de mélange spatial**.
  L'œil le confirme (§7.3) : le mixer ne supprime pas le pavage, il y superpose un tramage fin.
  Le vrai goulot est **état → frame** (le h-oracle donne 0.0004, le décodeur 0.71 : l'information
  est dans h, elle n'est pas dans ce que le lecteur sait extraire de l'ÉTAT).

---

## 1. GATE DINO AU STEP 0 (LE CHIFFRE QUI DÉCIDE)

`train_writer.py --dino_probe` sur **écrivain A2 + décodeur A2** (le point de départ exact de la
phase B), 10 vidéos val held-out, 20 ancres, négatifs = **toute la banque val (632 entrées)**
avec le masque F1.3 (±2 chunks autour de la cible dans la même vidéo) :

| | top-1 | cos(ancre décodée, sa cible) | cos(ancre, meilleur négatif) | **pos_gap** |
|---|---|---|---|---|
| **espace DINO ViT-B/14 (phase B)** | **0.000** | **0.113** | 0.298 | **−0.1848** |
| espace latent crop60 (J3, rappel) | 0.000 | 0.42–0.55 | 0.61–0.83 | −0.19 à −0.36 |

Par vidéo : gap ∈ [−0.256, −0.084], **négatif 10 fois sur 10** (aucune n'est proche de zéro).
Log complet : `mem2_runs/logs/B_probe.log`. Code de sortie 2 = gate FAIL (le probe est un gate
exécutable, pas une mesure décorative).

**Verdict : le passage du latent au DINO ne change PAS le signe.** Le mandat prévoyait
« vérifie que ça démarre du BON côté en espace DINO (contrairement au latent) » : ce n'est pas
le cas. B2 aurait passé ses 300 premiers steps à pousser sur un signal de signe faux (et, avec
le gate live que j'ai implémenté, à ne rien faire du tout).

---

## 2. POURQUOI : DIAGNOSTIC RENDU vs CONTENU (`probe_dino_render.py`, 20 ancres val)

Deux hypothèses opposées, trois mesures pour les séparer.

### 2.1 Dose-réponse de flou sur les VRAIES cibles
On floute la vérité terrain de σ pixels, on ré-embarque, on refait exactement la même
récupération (mêmes négatifs, même masque) :

| σ (px) | top-1 | pos_gap | cos_pos | cos meilleur négatif |
|---|---|---|---|---|
| 0 (vérité) | **1.000** | **+0.086** | 1.000 | 0.914 |
| 1 | 1.000 | +0.072 | 0.982 | 0.910 |
| 2 | 0.950 | +0.057 | 0.943 | 0.887 |
| 4 | 0.600 | +0.022 | 0.854 | 0.832 |
| 8 | 0.400 | **−0.010** | 0.640 | 0.650 |
| 16 | 0.150 | −0.066 | 0.320 | 0.386 |
| 32 | 0.000 | −0.127 | 0.172 | 0.298 |
| **ancre décodée (A2)** | **0.000** | **−0.190** | **0.103** | 0.293 |

**Lecture, et c'est LE résultat de la phase B côté gate** : la frame décodée est, du point de
vue de DINO, **plus loin de sa propre cible qu'une version de cette cible floutée à 32 pixels**
(cos 0.103 vs 0.172) — et le gate est déjà mort à σ=8. Autrement dit, **à la fidélité actuelle
le gate serait négatif même si le décodeur restituait la BONNE identité parfaitement** : le
cosinus DINO d'une image hors-variété mesure « à quel point c'est flou », pas « qui c'est ».
Le terme d'instance ne peut pas précéder la réparation du rendu.
(Note : le plafond vérité-terrain n'est que +0.086 parce que F1.3 laisse dans les négatifs les
frames LOINTAINES de la même vidéo — même scène, cos 0.91. La tâche est « quelle FRAME », pas
« quelle vidéo ». C'est voulu — c'est la pression anti-contamination — mais ça rend la marge
intrinsèquement fine, à garder en tête au moment de re-gater.)

### 2.2 Centrage (le correctif standard quand un côté vit dans une direction commune)
En retirant la moyenne de population de chaque côté (moyenne des ancres décodées d'un côté,
moyenne de la banque de l'autre) puis renormalisation : gap **−0.067** (contre −0.190 brut),
top-1 0.05. **Ça aide beaucoup mais ça ne renverse pas le signe** : le centrage n'est pas un
sauvetage, il confirme juste qu'une grosse part de l'écart est une direction « rendu » commune.

### 2.3 Structure métrique préservée par le rendu
Corrélation de rang entre les similarités décodé-décodé et vérité-vérité : **Spearman +0.083**,
accord du plus proche voisin **0/20** (hasard 1/19). À ce niveau de rendu, la géométrie DINO des
frames décodées ne porte quasiment rien de la géométrie réelle. Ça **ne dit pas** que l'état ne
contient pas l'instance (le test 3-bras cheval prouve le contraire, et le h-oracle donne
val 0.0004) : ça dit que **rien de ce contenu n'est visible à travers ce rendu**.

---

## 3. VRAM MESURÉE (avant tout run, mandat)

`--vram_probe` avec le backward COMPLET de la phase B (replay + décodeur v2 + **VAE 1 frame +
DINO**), écrivain gelé, RTX 6000 Ada 49 GB, `torch.cuda.reset_peak_memory_stats()` avant chaque
cellule :

| K (positions) | n_bptt | pic VRAM | s/step |
|---|---|---|---|
| 1 | 0 | 9.0 GB | 2.3 |
| 1 | full | 9.8 GB | 2.2 |
| **3** | **0** | **10.9 GB** | **2.0** |
| 3 | full | 11.4 GB | 2.2 |

**Très en dessous du plafond de 40 GB du mandat : aucune réduction nécessaire** (ni `n_bptt`,
ni décodage 1 position sur K — le terme DINO ne paie qu'**une** frame par vidéo, `--dino_k 1`).
Deux remarques :
- l'écart n_bptt 0 vs full est marginal (0.5–0.8 GB) parce qu'avec l'écrivain gelé **aucun
  paramètre du replay ne demande de gradient** : le graphe n'existe pas de toute façon ;
- le `peak VRAM` imprimé au step 1 d'un run (24.8 GB) inclut le transitoire de construction du
  pipeline 1.3B + T5 (`max_memory_allocated` n'est jamais remis à zéro) ; le vrai pic par step
  est celui du tableau. Même caveat sur les 25.6 GB annoncés au J3.

**Cache disque des états : REJETÉ, à raison.** Un état = 3 blocs × 3 tenseurs 1536×6144 bf16
= **170 MB par position** ; 139 vidéos × ~35 positions ⇒ ~800 GB. Inutile de toute façon :
selftest 5 prouve que `replay n_bptt=0` est **bitwise** le replay full-BPTT (le forward de
`replay_writes` est inchangé, seul le graphe est coupé), donc l'état consommé par le décodeur
est exactement celui d'un replay complet, pour ~0 GB et 2 s/step.

---

## 4. SELFTESTS (`selftest_phaseB.py`, 17/17 PASS, GPU 0)

- **mixer = identité bitwise** : un `state_dict` sans mixer se charge dans le décodeur v2
  (14 clés manquantes, **toutes `mixer.*`**, 0 inattendue) et la frame de sortie est
  `torch.equal` à celle de v1 (max |diff| **0.000e+00**) — après avoir perturbé la tête,
  sinon un décodeur zéro-init passerait le test en trichant. Contrôle inverse : un `pw2`
  non nul CHANGE la frame (le mixer n'est pas mort).
- **warm-start A2** : 0 clé inattendue, manquantes ⊆ `mixer.*`, et **23/23 tenseurs non-mixer
  bitwise identiques au checkpoint** (max |delta| 0.000e+00) — un `fuse`/`head` silencieusement
  ré-initialisé aurait ressemblé à « le mixer aide ».
- **structure** : dernier pointwise zéro-init (poids ET biais), depthwise 3×3 groupé sur la
  grille 30×52, **37.89M params** (v1 33.14M, mixer +4.76M) sous le budget 40M.
- **banque DINO** : (4001, 768) L2-normalisée, alignée ligne à ligne sur `bank.pt["entries"]`,
  auto-récupération 200/200, **30/30 paires de jumeaux du MÊME côté du split** (appariées par
  `corpus_idx`), 105/105 ancres switch adressables par `(video_id, frame)`, jumeau =
  négatif dur mais pas doublon (cos moyen 0.767, max 0.942).
- **VAE 1 frame différentiable** : états réels → décodeur v2 → VAE (1 latent → 1 frame
  480×832 ∈ [0,1]) → crop60 → DINO → InfoNCE → backward : gradient **non nul et fini** sur la
  tête (2.1e+02), le **mixer** (8.3e-01), les **queries** (3.2e+02), et jusqu'à l'**écrivain**
  quand on garde le graphe (7.6e+02). Pic 11.6 GB avec le checkpointing du VAE.
- **états = données** : `n_bptt=0` == full BPTT **bitwise** sur les 3 blocs.

---

## 5. RUNS

### B1 — LANCÉ ET TERMINÉ (GPU 0, 2500 steps, 2.2–2.4 s/step, 1.6 h) — résultats §7
```
train_writer.py --phase B --run_name mem2_writer_B1 --steps 2500
  --writer_ckpt writer/mem2_writer_A2_final.pt --freeze_writer
  --mixer --mixer_blocks 2 --mixer_hidden 768
  --dec_ckpt writer/mem2_writer_A2_final.pt
  --ages 5 10 20 35 --min_age 5 --anchor_frac 0.7   (les hyperparams data de A2)
  --lam_inst 0 --lam_dino 0 --dino_log              (DINO MESURÉ, jamais optimisé)
  --lr_d 1e-4 --wd_d 0.01 --warmup 200 --K 3 --bs 1
```
wandb `mem2_writer_B1`, log `mem2_runs/logs/B1.log`, pid `logs/B1.pid`.
`--dino_log` mesure `dino/gap` à CHAQUE step et `val/dino_gap` à chaque val (λ=0, sous
`no_grad` : aucune influence sur le gradient) — donc **B1 répond lui-même à la question
« le gate s'ouvrirait-il si le rendu était réparé ? »**, gratuitement.

Contrat de gel vérifié : `writer/delta_ref` asserté `== 0.0` à chaque val ET à la fin du run
(l'écrivain n'est même pas dans l'optimiseur ; `prox` désactivé). Au step 1 : `delta_ref
0.000e+00`, 4 tenseurs du mixer avec un gradient non nul.

### B2 — NON LANCÉ (gate §1)
Commande prête, à une variable près (le décodeur de départ) :
```
train_writer.py --phase B --run_name mem2_writer_B2 --steps 2500
  --writer_ckpt writer/mem2_writer_A2_final.pt --freeze_writer --mixer
  --dec_ckpt writer/mem2_writer_B1_final.pt        # <- le décodeur RÉPARÉ, pas A2
  --lam_dino 0.3 --lam_dino_ramp 300 600 --dino_gate 1 --dino_k 1
  ... (mêmes hyperparams data)
```
Le gate est aussi **live** : `--dino_gate 1` remet λ_dino à 0 tant que le `val/dino_gap` le plus
récent est ≤ 0, donc B2 est sûr par construction — mais le lancer aujourd'hui, c'est au mieux
un doublon de B1 (λ resterait à 0), au pire un run qui pousse sur du bruit. GPU 1 laissé libre.

---

## 6. COURBES ATTENDUES / CRITÈRES DE LECTURE DE B1

- `val/loss` (non pondérée, cond ON, 26 vidéos val, états rejoués) part de **0.7153** (A2 au
  step 2500). Le mixer ne change rien au step 0 (identité) : toute baisse est imputable à lui
  seul, l'écrivain étant gelé bit pour bit. Seuil de lecture : ≥5 % relatif (≤0.680) = le
  mélange spatial sert ; plat = le goulot du rendu n'est pas l'absence de mélange spatial.
- `val/dino_cos_pos` : **0.113** au départ. C'est la métrique de FIDÉLITÉ qui décide de B2 ;
  d'après la dose-réponse §2.1 il faut la remonter vers **≥ 0.6** (≈ σ=8 px) pour espérer un
  `pos_gap` proche de 0, et vers **≥ 0.85** (σ=4) pour un gap franchement positif.
- `val/dino_gap` : **−0.185**. C'est le gate de B2, re-mesuré toutes les 100 steps.
- `val/loss_by_kind/switch/anchor` (0.7130 chez A2) : la tâche du revisit proprement dite.
- Tripwires inchangés : `state/rowcap_frac` (0.173), `mem/influence_decay@k`.

---

## 7. RÉSULTATS DE B1 (2500 steps, terminé, `mem2_writer_B1_final.pt`)

### 7.1 Contrat de gel — TENU
`writer/delta_ref = 0.000e+00` à **chaque val** (25 mesures) **et** au check de fin de run
(`PHASE-B FREEZE CHECK at step 2500`). L'écrivain A2 n'est même pas dans l'optimiseur ; `prox`
désactivé ; `n_bptt` forcé à 0 (états bitwise ceux d'un replay complet, selftest 5).
Donc **tout delta observé est imputable au décodeur seul**.

### 7.2 Val (26 vidéos held-out, non pondérée, cond ON) — PAS D'AMÉLIORATION

| | A2 (départ) | B1 step 2500 |
|---|---|---|
| `val/loss` | 0.7153 | **0.7400** |
| `val/loss_by_kind/switch/anchor` (le revisit) | 0.7130 | 0.7418 |
| `val/loss_by_kind/switch/recency` | 0.7221 | 0.7378 |
| `val/loss_by_kind/single/recency` | 0.7098 | 0.7365 |
| `val/loss_by_age` a5 / a10 / a20 / a20p | 0.699 / 0.712 / 0.715 / 0.839 | 0.713 / 0.730 / 0.749 / 0.916 |
| `val/dino_cos_pos` | 0.113 (probe) | **0.0957** |
| `val/dino_gap` | −0.185 (probe) | −0.1677 |
| `state/rowcap_frac` (tripwire) | 0.173 | 0.173 (plat, sain) |

La trajectoire est **plate du step 100 au step 2500** (`val/loss` 0.729 → 0.740 en oscillant
entre 0.725 et 0.776 ; `val/dino_cos_pos` 0.0997 → 0.0957 sans jamais bouger). Les vals sont
bruitées (positions retirées au hasard à chaque val), d'où la mesure appariée ci-dessous.

### 7.3 Mesure APPARIÉE A2 vs B1 (`viz_phaseB.py`, 20 ancres val, mêmes états, mêmes positions)

```
MEAN over 20 PAIRED val anchors: mse 0.7672 -> 0.7909 (5/20 meilleures)
                                 cos_DINO 0.1027 -> 0.0919 (8/20 meilleures)
```
**15/20 vidéos se dégradent en reconstruction.** Ce n'est pas du bruit de val : c'est un
**sur-apprentissage**. La phase B dégèle **tout** le décodeur (37.89M params) alors que la phase
A ne dégelait que l'adressage (7.32M, garde-fou F2.1) — 139 vidéos d'entraînement ne portent pas
37.9M paramètres libres pendant 2500 steps.

### 7.4 Verdict à l'ŒIL — `reference/runs/mem2/eyepack_phaseB/target_vs_A2_vs_B.png`
6 lignes (cible | A2 v1 | B1 v2+mixer), pleine résolution VAE.
- Les frames A2 sont des **taches de couleur justes mais sans structure** : la teinte et la
  disposition générale sont là (le cheval brun sur l'herbe verte, la casserole claire, le
  ciel), la géométrie ne l'est pas.
- Les frames B1 **ne suppriment pas le pavage** : elles y **ajoutent un tramage régulier fin**
  visible sur toute l'image (net sur les lignes 2, 4, 5). Le mélange spatial a appris à
  redistribuer de l'énergie haute fréquence, pas à recoller les patchs voisins.
⇒ **Le diagnostic « damier = pas de mélange spatial » est réfuté.** Le pavage visible n'est pas
un artefact de recollement : c'est la signature d'un lecteur qui n'extrait de l'état qu'une
information basse fréquence par patch.

### 7.5 Re-gate DINO sur le décodeur B1 (`logs/B1_regate.log`, mêmes 10 vidéos)
`top-1 0.000 | cos_pos 0.0999 vs meilleur négatif 0.2771 | **pos_gap −0.1772**` (A2 : −0.1848).
**FAIL** — l'amélioration de 0.008 est du bruit, très loin des ~0.6 de `cos_pos` que la
dose-réponse §2.1 exige pour espérer un gap nul. **B2 reste bloqué.**

---

## 8. CE QUE ÇA IMPLIQUE POUR LA SUITE (décision Aden)

1. **Les deux défauts de J3 ne sont pas deux problèmes, c'est UN problème.** Le rendu et
   l'instance butent sur la même chose : la frame décodée n'a que ~29 % de l'énergie de sa
   cible (norm_mse 0.71) et vit hors de la variété des images naturelles. Tant que ce chiffre
   ne bouge pas, aucun juge perceptuel (DINO) ne peut récompenser l'identité, donc **aucun
   terme d'instance perceptuel n'est entraînable**. C'est démontré, pas supposé : à cette
   fidélité, la vérité terrain elle-même rate le gate (§2.1).
2. **Le mixer est mort comme levier de rendu** (§7.3/7.4) : ne pas le re-tenter, ne pas monter
   sa capacité. Le code reste additif et inerte par défaut (`mixer=False`), donc rien à annuler.
3. **Le prochain levier n'est pas le lecteur de l'ÉTAT, c'est l'ÉTAT.** Chaîne de chiffres :
   h contient la cible presque exactement (h-oracle val **0.0004**) ; l'état, lu par le meilleur
   décodeur qu'on sache entraîner, en rend **0.71**. Toute la perte est dans l'écriture
   h → état. Or A2 a entraîné l'écrivain… et `mem/influence_decay@k` reste plat (0.04–0.07,
   Δloss ±0.002) : **neutraliser n'importe quelle écriture ne change quasiment rien à l'état**.
   C'est LA mesure à faire bouger, et elle ne bougera pas en changeant de décodeur.
4. **Deux pistes concrètes, par ordre de coût** :
   - *(0 $, 1 GPU-h)* re-jouer une phase B **avec le garde-fou F2.1** (adressage seul dégelé,
     ou `--lr_d 3e-5`) pour trancher si le 0.7400 est du sur-apprentissage pur ou un plafond :
     si l'adressage seul redescend sous 0.7153, on sait que le budget de paramètres était le
     problème, pas la fonction ;
   - *(la vraie)* attaquer la capacité d'écriture : l'état est 3×(1536×6144) mais
     `state/rowcap_frac` est à 0.17 et l'influence par écriture est plate ⇒ l'écrivain n'écrit
     presque rien d'exploitable, exactement le diagnostic du J3 §5.5. C'est le sujet de la
     mémoire 2.x, pas d'un décodeur.
5. **Piste à ne pas perdre pour l'instance** : le centrage (§2.2) enlève déjà 65 % de l'écart
   (−0.190 → −0.067) sans rien apprendre. Une tête de métrique apprise sur la banque (option (c)
   du J3 5.1) ou des négatifs **décodés** plutôt que réels contourneraient l'artefact
   « hors-variété » sans attendre un rendu photo-réaliste. Hors spec figée ce coup-ci ; c'est
   la seule route connue vers un terme d'instance AVANT la réparation du rendu.

---

## 9. INVENTAIRE (chemins absolus)

| | |
|---|---|
| décodeur v2 + mixer | `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/video_memory.py` (`FrameMixerBlock`, `AnchorFrameDecoder(mixer=…)`) |
| trainer phase B | `.../reference/runs/mem2/train_writer.py` |
| banque DINO | `.../reference/runs/mem2/build_bank_dino.py` → `/shared/ssd_14T_project/adenelmi/mem2_runs/bank/bank_dino.pt` (4001×768, 6 MB) |
| selftests (17/17) | `.../reference/runs/mem2/selftest_phaseB.py` |
| diagnostic rendu/contenu | `.../reference/runs/mem2/probe_dino_render.py` → `mem2_runs/logs/dino_render_probe.{log,json}` |
| eye-pack | `.../reference/runs/mem2/viz_phaseB.py` → `.../reference/runs/mem2/eyepack_phaseB/target_vs_A2_vs_B.png` |
| ckpt B1 | `/shared/ssd_14T_project/adenelmi/mem2_runs/writer/mem2_writer_B1_final.pt` (+ `_step{500..2500}.pt`) |
| logs | `mem2_runs/logs/{B_probe,B1,B1_regate,viz_B1,dino_render_probe,bank_dino}.log`, pid `B1.pid` |
| wandb | `mem2_writer_B1`, `mem2_writer_B_probe`, `mem2_writer_B1_regate` (projet `anchor-forcing-memory`) |

Rien n'est commité. Aucun hook d'inférence ni fichier du harness modifié (l'éval `memdec21`
reste hors périmètre). GPU 1 n'a jamais été mobilisé pour un run long (B2 stoppé au gate).
