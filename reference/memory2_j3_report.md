# MÉMOIRE 2.0 — RAPPORT J2+J3 (write-replay : banque, gates, trainer écrivain) — 2026-07-27

Mandat : `reference/memory2_brief.md` + `memory2_design.md`, patchs DIE-ON de
`memory2_redteam.md` (section RÉSOLUTION DU DÉBAT) prioritaires en cas de conflit.
Code livré (tout additif, rien de modifié dans `wan/`, `collect_pairs.py` ou le harness) :

| Fichier | Rôle |
|---|---|
| `reference/runs/mem2/mem2_data.py` | couche DONNÉES partagée : index vidéo, split par vidéo, jumeaux, cibles/âges, embedding latent crop60 |
| `reference/runs/mem2/selftest_s1b.py` | **gate F5.2** : MEM_ZERO_ALL = multiplication (danger réel) + scan NaN/Inf des 22 parts |
| `reference/runs/mem2/build_bank.py` | **A** : banque InfoNCE (cibles + cond + kNN cond + split + jumeaux) + 9 selftests |
| `reference/runs/mem2/probe_h_oracle.py` | **B / gate F5.3a** : borne supérieure « ce que h contient » |
| `reference/runs/mem2/train_writer.py` | **D** : trainer write-replay (phases A/B, toy F3.1, sondes VRAM/retrieval) |
| `reference/runs/mem2/selftest_trainer.py` | **E** : 20 selftests CPU + smoke GPU |

Artefacts lourds : `/shared/ssd_14T_project/adenelmi/mem2_runs/` (`bank/`, `h_oracle/`,
`writer/`, `logs/`). Aucun commit git.

---

## 1. VERDICTS DES GATES

### S1b (F5.2) — **PASS**, gate ouvert   (`logs/s1b.log`, 2.6 min, GPU 0)
- **Le danger est RÉEL et démontré** : un NaN artificiel injecté dans la sortie *read* de la
  cellule du bloc 5 (monkeypatch de `_project_output`, restauré ensuite) sous MEM_ZERO_ALL=1
  contamine les h des blocs aval (`non-finite norm1 outputs {5: 28, 16: 29, 27: 29}`) **et**
  les latents : `mem_out * 0.0` est bien une multiplication, pas un skip structurel.
  Defense-in-depth constatée au passage : l'assert de range fp16 du collecteur v3 se déclenche
  aussi (`|h| max nan out of fp16 range`) → une telle vidéo n'aurait jamais été écrite.
- **Ça n'est pas arrivé sur la campagne réelle** : scan de **22 parts / 229 GB / 165 vidéos /
  15 525 tenseurs h / 111.6 G éléments** → **0 non-fini**, 0 latent non-fini.
  Par bloc : absmax 35.5 / 37.5 / 33.0, RMS 1.0000 (h est RMS-normé par `norm1`, cohérent).
- Contrôle de restauration : le rollout post-patch est byte-identique au contrôle propre.

### h-oracle (F5.3a) — **PASS très large : la borne n'est pas le facteur limitant**
(`logs/h_oracle.log`, extraction 3 s + 3 000 steps en ~2 min, GPU 2, wandb `mem2_h_oracle`)
- Protocole : lecture per-token du **h brut** du chunk qui contient la cible (tokens de la
  dernière frame, `h[:, 2*1560:3*1560]`, 3 blocs), readout = **le chemin CONTENU exact de
  l'AnchorFrameDecoder** (read_norms → 4608→3072→GELU→1536 → LN → head 64 → même unpatchify),
  **sans cond ni FiLM** (pas de raccourci texte, F1.1), split **par vidéo**.
- **val norm_mse (non pondérée, vidéos held-out) = 0.0004** — à comparer à
  **0.867 (décodeur v1.1)** et **0.812 (v1.2)**, mêmes unités (1.0 = prédire zéro).
- Cibles ANCRE seules (dernière frame de A) : **0.0005**. Contrôle h mélangé : **1.87** (> 1.0 : un h faux fait pire que prédire zéro, la sonde ne fuit rien).
- **Lecture** : le flux h contient la frame cible de façon quasi exacte ⇒ **le goulot n'est
  ni h ni le lecteur : c'est l'ÉTAT** (la thèse du brief est confirmée côté information).
  Caveat honnête à garder au papier : les val-sets diffèrent (v1.x = 16 échantillons / 8
  vidéos, F4.1) et le h-oracle lit un h *aligné* sur la cible (c'est bien une borne
  supérieure, pas une prédiction de performance).

### Toy-overfit (F3.1) — **FAIL (2 critères sur 3), mais le FAIL porte sur (b) à 1.1 point**
(`logs/toy.log` + `logs/toy_hshuffle.log`, 8 vidéos switch, cibles **d'âge ≥ 10 writes**,
200 steps/variante, K=3, batch 1, **cond coupé (`cond_dropout=1.0`)** pour interdire tout
raccourci texte : l'état est la SEULE route vers la cible, wandb `mem2_writer_A_toy`)

| critère | mesure | seuil | verdict |
|---|---|---|---|
| (a) la loss s'écrase sur des cibles ≥10 writes | **0.9829 → 0.4704** (ratio 0.479) | < 0.75 | **PASS** |
| (b) `n_bptt_writes=1` fait NETTEMENT pire | 0.5125 vs 0.4704 (**+8.9%**) | +10% | **FAIL** |
| (c) h-shuffle dégrade | 0.9395 vs 0.4704 (**+99.7%**) | +10% | **PASS** |

**Ce que ça dit vraiment** :
1. **Le gradient traverse la chaîne et l'écrivain apprend** : −52% de loss en 200 steps sur des
   cibles vieilles de 10 à 20 writes, sans aucun accès au texte (risque n°1 du design levé sur
   le plan mécanique : ni explosion, ni évanouissement total, pas de NaN, grad clip jamais
   saturé).
2. **La récurrence est indispensable** : rejouer la séquence h d'une AUTRE vidéo (tirée au sort à
   chaque step) laisse la loss à ~0.94, c'est-à-dire au niveau du prédicteur nul. L'état porte
   donc bien de l'information spécifique à SA vidéo — c'est le contrôle le plus informatif du
   lot et il est net.
3. **En revanche la PROFONDEUR du gradient n'apporte que ~9%** : couper le BPTT au dernier write
   coûte moins de 10% de loss… pour 3.3× moins de temps (2.4 s/step vs 8.0). Le critère
   pré-enregistré tombe juste du mauvais côté (+8.9% vs +10% requis) ⇒ **verdict FAIL, à lire
   comme « inconclusive sur la profondeur », pas comme « la chaîne n'apprend pas »**.
   Deux façons de le trancher proprement, au choix d'Aden :
   - **(rapide, ~1 GPU-h)** re-jouer le toy avec `--fixed_positions` (implémenté : un jeu de
     positions FIXE par vidéo, donc un vrai overfit au lieu d'un tirage neuf à chaque step, la
     variance par step allant aujourd'hui de 0.25 à 1.10) et 400 steps ;
   - **(gratuit)** lancer la phase A avec `--n_bptt 12` et lire `mem/influence_decay@k`, qui
     mesure DIRECTEMENT si la profondeur sert, sur le vrai corpus.
4. **Bug trouvé et corrigé pendant le gate** : dans le run 1, le contrôle h-shuffle tirait sa
   vidéo de remplacement dans le BATCH (taille 1) → aucun échange, courbe bit-identique à
   `main`. Corrigé (`self.shuffle_pool`, assert si le pool est vide) et la variante a été
   re-jouée seule (`--toy_variants hshuffle`, les deux autres relues depuis `toy_result.json`).

---

## 2. MESURES D'INGÉNIERIE

### VRAM du backward complet (F5.4, mesurée AVANT de fixer K/batch)
`--vram_probe`, GPU RTX 6000 Ada 49 GB, 1 vidéo switch (35 writes × 3 blocs), backward complet
(replay + décodage + recon + InfoNCE), **pas de VAE dans la loss** :

| K (positions) | n_bptt | pic VRAM | s/step |
|---|---|---|---|
| 1 | full (35) | 15.9 GB | 5.0 |
| 1 | 12 | 11.2 GB | 3.3 |
| 1 | 4 | 8.7 GB | 2.1 |
| 3 | full (35) | 23.4 GB | 7.4 |
| **3** | **12** | **15.2 GB** | **3.9** |
| 3 | 4 | 12.3 GB | 2.6 |

(le smoke complet, K=3 + banque + prox + optimiseur, a piqué à **25.6 GB** : le budget annoncé
au design, 16-22 GB, était donc juste mais SANS marge à K=3/full-BPTT — d'où le choix n_bptt=12
au §4, qui rend la règle maison « marge >25 GB » confortable.)

### Espace de retrieval de l'InfoNCE (mesure ajoutée : `--acc_probe`)
`--acc_probe` : on décode 10 vidéos val avec le décodeur **v1.2 gelé** sur des états rejoués,
puis on mesure le retrieval de l'ancre contre les 632 entrées val de la banque.

| pool | centré | top-1 | cos(ancre, sa cible) | cos(ancre, meilleur négatif) |
|---|---|---|---|---|
| 2 | non | 0.000 | 0.423 | 0.612 |
| 2 | oui | 0.000 | 0.294 | 0.513 |
| 4 | non | 0.000 | 0.454 | 0.654 |
| 4 | oui | 0.000 | 0.320 | 0.556 |
| 8 | non | 0.000 | 0.500 | 0.726 |
| 8 | oui | 0.000 | 0.365 | 0.640 |
| 16 | non | 0.000 | 0.547 | 0.830 |
| 16 | oui | 0.000 | 0.404 | 0.767 |

**Lecture : l'InfoNCE en espace latent démarre du mauvais côté** (l'ancre est plus proche de la
frame d'une AUTRE vidéo que de sa cible, à tous les poolings, centrage compris). Conséquence
opérationnelle au §5.1.

---

## 3. CE QUI EST IMPLÉMENTÉ (et où le vérifier)

### `mem2_data.py` (couche données, importée par A/B/D/E)
- Index des shards : **165 vidéos** (135 campagne + 30 jumeaux), **5 175 writes**, 22 parts,
  229 GB, construit en 2.7 s (`torch.load(mmap=True)` : les parts de 11-14 GB ne sont jamais
  matérialisées). `load_part` **ASSERTE `mem_zero_all is True`** : un shard memon serait refusé.
- **Convention d'âge** (utilisée partout, y compris les buckets wandb) : `age = i - j + 1`,
  âge 1 = « décode ta dernière écriture » (la cible triviale que F1.6 dénonce), âge 20 = 20
  writes en arrière. `sample_positions` tire l'âge dans {1,2,5,10,20} et refuse tout ce qui est
  plus jeune que `min_age`.
- Positions **ANCRE** = la tâche du revisit : cible = dernière frame de A (`switch_frame-1`),
  état post-switch (`chunk >= switch_chunk + 5`, la règle min_gap du collecteur), cond = A.
- Split **par vidéo** (F4.1), stratifié, **26 vidéos val** (18 switch + 8 single, les jumeaux
  suivent leur frère), déterministe et re-testé.
- `latent_embed` : crop60 → avg-pool 2×2 → RMS-norm → L2-norm (l'espace de l'InfoNCE, **sans
  VAE**, conformément au mandat), différentiable côté ancre.

### `build_bank.py` (A) — banque construite, 9 selftests VERTS
`bank/bank.pt` (73 MB) + `bank/bank_latents.pt` (0.80 GB) :
**4 001 entrées** (24 cibles réparties sur toute la séquence de writes par vidéo + l'ancre des
105 switch), embeddings latents (dim 8 928), cond T5 poolé par vidéo (165×4096), **kNN cond
k=8** (self et jumeau exclus), split (3 369 entrées train / 632 val), index des jumeaux.
Selftests : held-out grep 0 hit sur prompt ET prompt_b (16 prompts probe, les deux sens),
≥20 vidéos val, jumeaux jamais séparés, **30/30 jumeaux = même prompt + seed décalé de
+100000**, embeddings unitaires, ancre == `switch_frame-1` pour les 105 switch, étalement des
âges (span médian 34 writes), kNN propre, auto-retrieval 401/401, aucun `corpus_idx` à cheval.
**Diagnostic clé** : pour **93/145** sondes, le négatif le plus dur est le JUMEAU (même prompt,
autre seed) ⇒ F1.1b est validé empiriquement, le jumeau est bien le négatif qui compte.

### `train_writer.py` (D)
Chaque patch DIE-ON est implémenté et tracé dans le code (tag `Fx.y` en commentaire) :
- **F2.3** warm-start = θ_l1 lu **directement dans le ckpt** (`generator_ema` de
  `l1_longlive/keep/checkpoint_model_002000/model.pt`) ; `assert delta_l1 == 0.0` au step 0
  (mesuré : `0.000e+00`, chemins `pipeline` ET `direct`). θ_w = 18 tenseurs, **21.27M** params.
- **F3.3** `AnchorFrameDecoder(detach_states=False)` + assert de gradient non nul sur
  `qkv.weight` après le premier backward (mesuré : `|grad|_1 = 4.18e+01`).
- **F2.1** phase A = **adressage seul** dégelé (`queries`, `pos_row`, `pos_col`, `q_norms`) =
  7.32M / 33.1M ; contenu (`fuse*`, `out_norm`, `head`, `read_norms`) et **FiLM cond** gelés.
  Phase B (`--phase B`) dégèle tout.
- **F1.6** cibles vieillies + `loss_by_age` (a1/a2/a5/a10/a20) + `loss_by_kind`
  (switch/anchor, switch/recency, single/recency) en train ET en val ; `--single_w 0.3`.
- **F1.1** négatifs = toute la banque du bon côté du split (3 369 : contient par construction
  le jumeau et les 8 plus proches voisins cond ; avec `--n_neg K` ils sont AJOUTÉS de force),
  `--cond_dropout 0.5`, **marge causale** `acc(état vrai) − acc(état d'une AUTRE vidéo, même
  cond)` loggée dès le step 0 (état de l'échantillon précédent : marche à batch 1).
- **F1.3** masquage intra-vidéo **uniquement** dans ±2 chunks autour de la cible : les frames de
  l'autre segment restent des négatifs (testé : 4 masquées / 21 conservées sur un exemple).
- **F3.2** `mem/influence_decay@k` (rejeu avec `h_{T−k}` remplacé par sa moyenne par token,
  k ∈ {1,2,5,10,20}) → `||Δétat||/||état||` **et** Δloss décodée ; `loss_by_age`.
- **F4.1** split par vidéo + proximal `||θ_w − θ_l1||²` **obligatoire** (`--prox 1e-4`,
  accumulé APRÈS la copie des gradients bf16→masters fp32, ordre vérifié).
- **F5.4** VRAM du backward complet mesurée et loggée au step 1 + `--vram_probe` (table §2).
- Précision : **masters fp32 + forward bf16** (le round-trip explicite du design 1.4c : la
  fonction entraînée EST la fonction déployée) ; `wd = 0` sur θ_w, `wd 0.01` sur le décodeur.
- Ckpt écrit au format **MEM_WRITER** du design §4 : `{"writer": {"5": sd, "16": sd, "27": sd},
  "decoder": sd, "step", "args", "theta_l1_ckpt"}`, poids écrivain en **bf16**.

### `selftest_trainer.py` (E) — 20 checks CPU + smoke GPU, TOUS VERTS
Identité `norm_mse_per(...).mean() == train_decoder.norm_mse` (les deux subject_weight),
prédiction nulle = 1.0 exactement, masquage F1.3, jumeau dans le softmax (banque pleine ET
sous-échantillonnée), split/jumeaux/déterminisme, âges et frames des cibles, ancre,
**delta_l1 == 0 sur CPU**, **F3.3 sur une cellule miniature** (le gradient atteint
qkv/lrs/k_norm/decay_logit à travers le décodeur, `q_norm` n'en reçoit aucun = contrôle négatif)
**et la preuve de la bombe** : avec le décodeur PAR DÉFAUT (`detach_states=True`) l'écrivain
reçoit exactement ZÉRO gradient — les 2 jours de courbe plate annoncés par F3.3.
Smoke GPU (2 steps, `--load direct`, 20 s de démarrage) : loss finie, gradient non nul,
marge causale loggée, θ_w bouge (delta_l1 2.4e-07 après 2 steps).

---

## 4. HYPERPARAMS RETENUS POUR LE RUN LONG (phase A)

```
--phase A --steps 2500 --bs 1 --K 3 --n_bptt 12        # 15.2 GB, 3.9 s/step -> ~2.7 h
--lr_w 1e-4 --lr_d 1e-4 --wd_d 0.01 --prox 1e-4 --clip 1.0 --warmup 200 --min_lr_ratio 0.05
--subject_weight 6.0 --single_w 0.3 --ages 1 2 5 10 20 --anchor_frac 0.5
--tau 0.07 --n_neg 0 (banque pleine) --mask_chunks 2 --cond_dropout 0.5
--lam_inst 0.0  (voir §5.1 : le terme d'instance NE DOIT PAS être rampé en l'état)
--val_every 100 --probe_every 250 --ckpt_every 500 --load pipeline
```
Justifications mesurées : `n_bptt 12` divise la VRAM par 1.5 et le temps par ~2 vs full BPTT
tout en gardant 12 writes de gradient (le decay 0.95^12 ≈ 0.54, donc la fenêtre de gradient
couvre l'essentiel de l'influence réelle) **et** respecte la règle maison marge >25 GB (full
BPTT à K=3 laisse 25.6 GB libres sur 49, à la limite). `--load pipeline` (chemin canonique du
harness) : le démarrage coûte ~2 min, négligeable sur un run de 2.7 h.
Baseline step-0 mesurée au smoke (val non pondérée, cond ON, vidéos held-out, états rejoués,
**seulement 2 vidéos × 3 positions : à re-mesurer sur les 26 vidéos val avant le run**) :
**val/loss = 0.696** (par type : ancre 0.608, recency switch 0.672, single 0.762 ; par âge :
a1 0.799, a10 0.660, a20 0.698). C'est LE chiffre que la phase A doit battre de ≥5% relatif
(go/no-go du design 3.3), pas le 0.812 de v1.2 (autre val-set, autres états).

---

## 5. CE QUI RESTE AVANT LE TRAIN LONG

### 5.0 (à trancher) Le critère (b) du toy : profondeur de BPTT
Voir §1 : +8.9% au lieu des +10% requis. Recommandation : lancer la phase A avec `--n_bptt 12`
(3× moins cher que le full BPTT pour ≤9% de signal en moins d'après le toy) et lire
`mem/influence_decay@k` sur le vrai corpus ; re-jouer le toy avec `--fixed_positions` seulement
si Aden veut un verdict binaire propre avant de payer la phase A.

### 5.1 (BLOQUANT, arbitrage Aden) Le terme d'instance en espace LATENT est inutilisable tel quel
Mesure `--acc_probe` (§2) : **top-1 = 0.000 à tous les poolings, centré ou non**, et l'écart
`cos(ancre, sa cible) − cos(ancre, meilleur négatif)` est **négatif de 0.19 à 0.36**. Autrement
dit une ancre décodée par v1.2 est SYSTÉMATIQUEMENT plus proche de la frame d'une autre vidéo
que de sa propre cible : l'InfoNCE latent démarre 100% du mauvais côté, son gradient pousserait
d'abord sur l'artefact « frame décodée = passe-bas » et pas sur l'instance. Trois issues :
(a) **phase A en reconstruction seule** (`--lam_inst 0`) et n'activer le terme que quand
`inst/pos_gap` (loggé à chaque val) devient positif — recommandé, zéro risque ;
(b) revenir à l'espace **DINO** du design 3.4bis, ce qui exige un décodage VAE différentiable
dans la loss : **contredit le mandat J3** et rouvre F1.5 (ancre adverse) → décision Aden ;
(c) apprendre une projection sur la banque (tête de métrique) : nouveau composant, non gaté.
Le flag `--lam_inst` est inerte à 0 (le terme n'est même pas construit), donc l'option (a) est
byte-safe.

### 5.2 Le hook MEM_WRITER n'est PAS posé (volontairement)
Le mandat interdit de toucher au harness. Il reste donc à faire au J6 (design §4) :
`run_reid_probe.build_pipeline` + `cfg._memwriter_ckpt` + arms `memdec20`/`memdec20w`, et au
passage les 2 bugs F5.8 (`memdec*` absents de `all_arms`, `memdec10` dans `MAG_ARMS` sans
entrée `ARM_ENV`). Le ckpt écrivain est déjà écrit au format attendu (bf16, par bloc).

### 5.3 Décalage de profondeur train/éval (F5.1) toujours non mesuré
Train = 35 writes (120f) ; l'éval du harness décode à ~49 writes après un A→B→A. Aucune vidéo
val-only 240f n'a été collectée. Coût : ~1 GPU-h + 35 GB. À faire AVANT de conclure au J7,
sinon `val/loss` peut s'améliorer pendant que la profondeur d'éval se dégrade.

### 5.4 Contrôles à budget apparié (F2.2) hors périmètre J3
`memdec20w` reste biaisé tant qu'un décodeur FRAIS n'est pas ré-entraîné sur les états de
CHAQUE écrivain. Le trainer sait déjà geler l'écrivain (`--lr_w 0`) : le contrôle coûte 2
entraînements de décodeur, à planifier au J4/J6.

### 5.5 Points de vigilance mesurés au step 0 (à surveiller dès les premiers 500 steps)
- `state/rowcap_frac = 0.20` **dès l'init** : 20% des lignes de l'état sont déjà au cap de norme
  avec les poids l1. Si ça monte vers 1 avec `writer/decay → 0.999`, c'est la pathologie
  « tricher court » de F3.2/F3.5 : condition d'arrêt pré-enregistrée.
- `mem/influence_decay@k` du writer l1 (baseline) : **plate et faible**, 0.075 / 0.064 / 0.074 /
  0.060 / 0.095 pour k = 1/2/5/10/20 avec **Δloss ≈ 0 (±0.004)** : neutraliser N'IMPORTE quel
  write ne change quasiment ni l'état ni la frame décodée. C'est le chiffre que 2.0 doit faire
  bouger ; c'est aussi la mesure la plus directe du diagnostic « l'écrivain l1 n'écrit presque
  rien d'exploitable ».
- `inst/acc` et la marge causale valent 0 au départ (cf. 5.1) : ne pas les lire comme un signal
  avant que 5.1 soit tranché.

### 5.6 Divers
- Disque projet : 222 GB libres après la banque (0.8 GB) et le cache h-oracle (14 GB, supprimable
  via `rm -rf mem2_runs/h_oracle/cache_p6.pt` une fois le gate acquis).
- Rien n'a été commité ; tout le code 2.0 est dans `reference/runs/mem2/` (gitignored comme le
  reste de `reference/`).

---

## VERDICTS ŒIL ADEN (2026-07-28, eyepack reference/runs/mem2/eyepack/)
1. h-oracle : « indiscernable » → gate F5.3a confirmé HUMAINEMENT, pas seulement métrique.
2. toy avant/après : « beaucoup mieux » → l'écriture stocke du contenu, confirmé à l'œil.
3. jumeaux : « je vois bien la différence pour les deux » → le test d'instance est bien posé.

## VERDICTS ŒIL ADEN 2 (2026-07-28, clips + composites 3-bras, eyepack20)
- Cheval 3-bras : confirme pâle SEULEMENT avec écrivain 2.0 (v1.2 et contrôle bruns).
- Clips : "les sujets reviennent bien" (y compris paires où l'ancre brute semblait fausse :
  le splice est un nudge, pas une copie) ; cut "bon et propre" paires 1-2 ; vélo récupéré
  "avec difficulté", artefacts mur de briques ; FOND : catégorie conservée mais pas
  l'instance ("bien une plage mais plus la même") -> axe rejoin, v2 composite.
- Hybride A->B au switch1 : prouvé identique v1.2/2.0 (substrat, arm-independent).
