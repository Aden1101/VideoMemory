# Plan local C : entraînement RSD + éval sans Goodhart (volets c+d du brief sink-virtuel)

Date : 2026-07-23. Auteur : agent plan (lecture seule du code, rien lancé).
Base : `reference/ultraplan_brief_sink_virtuel.md` (version 2026-07-23 avec l'extension
MULTI-SWITCH actée par Aden : cible = A->B->C->A->D..., pas seulement A->B->A), code au HEAD.
Tous les claims code ci-dessous ont été vérifiés dans les fichiers cités (fichier:lignes).

## 0. Chiffre de gating tombé (section 3 du brief, confirmé)

Brief (24 cellules finales) : oracle1 = +0.224 full / +0.241 crop60, signes 24+/0-,
IDENTIQUE à l'oracle 12 frames (+0.220/+0.240) ; oracle3 pareil. Vérifié en cours de
campagne sur `reference/runs/reid_probe_l1/results/summary.json` (12 cellules mi-parcours) :

| arm vs memzero | full | crop60 | signes | sign-p |
|---|---|---|---|---|
| oracle (12 frames A) | +0.215 | +0.235 | 11+/1-, 12+/0- | 6e-3 / 5e-4 |
| oracle3 (3 frames A) | +0.218 | +0.239 | 11+/1-, 12+/0- | 6e-3 / 5e-4 |
| **oracle1 (1 frame A)** | **+0.216** | **+0.235** | **12+/0-, 12+/0-** | **5e-4 / 5e-4** |

Lecture : le plancher de compression est AU SOL. **1 seule frame latente de KV-A (1560 tokens)
dans le dernier slot de la fenêtre récupère 100% du gain oracle.** L'ask de compression du
sink-virtuel (m tokens ~ une fraction de frame) est donc raisonnable ; le risque n°1 du brief
(il faudrait les 12 frames) est ÉLIMINÉ. Corollaire de design : viser m <= 1560 tokens/bloc,
commencer petit (m = 256). Corollaire multi-switch : ~1 frame d'info PAR SUJET => 3-4 sujets
simultanés ~ 3-4 frames d'équivalent-info dans l'état, minuscule (brief, section 2).

## 1. État du code vérifié (ce sur quoi le plan s'appuie)

- **Le rollout d'entraînement ne switch JAMAIS** : `l1_longlive.yaml` a `switch_mode:
  random_choice`, `switch_choices: [1000]` ; `trainer/distillation.py:779-796` garde le
  sentinel hors-plage quand tous les choix >= max_length ("deliberate no-switch") donc
  `_should_switch_prompt` (`model/streaming_training.py:216-245`) ne tire jamais.
- **Un seul switch supporté en training** : `state["has_switched"]` (bool), un seul
  `switch_frame_index` dans `switch_info`. Le retour B->A (`second_switch_frame_index`,
  `text_prompts_third`) n'existe QUE côté inference (`pipeline/switch_causal_inference.py:130-345`).
- **ORACLE_SPLICE (validé)** : `pipeline/switch_causal_inference.py:237-259` (env) ->
  `_recache_after_switch(frames_override=...)` (L56-128) : recache le KV de la fenêtre depuis
  la queue du A ORIGINAL aux positions courantes. `ORACLE_SPLICE_FRAMES` = splice partiel.
  Le `_recache_after_switch` du pipeline TRAINING (`pipeline/streaming_switch_training.py:304-382`)
  n'a PAS `frames_override` : à porter (petit diff, même signature que la version inference).
- **Machinerie twin e3 (validée)** : `model/streaming_training.py::_twin_snapshot` (L559-602) /
  `_twin_restore` (L604-622) : clone CPU de kv_cache1 + crossattn + flow + prompt caches, refs
  ttt state/ema/fifo, RNG cpu+cuda+python, has_switched. Complétude prouvée par
  `E3_TWIN_SELFTEST=1` (L713-749 : twin mem-ON == réel bitwise). Coût mesurable : le kv_cache
  fait 30 blocs x [1, 39x1560, 12, 128] bf16 x (k+v) ~ **11 GB de clones CPU** par snapshot
  (déjà payé par les runs e3/e4 : praticable, quelques secondes de copie).
- **BPTT fenêtré** : `trainer/distillation.py:1178-1277`. K_bptt ~ U{4..14} rank-synced,
  `grad_tail_W: 1`, seed-step tous les 4 gen-steps (chunk 0 en graphe, backward par chunk),
  snapshot/restore ttt autour de CHAQUE backward (L1869-1903, le recompute corrompt l'état),
  all-reduce manuel des grads mémoire FSDP-ignored (L1474-1477), clip mem séparé (L1388-1393),
  EMA trainable-only (EMA_FSDP), cosine lr optionnel (L1395-1408).
- **Isolation mémoire déjà câblée par le NOM "memory"** : groupe optimizer lr_mem
  (L414-422), exclusion LoRA (L1674), FSDP-ignored + broadcast init (L319-332), sauvegarde
  `generator_memory` (L811), chargements strict=False. => le décodeur état->KV du volet (b)
  DOIT être nommé `block.memory.kv_dec.*` pour hériter de TOUT ça gratuitement.
- **Timing réel mesuré** (tfevents du run l1, 2000 steps, 4 GPUs partagés) : médiane
  29.3 s/step, moyenne 55.7 s (p10 24 s, p90 176 s : la traîne = contention GPU), total
  31.0 h wall. C'est la référence budget.
- **Harness re-ID** : `reference/runs/reid_probe_l1/run_reid_probe.py` : 8 paires x 3 seeds,
  arms memon/memzero/memsnap/oracle/oracle3/oracle1/stock, même seed par cellule inter-arm,
  scoring DINOv2 full+crop60, stats appariées + sign-test exact, selftest intégré.
- **TwoTextDataset** (`utils/dataset.py:39-76`) : appariement A/B strictement 1-à-1 par ligne
  de `data_path` / `switch_prompt_path`.

## 2. Plan d'entraînement RSD

### 2.1 Curriculum A->B->A auto-généré

**Tirage par séquence** (rank-0 + broadcast, comme tous les tirages existants) :
- `p_revisit = 0.5` : la séquence est un revisit A->B->A ; sinon plain-long (DMD pur, aucune
  RSD) = régression do-no-harm intégrée au training, gratuite.
- Positions (frames latentes, multiples de 3 ; switch2 aligné sur une frontière de chunk
  21+18k pour que le splice teacher et le grad-tail tombent proprement) :
  - `switch1 ~ U{27, 36, 45, 54, 63}` : A établi (>= 1.7 s vidéo) avant l'interlude.
  - `gap = switch2 - switch1 ~ U{27, 45, 63}` avec la contrainte dure **gap >= 24** :
    local_attn_size 12 (fenêtre) + 12 (frames recachées au retour) => la fenêtre recachée au
    switch2 est 100% B et A est TOTALEMENT évincé du cache local (sinon la tâche est triviale
    et le gradient n'apprend rien : assert runtime, selftest #4).
  - `switch2 <= 147` : laisse >= 5 chunks de résurrection avant max_length.
- Le retour = prompt A (pas de 3e fichier de prompts ; `text_prompts_third=None` côté probe
  fait déjà pareil).
- Sink : garder `global_sink: true` + sink 3 comme l1 (le sink garde les 3 premières frames
  de A pendant B ; le probe montre que ça ne suffit PAS : memzero 0.66 vs oracle 0.906 ;
  ne pas "résoudre" la tâche en agrandissant le sink).

**Code à changer (résumé, détail au volet b)** : (i) `switch_choices` in-range + tirage du
second switch dans `_get_switch_frame_index` ou un nouveau `_get_revisit_schedule` ;
(ii) `has_switched: bool` -> compteur 2 états + `second_switch_frame_index` dans switch_info ;
(iii) porter `frames_override` dans le `_recache_after_switch` training ; (iv) stocker
`state["A_tail"]` = les 12 dernières frames latentes générées avant switch1 (détachées, elles
sont dans `output`/`previous_frames` au moment du switch) pour le splice teacher.

**Paires de prompts** : construire 2 fichiers (1-à-1 par ligne, format TwoTextDataset) :
- `rsd_A.txt` : sous-ensemble du corpus vidprom 248k filtré "sujet re-identifiable"
  (heuristique regex : groupe nominal sujet + >= 1 attribut discriminant couleur/vêtement/
  matière, ex. "a brown dog with a white chest", cible 10-20k lignes).
- `rsd_B.txt` : interludes tirés du corpus général (shuffle, mêmes lignes count).
- **EXCLUSION held-out obligatoire** : retirer (match exact + substring des syntagmes clés)
  les 8 paires du probe `run_reid_probe.py::PAIRS` ET les 8 paires fraîches d'éval finale
  (section 3.2). Selftest #0 : grep des 16 paires dans rsd_A/B.txt = 0 hit.

### 2.2 Teacher oracle : réutiliser le twin e3 ? VERDICT : OUI, sans réserve

Le teacher = LE MÊME modèle, même VRAM, roulé une fois de plus en no_grad sur les chunks de
résurrection avec le KV spliced. Protocole par step generator "revisit" :

1. Rollout normal (prefix no_grad du BPTT existant) : chunks A puis B jusqu'à switch2.
2. `_twin_snapshot()` (machinerie e3 réutilisée telle quelle : caches -> CPU, ttt refs, RNG).
3. **Pass TEACHER** (no_grad) : recache oracle (`frames_override = state["A_tail"]`,
   version training du splice) + `MEM_ZERO_ALL=1`, rouler R=2 chunks -> stocker
   `x0_teacher` (2 x [1,18,16,60,104] bf16 ~ 64 MB/chunk : négligeable).
4. `_twin_restore()` : état bit-exact pré-retour, RNG comprise.
5. **Pass STUDENT** (grad) : recache normal (queue de B) + mémoire ON + tokens virtuels,
   MÊMES bruits et MÊMES exit_flags de denoising (garantis par la restauration RNG : c'est
   exactement le pairing du twin e3, `generate_and_sync_list` re-tire les mêmes indices
   depuis la RNG rank-0 restaurée).
6. Loss RSD sur les chunks de résurrection (2.3).

Pourquoi c'est le bon choix :
- **Aucun second modèle en VRAM** (un second generator 1.3B + LoRA rank 256 coûterait
  ~10-15 GB/rank et un cauchemar FSDP) ; le teacher est le même graphe en no_grad.
- La complétude du snapshot est DÉJÀ prouvée bitwise par E3_TWIN_SELFTEST ; on la re-prouve
  dans le chemin RSD par le selftest TEACHER_NOOP (#2, section 4).
- Coût : ~11 GB de clones CPU aller-retour + R x 5 forwards no_grad (4 denoise + 1 write)
  par step generator revisit seulement (~10% des steps trainer) : +15-25% de temps moyen.
- **Teacher = memzero + splice** (l'arm "oracle" validée à 0.906), PAS memon + splice : la
  cible ne doit pas dépendre de la mémoire en cours d'apprentissage (cible mouvante, boucle
  d'auto-renforcement). MEM_EPISODIC n'est pas utilisé (memsnap = -0.03 n.s., voie morte).

**Alignement fenêtre BPTT** : sur séquence revisit, forcer une NOUVELLE séquence à ce step
generator (mécanisme seed-step/e3 existant, `trainer/distillation.py:1213-1215`) et choisir
switch2 = frontière du chunk K_bptt - W : le grad-tail couvre exactement les R chunks de
résurrection. `rsd_grad_tail_W: 2` sur ces steps (le graphe MSE est gratuit, pas de forward
supplémentaire ; fallback OOM : W=1 et un 2e step generator sur la même séquence, permis par
le schedule).

### 2.3 Loss exacte

```
L_total(chunk) = L_DMD(chunk)                                  # inchangée, tous les chunks
               + lambda_RSD * 1[chunk in résurrection] * L_RSD(chunk)

L_RSD(chunk)   = mean over new frames( (x0_student - sg(x0_teacher))^2 ) / RMS^2(x0_teacher)
```

- **MSE x0 dense** sur les latents des frames NOUVELLES du chunk (via gradient_mask), sg =
  stop-grad, normalisée par le RMS du teacher (échelle stable inter-prompts).
- **Chunks de résurrection : R = 2** premiers chunks post-switch2 (36 frames latentes ~ 9 s).
  Au-delà, l'oracle lui-même diverge du splice (signal dilué) ; le probe score resA sur
  [sw2+12, fin] donc R=2 couvre le début du segment jugé.
- **Pourquoi une cible dense et pas DMD seul** : le mur d'invariance (fresh_eyes B) : le
  critic fenêtré ne voit jamais A, le gradient DMD est invariant à l'identité du sujet. Le
  teacher oracle CONTIENT l'identité (re-ID 0.906 > stock 0.813) : la MSE la transporte.
  Mécanisme = self-distillation à contexte privilégié, publié (ARL, ISPA, OPSD-V) : à citer.
- **Garder la DMD sur les chunks de résurrection aussi** : ancre on-manifold, anti-blur de la
  MSE (la MSE seule régresse vers la moyenne des 4 pas de denoising). Coût marginal nul.
- **lambda_RSD** : départ 1.0 (MSE normalisée ~ O(0.1-1)) ; calibrer sur les 20 premiers
  gen-steps revisit pour `lambda*L_RSD / L_DMD` dans [0.3, 0.5] (loggé `rsd/lambda_ratio`).
- **Pas de warmup de lambda** : le gate zero-init du décodeur KV EST le warmup mécanique
  (injection strictement nulle au step 0, flag-OFF byte-identical).
- Pas de pondération horizon par frame en v1 (KISS ; option v2 à la AdaState).
- `cache_dropout_prob: 0.0` et `text_dropout_prob: 0.0` en v1 : le signal est maintenant
  supervisé (teacher), ces bruits data-free du c2/c3 ajoutent de la variance sans payeur.

### 2.4 Routage du gradient

- **Phase 1 (steps 0-2000)** : `lora_unfreeze_from_block: 30` => TOUTE la LoRA generator
  gelée (mécanisme existant, trainer L296-317 ; la LoRA critic reste entraînée, le flag ne la
  touche pas). Seuls `memory.*` (cellule TTT + décodeur kv_dec) s'entraînent, groupe
  `lr_mem`. Attribution propre : tout gain re-ID = la mémoire, et la LoRA ne peut pas fermer
  le gap RSD en contournant la mémoire (le Goodhart d'attribution est structurellement mort).
- **Phase 2 (conditionnelle)** : si `rsd/gap_r*` plafonne > 0 avec un causal_gain positif
  mais un re-ID intermédiaire qui stagne : repartir du ckpt phase 1,
  `lora_unfreeze_from_block: 0`, lr 1e-5, +1000 steps de co-train. Ne PAS commencer par ça.

### 2.5 Budget VRAM (4x48 GB) et débit

- Baseline l1 mesurée ~20-22 GB/rank (memory note, full unfreeze). Ajouts RSD :
  teacher no_grad = 0 graphe ; snapshot = CPU RAM (~11 GB), 0 VRAM ; W=2 grad-tail ~ +4-6 GB
  au pic ; m<=1560 tokens virtuels/bloc x 3 blocs = négligeable ; cibles teacher 128 MB.
  => confortable sous 45 GB/rank. Garde-fous : `mem_breakdown: false` (règle maison),
  `profile_vram` 5 steps avant launch (selftest #8), jamais 5 GPUs sans demander.
- Débit : l1 réel = 29.3 s médian / 55.7 s moyen par step (GPUs partagés). Surcoût RSD
  concentré sur ~10% des steps (generator x revisit) : +2 chunks teacher no_grad (+~10 s)
  + 2 copies snapshot (~+10-20 s). Estimation honnête : **+15-25% => 2000 steps ~ 36-40 h
  wall** dans les mêmes conditions de partage (l1 : 31 h). ~55-65 steps/h en moyenne.

### 2.6 Schedule

- `max_iters: 2000`, `log_iters: 250` (8 ckpts, `max_checkpoints: 3` + copie manuelle keep/).
- `ema_weight: 0.99`, `ema_start_step: 250` (inchangé ; l'éval charge use_ema:true).
- `lr_mem: 5e-5`, `max_grad_norm_mem: 1.0`, `weight_decay: 0.05` (hérités c3, ont tenu l1).
- `lr_decay_start: 800` -> cosine, `lr_mem_floor_frac: 0.05` (décale de 400 à 800 : la RSD
  a besoin de plus de temps plein-lr que le pur DMD, le gap est une vraie courbe de perte).
- `dfake_gen_update_ratio: 5`, `bptt_window: 14 / min 4`, `train_seed_every: 4` : inchangés.
- Go/no-go à 500 : si `rsd/gap_r1` n'a pas baissé de >= 20% ET `rsd/causal_gain` ~ 0 =>
  arrêter, autopsier (probablement décodeur sous-dimensionné ou attention mass nulle).

### 2.7 Métriques wandb (LIVE, toutes les courbes du verdict)

Nouvelles (préfixe `rsd/` et `memory/`) :
- `rsd/gap_r1`, `rsd/gap_r2` : L_RSD par position de chunk de résurrection. LA courbe.
- `rsd/gap_memzero_r1` (1 gen-step revisit sur 10, via un twin memzero 1-chunk réutilisant
  `_twin_snapshot`) et `rsd/causal_gain = gap_memzero - gap_memon` : si le gap baisse mais
  causal_gain ~ 0, c'est le reste du réseau qui ferme le gap, pas la mémoire (tripwire
  d'attribution ; quasi impossible en phase 1 LoRA gelée, mais la courbe est le juge).
- `rsd/lambda_ratio` : lambda*L_RSD / L_DMD (calibration visible).
- `memory/vtok_rms_b{5,16,27}` : RMS des k/v virtuels décodés par bloc.
- `memory/vtok_gate_b{i}` : gate du décodeur (doit décoller de 0, sans exploser).
- `memory/vtok_attn_b{i}` : masse d'attention softmax sur les tokens virtuels (proxy pas
  cher : 1 tête, queries de la dernière frame, log 1 step sur 25). Monter = lu ; exploser et
  ne jamais redescendre = wallpaper (croiser avec l'éval 3.4).
- Existants à garder ON : `memory/drift_b*`, `statenorm_ratio_b*` (+ `assert_memory_drift`),
  gate/gatew, generator/critic loss + grad norms, `time/per_iteration_time`.
- **Re-ID périodique OFFLINE** (jamais dans la boucle train) : à chaque ckpt (250), job sur
  GPU 0 : `run_reid_probe.py` 2 paires x 1 seed, arms memon/memzero (~40 min), résultat
  poussé en wandb (run séparé `rsd_eval`) : la courbe re-ID vs step est le juge intermédiaire.

## 3. Plan d'éval (sans artefact Goodhart)

### 3.1 Arm "memvirt" dans le harness existant

- `probe_memvirt.yaml` = copie de probe_l1.yaml pointant le ckpt RSD@2000 (use_ema true,
  sink 1, mêmes model_kwargs + les nouveaux flags sink-virtuel).
- Ajouter `"memvirt"` à `MAG_ARMS`/`ARM_ENV` (env = _OFF, comme memon ; le nom distinct sert
  les JSON). Le memzero apparié est celui DU MÊME ckpt (MEM_ZERO_ALL doit zéroer AUSSI les
  tokens virtuels : selftest #5).
- 8 paires x 3 seeds, stats appariées + sign-test exact, full + crop60.
- **Seuils pré-enregistrés (avant de lancer)** : succès = mean delta(memvirt - memzero)
  > +0.05 ET sign-p < 0.05 sur full ET crop60. Références absolues : memzero ~0.66,
  stock 0.81, oracle 0.906. Cible réaliste annoncée : combler >= 25% du gap oracle-memzero.

### 3.2 Held-out (paires jamais vues au training)

- Niveau 1 : les 8 paires du probe sont exclues du corpus training (2.1) : selftest #0.
- Niveau 2 : **8 paires FRAÎCHES** écrites au moment de l'éval finale SEULEMENT (jamais
  utilisées pour aucun tuning ni go/no-go intermédiaire), catégories équilibrées
  (2 personnes, 2 animaux, 2 objets rigides, 2 mixtes). Rapporter tuning-set et fresh-set
  SÉPARÉMENT dans le papier : c'est la seule preuve anti-surapprentissage-du-bench.

### 3.3 Régression do-no-harm sur dérive simple

- 3-way classique memvirt | memzero | stock LongLive (jamais on/off seul : règle maison),
  240f sans switch, harness `reference/runs/l1_longlive/eval*` (VBench SC + BC + Dynamic
  Degree PRIORITAIRES), mêmes prompts que les évals l1 précédentes pour comparabilité.
- Critère : memvirt >= memzero sur les 3 axes à bruit près (noise floor DINO connu 0.18 ;
  pour VBench, utiliser la dispersion historique des évals l1) ; DD ne doit PAS chuter.
- Plus œil humain côte-à-côte (SendUserFile plein-res, crf 20-24) : obligatoire avant tout
  verdict, VBench seul est aveugle (leçons e2/c2 : dissolution et flicker invisibles).

### 3.4 Anti-Goodhart spécifiques au sink-virtuel

Menace n°1 : le token virtuel "imprime" une image statique de A (wallpaper KV) : le re-ID
DINO monte, la vidéo est morte ou contaminée. Détections croisées :
- **a. DD segmenté** : Dynamic Degree calculé PAR SEGMENT (origA / B / resA). Drapeau rouge :
  DD(resA, memvirt) << DD(resA, memzero) alors que reid monte = résurrection par gel.
- **b. `self_resA`** : ajouter au scorer la cohérence intra-segment de resA (miroir du
  `self_origA` existant). ~ self_origA attendu ; -> 1.0 = segment figé.
- **c. `ctrl_B_origA`** (déjà loggé) : si les tokens virtuels fuient A PENDANT B, ce score
  monte vs memzero = tripwire de fuite (le teacher n'est spliced qu'au retour, le student ne
  doit injecter qu'au retour).
- **d. `memory/vtok_attn` au fil du rollout d'éval** : masse d'attention qui explose au
  retour puis ne redescend jamais = sur-injection permanente ; à croiser avec (a).
- **e. Œil humain** : côte-à-côte memvirt | memzero | stock | oracle sur 2-3 paires, regarder
  spécifiquement le segment resA (gel, ghosting de A sur B, texture plaquée).
- (Les dcfrac/ampdisp du canal résiduel ne s'appliquent plus ; leurs héritiers = a+d.)

### 3.5 Baselines : à battre / à citer, verdicts reproductibilité honnêtes

- **AdaState (2605.30349)** : le voisin mécanisme (sink synthétique data-free). Pas de code
  public constaté à l'audit préemption. Réimplémentation sur notre substrat POSSIBLE (état =
  1 frame latente co-dénoisée en position 0) mais 1-2 semaines + risque de sous-tuning
  (comparaison contestable dans les deux sens). Verdict : **citer + différencier**
  (TTT/écriture à l'éviction/résurrection SUJET vs état recopié chaque chunk) ; réimplémenter
  seulement sur exigence reviewer. Rebuttal anticipé "même sink, autre updater" : notre
  réponse = le delta MESURÉ sur le revisit A->B->A, qu'AdaState n'évalue pas.
- **MAG/MemGen (2512.18741)** : O(T/3), 390K vraies vidéos, loss de reconstruction =>
  **non reproductible data-free chez nous, ne pas essayer**. Citer + situer MAG-Bench
  (revisit CAMÉRA/scène) vs notre revisit IDENTITÉ SUJET en T2V pur.
- **WorldKV (2605.22718)** : training-free, stockage O(T) des KV réels + retrieval indexé
  caméra : pas d'index caméra en T2V texte => non applicable tel quel. MAIS notre **arm
  oracle EST l'upper bound WorldKV-parfait** (retrieval divin de la queue A, 0$ et déjà
  mesuré : 0.906 full / 0.875 crop60) : le présenter comme tel dans le papier.
- **Stock LongLive** (0.81/0.73) : la référence externe systématique du 3-way.
- Tableau final du papier : memzero, memvirt, memsnap (négatif audité), oracle/oracle1
  (upper bound + plancher de compression), stock : tout existe déjà dans le harness.

## 4. Selftests OBLIGATOIRES avant le premier launch (ordre d'exécution)

0. **Held-out exclusion** : grep des 8 paires probe (+ 8 fraîches si déjà écrites) dans
   rsd_A.txt/rsd_B.txt = 0 hit.
1. **FLAG-OFF BYTE-IDENTICAL** (règle maison n°1) : config avec décodeur sink-virtuel absent
   + curriculum OFF (`switch_choices: [1000]`, p_revisit 0) : 20 steps, pertes et latents
   bitwise identiques au HEAD l1 (hash des chunks). Idem : nouveau code chargé mais
   `virtual_sink` off => byte-identical.
2. **TEACHER_NOOP** (le twin du chemin RSD) : rollout revisit, pass teacher avec recache
   NORMAL (sans splice, mêmes env que le student) : x0_teacher == x0_student bitwise.
   Valide snapshot/restore + pairing RNG dans le nouveau chemin (équivalent E3_TWIN_SELFTEST).
3. **TEACHER-SPLICE == ORACLE PROBE** : sur pair0/seed0 en config probe (240f, sw 81/162),
   le chemin training-teacher (snapshot -> splice -> rollout no_grad -> restore) reproduit
   les latents de l'arm oracle du harness (même seed, même config => viser bitwise ; tout
   écart s'investigue AVANT launch, pas de tolérance).
4. **ÉVICTION RÉELLE** : assert runtime au switch2 : aucune frame de A dans la fenêtre
   locale recachée (gap >= 24 le garantit ; l'assert protège des changements futurs de
   local_attn_size). + MEM_EVICT_DEBUG=1 une fois pour tracer le fifo.
5. **MEM_ZERO_ALL zéroe AUSSI les tokens virtuels** : arm memzero : vtok_attn == 0
   exactement ET latents == config `memory_layers: []` à sink égal (60f). L'ablation doit
   être un vrai OFF (piège historique du wrapper : mémoire random si config incomplète).
6. **GRAD PROBE décodeur** : étendre le probe existant (trainer L1256-1267) : grads non
   nuls sur `memory.kv_dec.*` ET `memory.lrs/qkv/proj` aux steps 0/5/10 ; grads NULS sur la
   LoRA generator (phase 1 gelée).
7. **RANK-SYNC 2-GPU** : 20 steps sur 2 GPUs : aucun deadlock (tous les tirages
   revisit/switch1/switch2 broadcast src=0), même schedule loggé sur les 2 ranks.
8. **PROFILE_VRAM** : `profile_vram: 5` sur 4 GPUs, pic alloc < 45 GB/rank,
   `mem_breakdown: false`.
9. **Selftest harness éval** : `run_reid_probe.py --mode selftest` avec probe_memvirt.yaml
   (ttt écrit sur les bons blocs, memzero diverge de memvirt, arms oracle/memsnap vivants)
   avant l'éval finale.

Rappels d'exploitation : wandb obligatoire (jamais WANDB_MODE=disabled), 4 GPUs max sans
demander, ne jamais tuer un run sain pour ce plan (attendre une fenêtre), artefacts dans
reference/ uniquement, Aden commit lui-même.
