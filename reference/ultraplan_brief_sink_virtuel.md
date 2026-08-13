# Brief ultraplan : mémoire "sink-virtuel" pour la résurrection de sujet (A→B→A) sur LongLive

## 0. Ce qu'on attend de l'ultraplan

Un plan d'exécution COMPLET et ADVERSARIAL pour l'idée ci-dessous : (a) design review à charge (modes de défaillance, alternatives, ce qui casse), (b) plan d'implémentation fichier-par-fichier avec les pièges connus du codebase, (c) plan d'entraînement (curriculum, VRAM 4 GPUs max, selftests AVANT launch), (d) plan d'éval sans artefact Goodhart, (e) liste des bugs probables à tester en priorité. Ne pas re-débattre "faut-il une mémoire" : la décision est prise, l'objectif est un papier POSITIF (une mémoire qui marche), pas un négatif audité.

## 1. Contexte en 5 lignes

Backbone Wan2.1-1.3B causal streaming distillé DMD 4 pas (Self-Forcing→LongLive), baseline officielle = LongLive (attention locale 12 frames + sink 3). Notre mémoire actuelle : MLP SwiGLU fast-weight (TTT, écriture Muon multi-pas, eviction-feed FIFO) greffée blocs [5,16,27], readout injecté dans le flux RÉSIDUEL via gated-AdaLN, co-entraînée data-free par DMD (+LoRA). Résultat définitif après ~10 runs + campagne statistique : l'injection résiduelle est inerte ou nuisible, JAMAIS utile ; sur le revisit A→B→A (24 cellules appariées, l1@2000-EMA), delta mémoire = nul (mean −0.02, sign-p 0.54). Diagnostic à 3 preuves : l'identité vit dans le canal KV (sink stock : re-ID 0.81 vs nous 0.67 ; toutes les dégénérescences passées = canal résiduel ; AdaState prouve qu'un sink synthétique appris marche en data-free).

## 2. L'idée à planifier : mémoire sink-virtuel + entraînement RSD

**Mécanisme** : garder la cellule fast-weight (état O(1), écriture TTT sur le chunk évincé), remplacer le readout résiduel par un DÉCODEUR d'état → m tokens k/v virtuels écrits dans le cache KV à côté du sink (par bloc mémoire). Le backbone les lit par attention softmax standard = l'interface qu'il maîtrise (il a été entraîné avec un sink). Gate/zero-init obligatoire sur les tokens virtuels (un k/v synthétique mal entraîné empoisonne l'attention globalement, rayon d'explosion >> résiduel).

**Entraînement (RSD, data-free)** : curriculum A→B→A auto-généré (l'entraînement actuel ne voit JAMAIS de switch, switch_choices [1000] : à corriger). Teacher = le MÊME modèle avec ORACLE_SPLICE (recache KV depuis la queue du A original, hook déjà implémenté et validé) ; student = cache normal + mémoire ; loss dense student-vs-oracle (x0/latents) sur les chunks de résurrection, gradient routé vers mémoire+décodeur. Le mécanisme "self-distillation à contexte privilégié" est PUBLIÉ (ARL, ISPA, OPSD-V : à citer, pas revendiquer) ; la capacité "résurrection d'identité SUJET, état O(1), data-free" est libre.

**Composants optionnels à arbitrer dans le plan** : gestion épisodique de l'état (MEM_EPISODIC snapshot/restore au switch, hook implémenté) ; gate de phase (frames_since_switch déjà câblé dans gate_extra) ; écriture à deux clés (moduler les clés par prompt-embedding).

**Extension MULTI-SWITCH (décision Aden 2026-07-23, à intégrer aux plans)** : la cible n'est pas que A→B→A mais le multi-switch interactif (A→B→C→A→D..., l'usage natif de LongLive). Conséquences : (a) le curriculum d'entraînement mélange des schedules de revisit variés (single + multi, positions aléatoires) ; (b) l'écriture à deux clés (clés modulées par prompt-embedding) passe de optionnel à probablement nécessaire (interférence entre traces : écrire B/C/D ne doit pas écraser A) ; (c) l'éval ajoute la courbe re-ID vs nombre de segments intercalés = L'ablation décisive contre un état-récence type AdaState (1 frame = dernier contexte seulement) et l'argument O(1) vs les banques retrieval O(T) ; (d) le decay doit tolérer des interludes de plusieurs minutes. Capacité plausible : plancher = 1 frame/sujet, donc 3-4 sujets ≈ 3-4 frames d'info dans l'état, minuscule.

## 3. Les 3 chiffres de gating (À REMPLIR avant de lancer l'ultraplan)

- **oracle − memzero = +0.220 (full) / +0.240 (crop60), signes 23+/1− et 24+/0−, sign-p < 1e-5.** Oracle absolu = 0.906, AU-DESSUS de stock (0.813). Le signal à distiller existe, il est énorme, et il est universel (les 8 paires positives). Détail : summary.json, 24 cellules appariées.
- **memsnap − memzero = −0.031 (full) / −0.045 (crop60), n.s.** La gestion épisodique de l'état résiduel n'apporte RIEN : le canal résiduel est mort même avec le bon contenu au bon moment → la voie "RSD sur l'état résiduel" est éliminée, le sink-virtuel est LA branche.
- **plancher de compression = 1 FRAME.** oracle1 (une seule frame latente de A dans la fenêtre recachée) = +0.224 (full) / +0.241 (crop60), signes 24+/0−, IDENTIQUE à l'oracle 12 frames (+0.220/+0.240) ; oracle3 pareil (+0.223/+0.245). L'information nécessaire à la résurrection tient dans UNE frame latente de KV (1560 tokens/bloc) : l'ask de compression du sink virtuel est minimal, m ≈ 1 frame voire moins (à ablater). Meilleur scénario possible.

Décision actée par ces chiffres : **branche sink-virtuel confirmée** (oracle fort + memsnap nul). L'ultraplan planifie CE mécanisme.

## 4. Points d'entrée code (vérifier chaque claim, ne rien croire sur parole)

- `wan/modules/causal_model.py` : attention + cache KV (`sink_tokens`, `_sink_runtime`, rope_enhance et son chemin sink séparé `grid_sizes_sink`, masques, fenêtre locale) : c'est LÀ que les tokens virtuels s'insèrent ; attention au chemin sink=0+rope_enhance (fix récent), au gradient checkpointing (entrées explicites obligatoires, cf. mécanisme ttt_state/mem_ema_prev), à MEM_ZERO_ALL (l'ablation doit zéroer AUSSI les tokens virtuels).
- `wan/modules/video_memory.py` : la cellule (état w0/w1/w2, step(), x_write eviction-feed, use_decay) ; le décodeur d'état→k/v est à créer ici.
- `pipeline/switch_causal_inference.py` : hooks MEM_EPISODIC + ORACLE_SPLICE (validés par selftest), _recache_after_switch (NB : tourne injection active = fuite de B connue), driver A→B→A (second_switch_frame_index).
- `pipeline/streaming_training.py` + `trainer/distillation.py` : rollout self-forcing, cache-dropout, BPTT fenêtré, EMA trainable-only, all-reduce manuel des grads mémoire (FSDP-ignored), snapshot/restore ttt autour du backward (recompute corrompt l'état). Le curriculum switch d'entraînement s'insère ici (`switch_choices`, `dmd_switch`).
- Harness d'éval : `reference/runs/reid_probe_l1/` (8 paires × 3 seeds, arms memon/memzero/memsnap/oracle/stock, stats appariées + sign-test). VBench 3-way : `reference/runs/l1_longlive/eval*/`.
- Pièges historiques à ne PAS réintroduire : switch/interactive inference qui bypassait la mémoire ; use_ema silencieux ; mem_breakdown VRAM ; id_loss Goodhart (gel/flicker) ; renorm-après-soustraction (dissolution e2).

## 5. Références à lire

- `reference/virtual_sink_preemption_check.md` : préemption sink-virtuel. Voisins = **AdaState 2605.30349** (sink synthétique data-free, MAIS 1 frame dénoisée, pas de TTT/éviction/revisit : baseline n°1 à battre + risque rebuttal "même sink autre updater"), **MAG/MemGen 2512.18741** (KV synthétiques + bench revisit caméra, O(T), vraie vidéo), **WorldKV 2605.22718** (réinsertion KV réel O(T) training-free). Benchs existants : MAG-Bench, NarraStream-Bench (revisit caméra : notre delta = identité SUJET).
- `reference/arl2_preemption_check.md` : ARL/ISPA/OPSD-V (mécanisme distillation pris).
- `reference/fresh_eyes_A.md`, `reference/fresh_eyes_B.md` : audits du brief précédent + idées (dont le mur d'invariance : le windowed-DMD seul ne peut pas apprendre l'identité, d'où la loss RSD).
- `reference/discussions/2026-07-22.md` : journal du jour (campagnes, résultats nuls, décisions).

## 6. Contraintes

4 GPUs max sans demander (jamais 5+) ; GPUs partagés avec d'autres users (prévoir OOM/relance, fenêtres volatiles) ; wandb obligatoire sur tout entraînement ; selftest twin/byte-identical AVANT tout launch ; tout artefact généré dans reference/ (gitignoré) ; Aden commit lui-même ; data-free strict (aucune vraie vidéo) ; éval finale = 3-way + œil humain sur côte-à-côte, jamais VBench seul.
