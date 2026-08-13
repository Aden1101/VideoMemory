# Rapport "fresh eyes A" : vérification du brief, carte littérature, propositions

Chercheur indépendant, 2026-07-22. Lecture seule du repo (aucun script lancé, aucun GPU touché).
Brief vérifié : `reference/brief_memoire_longlive.md`.

## §1. Vérification du brief contre le code

### 1.1 Claims confirmées (avec fichier:ligne)

| # | Claim du brief | Verdict | Preuve |
|---|---|---|---|
| 1 | Backbone Wan2.1-T2V-1.3B, 4 pas DMD | CONFIRMÉ | `reference/runs/l1_longlive/l1_longlive.yaml:130-139` (`denoising_step_list: [1000,750,500,250]`, `real_name/fake_name/generator_name: wan_models/Wan2.1-T2V-1.3B`) |
| 2 | LongLive = fenêtre locale 12 frames latentes + sink 3 frames | CONFIRMÉ | `l1_longlive.yaml:143-144` (`local_attn_size: 12`, `sink_size: 3`) ; masque d'attention `wan/modules/causal_model.py:969-972` ; le header du yaml (l.4-15) documente base.pt byte-identique à `longlive_base.pt` et le lora.pt LongLive drop-in (constat repris de leur selftest, non re-hashé par moi) |
| 3 | Cellule MLP SwiGLU à poids rapides, état de taille fixe | CONFIRMÉ | `wan/modules/video_memory.py:221-307` (`VideoMemoryLayer`; fast weights `w0/w1/w2` l.297-299 ; `swiglu_apply` l.134-138 ; état = (w0,w1,w2) répliqué par batch, `init_state` l.432-435) |
| 4 | Insérée dans 3 blocs [5,16,27] | CONFIRMÉ (config c3/l1) | `l1_longlive.yaml:145-148`. Nuance : `configs/ttt_memory.yaml:79` (config de base) n'a que `[5]` ; le brief décrit bien la config des runs c3/l1 |
| 5 | Lit : interroge l'état avec les tokens courants (apply-first) | CONFIRMÉ | `video_memory.py:464-468` (`o = swiglu_apply(q, w0, w1, w2)` avec l'état d'AVANT le chunk) ; en eviction-feed la query de lecture reste le chunk courant : `causal_model.py:632` et `video_memory.py:459-463` (docstring de `step`) |
| 6 | Injecte via porte apprise gated-AdaLN dans le flux résiduel | CONFIRMÉ | `video_memory.py:389-430` (`proj` -> shift/scale/gate l.399, `tanh` borné l.404-407, `out = gate*(carrier*(1+scale)+shift)` l.411) ; ajout au résiduel après self-attn : `causal_model.py:698` (`x = x + mem_out`) ; `gate_init: 0.1` `l1_longlive.yaml:149` |
| 7 | Écrit via Muon (Newton-Schulz) avec renorm vers l'init | CONFIRMÉ avec 2 précisions | `video_memory.py:170-210` (`weight_norm_update`), NS l.92-131. Précision (a) : `muon_update_steps=5` = 5 itérations Newton-Schulz d'orthogonalisation d'UNE écriture par chunk, pas 5 pas de gradient. Précision (b) : le défaut renorme à la norme PRÉ-update (l.196-198, invariante donc = norme d'init) ; le mode réellement utilisé (`use_decay: true`, yaml:153) est gated-delta : decay appris + CAP à la norme d'init (l.201-209, `_decay_and_norms` l.440-450) |
| 8 | Eviction-feed : on écrit le chunk évincé, pas le courant | CONFIRMÉ | Param `x_write` : `video_memory.py:455, 470-472`. FIFO au niveau modèle : `causal_model.py:1342-1385` (`window_chunks = ceil(local_attn_size/F)` l.1361, chunks du sink JAMAIS écrits l.1373-1375). Append côté bloc : `causal_model.py:628-641`. Nuance honnête (documentée dans le code) : c'est un délai FIFO par chunk, pas une éviction token-exacte du KV |
| 9 | Co-entraînement LoRA + mémoire par la loss DMD | CONFIRMÉ | Groupe d'optimiseur mémoire séparé : `trainer/distillation.py:413-426` (`lr_mem`) ; `lr: 1e-5` + `lr_mem: 5e-5` (`l1_longlive.yaml:64,71`) ; loss DMD eq.7 de 2311.18828 : `model/dmd.py:161-227` ; pur DMD au run l1 (tous les poids de garde-fous à 0, yaml:88-106) |
| 10 | Data-free : aucun dataset vidéo, signal = teacher DMD seul | CONFIRMÉ | `model/base.py:106-160` (`_run_generator` : bruit + backward simulation) ; `model/dmd.py:240` ("removes the need of any datasets") ; `load_raw_video: false` (yaml:34) |
| 11 | Teacher à contexte court, ne voit ni KV ni passé lointain | CONFIRMÉ | `real_score` bidirectionnel gelé : `model/base.py:36-37` (`is_causal=False`, `requires_grad_(False)`) ; la loss ne voit que `slice_last_frames: 27` latents + 1 frame ré-encodée : `model/base.py:162-182`, yaml:167 ; le teacher est appelé sans kv_cache ni ttt_cache : `model/dmd.py:106-118` |
| 12 | Read à chaque passe de débruitage, write sur la passe clean | CONFIRMÉ | `pipeline/streaming_training.py:290,331` (`ttt_update=False` sur les passes bruitées) ; l.359-373 (`ttt_update=True` sur la passe contexte clean) ; write entraîné dans le chunk (`_write_grad`, l.354-358 ; `train_write: true` yaml:74) |
| 13 | Pack de stabilisation (e) : cap amplitude + EMA + decay lr + clip | CONFIRMÉ | `mem_stream_scale: 0.5` (yaml:161 ; code `causal_model.py:659-670`) ; EMA LoRA 0.99 dès step 250 (yaml:45-46 ; `trainer/distillation.py:390-406`) ; cosine + floors (yaml:79-81 ; trainer:433-438) ; clip mem 1.0 (yaml:72 ; trainer:428-432, les grads mémoire sont FSDP-ignorés donc non clippés avant c3) ; cache-dropout annealé 0.3->0.1 (yaml:83-86 ; `pipeline/streaming_training.py:79-146`) |
| 14 | Hooks d'ablation propres | CONFIRMÉ | `MEM_ZERO_ALL` : `causal_model.py:652-658` ; `MEM_STREAM_SCALE` env : l.663-664 ; probe `MEM_REJECT_STREAM` : l.687-697 ; sink runtime réglable : `set_runtime_sink_size` l.930-936 |
| 15 | A->B->A supporté à l'inférence | CONFIRMÉ | `pipeline/switch_causal_inference.py:108-141` (`text_prompts_third`, `second_switch_frame_index`, défaut = retour au prompt A l.139-140) ; recache aux deux switches l.197-210. Note : c'est une extension INFÉRENCE ; l'entraînement (`dmd_switch`) ne fait toujours qu'UN switch, et le run l1 avec `switch_choices: [1000]` > `streaming_max_length: 300` (yaml:183-185) ne voit en pratique AUCUN switch |
| 16 | Éval 3-way + DINO | CONFIRMÉ | `eval*/memon.yaml` vs `memzero.yaml` (diff = output_folder uniquement, MEM_ZERO_ALL injecté par run_eval.sh) + `stock_longlive.yaml` (memory_layers [], lora LongLive) ; `metrics.py` : id_cos = cos(DINO(f0), DINO(f_last)), DINOv2-vitb14, noise floor 0.18 (metrics.py:32) |

### 1.2 Corrections au brief (par ordre d'importance)

**C1. MAJEURE : le probe re-ID n'est PAS "négatif", il est MIXTE (n=2), et le brief omet la paire positive.**
Le brief (§3.g) affirme "elle n'aide pas, elle nuit légèrement (delta ~ -0.06/-0.07)". Vérité des fichiers :

- pair0 (chien/voiture) : delta memon-memzero = **-0.0707** (full) / **-0.0560** (crop60) (`reference/runs/reid_probe/results/pair0_scores.json`, résumé `run_pair0.log`). Le mécanisme visuel du brief est bien réel : sur `results/reid_compare_pair0.png` et `eyecheck_memon_resA.png`, la carrosserie/pare-brise de la voiture de B est visible derrière le chien ressuscité en mem-ON, absente en mem-OFF. Cette partie du diagnostic (la mémoire traîne le contexte récent) est confirmée.
- pair1 (femme ciré jaune/pingouins) : delta = **+0.1511** (full) / **+0.3111** (crop60) (`results/pair1_scores.json`, `run_pair1.log`). La mémoire AIDE, fortement, sur cette paire : le retour de A est nettement plus fidèle avec mémoire (0.642 vs 0.331 en crop60), et self_origA est aussi plus haut en mem-ON (0.92 vs 0.82).

Conclusion honnête : sur n=2 paires, 1 négative modérée et 1 positive forte. La phrase du brief n'est pas supportée par les données du repo ; le bon verdict est "signal mixte, sous-échantillonné, avec un mécanisme d'échec identifié (contamination par B) ET un mécanisme de succès observé". C'est en fait une meilleure nouvelle que ce que dit le brief : la mémoire PEUT porter l'identité au revisit ; ce qui manque, c'est le contrôle de QUEL contenu est actif au retour de A.

**C2. Le probe re-ID a tourné sur le checkpoint c3 (substrat Anchor-Forcing), pas sur le run LongLive.**
`probe.yaml` : `lora_ckpt: .../c3_stable/keep/checkpoint_model_001000/model.pt` (et header "substrate copied from c3_stable/eval1000_sink1"). La chronologie du brief (f = re-base LongLive, puis g = probe) laisse croire que le probe teste le modèle LongLive-basé ; c'est faux. De plus 1 seul seed par paire (seed 42+pair, `run_reid_probe.py:149`).

**C3. Toutes les évals l1 (et le probe) comparent des bras à sink affamé (sink_size=1) contre un stock à sink plein (sink_size=3).**
`eval*/memon.yaml:11` et `memzero.yaml` : `sink_size: 1` ; `stock_longlive.yaml` : `sink_size: 3` ; `probe.yaml` : `sink_size: 1`. Le delta memon/memzero reste une ablation valide (même sink), mais (a) la claim f "net-négative sur l'identité" est établie SOUS sink affamé, pas dans la config LongLive nominale, et (b) l'écart au stock mélange deux causes (sink 1 vs 3 ET co-train). D'ailleurs memzero lui-même est bien sous stock (id_cos 0.58-0.69 vs 0.710) : une partie de la perte d'identité vient du co-entraînement/sink, pas de l'injection.

**C4. Claim f "ne converge jamais vers do-no-harm" : vraie dans l'ensemble, avec une exception dans le bruit.**
Chiffres l1 (id_cos memon vs memzero, `eval*/metrics_c1_step250.json`) : 750 : 0.589/0.621 ; 750ema : 0.628/0.685 ; 1500 : **0.603/0.585 (positif +0.019)** ; 1500ema : 0.590/0.644 ; 2000 : 0.569/0.690 ; 2000ema : 0.581/0.623. Tendance nette au négatif, mais pas monotone ; le point 1500 non-EMA est légèrement positif (sous le noise floor).

**C5. Claim e (c3 : do-no-harm + dynamique) : confirmée, et même un peu pessimiste.**
Sur c3, memon id_cos ≥ memzero à la plupart des checkpoints sink1 (ex. eval1000_sink1 : 0.672 vs 0.598 ; motion lag1/lag2 conservés). Le "delta ~0" du brief est prudent : plusieurs points sont positifs mais restent sous stock (0.743) et le delta apparié (~0.26-0.32) doit être lu contre le noise floor 0.18 : non concluant, d'accord avec le brief.

**C6. Précisions mineures.** (i) "Muon multi-pas" = 5 itérations NS par écriture, une écriture par chunk (voir tableau #7). (ii) "renorm vers l'init" : exact en mode use_decay (cap à la norme d'init) ; le mode legacy renorme à la norme pré-update. (iii) Le run l1 ne voit en pratique aucun switch pendant l'entraînement (C f. tableau #15) : la mémoire n'a JAMAIS été entraînée sur un scénario où elle sert (revisit), alors que le probe l'évalue là-dessus ; ce mismatch train/éval mérite d'être dit explicitement dans le brief.

## §2. Carte littérature (2024-2026)

Méthode : abstracts vérifiés par fetch pour les papiers marqués (v) ; les autres au niveau recherche/résumé. Le champ publie ~5-10 papiers/mois sur exactement ce problème : tout trou identifié ici a une demi-vie courte.

### 2.1 Déjà fait / préempté

**Mémoire apprise entraînée par reconstruction sur vraie vidéo (la voie "qui marche") : PRÉEMPTÉ, ne pas refaire.**
- (v) **VideoSSM** (2512.04519) : mémoire SSM globale + fenêtre locale, distillation Self-Forcing, même substrat que nous, SOTA consistance/motion à l'échelle minute.
- (v) **MemGen / "Memorize-and-Generate"** (2512.18741) : modèle mémoire séparé qui compresse l'historique en KV compact + **MAG-Bench** (bench de rétention mémoire). L'ÉVAL revisit est donc déjà benchmarkée publiquement.
- (v) **Memento** (2606.14667) : "reconstruct to remember", mémoire dual-query (identité vs continuation), reconstruction du sujet DEPUIS la mémoire comme objectif explicite, pipeline de données ciné dédié. C'est la version supervisée-données de l'idée "la mémoire doit pouvoir régénérer le sujet". SOTA consistance sujet long terme.
- (v) **Memorize When Needed** (2604.18215) : branche mémoire découplée + gating caméra pour les revisits spatiaux.

**Le mécanisme TTT/fast-weight lui-même : PRÉEMPTÉ comme contribution.**
TTT-Video (2504.05298), LaCT (2505.23884), Titans (2501.00663, écriture par surprise), gated-delta (déjà dans notre code), Fast Spatial Memory / elastic TTT (2604.07350), tttLRM (2602.20160). Une "nouvelle règle d'écriture" seule n'est plus publiable (confirmé aussi par votre synthèse interne).

**Compression du contexte évincé + (self-)distillation : LARGEMENT PRÉEMPTÉ.**
- AHN (2510.07318, v au niveau résumé) : fenêtre KV lossless + module récurrent qui absorbe les KV évincés, entraîné par self-distillation avec backbone gelé (LLM). Notre eviction-feed en est déjà l'analogue vidéo (le code le cite).
- (v) **ARL2 "Attend Locally, Remember Linearly"** (2605.16579) : attention linéaire comme mémoire inter-frame absorbant les tokens évincés, conversion par distillation 2 étages, pour la vidéo AR. Aussi : Long-Horizon Streaming via Hybrid Attention distillation (2604.10103), Context Forcing (2602.06028).

**Gestion KV training-free du revisit/interactif : TRÈS ENCOMBRÉ.**
- (v) **Echo-Forcing** (2605.16003) : ancres stables + historique compressé + Scene Recall Frames + decay différentiel, TRAINING-FREE, gère explicitement le "long-range scene recall" (notre scénario A->B->A) en espace KV.
- (v) TetherCache (2606.13035) : recall gated + édition statistique des tokens rappelés, training-free sur Self-Forcing. (v) AdaState (2605.30349) : ancre latente auto-évolutive jamais rendue. SWIFT (2605.09442), FLEX (2602.14027), (v) IAMFlow (2605.18733) : mémoire d'entités LLM/VLM training-free + NarraStream-Bench (l'identité narrative est benchmarkée là aussi).
- Rétrieval : **LongLive-RAG** (latents générés = mémoire recherchable, bat LongLive en subject/background consistency sur 30/60/120s), **MemFlow** (2512.14699, Kling : retrieval KV par pertinence narrative, temps réel). Écosystème LongLive : LongLive-2.0, ShotStream, Stream-T1, KVPO.

**Récompense sur self-rollout : partiellement préempté.**
(v au niveau résumé) Identity-Preserving I2V via Reward-Guided Optimization (2510.14255, CVPR26) : reward facial calculé sur des rollouts type inférence, optimise directement la consistance d'identité longue ; MAIS c'est de l'I2V avec image de référence FOURNIE, sans module mémoire. Reward-Forcing (2601.16933) : feedback de reward pour l'AR vidéo, data-free, sans mémoire ni identité.

**Mondes/spatial** : WorldMem (2504.12369), Long-term Spatial Memory (2506.05284), Context-as-Memory (SIGGRAPH Asia 25), RELIC (2512.04040), DecMem (2605.31336), Geometry-Aware Implicit Memory (2606.02436) : revisit SPATIAL (caméra/pose), pas identité de sujet sous switch de prompt ; substrat différent.

### 2.2 Où il reste un trou publiable (pour NOTRE substrat imposé)

1. **Personne ne donne un signal d'identité DATA-FREE à une mémoire apprise.** Tous les systèmes qui apprennent QUOI retenir le font par reconstruction de vraie vidéo (VideoSSM, MemGen, Memento) ou avec une référence externe fournie (reward I2V). Utiliser le propre passé généré du modèle comme oracle d'identité au revisit (self-oracle) n'est pris par personne, et c'est exactement le chaînon qui manque à notre diagnostic ("rien ne dit à la mémoire quoi retenir").
2. **Personne ne gère l'état fast-weight de façon ÉPISODIQUE.** Snapshot/restauration/mélange d'états de poids rapides aux frontières de scène, indexés par prompt : inexistant en génération vidéo. L'analogue en espace KV existe (Echo-Forcing, Anchor-Forcing) et devra être cité et comparé, mais l'espace poids-rapides a un argument propre : coût O(1) indépendant de l'histoire, pas de recache, et notre autopsie (la mémoire stocke la récence -> traîne B) fournit la motivation mécaniste que les papiers KV n'ont pas.
3. **Le couplage des deux** (mémoire épisodique + objectif de résurrection self-oracle) n'existe nulle part et répond point par point à notre échec documenté. L'éval revisit seule n'est PLUS une contribution (MAG-Bench, NarraStream-Bench existent) : il faut mécanisme + objectif, avec ces benchs comme éval externe.
4. Trou secondaire : twin-distillation data-free du cache (AHN-style mais teacher = soi-même à cache complet sur self-rollouts, pas de données) : nouveauté moyenne (AHN/ARL2 proches), mais c'est le seul objectif DENSE dont on soit sûr qu'il donne un "job" à la mémoire par construction.

## §3. Mes propositions, classées (faisabilité x nouveauté x chance de marcher)

Protocole de mesure commun (contre les artefacts Goodhart, réutilise l'infra existante) :
probe re-ID étendu à **>=20 paires x 3 seeds**, sur le ckpt l1 ET c3, aux DEUX sinks (1 et 3) ; stats appariées (sign test + bootstrap du delta, pré-enregistrer le seuil) ; 3-way mem-ON | mem-OFF | stock ; DINO sujet crop60 (+ crop par tracker léger si dispo) ; tripwires : lag1 (gel), lag2 vs lag1 (flicker), dbg_dc_frac (biais DC), DD VBench en contrôle pas en headline ; côte-à-côte pour l'oeil d'Aden ; MAG-Bench (MemGen) comme éval externe si récupérable.

### Rang 1 : Mémoire épisodique fast-weight ("Episodic TTT") : snapshot/restore de l'état aux frontières de prompt

- **Mécanisme.** À chaque switch de prompt : snapshot de l'état (w0,w1,w2) de chaque bloc mémoire (+ flush du FIFO), indexé par l'embedding du prompt sortant. Au switch suivant, si le nouveau prompt matche un état stocké (cos des embeddings poolés > seuil) : restaurer cet état, ou le mélanger (w_actif = a*w_A + (1-a)*w_courant, a fixe 0.5-1.0 d'abord, appris ensuite). Généralisation : mini-banque LRU {prompt_emb -> état} pour N scènes. Variante d'écriture : pendant la scène B, continuer à écrire dans l'état de B mais jamais dans celui de A (ségrégation par scène).
- **Pourquoi ça donne un travail que le data-free actuel ne donne pas.** On ne change PAS ce que la mémoire apprend à stocker (la récence, on l'a prouvé) ; on change QUEL état est actif. Au retour de A, "la récence de A" EST l'identité de A : la pathologie observée (pair0 : la voiture de B traînée dans A) devient la feature. Aucun signal d'entraînement nouveau n'est requis pour le probe : c'est de la gestion d'état pure.
- **Coût.** $0 : ~30-50 lignes dans `switch_causal_inference.py` (les hooks existent : `ttt_cache[bi].state` est un objet accessible, `_recache_after_switch` marque déjà les frontières), inférence seule, GPU 0 pendant que le train tourne. Co-train optionnel ensuite (le read apprend à exploiter un état restauré) : nécessite les schedules A->B->A à l'entraînement (câbler le 2e switch dans dmd_switch).
- **Nouveauté.** Gestion épisodique d'états fast-weight en génération vidéo : pas vu (vérifié §2). À positionner contre Echo-Forcing (même idée en espace KV, training-free) : notre argument = état O(1), pas de recache, compatible streaming infini, et on documente le mode d'échec précis que ça corrige.
- **Chance de marcher : élevée.** La preuve d'existence est DANS le repo : pair1 montre +0.15/+0.31 quand le contenu actif est le bon ; pair0 échoue par contamination B ; restaurer l'état de A supprime mécaniquement la contamination. Prédiction falsifiable : pair0 passe de -0.07 à >=0.
- **Mesure.** Protocole commun, 4 bras : memzero | memon | memon+restore | memon+blend. Échec si le delta médian sur >=20 paires n'est pas >0 avec sign test p<0.05.

### Rang 2 : Objectif de résurrection self-oracle (le terme d'identité data-free qui manque)

- **Mécanisme.** Pendant le co-train, avec proba p (~0.5), la séquence devient A->B->A (2 switches, à câbler ; l'inférence l'a déjà). On décode 2-4 frames du segment origA (DÉTACHÉES : c'est l'oracle) et 2-4 frames du segment resA (avec grad) via l'infra `id_decode_frames` existante ; loss aux = 1 - cos(DINO(resA), DINO(origA)) (+ variante latente L2 pour éviter le décodage). **Gradient routé UNIQUEMENT vers les paramètres mémoire** (stop-grad sur la LoRA pour ce terme) : la mémoire est le seul canal causal reliant origA (écrit puis évincé pendant B) à resA, donc le terme lui apprend littéralement QUOI retenir (l'identité de A) et quoi ne pas traîner (B). DMD standard partout, inchangé.
- **Pourquoi ça marche là où le data-free échouait.** Le diagnostic §4 du brief dit : le teacher ne compare jamais à une instance de référence. Ici la référence EXISTE et elle est gratuite : c'est le propre passé généré du modèle, self-consistant par construction (self_origA ~0.8-0.9 mesuré). On n'introduit aucune donnée : on convertit la self-consistance du rollout en cible d'identité.
- **Anti-Goodhart.** (a) La loss ne compare jamais des frames adjacentes -> le gel n'est pas récompensé (et le DMD le punit) ; (b) borne naturelle = self_origA (on ne demande pas mieux que la self-consistance du modèle) ; (c) routage mémoire-only : le backbone ne peut pas "tricher globalement" (c'est la LoRA qui avait appris le do-no-harm inerte) ; (d) tripwires lag1/lag2 au monitoring, arrêt si DD s'effondre. Risque résiduel réel : la mémoire apprend juste à couper son injection pendant B (ce qui serait... un comportement correct, et mesurable via dbg_gate_mean par segment).
- **Coût.** 4 GPUs, run ~2000 steps type c3/l1 ; surcoût = décodage VAE de quelques frames x2 par rollout supervisé (infra déjà écrite) + câblage du 2e switch dans le trainer. Data-free strict (superviseur = DINO gelé, pas de vidéo).
- **Nouveauté.** Identité par self-oracle data-free pour une mémoire apprise : non pris (§2.2.1). Se différencie proprement de Memento (reconstruction sur vraies données ciné), du reward-I2V (référence externe fournie), de Memorize-When-Needed (boucles pseudo sur observations réelles). Le claim de papier : "on peut enseigner l'identité à une mémoire sans aucune donnée vidéo, en prenant le passé généré du modèle comme sa propre vérité terrain".
- **Chance de marcher : moyenne-haute.** Le signal est clairsemé (1 terme par rollout) et DINO est bruité (d'où >=20 paires à l'éval et EMA au train), mais le gradient a un chemin causal unique et court vers la mémoire : c'est structurellement l'anti-e2/e3 (pas de proxy fréquentiel à gamer).
- **Mesure.** Protocole commun ; le KPI est le delta re-ID apparié memon vs memzero SUR LE MÊME ckpt, plus la comparaison au rang 1 (episodic sans train) : si le train n'apporte rien de plus que le restore gratuit, on le dit (audited negative propre).

### Rang 3 : Twin-distillation du cache affamé (le seul objectif DENSE garanti d'avoir un job)

- **Mécanisme.** Chaque rollout de train tourne en jumeau : bras teacher = le même modèle, cache complet (sink 3 + fenêtre 12), mémoire coupée (MEM_ZERO_ALL, infra twin déjà validée bit-exacte) ; bras student = cache affamé (sink 0-1 via le cache-dropout existant) + mémoire active. Loss = cos/L2 dense entre les latents prédits par chunk du student et ceux du teacher (détachés), + DMD. La mémoire est par construction le SEUL canal qui peut fournir au student l'info que son cache n'a plus : plus de "no job".
- **Pourquoi.** Répond frontalement au diagnostic c ("pas de travail marginal") : on crée le travail par ablation contrôlée du cache, et le target est dense (chaque chunk), dynamique (un teacher qui bouge ne récompense jamais le gel) et data-free (self-rollouts).
- **Coût.** Le plus cher : ~2x forward par step (sous-échantillonner les chunks supervisés pour tenir 4 GPUs). Toujours zéro donnée.
- **Nouveauté : moyenne, à défendre.** AHN (LLM, self-distillation des KV évincés) et ARL2/2604.10103 (vidéo, distillation d'attention hybride, sur données) sont proches ; l'angle restant = twin data-free sur self-rollouts DMD + fast-weight TTT + démonstration que ça débloque le revisit (éval au sink plein ensuite : la mémoire doit apporter du hors-fenêtre, pas remplacer le sink).
- **Chance.** Haute pour "la mémoire apprend enfin un contenu utile" (le signal est dense et impossible à gamer par gel/flicker) ; moyenne pour battre LongLive en config nominale (le gain n'apparaît que là où la fenêtre+sink ne suffisent plus : >240s, multi-scènes, revisit).

### Rang 4 (support) : gate de lecture contextuel appris

`gate_extra: ["agreement", "frames_since_switch"]` est DÉJÀ câblé zero-init (`video_memory.py:315-355`, signaux construits dans `causal_model.py:606-627`, frames_since_switch se remet à 0 au switch). Co-entraîné avec le rang 2, il donne à la mémoire le moyen d'apprendre "injecter fort juste après un switch-back, s'abstenir sinon", ce qui est exactement la politique optimale vu pair0/pair1. Quasi gratuit, pas un papier seul.

### Rang 5 (support, nouveauté faible) : écriture résiduelle / surprise-scaled

Écrire seulement le complément du prédictible (h_evince - f_W(k_evince) est déjà la forme du gradient TTT ; renforcer via lr par token modulé par la surprise). Recouvre Titans et vos fix-ideas internes : utile en ablation, pas en contribution.

### L'histoire de papier que je défendrais

Rang 1 + Rang 2 (+ Rang 4) = un seul papier cohérent : **"Episodic fast-weight memory with a self-oracle resurrection objective : data-free identity for interactive long video"**. Le rang 1 fournit un résultat garanti-bon-marché dès la semaine 1 (probe $0, prédiction falsifiable sur pair0), le rang 2 fournit la contribution d'apprentissage, la triade d'échecs déjà auditée (gel/flicker/inerte + "la mémoire stocke la récence") devient la section d'analyse qui motive tout. Plan B si tout est négatif : le papier metric-integrity/audited-negative déjà envisagé en interne, renforcé par le protocole appariés-20-paires.

### Ce que je ne ferais pas

- Refaire une mémoire par reconstruction sur vraie vidéo (VideoSSM/MemGen/Memento : mort en nouveauté, dit aussi par le brief).
- Une nouvelle règle d'écriture comme contribution centrale (Titans et cie).
- Du KV-management training-free de plus (Echo-Forcing/TetherCache/LongLive-RAG saturent la niche, et ce n'est pas le substrat imposé).
- Conclure quoi que ce soit de plus sur le probe actuel avant de l'avoir ré-échantillonné : n=2 avec signes opposés, 1 seed, mauvais ckpt (c3, pas l1), sink 1. C'est la première dépense GPU utile, avant tout entraînement.
