# Programme mémoire vidéo longue: audit ULTRA code+littérature (master doc), 2026-06-28

13 agents (4 code + 5 littérature + 3 idées + synthèse), run wf_46d4d81e-bfe, ~1.1M tokens.

## TL;DR (3-5 lignes)

Tu as greffé une mémoire fast-weight (TTT/Scal3R, `VideoMemoryLayer`) dans Anchor-Forcing (Wan2.1-1.3B, DMD 4-step) pour corriger la dérive d'identité long-format, et le verdict mesuré est qu'elle DÉGRADE l'identité à toutes les longueurs. La littérature 2025-2026 a tranché trois choses qui tuent le cadrage initial : (1) pour la dérive avant et le revisit géométrique, c'est retrieval + sink sur le vrai KV qui gagne, le fast-weight + gate additif est la pire combinaison documentée (arXiv:2512.06983) ; (2) ta nouveauté annoncée "entraîner le write par self-rollout" est déjà préemptée par VideoSSM (arXiv:2512.04519) ; (3) ta métrique (VBench subject_consistency) est aveugle à la dérive lente et surnote les frames dégradées. La cause racine prouvée et consensuelle : DMD n'a aucun signal d'identité et l'horizon supervisé (chunk <=21f) <= fenêtre KV (local 12 + sink 3), donc AUCUN gradient ne tombe sur le long-range que la mémoire doit corriger. Le pivot défendable : redéfinir la cible sur la ré-identification de sujet (A->B->A), seule niche ouverte, avec une métrique réelle, et fournir l'objectif manquant (anchor-REPA + history-loss) ; valider d'abord contre des baselines retrieval/EMA-sink bon marché, sur un backbone dé-confondu.

## Ce que tu as construit et voulais faire (carte du code + intention)

**L'intention.** Soigner la dérive d'identité long-format dans un générateur causal distillé et figé (Wan2.1-T2V-1.3B, DMD 4-step, streaming AR) en greffant une mémoire fast-weight que le cache KV roulant (local 12 + sink 3) ne peut pas fournir. La nouveauté visée : entraîner le WRITE par self-rollout causal (BPTT tronquée sur fenêtre roulante), distinct du TTT teacher-forcé (TTT-Video, LaCT) ou du self-rollout sans write entraîné.

**Le module.** `wan/modules/video_memory.py` : `VideoMemoryLayer` (L199-411), portage causal du `FastWeightGluMLP` de Scal3R. State = fast weights (w0,w1,w2). `step()` (L366-395) est APPLY-FIRST : lecture `swiglu_apply(q,w0,w1,w2)` (L374) sur l'état d'avant le chunk, puis si `do_update` write via `swiglu_fast_weight_gradient` (montée de `<f_W(k),v>`, Hebbian, L129-155) + `weight_norm_update` (Muon Newton-Schulz steps=5 puis renorm-vers-norme-init de chaque ligne, L158-188). Sortie `_project_output` (L344-355) : `o_norm` (RMSNorm) PUIS AdaLN `gate*(a_norm(x)*(1+scale)+shift)`, gate = slice brute non bornée. Donc lecture DOUBLE-normalisée + gate non borné = injection à RMS fixe découplée de la pertinence.

**Le greffage.** `wan/modules/causal_model.py:556-604` : branche parallèle à la self-attn, `x = x + (mem_out.detach() if ttt_update else mem_out)` (L601). `TTTBlockCache` (L44-57) est un objet `__slots__` (pas un dict) pour survivre au pytree FSDP/checkpoint. `reset_parameters()` re-appelé après le xavier global (L795-799) : c'était le bug gate-clobber.

**L'entraînement.** Seuls les params `*memory*` s'entraînent : groupe AdamW dédié `lr_mem=1e-4` (`distillation.py:367-374`), backbone `lr=0` (figé). `train_write` (`pipeline/streaming_training.py:69-77`) flip `detach_state=False` et passe le write en `enable_grad` gardé sur `requires_grad` (`_write_context`, L79-84). C3 fix : détache l'état au bord de chunk (`model/streaming_training.py:294-298`). Perte = DMD pur (`model/dmd.py:161-227`), per-frame, masqué aux frames neuves, AUCUN terme d'identité/recall.

**Le verdict mesuré.** e2_write3L, ckpt1200, layers [5,16,27] : dégrade à toutes les longueurs, pire à ~6s (gap -0.16 à 6s, -0.084 à 60s, -0.070 à 120s), gap qui rétrécit avec la longueur, gap DINO per-frame ~uniforme. Pas l'horizon (cross-chunk écarté), pas le cold-start.

**Ce qui est à moitié construit / mort.** `forward_sequence` (L398-410) "chemin parallèle d'entraînement" jamais appelé ; `_forward_train` lève `NotImplementedError` (`causal_model.py:1306-1307`) ; curriculum d'horizon commenté (`model/streaming_training.py:393-415`) ; BPTT cross-chunk câblée mais `bptt_window` jamais configuré, OOM (`distillation.py:1112-1137`) ; `gate_extra` zero-init donc décoratif sauf si activé ; `frames_since_switch` OOD à l'inférence (`causal_model.py:577-578`, `switch_global_frame=None`).

## La carte du domaine (ce qui est réglé vs ouvert)

**RÉGLÉ (contre toi).**

- **L'identité long-format se tient par sink + retrieval sur le vrai KV, pas par mémoire paramétrique.** Toute la vague *-Forcing sur Wan-1.3B : Rolling Forcing (arXiv:2509.25161), LongLive (2509.22622, 240s), Deep Forcing (2512.05081), Reward Forcing EMA-sink (2512.04678), Relax Forcing (2603.21366), DySink (2605.21028), Future Forcing (2605.30083), Context Forcing (2602.06028). Plusieurs sont training-free et à ~1pt des entraînées.
- **Le grid contrôlé (arXiv:2512.06983) :** encodeurs {cache, neural-weights/Titans, SSM} x injecteurs {QK-bias, adaLN, cross-attn, additif} -> "cache + context-prepend" gagne ; fast-weight SOUS-performe le no-memory. Ta combinaison (neural-weights + gate additif) est exactement la pire paire documentée.
- **Le revisit de scène est résolu par la géométrie + les données, pas par un module malin :** WorldMem (2504.12369), VMem (2506.18903), Context-as-Memory (2506.03141), Memory Forcing (2510.03198), GIM-World (2606.02436). Toutes pose/FOV-keyed.
- **Ta nouveauté est préemptée :** VideoSSM (2512.04519) entraîne déjà une mémoire gated-delta par distillation Self-Forcing/DMD sur rollouts AR 60s+, et bat LongLive ET Self-Forcing.
- **Côté update-rule :** delta/error-correcting bat Hebbian au recall partout (Titans 2501.00663, TTT-Video 2504.05298) ; le gating scalaire couplé est le goulot nommé (Gated DeltaNet-2 2605.22791, KDA 2510.26692) ; le gate borné (tanh) est le défaut du domaine.

**OUVERT (pour toi).**

- **Ré-identification de SUJET non-géométrique (A->B->A) :** un sujet quitte fenêtre+sink puis revient. Toutes les mémoires qui marchent sont géométrie/pose-keyed, donc gèrent le revisit de scène mais PAS le retour d'un sujet (personne/objet). C'est la niche claire pour une mémoire apprise content-keyed.
- **Aucun objectif ne donne à une mémoire apprise un signal de stockage long-range sous distillation.** REPA/CREPA alignent même-ou-adjacent (dans la fenêtre), DMD supervise un chunk court : le long-range exact que la mémoire devrait corriger n'est dans aucun gradient. Personne n'entraîne le write avec un signal d'identité explicite.
- **La métrique.** VBench subject_consistency est adjacency-dominated, background-contaminée, surnote les frames dégradées (Self-Forcing++ 2510.02283). Pas de benchmark re-ID/revisit standardisé ni de métrique ArcFace/DINO-sur-crop-sujet établie.

**Note de collision de noms :** arXiv:2512.18741 s'appelle aussi "MAG" (Memorize-and-Generate), même base Wan2.1-1.3B + Self-Forcing, diagnostique ton gap exact ("solution dégénérée" : DMD ignore l'historique, s'appuie sur le texte) et le corrige par une history-loss (drop du texte, lambda=0.6).

## Les idées classées

### Idée 1 (TOP) -- Recadrer sur la ré-identification de sujet (A->B->A) : benchmark + vraie métrique re-ID
- **Mécanisme :** générer des rollouts A->B->A avec la machinerie de switch existante (`streaming_switch_training.py:99-103,295-373`, prompt_cache, recache ; `switch_global_frame` déjà câblé `causal_model.py:577-583`), en faisant revenir A après >local(12)+sink(3) frames de B pour que le vrai KV de A soit évincé. Scorer la RÉCUPÉRATION d'identité = cosinus(crop-sujet au retour vs ancre pré-départ) via ArcFace (visages) / DINOv2 / DreamSim sur crop SAM, plus un contraste début-vs-fin. Confronter chaque config mémoire à no-memory (`baseline.yaml`), EMA-sink, dynamic-sink retrieval sur le MÊME backbone.
- **Pourquoi ça peut gagner :** seule niche ouverte (revisit de sujet non-géométrique non testé) et seul régime où rolling-KV+sink échoue par construction. Publiable sur les deux signes : une victoire mémoire est nouvelle, une défaite + le premier benchmark/métrique re-ID est quand même citable.
- **Fit code :** très haut, le générateur A->B->A existe ; nouveau = crop SAM/grounding + scorer cosinus (quelques centaines de lignes, sans entraînement).
- **Coût :** faible-moyen, ~1-2 semaines, pas d'entraînement pour définir le benchmark.
- **Risque :** moyen, les baselines retrieval peuvent aussi récupérer l'identité (alors la mémoire n'a pas de job) ; mais c'est l'expérience décisive. Détection de crop bruitée sur contenu stylisé (mitiger avec DreamSim + boîtes manuelles).
- **Payoff :** haut et robuste, satisfait la directive superviseur (5 longues vidéos), produit un artefact standardisé que le domaine n'a pas.
- **Ancré :** code `pipeline/streaming_switch_training.py:99-141,295-373`, `causal_model.py:577-583`, `configs/benchmark_{60,120,240}s_single.yaml`, `reference/e2_eval_parallel.py`. Lit : 2512.18741, 2605.18733 (IAMFlow), 2311.17982, 1801.07698 (ArcFace), 2306.09344 (DreamSim).

### Idée 2 -- L'objectif manquant : anchor-REPA long-range + history-loss
- **Mécanisme :** au hook `compute_generator_loss` (`model/streaming_training.py:680-689`, gradient_mask déjà là), ajouter (1) ANCHOR-REPA : aligner l'état post-mémoire (après [5,16,27]) ou `mem_out` aux features DINOv2/VideoMAE d'une frame-ancre DISTANTE hors fenêtre, par alignement SOFT de matrice-de-relations (leçon VideoREPA : le hard match casse un backbone figé), sur crop-sujet SAM ; (2) HISTORY-LOSS : pour les chunks non-premiers, prob lambda~0.6 drop du texte (prompt vide) -> `grad = (1-lambda)*DMD + lambda*history`. Le chemin uncond CFG existe déjà (`dmd.py:106-120`).
- **Pourquoi ça peut gagner :** fabrique exactement le gradient de stockage long-range absent, prérequis nommé par tous les pools. L'anchor-REPA distant est plausiblement nouveau (CREPA = adjacent seul, VideoREPA = même-clip, GIM-World = géométrie). Method-agnostic : aide la mémoire OU le retrieval OU le backbone nu. Donne gratis l'interprétabilité mem_out demandée par le superviseur.
- **Fit code :** haut, point d'insertion unique ; history-loss ~10 lignes.
- **Coût :** moyen, ~2-3 semaines, DINOv2 figé + petit MLP de projection + crop + curriculum d'horizon pour qu'une ancre tombe hors fenêtre.
- **Risque :** moyen, REPA non testé en régime distillé few-step + backbone figé ; collapse-vers-statique si lambda trop fort (reporter Dynamic Degree).
- **Payoff :** haut, contribution-coeur potentielle, et c'est le moteur qui rend l'Idée 1 gagnable.
- **Ancré :** `model/streaming_training.py:680-689,483-648`, `causal_model.py:556-604`. Lit : 2410.06940 (REPA), 2505.23656 (VideoREPA), 2506.09229 (CREPA), 2606.02436 (GIM-World), 2512.18741 (history-loss), 2602.06028.

### Idée 3 -- Baselines bon marché qui dominent peut-être (et qui deviennent la contribution)
- **Mécanisme :** sur le MÊME backbone, (a) augmenter `sink_size` (config) / `local_attn_size` ; (b) EMA-sink 0-paramètre : plier le KV évincé dans un état courant `SK = a*SK + (1-a)*K_evict` au point roll (`causal_model.py:247-302`) ; (c) dynamic-sink retrieval training-free : top-k blocs KV historiques par cosinus d'un descripteur DINO/PE-Core figé, concat dans le cache.
- **Pourquoi ça peut gagner :** le consensus du domaine ; DySink rapporte +21 Dynamic Degree sur sink statique. Si un sink bon marché ferme le gap 60/120/240s, la mémoire apprise est formellement dominée et le pivot est justifié proprement.
- **Fit code :** haut, pur changement de politique KV (`causal_model.py:214-440`), leviers config ; EMA-sink sans entraînement.
- **Coût :** faible, ~1 semaine pour EMA-sink + retrieval basique.
- **Risque :** faible scientifiquement ; gérer la ré-alignement RoPE des tokens sink ajoutés (sinon sink collapse / flicker, fixes publiés : Deep Forcing temporal-only RoPE, DySink anomaly gate).
- **Payoff :** moyen-haut, la baseline matchée que l'éval réclame ; falsifie ou crédibilise toute victoire mémoire.
- **Ancré :** `causal_model.py:214-440`, `configs/ttt_memory.yaml:77-78`. Lit : 2512.04678, 2605.21028, 2512.05081, 2603.21366.

### Idée 4 -- Métrique d'abord : score de dérive ancré sur crop-sujet (re-score des vidéos EXISTANTES, sans GPU)
- **Mécanisme :** remplacer VBench subject_consistency par : garder seulement le terme cos-vers-première (ancre), ajouter contraste début(15%)-vs-fin(15%), sur crop-sujet. `reference/e2_perframe_slice.py` calcule DÉJÀ `sf=cos_to_first` (L30) et `sp=cos_to_prev` (L29) séparément ; le changement = lâcher sp, reporter la courbe ancrée. Re-scorer les vidéos e2_write3L déjà générées.
- **Pourquoi ça peut gagner :** ton verdict négatif est peut-être en partie un artefact VBench ; une métrique ancrée est la seule façon de réclamer une victoire. Re-scorer les vidéos existantes peut changer la conclusion AUJOURD'HUI.
- **Fit code :** excellent, `e2_perframe_slice.py:19-38` fait tout sauf le crop ; tourne en env longlive sur les .mp4 existants.
- **Coût :** heures (version frame-entière) ; +0.5-1 jour pour le crop SAM.
- **Risque :** très faible.
- **Payoff :** débloque toutes les autres directions, possible reframe du headline à coût ~nul.
- **Ancré :** `reference/e2_perframe_slice.py:19-38`, `e2_eval_parallel.py:105-138`. Lit : 2311.17982, 2510.02283, 2306.09344.

### Idée 5 -- Sondes de falsification gratuites (= interprétabilité superviseur)
- **Mécanisme :** sur le checkpoint 3L : (1) `MEM_ZERO_COLD_FRAMES` (`causal_model.py:597-600`) isole le cold-start ; (2) FREEZE-STATE-NEVER-WRITE : lire `init_state` chaque chunk (`video_memory.py:357-360`) avec `do_update` off -> si l'identité chute quand même, c'est la TRANSFORMATION de lecture, pas le carry/BPTT, le coupable ; (3) bins DINO cold-vs-warm (`e2_perframe_slice.py:41`, arm coldzero) ; (4) `feature_maps.py`/`probe_read.py` pour voir si le read pré-gate porte l'identité (Q3).
- **Pourquoi ça peut gagner :** convertit les opinions en preuves au moindre coût ; dit s'il faut corriger le gate, l'objectif, ou arrêter.
- **Fit code :** excellent, les 4 sondes sont déjà écrites/env-gated.
- **Coût :** <1 jour.
- **Risque :** très faible (diagnostics purs).
- **Payoff :** attribution défendable du POURQUOI, ordonne tous les retrains suivants ; livre l'interprétabilité mem_out demandée.
- **Ancré :** `causal_model.py:597-600`, `video_memory.py:357-363,393-394`, `reference/feature_maps.py`, `probe_read.py`, `why_ours_fails_vs_papers.md:87`.

### Idée 6 -- Parité train/inférence (LongLive) : corriger l'OOD frames_since_switch + matcher l'horizon
- **Mécanisme :** (1) passer `switch_global_frame` dans `pipeline/causal_inference.py` pour que `frames_since_switch` matche l'entraînement (`causal_model.py:577-578`) ; (2) ré-activer le curriculum commenté (`model/streaming_training.py:393-415`) et utiliser DMD sliding-window (Self-Forcing++) plutôt que BPTT longue.
- **Pourquoi ça peut gagner :** fermer le gap exposition/horizon est le levier dominant (LongLive, Self-Forcing++) indépendamment de toute mémoire ; l'OOD fss est un confond non corrigé qui empoisonne peut-être le gate à l'éval.
- **Fit code :** bon, plumbing switch déjà côté training.
- **Coût :** heures (fss) à 2-4 jours (retrain horizon).
- **Risque :** faible pour fss ; l'extension d'horizon touche le mur VRAM/BPTT (rester en sliding-window).
- **Payoff :** retire un confond d'éval gratuitement.
- **Ancré :** `causal_model.py:577-578,520-583`, `pipeline/causal_inference.py:141,197-237`, `model/streaming_training.py:393-415`. Lit : 2509.22622, 2510.02283.

### Idée 7 -- Petit retrain ciblé : gate borné + lecture couplée à la pertinence + arm propre g_init=0
- **Mécanisme :** dans `_project_output` (`video_memory.py:344-355`) : remplacer le gate non borné par `g_max*tanh(gate_raw)` (bias init gate~0) ; lâcher/adoucir le second norm (`a_norm` L355 après `o_norm` L346) et/ou injecter `||o||` pré-norm dans le gate ; arm matché `gate_init=0.0` pour step0==baseline.
- **Pourquoi ça peut gagner :** corrige la cause racine #2 (injection à RMS fixe) ; gate borné = défaut du domaine (TTT-Video), write pondéré-pertinence (OSDN 2605.13473).
- **Fit code :** excellent, fonction unique ; `gate_init` levier config.
- **Coût :** ~1 jour code + 1-3 jours retrain read-path 1 layer.
- **Risque :** faible-moyen, borner peut rendre la mémoire inerte (gate déjà observé ~6e-4) ; coupler à un gate-floor schedule.
- **Payoff :** arm falsifiable do-no-harm ; si inerte, preuve décisive que le read ne porte rien.
- **Ancré :** `video_memory.py:344-355,267-271,288-289`, `gating_design_exploration.md:22-32`. Lit : 2504.05298, 2605.13473, 2512.06983.

### Idée 8 -- Si tu gardes le fast-weight : delta rule + gates découplés (uniquement testé sur le benchmark re-ID)
- **Mécanisme :** (1) Hebbian -> delta `(v - M k)k^T` (`video_memory.py:129-155`) ; (2) gates DÉCOUPLÉS erase-clé / write-valeur, forget faible data-dependent (GDN-2) pour écrire B sans effacer A ; (3) terme de forget réel dans `weight_norm_update` (actuellement renorm-vers-init pur, L158-188).
- **Pourquoi ça peut gagner :** delta bat Hebbian au recall ; overwrite adressé-par-clé + erase/write découplés = mécanisme documenté pour "garder l'ancienne clé en écrivant la nouvelle" (cas A->B->A).
- **Fit code :** haut et chirurgical, localisé à `video_memory.py`.
- **Coût :** moyen, ~2-3 semaines (les 4 changements interagissent).
- **Risque :** moyen-haut, même bien construit le fast-weight peut perdre contre le retrieval ; préempté par VideoSSM ; INUTILE sans le gradient de stockage (Idée 2) en amont.
- **Payoff :** moyen, rend la mémoire compétitive sur sa meilleure tâche ; surtout l'arm "on a construit la bonne mémoire et elle (gagne/perd) au revisit".
- **Ancré :** `video_memory.py:129-188,344-355`. Lit : 2501.00663, 2412.06464, 2605.22791, 2503.14456, 2605.13473.

### Idée 9 (plancher) -- Papier négatif + diagnostic mécaniste
- **Mécanisme :** packager le post-mortem en grid contrôlé encodeur x injecteur sur le même student distillé + les falsifications de l'Idée 5 + le diagnostic "horizon supervisé <= fenêtre KV".
- **Pourquoi ça peut gagner :** peu de démonstrations vidéo contrôlées sur un vrai backbone AR distillé avec le mécanisme horizon<fenêtre nommé ; plancher de thèse garanti.
- **Fit code :** haut, switches de falsification déjà là ; exposer les knobs hardcodés (`base_lr`, `muon_update_steps`, `qk_norm`, `inter_multi`) via model_kwargs.
- **Coût :** faible-moyen, ~2 semaines.
- **Risque :** faible en livraison, réception incertaine (un négatif pur peut décevoir un superviseur qui voulait une méthode) ; cadrer comme filet sous Idées 1-2.
- **Payoff :** moyen, plancher solide ; combiné à l'Idée 1, papier complet fort.
- **Ancré :** `video_memory.py:362-363`, `causal_model.py:597-600,498-500`, `reference/why_memory_degrades.md`, `why_ours_fails_vs_papers.md`. Lit : 2512.06983, 2601.17067, 2506.05284.

### Idée 10 (réserve, ceiling haut / odds bas) -- Vidéo interactive infinie, état O(1)
- **Mécanisme :** cibler le streaming interactif temps-réel où le KV croît sans borne ; valeur = état O(1) à FPS fixe, pas SOTA d'identité brute. Chemin `pipeline/interactive_causal_inference.py` ttt_cache O(1) ; profiler FPS/VRAM-vs-longueur contre une baseline retrieval à KV croissant.
- **Pourquoi ça peut gagner :** O(1) horizon-infini est le seul axe où un état paramétrique bat structurellement le retrieval.
- **Fit code :** moyen, chemin O(1) existe ; passer `switch_global_frame` à l'inférence pour fixer l'OOD.
- **Coût :** haut, ~3-4 semaines.
- **Risque :** haut, littérature négative (RELIC/retrieval gagnent même l'interactif ; SSM/VideoSSM occupe déjà la niche d'état compact plus crédiblement).
- **Payoff :** plus gros ceiling, plus faibles odds ; uniquement APRÈS que les Idées 1-4 montrent que la mémoire fait ce que le retrieval ne peut pas.
- **Ancré :** `pipeline/interactive_causal_inference.py`, `configs/interactive_inference.yaml`, `video_memory.py:44-57`. Lit : 2510.20668 (Genie 3 survey), 2512.04040 (RELIC), 2601.17067.

## Le programme recommandé (séquence, semaine par semaine)

Le throughline : définir un problème gagnable et bien mesuré D'ABORD, puis amener le seul mécanisme nouveau bon marché que le diagnostic dit pouvoir le gagner. Ne pas dépenser d'effort lourd avant que la métrique re-ID montre que la mémoire a un job mesurable.

**Semaine 1 -- Mesurer et diagnostiquer (coût quasi nul, peut clore honnêtement).**
- Idée 4 : re-score ancré sur crop-sujet des vidéos e2_write3L existantes (heures, sans GPU). Peut reframer le headline immédiatement.
- Idée 5 : les 4 sondes de falsification (freeze-state-read, MEM_ZERO_COLD, cold/warm DINO, feature_maps) -> attribution read vs carry vs objectif + interprétabilité superviseur.
- Idée 6 (partie gratuite) : corriger l'OOD `frames_since_switch` à l'inférence (heures).

**Semaine 2 -- Les baselines matchées (décisif).**
- Idée 3 : EMA-sink 0-param + dynamic-sink retrieval training-free sur le même backbone, sur benchmark_{60,120,240}s. Si un sink bon marché ferme le gap, le pivot est justifié et c'est déjà un résultat publiable.

**Semaines 3-4 -- Construire la niche.**
- Idée 1 : benchmark A->B->A + métrique re-ID (crop SAM + ArcFace/DINO/DreamSim, ancré pas adjacency). Lancer mémoire vs no-memory vs EMA-sink vs retrieval dessus.

**Semaines 4-6 -- L'objectif qui peut gagner la niche.**
- Idée 2 : anchor-REPA soft-relation + history-loss au hook compute_generator_loss. Method-agnostic, plausiblement nouveau, donne l'interprétabilité. Retrain read-path court d'abord.

**Semaines 6-8 -- Affiner le substrat (conditionnel au signal de l'Idée 1).**
- Idée 7 (gate borné, do-no-harm) puis Idée 8 (delta + gates découplés) UNIQUEMENT si le benchmark re-ID montre que la mémoire a un job. Gardé derrière l'Idée 2 : un meilleur write sans gradient de stockage n'apprend rien.

**Filet et réserve.**
- Idée 9 (négatif + diagnostic) : packageable à tout moment comme plancher de thèse, renforcé par le benchmark de l'Idée 1.
- Idée 10 (interactif O(1)) : seulement si la course qualité paraît imperdable et que la mémoire atteint la parité.

## La première expérience à lancer

**Idée 4, version frame-entière, sur les vidéos déjà générées, aujourd'hui, sans GPU.**

Re-scorer les rollouts e2_write3L existants (mem vs base à 60/120/240s) avec une métrique de dérive ANCRÉE : garder uniquement le terme `cos_to_first` (ancre), lâcher le terme adjacency `cos_to_prev`, et ajouter un contraste début(15%)-vs-fin(15%). Le code est déjà là : `reference/e2_perframe_slice.py:30` calcule `sf = cos_to_first` et L29 `sp = cos_to_prev` séparément, avec le même backbone DINO que VBench, en env longlive sur les .mp4 existants.

Pourquoi celle-ci en premier : coût en heures, zéro inférence, zéro entraînement ; c'est un prérequis strict (aucune autre direction ne peut réclamer une victoire mesurable sans une métrique qui voit la dérive lente) ; et la littérature suggère fortement que ton verdict négatif est en partie un artefact VBench, donc ce re-score peut changer la conclusion headline le jour même. Si la métrique ancrée confirme la dégradation, tu passes aux sondes de falsification (Idée 5) avec une attribution propre ; si elle la nuance, tout le cadrage stratégique se reconsidère avant de dépenser un seul GPU-jour.

## Ce qu'on arrête

- **Arrêter d'optimiser VBench subject_consistency comme cible.** Adjacency-dominated, background-contaminée, surnote les frames dégradées (2510.02283, 2311.17982) ; elle est aveugle au problème exact que la mémoire vise. Continuer à pousser ce nombre maximise la probabilité d'un négatif confirmé sans contribution.
- **Arrêter de combattre la dérive AVANT (forward drift) avec un fast-weight.** Course réglée : retrieval + sink sur le vrai KV gagne, et fast-weight + gate additif est la pire paire documentée (2512.06983). N'y revenir que comme baseline-foil.
- **Arrêter de présenter "entraîner le write par self-rollout" comme LA nouveauté.** Préempté par VideoSSM (2512.04519). La nouveauté doit venir de l'angle re-ID de sujet + objectif de stockage, pas du mécanisme d'entraînement.
- **Ne pas investir dans la BPTT cross-chunk lourde / replayed-backprop maintenant.** Le mur VRAM/OOM est connu (`distillation.py:1112-1137`, mémoire crosschunk-bptt-infeasible) et l'horizon n'est PAS la cause racine prouvée ; n'y toucher que si la métrique re-ID montre que la dérive est horizon-limitée.
- **Ne pas lancer un gros retrain (Idées 7/8) avant les Semaines 1-2.** Les sondes gratuites doivent d'abord confirmer que le read porte un signal ; sinon un meilleur write rule entraîne du bruit.
- **Geler le travail sur `gate_extra`/agreement (ancre = prompt poolé).** Variante rejetée comme trop grossière (catégorie pas instance) ; le proj_extra zero-init est décoratif. Remplacé par l'ancre identité réelle de l'Idée 2.
- **Nettoyer le code mort plutôt que le maintenir :** `forward_sequence` (`video_memory.py:398-410`), `_forward_train` (`causal_model.py:1306`), branches GAN/decouple-DMD off, flotte de watchers GPU dans `reference/`.
