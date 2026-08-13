# Design review À CHARGE : mécanisme "sink virtuel" (état fast-weight → m tokens k/v dans le chemin d'attention)

Volet (a)+(e) de l'ultraplan. Lecture seule du code au 2026-07-23 ; toutes les références de lignes vérifiées dans l'arbre courant (branches modifiées non commitées incluses).
**Périmètre étendu (décision Aden 2026-07-23, brief §2 mis à jour) : la cible est le MULTI-SWITCH interactif (A→B→C→A→D…), pas seulement A→B→A.** L'extension est traitée en §3.6-3.8 et dans les bugs #11-13.
Fichiers audités : `wan/modules/causal_model.py`, `wan/modules/video_memory.py`, `wan/modules/attention.py`, `pipeline/switch_causal_inference.py`, `pipeline/streaming_training.py`, `pipeline/streaming_switch_training.py`, `trainer/distillation.py`, `utils/wan_wrapper.py`, configs `ttt_memory.yaml` / `reference/runs/l1_longlive/l1_longlive.yaml`, résultats `reference/runs/reid_probe_l1/results/summary.json`.

Chiffre de gating intégré (tombé pendant la review) : **oracle1 = +0.224 ≈ oracle full +0.220 (24+/0−)** ; dans le summary.json partiel (12 cellules) : oracle1 12+/0− p=0.0005, oracle3 ≈ oracle full aussi. **UNE frame latente de KV de A suffit.**

---

## 0. Verdict d'ensemble (résumé exécutif)

Le mécanisme est la bonne branche (le canal KV est prouvé par l'oracle, le canal résiduel est prouvé mort par memsnap), MAIS le design tel qu'esquissé dans le brief a **un trou de gating** et **deux pièges d'implémentation mortels** :

1. **Trou de gating : l'oracle splice les 30 blocs, le design n'insère qu'à [5,16,27].** Personne n'a mesuré un splice restreint aux 3 blocs mémoire. Les +0.22 ne garantissent RIEN pour une insertion 3-blocs. Probe $0 obligatoire avant d'entraîner (§4.1).
2. **Piège mortel n°1 : écrire physiquement les tokens dans le cache KV casse l'arithmétique de pointeurs à ~8 endroits** (roll, is_recompute, recache, clear, eviction-feed). La seule insertion saine est la **concat au moment de l'attention**, sur le template `prompt_cache` qui existe déjà (§3.1, §4.1).
3. **Piège mortel n°2 : le "zero-init" naïf (v=0 concaténé) n'est PAS inerte** — la softmax est à somme fixe, un token concaténé détourne de la masse même avec v=0, et une clé nulle a un logit 0 (pas −∞) donc attire ~1/N de masse par query. La byte-identity exige un merge par log-sum-exp avec gain scalaire g init 0 et un **skip structurel** à g=0 (§3.2, §4.3).

---

## 1. Faits de code qui contraignent le design (vérifiés, avec lignes)

### 1.1 Le chemin d'attention avec cache (`causal_model.py`, `CausalWanSelfAttention.forward`)
- Chemin sink>0 (l. 351-437) : `x = attention(roped_query, k_cat, v_cat)` avec `k_cat = [k_sink, (k_prompt_cache), k_local]` — **softmax pleine, sans masque** : chaque token query attend TOUTES les clés concaténées. C'est exactement là qu'un `k_virt/v_virt` s'insérerait : le **template `prompt_cache` (l. 386-426) fait déjà une insertion de KV supplémentaires entre sink et fenêtre**, avec sa propre convention RoPE.
- Sous `rope_enhance` (l1 : `true`), le cache stocke des clés **NON ropées** ; le rope est appliqué au moment de l'attention en **bandes de frames disjointes** : query à `rope_start_f = min(21 − gen, current_frame)` (l. 367, horloge bornée à 21 frames), sink à `[0, _sink_runtime)` (l. 374/380), fenêtre locale à `rope_start_f_k_local = max(rope_start_f − budget + gen, _sink_runtime)` (l. 368), prompt_cache à `rope_start_frame = 21 − local_attn_size − pf` (l. 1204). Toute clé jamais vue par le modèle est ropée dans une de ces bandes.
- Chemin sink==0 (l. 438-462) : branche séparée, **sans le concat prompt_cache**, réparée récemment pour rope_enhance (commentaire l. 443 : cette branche avait été oubliée). C'est la branche des rollouts cache-dropout (`cache_dropout_sink: 0` dans l1).
- Budget : `max_attention_size = 12×1560 = 18720` tokens ; sink 3 frames = 4680 ; `local_budget` = 9 frames = 14040. Ajouter m=1560 → 20280 clés attendues (+8.3%) : sans limite dure côté flash-attn, mais NE PAS déduire m du `local_budget` (ça rétrécirait la fenêtre réelle = confound vs baseline).
- L'arithmétique de cache (l. 251-345 + `_apply_cache_updates` l. 1140-1213) repose partout sur `sink_tokens = _sink_runtime × frame_seqlen` : roll (l. 275-278, 1164-1167), protection d'écriture recompute (l. 282, 321-323, désactivée par `sink_recache_after_switch`), détection `is_recompute` par `global_end_index` (l. 250), eviction-feed `sink_chunks` (l. 1363). **Une zone virtuelle physique dans le cache doit décaler chacun de ces calculs.**

### 1.2 Le contrat gradient-checkpointing (l. 1386-1450)
Tout état lu DANS le bloc checkpointé doit être une **entrée explicite** parce que le backward re-exécute le bloc : c'est le mécanisme `ttt_state` (l. 1399-1403, snapshot pris HORS du checkpoint), `mem_ema_prev` (l. 1407), `mem_x_write` (pop du fifo HORS du checkpoint, l. 1350-1385). Le fifo append dans le bloc (l. 640) est ré-exécuté au recompute et réparé par le snapshot/restore du trainer (`distillation.py` l. 1869-1893, autour de CHAQUE backward). Les k/v virtuels devront suivre EXACTEMENT ce contrat.

### 1.3 Le périmètre FSDP/"memory" (`distillation.py`)
La mémoire vit hors FSDP et TOUT son cycle de vie passe par des name-matchs :
- exclusion FSDP par **type** : `type(m).__name__ == "VideoMemoryLayer"` (l. 322-323) + broadcast rank-0 (l. 328-332) ;
- dégel : `"memory" in n` (l. 292-294) ; groupe d'optimiseur : l. 415-422 ; clip dédié `_clip_mem_grads` (les params FSDP-ignored ne sont PAS couverts par le clip FSDP, l. 429-432) ;
- all-reduce manuel des grads : `"memory" in _n` (l. 1474-1477 et 1542-1545) ;
- sauvegarde : `_gather_memory_state_dict` (l. 810-819) ; tolérance de load `strict=False` sur les clés "memory" (l. 134-149) ;
- exclusion des cibles LoRA : `".memory." not in full_submodule_name` (l. 1672-1674).
**Conséquence : le décodeur d'état→k/v DOIT être un sous-module de `VideoMemoryLayer`** (donc dans `video_memory.py`, comme le brief le prévoit) sinon 6 points d'intégration cassent en silence, le pire étant l'absence d'all-reduce (décodeur divergent par rank).

### 1.4 Les pipelines switch/recache (`switch_causal_inference.py`)
- `_recache_after_switch` (l. 56-128) : zère k/v mais **PAS les indices** (l. 63-65 commentées), tourne avec `ttt_cache` actif et `ttt_update=False` → les lectures mémoire sont actives pendant le recache (la "fuite de B" connue). `sink_recache_after_switch=not global_sink` autorise la réécriture du sink.
- Hooks validés : ORACLE_SPLICE (+ partiel `ORACLE_SPLICE_FRAMES`, l. 236-259), MEM_EPISODIC snapshot/restore (l. 37-54, 229-234, 266-269).
- **Le teacher RSD n'existe QUE dans ce pipeline d'inférence.** Le pipeline d'ENTRAÎNEMENT (`streaming_switch_training.py`) ne supporte qu'UN switch (A→B) et aucun splice-override ; `switch_choices: [1000]` dans l1 avec `streaming_max_length: 300` → **l'entraînement actuel ne voit jamais aucun switch** (confirmé). Tout le curriculum A→B→A + le teacher oracle sont à CONSTRUIRE côté training.

### 1.5 MEM_ZERO_ALL et l'attention wrapper
- MEM_ZERO_ALL (l. 653-658 de causal_model.py) zère `mem_out` APRÈS calcul — sémantique "compute path identique, injection nulle".
- `attention()` (`wan/modules/attention.py`) : FA3 renvoie `(out, lse)` et le code **jette la LSE** (`[0]`, l. 116) ; FA2 `flash_attn_varlen_func` peut l'exposer via `return_attn_probs=True`. La LSE est donc récupérable avec un petit changement de wrapper — nécessaire pour le merge exact (§3.2).

---

## 2. Modes de défaillance énumérés

### 2.1 Poison d'attention — quantification du rayon d'explosion
Un token virtuel mal entraîné à un bloc mémoire se propage en 3 cercles :

1. **Rayon instantané** : la softmax est partagée par toutes les queries → les m tokens sont lus par les **4680 tokens query du chunk (3 frames × 1560) × 12 têtes × 5 passes** (4 denoising + 1 clean). Contrairement au canal résiduel, il n'y a **aucune enveloppe d'amplitude** : `MEM_STREAM_SCALE`/dc_remove/rejection sont inapplicables dans la softmax. Une clé qui s'aligne sur les queries capte une masse O(1) sur CHAQUE query (comportement super-sink ; les sinks réels captent couramment 30-60%+ de la masse). Et même v=0 nuit : masse détournée = atténuation uniforme du signal réel (somme softmax fixe).
2. **Rayon cache** : l'injection à bloc 5 corrompt x en sortie de bloc 5 → les blocs 6-29 calculent leurs k/v depuis ce x corrompu et les ÉCRIVENT dans leur cache sur la passe clean → le poison persiste dans le cache KV des **30 blocs pendant local_attn_size = 12 frames (4 chunks)** même si on retirait les tokens ensuite. Bloc 5 étant tôt, 25 blocs aval sont touchés (c'est le bloc mémoire le plus dangereux ; 27 le moins).
3. **Rayon récursif** : les latents émis sont re-bruités en contexte (`context_noisy`) et ré-encodés → persistance **non bornée** via la boucle autorégressive pixel. Ce cercle existait déjà pour le canal résiduel ; les cercles 1-2 sont ceux qui grossissent (~10-100× de masse potentielle par rapport à une injection résiduelle cappée à 0.5×RMS du stream).

Mitigations OBLIGATOIRES dès la v1 : gain scalaire g par bloc (init 0), **cap RMS sur v_virt** (l'analogue softmax de l'enveloppe stream-RMS ; jamais de renorm-après-soustraction), tripwire wandb dès step 0 sur la masse d'attention captée par les tokens virtuels par bloc (seuil d'alerte ~30%) + RMS(k_virt), RMS(v_virt). Le pattern c2 (ramp lente des slow-params → dépassement d'un seuil de cohérence à step 1000+) se reproduira ICI EN PIRE si ces jauges n'existent pas dès le premier run.

### 2.2 Sémantique RoPE / positions des tokens virtuels
Trois options, une seule survit :

| Option | Analyse | Verdict |
|---|---|---|
| (a) Positions du chunk évincé | Sous rope_enhance l'horloge est **bornée à 21 frames** (l. 367) : la position d'origine de l'évincé n'existe plus dans la bande query. Sous non-rope_enhance, positions absolues croissantes = géométrie de phase jamais vue. | **REJET** |
| (b) Sans RoPE (clés brutes du décodeur) | Toutes les clés self-attn vues à l'entraînement sont ropées. Une clé non ropée voit son score q·k osciller avec la phase (temporelle ET spatiale) de chaque query ; le décodeur ne peut pas compenser pour toutes les positions à la fois. Apprenable en théorie, interface OOD en pratique. | REJET en v1 (ablation possible plus tard) |
| (c) Bande de frames FIXE sink-adjacente + grille spatiale réelle | La géométrie relative query↔sink (distance ≤ 21) est le régime entraîné ; le modèle lit déjà le `prompt_cache` dans une bande fixe intercalée (l. 1204 : `21 − local − pf`). Réutiliser cette convention : k_virt ropé à `start_frame` fixe entre le sink et la fenêtre, spatial = la vraie grille (m = 1 frame pleine). | **RETENU** |

Pièges de (c) : (i) **collision de bandes en début de vidéo** — tant que la fenêtre n'est pas pleine, `rope_start_f_k_local` descend jusqu'à `_sink_runtime` (l. 368) et chevaucherait la bande virtuelle → activer les tokens virtuels **seulement après la première éviction** (avant, la mémoire n'a de toute façon rien à ressusciter ; même logique de warm-up que le fifo eviction-feed) et ASSERT bandes disjointes ; (ii) sous non-rope_enhance (si un run l'utilise), les clés du cache sont déjà ropées absolues : le décodeur doit alors roper ses clés à la position absolue courante du slot — gérer les deux modes comme le fait `prompt_cache` (l. 393-396).

### 2.3 Interaction avec la troncature de fenêtre
- Insertion physique dans le cache : voir §1.1 — huit sites d'arithmétique à décaler, chacun un off-by-one silencieux (le roll copie `sink_tokens + num_evicted → sink_tokens` ; la protection recompute borne l'écriture à `sink_tokens` ; `is_recompute` compare des indices globaux qui ignoreraient la zone virtuelle ; `clear_kv_cache` et le recache zèrent tout). **REJET.**
- Concat au moment de l'attention : ZÉRO interaction avec le roll/évict — le cache et ses pointeurs ne changent pas d'un octet. Seul coût : +8.3% de clés attendues, +1 frame de latence d'attention.

### 2.4 Interaction avec le recache après switch
- Les tokens virtuels seront actifs pendant le forward de recache (le recache passe `ttt_cache`) : c'est **voulu** (c'est le vecteur de résurrection : la reconstruction du cache au retour de A se fait sous l'influence de la mémoire), mais c'est AUSSI le vecteur de la fuite de B : au retour A→B→**A**, l'état contient du B récent ; sans gestion (MEM_EPISODIC, décay, ou gate de phase `frames_since_switch` déjà câblé), les k/v virtuels re-injectent B pendant la reconstruction du cache A. Le probe memsnap a montré que l'épisodique n'apporte rien SUR LE CANAL RÉSIDUEL mort ; il doit être RE-testé sur le canal KV (probe d'inférence $0, hook déjà en place).
- Les indices ne sont pas resetés au recache (l. 63-65) et `current_start` y prend des valeurs inhabituelles : le décodage doit être **stateless par forward** (pas de cache de tokens décodés keyé sur current_start).

### 2.5 MEM_ZERO_ALL (validité des ablations)
Zérer `v_virt` ne suffit PAS (masse softmax toujours détournée) ; forcer g=0 dans le merge LSE est mathématiquement exact mais pas bit-exact (ordre d'opérations flottantes différent). **La seule sémantique acceptable : MEM_ZERO_ALL ⇒ skip STRUCTUREL de tout le chemin virtuel** (pas de décodage, pas de seconde attention, pas de merge) + le zérage existant de mem_out. Test d'acceptation : latents bit-identiques entre `MEM_ZERO_ALL=1` sur le ckpt entraîné et le run `memory_layers: []` (la colonne stockaf existe déjà). Sans ça, TOUTE la campagne 3-way est invalide et on ne le découvre qu'à l'audit final.

### 2.6 Gradient checkpointing
Si le décodeur lit `ttt_cache.state` (objet live) dans le bloc checkpointé, le recompute du backward après une passe d'écriture lit l'état POST-write → k/v virtuels différents entre forward et recompute → **gradients silencieusement faux, aucun crash**. Deux implémentations conformes :
- (i) décoder DANS le bloc mais uniquement depuis `ttt_state` (déjà entrée explicite) ;
- (ii) **préférée** : décoder UNE fois par forward HORS du checkpoint dans `_forward_inference` (comme le pop `mem_x_write`, l. 1350-1385) et passer `(k_virt, v_virt)` en kwargs explicites. Possible parce que le décodeur n'a PAS de dépendance à x (queries apprises) et que l'état est constant entre les passes d'un même chunk. Bénéfice : recompute-safe par construction + décodage 1× au lieu de 5×. Les kwargs-tenseurs sous `use_reentrant=False` reçoivent bien les gradients (mécanisme déjà exercé par ttt_state). Ne pas introduire de nouvel état mutable sur `TTTBlockCache` ; si inévitable, respecter le contrat replace-not-mutate + l'ajouter au `_snapshot_ttt_state` du trainer ET au `_snapshot_ttt` du pipeline switch.

### 2.7 FSDP
Voir §1.3. Le décodeur DANS `VideoMemoryLayer` hérite de tout (ignore-FSDP, broadcast init, all-reduce, groupe lr_mem, clip, EMA trainable-only, save/load, exclusion LoRA). Hors de ce périmètre, la défaillance est **silencieuse et non déterministe** (chaque rank entraîne un décodeur différent ; l'EMA et le checkpoint sauvent celui du rank 0 ; les évals sont du bruit). C'est le bug le plus cher de toute la liste.
Nota : le broadcast init (l. 328-332) couvre `m.parameters()`/`m.buffers()` du module — les queries apprises du décodeur doivent être des `nn.Parameter` (pas des tensors nus) pour être broadcastées.

---

## 3. Choix de design ouverts : options et verdicts

### 3.1 OÙ insérer — mécanisme
1. Écriture physique dans le cache (slots réservés à côté du sink) — REJET (§2.3, 8 sites d'arithmétique).
2. **Concat au moment de l'attention, template prompt_cache : `k_cat = [k_sink, k_virt, k_local]`** — RETENU. Aucune mutation du cache, gating structurel trivial, entrées explicites pour le checkpoint, byte-identical par construction quand le chemin est sauté. À implémenter dans les DEUX branches (sink>0 l. 351-437 ET sink==0 l. 438-462 — la branche cache-dropout, précisément celle où la mémoire doit porter l'ancre).
3. Variante "state dénoisé dans le flux" (à la AdaState) — hors périmètre (préempté, et ce serait un autre papier).

### 3.2 Gate / zero-init exact
1. Concat naïf + v zero-init — **PAS inerte** (masse détournée ; clé nulle = logit 0, pas −∞). REJET.
2. Biais de logit init très négatif — non supporté par le kernel flash sans bias tensor, jamais bit-exact. REJET.
3. **Seconde attention sur les m tokens seuls + merge log-sum-exp** : `out = (Z_b·x_b + g·Z_v·x_v)/(Z_b + g·Z_v)`, avec `g` **scalaire LINÉAIRE par bloc (option : par tête), init exactement 0**, et skip structurel du chemin quand g==0 / MEM_ZERO_ALL. — RETENU. À g=0 : byte-identical exact ; gradient ∂out/∂g|₀ = (Z_v/Z_b)·(x_v − x_b) ≠ 0 → pas de starvation. La LSE de l'attention principale est déjà calculée par FA3 (jetée l. 116) et exposable en FA2 (`return_attn_probs=True`) ; la seconde attention (m=1560 clés) est assez petite pour un sdpa manuel fp32 si besoin.
   **RÈGLE ANTI-STARVATION : exactement UN zéro dans la chaîne.** g=0 ET décodeur init normal (petit). Interdire g=0 + W_v=0 (gradient mort des deux — la leçon "gate=0 → all grads 0" est déjà payée). Interdire aussi g=λ² ou g=sigmoid(−∞) (gradient nul ou jamais exactement 0).
   Fallback si la LSE s'avérait indisponible : add résiduel pré-`self.o` de `g·x_v` — à documenter comme DÉGRADÉ (ce n'est plus une compétition de masse softmax, on se rapproche du canal résiduel mort).

### 3.3 Quels blocs
- Options : [5,16,27] seuls / les 30 / les 30 avec cellules partagées + décodeurs par bloc.
- **Fait bloquant : l'oracle (+0.220) splice les 30 blocs ; le splice 3-blocs n'a jamais été mesuré.** Le gating chiffré ne couvre donc PAS ce choix. Probe $0 avant tout entraînement : "oracle-3-blocs" — recache B normal, snapshot du cache, recache oracle, puis restauration des blocs ∉ {5,16,27} depuis le snapshot (~20 lignes dans `_recache_after_switch`, inference-only, réutilise le harness 24 cellules). Lecture : si ≥ ~70% du delta survit → [5,16,27] confirmé ; si le delta s'effondre → il faudra insérer plus large (toutes les ~3 couches, cellule partagée par segment pour contenir VRAM/params) et le coût du projet change de catégorie — à savoir AVANT d'écrire le décodeur.
- Le même hook donne gratuitement la variante "splice côté sink" (l'oracle1 actuel place la frame A au DERNIER slot de la fenêtre = avantage de récence ; la position sink-adjacente retenue en §2.2 n'est pas strictement testée par oracle1 — vérifier que le delta survit à la position).

### 3.4 m (nombre de tokens)
- Plancher mesuré : **oracle1 = +0.224 ≈ full +0.220 (24+/0−)** → 1 frame latente de vrai KV suffit à ressusciter le sujet.
- Options : m=1560 (1 frame pleine, grille spatiale réelle) / m≈390 (grille sous-échantillonnée) / m=32-128 (tokens abstraits sans grille).
- **Verdict : m = 1560 = exactement une "frame synthétique"** — interface 100% in-distribution (une 4e frame de sink), rope spatial trivial (la vraie grille), et c'est LITTÉRALEMENT ce que l'oracle1 prouve suffisant. Les m plus petits imposent une grille spatiale jamais vue (OOD) : à garder pour l'ablation de compression APRÈS un premier résultat positif, pas avant. Coût : décodeur = m queries apprises (1560×1536 ≈ 2.4M) + 2 têtes k/v (1536→1536) ≈ **~7M params/bloc, ~21M pour 3 blocs** ; activations k/v = 2×1560×1536 par bloc — négligeable devant le cache.

### 3.5 Read-only vs read-write pendant les passes bruitées
- **Écriture : inchangée** — `ttt_update=True` uniquement sur la passe clean/context (streaming_training.py l. 358-374). Écrire depuis les passes bruitées injecterait le bruit σ_t dans l'état et multiplierait par 5 les mutations d'état par chunk (le contrat snapshot/restore devient intenable).
- **Présence des tokens virtuels (lecture)** : (a) toutes les passes / (b) passes bruitées seulement / (c) dernière passe seulement. Verdict : **(a) toutes les passes** — l'oracle qui fournit le signal opère via un cache lu par toutes les passes ; (b)/(c) créent un mismatch denoise↔cache jamais vu à l'entraînement, et leur "confinement du poison" est illusoire (le cercle récursif pixel §2.1-3 le porte de toute façon). Décoder 1×/chunk (état constant entre passes) et réutiliser.
- Activation retardée : tokens virtuels actifs seulement après la première éviction (§2.2), signal propre pour le gate de phase déjà câblé (`frames_since_switch`).

---

## 4. Bugs les plus probables, classés par (probabilité × coût de détection tardive)

| # | Bug | P | Coût tardif | Test de détection PRÉCOCE |
|---|---|---|---|---|
| 1 | **Décodeur hors du périmètre "memory"** (nom ou type) → pas d'all-reduce → décodeur divergent par rank ; optimiseur/EMA/save/LoRA également cassés | Haute (6 points d'intégration par name-match, §1.3) | Run entier invalide, symptôme = "ça n'apprend pas" découvert à l'éval | Au setup : assert chaque param trainable ∈ (FSDP-géré ∪ mem_params) ; à step 0 et 50 : all-gather du hash des params du décodeur, assert identiques ; étendre le grad-probe existant (distillation.py l. 1256-1267) aux poids du décodeur |
| 2 | **MEM_ZERO_ALL ne coupe pas le chemin virtuel** → les arms memzero de TOUTE la campagne lisent la mémoire | Moyenne | Maximal : campagne 24 cellules à refaire, découvert à l'audit | Run court `MEM_ZERO_ALL=1` sur ckpt entraîné vs `memory_layers: []` : latents bit-identiques (harness stockaf existant) — À METTRE DANS LE SELFTEST STANDARD |
| 3 | **Recompute-divergence du checkpoint** (décodage depuis l'état live, ou nouveau state mutable non snapshot) → gradients faux silencieux | Moyenne-haute (le contrat a déjà mordu 3 fois : ttt_state, ema, fifo) | Entraînement qui "marche" mais converge mal ; indiscernable d'un mauvais hyperparam | Selftest dédié : forward+backward checkpointing ON vs OFF sur 2 chunks, comparer les grads du décodeur (tolérance bf16-aware) ; déjà le pattern du twin-selftest |
| 4 | **Byte-identity cassée à l'init** (concat naïf, ou merge à g=0 non skippé) | Moyenne | Toutes les comparaisons vs baseline confondues | Twin selftest byte-identical AVANT launch (protocole déjà standard) — il attrapera ce bug SI le skip structurel est le chemin testé |
| 5 | **Branche sink==0 sans tokens virtuels** → les rollouts cache-dropout (30% des rollouts !) tournent sans le canal KV, exactement là où la mémoire doit porter l'ancre | Haute (précédent historique : cette branche a déjà été oubliée, l. 443) | Co-training silencieusement amputé ; explique un résultat nul sans le signaler | Compteur/log wandb `virt_attn_mass` par rollout ; assert > 0 sur un rollout forcé `cache_dropout_prob=1.0` au selftest |
| 6 | **Curriculum switch inopérant** (`switch_choices` jamais atteint, ou segment B < fenêtre → A jamais évincé → teacher==student → RSD distille du vide) | Moyenne | Run de training entier pour un delta nul "inexpliqué" | Assert au setup : positions de switch garantissent éviction complète de la queue A (gap ≥ local_attn + marge) dans `streaming_max_length` ; log de ‖teacher − student‖ sur les chunks post-retour (doit être >> 0 aux premiers steps) |
| 7 | **Kwargs non threadés dans un call-site** (recache, interactive, un des 3+ sites de streaming_training, eval pipelines) → mémoire absente précisément sur les chunks de résurrection | Moyenne (répétition du bug historique "switch inference bypassait la mémoire") | Évals void, découvert des semaines après | Compteur global de décodages par forward ; assert ≥ 1 décodage par chunk post-éviction dans CHAQUE pipeline au selftest ; grep-audit des `self.generator(` |
| 8 | **Super-sink tardif** : normes k/v virtuels rampent pendant le training long → masse d'attention excessive → flicker/dissolution à step 1000+ (pattern c2) | Haute (c'est le mode de dégénérescence par défaut de la famille) | 2-3 semaines de GPU + autopsie | Tripwire wandb dès step 0 : masse captée par bloc (alerte >30%), RMS(k_virt), RMS(v_virt), g par bloc ; cap RMS sur v_virt dès la v1 |
| 9 | **Fuite de B au recache du retour** (état non géré au switch → les tokens virtuels injectent B pendant la reconstruction du cache A) | Moyenne | Delta re-ID amputé, diagnostic difficile | Probe d'inférence $0 sur le premier ckpt : MEM_EPISODIC on/off sur le canal KV (hooks déjà en place, harness 24 cellules) |
| 10 | **Collision de bandes RoPE** en début de vidéo ou sous non-rope_enhance | Basse-moyenne | Dégradation de qualité subtile, mal attribuée | Assert bandes disjointes à chaque forward (cheap) ; étendre la ligne de debug `[rope]` existante (l. 383) aux tokens virtuels |

Ordre de traitement recommandé : 1, 2, 3 (les trois silencieux-et-chers) sont à verrouiller par des asserts/selftests AVANT d'écrire la première ligne du décodeur ; 5, 6, 7 au moment du câblage ; 8, 9, 10 par instrumentation wandb dès le premier run.

---

## 5. Probes $0 à exécuter AVANT l'entraînement (rappel consolidé)

1. **Oracle-3-blocs** (§3.3) — le seul trou de gating restant ; décide [5,16,27] vs plus large.
2. **Oracle-position-sink** (même hook) — vérifie que le delta d'oracle1 survit quand la frame A est lue depuis une position sink-adjacente plutôt que le dernier slot de fenêtre.
3. **MEM_EPISODIC sur canal KV** (§4.9) — dès le premier ckpt, avant de payer l'écriture à deux clés.

## 6. Ce que ce design NE règle PAS (à dire honnêtement dans le papier)
- Le rayon récursif pixel (§2.1-3) est commun à toutes les mémoires actives : le gate + les jauges le bornent, ils ne l'éliminent pas.
- Le teacher RSD est le même modèle que le student (self-distillation à contexte privilégié) : cible mouvante à mesure que la LoRA bouge ; les précédents publiés (ARL/ISPA/OPSD-V) l'assument, citer et assumer pareil (option : teacher gelé au ckpt de départ, à trancher au volet (c)).
- AdaState reste le rebuttal n°1 ("même sink, autre updater") : la comparaison AdaState-like (état 1-frame dénoisé, sans TTT) est une baseline à budgéter dans le volet éval, pas un bug de design.
