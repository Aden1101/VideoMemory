# 3. Pipeline d'entraînement et modifications

Cette section décrit comment le modèle (générateur causal Wan 1.3B + mémoire fast-weight) est entraîné, puis l'ensemble des modifications apportées au pipeline au fil des expériences. Le socle est la recette Anchor-Forcing / Self-Forcing : une distillation de distribution (DMD) appliquée non pas à des vidéos réelles, mais aux propres rollouts autorégressifs du modèle (entraînement « data-free » : aucune vidéo réelle n'entre dans la boucle, seulement des prompts texte). Nos modifications se répartissent en trois couches : (i) le co-entraînement de la mémoire dans cette boucle (LoRA + paramètres mémoire, BPTT inter-chunks), (ii) une série d'objectifs « garde-fous » (série e) construits puis abandonnés, et (iii) trois runs de référence c1/c2/c3 qui reviennent au DMD pur en modifiant l'architecture d'écriture et la stabilité de l'optimisation.

## 3.1 La distillation DMD (model/dmd.py)

### 3.1.1 Principe : deux scores, un gradient KL

La classe `DMD` (`model/dmd.py:18`) implémente Distribution Matching Distillation (DMD, eq. 7 de arXiv:2311.18828). Trois réseaux interviennent :

- le **générateur** (étudiant) : le Wan causal few-step, celui qu'on entraîne ;
- le **teacher** `real_score` : le Wan 2.1 T2V 1.3B original, **figé** (`model/base.py:36-37`, `requires_grad_(False)`), qui représente la distribution des « vraies » vidéos ;
- le **critic** `fake_score` : une copie **entraînable** du même Wan (`model/base.py:39`), entraînée en parallèle (loss de débruitage flow-matching, `critic_loss`, `model/dmd.py:299`) pour représenter la distribution courante des sorties du générateur.

Le cœur est `_compute_kl_grad` (`model/dmd.py:64-159`). On bruite la sortie du générateur à un timestep aléatoire, puis les deux scores prédisent chacun le débruité :

```python
# model/dmd.py:145-153
grad = (pred_fake_image - pred_real_image)
if normalization:
    p_real = (estimated_clean_image_or_video - pred_real_image)
    normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
    grad = grad / normalizer
```

`pred_real_image` est calculé avec classifier-free guidance (branches conditionnelle et inconditionnelle, `model/dmd.py:106-120`, `real_guidance_scale: 3.0`). La différence `fake - real` est exactement le gradient de la KL inverse entre la distribution du générateur et celle du teacher : elle pointe depuis « ce que le générateur produit typiquement » vers « ce que le teacher trouverait plus probable ».

### 3.1.2 La MSE de substitution et le gradient_mask

Ce gradient est calculé entièrement sous `torch.no_grad()` (`model/dmd.py:185`). Pour l'injecter dans l'autograd, `compute_distribution_matching_loss` (`model/dmd.py:161-227`) utilise l'astuce classique de la MSE de substitution :

```python
# model/dmd.py:221-223
dmd_loss = 0.5 * F.mse_loss(
    original_latent.double()[gradient_mask],
    (original_latent.double() - grad.double()).detach()[gradient_mask],
    reduction="mean")
```

La cible `(x - grad).detach()` est constante ; la dérivée de `0.5*||x - (x - grad)||²` par rapport à `x` vaut exactement `grad`. Le backward propage donc le gradient KL vers x, puis à travers tout le rollout du générateur. Le `gradient_mask` restreint cette loss aux **nouvelles frames** du chunk courant : les frames de contexte recopiées du chunk précédent (overlap) sont exclues (construction du masque en 3.2). Le timestep de bruitage est échantillonné dans une plage pilotée par le nombre d'étapes de débruitage réellement effectuées par l'étudiant (`denoised_timestep_from/to`, `model/dmd.py:187-203`), avec le `timestep_shift: 5.0` de Wan.

### 3.1.3 Point crucial : le juge est bidirectionnel et aveugle au cache

Le teacher et le critic sont instanciés avec `is_causal=False` (`model/base.py:36-39`) : ce sont des `WanModel` bidirectionnels standards (`utils/wan_wrapper.py:165`), pas des `CausalWanModel`. Surtout, leurs appels dans `_compute_kl_grad` ne reçoivent **ni KV-cache, ni sink, ni état mémoire** (`model/dmd.py:85-116` : seulement le latent bruité, le texte et le timestep). Le juge DMD ne voit donc à chaque étape qu'une fenêtre isolée de 21 frames latentes, sans aucun accès à l'historique que le générateur, lui, transporte dans son cache et sa mémoire.

Cette asymétrie structure toutes les conclusions du projet : l'objectif ne récompense la cohérence à long terme (identité du sujet à la frame 200 vs la frame 0) qu'indirectement, via ce que le contexte injecté rend « probable » pour la fenêtre courante. Le gradient traverse bien le KV-cache et la mémoire du générateur (le contexte est en graphe), donc une mémoire co-entraînée reçoit un signal (c'est le mécanisme VideoSSM) ; mais rien dans la loss ne compare la fenêtre courante au début de la vidéo. Toute solution qui rend la fenêtre localement plausible, y compris en oubliant le sujet, est un optimum valide.

## 3.2 Le self-rollout streaming (model/streaming_training.py)

### 3.2.1 Rollout par chunks, contexte = sortie du modèle

`StreamingTrainingModel` (`model/streaming_training.py:25`) enveloppe le modèle DMD et gère la génération autorégressive pendant l'entraînement. Les constantes clés (`model/streaming_training.py:53-56`) : `chunk_size=21` frames latentes (la fenêtre de loss), `min_new_frame=18` nouvelles frames par pas, `streaming_max_length=300` (~240 frames vidéo générées par séquence, soit des rollouts de plusieurs chunks bien au-delà de la fenêtre d'attention).

`generate_next_chunk` (`model/streaming_training.py:624-833`) réalise un pas de rollout :

1. tirage (synchronisé entre ranks, `model/streaming_training.py:663-676`) du nombre de nouvelles frames (en pratique 18), l'overlap étant `21 - 18 = 3` frames reprises du chunk précédent ;
2. génération des nouvelles frames par `_generate_chunk` -> `generate_chunk_with_cache` (pipeline), à partir de bruit pur, conditionnées uniquement par le KV-cache (sink + fenêtre locale), la mémoire TTT et le texte : **le contexte est la propre sortie du modèle**, jamais une vidéo réelle ;
3. concaténation `[previous_frames, generated_new_frames]` et construction du `gradient_mask` : seules les nouvelles frames portent le gradient (`model/streaming_training.py:792-802`) ;
4. sauvegarde des 21 dernières frames comme contexte du prochain pas (`model/streaming_training.py:811`).

Côté pipeline, `generate_chunk_with_cache` (`pipeline/streaming_training.py:173-400`) découpe le chunk en blocs de 3 frames et applique le débruitage few-step (`denoising_step_list: [1000, 750, 500, 250]`) avec sortie à une étape aléatoire partagée entre ranks. Seule la dernière étape de débruitage du bloc est en graphe (`pipeline/streaming_training.py:308-338`) ; les étapes intermédiaires sont no-grad. Après chaque bloc, une passe supplémentaire sur le bloc débruité (re-bruité au `context_noise`) met à jour le KV-cache et **écrit dans la mémoire** (`ttt_update=True`, `pipeline/streaming_training.py:359-374`), alors que les passes de débruitage ne font que **lire** (`ttt_update=False`, lignes 290 et 331).

### 3.2.2 Text-dropout

Un ingrédient de la recette AF conservé dans tous nos runs : avec probabilité `text_dropout_prob: 0.3`, un chunk (jamais le chunk 0) est généré sous texte nul, cache cross-attention désactivé, tandis que le teacher et le critic gardent le vrai prompt (`model/streaming_training.py:298-310`). Le générateur doit alors porter le sujet par son seul historique (cache + mémoire), ce qui donne à la mémoire une occasion de « gagner » un rôle identitaire dans l'objectif DMD.

## 3.3 Le co-entraînement : LoRA + mémoire full-rank

### 3.3.1 Deux populations de paramètres

Le backbone Wan est gelé ; deux ensembles s'entraînent, avec des learning rates séparés :

- **LoRA rank-256** (`adapter: {rank: 256, alpha: 256}` dans les configs) sur toutes les couches linéaires des `CausalWanAttentionBlock` du générateur (et du critic), configuré dans `_configure_lora_for_model` (`trainer/distillation.py:1656-1707`), qui **exclut explicitement la branche mémoire** des cibles LoRA. Le flag `lora_unfreeze_from_block: 0` (`trainer/distillation.py:296-317`) laisse la LoRA de tous les blocs entraînable (une valeur K > 0 gèlerait la LoRA des blocs < K pour économiser la VRAM).
- **La mémoire, full-rank** : PEFT gèle tout paramètre non-LoRA, donc les paramètres mémoire sont explicitement dégelés après coup (`trainer/distillation.py:292-294`). Ils comprennent les projections de lecture/écriture (`qkv`, `proj`, `lrs`, `wan/modules/video_memory.py:278-302`) et la graine du fast-weight (`w0/w1/w2`, lignes 297-299).

Les modules `VideoMemoryLayer` sont **exclus de FSDP** (`ignored_modules`, `trainer/distillation.py:319-340`) : leur état fast-weight est muté à chaque chunk, ce qui est incompatible avec le sharding. Ils sont donc répliqués par rank (init broadcastée depuis le rank 0, lignes 328-332), et leurs gradients sont moyennés à la main avant chaque pas d'optimiseur :

```python
# trainer/distillation.py:1474-1477
for _n, _p in self.model.generator.named_parameters():
    if "memory" in _n and _p.grad is not None:
        dist.all_reduce(_p.grad, op=dist.ReduceOp.AVG)
```

L'optimiseur AdamW a deux groupes (`trainer/distillation.py:414-427`) : les adaptateurs LoRA à `lr: 1e-5` et les paramètres mémoire à `lr_mem` (1e-4 en c1/c2, 5e-5 en c3). Détail qui aura son importance en c3 : `clip_grad_norm_` étant une méthode FSDP, elle **ignore** les paramètres FSDP-ignored ; les gradients mémoire étaient donc non clippés en c1/c2 (commentaire `trainer/distillation.py:428-432`).

### 3.3.2 Entraîner l'écriture : train_write et la fenêtre BPTT

Avec `train_write: true` (`pipeline/streaming_training.py:68-77`), la passe d'écriture s'exécute sous `enable_grad` et l'état fast-weight garde son graphe (`detach_state=False` sur les `VideoMemoryLayer`) : la règle d'écriture (learning rates par tête via `lrs`, update Muon `zeropower_via_newtonschulz5`, `wan/modules/video_memory.py:92-189`) devient différentiable et reçoit un gradient dès qu'une lecture ultérieure influence la loss.

Mais un gradient intra-chunk ne suffit pas : l'intérêt d'une mémoire est de porter l'information **entre** chunks. D'où la BPTT tronquée inter-chunks (`trainer/distillation.py:1178-1277`) :

- à chaque pas générateur, un horizon `K ~ U{bptt_window_min .. bptt_window}` = U{4..14} est tiré (synchronisé entre ranks, lignes 1181-1194) : la fenêtre supervise ainsi des horizons variés, jusqu'au-delà de l'éviction KV (fenêtre locale de 12 frames + sink 3) ;
- les `K - W` premiers chunks sont générés **sans gradient** (état profond, peu coûteux), seuls les `W = grad_tail_W = 1` derniers portent le graphe (lignes 1196-1228). `W=1` évite la recomputation croisée gradient-checkpointing / éviction KV, et maintient la VRAM au niveau prouvé d'un seul chunk en graphe ;
- `detach_state` n'est appliqué qu'au début de fenêtre et `keep_last_write_grad` maintient le graphe de l'écriture sortante quand l'état doit porter le gradient vers le chunk suivant (ligne 1228 ; logique côté pipeline `pipeline/streaming_training.py:354-374`) ;
- **seed step** : la graine `w0/w1/w2` n'est atteignable que via l'initialisation de l'état au chunk 0. Tous les `train_seed_every: 4` pas générateur, le chunk 0 est aussi mis en graphe et backwardé chunk par chunk (lignes 1206-1245), de sorte que la graine reçoive un gradient. Un probe (`trainer/distillation.py:1257-1267`) vérifie aux premiers steps que les gradients WRITE (`lrs`, `decay`), READ (`qkv`, `proj`) et SEED (`w0`) sont non nuls.

Les losses des chunks en graphe sont moyennées et un **unique backward** couvre la fenêtre (ligne 1253), ce qui entraîne l'écriture à travers les frontières de chunks sans fuite de graphe.

### 3.3.3 Snapshot/restore de l'état TTT autour du backward

Une subtilité imposée par le gradient checkpointing : la recomputation au backward **ré-exécute la passe d'écriture**, donc mute l'état fast-weight, le buffer EMA d'injection (e4) et la FIFO d'éviction (c2). Sans précaution, l'état entrant dans le chunk suivant serait celui d'une recomputation, pas du forward. Le trainer encadre donc chaque backward :

```python
# trainer/distillation.py:1309-1313
_ttt_snap = self._snapshot_ttt_state()
try:
    scaled_generator_loss.backward()
finally:
    self._restore_ttt_state(_ttt_snap)
```

`_snapshot_ttt_state` (`trainer/distillation.py:1869-1903`) copie sur CPU l'état (`state`), l'EMA et la FIFO de chaque `TTTBlockCache`, et `_restore_ttt_state` les remet en place après le backward. Le même mécanisme, étendu à tous les caches (KV, cross-attn, RNG), sert au jumeau contrefactuel décrit ci-dessous (`_twin_snapshot`/`_twin_restore`, `model/streaming_training.py:558-622`), validé par un selftest bit-exact (`E3_TWIN_SELFTEST=1`, ligne 746-749).

## 3.4 Les objectifs « garde-fous » de la série e

### 3.4.1 Le jumeau mem-OFF contrefactuel

Le problème récurrent des runs e1/e2 : impossible d'attribuer proprement un effet à la mémoire, car toute comparaison à un run séparé confond l'injection avec la dérive LoRA et le bruit. La série e3 introduit donc un **jumeau contrefactuel** : avant de générer le dernier chunk de la fenêtre, l'état complet est snapshotté, le même chunk est généré sous `MEM_ZERO_ALL=1` (injection mémoire annulée), même bruit, mêmes flux RNG, puis tout est restauré (`model/streaming_training.py:706-735`). Le seul écart entre `chunk` et `twin_new_frames` est l'injection de la mémoire pendant ce chunk. Le jumeau est no-grad : rien ne peut être « gagné » en dégradant la baseline.

### 3.4.2 Les termes de la loss

`_compute_advantage_loss` (`model/streaming_training.py:939-1070`) combine, autour de la loss DMD (ajout dans `compute_generator_loss`, `model/streaming_training.py:1151-1155`) :

```python
# model/streaming_training.py:1053-1058
loss = (self.dnh_weight * dnh          # do-no-harm : relu(cos_CF - cos_ON - deadband)
        + self.act_weight * act        # activity floor : relu(act_floor - dose_AC)
        + self.mm_rel_weight * mm_rel  # motion-match : (log m_ON - log m_CF)^2
        + self.mm_ext_weight * mm_ext  # plancher absolu vs la courbe stock-AF
        + self.flick_weight * flick    # anti-flicker : hinge sur la diff temporelle seconde
        + self.adv_loss_weight * (-adv))  # bonus clampe, defaut OFF
```

- **dnh** (do-no-harm, ligne 1046) : hinge sur l'écart de cosinus DINO à l'ancre identité du chunk 0 (`_set_id_anchor`, lignes 867-878) entre le chunk réel (`cos_on`) et le jumeau (`cos_cf`). Punit seulement quand l'injection **dégrade** l'identité au-delà d'un deadband ; rien ne paie pour la sur-similarité (le gel n'a plus de payeur).
- **act** (activity floor, lignes 1004-1016) : la « dose AC » de l'injection (RMS de `chunk - twin` après retrait de la moyenne par canal, pour qu'un biais DC/style ne compte pas) doit dépasser un plancher. C'est un `relu` : gradient nul dès que satisfait, un plancher, jamais un maximiseur.
- **mm_rel / mm_ext** (lignes 997-1002) : le mouvement latent du chunk réel doit égaler celui du jumeau (deux-côtés) et ne pas descendre sous une fraction d'une courbe de mouvement stock-AF pré-mesurée (`motion_target_path`), ce qui interdit le co-gel des deux bras.
- **flick** (e4, lignes 1023-1032) : hinge sur la rugosité de différence temporelle seconde (D2) que l'injection ajoute au-delà du jumeau, un « tripwire » anti-scintillement.
- **e4** ajoute aussi la statistique de mouvement **lag-2** (lignes 987-994 : l'alternance frame à frame contribue ~0 à un delta lag-2, le shimmer ne peut plus contrefaire du mouvement), le retrait de moyenne **temporelle** dans la dose (lignes 1008-1015 : un « papier peint » statique gagne une dose nulle) et un lissage EMA temporel de l'injection côté modèle (buffer `ema` du `TTTBlockCache`, snapshotté en 3.3.3).

### 3.4.3 Pourquoi c'est devenu du whack-a-mole

L'historique des runs e est une leçon sur le Goodhart structurel. Chaque garde-fou fermait un mode de dégénérescence et en révélait un autre, car aucun terme ne définit ce que la mémoire **devrait** faire ; ils ne définissent que ce qu'elle ne doit pas faire :

1. la loss d'identité (e1) paie pour la self-similarité -> le modèle **gèle** (vidéo statique, identité parfaite triviale) ;
2. le retrait DC + normalisation d'amplitude (e2_amp) épingle une dose fixe -> l'injection hors-distribution s'accumule et la vidéo **se dissout** à 240 frames ;
3. les planchers de mouvement (e3) interdisent le gel -> le modèle produit du **flicker** (le shimmer haute fréquence compte comme mouvement pour une statistique lag-1) ;
4. les termes anti-flicker lag-2/D2 (e4) ferment ce canal -> le gel revient plus tard dans l'entraînement (e3 propre à 250 steps, gelé à 500), et le canal résiduel identifié était le « papier peint », déjà l'escalade suivante.

Le triptyque gel / dissolution / flicker balaie l'espace des fréquences temporelles : chaque métrique-ruler a un angle mort, et l'ensemble faisable de l'optimisation converge vers « les angles morts des rulers » union « l'inertie ». La synthèse multi-agents de juillet 2026 a conclu que ce jeu ne se gagne pas en ajoutant des contraintes : d'où l'abandon complet des garde-fous pour la série c. Le code reste en place mais **tous les poids sont à 0** dans c1/c2/c3 (`_e3_on` est faux, `model/streaming_training.py:121-124` : ni jumeau ni décodage DINO ne s'exécutent, coût nul, chemin octet-identique).

## 3.5 Les runs c1, c2, c3

### 3.5.1 c1_cotrain : le co-entraînement DMD pur

c1 (`reference/runs/c1_cotrain/c1_cotrain.yaml`) est le premier run « honnête » : DMD pur (tous les garde-fous à 0), substrat sain de 248k prompts (VidProM filtré, plus le set 8-prompts sur-appris des runs e), LoRA entièrement dégelée (`lora_unfreeze_from_block: 0`), mémoire co-entraînée (`lr_mem: 1e-4`, `train_write: true`), BPTT U{4..14} avec `grad_tail_W: 1` et seed step tous les 4, injection déjà domestiquée par construction (`rel_norm`, `dc_remove`, `bounded_gate`, `gate_init: 0.1`). C'est la reproduction du régime VideoSSM : une mémoire globale co-entraînée reçoit-elle un gradient DMD fenêtré utile ?

Résultat (notes `c1-cotrain-step250-result.md`) : pas de gel (l'attracteur e3/e4 est évité, motion saine), mais l'injection est nettement **négative sur l'identité** à 250 steps (id_cos tail-mean mem-ON vs mem-OFF : -0.123, 6/6 prompts), se neutralise vers do-no-harm à 500 (-0.019), puis **redevient nocive** à 1250 (-0.119, et l'injection tue le mouvement, DD 0.23 vs 0.68). Diagnostic : l'écriture de c1 est redondante par construction, la mémoire écrit le chunk **courant**, dont les k/v sont déjà dans le KV-cache ; elle n'a aucun rôle marginal, et le co-entraînement apprend surtout à s'appuyer sur une injection nuisible.

### 3.5.2 c2_eviction : donner un rôle à la mémoire

c2 (`reference/runs/c2_eviction/c2_eviction.yaml`) corrige ce défaut par deux mécanismes :

**Eviction-feed** (`eviction_feed: true`). La mémoire n'écrit plus le chunk courant mais le chunk qui vient de **sortir de la fenêtre d'attention** : chaque bloc mémoire garde une FIFO des chunks passés (`TTTBlockCache.fifo`, `wan/modules/video_memory.py:54-66`) ; sur la passe d'écriture, le chunk évincé est poppé hors du bloc checkpointé (`wan/modules/causal_model.py:1343-1384`, avec le lag exact `window_chunks = ceil(local_attn_size / frames_per_chunk) = 4` vérifié par selftest), et passé comme `x_write` au module mémoire (`wan/modules/video_memory.py:455-472`), la lecture restant interrogée par le chunk courant. La mémoire détient ainsi le **complément du cache** (ce que le sink + la fenêtre ont perdu), à la façon VideoSSM/AHN.

**Cache-dropout co-train** (`cache_dropout_prob: 0.3`, `pipeline/streaming_training.py:79-145`). À chaque rollout, avec probabilité 0.3 (tirage synchronisé entre ranks), le sink d'attention est réduit à l'exécution de 3 à 0 frames (`_set_runtime_sink`, lignes 114-123) pour le rollout **entier**, mémoire active. Le DMD récompense alors directement la mémoire pour porter l'ancre que le sink ne fournit plus : c'est le gradient qui casse la redondance « le sink fait déjà le travail ». La décision est prise dans `clear_kv_cache()` en début de rollout (ligne 502) et reste stable pendant tous les chunks et leurs backwards (la recomputation du checkpointing doit voir le même sink que le forward).

Résultat : nette amélioration au début (sous starvation du sink, gap identité -0.055 à 250 vs -0.102 pour c1, motion réparée), convergence apparente vers do-no-harm jusqu'à 750 (-0.040), puis **bascule en flicker** à 1000+ (-0.253 puis -0.138, SC 0.95 -> 0.80). L'autopsie des tfevents a montré que ce flicker n'est pas le mur data-free lui-même mais une **pathologie d'optimisation réparable** : les paramètres lents de la mémoire (qkv -> RMS de lecture x8 ; gate 0.30 -> 1.76) rampent de façon monotone sans aucun amortissement (lr constant, EMA désactivée en mode LoRA, weight decay faible, gradients mémoire non clippés), jusqu'à ce que l'amplitude d'injection franchisse un seuil de cohérence vers le step 1000. Un probe à coût nul l'a confirmé : `MEM_STREAM_SCALE=0.5` appliqué au checkpoint 1000 réduit de moitié le dégât.

### 3.5.3 c3_stable : le pack stabilité

c3 (`reference/runs/c3_stable/c3_stable.yaml`) garde l'architecture de c2 (eviction-feed, cache-dropout, DMD pur) et ajoute uniquement des stabilisateurs d'optimisation, chacun ciblant la rampe des paramètres lents :

1. `mem_stream_scale: 0.5` : plafond dur de l'injection à 0.5x le RMS du résiduel local (le frein d'amplitude direct) ;
2. `lr_mem: 5e-5` (au lieu de 1e-4) : moteur de la rampe divisé par deux ;
3. `weight_decay: 0.05` (au lieu de 0.01) ;
4. **EMA des poids réactivée en mode LoRA**, trainable-only : `EMA_FSDP` (`utils/distributed.py:93-134`) ne suit que les paramètres `requires_grad` (adaptateurs LoRA + mémoire FSDP-ignored), shadow CPU float32 mis à jour après chaque `optimizer.step()` (`trainer/distillation.py:1488-1489`), sauvegardé/restauré au resume (`trainer/distillation.py:274-278, 400-408`). Read-only : ne change pas la trajectoire d'entraînement, mais l'éval doit charger le shadow (`use_ema: true`) ;
5. décroissance cosinus des deux lr (`_lr_sched_step`, `trainer/distillation.py:1395-1408`) : de `lr_decay_start: 400` jusqu'à `max_iters`, planchers 0.1x (LoRA) et 0.05x (mémoire) ;
6. `max_grad_norm_mem: 1.0` : clip explicite des gradients mémoire (`_clip_mem_grads`, `trainer/distillation.py:1388-1393`), qui comble le trou du clip FSDP ;
7. anneal du cache-dropout 0.3 -> 0.1 sur 800 steps (`_effective_cache_dropout_prob`, `pipeline/streaming_training.py:105-112`, step poussé par le trainer ligne 1436-1438) : retire l'essentiel de la récompense sink-0 une fois l'injection forte, en gardant un rôle résiduel ;
8. `use_decay: true` : forget gated-delta sur le fast weight (couverture secondaire, la norme d'état était déjà contrôlée par renorm).

Chaque mécanisme a été validé par un selftest (`reference/runs/c3_stable/selftest.py`) : pack désactivé = sortie octet-identique à c2, chaque knob vérifié actif quand allumé.

### 3.5.4 Tableau comparatif des runs

| Knob | c1_cotrain | c2_eviction | c3_stable |
|---|---|---|---|
| Objectif | DMD pur (garde-fous 0) | DMD pur | DMD pur |
| Substrat | 248k prompts VidProM | idem | idem |
| LoRA | rank 256, tout dégelé, lr 1e-5 | idem | idem + cosine decay (plancher 0.1x) |
| `lr_mem` | 1e-4 | 1e-4 | 5e-5 + cosine (plancher 0.05x) |
| `train_write` / BPTT | true / U{4..14}, W=1, seed 4 | idem | idem |
| `text_dropout_prob` | 0.3 | 0.3 | 0.3 |
| `eviction_feed` | non (écrit le chunk courant) | **oui** (écrit le chunk évincé) | oui |
| `cache_dropout_prob` | 0 | **0.3** (sink 3 -> 0) | 0.3 -> 0.1 (anneal 800 steps) |
| `mem_stream_scale` | 1.0 | 1.0 | **0.5** |
| `weight_decay` | 0.01 | 0.01 | **0.05** |
| `max_grad_norm_mem` | inactif (non clippé) | inactif | **1.0** |
| EMA poids | inopérante en mode LoRA | idem | **trainable-only, 0.99 dès 250** |
| `use_decay` (fast weight) | false | false | **true** |
| Résultat identité (gap mem-ON - mem-OFF, tail id_cos, sink starvation) | -0.102 (et motion gelée) | -0.055 -> -0.040 (250-750) puis flicker -0.253/-0.138 (1000+) | en cours (question : tenir do-no-harm + dynamique jusqu'à 2000) |

## 3.6 Bilan de la partie entraînement

Le pipeline final entraîne donc, par distillation DMD data-free sur les propres rollouts du modèle : une LoRA rank-256 du backbone, et une mémoire fast-weight full-rank (lecture, écriture différentiable via BPTT tronquée inter-chunks, graine). Les modifications décisives ne sont pas venues d'objectifs supplémentaires, la série e ayant démontré empiriquement que des garde-fous métriques sur un objectif aveugle à l'historique se contournent l'un après l'autre, mais de deux leviers plus structurels : donner à la mémoire une information que le cache n'a plus (eviction-feed) et un régime d'entraînement où elle est le seul porteur de l'ancre (cache-dropout), puis stabiliser l'optimisation des paramètres lents (c3). La limite de fond demeure celle du 3.1.3 : le juge DMD étant bidirectionnel, fenêtré et aveugle au cache, rien dans l'objectif ne définit positivement « retenir le sujet » ; les runs c mesurent précisément ce qu'une mémoire peut apprendre malgré cela.
