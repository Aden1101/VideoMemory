# localplan_D : stress-test adversarial du pivot "frame synthétique recachée" (post NO-GO oracle3b)

Rédigé le 2026-07-23, lecture seule du code (rien exécuté). Fichiers audités :
`pipeline/switch_causal_inference.py` (recache l.56-128, hooks oracle l.226-283),
`pipeline/streaming_switch_training.py` (recache l.304-383, boucle chunk l.100-302),
`wan/modules/causal_model.py` (attention+cache l.108-472, `_apply_cache_updates` l.1140-1213,
`_forward_inference` l.1215-1461, eviction-feed l.1350-1385), `wan/modules/video_memory.py`
(état w0/w1/w2, swiglu), `model/streaming_training.py` (`_clear_cache_gradients` l.1191-1213,
BPTT), `trainer/distillation.py` (bptt_window/grad_tail_W l.1178-1247),
`pipeline/causal_inference.py` (allocation cache l.121-133, max_attention_size l.346-357).
Contexte : plan_final_sink_virtuel.md (RÉSULTAT oracle3b : +0.023 sur +0.220, NO-GO 3-blocs),
localplan_A/B, virtual_sink_preemption_check.md.

---

## 0. Verdict exécutif

Le pivot est la bonne direction MAIS pas sous la forme énoncée. Trois découvertes de code
changent le plan :

1. **La "position sink persistante" ne demande AUCUNE chirurgie d'arithmétique** : après un
   switch non-global-sink, le recache reconstruit sink + fenêtre depuis les 12 frames passées
   (le "sink" post-switch n'est PAS l'ancre vidéo, c'est les 3 premières frames recachées).
   Mettre la frame synthétique en PREMIÈRE position de `frames_override` la place dans le sink,
   où elle persiste jusqu'au prochain switch, à coût de code nul. Le slot dédié "+1 frame de
   sink" (6 sites d'off-by-one, détail §1.3) est INUTILE en v1.
2. **La croyance de localplan_B "le kv_cache est un buffer détaché, la concat est le seul
   chemin de gradient" est FAUSSE dans ce codebase** : les écritures in-place sous enable_grad
   promeuvent le buffer en tenseur de graphe (CopySlices), le BPTT inter-chunk existant
   (bptt_window/grad_tail_W) passe DÉJÀ par le cache physique, et `_clear_cache_gradients`
   (model/streaming_training.py l.1191-1213) existe précisément pour le détacher avant le
   critic. Le gradient RSD PEUT traverser un recache 1-frame, pour ~0.15-0.5 GB (§3).
3. **Le vrai trou de gating du pivot n'est pas l'arithmétique, c'est la courbe de tolérance à
   l'imperfection** : oracle1 splice la frame EXACTE du modèle ; personne n'a mesuré le delta
   avec une frame A DÉGRADÉE. Si +0.22 s'effondre au premier flou, le décodeur devra être
   quasi parfait et le pari change de catégorie. Probe $0 "oracle-dose" AVANT toute ligne de
   code (§5), même logique que le probe oracle3b qui vient de sauver 6 jours de dev.

Recommandation : **(a') en deux étages** : v1 = splice événementiel de la frame décodée via
`frames_override` EXISTANT (zéro modification du cache, décodeur entraîné par la cible
auxiliaire de reconstruction, zero-shot côté backbone, exactement le mécanisme d'oracle1) ;
v2 (si v1 positif mais sous-oracle) = hybride "phantom forward + concat" qui garde les 30
blocs gratuits ET le gradient RSD propre. Le design (b) pur (30 têtes k/v abstraites) est
DOMINÉ par cet hybride sur tous les axes sauf un (§2.3).

---

## 1. Q1 : faisabilité exacte du recache 1-frame contre le code

### 1.1 Ce que fait réellement le chemin recache

Inférence (`switch_causal_inference.py::_recache_after_switch`, l.56-128) :
- zère k/v de TOUS les blocs (si `not global_sink`), PAS les indices (l.63-65 commentées) ;
- `num_recache_frames = min(local_attn_size, current_start) = 12` ; forward des 12 frames à
  `context_noise` avec `current_start = recache_start_frame * 1560`,
  `sink_recache_after_switch=True` (lève la protection d'écriture du sink),
  `ttt_update=False`, sous `torch.no_grad()` (l.110) ;
- `frames_override` (l.80-87) remplace le CONTENU des 12 frames aux MÊMES positions : c'est le
  hook oracle validé (+0.220 full, +0.224 oracle1 en dernier slot via ORACLE_SPLICE_FRAMES=1).

Arithmétique côté attention (causal_model.py) : les pointeurs n'ayant pas été zérés,
`current_end <= global_end_index` donc `is_recompute=True` (l.250), branche direct_insert
(l.312-345) : `local_end_index` reste plein (18720), `local_start_index = local_end - 12f = 0`,
`sink_recache_after_switch` force `write_start_index = local_start_index = 0` (l.322-323).
**Les 12 frames recachées couvrent exactement le buffer [0, 18720) = [3 sink | 9 fenêtre], et
les pointeurs ne bougent pas (l.1210-1213 : pas d'update sur is_recompute).** Chaque bloc des
30 calcule ses k/v depuis SA représentation de la frame (le x du bloc i dépend des attentions
des blocs 0..i-1) : c'est bien "les 30 blocs alimentés naturellement", l'exigence post-oracle3b,
par construction.

Nota importants :
- l'attention du recache est PLEINE sur le chunk recaché (le block_mask préparé n'est pas
  branché sur ce chemin, commentaire chinois l.362 du pipeline training) : la frame synthétique
  contamine les k/v des 11 autres frames recachées. Ce n'est PAS un bug pour nous : oracle1 est
  passé par exactement ce chemin, son +0.224 INCLUT cette propagation (elle aide probablement).
- le recache tourne avec `ttt_cache` actif (lectures mémoire actives, la "fuite de B" connue) ;
  dans le pivot, la frame synthétique EST le readout : couper la lecture résiduelle legacy
  pendant le recache (mem_readout != residual) supprime la fuite par ce canal.
- Training (`streaming_switch_training.py::_recache_after_switch`, l.304-383) : même mécanique,
  `switch_recache_frames` permet déjà d'injecter des frames externes (concat + `-local_attn:`),
  mais PAS de `frames_override` par slot : le miroir prévu à l'étape B3 reste à écrire (~15
  lignes). UN seul switch supporté à l'entraînement : inchangé, c'est le chantier B3 connu.

### 1.2 Position et RoPE : trois variantes, deux gratuites, une chère

**(P1) Dernier slot de fenêtre (la position d'oracle1, validée 24+/0-)** :
`frames_override = cat([B_tail 11f, frame_décodée 1f])`. Zéro code nouveau (hook
ORACLE_SPLICE_FRAMES=1 déjà en place). RoPE : sous rope_enhance le cache stocke des clés NON
ropées ; à la lecture, le slot est ropé dans la bande fenêtre côté récence (l.363-377), la
géométrie exacte du probe. Durée de vie : le slot roule et s'évince après ~9 frames = 3 chunks.
C'est la SEULE position couverte par une mesure.

**(P2) Slot sink par substitution de contenu (la "position persistante" du pivot, gratuite)** :
`frames_override = cat([frame_décodée 1f, B_tail 11f])`. La frame atterrit au slot sink 0,
ropée à la lecture dans la bande sink [0, _sink_runtime) (l.374-380), distance relative aux
queries <= 21 = régime entraîné. Persistance : le sink n'est jamais roulé (roll copie à partir
de sink_tokens, l.275-278/1164-1167) et la protection is_recompute (l.282/321) interdit sa
réécriture hors recache : **le slot persiste jusqu'au prochain switch, sans aucune chirurgie**.
Coût réel : on sacrifie 1 des 3 frames de sink post-switch (qui ne sont que la queue B promue
sink, pas l'ancre vidéo) ; déplacement d'information identique en nature à celui d'oracle1
(qui sacrifiait 1 slot de fenêtre) et DÉJÀ intégré dans la comptabilité du harness.
NON MESURÉ : c'est exactement le probe "oracle-position-sink" du plan_final, réalisable à $0
en réordonnant `frames_override`. À faire tourner AVANT de choisir P1 vs P2.

**(P3) Slot dédié "+1 frame de sink" (REJET v1)** : agrandir le buffer à 13 frames avec
`_sink_runtime = 4`. Le roll/évict/budget suivent automatiquement (`sink_tokens` est dérivé
partout de `_sink_runtime`), MAIS six sites manuels à off-by-one silencieux : (1) allocation
`kv_cache_size = local_attn * 1560` (causal_inference.py l.121) ; (2) `max_attention_size`
(l.346-357), sinon fenêtre réelle -1 frame = confound baseline ; (3) le recache 12-frames se
désaligne (local_start = 20280 - 18720 = 1560 : il écrirait les frames buffer 1..12, laissant
le slot 0 périmé, ou l'inverse selon le layout) ; (4) la bande rope sink s'élargit et décale
la géométrie des vrais sinks ; (5) eviction-feed `sink_chunks = ceil(_sk/_F)` (l.1362-1363)
compte le slot synthétique comme sink et décale la comptabilité fifo ; (6) écrire ce slot HORS
recache est impossible proprement : dans le forward existant, position d'écriture et rope de la
query sont couplées via `current_start` (write au slot 3 exige
`current_end = global_end - local_end + 4f`, ce qui rope la query à ~frame 20, et vice-versa) ;
il faudrait un kwarg write-override, un 9e site. **Tout ça pour un bénéfice que P2 donne
gratuitement, sauf le rafraîchissement entre les switches, dont la v1 n'a pas besoin.**

### 1.3 Fréquence et coût

- **v1 (événementiel, recommandé)** : décoder UNIQUEMENT au switch (celui du retour au minimum,
  tous les switches si multi-switch). Coût : forward du décodeur (~1560 queries x swiglu
  1536->6144->1536, quelques ms) + ZÉRO forward DiT supplémentaire : le recache 12-frames est
  déjà payé à chaque switch, on ne change que le contenu d'un slot. Entre les switches, aucune
  injection : do-no-harm STRUCTUREL sur le drift simple (les 3 cercles de poison de
  localplan_A §2.1 n'existent qu'aux événements de switch), MEM_ZERO_ALL = pas de splice =
  ablation exacte triviale.
- **Refresh par chunk (le "1x/chunk à l'écriture d'état" du plan_final)** : nécessite P3 + le
  kwarg write-override (§1.2) OU l'hybride concat (§2.2). Coût : 1-frame forward = 1560 tokens
  = 1/3 d'une passe chunk, soit ~+6-7% du forward par chunk (5 passes de 4680 tokens). À ne
  considérer QUE si l'éval segmentée montre que le delta re-ID décroît avec la distance au
  switch (P1) ou que la persistance inter-switch (P2) ne suffit pas.

---

## 2. Q2 : (a) frame-recachée vs (b) k/v abstraits 30-blocs, honnêtement

### 2.1 Ce que (a) a pour lui, contre le code

- Les 30 blocs alimentés GRATUITEMENT et in-distribution par construction : les k/v de chaque
  bloc sortent des qkv gelés du backbone appliqués à une frame latente, la seule interface dont
  on a une PREUVE chiffrée qu'elle porte la résurrection (+0.220/+0.224). Après oracle3b, c'est
  l'argument dominant : (b) n'a AUCUNE mesure de gating pour "k/v appris aux 30 blocs" ; c'est
  structurellement le même trou de gating que celui qui vient de tuer [5,16,27].
- UN décodeur (~2.4M queries + tête 1536->64/patch + unpatchify vers 16x60x104), pas 30 paires
  de têtes (~142M pour (b) à 2x1536² par bloc, avec 30 gates LSE, 30 tripwires, 30 chances de
  starvation).
- **La cible auxiliaire n'existe QUE dans (a)** et c'est le jackpot d'entraînement : à CHAQUE
  écriture (éviction), décoder immédiatement et régresser la frame décodée sur la frame évincée
  (self-generated, data-free, AHN-style) = un autoencodeur à travers le goulot fast-weight,
  entraînable à CHAQUE step SANS rollout, SANS teacher, SANS RSD. Le write reçoit un gradient
  dans le même step (write->read-back intra-step, graphe d'état vivant dans le chunk). Ajouter
  une variante RETARDÉE (reconstruire une frame écrite G chunks avant, cibles gardées côté
  host) pour entraîner la rétention sous interférence de B, ce que la reconstruction immédiate
  ne teste pas. (b) n'a que le signal RSD end-to-end, rare (1-2 résurrections/séquence) et
  bruité, pour 30 têtes.
- La qualité du décodeur se mesure OFFLINE (MSE/DINO frame décodée vs frame évincée), métrique
  intrinsèque sans rollout : le débuggage de (a) est découplé du débuggage du rollout. (b) n'est
  observable qu'à travers le delta re-ID final.

### 2.2 Ce que (a) a contre lui, et l'hybride qui répare

- La frame décodée doit vivre sur le manifold latent du VAE : un code "fonctionnel mais non
  pixel-fidèle" est interdit à (a) alors que (b) peut l'apprendre. Si la courbe de tolérance
  (probe oracle-dose, §5) est raide, (a) exige une reconstruction quasi parfaite.
- v1-(a) écrit physiquement dans le cache : le gradient est possible (§3) mais passe par la
  promotion du buffer en graphe, discipline detach à respecter ; et le splice ne vit qu'aux
  événements de switch.
- **Hybride (v2) "phantom forward + concat"** : le même décodeur produit la frame ; un forward
  1-frame par chunk la passe dans les 30 blocs SANS appliquer `_apply_cache_updates` (skipper
  l'unique call-site l.1453-1455 sous un flag) en capturant les `new_k/new_v` par bloc (k non
  ropés, exactement ce que la concat attend) ; insertion par concat-attention (template
  prompt_cache, bande rope fixe, gate LSE g=0, skip structurel : TOUTES les décisions de
  plan_final §Arbitrages 1-2 se transposent telles quelles). On garde : 30 blocs gratuits,
  interface in-distribution, UN décodeur, cible aux, ET persistance par chunk + gradient RSD
  propre sans toucher au buffer. Coût : +6-7%/chunk.

### 2.3 Verdict motivé

**(b) pur est dominé par l'hybride sur tous les axes sauf l'absence de contrainte manifold** :
mêmes piliers de nouveauté (l'état TTT, l'écriture à l'éviction, l'objectif résurrection et
l'éval re-ID sont indépendants du readout), mais 30 têtes à entraîner sans cible aux, interface
sans aucune évidence de gating, 30x la surface de poison. Sa seule niche de survie : si le
probe oracle-dose montre que le canal exige une fidélité inatteignable par reconstruction,
alors le code appris librement de (b)/hybride-RSD redevient le seul chemin, et il faudra
l'assumer sans plancher zero-shot. Verdict : **(a') deux étages (v1 splice + aux ; v2 hybride
concat)**, (b) pur abandonné.

---

## 3. Q3 : le gradient à travers le recache

Constat de code qui CORRIGE localplan_B §2 : le cache physique n'est pas un pare-feu de
gradient. `_apply_cache_updates` assigne in-place (`cache["k"][:, ws:we] = new_k`, l.1171/1189)
des tenseurs qui, sous enable_grad, portent un grad_fn (ils sortent du checkpoint) : le buffer
requires_grad=False est PROMU (CopySlices) et les lectures suivantes (`temp_k =
kv_cache["k"].clone()`, l.271/318) transportent le graphe. Preuves dans l'arbre : le BPTT
inter-chunk (trainer/distillation.py l.1178-1247) n'a pas d'autre chemin ; et
`_clear_cache_gradients` (model/streaming_training.py l.1191-1213) teste explicitement
`cache_block['k'].requires_grad` avant le critic. Les passes no_grad intermédiaires qui
écrivent d'autres slices du même buffer promu coexistent déjà avec ce mécanisme en production
(clone avant lecture, CopySlices ne sauvegarde pas de valeurs : pas de conflit de version).

Ce qu'il faut changer pour le RSD end-to-end en (a) :
1. Sortir du `torch.no_grad()` la sous-passe de la frame synthétique dans le recache TRAINING
   (l.363 de streaming_switch_training.py) ; découper le recache en 2 forwards : 11 frames B
   sous no_grad, puis 1 frame synthétique sous enable_grad (sinon on paye le graphe sur 18720
   tokens au lieu de 1560). Le pipeline d'inférence (l.110) reste no_grad : rien à changer.
2. Décoder l'état HORS de la région checkpointée et passer la frame en entrée explicite
   (contrat ttt_state, déjà 3 fois mordu : inchangé).
3. Vérifier l'ordonnancement des detach : la loss générateur backward AVANT
   `_clear_cache_gradients` (appelé au critic : OK aujourd'hui) ; aucun `detach_state` entre le
   recache et les R chunks de résurrection ; le flag type keep_last_write_grad pour que la
   passe recache-grad ne soit pas jetée.
4. VRAM du recache 1-frame avec graphe (1560 tokens x 30 blocs) : checkpointé par bloc,
   ~30 x 1560x1536 bf16 = **~145 MB d'inputs retenus** + transitoires recomputés au backward
   (clones temp_k/v ~110 MB/bloc, non retenus) + décodeur/frame < 10 MB : **< 0.5 GB retenu**.
   Sans découpage (12 frames avec graphe) : ~1.7 GB retenus. Les deux tiennent dans le budget.

MAIS : la v1 n'en a PAS BESOIN. La cible auxiliaire entraîne décodeur + write sans aucun
gradient à travers le recache, et oracle1 prouve que le backbone consomme une bonne frame en
zero-shot (+0.224 sans un seul step de co-training). Le RSD-through-recache est l'outil de la
v2 (hybride concat, où il est de toute façon plus propre). Découpler ces deux paris est le
principal dé-risquage de ce plan.

---

## 4. Q4 : préemption AdaState, re-check sur le pivot

AdaState (2605.30349) : état = 1 frame latente, dénoisée CONJOINTEMENT à chaque chunk, son KV
clean écrit en position 0 du cache, récurrence détachée, data-free, AUCUNE loss dédiée à
l'état, aucune éval revisit. Le pivot s'en rapproche, gradient de dangerosité selon la variante :

- **(a2) frame au sink rafraîchie chaque chunk : DANGER MAXIMAL.** La phrase-mécanisme devient
  "une frame latente synthétique dont le KV occupe la position sink, entretenue au fil des
  chunks, data-free" : c'est LEUR phrase. Il ne resterait que l'updater (TTT vs dénoising
  conjoint) : rebuttal "you changed the updater" quasi imparable. À éviter aussi pour ça.
- **(a1/P2) synthèse ÉVÉNEMENTIELLE au revisit : défendable.** Différenciateurs restants,
  précis : (i) état = mémoire associative fast-weight multi-segments écrite À L'ÉVICTION
  (capture ce qui SORT de la fenêtre ; AdaState = résumé courant de capacité 1 frame, sans lien
  à l'éviction) ; (ii) readout = frame SYNTHÉTISÉE À LA DEMANDE, adressée par le prompt du
  segment de retour (FiLM sur les queries du décodeur) : un recall épisodique déclenché par
  l'événement, pas une ancre continue ; testable : courbe re-ID vs nombre de segments stockés,
  chose qu'un état d'1 frame ne peut pas faire ; (iii) objectif dédié (reconstruction aux
  data-free de l'évincé + RSD teacher-oracle) là où AdaState n'a AUCUNE loss d'état ;
  (iv) éval résurrection d'identité SUJET A->B->A, absente de toute la littérature (les benchs
  revisit sont caméra/pose). En rebuttal : "AdaState maintient une ancre pour le drift ; nous
  ressuscitons un sujet ÉVINCÉ que leur état d'une frame, écrasé par B pendant le segment B, ne
  peut pas retenir" ; et la baseline AdaState-like du volet éval le montre empiriquement (à
  budgéter, déjà prévu).
- **(b)/hybride concat** : mécaniquement plus loin d'AdaState (pas d'écriture cache, tokens
  virtuels par concat) mais le mécanisme "état->K/V projetés" est déjà pris côté LLM (Latent
  Recurrent Transformer 2605.26797, Kwai Summary Attention 2604.24432) : la nouveauté de la
  règle de lecture seule est faible dans les deux cas. Les piliers défendables (TTT+éviction,
  synthèse à la demande, objectif+éval résurrection) sont IDENTIQUES en (a') et (b) : la
  préemption ne départage PAS les deux designs, la faisabilité et l'évidence oracle1 le font.
- MemGen/MAG reste le rebuttal use-case (revisit caméra, mémoire O(T), 390K vraies vidéos vs
  notre O(1) data-free) ; WorldKV le rebuttal "résurrection par cache KV" (stockage O(T) de KV
  réels indexé pose, zéro apprentissage). Citer AHN pour la reconstruction de l'évincé.
- Rappel du check de préemption : AdaState/ISPA/DySink/WorldKV datent de mai-juillet 2026,
  fenêtre concurrent-work, re-vérifier arXiv avant soumission.

---

## 5. Q5 : recommandation finale, risques, tests précoces

**Recommandation : (a') deux étages.**
V1 : décodeur état->frame latente entraîné par reconstruction aux (immédiate + retardée),
injection par splice `frames_override` au recache du revisit (position P1 dernier-slot par
défaut, P2 sink si le probe position la valide), backbone zero-shot, pas de RSD, pas de
modification de causal_model.
V2 (si v1 positif mais sous le plafond oracle) : hybride phantom-forward + concat, RSD
end-to-end, persistance par chunk.

Les 3 risques principaux du gagnant, avec détection PRÉCOCE :

| # | Risque | Test de détection précoce |
|---|---|---|
| 1 | **La courbe de tolérance est raide** : le canal splice exige une frame quasi parfaite, la reconstruction ne l'atteindra pas | Probe $0 "oracle-dose" AVANT toute ligne de code : splicer la frame A dégradée (flou latent croissant, bruit sigma, roundtrip VAE, mix alpha A/B) sur le harness 24 cellules ; tracer delta vs fidélité. Effondrement à faible dégradation = NO-GO du pivot (a), pivot vers hybride-RSD assumé sans plancher zero-shot |
| 2 | **Le décodeur ne reconstruit pas** (état 3 blocs = features 5/16/27, la synthèse pixel-latente est peut-être hors de portée ; ou rétention détruite par l'interférence de B) | Entraînement OFFLINE du décodeur seul (aux loss, pas de rollout, GPU 0) ; MSE/DINO vs frames évincées, versions immédiate ET retardée-post-B, comparées au seuil de fidélité du probe #1. Verdict en ~2 jours sans toucher au harness |
| 3 | **Fuite/adressage : au revisit l'état contient B**, la frame décodée mélange A et B et injecte la contamination au moment le plus sensible (reconstruction du cache) | Dès le premier ckpt décodeur : probe $0 splice-frame-décodée vs splice-oracle vs MEM_EPISODIC on/off (hooks en place) ; métriques ctrl_B_origA + oeil humain ; FiLM par prompt-embedding sur les queries du décodeur préparé en mitigation (étape B-multi-switch déjà planifiée) |

Ordre des probes $0 révisé (tous sur le hook `frames_override`/oracle existant, ~40 min
chacun, mêmes 24 cellules) : (1) oracle-dose ; (2) oracle-position-sink (P2 vs P1, frame A en
tête vs queue de frames_override) ; (3) MEM_EPISODIC sur canal splice (après premier ckpt).

---

## 6. Impact sur B0-B6 (localplan_B)

- **B0 (flags)** : conservé ; `mem_readout: frame_splice` (v1) | `vsink_concat` (v2) ;
  `vsink_tokens` remplacé par `frame_decoder: true`.
- **B1 (décodeur)** : remplacé : UN décodeur état->frame latente (1560 queries apprises ->
  swiglu_apply -> tête 1536->64/patch -> unpatchify 16x60x104), sous-module de VideoMemoryLayer
  (périmètre "memory.*" inchangé, broadcast/all-reduce/EMA/LoRA-exclusion hérités) + tête de
  reconstruction aux. Les 3 paires vsink_to_k/to_v par bloc DISPARAISSENT. Le zero-init exact
  n'a plus de sens en v1 (pas d'injection continue à neutraliser) : l'inertie est structurelle
  (pas de splice = baseline octet-identique).
- **B2 (insertion attention)** : SUPPRIMÉ en v1 (aucune modification de
  CausalWanSelfAttention : gain de risque majeur, les selftests S2 se réduisent à
  "flag OFF = bitwise"). Reporté en v2 sous la forme hybride (concat + LSE gate g=0 + skip
  structurel + phantom-forward sans _apply_cache_updates), décisions plan_final inchangées.
- **B3 (curriculum multi-switch)** : INCHANGÉ (le registre segment_tails fournit en prime les
  cibles de reconstruction retardée) + ajout du miroir `frames_override` dans le recache
  training (~15 lignes, déjà prévu).
- **B4 (teacher oracle)** : conservé pour l'éval et la v2, mais la V1 TOURNE SANS RSD :
  loss = DMD + L_rec(aux). Le run v1 se simplifie d'autant (pas de snapshot 9.2GB/step).
- **B5 (loss)** : RSD remplacé en v1 par L_rec = MSE latente normalisée (frame décodée vs frame
  évincée, immédiate chaque step + retardée échantillonnée) ; le grad-probe cible les params du
  décodeur ; RSD réintroduit en v2 (avec le découpage recache 11+1 et les vérifs detach de §3).
- **B6 (ops/VRAM)** : inchangé ; budget v1 PLUS LÉGER que prévu (pas de teacher par step, pas
  de concat) ; ajouter au selftest : byte-identity flag OFF, et assert que le splice est actif
  au recache du revisit dans CHAQUE pipeline (le bug historique n°7 de localplan_A).
- Dev estimé v1 : ~3-4 jours au lieu de 6 (B2 supprimé, B4 différé), APRÈS les probes #1-2.
