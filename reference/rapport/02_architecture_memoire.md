# 2. Architecture du module de mémoire (MAG / VideoMemoryLayer)

Cette section décrit la construction du module de mémoire fast-weight (nom interne MAG, pour un arrangement "Memory as a Gated branch", inspiré de la terminologie des variantes de Titans) et la manière dont il est greffé dans le DiT causal Wan-1.3B utilisé par Anchor-Forcing. Le module vit dans deux fichiers :

- `wan/modules/video_memory.py` : la cellule de mémoire proprement dite (`VideoMemoryLayer`), adaptée du mécanisme TTT de Scal3R (`FastWeightGluMLPMultihead`, arXiv 2604.08542), lui-même dans la lignée de LaCT (arXiv 2505.23884) et de TTT-Video (arXiv 2504.05298) ;
- `wan/modules/causal_model.py` : le point de greffe dans le bloc transformer (`CausalWanAttentionBlock.forward`) et la plomberie au niveau modèle (`CausalWanModel._forward_inference`), notamment le mécanisme d'eviction-feed introduit par le run c2.

L'idée générale : le DiT causal génère la vidéo chunk par chunk avec un cache KV borné (un sink de quelques trames initiales plus une fenêtre glissante). Tout ce qui sort de cette fenêtre est perdu pour l'attention. La mémoire fast-weight est un petit MLP dont les poids sont modifiés au fil de la génération pour retenir une version compressée de ce contenu, et dont la lecture est réinjectée dans le flux résiduel du DiT.

## 2.1 La cellule fast-weight SwiGLU : trois matrices comme état

### 2.1.1 L'état w0/w1/w2 et la lecture

L'état de la mémoire n'est pas un cache de tokens mais les poids d'un MLP SwiGLU à trois matrices, initialisées comme des paramètres appris (`video_memory.py:296-299`) :

```python
# TTT fast weight parameters initialization -- LaCT/Scal3R: 1/sqrt(fan_in)
self.w0 = nn.Parameter(torch.randn(num_heads, di, dh) / math.sqrt(di))
self.w1 = nn.Parameter(torch.randn(num_heads, dh, do) / math.sqrt(dh))
self.w2 = nn.Parameter(torch.randn(num_heads, di, dh) / math.sqrt(di))
```

Le module est instancié avec `num_heads=1` et `inter_multi=4` (`causal_model.py:537-542`), donc pour Wan-1.3B (dim 1536) : `di = do = 1536` et `dh = 6144`. L'état par bloc mémoire est donc constitué de deux matrices 1536 x 6144 et d'une matrice 6144 x 1536, soit environ 28 M de scalaires par échantillon et par bloc (~57 Mo en bf16). Une seule tête de dimension maximale est un choix délibéré : pour la TTT, un grand `head_dim` donne un état plus expressif (commentaire `video_memory.py:232`).

La lecture (le "apply") évalue simplement ce MLP sur une requête q :

```python
def swiglu_apply(q, w0, w1, w2):
    """ f_W(q) = (silu(q @ w0) * (q @ w2)) @ w1 """
    return (F.silu(q @ w0, inplace=False) * (q @ w2)) @ w1
```

(`video_memory.py:134-138`). C'est la formule d'application de Scal3R, isolée dans une fonction libre pour que la version causale puisse lire avant d'écrire (voir 2.2.4).

Les requêtes, clés et valeurs sont produites par une projection linéaire `qkv` sur les états cachés du chunk courant, avec RMSNorm optionnelle sur q et k (`qk_norm=True` par défaut, pour la stabilité autorégressive), dans `_qkv_lr` (`video_memory.py:367-387`). La même fonction produit aussi des learning rates par token : une petite couche linéaire `self.lrs` suivie d'un softplus décalé par `inv_softplus(base_lr)` avec `base_lr = 1e-2` (`video_memory.py:302-304, 380-386`). Le taux d'écriture est donc appris et modulé token par token, ce qui donne à la mémoire un moyen de choisir quoi écrire fort et quoi ignorer.

### 2.1.2 Le sens de "fast weight"

Il y a deux échelles de temps de paramètres :

- les poids lents : `qkv`, `lrs`, `proj`, `o_norm` et l'initialisation `w0/w1/w2` elle-même. Ils sont entraînés par la descente de gradient externe (DMD, co-entraînement LoRA) et sont identiques pour toutes les vidéos ;
- les poids rapides : le triplet `(w0, w1, w2)` courant, appelé `state`. À chaque nouvelle vidéo, il est réinitialisé par `init_state` (`video_memory.py:432-435`), qui réplique les paramètres appris sur le batch, puis il est mis à jour à chaque chunk pendant la génération, sans aucune backprop du loss externe : c'est du test-time training. La mémoire de la vidéo est littéralement encodée dans ces matrices de poids.

La lecture `f_W(q)` est donc une mémoire associative non linéaire : l'écriture stocke des associations clé -> valeur dans les poids (section suivante), et la lecture ressort, pour une requête proche d'une clé stockée, quelque chose de proche de la valeur associée.

## 2.2 L'écriture : objectif TTT, mise à jour Muon, coquille de norme

### 2.2.1 Le gradient interne

L'écriture d'un chunk consiste à faire un pas de gradient sur l'objectif interne TTT `dot = <f_W(k), v>` : rendre la sortie de la mémoire pour la clé k alignée avec la valeur v. La fonction `swiglu_fast_weight_gradient` (`video_memory.py:141-167`) calcule ce gradient à la main (backprop manuelle à travers le SwiGLU, avec `silu_backprop`), en repliant les learning rates par token directement dans le gradient :

```python
w0_grad = (k * lr0).transpose(-1, -2) @ dgate_before_act   # (B, C, Ch)
w1_grad = (hidden * lr1).transpose(-1, -2) @ v             # (B, Ch, C)
w2_grad = (k * lr2).transpose(-1, -2) @ dhide_before_mul   # (B, C, Ch)
```

Point important du docstring (`video_memory.py:146-147`) : il n'y a pas de signe moins, le fast weight fait une montée de gradient sur ce produit scalaire (c'est l'objectif TTT, pas une perte à minimiser), et l'orthogonalisation Muon est appliquée plus tard, dans `weight_norm_update`. Le corps est identique à celui de Scal3R ; la seule adaptation est la signature, pré-découpée à un seul chunk causal (pas de réordonnancement `ttt_ua_order` bidirectionnel comme en reconstruction 3D).

### 2.2.2 Muon : orthogonalisation par Newton-Schulz

Avant d'être ajouté aux poids, chaque gradient est orthogonalisé par `zeropower_via_newtonschulz5` (`video_memory.py:92-131`), l'itération quintique de Newton-Schulz popularisée par l'optimiseur Muon (code adapté de Moonlight/MoonshotAI, généralisé au cas batché `(B, C, C)`) :

```python
X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)   # norme spectrale <= 1
for _ in range(steps):                               # steps = 5 par défaut
    A = X @ X.transpose(1, 2)
    B = b * A + c * A @ A
    X = a * X + B @ X
```

L'itération approxime `US'V^T` où `USV^T` est la SVD du gradient, c'est-à-dire qu'elle égalise les valeurs singulières (S' ~ Uniform(0.5, 1.5)). Intuitivement : au lieu de laisser une ou deux directions dominantes écraser la mise à jour, toutes les directions du gradient contribuent avec une amplitude comparable, ce qui donne des écritures mieux conditionnées et une capacité mémoire mieux utilisée. `muon_update_steps=0` court-circuite l'orthogonalisation et retombe sur un SGD brut (`video_memory.py:111-112`), ce qui sert d'ablation. Le cast en bfloat16 de l'itération (`video_memory.py:115`) est signalé dans le code comme source possible de 2 à 4 % d'erreur relative de gradient sous checkpointing.

### 2.2.3 La mise à jour : renorm-to-init ou gated-delta

`weight_norm_update` (`video_memory.py:170-210`) combine le gradient Muon avec les poids. Le mode par défaut (`decay=None`) est la renormalisation de Scal3R :

```python
w0 = w0 + w0_grad                                            # montée de gradient (Muon)
w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm    # renorm à la norme d'AVANT
```

Chaque ligne de poids est ramenée à la norme qu'elle avait avant la mise à jour. Conséquence structurelle : la norme de l'état est conservée exactement, l'état ne fait que tourner dans une coquille de norme fixe. Il n'y a ni explosion ni oubli par décroissance : écrire du nouveau contenu se fait uniquement en réorientant les lignes, ce qui écrase progressivement l'ancien contenu par rotation. (Empiriquement, sur le run c2, la norme d'état est restée épinglée à 0.999 pendant tout l'entraînement, ce qui a permis d'innocenter l'écriture Muon lors de l'autopsie de l'instabilité tardive.)

Le mode optionnel `use_decay=True` (`video_memory.py:201-210`) implémente une variante gated-delta : `w <- sigmoid(decay_logit) * w + muon(grad)`, avec un logit de forget-gate scalaire appris (init `decay_init=0.95`, `video_memory.py:272-275`), puis un plafonnement de chaque ligne à sa norme d'initialisation (le rétrécissement est autorisé : c'est un oubli explicite, à la Gated DeltaNet). Le run c2 tourne avec `use_decay: false` (renorm pure) ; c3 l'active en couverture.

### 2.2.4 Apply-first : lire avant d'écrire

La méthode `step` (`video_memory.py:453-493`) impose l'ordre causal :

```python
o = swiglu_apply(q, w0, w1, w2)     # APPLY-FIRST (état issu des chunks < courant)
new_state = state
if do_update:                        # écriture (chunk évincé, ou courant)
    ...
```

La lecture du chunk t utilise l'état construit strictement à partir des chunks antérieurs ; le chunk courant ne lit jamais sa propre écriture. C'est le pendant mémoire de la causalité de l'attention : sans cela, la mémoire pourrait servir de canal de triche intra-chunk (recopier le présent au lieu de retenir le passé). L'écriture est optionnellement sous gradient-checkpoint (`checkpoint_write`, `video_memory.py:479-481`) et l'état est détaché en évaluation, gardé dans le graphe en entraînement (`detach_state=None` -> comportement automatique, `video_memory.py:437-438, 484-485`), ce qui permet la BPTT à travers les écritures pendant le co-entraînement.

## 2.3 L'injection dans le flux résiduel du DiT

### 2.3.1 Le point de greffe : trois blocs, une branche parallèle

Chaque `CausalWanAttentionBlock` peut porter une branche mémoire, construite seulement pour les blocs listés dans `memory_layers` (`causal_model.py:873-886`). La configuration c2 (`reference/runs/c2_eviction/c2_eviction.yaml:118-137`) en place trois, réparties sur la profondeur des 30 blocs de Wan-1.3B : `memory_layers: [5, 16, 27]` (début, milieu, fin du réseau), avec `gate_init: 0.1`.

Dans le forward du bloc (`causal_model.py:584-599`), la mémoire se branche juste après le résidu de self-attention, en parallèle, sur l'entrée normalisée du bloc :

```python
x_in = x
h = self.norm1(x_in)          # entrée normalisée, réutilisée par la branche mémoire
... self_attn ...
x = x_in + (y ... * e[2])     # résidu de self-attention
# parallel memory branch
if self.memory is not None and ttt_cache is not None:
    ...
    mem_out, new_state = self.memory.step(h, state=state_in, do_update=..., x_write=..., ...)
    ...
    x = x + (mem_out.detach() if ttt_update else mem_out)   # injection résiduelle
```

(`causal_model.py:601-701`, injection ligne 698). La requête de lecture est donc `h = norm1(x_in)`, le même tenseur que voit la self-attention, et la sortie de la mémoire s'ajoute au flux résiduel avant la cross-attention et le FFN. Sur la passe d'écriture (`ttt_update=True`, la passe qui alimente le cache propre détaché), `mem_out` est détaché : le gradient d'écriture passe par `new_state`, pas par x, ce qui évite de construire un graphe backward inutile d'environ 15 Go à travers les blocs gelés en aval (commentaire `causal_model.py:649-651`).

Un détail d'initialisation compte : `CausalWanModel.init_weights()` xavier-initialise toutes les nn.Linear du modèle, ce qui casserait le démarrage inerte de la mémoire ; `reset_parameters()` est donc rappelé sur chaque branche mémoire après l'init globale (`causal_model.py:914-918`, `video_memory.py:357-364`).

### 2.3.2 La sortie gated-AdaLN

`_project_output` (`video_memory.py:389-430`) transforme la lecture brute o en contribution résiduelle par une modulation de type AdaLN :

```python
shift, scale, gate = self.proj(x).chunk(3, dim=-1)
if self.bounded_gate:
    gate = torch.tanh(gate)
carrier = x if self.rel_norm else self.a_norm(x)
out = gate * (carrier * (1 + scale) + shift)
```

`proj` est initialisé à zéro, seul le biais du gate reçoit `g_init` (`video_memory.py:309-313`). Avec `g_init=0` le module démarre exactement inerte (sortie nulle, byte-identical à la baseline au step 0) ; les runs utilisent `gate_init: 0.1`, un démarrage doux où l'injection initiale vaut 10 % du carrier. Des entrées de gate additionnelles optionnelles (`gate_extra` : accord mémoire/prompt, pas de débruitage, trames depuis le switch) passent par une projection séparée zéro-initialisée (`video_memory.py:319-355`) ; elles sont désactivées dans c2 (`gate_extra: []`).

### 2.3.3 Les garde-fous d'injection

Une leçon centrale des runs e1-e4 est que l'amplitude de l'injection est le point de fragilité (artefact DC, dissolution, gel). La sortie est donc entourée d'une série de knobs, chacun désactivable et byte-identical quand il est off :

- `rel_norm` (`video_memory.py:391-393, 410`) : remplace la RMSNorm par token `o_norm` (qui effaçait la magnitude de lecture) par un unique scalaire RMS par chunk. La magnitude relative par token survit jusqu'à la projection et au carrier (l'injection reste couplée à la pertinence de la lecture), tandis que la croissance globale liée au nombre d'écritures s'annule à chaque chunk. Actif dans c2.
- `dc_remove` (`video_memory.py:428-429`, aussi forçable par l'env `MEM_DC_REMOVE`) : soustrait la moyenne sur les tokens de l'injection, à l'entraînement comme à l'inférence, pour interdire à `proj` d'apprendre la béquille du biais DC constant par token (l'artefact sombre/quadrillé diagnostiqué sur les premières ablations). Actif dans c2.
- `bounded_gate` et `bounded_mod` (`video_memory.py:404-407`) : bornent gate, puis scale/shift, par tanh, ce qui tue l'injection non bornée à RMS fixe et rend l'amplitude d'injection indépendante de l'état. Actifs dans c2.
- `mem_stream_scale` (`causal_model.py:659-670`, env `MEM_STREAM_SCALE` ou config) : enveloppe RMS au niveau du bloc, `mem_out <- mem_out * (scale * rms(x))`, où `rms(x)` est le RMS détaché du flux résiduel local. Borne l'injection à une fraction du signal local pour qu'une injection active ne puisse pas s'accumuler en artefact sur un long rollout hors distribution ; le detach en fait une borne, pas une grandeur que la mémoire peut manipuler. c2 tourne avec `mem_stream_scale: 1.0`.
- `ema_alpha` (`causal_model.py:671-686`) : passe-bas temporel causal sur l'injection de lecture, `mem_out <- alpha * mem_out + (1 - alpha) * ema_prev`, avec des buffers EMA indexés par timestep de débruitage et stockés (offload CPU) dans le `TTTBlockCache`. C'est le garde-fou anti-flicker d'e4 ; il vaut 0.0 (off) dans c2.

Deux hooks d'analyse complètent le tableau. `MEM_ZERO_ALL` (`causal_model.py:653-658`) met l'injection à zéro sur toutes les passes : c'est le bras d'ablation mem-OFF utilisé dans toutes les évaluations trois voies (mem-ON, mem-OFF, AF stock) ; l'annonce est imprimée une fois par process pour rendre le bras auditable. `MEM_REJECT_STREAM` (`causal_model.py:687-697`) est une sonde à coût nul qui soustrait par token la composante de l'injection parallèle au flux résiduel courant (rejeter "injecter plus de ce qui est déjà là", le mode auto-similaire qui amortit le mouvement), sans renormalisation après soustraction. Enfin, le module expose des diagnostics scalaires sans synchronisation hôte (`dbg_dc_frac`, `dbg_amp_disp`, décomposition spatiale/temporelle de la dispersion d'amplitude, `dbg_gate_mean`, `dbg_read_rms`, `video_memory.py:412-427, 490-492`) qui alimentent le monitoring wandb.

## 2.4 L'eviction-feed (c2) : écrire ce que le cache oublie

### 2.4.1 Le problème : l'écriture de c1 était redondante par construction

Le DiT causal maintient par bloc un cache KV borné : les `sink_size` premières trames latentes sont épinglées (jamais évincées : `sink_tokens = self._sink_runtime * frame_seqlen`, `causal_model.py:232`), et le reste est une fenêtre glissante de `local_attn_size` trames (`max_attention_size = local_attn_size * 1560` tokens, `causal_model.py:98`). Quand le cache déborde, les tokens les plus anciens hors sink sont évincés par un roll (`num_evicted_tokens`, `causal_model.py:250-306`), et l'attention lit la concaténation sink + fenêtre (`causal_model.py:351-437`).

Dans la version c1, la mémoire écrivait le chunk courant, celui dont les k/v venaient précisément d'entrer dans le cache KV. La mémoire stockait donc une copie de ce que l'attention voyait déjà parfaitement : elle n'avait, par construction, aucun travail à faire, et le gradient DMD ne pouvait pas la récompenser d'exister. Le fix c2, inspiré du complément de cache d'AHN (arXiv 2510.07318) et de VideoSSM : la mémoire doit écrire le chunk qui vient de tomber hors de la fenêtre d'attention, c'est-à-dire exactement ce que sink + fenêtre viennent de perdre.

### 2.4.2 Le mécanisme : FIFO par bloc, pop au niveau modèle, x_write

Le mécanisme a deux moitiés, séparées à dessein par la frontière du gradient checkpointing.

Côté bloc (`causal_model.py:628-641`), sur la passe d'écriture, le chunk courant normalisé est empilé dans un FIFO par bloc :

```python
if self.eviction_feed and ttt_update:
    _cur_chunk = int(current_start) // (frame_seqlen * num_frames)
    ttt_cache.fifo.append((_cur_chunk, h.detach()))
    _mem_do_update, _mem_xw = mem_x_write is not None, mem_x_write
```

Côté modèle (`causal_model.py:1342-1385`), avant la boucle sur les blocs et hors de toute région checkpointée, `_forward_inference` dépile l'entrée la plus ancienne du FIFO de chaque bloc mémoire et la passe au bloc comme kwarg explicite `mem_x_write` (`causal_model.py:1408, 1439`) :

```python
_wc = max(1, math.ceil(_law / _F))          # window_chunks = ceil(local_attn_size / F)
_sk = int(getattr(self.blocks[0].self_attn, "_sink_runtime", 0))
_sc = math.ceil(_sk / _F)                    # sink_chunks
...
if len(_fifo) >= _wc:                        # profondeur FIFO = la fenêtre d'attention
    _popped, _hev = _fifo.pop(0)
    if _popped >= _sc:                       # au-delà du sink -> réellement évincé du KV
        _mem_x_write[_bi] = _hev
    # else: chunk du sink, épinglé pour toujours dans le cache -> jamais écrit
```

C'est un délai FIFO simple et robuste, pas une éviction token-exacte du KV (le commentaire `causal_model.py:1346-1349` l'assume) : un chunk est considéré évincé `window_chunks` chunks après son écriture dans le FIFO, avec `window_chunks = ceil(local_attn_size / F)` où F est le nombre de trames latentes par chunk. Dans la config c2 (`local_attn_size: 12`, chunks de 3 trames latentes, `sink_size: 3`), cela donne `window_chunks = 4` et `sink_chunks = 1` : la mémoire écrit à l'instant t le chunk t-4, et le tout premier chunk (le sink, qui reste dans le cache pour toujours) n'est jamais écrit. Le selftest du run a vérifié ce lag de 4 chunks exactement, ainsi que la bit-identité du chemin `eviction_feed=false`.

Enfin, dans `VideoMemoryLayer.step` (`video_memory.py:470-472`), `x_write` sépare lecture et écriture :

```python
if do_update:
    if x_write is not None:                 # eviction-feed: source d'écriture = tokens évincés
        _, k, v, lr0, lr1, lr2, _ = self._qkv_lr(x_write)
```

La requête de lecture q reste celle du chunk courant (ce qu'on est en train de générer), mais les k, v et learning rates de l'écriture sont recalculés sur `x_write`, le chunk évincé. Tant que rien n'a été réellement évincé (warm-up du FIFO, ou chunk encore dans le sink), l'écriture est entièrement sautée (`_mem_do_update = mem_x_write is not None`), donc la mémoire n'accumule que du contenu complément-de-cache. `x_write=None` (défaut) écrit le chunk courant, byte-identical au comportement pré-c2.

### 2.4.3 Pourquoi c'est le fix clé

Après l'eviction-feed, la partition de l'information est propre : l'attention voit le sink (ancre initiale) et la fenêtre (passé récent) ; la mémoire détient exactement le complément, le passé intermédiaire que le cache a perdu. La mémoire a désormais un contenu que rien d'autre ne fournit, donc un travail que le gradient peut récompenser, ce que le co-entraînement c2 exploite en plus par le cache-dropout (des rollouts entiers avec le sink affamé à 0 trame, pour que la mémoire soit la seule à porter l'ancre). Empiriquement, sur l'évaluation sink-starvation, c2 a corrigé le gel du mouvement de c1 et réduit nettement son atteinte à l'identité aux checkpoints précoces et moyens ; par honnêteté, il faut préciser que cela n'a pas suffi à stabiliser le comportement en fin d'entraînement (dérive des paramètres lents sur-amplifiant l'injection après le step 1000, adressée par le pack de stabilisation du run c3), mais ce point relève de l'objectif d'entraînement, pas de l'architecture décrite ici.

## 2.5 Le TTTBlockCache : l'état sous FSDP et gradient checkpointing

L'état par bloc circule dans un objet dédié (`video_memory.py:45-66`) :

```python
class TTTBlockCache:
    __slots__ = ("state", "ema", "fifo")
    def __init__(self):
        self.state = None    # triplet (w0, w1, w2) courant
        self.ema = {}        # buffers passe-bas e4, indexés par timestep
        self.fifo = []       # (chunk_idx, h détaché) pour l'eviction-feed c2
```

Le choix d'un objet plutôt qu'un dict est fonctionnel, pas cosmétique (docstring `video_memory.py:46-53`) : le cast des entrées en précision mixte de FSDP et le gradient/activation checkpointing parcourent les kwargs comme des pytrees et reconstruisent les conteneurs dict/list (un dict neuf à chaque forward), ce qui avale silencieusement les écritures in-place dans un dict. Un objet custom est une feuille de pytree : il est passé par référence, donc `ttt_cache.state = new_state` (`causal_model.py:700-701`) mute l'objet partagé et survit à la pile d'appels. Scal3R faisait circuler un tuple, ce qui marche dans son chemin d'inférence mono-process mais pas sous notre pile d'entraînement FSDP.

Le checkpointing impose une seconde précaution : la fonction checkpointée est rejouée au backward, donc tout état lu à l'intérieur doit être identique entre les deux exécutions. Trois mécanismes s'alignent sur ce principe :

- l'état fast-weight est photographié hors de la région checkpointée et passé en tenseurs explicites `ttt_state` (`causal_model.py:1397-1403`), que checkpoint sauvegarde comme entrées ; le bloc lit `state_in = ttt_state` plutôt que l'objet vivant (`causal_model.py:603-605`) ;
- le buffer EMA précédent est lu dehors et passé en `mem_ema_prev` (`causal_model.py:1330-1341`) ;
- le chunk évincé est dépilé dehors et passé en `mem_x_write` (`causal_model.py:1342-1385`) ; le seul effet de bord restant dans le bloc, l'append au FIFO, est ré-exécuté par le recompute du backward, ce que le snapshot/restore du cache TTT côté trainer annule (même mécanisme que pour `.state` et `.ema`, commentaire `causal_model.py:635-638`). Les entrées du FIFO sont des tuples jamais mutés en place, donc une copie superficielle `list()` suffit pour les snapshots twin/BPTT.

## 2.6 Récapitulatif de la configuration effective (run c2_eviction)

Pour fixer les idées, les valeurs du run c2 (`reference/runs/c2_eviction/c2_eviction.yaml:116-137`) :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `memory_layers` | [5, 16, 27] | trois branches mémoire sur 30 blocs |
| `local_attn_size` / `sink_size` | 12 / 3 trames latentes | fenêtre glissante + sink épinglé |
| `gate_init` | 0.1 | gate AdaLN au démarrage |
| `muon_update_steps` | 5 | itérations Newton-Schulz de l'écriture |
| `use_decay` | false | renorm-to-init (coquille de norme fixe) |
| `rel_norm`, `dc_remove`, `bounded_gate`, `bounded_mod` | true | garde-fous d'injection |
| `mem_stream_scale` | 1.0 | enveloppe RMS de flux |
| `ema_alpha` | 0.0 | passe-bas temporel (off) |
| `eviction_feed` | true | écriture du chunk évincé (window_chunks=4, sink_chunks=1) |
| `ckpt_write` | true | écriture sous gradient-checkpoint |

En résumé, l'architecture assemble quatre pièces : une mémoire associative dont l'état est un MLP SwiGLU de ~28 M de scalaires par bloc, mis à jour au test-time par montée de gradient orthogonalisée (Muon) dans une coquille de norme fixe ; une lecture apply-first injectée en résiduel via une modulation AdaLN gated, bornée par une pile de garde-fous d'amplitude tous désactivables ; un routage d'écriture eviction-feed qui donne à la mémoire le complément exact du cache KV ; et un porteur d'état (`TTTBlockCache`) conçu pour survivre à FSDP et au gradient checkpointing. Chaque ajout est gardé derrière un flag dont la valeur par défaut reproduit bit à bit le modèle de base, ce qui a permis de valider chaque étape par selftest de bit-identité avant lancement.
