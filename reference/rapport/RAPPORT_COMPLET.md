# Mémoire à poids rapides (fast-weight / test-time training) pour la génération de vidéo longue

## Étude d'un module MAG greffé dans Anchor-Forcing (Wan2.1-T2V-1.3B, DMD self-rollout)

*Document technique, état au 15 juillet 2026. Généré comme base de rapport de stage: il documente le code, le pipeline d'entraînement, l'ensemble des expériences et les résultats (négatifs inclus), de façon honnête et vérifiée contre le code et les métriques réelles.*

---

## Résumé exécutif

**Objectif.** Corriger la dérive d'identité/sujet dans la génération autoregressive de vidéos longues, en greffant une **mémoire à poids rapides** (fast-weight / test-time training, notée MAG) dans **Anchor-Forcing** (backbone Wan2.1-T2V-1.3B, diffusion vidéo causale streaming distillée en 4 pas par **DMD self-rollout**, entraînement **sans données vidéo longues réelles**).

**Parcours expérimental.** (1) Des **garde-fous** explicites (série e3/e4) pour forcer un comportement non-dégénéré: ils gèlent puis "flickerent" (whack-a-mole structurel, triade de Goodhart). (2) Un **co-entraînement propre pur-DMD** (c1): plus de gel, mais l'injection de mémoire est neutralisée vers un régime **inerte** (do-no-harm sans aide) sur la dérive simple. (3) Le fix **eviction-feed** (c2): la mémoire écrit désormais le chunk **évincé** (hors fenêtre d'attention) au lieu du chunk courant redondant, façon AHN/VideoSSM. Résultat: aux checkpoints tôt/moyens, une **vraie mémoire MLP co-entraînée do-no-harm et dynamique** (ce que c1 ratait), mais une **instabilité (flicker)** apparaît vers l'étape 1000. (4) Une **autopsie** de la télémétrie montre que ce flicker n'est **pas** le "mur" fondamental mais une **pathologie d'optimisation réparable** (une rampe monotone non-amortie des paramètres lents qui sur-pilote l'injection). (5) Un **pack de stabilisation** (c3, inspiré de VideoSSM) est en cours d'évaluation.

**Bilan honnête.** Le module de mémoire n'est **pas** le coupable des dégénérescences: le fix eviction-feed lui donne enfin un rôle (do-no-harm) et le flicker est réparable. Le **mur réel** est l'**objectif data-free**: le juge DMD (un teacher à contexte court qui ne voit pas le cache) ne **supervise pas** l'identité d'instance sous-spécifiée par le texte. Stabiliser **amortit** mais ne **redirige pas** cet objectif: le mieux atteignable en data-free est le **do-no-harm**; un effet d'identité **positif** exige une cible supplémentaire (reconstruction sur **vraie vidéo**, famille VideoSSM/MemGen, ou un **témoin non-factorisé data-free**, l'idée "Anchor Resurrection" A→B→A par splice de KV). Vis-à-vis de la littérature, le mécanisme est une **reproduction** (AHN/VideoSSM); la contribution défendable est l'**analyse rigoureuse**, l'**intégrité métrique** (les métriques VBench sont trompeuses: le gel gonfle la Subject Consistency, le flicker gonfle le Dynamic Degree), et le **diagnostic** "on ne peut pas récompenser une mémoire longue-portée avec un juge court-portée data-free".

---

## Sommaire

1. **Contexte et état de l'art** — le problème de la dérive en vidéo longue, la lignée Self-Forcing → LongLive → Anchor-Forcing, la distillation DMD, les mémoires fast-weight/TTT, et les travaux qui préemptent (VideoSSM, MemGen, AHN, SSD).
2. **Architecture du module de mémoire (MAG)** — la cellule fast-weight SwiGLU, l'écriture Muon (Newton-Schulz + renorm-to-init), l'injection gated-AdaLN, les garde-fous d'injection, et le mécanisme clé eviction-feed (FIFO + `x_write`), code à l'appui.
3. **Pipeline d'entraînement et modifications** — la distillation DMD, le self-rollout streaming, le co-entraînement LoRA + mémoire, le BPTT, les garde-fous de la série e, et les runs c1 / c2 / c3 (tableau des configurations).
4. **Expériences et résultats** — le parcours chronologique complet avec les trajectoires chiffrées (e → c1 → c2 → c3), l'autopsie du flicker, honnête sur les résultats négatifs.
5. **Méthodologie d'évaluation** — le protocole 3-way, l'évaluation sink-starvation, les métriques (VBench, motion lag-2, id_cos DINO tail-mean), et les leçons d'intégrité métrique.
6. **Conclusions et travaux futurs** — ce qui marche, les murs structurels, le positionnement vis-à-vis de la littérature, et les pistes concrètes (Anchor Resurrection, reconstruction sur vraie vidéo, benchmark revisit/re-ID).

---

# 1. Contexte et état de l'art

## 1.1 Le problème : générer des vidéos longues sans perdre le sujet

Les modèles de diffusion vidéo actuels (famille Wan2.1, notamment le modèle texte-vers-vidéo Wan2.1-T2V-1.3B utilisé dans ce travail) produisent des clips courts, typiquement 5 secondes, d'excellente qualité. Étendre cette génération à des durées de plusieurs minutes impose de passer à un régime **autorégressif** : le modèle génère la vidéo par blocs de quelques images (chunks), chaque bloc étant conditionné sur les blocs précédents, en flux continu (streaming). Ce changement de régime introduit deux difficultés fondamentales.

La première est l'**accumulation d'erreurs** (exposure bias). Pendant l'entraînement classique, le modèle est conditionné sur des images vraies ; à l'inférence, il est conditionné sur ses propres sorties, imparfaites. Chaque petite erreur (une texture légèrement fausse, un contraste dérivé) est réinjectée comme contexte du bloc suivant, et les erreurs se composent. Après quelques dizaines de secondes, on observe des dérives caractéristiques : saturation ou assombrissement progressif des couleurs, perte de netteté, et surtout **dérive d'identité du sujet** : le visage, les vêtements ou les attributs d'un personnage se transforment lentement jusqu'à ne plus correspondre au sujet initial.

La seconde difficulté est **structurelle** : pour rester calculable en streaming, l'attention du modèle est restreinte à une fenêtre locale sur les derniers blocs (voir section 1.2.2). Toute information plus ancienne que cette fenêtre est littéralement évincée du contexte. Or l'identité du sujet est précisément une information à longue portée : si le sujet établi à la seconde 0 n'est plus dans le contexte à la seconde 60, rien ne contraint le modèle à le préserver, sinon le texte du prompt. Le prompt agit comme une ancre globale, mais il ne spécifie l'identité qu'au niveau de la **catégorie** (« une femme en manteau rouge ») et non de l'**instance** (ce visage précis, cette coupe de cheveux précise). Les attributs d'instance, sous-spécifiés par le texte, dérivent donc librement. C'est cette dérive d'identité sur des horizons de plusieurs minutes que ce stage cherche à corriger.

Enfin, la difficulté est aggravée par le mode d'entraînement : les recettes récentes (section 1.2) entraînent le modèle **sans vidéos longues réelles**, par auto-déroulé (self-rollout) jugé par un professeur à horizon court. Le signal d'apprentissage porte alors sur la plausibilité locale de chaque bloc, pas directement sur la cohérence à longue portée, ce qui rend le problème à la fois pratique (les métriques locales peuvent être bonnes alors que l'identité est perdue) et théorique (quel gradient récompense la mémoire longue ?).

## 1.2 La lignée de base : de Self-Forcing à Anchor-Forcing

Notre code repose sur une lignée de trois travaux successifs, tous construits sur le backbone Wan2.1-T2V-1.3B : Self-Forcing (arXiv 2506.08009), LongLive (arXiv 2509.22622), puis Anchor-Forcing (arXiv 2603.13405), qui est le dépôt directement étendu par ce stage. Aucun des trois n'utilise de mémoire apprenable à poids rapides : ce sont des architectures « attention + cache KV + sink » pures, point important pour situer notre contribution.

### 1.2.1 Distillation DMD et self-rollout causal (Self-Forcing)

Un modèle de diffusion standard nécessite des dizaines de pas de débruitage par image, incompatible avec le temps réel. La **Distribution Matching Distillation (DMD)** distille ce professeur multi-pas en un étudiant à très peu de pas (4 pas dans notre configuration, cf. `denoising_step_list` dans `configs/ttt_memory.yaml`). Le principe : plutôt que d'imiter les trajectoires du professeur pas à pas, on aligne la **distribution** des sorties de l'étudiant sur celle du professeur. Concrètement, le gradient DMD est la différence entre deux fonctions de score, celle d'un modèle « réel » (le professeur figé, ici Wan2.1-T2V-14B pour Anchor-Forcing) et celle d'un modèle « fake » (un critique entraîné en ligne sur les sorties de l'étudiant), évaluées sur les échantillons générés. L'étudiant est poussé vers les régions que le professeur juge plausibles, sans nécessiter de données appariées.

**Self-Forcing** (arXiv 2506.08009) applique cette distillation en régime **causal et autorégressif** : l'étudiant est rendu causal (chaque bloc n'attend que sur le passé, via un cache KV roulant), puis entraîné par **self-rollout** : il déroule lui-même une vidéo bloc par bloc, conditionné sur ses propres sorties, et la perte DMD est appliquée sur les blocs générés (avec troncature du gradient sur les blocs les plus anciens pour tenir en mémoire). L'intérêt décisif est que l'entraînement reproduit exactement les conditions de l'inférence : le modèle apprend à être conditionné sur ses propres erreurs, ce qui atténue l'exposure bias. Notre `pipeline/self_forcing_training.py` en est directement adapté. Une propriété centrale pour la suite : le professeur et le critique DMD notent chaque fenêtre de quelques secondes **isolément** (sans cache long) ; le juge est à horizon court, même si le déroulé est long.

### 1.2.2 Cache KV, fenêtre d'attention locale et attention sink

Le mécanisme qui rend le streaming possible est le **cache KV** : pour chaque bloc transformer, les clés (K) et valeurs (V) des tokens déjà générés sont mémorisées, de sorte que les nouveaux tokens attendent sur le passé sans le recalculer. Sans borne, ce cache croît linéairement avec la durée ; l'attention y devient inabordable. Deux dispositifs classiques bornent le coût, tous deux visibles dans `wan/modules/causal_model.py` :

- La **fenêtre d'attention locale** : seuls les derniers `local_attn_size` frames (12 dans notre configuration) restent dans le cache. Quand un nouveau bloc arrive, les tokens les plus anciens sont **évincés** par décalage du cache (rolling). C'est cette éviction qui crée le déficit de mémoire longue : au-delà de ~12 frames latentes, l'information est perdue.
- L'**attention sink** : les tokens des tout premiers frames (`sink_size` = 3) sont conservés en permanence en tête du cache et concaténés à la fenêtre locale à chaque étape. Ce mécanisme, hérité des observations sur les LLM en streaming (les premières positions absorbent une masse d'attention disproportionnée et leur suppression effondre la génération), joue un double rôle : stabilité numérique de l'attention et, de fait, **ancre visuelle** : le début de la vidéo, donc souvent le sujet initial, reste visible pour toujours. Une part importante de la tenue d'identité des systèmes de cette lignée passe par ce sink, ce qui devra être gardé à l'esprit en évaluant toute mémoire additionnelle (le sink fait déjà une partie du travail).

### 1.2.3 LongLive : passage à l'échelle de la minute et interactivité

**LongLive** (arXiv 2509.22622, NVIDIA) est le parent immédiat de notre base de code. Il part de Self-Forcing et l'étend à la génération longue (jusqu'à 240 s) et **interactive** (changement de prompt en cours de génération). Ses apports : (i) le **KV re-cache** lors d'un changement de prompt, qui recalcule le cache des frames récentes sous le nouveau prompt pour éviter une transition incohérente ; (ii) le **sink au niveau frame** décrit ci-dessus ; (iii) un réglage fin long en streaming (streaming long tuning) où le modèle s'entraîne sur des déroulés longs. LongLive atteint la minute avec la seule combinaison fenêtre locale + sink, sans mémoire apprise (de même que Self-Forcing++, arXiv 2510.02283, sur le même tronc, jusqu'à plus de 4 minutes), ce qui fixe une base de comparaison exigeante.

### 1.2.4 Anchor-Forcing : Anchor Memory et Tri-Region RoPE

**Anchor-Forcing** (arXiv 2603.13405, vivoCameraResearch) est la couche finale de notre base : un réglage fin (LoRA) au-dessus de LongLive, distillé par DMD depuis le professeur Wan2.1-T2V-14B. Ses deux mécanismes :

- L'**Anchor Memory** : une réorganisation **structurelle** du cache KV pour le scénario multi-prompts. Au lieu d'un simple sink + fenêtre, le cache est partitionné en cache sink, caches d'ancrage/jonction figés après chaque changement de prompt, et caches locaux roulants. Insistons sur un point de terminologie : malgré son nom, l'Anchor Memory n'est **pas une mémoire apprenable** ; c'est une politique de conservation de tokens KV existants. Elle est d'une nature entièrement différente du module de mémoire à poids rapides introduit dans ce stage, et les deux ne doivent pas être confondus.
- Le **Tri-Region RoPE** : les indices de position rotatifs (RoPE) sont réassignés par région (sink / local / clip courant) et bornés, au lieu de croître indéfiniment avec la durée. Cela évite l'extrapolation de position hors du domaine d'entraînement, cause connue de dégradation sur les longues séquences.

C'est sur ce système (checkpoints publics `base.pt` + LoRA `final.pt`) que notre module de mémoire est greffé.

## 1.3 Mémoires à poids rapides et test-time training

### 1.3.1 L'idée

Le déficit identifié en 1.2.2 est précis : tout ce qui sort de la fenêtre de 12 frames et n'est pas dans les 3 frames du sink est perdu. L'idée des **mémoires à poids rapides** (fast weights) est de compléter ce cache sans en payer le coût : une petite matrice (ou un petit MLP) d'état M, de taille **constante**, est mise à jour à chaque bloc en fonction du flux de tokens (écriture), puis interrogée par les tokens courants (lecture), et sa sortie est réinjectée dans le transformer via une porte (gate). La mémoire pourrait alors retenir, en O(1) mémoire et calcul par pas, l'information évincée du cache KV, en particulier l'identité d'instance du sujet.

Deux familles de règles de mise à jour existent, et la distinction est structurante pour ce projet :

- Le **test-time training (TTT)** proprement dit : l'écriture est une **boucle de descente de gradient interne**, exécutée au moment de l'inférence, sur une perte interne (typiquement reconstruire v à partir de k). **TTT-Video** (arXiv 2504.05298) insère de tels couches TTT-MLP dans un DiT pré-entraîné et génère des vidéos d'une minute cohérentes (dessins animés Tom & Jerry), mais sous des conditions précises : perte de reconstruction sur vraies vidéos, backbone co-entraîné, gate borné en tanh, curriculum de 3 s à 63 s. **LaCT** (arXiv 2505.23884) rend ce schéma efficace sur GPU en effectuant la mise à jour par grands chunks (large-chunk TTT) plutôt que token par token. **Titans** (arXiv 2501.00663) ajoute une modulation par la « surprise » (le gradient interne module la force d'écriture) et un oubli appris.
- La règle **delta / gated-delta** (DeltaNet, Gated DeltaNet, et plus largement les SSM linéaires type Mamba2) : une **récurrence en forme close**, M_t = décroissance(g_t) · M_{t-1} + k(v − M_{t-1}k)^T, entraînée bout à bout par rétropropagation dans le temps. Mathématiquement, la mise à jour delta est exactement **un** pas de gradient sur ½‖v − Mk‖² : la famille gated-delta est donc une forme dégénérée (un pas, forme close) de TTT, beaucoup moins expressive mais nettement plus stable à entraîner.

### 1.3.2 Notre substrat : Scal3R

Le module implémenté pendant ce stage (`wan/modules/video_memory.py`, `VideoMemoryLayer`) est adapté de **Scal3R** (arXiv 2604.08542, CVPR 2026), un système de reconstruction 3D à grande échelle : mémoire fast-weight en MLP SwiGLU multi-têtes, gradient interne par mini-batch, orthogonalisation du gradient par itérations de Newton-Schulz (optimiseur Muon) et renormalisation des poids (weight-norm) après chaque écriture. Nous avons donc transplanté le **mécanisme** TTT le plus expressif de la littérature (multi-pas, Muon) hors de son domaine d'origine : Scal3R traite de la reconstruction 3D supervisée, pas de la diffusion vidéo causale. Ce choix de substrat (TTT « dur » plutôt que gated-delta « doux ») et le risque de transfert de domaine associé se sont révélés être des facteurs déterminants des résultats, discutés dans les sections suivantes du rapport.

La question scientifique posée est donc : une telle mémoire, greffée dans quelques blocs du transformer d'Anchor-Forcing et entraînée **sans données vidéo réelles** (uniquement par le gradient DMD du self-rollout), peut-elle apprendre à retenir l'identité au-delà de la fenêtre KV ?

## 1.4 Travaux proches : ce qui préempte, ce qui inspire, ce qui reste ouvert

Le paysage a évolué rapidement pendant le stage ; quatre travaux doivent être positionnés honnêtement.

**VideoSSM** (arXiv 2512.04519) est le plus proche, au point de préempter largement le mécanisme général. Même backbone (Wan2.1-T2V-1.3B), même régime d'entraînement (long self-rollout avec correction DMD fenêtrée, sans vidéos longues réelles), et une mémoire globale **gated-delta** hybridée avec la fenêtre KV locale : la mémoire n'absorbe que les tokens évincés de la fenêtre, et un routeur de fusion sensible à la position mélange local et global. VideoSSM dépasse LongLive et Self-Forcing en cohérence et en dynamisme à 60 s. Les différences avec notre approche sont exactement les axes durs : VideoSSM utilise la récurrence en forme close (stable) là où nous utilisons le TTT multi-pas Muon (expressif mais instable), et co-entraîne **tout** le backbone là où le nôtre est gelé (LoRA au mieux). VideoSSM constitue aussi une preuve d'existence importante : une mémoire globale co-entraînée **reçoit** bien un gradient utile du DMD fenêtré (son état au bloc i change la plausibilité de la fenêtre i), ce qui invalide la version forte de l'objection « un juge à horizon court ne peut pas récompenser une mémoire longue ».

**Memorize-and-Generate / MemGen** (arXiv 2512.18741), même backbone, diagnostique précisément notre mode d'échec : la solution dégénérée p(x | historique, texte) → p(x | texte), où le modèle ignore l'historique et ne s'appuie que sur le prompt. Leur remède passe par de **vraies vidéos** : une perte d'historique (conditionnement à texte vide avec probabilité 0.6 pendant l'entraînement du générateur, forçant le modèle à puiser dans le contexte visuel) et une mémoire par **cache KV compressé** (pas de fast-weight). Ils introduisent MAG-Bench, un banc de test de re-visite de scène (la caméra quitte puis retrouve un lieu) avec métriques pixel. Différence clé avec nous : leur cible d'apprentissage contient l'identité (reconstruction de vraies vidéos), la nôtre non (DMD pur, sans données) ; c'est précisément la variable que notre étude isole.

**AHN, Artificial Hippocampus Networks** (arXiv 2510.07318, ByteDance) opère sur les LLM mais fournit le patron architectural que notre seconde itération (c2) adopte : la fenêtre KV glissante sert de mémoire courte **sans perte**, et un module récurrent apprenable (Mamba2 / DeltaNet / GatedDeltaNet) compresse **les tokens évincés** de la fenêtre dans un état de taille fixe, le backbone restant gelé (auto-distillation). L'idée directrice, que nous reprenons, est que la mémoire ne doit pas ré-encoder ce que le cache voit déjà (redondant, donc sans gradient utile), mais exactement le **complément** du cache : ce qui vient d'en tomber.

**SSD** désigne ici le Scheduled Sliding-window Dropout proposé dans « Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods » (arXiv 2510.05901). Ce travail montre qu'un modèle hybride (attention à fenêtre + composant récurrent linéaire) apprend spontanément à **contourner** le composant récurrent : la fenêtre softmax suffit à la perte d'entraînement, le composant mémoire n'est jamais sollicité. Le remède est de supprimer aléatoirement la fenêtre pendant l'entraînement pour forcer le flux à passer par la mémoire. Nous reprenons ce principe sous forme de **cache-dropout** (réduction aléatoire du sink pendant certains déroulés d'entraînement), afin que la mémoire « gagne » un gradient au lieu d'être court-circuitée par le sink.

### Positionnement

En synthèse : la lignée Self-Forcing → LongLive → Anchor-Forcing établit qu'attention locale + sink + réorganisation du cache portent la génération jusqu'à plusieurs minutes sans mémoire apprise ; TTT-Video, LaCT, Titans et Scal3R fournissent les mécanismes de mémoire à poids rapides ; VideoSSM et MemGen démontrent, sur le même backbone, qu'une mémoire récurrente co-entraînée (respectivement avec juge DMD fenêtré, et avec cible de reconstruction sur vraies vidéos) réduit effectivement la dérive. La combinaison spécifique étudiée dans ce stage, une mémoire TTT multi-pas de type Scal3R greffée sur un Anchor-Forcing (quasi) gelé et entraînée **sans aucune donnée réelle** par DMD self-rollout, n'est couverte par aucun de ces travaux, mais elle n'est pas non plus, au vu de VideoSSM, la seule voie vers une mémoire fonctionnelle. Le rapport doit donc être lu comme une **étude empirique de ce régime précis** : quelles conditions (cible d'écriture, co-entraînement, dropout de cache, stabilisation) sont nécessaires pour qu'une telle mémoire soit au minimum inoffensive, et où se situe la limite de l'entraînement sans données, en particulier pour l'identité d'instance que le texte ne spécifie pas.
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
# 4. Expériences et résultats

Cette section retrace le parcours expérimental complet du stage, dans l'ordre chronologique. Le fil conducteur est une seule question: une mémoire à poids rapides (fast-weight, de type TTT), injectée dans un générateur vidéo autorégressif entraîné par distillation DMD sans données réelles, peut-elle améliorer la cohérence à long terme (identité du sujet) sans dégrader le mouvement? La réponse honnête, construite run après run, est nuancée: chaque tentative a produit un enseignement précis, souvent négatif, et ces résultats négatifs constituent une part essentielle des conclusions du stage.

## 4.1 Protocole d'évaluation commun

Toutes les évaluations rapportées ci-dessous suivent le même protocole, durci au fil des runs à mesure que nous découvrions les angles morts des métriques.

**Générations.** Six prompts held-out (jamais vus à l'entraînement), graine fixée (seed 42), vidéos longues de 240 frames latentes, soit 957 images décodées (environ 40 secondes), très au-delà de l'horizon d'entraînement (environ 5 secondes par fenêtre).

**Trois bras systématiques.** Une leçon des premiers mois (audit de validité des baselines) est qu'une comparaison à deux bras est trompeuse. Chaque évaluation comporte donc:

- **mem-ON** (memon): le checkpoint entraîné, mémoire active;
- **mem-ZERO** (memzero): le même checkpoint, injection mémoire mise à zéro à l'inférence (variable d'environnement `MEM_ZERO_ALL=1`). Ce bras isole la dérive due au co-entraînement LoRA seul; la différence memon moins memzero mesure l'effet spécifique de l'injection;
- **AF-stock**: le checkpoint Anchor-Forcing officiel, sans mémoire ni fine-tuning. C'est la référence absolue.

**Métriques.** VBench (Subject Consistency SC, Background Consistency BC, Dynamic Degree DD), une métrique de mouvement lag-2 insensible au scintillement (flicker-blind, seuil de gel pré-enregistré à 0.01), la dérive DINO par frame par rapport à une ancre initiale, et surtout la similarité cosinus d'identité id_cos(f0, ft). Après avoir constaté que la valeur id_cos sur une seule frame finale est bruitée et a produit un faux verdict (voir 4.4), la métrique primaire est devenue le **tail-mean id_cos**: la moyenne de la courbe id_cos sur les 300 dernières frames, et l'écart memon moins memzero de ce tail-mean, avec un décompte par prompt (n/6 positifs). Les seuils sont pré-enregistrés: plancher de bruit DINO 0.18, tolérance de parité 0.03.

**Évaluation sink-starvation.** À partir de c2, nous ajoutons une condition d'évaluation où le sink d'attention (les tokens d'ancrage que le cache KV conserve en permanence) est réduit de 3 à 1. Cette privation crée artificiellement un travail pour la mémoire: si elle retient réellement le contenu que le cache a perdu, elle doit aider dans cette condition. C'est le test « it works » le plus direct que nous ayons construit.

## 4.2 Point de départ: une injection qui nuit ou se dissout

Avant la série décrite ici, deux constats avaient été établis sur les premières versions du module (runs e1 et e2, backbone gelé, seule la mémoire entraînée).

La première ablation propre (e1_rerun2, step 250 puis confirmée à 1000) montrait mem-ON avec une dérive DINO tardive de 0.501 contre 0.204 pour mem-OFF, la mémoire nuisant sur 8 prompts sur 8. Un audit a ensuite requalifié ce résultat: environ 97 % de l'écart provenait d'un artefact d'injection quasi-DC (biais constant par token, assombrissement et moirage), et une fois ce biais retiré (`MEM_DC_REMOVE=1`), l'effet net de la mémoire tombait dans le plancher de bruit: la mémoire était **inerte**, pas toxique. Le correctif d'amplitude qui a suivi (e2_amp: normalisation RMS relative par chunk plus retrait du DC) a été pire: en épinglant l'injection à une amplitude fixe à chaque chunk, il l'accumulait hors distribution sur 240 frames et **dissolvait** le sujet en taches abstraites (DINO 0.753 contre 0.271 pour mem-OFF, pire sur 8/8), tout en restant quasiment invisible pour VBench (SC 0.952 contre 0.966). Ce dernier point, la cécité de VBench devant une vidéo qui se dissout, a motivé le durcissement du protocole de la section 4.1.

Le tableau était donc posé: soit l'injection est neutralisée (inerte), soit elle dégrade. La suite du stage a consisté à chercher un régime où elle aide.

## 4.3 La série e: les garde-fous (e3_antifreeze, e4_antiflicker)

L'hypothèse de la série e était qu'on pouvait contraindre l'optimisation par des termes de régularisation (garde-fous) pour empêcher les dégénérescences connues (gel du mouvement, dissolution, biais DC) et laisser émerger un comportement utile de la mémoire.

**e3_antifreeze** ajoutait un terme d'appariement de mouvement à la référence AF-stock (mm_ext). À l'étape 250, le résultat semblait être le premier résultat non dégénéré du projet:

| VBench, e3 step 250 | mem-ON | mem-OFF | AF-stock |
|---|---|---|---|
| SC | 0.914 | 0.935 | 0.936 |
| BC | 0.927 | 0.942 | 0.948 |
| DD | 0.996 | 1.000 | 1.000 |

L'entraînement ne dégradait plus le mouvement (mem-OFF égal au stock sur toutes les dimensions) et la mémoire ne gelait plus la vidéo (DD 0.996), au prix d'environ 2 points d'identité (SC 0.914 contre 0.935).

Ce résultat était **transitoire**. À l'étape 500, le même run avait gelé:

| e3, trajectoire mem-ON | step 250 | step 500 |
|---|---|---|
| DD (mouvement) | 0.996 | **0.046** |
| SC (consistance sujet) | 0.914 | **0.966** |

Le point crucial est que SC **monte** quand la vidéo gèle: une vidéo immobile est parfaitement « consistante ». C'est le piège de Goodhart observé en direct dans un seul run: l'optimiseur a trouvé que geler bat la métrique de consistance. Deux contrôles ont écarté les explications faciles. D'une part, l'inspection visuelle à l'étape 250 révélait déjà un scintillement de luminance par patch (+22.7 % d'écart-type temporel par patch par rapport au jumeau, netteté -28 %): le run n'était « propre » qu'aux yeux des métriques. D'autre part, une évaluation sur 6 prompts jamais vus a montré que le gel n'est **pas** du surapprentissage des 8 prompts d'entraînement: mem-ON held-out DD 0.072 (gèle aussi fort que les prompts entraînés à 0.046) alors que AF-stock reste à 0.994. Le gel est un **attracteur général du self-rollout DMD**, pas une mémorisation.

**e4_antiflicker** a été la dernière tentative de cette famille: une pile de contraintes conçue pour bloquer le scintillement (passe-bas temporel EMA sur l'injection, métrique de mouvement lag-2 aveugle au flicker, retrait de la moyenne temporelle de la dose, tripwire de détection). Le résultat a été encore plus net:

| e4, VBench | mem-ON @250 | mem-OFF @250 | mem-ON @500 | mem-OFF @500 |
|---|---|---|---|---|
| SC | 0.943 | 0.939 | 0.989 | 0.989 |
| DD | 0.450 | 0.942 | **0.000** | **0.000** |

À l'étape 500, les deux bras sont totalement gelés (DD 0.0) avec une SC record de 0.989: la pile anti-flicker a simplement renvoyé l'optimiseur vers le gel, encore plus vite.

**Conclusion de la série e: le whack-a-mole est structurel.** Chaque garde-fou ferme un mode de dégénérescence et l'optimisation en trouve un autre dans l'espace des fréquences laissé libre: le retrait du DC mène à la dissolution, l'anti-dissolution mène au gel, l'anti-gel mène au flicker, l'anti-flicker ramène au gel. Nous appelons ce constat la **triade de Goodhart**: le gel bat SC, le biais DC bat les métriques de dérive, le flicker bat DD (qui compte le scintillement comme du mouvement). L'ensemble des solutions accessibles à l'optimiseur sous ces contraintes est l'union des angles morts des métriques et des solutions inertes. La série e a donc été arrêtée, avec deux décisions: ne plus jamais conclure sur VBench seul, et changer de stratégie, en s'attaquant à la cause plutôt qu'aux symptômes.

## 4.4 c1_cotrain: co-entraînement pur DMD propre

Le diagnostic issu de la série e et de la littérature (VideoSSM notamment, qui fait fonctionner une mémoire globale sous un DMD fenêtré) désignait deux différences majeures avec les systèmes qui marchent: notre backbone était gelé (la mémoire ne pouvait pas être « absorbée » par le réseau) et nos garde-fous déformaient l'objectif. **c1_cotrain** est le premier run honnête de la nouvelle famille: tous les garde-fous à zéro (DMD pur), substrat d'entraînement sain de 248 000 prompts (au lieu des 8 prompts sur-appris), LoRA du backbone entièrement dégelée et co-entraînée avec la mémoire, injection bornée (normalisation relative, retrait du DC, gate borné à 0.1).

**Trajectoire (évaluation normale, écart tail-mean id_cos memon moins memzero, vérifié dans les courbes sauvegardées):**

| c1_cotrain | écart id_cos (tail 300) | prompts positifs | DD memon | DD memzero |
|---|---|---|---|---|
| step 250 | -0.123 | 0/6 | 0.878 | 0.872 |
| step 500 | **-0.019** | 3/6 | 0.739 | 0.678 |
| step 1250 | -0.119 | 1/6 | **0.228** | 0.683 |

Trois enseignements, dans l'ordre où nous les avons appris.

**Premier: pas de gel.** À l'étape 250, le mouvement est sain (DD environ 0.88, lag-2 très au-dessus du seuil de gel), sur les deux bras. La recette de co-entraînement propre échappe à l'attracteur qui a tué e3 et e4. C'est le gain net de c1.

**Deuxième: l'injection nuit d'abord, puis est neutralisée.** À 250, l'effet spécifique de l'injection sur l'identité est franchement négatif (-0.123, dommageable sur 6 prompts sur 6). À 500, cet écart est réduit à -0.019, à l'intérieur de la tolérance de parité de 0.03: le co-entraînement a appris au backbone à absorber la lecture mémoire, atteignant le niveau « do-no-harm » (ne pas nuire). Il faut être précis sur ce que cela signifie: do-no-harm veut dire **inerte**, pas utile. Sur une dérive simple (le sujet reste à l'écran, le cache KV et le sink suffisent), la mémoire n'a pas de travail net à fournir; c'est le « mur » diagnostiqué en amont.

Une parenthèse méthodologique importante: le script d'évaluation avait étiqueté l'étape 500 « ACTIVE-NEGATIVE » sur la base de l'id_cos à une seule frame finale et d'une comparaison au stock qui facturait à la mémoire la dérive commune de la LoRA. Deux audits indépendants du code (chemin d'inférence propre, mémoire chargée et vérifiée 36/36, memzero bit-exact) plus un recalcul en tail-mean ont montré que c'était un **artefact de métrique**, pas une régression réelle: la trajectoire honnête est bien -0.123 puis -0.019. Cette leçon (échantillonner une courbe bruitée en un point produit des verdicts instables) a fixé la métrique primaire du reste du stage.

**Troisième: le do-no-harm de l'étape 500 est une fenêtre transitoire.** À 1250, l'écart d'identité redevient nettement négatif (-0.119, 1/6) et surtout l'injection **tue le mouvement**: DD memon 0.228 contre memzero 0.683. Le bras memzero, lui, reste sain et stable (identité tail 0.784, supérieure même au stock 0.759): la recette de co-entraînement est propre, c'est bien **notre injection** qui est le problème, et le co-entraînement prolongé aggrave la dépendance du modèle à cette injection nocive. Le meilleur checkpoint de c1 est vers 500; continuer n'améliore rien.

**Probes d'inférence à coût nul (checkpoint 1250).** Avant de reconstruire quoi que ce soit, deux hypothèses de correctif ont été testées sans entraînement, directement à l'inférence:

| Probe c1 (ckpt 1250, sink-starvation) | écart id_cos (tail 300) | DD memon | DD memzero |
|---|---|---|---|
| sink réduit à 1 (la mémoire devrait aider) | -0.102 (1/6) | **0.211** (figé) | 0.650 |
| MEM_REJECT_STREAM (réjection par soustraction) | -0.127 (1/6) | 0.217 | 0.683 |

Les deux sont **négatifs**: même privée de son sink, la mémoire de c1 n'aide pas (elle nuit et fige le mouvement), et la réjection de flux n'améliore rien. Conclusion: c1 n'est pas rattrapable par un patch d'inférence; le défaut est architectural et il faut reconstruire.

## 4.5 c2_eviction: le correctif eviction-feed

L'autopsie de c1 a identifié un défaut de conception fondamental: la mémoire **écrivait le chunk courant**, c'est-à-dire exactement ce que la fenêtre d'attention voit déjà. Elle était structurellement redondante avec le cache KV, donc sans travail à faire, donc disponible pour être « goodhartée ». **c2_eviction** corrige cela: la mémoire écrit désormais le chunk **évincé**, celui qui vient de sortir de la fenêtre d'attention (FIFO par bloc, avec un décalage vérifié exactement égal à la fenêtre de 4 chunks). Elle devient le complément du cache: elle détient ce que le sink et la fenêtre ont perdu. S'y ajoute un co-entraînement avec cache-dropout (probabilité 0.3 de réduire le sink de 3 à 0 par rollout), pour que la mémoire reçoive un gradient quand le cache est privé de son ancre. Le reste est identique à c1 (DMD pur, garde-fous à zéro, substrat 248k, gate 0.1). L'implémentation a été validée avant lancement: selftest bit-exact quand le mécanisme est désactivé, revue indépendante du chemin BPTT/FSDP.

**Trajectoire sink-starvation (sink 1, la condition où la mémoire doit aider), tous chiffres recalculés depuis les courbes sauvegardées:**

| c2, sink 1 | écart id_cos (tail 300) | prompts positifs | DD memon | DD memzero | SC memon |
|---|---|---|---|---|---|
| c1 (référence) | -0.102 | 1/6 | 0.211 | 0.650 | 0.966 |
| c2 step 250 | -0.055 | 3/6 | 0.783 | 0.839 | 0.971 |
| c2 step 500 | -0.053 | 3/6 | 0.356 | 0.856 | 0.979 |
| c2 step 750 | **-0.040** | 2/6 | 0.239 | 0.311 | 0.979 |
| c2 step 1000 | **-0.253** | 0/6 | 0.911 | 0.711 | 0.958 |
| c2 step 1250 | -0.138 | 1/6 | 0.989 | 0.994 | **0.800** |

En évaluation normale (sink 3), l'étape 500 donne un écart de -0.071 (1/6) avec DD memon 0.589: sur la dérive simple, la mémoire reste légèrement nocive.

**Lecture en deux phases.**

*Phase 1 (250 à 750): un vrai progrès.* Le problème de mouvement de c1 est corrigé dès 250: memon DD 0.78, dynamique comme memzero (0.84), là où c1 figeait à 0.21. Le dommage d'identité est divisé par deux (-0.055 contre -0.102) et **converge** de façon monotone vers zéro (-0.055, -0.053, -0.040). À 750, l'analyse d'attribution est encourageante: la baisse de DD visible sur memon (0.24) touche **aussi** memzero (0.31), donc c'est l'attracteur d'auto-similarité du co-entraînement DMD qui dé-dynamise les deux bras, pas l'injection (l'écart memon moins memzero sur DD se réduit de -0.50 à -0.07). Tout indiquait une convergence vers « actif et do-no-harm », l'objectif que c1 avait manqué.

*Phase 2 (1000 et 1250): la bascule.* La convergence ne tient pas. À l'étape 1000, l'écart d'identité s'effondre à -0.253 (0/6 positifs, pire que c1) alors que le mouvement remonte à DD 0.91: le modèle est entré dans un régime de **flicker à haute énergie**, très dynamique mais incohérent (le sujet se métamorphose). À 1250 le régime persiste (id_cos memon 0.353 en tail, DD 0.99) et la SC chute à 0.800 contre 0.951 pour le stock: cette fois, même VBench voit le scintillement. La non-monotonie des deux colonnes DD (0.78, 0.36, 0.24, 0.91, 0.99) est la signature de l'oscillation gel-flicker déjà rencontrée dans la série e.

**Verdict honnête sur c2.** L'eviction-feed est une amélioration claire et mesurable par rapport à c1 aux checkpoints précoces et intermédiaires (mouvement réparé, dommage d'identité divisé par deux et décroissant), mais il n'atteint pas un do-no-harm **stable**: avec plus d'entraînement, le système retombe dans la dégénérescence par scintillement. À ce stade, l'interprétation par défaut était que le mur du data-free (aucun terme d'identité dans l'objectif DMD) finit toujours par gagner, l'eviction-feed n'ayant fait que retarder l'échéance. L'autopsie qui suit a changé cette interprétation.

## 4.6 Autopsie du flicker: une pathologie d'optimisation, pas le mur

Plutôt que de conclure sur les seules métriques d'évaluation, nous avons dépouillé la télémétrie d'entraînement complète de c2 (tfevents, étapes 0 à 1355). Le résultat contredit l'hypothèse « oscillation de Goodhart » et localise précisément la cause.

**Ce que la télémétrie montre.** Le flicker n'est pas une oscillation: c'est une **rampe monotone non amortie des paramètres lents** de la mémoire. La dérive des poids qkv fait croître la RMS de la lecture d'un facteur 8 sur le run; la norme du gate (gatew) monte linéairement de 0.30 à 1.76 sans jamais atteindre de plateau, et accélère vers l'étape 900. L'amplitude d'injection croît donc continûment jusqu'à franchir, vers l'étape 1000, un seuil de cohérence au-delà duquel la dispersion temporelle d'amplitude (notre canal ampdisp_temporal) explose: c'est le flicker. Trois suspects sont **innocentés** par la même télémétrie: l'état à poids rapides lui-même (statenorm épinglé à 0.999 tout le run, la renormalisation de l'écriture Muon fonctionne, contrairement à notre première hypothèse), le critique DMD et les normes de gradient (plats, donc ni cycle limite ni instabilité du critique). Les facteurs facilitants sont tous identifiés dans le code: learning rate constant (aucun scheduler), EMA de poids désactivée en mode LoRA, weight decay faible, et le plafond d'injection (mem_stream_scale) jamais atteint (pic à 0.37 pour un seuil de 1.0).

**Confirmation causale à coût nul.** Si la cause est une sur-amplitude de l'injection, la brider à l'inférence doit réduire le dommage sans aucun réentraînement. La probe `MEM_STREAM_SCALE=0.5` sur le checkpoint 1000 (sink 1) donne:

| ckpt 1000, sink 1 | écart id_cos (tail 300) | SC memon | DD memon |
|---|---|---|---|
| injection pleine | -0.253 (0/6) | 0.958 | 0.911 |
| injection bridée à 0.5 | **-0.120** (0/6) | 0.974 | 0.944 |

Le dommage est divisé par deux par un simple facteur scalaire à l'inférence. Il n'est pas annulé (la dérive des poids lents est déjà « cuite » dans le checkpoint), ce qui indique que le correctif doit s'appliquer **pendant l'entraînement**.

**Portée du diagnostic.** C'est le résultat le plus important de cette phase: la dégénérescence de c2 à 1000+ n'est pas la manifestation inévitable du mur data-free, c'est une **pathologie d'optimisation identifiée et bornable** (rampe non amortie de paramètres lents sur-pilotant l'injection). Elle relève d'outils standards: plafonnement d'amplitude, schedulers, EMA, clipping. La littérature fournit d'ailleurs une preuve d'existence: VideoSSM entraîne une mémoire globale sous le même DMD fenêtré data-free, de façon stable, avec précisément ce type d'hygiène (mémoire contractive, co-entraînement complet, EMA et schedules). c2 prouve donc que le data-free **non stabilisé** dégénère, pas que le data-free est impossible.

## 4.7 c3_stable: le pack de stabilité (en cours)

**c3_stable** traduit l'autopsie en un correctif d'entraînement: c2 (eviction-feed conservé) plus un « pack stabilité à parité VideoSSM » dont chaque élément cible la rampe identifiée, et non l'état à poids rapides (déjà sous contrôle):

1. `mem_stream_scale 0.5`: plafond dur de l'injection à 0.5 fois la RMS locale du flux résiduel (borne l'amplitude quelle que soit la dérive des poids lents, le levier le plus directement causal);
2. `lr_mem` réduit de moitié (5e-5): réduit le moteur de la rampe;
3. `weight_decay 0.05`: rappel des poids qkv/proj;
4. EMA de poids réactivée (0.99, trainable-only, à partir de l'étape 250; elle était désactivée en mode LoRA dans c1/c2), l'évaluation chargeant obligatoirement l'ombre EMA;
5. décroissance cosinus des learning rates à partir de l'étape 400 (planchers 0.1x pour la LoRA, 0.05x pour la mémoire): supprime la rampe tardive;
6. clipping explicite des gradients mémoire à 1.0 (le clipping FSDP les ignorait, ils étaient donc non clippés dans c1 et c2);
7. annealing du cache-dropout de 0.3 vers 0.1 sur 800 étapes;
8. `use_decay` (oubli des poids rapides) activé en couverture, secondaire d'après la télémétrie.

Le run part d'un redémarrage propre (base.pt + final.pt, mémoire réinitialisée), cible 2000 étapes. La validation pré-lancement a suivi le même standard que c2: selftest en 8 phases (bit-exact pack désactivé, aller-retour de sauvegarde EMA exact), et une revue de code indépendante qui a intercepté un bug bloquant avant lancement (l'ombre EMA n'était jamais sauvegardée en mode LoRA, ce qui aurait rendu tout l'amortissement invisible à l'évaluation).

**État au moment de la rédaction:** c3_stable a été lancé le 15 juillet 2026 et est **en cours d'entraînement** (4 GPU, sondes de démarrage saines: gradients mémoire et gradients de decay non nuls, EMA trainable-only active). Aucun checkpoint n'a encore été évalué; le dossier `reference/runs/c3_stable/` ne contient pas encore d'évaluation. La trajectoire (250, 500, ..., 2000, en conditions normale et sink-starvation) est à compléter. La règle de décision est pré-enregistrée: si c3 tient « actif, dynamique et do-no-harm » jusqu'à 2000 étapes, c'est la démonstration recherchée que la dégénérescence de c2 était bien une pathologie d'optimisation corrigeable; si c3 dégénère malgré la totalité des stabilisateurs standards, c'est au contraire la meilleure preuve accumulée en faveur du mur data-free. Dans les deux cas, une limite reste acquise: la stabilisation amortit, elle ne crée pas d'objectif; un effet **positif** sur l'identité (au-delà du do-no-harm) exigerait un changement de cible d'entraînement (reconstruction sur vidéo réelle, ou témoin non factorisé de type « Anchor Resurrection »), discuté en section suivante.

## 4.8 Bilan de la campagne

| Run | Hypothèse testée | Résultat | Enseignement |
|---|---|---|---|
| e1 / e2 | injection sur backbone gelé | artefact DC puis dissolution; VBench aveugle | l'amplitude d'injection découplée du flux est structurellement dangereuse; durcir les métriques |
| e3 / e4 | garde-fous anti-dégénérescence | gel @500 (DD 0.996 vers 0.046, SC monte); e4 gèle à DD 0.0 | whack-a-mole structurel; triade de Goodhart (gel bat SC, DC bat la dérive, flicker bat DD); attracteur DMD général, pas du surapprentissage |
| c1_cotrain | co-entraînement DMD pur, propre | pas de gel; injection nocive @250 (-0.123) neutralisée @500 (-0.019), puis re-dégradation @1250; probes de sauvetage négatives | le co-entraînement absorbe l'injection mais do-no-harm = inerte; la mémoire écrit un contenu redondant, défaut architectural |
| c2_eviction | écrire le chunk évincé (complément du cache) | mouvement réparé, dommage divisé par deux et décroissant (250-750), puis flicker (1000+) | amélioration réelle mais instable; l'autopsie montre une rampe d'optimisation corrigeable, pas le mur |
| c3_stable | pack de stabilité ciblant la rampe | en cours | trajectoire à compléter |

Aucun run n'a atteint l'objectif initial (une mémoire qui **améliore** l'identité à long terme). Mais la campagne a produit trois résultats solides et vérifiés: une caractérisation précise des modes de dégénérescence du self-rollout DMD data-free et de la manière dont ils trompent les métriques standards (la triade de Goodhart, avec le protocole d'évaluation durci qui en découle); l'identification du défaut architectural de la mémoire initiale (redondance avec le cache) et sa correction mesurable par l'eviction-feed; et le diagnostic télémétrique complet de l'instabilité résiduelle, avec sa confirmation causale à coût nul et son correctif en cours de test. La frontière exacte entre « pathologie d'optimisation » et « mur de l'objectif data-free » est précisément ce que c3_stable est en train de mesurer.
# 5. Méthodologie d'évaluation

Évaluer une mémoire longue pour la génération vidéo autoregressive est plus difficile que de l'entraîner : la quasi-totalité des faux positifs rencontrés pendant ce stage venaient de la mesure, pas du modèle. Cette section décrit le protocole final, construit itérativement après plusieurs verdicts erronés, et les leçons d'intégrité métrique qui l'ont motivé. Le harnais complet est versionné par run dans `reference/runs/*/eval*/` (un script `run_eval.sh`, un `metrics.py` autonome, des configs YAML par bras), de sorte que chaque évaluation est reproductible et auditable a posteriori.

## 5.1 Le protocole 3-way : mem-ON, mem-OFF, AF-stock

Chaque checkpoint est évalué avec trois bras, jamais deux :

1. **mem-ON (memon)** : le checkpoint entraîné, mémoire active. Config type : `memon.yaml` (couches mémoire `[5, 16, 27]`, `eviction_feed: true`, LoRA chargée depuis `checkpoint_model_000XXX/model.pt`).
2. **mem-OFF (memzero)** : le **même** checkpoint, même LoRA, même seed, mais avec l'injection mémoire mise à zéro via la variable d'environnement `MEM_ZERO_ALL=1`. Le code correspondant est dans `wan/modules/causal_model.py` (lignes ~652-658) : `mem_out = mem_out * 0.0` sur chaque frame, avec un message d'audit imprimé une fois par processus pour garantir que le bras est bien celui qu'on croit. C'est un **proxy inerte** : tout le reste (poids LoRA co-entraînés, cache KV, sink, RoPE) est strictement identique.
3. **AF-stock** : le vrai modèle Anchor-Forcing de référence, c'est-à-dire `memory_layers: []` avec le checkpoint public `final.pt` (config `stockaf.yaml`). Il est généré une fois sur les prompts held-out et réutilisé comme colonne fixe entre les évaluations.

**Pourquoi mem-OFF n'est pas AF-stock.** C'est un point qu'un audit à trois agents a dû trancher explicitement après un doute légitime sur la validité de la baseline. Deux vérités distinctes, à ne pas confondre :

- Le **code** du chemin mem-OFF est identique au code AF d'origine : le commit de base est byte-identique à l'upstream `vivoCameraResearch/Anchor-Forcing`, les checkpoints `base.pt`/`final.pt` correspondent aux sha256 de HuggingFace, et empiriquement la différence mem-OFF-vs-stock (même checkpoint, seed 42) tient entièrement dans le non-déterminisme bf16/flex-attention du modèle lui-même (MSE ~11.5 contre ~11.7 pour stock-vs-lui-même). Le zeroing n'a aucun effet de bord sur le KV, le sink ou le RNG.
- Mais les **poids** de mem-OFF sont notre checkpoint entraîné, pas le modèle stock : la LoRA a été co-adaptée à la présence de la mémoire pendant l'entraînement. Donc mem-ON vs mem-OFF est une **ablation propre de l'injection** (la seule différence est le vecteur injecté), tandis que mem-OFF vs AF-stock mesure la **dérive du co-entraînement LoRA/DMD** lui-même. Sans le troisième bras, on facturerait cette dérive à la mémoire, ou inversement on la manquerait.

Deux pièges pratiques rendent ce protocole non trivial : `memory_layers: []` fait crasher le chargement d'un checkpoint entraîné avec mémoire (les clés `generator_memory` ne mappent plus), donc le seul bras sans-mémoire propre sur un tel checkpoint est `MEM_ZERO_ALL=1` ; et le wrapper (`utils/wan_wrapper.py`) a un défaut de `memory_layers=(5,)` avec `gate_init=0.1`, si bien qu'une config qui omet à la fois `memory_layers` et `MEM_ZERO_ALL` exécute silencieusement une mémoire aléatoire. Le protocole impose donc de déclarer chaque bras explicitement.

## 5.2 L'évaluation sink-starvation

Le diagnostic central du projet (section 4) est que sur une dérive monotone, le sink d'attention (les 3 premières frames latentes, épinglées en permanence dans le cache KV) détient déjà l'ancre d'identité : la mémoire n'a pas de travail marginal, et son optimum sous DMD est de ne rien faire. Pour donner un travail mesurable à la mémoire, on l'évalue aussi en **privant le sink de l'ancre** : `sink_size: 1` (voire 0) au lieu de 3 dans la config d'inférence (cf. `eval*_sink1/memon.yaml`), sur un checkpoint entraîné avec cache-dropout précisément pour survivre à ce régime.

L'interprétation est directionnelle. Avec un sink affamé, la fenêtre d'attention perd l'ancre du sujet au fil du rollout ; si la mémoire accumule réellement le complément du cache (les chunks évincés de la fenêtre, ce que fait le mécanisme d'eviction-feed de c2), elle est le seul chemin de retour vers l'identité initiale, et mem-ON devrait battre mem-OFF sur la rétention d'identité. Si mem-ON n'apporte rien même là, la mémoire est inerte ; si mem-ON dégrade, l'injection est nuisible. C'est ce protocole qui a permis de voir que c2 corrige le blocage moteur de c1 (Dynamic Degree 0.78 contre 0.21 sous starvation au step 250) et divise par deux sa nuisance d'identité, résultats invisibles dans le régime sink complet où les deux bras se confondent.

## 5.3 Les métriques

Le `metrics.py` de chaque dossier d'évaluation est autonome (il n'importe pas le stack d'entraînement) et calcule, par vidéo et par bras :

**VBench (SC, BC, DD).** Subject Consistency, Background Consistency et Dynamic Degree, calculés via `vbench2_beta_long` en mode `long_custom_input`. Le DD est traité en priorité : c'est l'axe vivant à ces longueurs, là où SC/BC saturent au-dessus de 0.93 et deviennent aveugles (cf. 5.4).

**Motion lag-2, aveugle au flicker.** La détection de gel naïve (différence inter-frames lag-1) est trompée par un artefact réel observé dans nos runs : le flicker période-2, où le modèle alterne entre deux images. Le lag-1 y voit du mouvement ; le lag-2 (différence `|x[t] - x[t-2]|`) le voit correctement comme nul. La combinaison sépare trois régimes : lag-2 ~0 et lag-1 ~0 = **gel** ; lag-2 ~0 et lag-1 élevé = **flicker** (faux mouvement) ; les deux élevés = mouvement réel. Un flux optique Farneback optionnel sert de contre-vérification. Seuil pré-enregistré : `FREEZE_LAG = 0.010` en différence de pixels normalisée.

**Identité DINO.** Chaque frame est encodée par un DINOv2-vitb14 gelé (chargé exactement comme dans `model/streaming_training.py`), et on calcule la courbe de dérive complète `cos(DINO(f0), DINO(ft))` pour tout t, dont le point terminal `id_cos(f0, fN)`.

**Le tail-mean id_cos (memon moins memzero), la métrique primaire.** La leçon la plus coûteuse du stage : le verdict basé sur le `id_cos` single-frame (frame 0 contre dernière frame) est un **artefact**. Au step 500 de c1, le label automatique a basculé de « positif » à « ACTIVE-NEGATIVE » alors que deux audits indépendants et un recalcul ont montré que rien n'avait régressé : la dernière frame est un échantillon unique d'une courbe bruitée, et le basculement tenait à une oscillation de 0.005 de SC à l'intérieur de la tolérance de 0.02. La métrique corrigée est la **moyenne de la courbe id_cos sur la queue du rollout** (~300 dernières frames décodées, là où la dérive longue portée s'exprime), et surtout **l'écart apparié memon moins memzero** de cette moyenne, qui isole l'effet spécifique de l'injection. C'est cette quantité qui donne la trajectoire honnête de c1 (nuisance réelle -0.123 au step 250, 6/6 prompts négatifs, puis neutralisation à -0.019 au step 500) et de c2 (section 6). Le compte de prompts positifs/négatifs accompagne toujours la moyenne.

**Le noise floor 0.18.** Le delta apparié memon/memzero en distance cosinus DINO par frame est comparé à un plancher de bruit de 0.18, calibré empiriquement : c'est l'ordre de grandeur de la divergence que produit le seul non-déterminisme d'exécution accumulé sur un long rollout entre deux runs identiques. En dessous, la mémoire n'a fait « ~aucune différence » (verdict INERT) ; au-dessus, elle est ACTIVE et on peut alors demander si elle aide ou nuit. Sans ce plancher, on interpréterait du bruit.

**La méthode des paires.** Tous les bras d'un checkpoint sont générés avec le même seed (42), le même nombre de frames (240, le maximum reporté par Anchor-Forcing ; au-delà on est en extrapolation hors distribution), les mêmes prompts, via `inference/inference.py` (le seed est fixé globalement, les vidéos sont nommées par index de prompt pour l'appariement automatique). Le bruit de diffusion étant identique entre memon et memzero, la seule source de différence est l'injection : les deltas par frame sont donc de vraies mesures causales, pas des comparaisons entre échantillons. Les prompts sont **held-out** (6 prompts jamais vus à l'entraînement, `reference/runs/e3_antifreeze/heldout/prompts.txt`), ce qui a permis d'écarter l'hypothèse de sur-apprentissage : le gel observé sur e3 était identique sur prompts vus et non vus.

## 5.4 Leçons d'intégrité métrique

Ces choix ne sont pas des raffinements : chacun corrige un faux verdict réellement produit.

**SC/BC montent quand le mouvement baisse (Goodhart).** VBench Subject Consistency est essentiellement une similarité DINO entre frames adjacentes : une vidéo gelée, ou effondrée en bouillie uniforme, obtient un score de consistance quasi parfait. Nous l'avons constaté visuellement : des vidéos mem-ON totalement effondrées en fin de rollout (sujet disparu) scoraient 0.938 contre 0.978 pour la baseline nette. Plus généralement, le projet a documenté une **triade de Goodhart** : le gel bat la métrique de consistance, le biais quasi-DC bat la métrique de dérive, le flicker bat le Dynamic Degree (qui compte le clignotement comme du mouvement). Un score VBench n'est donc jamais interprété seul : toute hausse de SC/BC est suspecte tant que le DD et le lag-2 n'excluent pas une perte de mouvement, et la règle du projet impose une vérification humaine sur vidéos côte-à-côte (mem-ON | mem-OFF | AF-stock, dossiers `sidebyside/` de chaque éval) avant toute déclaration de victoire.

**Comparer memon à memzero, pas au stock.** Le verdict automatique initial comparait mem-ON à AF-stock, ce qui **facture la dérive LoRA à la mémoire** : la LoRA co-entraînée dérive du stock indépendamment de l'injection (memzero est lui-même ~-0.03 sous le stock en identité). L'effet de la mémoire est l'écart memon moins memzero ; l'écart memzero moins stock est un garde-fou d'attribution qui mesure la dérive du recette d'entraînement. Confondre les deux a produit au moins un faux « ACTIVE-NEGATIVE ».

**n=6, seed unique : un caveat permanent.** Toutes les trajectoires rapportées reposent sur 6 prompts held-out et un seul seed. Les écarts d'identité de l'ordre de 0.02-0.05 sont dans la zone où quelques prompts font basculer la moyenne ; la convention du rapport est de qualifier ces résultats de suggestifs, jamais de prouvés, et de toujours donner le décompte par prompt. Les seuils de décision (`NOISE_FLOOR=0.18`, `FREEZE_LAG=0.010`, tolérances de parité VBench 0.02 et de dérive 0.03) sont pré-enregistrés en tête de `metrics.py` et imprimés dans chaque rapport, précisément pour empêcher l'ajustement a posteriori.

## 5.5 Le harnais

Concrètement, une évaluation est un dossier `reference/runs/<run>/eval<step>[_sink1]/` contenant : les trois configs YAML (memon, memzero, stockaf), un `run_eval.sh` à point d'entrée unique et un `metrics.py`. Le script est **gardé** (il refuse de tourner tant que le checkpoint cible n'existe pas) et **idempotent** (chaque étape vérifie ses sorties et se saute si elles existent) : génération memon puis memzero via `torchrun ... inference/inference.py --config_path <arm>.yaml` (le bras memzero avec `MEM_ZERO_ALL=1` dans l'environnement), scoring VBench SC/BC/DD des deux bras, complétion des dimensions manquantes sur la colonne stock réutilisée, puis `metrics.py` qui produit le rapport texte, le JSON complet (y compris les courbes de dérive par prompt, ce qui a permis de recalculer le tail-mean sans regénérer les vidéos) et le verdict pré-enregistré. Une leçon d'exploitation mérite d'être notée : les dossiers d'éval étant dérivés les uns des autres par substitution de chemin, une substitution incomplète a une fois silencieusement ré-évalué le mauvais checkpoint ; la parade est de vérifier dans le log de run que le répertoire et le checkpoint annoncés sont les bons avant de lire le moindre chiffre.

# 6. Conclusions et travaux futurs

## 6.1 Ce qui fonctionne

**c2 (eviction-feed) corrige les deux défauts identifiés de c1.** Le run c1 avait établi deux problèmes précis de la mémoire co-entraînée : sous sink-starvation elle tuait le mouvement (DD 0.21, vidéo quasi gelée), et son injection nuisait à l'identité (écart tail-mean memon moins memzero de -0.102). Le correctif architectural de c2 (écrire dans la mémoire le chunk **évincé** de la fenêtre d'attention, via une FIFO par bloc, au lieu du chunk courant redondant avec le cache ; plus un co-entraînement avec cache-dropout pour que la mémoire gagne un gradient) répond à ces deux problèmes aux checkpoints tôt et moyens : au step 250 sous starvation, le mouvement est restauré (DD 0.78, contre 0.84 pour memzero) et la nuisance d'identité est divisée par deux (-0.055, 3 prompts sur 6 positifs) ; la trajectoire 250 vers 750 converge ensuite vers le do-no-harm (-0.055, -0.053, -0.040). C'est le premier régime du projet où une mémoire MLP à fast-weights co-entraînée est simultanément **active** (delta apparié au-dessus du plancher de bruit) et **non nuisible**, ce que c1 et toutes les variantes gelées avaient raté. Ce résultat est modeste mais réel, et il valide le diagnostic mécaniste : la mémoire de c1 n'avait littéralement rien d'utile à stocker (elle ré-écrivait ce que le cache détenait déjà) ; celle de c2 stocke le complément du cache et cesse d'être un pur perturbateur.

**Le flicker de c2 à partir du step 1000 est une pathologie d'optimisation, vraisemblablement corrigeable, pas le mur.** À partir du step 1000, c2 bascule dans un régime de scintillement incohérent (DD élevé 0.91-0.99 mais id_cos effondré, écart -0.253 puis -0.138, SC qui chute à 0.80), qui persiste au step 1250. L'autopsie des courbes d'entraînement montre que ce n'est **pas** une oscillation de Goodhart du fast-weight : la norme de l'état fast-weight reste épinglée (0.999) tout le run, le critic et les normes de gradient sont plats. C'est une **rampe monotone non amortie des paramètres lents** de la mémoire (RMS du read multiplié par 8, gate de 0.30 à 1.76, jamais de plateau) qui sur-alimente l'injection jusqu'à franchir un seuil de cohérence vers le step 1000. Les facteurs habilitants sont tous identifiés dans le code : learning rate constant sans scheduler, EMA de poids désactivée sur le chemin LoRA, weight decay faible, enveloppe `mem_stream_scale` jamais contraignante. Une sonde à coût nul le confirme : borner l'injection à l'inférence (`MEM_STREAM_SCALE=0.5` sur le checkpoint 1000) divise par deux la nuisance sans réentraînement. Le run c3 (pack de stabilité à parité avec les recettes publiées : EMA de poids, décroissance cosinus du lr, clipping de gradient dédié, decay du fast-weight, enveloppe de flux active) teste directement cette hypothèse ; il est en cours au moment de la rédaction. Si c3 tient le do-no-harm dynamique jusqu'au step 2000, la démonstration « mémoire co-entraînée stable et inoffensive en régime data-free » est acquise ; s'il oscille malgré tous les stabilisateurs standards, ce sera au contraire la meilleure évidence du mur.

## 6.2 Les vrais murs

Il faut être clair sur ce que ces progrès ne changent pas. Trois limites sont structurelles, pas des questions d'hyperparamètres.

**1. Sur la dérive monotone, la mémoire a peu de travail.** L'architecture Anchor-Forcing épingle les premières frames dans le sink d'attention : l'ancre d'identité est déjà détenue, gratuitement, pour toute la durée du rollout, et la littérature (LongLive, Self-Forcing++, Rolling Forcing) montre que KV roulant plus sink suffisent à 4 minutes sans mémoire apprise. Le travail marginal d'une mémoire n'existe que là où le sink perd l'ancre : notre éval sink-starvation le crée artificiellement, mais le cas d'usage réel est le **revisit** (le sujet sort du champ au-delà de la fenêtre, puis revient, scénario A puis B puis retour à A), que notre protocole de rollout monotone n'a jamais testé et qu'aucun de nos entraînements n'optimise.

**2. Injecter du « passé » auto-similaire tend à réduire le mouvement.** Une mémoire du flux généré ré-injecte par construction du contenu corrélé au présent ; le chemin de moindre résistance pour baisser la loss est d'amplifier cette auto-similarité, ce qui amortit la dynamique. C'est le mécanisme derrière le motion-kill de c1 et derrière la dé-dynamisation lente observée sur les **deux** bras (memzero aussi) du co-entraînement : l'attracteur d'auto-similarité appartient à la recette DMD en self-rollout, pas seulement à la mémoire.

**3. Le mur principal : le juge DMD ne supervise pas l'identité d'instance.** Les scores real/fake du DMD sont calculés par un teacher bidirectionnel appelé sur le chunk courant (~5 s) avec le prompt constant, sans cache ni sink. Il faut énoncer ce point avec précision, car sa version forte est fausse : le texte **est** une ancre globale (l'identité de catégorie, « un chien roux », est notée à chaque chunk via le CFG), et une mémoire globale co-entraînée **reçoit** bien un gradient du DMD fenêtré (son état au chunk i change la plausibilité du chunk i ; VideoSSM en est le contre-exemple vivant, qui réduit la dérive avec un pur DMD fenêtré). Ce qui n'est supervisé par rien, c'est l'**identité d'instance sous-spécifiée par le texte** : les attributs particuliers de ce chien-ci, que le prompt ne fixe pas, peuvent dériver librement sans que le juge le voie jamais. Conséquence : la stabilisation (c3) peut **amortir** le système, elle ne peut pas **rediriger** l'objectif. Le mieux atteignable en pur data-free sur dérive monotone est donc le do-no-harm ; un effet d'identité **positif** exige une cible qui contienne l'identité, soit une reconstruction sur vraie vidéo (la famille VideoSSM/MemGen : la cible est une vidéo réellement cohérente), soit un témoin non factorisé fabriqué sans données (l'idée « Anchor Resurrection » ci-dessous). Notre trajectoire expérimentale complète, du gel d'e3 au flicker de c2 en passant par la dissolution d'e2, est cohérente avec cette lecture : chaque garde-fou métrique déplace le mode d'échec sans le supprimer, parce que l'objectif ne contient pas la quantité qu'on veut préserver.

## 6.3 Positionnement par rapport à la littérature

Il serait malhonnête de présenter le mécanisme comme une contribution. VideoSSM (arXiv 2512.04519) réalise, sur le même backbone Wan2.1-1.3B et le même schéma DMD en self-rollout, une mémoire globale à récurrence gated-delta co-entraînée de bout en bout qui bat LongLive et Self-Forcing ; MemGen (2512.18741) diagnostique exactement notre mode d'échec (la solution dégénérée qui ignore l'histoire au profit du texte) et le corrige par une history-loss, avec un benchmark de revisit de scène ; nos correctifs c2 (écriture sur éviction) suivent la ligne AHN (2510.07318). La valeur de c2/c3 est celle d'une **reproduction contrôlée** : montrer qu'une mémoire à fast-weights peut atteindre le do-no-harm dans notre régime, en isolant précisément quel choix (écrire le complément du cache plutôt que le chunk courant) fait la différence.

La contribution défendable du stage est ailleurs :

- **Le diagnostic précis.** Notre configuration initiale échouait pour des raisons identifiées et vérifiées dans le code : backbone gelé, magnitude du read découplée de sa pertinence (double normalisation plus gate non borné), écriture redondante avec le cache, et surtout absence de terme d'identité dans l'objectif. La formulation finale, « on ne peut pas récompenser une mémoire longue avec un juge court-portée data-free, sauf à fabriquer un témoin ou à cibler de la vraie vidéo », est énoncée dans sa version précise (compatible avec le succès de VideoSSM) et étayée par une trajectoire expérimentale complète.
- **L'intégrité métrique.** La triade de Goodhart (le gel bat la consistance, le biais DC bat la dérive, le flicker bat le Dynamic Degree), la nécessité du 3-way avec proxy inerte, le tail-mean apparié contre le single-frame, le plancher de bruit : autant d'éléments réutilisables par quiconque évalue de la génération vidéo longue, et qui invalident plusieurs conclusions qu'une lecture naïve de VBench aurait signées.
- **Des négatifs audités.** Chaque échec (e2, e3, e4, c1) est documenté avec son mécanisme, ses confondeurs levés et ses sondes de falsification à coût nul, ce qui est la condition pour que le négatif ait une valeur scientifique.

## 6.4 Travaux futurs

Trois directions concrètes, par ordre de coût :

**1. La sonde re-ID à coût nul, puis Anchor Resurrection.** L'idée : aucun juge data-free factorisé n'est incontournable, mais une trajectoire réelle est un **témoin non factorisé** (identité et mouvement y sont joints, donc le gel n'y est pas optimal), et on peut en fabriquer sans données en donnant à une seconde copie du modèle un contexte privilégié. Concrètement : rollout A puis B puis retour à A ; la branche oracle ré-injecte le KV de l'ancre du sujet A au moment du retour (en réutilisant la réorganisation de KV et le Tri-Region RoPE propres à Anchor-Forcing) ; l'étudiant, lui, a perdu A par éviction et la mémoire est son seul chemin de retour ; la loss aligne les latents du revisit de l'étudiant sur ceux de l'oracle, en plus du DMD. C'est data-free, non factorisé par construction, et le travail demandé (« retenir plus d'ancres que le sink ») est exactement le créneau identifié en 6.2. La première étape ne coûte aucun entraînement : une sonde d'inférence A-B-A avec splice du KV au retour, scorée en re-identification (ArcFace pour les humains, DINOv2/DreamSim sur crops segmentés sinon, normalisée par la variabilité intra-A), avec règle de décision pré-enregistrée. Si les baselines re-identifient déjà bien, le travail n'existe pas et on s'arrête là ; si elles échouent et que le splice répare, la distillation d'oracle est justifiée.

**2. La reconstruction sur vraie vidéo.** La voie éprouvée par TTT-Video, VideoSSM et MemGen : pré-entraîner ou co-entraîner la mémoire avec une loss de reconstruction sur de vraies vidéos longues (par exemple MiraData), où l'identité est dans la cible par construction, avant ou en plus du DMD on-policy. C'est la solution la plus sûre mais la moins originale (elle est publiée, jusqu'à la variante « reconstruire pour se souvenir ») ; pour un effort solo à 4 GPU, elle vaut comme validation, pas comme contribution.

**3. Un benchmark revisit/re-ID.** Le manque le plus net côté évaluation : aucun benchmark public ne score le retour d'un **sujet** (par opposition au revisit de scène/caméra de MAG-Bench, qui est géométrique et en métriques pixel) en génération streaming, avec des embeddings de re-identification plutôt que de la consistance frame-adjacente. Un jeu de triplets A-B-A dont le segment B excède la fenêtre KV plus le sink, avec métrique de récupération d'identité au retour normalisée par un oracle, servirait à la fois de gate pour Anchor Resurrection et d'outil pour la communauté, et prolonge directement l'appareil d'intégrité métrique de la section 5.

**En une phrase.** Le stage n'a pas produit une mémoire qui améliore l'identité longue, et l'analyse explique pourquoi c'était improbable dans ce régime ; il a produit une mémoire co-entraînée qui cesse de nuire (c2, avec c3 en test pour la stabilité), un appareil d'évaluation qui résiste aux trois façons dont ces systèmes trompent leurs métriques, et un diagnostic précis qui désigne les deux seules sorties du mur : une cible contenant l'identité, ou un témoin qu'on fabrique soi-même.
