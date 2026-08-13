# Due diligence de nouveauté : « virtual sink » (fast-weight O(1) → tokens K/V virtuels dans le cache d'un modèle vidéo streaming, résurrection post-éviction, data-free)

Date : 2026-07-22. Sources : arXiv 2024-2026 via WebSearch/WebFetch (abstracts + HTML plein texte quand disponible).
Idée auditée : mémoire fast-weight MLP TTT, état de taille FIXE O(1), dont le READOUT est **décodé en tokens K/V virtuels insérés dans le cache KV à côté du sink** (« sink appris/synthétique »), pour **ressusciter** un sujet entièrement évincé (revisit A→B→A), entraînée **data-free** (self-distillation type DMD/Self-Forcing).

Précédents LLM assumés (à citer, hors périmètre) : Titans-MAC (2501.00663), RMT, Compressive Transformer, Infini-attention, Memorizing Transformers. **Ajouter à cette liste** (trouvés pendant l'audit, côté LLM) :
- **Kwai « Summary Attention » (2604.24432)** : summary tokens APPRENABLES qui compressent l'historique, K/V dans le cache, O(n/k). LLM uniquement, aucune application vidéo.
- **Latent Recurrent Transformer (2605.26797)** : langage (cs.CL) ; la variante « KV Projection » projette un état récurrent dans les espaces K et V d'une couche via des projections apprises → l'attention attend l'état comme entrées KV supplémentaires. **C'est le précédent LLM le plus proche du mécanisme exact « état → K/V décodés »** (extrait du plein texte via résumé de recherche ; l'abstract ne le détaille pas, à vérifier au moment de citer).

---

## 1. Les 7 papiers vidéo nommés : mécanisme de LECTURE exact

| Papier | Mécanisme de lecture | Verdict lecture-KV ? |
|---|---|---|
| 2604.10103 | Branche linear-attention parallèle | NON |
| MemFlow 2512.14699 | Attention sparse sur KV RÉELS retenus | KV réels, pas synthétiques |
| AdaState 2605.30349 | KV d'un état latent en position 0 du cache (sink) | **OUI, KV synthétique au sink** |
| ARL 2605.16579 | Branche linear-attention gated récurrente | NON |
| ISPA 2607.00712 | Modulation de poids (paramétrique) | NON |
| VideoSSM 2512.04519 | Branche résiduelle gated (Q·M) | NON |
| MemGen/MAG 2512.18741 | KV compressés préfixés, self-attention standard | **OUI, KV synthétiques en préfixe** |

### 1.1 arXiv 2604.10103 — « Long-Horizon Streaming Video Generation via Hybrid Attention with Decoupled Distillation »
- Mémoire : état K/V compact qui **absorbe incrémentalement les tokens évincés** (écriture à l'éviction, comme notre c2).
- **Lecture : linear attention, branche PARALLÈLE**, PAS des tokens dans le cache. Eq. 4 : `O_history = Linear(RoPE(φ(Q))·L / φ(Q)·H)` (produit requête×matrice d'état accumulée L, normalisé par H appris) ; sortie finale `O = O_history + O_local` (Eq. 7, somme avec la block-sparse attention locale).
- Entraînement : distillation découplée (phase DMD dense, puis activation linear+sparse). Pas d'éval revisit/re-ID.
- **Différence avec nous** : même règle d'écriture (éviction), mais readout = canal linear-attention, jamais décodé en tokens K/V softmax. Pas de résurrection.

### 1.2 MemFlow — 2512.14699 « Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives »
- Banque mémoire = **KV RÉELS** des chunks générés (« aggregating historical visual token (KV cache) from streamingly generated chunks »). Mise à jour = retrieval sémantique par le prompt + dédoublonnage heuristique (KV de la 1re frame du chunk précédent comme prototype). **Aucun état fast-weight appris.**
- Lecture : attention sparse top-k sur les tokens de la banque (« only activate the most relevant tokens in the memory bank for each query in the attention layers »).
- Entraînement : streaming long-tuning avec DMD. Pas d'éval revisit.
- **Différence** : retrieval de KV réels, pas de génération de KV par une mémoire ; croissance non-O(1) de la banque.

### 1.3 AdaState — 2605.30349 « Self-Evolving Anchors for Streaming Video Generation » ⚠️ LE PLUS PROCHE (mécanisme)
- État = **1 frame latente cachée** (Fs=1), dénoisée conjointement avec le contenu à chaque chunk, jamais rendue.
- **Lecture : « the adaptive state's clean KV is written to position 0 and carried to subsequent chunks via the state recurrence »** — le KV de l'état occupe la **position sink privilégiée** du cache et est attendu par la self-attention standard. C'est un **sink appris/auto-évolutif dans le cache KV**. « The KV cache serves as the carrier, requiring no external module. »
- Écriture : par le DÉNOISING lui-même (l'état part du bruit, est dénoisé avec le contenu, sa version clean écrase la position 0) ; récurrence détachée aux frontières de chunk. **Pas de fast-weight, pas d'écriture par gradient/TTT, pas de déclenchement à l'éviction.**
- Entraînement : **DATA-FREE** (self-distillation DMD, horizon-weighted `w_i = 1+α·i/(N-1)`) ; « No separate loss is applied to the state; gradient reaches it entirely through the attention that content frames pay to it. »
- Éval : consistance/dynamique standard, **PAS de revisit A→B→A ni re-ID**.
- **Différence exacte avec nous** : (i) l'état est une latente de la dimension d'une frame dénoisée PAR le backbone, pas un module fast-weight MLP externe à règle d'écriture gradient/TTT ; (ii) capacité = 1 frame latente (pas un MLP profond) ; (iii) pas d'écriture à l'éviction (l'état évolue à chaque chunk, il ne capture pas spécifiquement ce qui sort de la fenêtre) ; (iv) aucun objectif ni éval de résurrection. **Mais le claim « KV synthétique appris en position sink, data-free, streaming vidéo » est pris par ce papier.**

### 1.4 ARL — 2605.16579 « Attend Locally, Remember Linearly »
- « An inter-frame gated recurrent linear branch that maintains a fixed-size state for streaming context. »
- **Lecture : canal linear-attention** (branche parallèle à la softmax de fenêtre), pas de tokens KV. Pas d'éval revisit.

### 1.5 ISPA — 2607.00712 « Instance-Specific Parametric Absorption »
- Écriture : problème moindres-carrés en forme close → **modulation de poids instance-spécifique** qui compense l'historique manquant (absorption paramétrique de l'histoire dans les poids, explicitement « beyond external token-level cache management »).
- **Lecture : à travers les poids modulés**, aucun token KV. C'est le pôle opposé au nôtre (poids purs vs tokens purs). Concurrent très récent (juillet 2026).

### 1.6 VideoSSM — 2512.04519
- Lecture (Sec. 4.2.2) : `H_global_t = Swish(g_out_t ⊙ RMSNorm(Q_t·M_t))`, fusion **résiduelle gated** `H_fused = H_local + γ_t·H_global` (Eq. 10) avec router gate position-aware. **PAS de tokens dans le cache KV.**
- Entraînement : 2 étapes (régression sur trajectoires teacher, puis DMD long self-rollout). Pas d'éval revisit (reconnu en conclusion).

### 1.7 MemGen/MAG — 2512.18741 « Memorize-and-Generate » ⚠️ LE PLUS PROCHE (use-case)
- Mémoire = le backbone lui-même utilisé en ENCODEUR : compresse chaque bloc de 3 frames en un cache KV d'~1 frame (compression 3×). **Le cache compressé CROÎT linéairement avec la longueur (O(T/3)), PAS O(1). Pas de fast-weight.**
- **Lecture : les K/V compressés sont préfixés comme contexte historique et attendus par la self-attention standard** du générateur (KV synthétiques dans le chemin d'attention).
- Entraînement : **loss de reconstruction pixel** (reconstruire toutes les frames du bloc depuis le cache compressé), **390K vraies vidéos (VPData)** → **PAS data-free**.
- Éval : **MAG-Bench = benchmark de REVISIT explicite** (« videos with camera trajectories that leave and subsequently return to the scene ») — c'est notre use-case résurrection, pris au niveau scène/caméra.
- **Différence exacte** : mémoire non-bornée (O(T)), entraînement supervisé par reconstruction sur données réelles, pas d'état fast-weight, revisit = trajectoire caméra (scène), pas identité sujet en text-to-video.

---

## 2. Recherche large : toute mémoire vidéo qui APPREND à produire des tokens/K-V attendus par l'attention

### Tokens synthétiques dans le cache (la famille dangereuse)
- **MemRoPE — 2603.12513 « Training-Free Infinite Video Generation via Evolving Memory Tokens »** : compresse les CLÉS passées par **EMA** en un jeu fini de memory tokens dans un cache de taille fixe (streams long-terme + court-terme) ; « Online RoPE Indexing » découple le RoPE des clés cachées (appliqué au moment de l'attention) pour éviter l'annulation de phase. Lecture = softmax attention standard sur ces tokens synthétiques. Claims de préservation d'identité sujet. **MAIS : training-free, EMA fixe, aucun état fast-weight appris, pas d'écriture par gradient, pas d'éval A→B→A.** C'est « tokens synthétiques dans le cache » sans apprentissage.
- **DiT-Mem — 2511.19229 « Learning Plug-and-play Memory for Guiding Video Diffusion Models »** : encodeur mémoire appris (3D CNN + filtres + self-attention, 150M params) qui mappe des vidéos de référence en un **jeu compact de memory tokens concaténés dans les couches de self-attention du DiT** (Q/K/V avec les tokens vidéo). Backbone gelé. **MAIS : DiT bidirectionnel OFFLINE, injection de connaissance physique/monde, pas de streaming, pas de cache KV autoregressif, pas d'éviction/résurrection, entraîné sur 10K échantillons.**
- **RELIC — 2512.04040 (world model interactif)** : « highly compressed historical latent tokens encoded with both relative actions and absolute camera poses **within the KV cache** », attendus par self-attention ; distillation self-forcing memory-efficient. **MAIS : compression de l'historique réel (pas un état fast-weight génératif), conditionné poses/actions, données curées, pas de re-ID sujet.**

### Sinks en vidéo : tous des tokens RÉELS
- **LongLive (2509.22622)** : frame sink = vraies frames initiales. **Deep Forcing (2512.05081)** : training-free, moitié de la fenêtre en sink persistant (tokens réels) + réalignement RoPE. **DySink (2605.21028)** : frame sinks dynamiques = vraies frames historiques **récupérées** d'une banque compacte (adaptive retrieval + réalignement de phase RoPE). **Anchor-Forcing (2603.13405, notre base)** : Anchor Memory = réorganisation KV + Tri-Region RoPE sur tokens réels. → **Aucun sink APPRIS/synthétique généré par une mémoire dans la littérature vidéo trouvée, sauf AdaState (état dénoisé) ; aucun décodé depuis un fast-weight.**

### Résurrection/revisit : pris, mais jamais par une mémoire apprise O(1)
- **WorldKV — 2605.22718** : training-free ; **stocke les chunks KV RÉELS évincés** en GPU/CPU, les récupère par correspondance caméra/action au moment du revisit et les **réinsère dans une « retrieved region » de la fenêtre d'attention** sans ré-encodage. **Éval loop-closure explicite** (frame de revisit vs frame de première visite, PSNR/SSIM/LPIPS/FID). → prend « résurrection post-éviction via le cache KV », mais par STOCKAGE O(T) + retrieval indexé pose, zéro apprentissage.
- **MAG-Bench (MemGen, ci-dessus)** : revisit de scène par trajectoire caméra, mémoire entraînée sur données réelles.
- **WorldPack — 2512.02473** : trajectory packing + memory retrieval, benchmark **LoopNav** (Minecraft, consistance long-terme/loop-closure).
- Famille retrieval world-models : **WorldMem (2504.11454)** (banque de frames + poses, lecture par attention sur frames mémoire, revisit Minecraft), **Context-as-Memory (2506.03141, SIGGRAPH Asia 2025)** (retrieval de frames de contexte par FOV caméra), **Long-term Spatial Memory (2506.05284)** (mémoire spatiale géométrique), **MemLearner (2606.31734)** (query tokens appris pour récupérer le contexte, données annotées en poses). → tout est retrieval de contenu RÉEL, indexé caméra/pose, en setting world-model conditionné actions.
- Constat transversal : **le revisit « identité d'un SUJET » (pas retour caméra) en text-to-video streaming n'est évalué nulle part** ; les benchs existants (MAG-Bench, LoopNav, WorldKV) sont tous des retours de point de vue caméra.

### Mémoires à état fixe en vidéo : readout toujours HORS cache KV
- **Tensor Memory — 2605.27686** : état voxel 3D récurrent de taille fixe (langage+image+vidéo) ; lecture = échantillonnage continu + **fusion résiduelle gated dans le flux de tokens**, explicitement PAS des K/V de cache.
- **MemoryPack / Pack-and-Force — 2510.01784** : état compressé appris ψ de taille fixe, mis à jour itérativement ; lecture = **CROSS-attention** (sortie de Memorize en query, ψ en K/V) — canal parallèle, pas le cache de self-attention.
- **Memento — 2606.14667** : banque mémoire à double query (identité + keyframes), reconstruction du sujet depuis la mémoire seule (« A memory bank faithfully preserving a subject should support reconstructing that subject from memory alone ») ; pipeline de données cinématiques dédié → pas data-free ; canal séparé, pas d'injection KV documentée ; pas fast-weight.
- **TTT-Video (2504.05298), LaCT (2505.23884)** : couches TTT/fast-weight lues par application directe f(q) en branche (résiduelle/hybride fenêtre), jamais décodées en tokens KV. **« Forget, Anticipate and Adapt » (2606.26515)** : TTT vidéo mais COMPRÉHENSION (segmentation/classif), lecture par poids adaptés.
- **Long-Context State-Space Video World Models (ICCV 2025, Po et al.)** : état SSM interne, canal SSM.

### Compression d'historique réel en tokens (famille non-apprise ou à schedule fixe)
- **FramePack (2504.12626)**, **Helios (2603.04379)** (Multi-Term Memory Patchification) : historique distant patchifié/poolé en tokens grossiers lus par attention — schedule fixe, tokens dérivés du réel. **TokensGen (2507.15728)** : tokenizer vidéo → tokens condensés guidant via cross-attention, offline 2 étapes. **VideoMLA (2605.30351)**, **Forcing-KV (2605.09681)**, **Future Forcing (2605.30083)**, **Pyramid Forcing (2605.13111)** : compression/éviction/quantization de cache = efficacité, pas de mémoire générative.
- Surveys utiles : **« On Memory: A comparison of memory mechanisms in world models » (2512.06983)** ; liste GitHub « Awesome-World-Models-with-Memory » (taxonomie 3D-as-memory / feature-as-memory / context-as-memory).

---

## 3. VERDICT : (b) PARTIELLEMENT PRISE

**La conjonction exacte (état fast-weight O(1) à écriture TTT/gradient → readout DÉCODÉ en tokens K/V virtuels persistants au sink d'un cache KV vidéo streaming → résurrection A→B→A → data-free) n'existe dans aucun papier trouvé.** Mais chaque face est prise séparément, et la surface restante est étroite :

1. **AdaState (2605.30349)** prend « KV synthétique appris en position sink, data-free, streaming » — c'est LA citation qui tue le claim mécanisme seul. Delta restant vs eux : writer fast-weight/TTT à l'éviction (vs état-latente dénoisée par le backbone), capacité au-delà d'1 frame latente, et l'objectif/éval résurrection qu'ils n'ont pas.
2. **MAG/MemGen (2512.18741)** prend « KV synthétiques lus par l'attention + benchmark de revisit » — delta restant : O(1) (leur cache croît), data-free (eux : reconstruction sur 390K vidéos), identité SUJET en T2V (eux : retour caméra).
3. **WorldKV (2605.22718)** prend « résurrection post-éviction par réinsertion dans le cache KV » en training-free — delta : mémoire APPRISE O(1) vs stockage O(T) de KV réels indexés pose.
4. Toutes les mémoires à état fixe en vidéo (VideoSSM, ARL, 2604.10103, Tensor Memory, LaCT/TTT-Video, MemoryPack) lisent par linear-attention/résiduel-gated/cross-attention — **personne ne décode un état fast-weight en tokens K/V softmax** ; et côté LLM le mécanisme état→KV existe déjà (Titans-MAC, Latent Recurrent Transformer « KV Projection », Kwai Summary Attention), donc la règle de lecture seule n'est pas une nouveauté de classe.
5. Le claim défendable = la conjonction complète, vendue comme : **premier readout fast-weight décodé en KV-sink dans un modèle vidéo streaming + première résurrection d'IDENTITÉ SUJET post-éviction apprise data-free** (les benchs revisit existants sont tous caméra/pose, en world-models conditionnés actions). Prévoir le rebuttal AdaState (« vous avez changé l'updater ») et MAG (« votre éval existe ») dès l'écriture. Attention : AdaState/ISPA/DySink/WorldKV datent de mai-juillet 2026 → fenêtre « concurrent work », re-vérifier arXiv avant soumission.
