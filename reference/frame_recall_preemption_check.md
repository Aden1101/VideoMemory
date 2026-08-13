# Due diligence de nouveauté n°3 : « frame de rappel synthétisée » (état TTT O(1) écrit à l'éviction → DÉCODEUR qui synthétise une frame latente de rappel → splice événementiel au revisit → data-free, T2V pur)

Date : 2026-07-24. Méthode : WebSearch + WebFetch arXiv 2024-2026 (abstracts + HTML plein texte quand disponible). Complète les audits 1 (virtual_sink_preemption_check.md) et 2 (arl2_preemption_check.md) ; les papiers déjà traités là-bas (AdaState, MAG/MemGen, WorldKV, ARL, ISPA, OPSD-V, VideoSSM, 2604.10103, MemRoPE, DySink, WorldMem, MemoryPack, etc.) ne sont pas re-décrits, seulement re-cités quand ils bornent le delta.

Idée auditée (« design 2 ») : mémoire O(1) fast-weight MLP TTT écrite au fil de l'éviction du cache KV, dont le readout est un DÉCODEUR qui SYNTHÉTISE UNE FRAME LATENTE DE RAPPEL du sujet ; au revisit (sujet entièrement évincé), la frame décodée est réinjectée dans le contexte via recache (splice événementiel) et le backbone GELÉ la lit par attention comme n'importe quelle frame. Entraînement data-free : reconstruction des frames évincées auto-générées.

---

## 1. Adjacence PERSONNALISATION / référence : personne ne SYNTHÉTISE la référence depuis une mémoire interne

### 1.1 Famille « référence fournie par l'utilisateur » (hors périmètre mais à citer en related work)
IP-Adapter, ConsisID, MAGREF (2505.23742), Stand-In (2508.07901), Movie Weaver (2502.07802), PersonalVideo (2411.17048), RefAlign (2603.25743), Spatial-Temporal Decoupled Reference Conditioning (2606.02441), Identity-GRPO (2510.14256), **ARGUS (2606.11670)** (vérifié : « MLLM-selected image/video identity evidence » de l'UTILISATEUR → mosaïque 3×3 injectée en « negative-time read-only memory » dans l'espace token de Wan ; clip-level, pas de streaming). Toute la famille conditionne sur des références EXTERNES. Aucun ne fabrique la référence depuis le propre passé généré du modèle.

### 1.2 Famille « référence issue du propre passé généré » : elle EXISTE, mais toujours STOCKÉE réelle, jamais synthétisée
- **StreamingT2V (2403.14773)** : le précédent canonique. APM (Appearance Preservation Module) conditionne sur une **anchor frame = frame réelle du premier chunk généré** (CLIP image + cross-attention), FIXE pour toute la vidéo. C'est bien une « generated reference », mais : frame réelle stockée, choisie une fois, lecture continue, aucun état compressé, aucun revisit. **Helios (2603.04379)** : même principe (First Frame Anchor).
- **IAMFlow (2605.18733, « Training-Free Identity-Aware Memory »)** ⚠️ le plus proche en USE-CASE de l'aire 1. Vérifié plein texte : archive de **frames réelles générées + leur KV cache par couche**, indexées par entity-ID + scores VLM ; au retour d'un personnage, sélection greedy set-cover des frames archivées couvrant les IDs courants, **KV réels concaténés dans la memory bank avant le denoising** (sink 3 + buffer mémoire 2-4 + fenêtre locale 9 ; archivage quand le chunk local est évincé). **Training-free, zéro synthèse, zéro état appris.** C'est « notre pipeline revisit-identité », version retrieval de réel.
- **GroundShot (2606.20799)** : crops réels d'entités extraits des shots générés acceptés, stockés (référence canonique protégée + variantes), re-fournis à un générateur reference-conditioned (Ref2V) quand l'entité revient. Training-free, agentic, pipeline par shots (pas streaming KV).
- **EM-Vid (2605.23610)** : banque de latent patches indexée par entité (contenu généré réel compressé), sparse token conditioning, training-free, multi-shot.
- **VideoMemory (2601.03655)** : Dynamic Memory Bank de descripteurs visuels/sémantiques par entité, retrieval-update (abstract seul, mécanisme fin non публié en HTML ; à re-vérifier si cité).
- **LongLive-RAG (2606.02553)** : retriever APPRIS (autoencodeur latent offline + Window Temporal Delta Loss, backbone fixe, données générées) sur les latents passés, mais ce qui est réinjecté = **tokens de contexte réels appariés**, concaténés [sink ‖ retrieved ‖ local], récupération CONTINUE à chaque bloc, « no compressed memory state ».
- StoryDiffusion / ConsiStory : self-attention partagée entre images générées d'un même batch (lecture de ses propres tokens), pas de mémoire compressée, pas de streaming, pas d'éviction.

**Constat aire 1 : « la référence vient du propre passé généré » est PRIS (StreamingT2V, IAMFlow, GroundShot, EM-Vid, LongLive-RAG), mais TOUJOURS par stockage/retrieval de contenu RÉEL (frames, crops, KV, patches). Personne ne SYNTHÉTISE la référence depuis un état compressé appris. La case « self-reference générative » est libre.**

---

## 2. Adjacence KEYFRAME / ANCHOR mémoire : personne ne décode un état compressé en frame réinjectée

### 2.1 Echo-Forcing — 2605.16003 « A Scene Memory Framework for Interactive Long Video Generation » ⚠️ LE papier que le reviewer brandira
- Concept homonyme du nôtre : **« Scene Recall Frames »**. Vérifié plein texte : recall frame = **compression spatiale (fusion pondérée) des KV RÉELS** de M blocs stables d'une scène → K^rec/V^rec « spatially structured », stockés dans un **scene memory pool** (croît avec le nombre de scènes, pas O(1)).
- **Déclenchement ÉVÉNEMENTIEL au revisit** : routing par similarité de prompt, « recall » si la meilleure correspondance est une scène non-adjacente et s_max ≥ τ_rec → **les KV de rappel sont concaténés dans le cache KV**.
- **Training-free, backbone GELÉ** (appliqué à Self-Forcing et LongLive), donc trivialement « data-free ».
- Éval : subject consistency sur la vidéo entière + background consistency sur segments appariés de revisit de scène ; **pas d'isolation identité-sujet vs rappel de scène**, pas de re-ID.
- **Delta exact vs nous** : (i) aucune mémoire APPRISE, aucun état fast-weight, rien n'est synthétisé (KV réels compressés) ; (ii) O(nombre de scènes), pas O(1) ; (iii) écriture à la complétion de scène, pas à l'éviction ; (iv) granularité scène, pas identité sujet. **Mais il prend « recall frame + trigger événementiel au revisit + injection dans le cache KV + backbone gelé ». C'est notre baseline obligatoire.**

### 2.2 Echo-Infinity — 2606.04527 « Learning Evolving Memory for Real-Time Infinite Video Generation » ⚠️ resserre l'audit 1
- **Memory Queries apprenables, mises à jour par attention + gating AU MOMENT DE L'ÉVICTION** de la fenêtre locale (« write-on-eviction »), taille fixe (**O(1)**, « constant computation independent of video length »).
- Lecture, vérifiée plein texte : « 𝒬 is projected into K𝒬/V𝒬 by two linear maps... and **concatenated with the sink and local KVs before causal attention** » → **tokens K/V dans le cache**, lecture CONTINUE. Pas de frame décodée.
- Entraînement : end-to-end avec le DiT (Wan2.1-T2V-1.3B, **backbone entraîné, pas gelé**), causal-forcing DMD + streaming long tuning 60 s, prompts VidProM (rollouts auto-générés). Pas d'éval revisit/re-ID.
- **Conséquence pour l'audit 1** : la conjonction « état appris O(1) écrit à l'éviction → lu comme KV virtuels dans le cache, entraîné DMD self-forcing » est maintenant PRISE par un seul papier (AdaState ne couvrait que le sink-state, 2604.10103 que l'éviction en linear-attention). Le design « virtual sink » a encore rétréci. Le design 2 (frame décodée) n'est pas touché.

### 2.3 Le reste de la famille anchor/keyframe : réel ou géométrique, jamais décodé d'un état
- **DySink (2605.21028)**, **Relax Forcing (2603.21366)** (Sink/History/Tail, relaxed KV), **Stream-T1 (2605.04461)** (memory sinking par frontières sémantiques) : réorganisation/rétention de KV réels.
- **AnchorWeave (2602.14941)** : « retrieved local spatial memories », retrieval spatial de contenu réel.
- **Mirage / Latent Spatial Memory (2606.09828)** : mémoire spatiale 3D par back-projection de latents, et REVENDIQUE d'« éliminer la reconstruction pixel-space » : le champ va dans la direction OPPOSÉE au décodage en frame.
- **TinyHistory (2512.23851)** : embeddings d'historique légers appris en 2 étapes (pré-entraînés sur vraies vidéos) ; rien de décodé en frame réinjectée. (NB : le snippet de recherche titrait « Pretraining Frame Preservation in Autoregressive Video Memory Compression » pour le même ID, la page abs donne TinyHistory ; probable renommage v1→v2, re-vérifier le titre si cité.)
- **HiAR (2603.08703)** : ordre de denoising hiérarchique, pas une mémoire.
- Multi-shot par keyframes (EM-Vid, KeyVID 2504.09656, pipelines FLUX+I2V) : les keyframes sont des PLANS générés en avant, pas des rappels décodés d'une mémoire du passé.

**Constat aire 2 : « recall frame » existe (Echo-Forcing) mais = KV réels compressés retrievés ; « état appris écrit à l'éviction lu par le cache » existe (Echo-Infinity) mais = tokens KV continus. Personne ne DÉCODE un état de mémoire compressé en une frame (latente) remise en contexte.**

---

## 3. Adjacence RECONSTRUCTION d'évincé : la reconstruction existe comme LOSS, jamais comme acte d'inférence

- **Memento (2606.14667)** ⚠️ l'antériorité conceptuelle la plus proche de notre OBJECTIF d'entraînement. Vérifié plein texte : banque mémoire à **longueur fixe** de latents VAE (TopK remplace les vieux tokens), dual-query (identité + keyframes) lue par **cross-attention à l'inférence** ; « memory-based subject reconstruction » = **ℒ_recon d'ENTRAÎNEMENT uniquement** (débruiter une image du sujet conditionné sur la mémoire seule) ; « à l'inférence, aucune synthèse explicite de sujet depuis la mémoire ». Données : **vraies vidéos cinématographiques** (2 033 séquences), génération shot-by-shot, pas de streaming KV/éviction, pas d'éval revisit long. **Delta : leur reconstruction supervise CE QUE la mémoire retient ; la nôtre EST le readout d'inférence.**
- **MAG/MemGen (2512.18741, audit 1)** : même statut, loss de reconstruction pixel des blocs compressés à l'ENTRAÎNEMENT (390K vraies vidéos), à l'inférence les KV compressés sont lus directement, jamais re-décodés.
- **Composition of Memory Experts (2605.18813)** : LTM = **LoRA finetuné au test sur ses propres frames générées** (TTT véritable), recall testé par rollouts arrière reconstruisant les états visités ; mais readout = **influence des poids sur le score de diffusion**, pas de décodage-réinjection ; world model action-conditionné, vraies données (Memory Maze, Minecraft, RealEstate10K). Idem **SlowFast-VGen (2410.23277)** (temporal LoRA épisodique, action-driven, readout poids).
- **Generative replay / pseudo-rehearsal** (Shin 1705.08690 ; Dream Rehearsal 2607.19749 pour le RL continu) : la famille conceptuelle « synthétiser le passé pour le rejouer contre l'oubli » est vieille et établie, mais au niveau TÂCHE/continual-learning ; **jamais appliquée comme « rejouer une frame synthétisée DANS la fenêtre de contexte d'un générateur vidéo streaming »**. Bon cadrage à revendiquer : design 2 = « generative replay inside the context window ».
- **Fast Spatial Memory (2604.07350)**, **tttLRM (2602.20160)**, **Mem3R (2604.07279)** : états fast-weight LaCT/TTT (SwiGLU MLP en ligne) **décodés en RGB / Gaussian Splatting**, donc « état TTT → pixels » est publié... en **reconstruction 3D/4D**, où le décodage EST la sortie de la tâche, pas un rappel réinjecté dans le contexte d'un générateur. À citer : ça affaiblit la nouveauté « mécanisme brut state→frame » (elle existe cross-domaine, comme Dreamer/RSSM décode son état depuis 2019) mais ça prouve la faisabilité du décodeur depuis un état LaCT-like.
- **ISPA / ARL / AHN / OPSD-V (audit 2)** : distillation/absorption de l'évincé vers des poids ou un état linéaire, readout jamais en frames.

**Constat aire 3 : « reconstruire les frames évincées depuis un état compressé ET les remettre en contexte à l'inférence » n'existe nulle part. La reconstruction-des-évincés n'existe que (a) comme loss d'entraînement (MemGen, Memento), (b) comme tâche terminale (3D recon), (c) comme replay au niveau tâche (continual learning).**

---

## 4. VERDICT : (b) PARTIELLEMENT PRISE, mais le cœur génératif est LIBRE

**La conjonction exacte (état fast-weight O(1) écrit à l'éviction → frame de rappel SYNTHÉTISÉE par un décodeur → splice événementiel au revisit → data-free par reconstruction d'évincés auto-générés → T2V pur, backbone gelé) n'existe dans aucun papier trouvé.** Le composant discriminant, le readout GÉNÉRATIF (décoder l'état en une frame de rappel), n'est pris par personne en génération vidéo. Mais la triangulation est serrée :

| Face de l'idée | Prise par | Ce qui leur manque |
|---|---|---|
| Recall frame + trigger événementiel au revisit + injection cache KV, backbone gelé | **Echo-Forcing 2605.16003** (training-free) | KV réels compressés, pas de synthèse, pas d'état appris, O(scènes), pas de re-ID sujet |
| État appris O(1) écrit à l'éviction, lu par le cache, entraîné e2e DMD self-forcing | **Echo-Infinity 2606.04527** | Lecture continue en tokens KV, pas de frame décodée, backbone entraîné, pas de revisit |
| Reconstruction du sujet/des frames DEPUIS la mémoire comme objectif | **Memento 2606.14667**, MemGen 2512.18741 | Loss d'entraînement seulement, vraies données, pas de décodage à l'inférence |
| Référence tirée du propre passé généré, réinjectée au retour du sujet | **IAMFlow 2605.18733**, GroundShot, StreamingT2V-APM | Stockage/retrieval de RÉEL (frames/KV/crops), pas de synthèse, pas d'O(1) |
| État TTT/fast-weight décodé en pixels | Fast Spatial Memory 2604.07350 (+tttLRM, Mem3R) | Reconstruction 3D/4D, le décodage est la tâche, pas un rappel en contexte |
| Splice de contenu post-éviction dans le cache au revisit | WorldKV 2605.22718 (audit 1) | Stockage O(T) de KV réels indexés pose, world model, zéro apprentissage |

1. **Le papier que le reviewer brandira en premier : Echo-Forcing (2605.16003).** Il a le pipeline complet (pool de « Scene Recall Frames », routing événementiel au revisit, concat au cache, frozen backbone, sur LongLive/Self-Forcing) en training-free. Notre soutenance doit répondre à UNE question : **pourquoi SYNTHÉTISER depuis un état appris O(1) plutôt que compresser les KV réels ?** Réponses possibles : borne mémoire stricte O(1) vs O(scènes), consolidation multi-vues du sujet (la synthèse peut fusionner ce qu'aucune frame réelle unique ne contient), identité SUJET vs scène. Il faut la baseline Echo-Forcing-style (real-KV recall) dans le papier, et si elle fait aussi bien, la mémoire générative n'a pas de job (écho direct du verdict « memory has no job »).
2. **Deuxième ligne de rebuttal : Memento** (« la reconstruction depuis la mémoire, on l'a déjà comme objectif ») ; réponse : chez eux c'est une régularisation d'entraînement sur vraies données cinéma, chez nous c'est le MÉCANISME d'inférence, data-free. Et **Echo-Infinity** (« état à l'éviction e2e, on l'a ») ; réponse : readout tokens-KV continu vs frame décodée événementielle, backbone entraîné vs gelé.
3. **Ce qui reste réellement à nous** : (i) le readout génératif frame-de-rappel depuis un état fast-weight en génération vidéo, (ii) le cadrage « generative replay dans la fenêtre de contexte », (iii) l'éval re-ID d'IDENTITÉ SUJET au revisit en T2V pur (toujours personne : Echo-Forcing mesure la scène, MAG-Bench la caméra, IAMFlow ne sépare pas), (iv) le tout data-free sur backbone gelé.
4. **Risques** : (a) l'argument « pourquoi pas retrieval de réel ? » est structurel, pas cosmétique : c'est la même falaise que WorldKV vs nous à l'audit 1, maintenant instanciée par un papier training-free directement sur NOTRE backbone de base ; (b) une frame synthétisée depuis un état compressé sera FLOUE/moyennée, et le backbone gelé attendra dessus comme sur du vrai : risque d'injection d'artefacts (notre historique c1/c2 le prédit) ; (c) fenêtre concurrente TRÈS chaude : Echo-Forcing, Echo-Infinity, Memento, IAMFlow, GroundShot, EM-Vid datent tous de mai-juin 2026, re-scanner arXiv juste avant toute soumission.

Papier le plus proche, réponse en un mot : **Echo-Forcing (2605.16003)** pour le pipeline revisit-splice, **Memento (2606.14667)** pour l'idée de reconstruction-depuis-mémoire.
