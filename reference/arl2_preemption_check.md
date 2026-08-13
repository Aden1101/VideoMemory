# Due diligence de nouveauté : "distiller le contexte évincé du KV cache dans une mémoire compressée via self-distillation" pour la GÉNÉRATION VIDÉO

Date de vérification : 2026-07-22. Méthode : WebFetch des abstracts/HTML arXiv + WebSearch large. Tous les faits ci-dessous sont vérifiés à la source (arxiv.org), pas de citation de seconde main.

## TL;DR

- L'agent A a confondu DEUX papiers réels mais son fond est correct : le créneau vidéo n'est PAS libre. "ARL2" = arXiv **2605.16579 v2** ("Attend Locally, Remember Linearly", d'où l'acronyme ARL). 2604.10103 est un AUTRE papier vidéo, lui aussi pertinent.
- L'agent B a raison sur AHN (LLM texte seulement) et sur RELIC (données Unreal), mais TORT sur la conclusion "créneau vidéo libre" : il a raté ARL, ISPA, OPSD-V et 2604.10103.
- Verdict : **(b) partiellement pris, et le delta restant est mince.** Chaque composant de l'idée est publié séparément en vidéo ; seule la synthèse exacte (module mémoire APPRIS, écrit à l'éviction, entraîné uniquement par self-distillation cache-affamé-vs-cache-complet sur vidéo auto-générée, style AHN-pour-la-vidéo) n'existe pas dans UN papier.

---

## 1. Vérification des deux IDs de l'agent A

### arXiv 2604.10103 : "Long-Horizon Streaming Video Generation via Hybrid Attention with Decoupled Distillation"
- Li, Yang, Ai, Wu, Wen, Peng, Zhang. Soumis 11 avril 2026 (v2 : 28 avril 2026). cs.CV.
- **Vidéo, oui. Éviction → état compressé, oui, explicitement** : "we maintain a dynamically updated KV state that aggregates evicted frames. The linear state update is performed when a frame is evicted from the cache." L'état linéaire (matrices L, H) absorbe les frames évincées.
- **MAIS pas de self-distillation cache-réduit-vs-cache-complet** : le teacher est le modèle de diffusion BIDIRECTIONNEL classique (pipeline DMD deux phases, "decoupled" = phase 1 DMD en attention dense, phase 2 activation de l'attention hybride linéaire + block-sparse). Pas de perte dédiée à la compression : l'état est entraîné indirectement via la perte DMD globale.
- Données : prompts VidProM (auto-génération style Self-Forcing), pas de vidéos réelles.
- Delta vs notre idée : le mécanisme d'écriture-à-l'éviction est pris ; l'objectif self-distillation (student au cache réduit matche le MÊME modèle au cache complet) n'y est pas.

### arXiv 2605.16579 : "Attend Locally, Remember Linearly: Linear Attention as Cross-Frame Memory for Autoregressive Video Diffusion" (= le "ARL"/"ARL2" de l'agent A)
- Kunyang Li, Mubarak Shah, Yuzhang Shang. v1 15 mai 2026, v2 20 mai 2026. cs.CV.
- **Vidéo autoregressive, et c'est le hit le plus proche de l'OBJECTIF self-distillation** :
  - Student = modèle hybride : softmax locale + branche linéaire récurrente gated à état de taille fixe pour le contexte streaming ("an inter-frame gated recurrent linear branch that maintains a fixed-size state for streaming context").
  - Teacher = le MÊME modèle de base (Causal Forcing) full-attention, GELÉ.
  - Distillation en 2 étages : (1) per-layer, MSE entre sorties teacher/student couche par couche "under the intact teacher context" ; (2) joint, matching du champ de vitesse : ℒjoint = ‖v_student − v_teacher‖².
  - **Data-free** : "Stage 2 jointly distills the hybrid student against the frozen Causal Forcing teacher for 12K steps on 6K teacher-synthesized videos." Les vidéos d'entraînement sont générées par le teacher lui-même.
- Nuance : PAS d'écriture explicite du contexte "évincé" (pas d'événement d'éviction) ; c'est une conversion d'architecture (l'attention linéaire remplace la softmax pour l'inter-frame), la compression du long contexte est implicite dans l'état récurrent.
- Delta vs notre idée : ARL fait "student à état borné distille depuis soi-même au contexte complet, sur données auto-générées" = le cœur de notre pitch. Il ne fait PAS "écriture déclenchée par l'éviction d'un chunk du KV cache" ni "module mémoire ajouté à backbone intact" (il remplace des couches d'attention).

**Conclusion agent A** : citations incohérentes (deux papiers distincts confondus sous un seul nom), mais la substance de sa claim tient : 2605.16579 préempte l'objectif self-distillation en vidéo, 2604.10103 préempte l'écriture-à-l'éviction.

## 2. Vérification AHN (claim de l'agent B)

### arXiv 2510.07318 : "Artificial Hippocampus Networks for Efficient Long-Context Modeling"
- ByteDance Seed (Fang, Yu, Zhong, Ye, Xiong, Wei). v1 8 oct 2025, v2 17 déc 2025.
- **LLM texte UNIQUEMENT** (Qwen etc., instancié Mamba2/DeltaNet/GatedDeltaNet). Confirmé.
- Mécanisme = exactement notre schéma, mais en texte : fenêtre glissante de KV cache = mémoire court terme sans perte ; l'AHN "recurrently compresses out-of-window information into a fixed-size compact long-term memory" ; entraînement = "an efficient self-distillation training method where the base model's all parameters are frozen and only the parameters from AHNs are optimized."
- L'agent B a raison sur ce point. Aucun papier "AHN-pour-la-génération-vidéo" trouvé (recherche dédiée : rien).

## 3. Vérification RELIC (claim de l'agent B)

### arXiv 2512.04040 : "RELIC: Interactive Video World Model with Long-Horizon Memory"
- Modèle 14B, monde interactif. Mémoire = "highly compressed historical latent tokens encoded with both relative actions and absolute camera poses within the KV cache" (tokens compressés DANS le cache, pas un état/mémoire externe alimenté par l'éviction). "Full-context distillation" = self-forcing memory-efficient sur teacher bidirectionnel étendu au long horizon.
- Données : "trained on a curated Unreal Engine-rendered dataset". Confirmé : synthétique Unreal, pas data-free auto-généré, et pas de distillation du contexte évincé vers une mémoire.
- L'agent B a raison sur la caractérisation de RELIC, tort sur la conclusion générale.

## 4. Recherche large : autres papiers proches (2025-2026)

### arXiv 2607.00712 : "Towards Memory-Efficient Autoregressive Video Generation via Instance-Specific Parametric Absorption" (ISPA) -- LE hit le plus dangereux
- Fu, Li, Hu, Wang, So, Dai, Chu, Han. Soumis 1er juillet 2026.
- Abstract, mot pour mot : "we propose Instance-Specific Parametric Absorption (ISPA), a novel framework that **shifts the KV cache compression from discarding to distilling**" ; absorbe le contexte historique DANS LES POIDS du modèle en faisant passer des couches de Full-Attention à Local-Attention.
- Mécanisme : phase de warmup à l'inférence, monitoring de l'écart entre sorties attention globale vs locale, puis à la transition résolution en forme close (moindres carrés) d'une modulation de poids ΔW telle que la sortie du flux local matche la sortie du flux complet. **Self-distillation du même modèle (local vs full) au test-time, par instance, sans données** (opère sur la vidéo en cours de génération : de facto data-free).
- Résultats : -50% de KV cache "near-lossless (<1%)" ; ablation : sans l'absorption paramétrique → "catastrophic Subject Mismatch" (perte d'identité). Bases : LongLive 1.3B, Reward-Forcing 1.3B, Krea-Realtime 14B, LiveAvatar ; VBench-Long.
- **Limites/delta** : absorption UNE SEULE FOIS au point de transition ("At the dividing frame T_warm... For all subsequent frames t>T_warm, the designated K layers transition into L-Layers"), pas d'écriture continue à chaque éviction ; forme close, pas de module mémoire appris offline ; training-free.
- C'est littéralement "instance-specific parametric absorption" = fast weights par instance qui compensent l'histoire manquante via self-match : très proche de notre pitch TTT, avec en plus la claim identité/subject-mismatch (notre angle re-ID).

### arXiv 2607.08766 : "OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators"
- Liu et al. Soumis 9 juillet 2026.
- "Cache-aware on-policy self-distillation" : student déroule son propre rollout ("writes its own generated chunks into the KV cache"), teacher = LE MÊME modèle évalué aux mêmes latents bruités mais avec un cache privilégié ("data-assisted cache" : l'historique ancien remplacé par les chunks de vidéo RÉELLE correspondants). Perte = matching des vitesses.
- Delta : le contexte privilégié vient de VIDÉOS RÉELLES (3 800 vidéos d'une minute, pas data-free) ; l'objectif est la dégradation long-horizon des générateurs few-step, PAS la compression du contexte évincé dans une mémoire (aucun état compressé). Mais il installe en vidéo le principe "teacher = soi-même à cache privilégié, student = soi-même à son propre cache".

### arXiv 2512.04519 : VideoSSM (déjà connu)
- Mémoire globale SSM qui absorbe les tokens évincés de la fenêtre locale ("introduces a global memory to absorb tokens evicted from the local window and relies on a state space model to recurrently compress them into a compact, fixed-size state"), MAIS entraînée par distillation Self-Forcing depuis un teacher BIDIRECTIONNEL, pas par self-distillation cache-réduit-vs-cache-complet.

### Écartés après vérification
- **Future Forcing (2605.30083)** : politique de cache training-free, merge des entrées évincées selon leur profil d'attention futur prédit. Aucun apprentissage, aucune distillation.
- **Forcing-KV (2605.09681, 10 mai 2026)** : compression de cache training-free par pruning tête par tête (têtes statiques vs dynamiques). Pas de distillation.
- **DecMem (2605.31336, 29 mai 2026)** : mémoire découplée = retrieval sparse par blocs (SGM) + fenêtre locale ancrée (ALM), entraînée en teacher forcing sur WorldMem. Pas de compression d'éviction, pas de self-distillation.
- **AHN (2510.07318)** : texte seulement (cf. section 2).
- **RELIC (2512.04040)** : Unreal, tokens compressés dans le cache (cf. section 3).

## 5. Verdict final : (b) PARTIELLEMENT PRIS, delta mince, risque de préemption élevé

Le créneau exact "student vidéo au cache affamé distille depuis soi-même au contexte privilégié, data-free" est couvert par morceaux qui, ensemble, ne laissent qu'un interstice de synthèse :

| Composant de notre idée | Déjà publié en vidéo par |
|---|---|
| Self-distillation student-état-borné vs teacher = même modèle full-context, sur vidéos auto-générées (data-free) | **ARL 2605.16579** (mai 2026) |
| Écriture d'un état compressé DÉCLENCHÉE par l'éviction du cache | **2604.10103** (avril 2026) ; VideoSSM 2512.04519 |
| "Du discard au distill" : absorber l'historique manquant dans des poids par instance en matchant local vs full du même modèle, avec claim préservation d'identité | **ISPA 2607.00712** (1er juil 2026) |
| Teacher = soi-même à cache privilégié sur les propres rollouts du student | **OPSD-V 2607.08766** (9 juil 2026) |
| Le schéma AHN complet (fenêtre + compression récurrente de l'out-of-window + self-distillation, backbone gelé) | AHN 2510.07318, **texte seulement** : PAS encore fait en vidéo dans un seul papier |

Ce qui reste réellement libre : la combinaison AHN-pour-la-vidéo-générative stricte, c'est-à-dire un module mémoire APPRIS offline (backbone gelé), alimenté en continu par chaque chunk évincé, entraîné UNIQUEMENT par la perte de self-distillation cache-affamé-vs-cache-complet sur vidéo auto-générée, évalué sur la ré-identification. Mais : (1) ARL a déjà la perte et le data-free, (2) ISPA a déjà le "distill l'évincé, self-match, identité préservée" (en training-free, ce qui rend un module appris plus lourd difficile à motiver s'il ne bat pas la forme close), (3) le rythme du champ (ISPA et OPSD-V datent de CE mois-ci) rend la fenêtre très courte. Un papier vendant cette idée comme contribution principale se ferait opposer ARL + ISPA + AHN en review ; en revanche, comme OBJECTIF AUXILIAIRE d'identité (notre angle re-ID/Anchor-Resurrection) au-dessus d'un de ces mécanismes, la self-distillation d'éviction n'est plus une nouveauté en soi mais un outil établi, citable.

### Correctif sur les deux agents
- Agent A : conclusion correcte (créneau pris), citations fautives (deux papiers fusionnés sous "ARL2" ; le vrai ARL = 2605.16579, et 2604.10103 est un second papier pertinent distinct).
- Agent B : faits corrects sur AHN et RELIC, conclusion fausse ("libre") car recherche incomplète : ARL, ISPA, OPSD-V et 2604.10103 manquaient.
