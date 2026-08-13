# Fresh Eyes B : vérification du brief, carte littérature, propositions

*Rapport d'un relecteur sceptique, 2026-07-22. Lecture seule du code (aucun script lancé, aucun GPU touché). Transparence : le contexte qui m'a été fourni inclut les notes de projet (MEMORY.md) ; je ne peux donc pas prétendre à une redécouverte "en aveugle" des pistes internes (Anchor Resurrection y figure). Ce que j'apporte : la vérification contre le code, la carte littérature à jour, et une évaluation critique indépendante de ce qui a une chance de marcher.*

---

## §1. Vérification du brief contre le code

### 1.1 Confirmé (avec les références exactes)

| Claim du brief | Verdict | Preuve (fichier:ligne) |
|---|---|---|
| §2 Cellule MLP SwiGLU fast-weight, état de taille fixe | CONFIRMÉ | `wan/modules/video_memory.py:134-167` (swiglu_apply + gradient interne) ; état = (w0,w1,w2), nh=1, head_dim=1536, inter 4x → ~28M par couche (`:296-299`) |
| §2 Lit avec les tokens courants, apply-first | CONFIRMÉ | `video_memory.py:464-468` (`step()` : q du chunk courant, `swiglu_apply` AVANT le write) |
| §2 Injection via porte apprise gated-AdaLN | CONFIRMÉ | `video_memory.py:399-411` (proj → shift/scale/gate ; `bounded_gate`/`bounded_mod` tanh :404-407 ; `rel_norm` :391-393 ; `dc_remove` :428-429) + cap dur `mem_stream_scale` 0.5×RMS local `wan/modules/causal_model.py:659-670` |
| §2 Écriture Muon (Newton-Schulz) | CONFIRMÉ, une précision (voir 1.2.4) | `video_memory.py:92-131` (NS quintique), `:170-210` (update) |
| §2 3 blocs [5,16,27] | CONFIRMÉ pour les runs c2/c3/l1 | `reference/runs/l1_longlive/l1_longlive.yaml:145-148` (NB : `configs/ttt_memory.yaml:79` = [5] seul, config historique) |
| §2 Eviction-feed = écrire le chunk évincé, pas le courant | CONFIRMÉ | FIFO par bloc `causal_model.py:628-641` ; pop au niveau modèle `:1350-1385` ; `window_chunks = ceil(12/3) = 4` chunks de délai `:1361` ; les chunks du sink ne sont JAMAIS écrits `:1373-1375` ; la query de lecture reste le chunk courant `video_memory.py:459-463`. C'est un FIFO par chunk, PAS une éviction token-exacte (approximation assumée en commentaire `:1346-1349`) |
| §2 Co-entraînement LoRA + mémoire par la loss DMD, data-free | CONFIRMÉ | groupes d'optimiseur séparés (lr / lr_mem) `trainer/distillation.py:413-427` ; rollout depuis le bruit sans dataset (backward simulation) `model/dmd.py:229-263` ; write entraîné (`train_write`) `pipeline/streaming_training.py:68-77` |
| §2 Le teacher DMD ne voit ni le KV ni le passé lointain | CONFIRMÉ | `model/dmd.py:85-118` : real_score/fake_score reçoivent uniquement les latents bruités de la fenêtre + les embeddings texte ; fenêtre jugée = `slice_last_frames` = 27 frames latentes (~6.75 s) `dmd.py:254`, `l1_longlive.yaml:167` |
| §3.e Pack de stabilisation | CONFIRMÉ | EMA `distillation.py:390-406` ; clip mem-grads (FSDP les ignorait) `:428-432` ; cosine lr + floors `:433-438` ; cache-dropout annealé `streaming_training.py:79-98` ; caps dans `l1_longlive.yaml:141-164` |
| §3.f l1 (LongLive) net-négatif identité, mémoire active | CONFIRMÉ par les fichiers d'éval | `reference/runs/l1_longlive/eval2000_sink1/metrics_c1_step250.txt` : id_cos(f0,fL) memon 0.569 / memzero 0.690 / stock 0.710 ; DD memon 0.589 > memzero 0.567 > stock 0.489 (active mais coûte l'identité) |
| §3.g Le probe re-ID A→B→A existe, même seed/bruit entre arms | CONFIRMÉ | `pipeline/switch_causal_inference.py:109-141,196-214` (2e switch, prompt 3 = prompt A par défaut) ; ttt_cache créé par séquence `:182` (le vieux bug de bypass est corrigé) ; arms `run_reid_probe.py:144-153` |
| §3.g Visuel pair0 : la voiture de B traîne dans le retour de A | CONFIRMÉ | `reference/runs/reid_probe/results/reid_compare_pair0.png` : mem-ON = chien devant une structure de voiture rouge, mem-OFF = chien propre sur herbe |

### 1.2 Corrections et sur-simplifications

**1.2.1 — MAJEUR : le résultat du probe re-ID n'est PAS "elle nuit légèrement". Il est MIXTE, n=2, et la paire ignorée est un gros POSITIF.**
Le brief (§3.g) cite "delta ~ −0.06/−0.07". C'est uniquement **pair0** (chien/voiture) : full −0.071, crop60 −0.056 (`results/pair0_scores.json`). Mais **pair1** (femme en imperméable jaune / pingouins) donne **+0.151 en full et +0.311 en crop60** en faveur de la mémoire (`results/pair1_scores.json`, résumé dans `run_pair1.log`) : sans mémoire, A ressuscité ne ressemble presque plus à l'original (0.393/0.331), avec mémoire il revient (0.544/0.642), et self_origA est même meilleur (0.913 vs 0.839). Caveat sceptique dans les deux sens : (a) le +0.31 de pair1 peut refléter un effondrement de l'arm memzero plutôt qu'un "sauvetage" par la mémoire (reid memzero ~0.33-0.39 = le sujet n'est simplement pas revenu ; à vérifier à l'œil sur `videos/pair1_mem*.mp4`) ; (b) symétriquement, le −0.07 de pair0 est UNE vidéo, un seed. **Conclusion honnête : le probe est sous-puissant (n=2, 1 seed, signes opposés) et ne permet NI la conclusion du brief NI son contraire.** pair2 (chat/tabby, défini dans `run_reid_probe.py:51-52`) n'a jamais été lancé. L'action à ~0$ la plus rentable du projet : lancer 8-10 paires × 2 seeds avant toute décision.

**1.2.2 — Le probe n'a PAS testé la mémoire LongLive.** `probe.yaml:40` : checkpoint = `c3_stable/keep/checkpoint_model_001000` = la mémoire **AF-based "do-no-harm"** (celle du §3.e), pas la mémoire l1 re-basée LongLive (§3.f). L'enchaînement du brief (f puis g) laisse croire que le probe teste le système LongLive : non. De plus le probe tourne avec **sink_size 1** (`probe.yaml:15`) alors que l'entraînement a vu sink_size 3 : c'est un régime sink-affaibli, jamais vu à l'entraînement, ce qui change l'interprétation ("le scénario où la mémoire devrait servir" est aussi un scénario OOD pour elle).

**1.2.3 — "renorm vers l'init" est imprécis.** Le chemin par défaut renorme chaque ligne vers sa norme **d'avant l'update** (pas l'init) : `video_memory.py:191-199`. Avec `use_decay: true` (le cas réel de c3/l1, `l1_longlive.yaml:153`) c'est `w ← σ(decay)·w + muon(grad)` avec **cap** (pas renorm) à la norme d'init : `:201-210`. Nuance qui compte : le mode réel PERMET l'oubli (shrink), contrairement à ce que "renorm vers l'init" suggère.

**1.2.4 — "Muon multi-pas" prête à confusion.** `muon_update_steps: 5` = 5 itérations de Newton-Schulz pour **orthogonaliser un seul** gradient interne (`video_memory.py:110-131`). Il y a **un seul pas d'écriture par chunk**, pas cinq. La "multi-step TTT" des notes internes désigne l'orthogonalisation, pas des pas de gradient multiples.

**1.2.5 — Un deuxième chemin de fuite de B non mentionné : le recache au switch lit la mémoire.** `switch_causal_inference.py:82-94` : `_recache_after_switch` ré-encode les 12 dernières frames (du B) sous le prompt A avec `ttt_update=False` mais `ttt_cache` passé → **l'injection mémoire est active pendant le recache** et contamine le KV recaché. Le "la mémoire traîne B dans le retour de A" passe donc par 2 canaux : la lecture pendant la génération ET le KV recaché modifié. Une ablation à 0$ (zéroter l'injection uniquement pendant le recache) départagerait les deux.

**1.2.6 — "sans mémoire" (memzero) = injection zérotée, écritures maintenues.** `causal_model.py:653-658` + doc de `run_reid_probe.py:8-9`. C'est la bonne ablation d'injection, mais la formulation "sans mémoire" du brief est légèrement abusive (l'état continue d'être écrit ; sans injection il ne peut toutefois rien influencer, donc l'ablation est valide).

**1.2.7 — "Data-free" = sans vidéo, pas sans données.** L'entraînement consomme 248k prompts texte réels (VidProM, `l1_longlive.yaml:51`). Aucune conséquence sur le diagnostic, mais à formuler proprement dans un papier.

**1.2.8 — Divers.** (a) La FIFO d'éviction retarde de 4 chunks : au switch-back (frame 162), la FIFO contient encore ~4 chunks de B qui seront écrits PENDANT les premiers chunks du retour de A — la mémoire écrit activement du B pendant la résurrection de A, ce qui aggrave mécaniquement la récence. (b) `configs/ttt_memory.yaml` (pointé par le brief §7) est une config historique (memory_layers [5], lr=0) qui ne décrit PAS les runs discutés ; la config qui fait foi est `l1_longlive.yaml`. (c) Les fichiers d'éval s'appellent `metrics_c1_step250.*` dans tous les dossiers eval* (nom hérité, contenu correct) — piège d'archéologie.

---

## §2. Carte littérature (2024-2026)

### 2.1 Ce qui est fait / préempté

| Travail | Ce que c'est | Ce que ça préempte pour nous |
|---|---|---|
| **VideoSSM** (arXiv 2512.04519) | Même substrat (Wan-1.3B, DMD self-rollout), mémoire SSM/gated-delta globale co-entraînée e2e, fenêtre locale + mémoire d'état ; SOTA consistance/motion à l'échelle minute | Le mécanisme "mémoire apprise co-entraînée sur AR-DMD windowed" : fait, et ça marche (avec reconstruction sur vraie vidéo). Nouveauté mécanisme ≈ 0 |
| **MemGen / MAG** (2512.18741) | Modèle mémoire séparé qui compresse l'historique en KV compact + générateur ; MAG-Bench pour la rétention d'historique | La mémoire-compression d'historique + le benchmark de rétention/revisit |
| **Memento** (2606.14667) | Génération next-shot AR + **reconstruction de sujet depuis la mémoire** (dual-query : identité vs cohérence courte), pipeline de données cinématiques "subject-aware" | LE gros préempt du créneau "mémoire d'identité de sujet" — mais avec supervision réelle dédiée (pipeline d'annotation). Le data-free reste ouvert |
| **TTT-Video** (2504.05298), **LaCT** (2505.23884) | TTT/fast-weight pour la vidéo (fine-tuning avec vraie vidéo ; large-chunk TTT) | Le substrat fast-weight en vidéo. Pas le data-free, pas le re-ID |
| **AHN** (2510.07318, ByteDance) | LLMs : fenêtre glissante + module récurrent (Mamba2/GatedDeltaNet) qui compresse le **KV évincé** ; entraîné par **self-distillation** depuis le modèle full-attention gelé | (a) L'eviction-feed exact de c2 (notre code le cite déjà) ; (b) SURTOUT : la recette d'objectif qui nous manque — le teacher à contexte privilégié — démontrée côté LLM |
| **RELIC** (2512.04040) | World model interactif 14B, mémoire = KV historique compressé avec poses caméra, **full-context distillation** vers un student causal | Le principe "teacher long contexte → student mémoire" en vidéo — mais avec dataset Unreal + poses caméra, revisit spatial, pas de sujet, pas data-free |
| **WorldMem** (2504.12369), **Context-as-Memory** (2506.03141, SIGGRAPH Asia 25), **CaR** (2606.23105), **Retrieve What's Missing** (2606.02479) | Revisit de SCÈNE par retrieval de contexte/frames (souvent avec poses caméra) | Le revisit spatial par retrieval : très couvert. Confirme que si le point de revisit est connu, le retrieval gagne trivialement (voir §2.3) |
| **IAMFlow** (2605.18733) | Mémoire d'identité **training-free** pour vidéo narrative (registre d'entités LLM/VLM) + NarraStream-Bench | Le "identity memory" côté système/training-free + un benchmark narratif multi-prompts qui ressemble à notre éval revisit |
| **TetherCache** (2606.13035) | Sur **Self-Forcing**, training-free : recall gated de frames longues + alignement statistique ; réduit drastiquement le drift à 240 s | La barre "améliorer la baseline en long" est déjà montée par des méthodes SANS entraînement — un mem-TTT entraîné doit battre ça pour justifier son existence |
| **DySink** (2605.21028), **AdaState** (2605.30349) | Sinks dynamiques (contre le sink collapse) ; ancre latente auto-évolutive co-débruitée jamais rendue | Les alternatives "pas-de-mémoire-apprise" qui attaquent le même symptôme ; AdaState est conceptuellement la plus proche d'une mémoire sans mémoire |
| **Echo-Memory** (2606.09803) | Étude contrôlée des mécanismes de mémoire dans les world models (raw context vs compression vs SSM), protocole replay/revisit | (a) Le constat "raw context est une baseline très forte, la compression sacrifie l'évidence nécessaire" = notre "no job" mesuré indépendamment ; (b) préempte partiellement le papier "metric-integrity" de repli (les métriques de replay et de revisit divergent, déjà documenté) |
| **IPRO** (2510.14255, CVPR26), **Identity-GRPO** (2510.14256), **VCD** (2510.19193) | RL/reward fine-tuning avec ArcFace/consistance vs une image de référence (I2V) | Le "reward identité vs référence" — mais toujours avec une référence EXTERNE fournie (image conditionnelle). Le self-référentiel (référence = propre sortie du modèle) n'y est pas |
| **Bases** : Self-Forcing (2506.08009) → LongLive (2509.22622) → Anchor-Forcing (2603.13405) | Lignée du substrat | — |

### 2.2 Où il reste un trou

Le créneau encore vide au croisement de trois axes : **(i) mémoire fast-weight/TTT, (ii) identité d'INSTANCE de sujet (pas de scène/caméra), (iii) zéro vidéo réelle (le signal vient du modèle lui-même)**. Chaque paire d'axes est prise (VideoSSM = i sans ii/iii ; Memento = ii sans iii ; AHN/RELIC = i+"iii-ish" hors vidéo-gen ou avec données synthétiques), le triplet ne l'est pas. Deuxième trou plus petit : personne n'a publié de **fenêtre jugée non contiguë** dans un cadre DMD (donner au juge court l'ancre ET le présent).

### 2.3 Sceptique : trou = mur ou opportunité ?

Il faut découper le trou en deux :

- **Sous-trou A ("windowed-DMD-only", l'objectif actuel) : c'est un MUR, pas une opportunité.** Argument (formalisable, voir §4) : l'objectif est invariant au remplacement de l'instance du sujet dès que l'information sort de {fenêtre, sink, prompt} → gradient d'identité structurellement nul. Vos 8 runs le confirment empiriquement, et Echo-Memory le confirme par un tiers (la compression perd face au raw context sur le revisit). Personne n'y arrive parce que c'est impossible SOUS CET OBJECTIF, pas parce que personne n'a essayé.
- **Sous-trou B ("data-free avec contexte privilégié") : opportunité réelle, non essayée en vidéo-gen.** "Data-free" n'implique pas "windowed-DMD-only" : les propres rollouts du modèle contiennent l'instance exacte du sujet (le segment A original), et un teacher = le même modèle avec un contexte privilégié (KV splicé/retenu) fournit une cible SPÉCIFIQUE À L'INSTANCE sans une seule frame de vraie vidéo. AHN l'a démontré côté LLM (module mémoire seul entraîné, base gelée, self-distillation) ; personne ne l'a fait pour la vidéo AR distillée. Deux indices que ce n'est pas un mur : (a) la cible existe et est dense (pas un scalaire de reward) ; (b) pair1 du probe montre qu'une injection même NON ciblée peut déplacer le re-ID de +0.31 — la bande passante de l'injection est suffisante pour changer l'issue d'une résurrection.
- Lucidité sur le plafond : dans le setting interactif où le point de revisit est CONNU (switch explicite), un recache-depuis-l'historique training-free (les frames de A sont dans le buffer de sortie !) résoudrait probablement le revisit sans aucune mémoire apprise — c'est la famille Context-as-Memory/TetherCache. La mémoire fast-weight ne se justifie que par : état O(1) en streaming, revisit NON signalé, et pas de banque de frames qui grossit. Le papier doit assumer ce positionnement (compression à état constant), sinon un reviewer sort le retrieval en 2 lignes.

---

## §3. Mes propositions, classées (faisabilité × nouveauté × chance de marcher)

### Idée 1 — RSD : Resurrection Self-Distillation (teacher à KV splicé, data-free) — RANG 1

**Mécanisme.** Curriculum d'entraînement A→B→A (l'infra switch existe, `switch_causal_inference.py` ; à porter côté training, le trainer ne fait qu'un switch aujourd'hui). Sur les chunks de résurrection, on fabrique un **teacher-oracle = le même modèle (gelé, mem-OFF) dont le KV est splicé avec la fenêtre du segment A original** (contenu du A d'origine re-caché aux positions récentes — exactement le mécanisme `_recache_after_switch` existant, mais alimenté par les frames de A stockées dans le buffer de sortie au lieu de la queue de B ; attention shape in-distribution, contrairement à un full-KV jamais vu à l'entraînement). Le **student = fenêtre normale + mémoire TTT**. Même bruit, mêmes seeds sur les deux branches.

**Objectif.** `L_res = || x0_student − sg(x0_oracle) ||²` sur les chunks post-switch-back (+ la DMD normale ailleurs, pour le réalisme). Le gradient traverse la lecture ET l'écriture (train_write existe déjà). C'est un signal **dense, spécifique à l'instance, et data-free** : la cible est la propre génération du modèle quand il se souvient de A.

**Pourquoi ça donne un job à la mémoire (ce que le data-free actuel ne donne pas).** L'inerte n'est plus l'optimum : si la mémoire n'injecte rien, le student ne peut pas reproduire l'oracle (l'info de A n'est ni dans la fenêtre ni — si on entraîne avec sink réduit/cache-dropout, déjà implémenté — dans le sink). L'écart student-oracle EST l'information évincée : exactement le contenu que l'eviction-feed écrit déjà. Mécanisme et objectif s'emboîtent enfin. Et ça pénalise directement la pathologie observée (traîner B) : l'oracle, lui, ne traîne pas B.

**Coût.** Zéro donnée. +1 forward teacher no_grad par chunk entraîné (~+30-50 % du coût générateur sur ces chunks) ; 4 GPUs OK. Ingénierie : étendre le trainer au double switch + le splice (le recache existant fait 80 % du travail).

**Nouveauté.** AHN = self-distillation de contexte pour LLM ; RELIC = full-context distillation avec données Unreal + poses ; Memento = reconstruction de sujet avec vraies données. **Personne n'a fait "le modèle est son propre témoin d'identité" en vidéo AR data-free.** (Honnêteté : c'est la formalisation de l'"Anchor Resurrection" déjà dans vos notes — convergence, pas découverte ; je la juge correcte et je la précise : teacher par recache-splice in-distribution, loss x0 dense, curriculum double-switch.)

**Mesure propre.** Pré-enregistrée : 10 paires A/B × 2 seeds × {memon, memzero, stock-LongLive}, reid_resA_origA (DINO crop) + **ctrl_resA_B comme métrique anti-récence** (elle existe déjà dans le probe et mesure exactement "B traîne") + DD/lag-2 pour exclure gel/flicker + côte-à-côte humain. Gate avant tout entraînement : étendre le probe actuel à 10 paires (~0$) pour établir la vraie baseline mixte.

**Risques réels.** (a) Qualité de l'oracle : le recache-splice au retour de A peut lui-même être médiocre (discontinuité de mouvement) — mitigation : ne distiller que les 2-4 premiers chunks de résurrection, là où l'oracle est le plus fiable ; (b) capacité : 3 points d'injection pour transporter une apparence — pair1 suggère que ça suffit pour basculer une résurrection, pas garanti pour la finesse ; (c) le student peut apprendre à "recopier l'oracle" via des raccourcis non-mémoire (le LoRA) — contrôle : arm LoRA-seul (mémoire off) dans l'ablation.

### Idée 2 — Anchor-DMD : fenêtre jugée non contiguë — RANG 2 (novel, risqué, quasi gratuit)

**Mécanisme.** Aujourd'hui le juge DMD voit les 27 dernières frames latentes. Proposition : sur une fraction des steps, juger une fenêtre composite **[chunk ancre (le sujet au début) ⊕ 24 dernières frames]**, positions RoPE avec un vrai gap temporel (assumer le "jump cut"), prompts identiques. Le teacher bidirectionnel évalue alors la plausibilité d'une vidéo qui CONTIENT l'ancre et le présent : un sujet qui a dérivé devient une incohérence visible par le juge court. C'est l'attaque frontale du slogan "on ne peut pas récompenser une mémoire longue avec un juge court" : on ne rallonge pas le juge, on plie la fenêtre.

**Pourquoi un job pour la mémoire.** Le gradient d'identité existe enfin dans la loss principale (pas un terme auxiliaire DINO gameable) et il est aligné distributionnellement (même juge, même échelle que la DMD). **Poids faible** (0.1-0.3 de la DMD) et seulement quand le prompt n'a pas changé.

**Coût.** Zéro donnée, quasi zéro compute additionnel (même forward de juge, fenêtre différente). Implémentation : construction de la fenêtre composite + RoPE offsets, quelques jours.

**Nouveauté.** Fenêtre jugée non contiguë dans un cadre DMD : rien trouvé dans la littérature. Publiable même en résultat mitigé (analyse "que voit un juge court d'une paire ancre-présent ?").

**Risques (élevés, à dire franchement).** (a) OOD pour le teacher : Wan a surtout vu des clips mono-plan ; un jump-cut peut être jugé implausible EN SOI → le modèle pourrait minimiser en régressant le présent vers l'ancre = **sink collapse Goodhart** (DySink documente exactement cette pathologie). Tripwires motion déjà en place ; à poids faible c'est testable et tuable vite. (b) Le gradient passe par le juge fake_score qui doit lui aussi apprendre les fenêtres composites (l'entraîner dessus symétriquement). Je le classe 2 pour la nouveauté/coût, pas pour la probabilité de succès (~30 %).

### Idée 3 — Le framing "compression" : distiller l'oracle-retrieval en poids rapides, avec l'histoire O(1) — RANG 3 (même machinerie que 1, papier différent)

**Mécanisme.** Identique à l'idée 1, mais le teacher est explicitement un **retrieval-augmenté** (banque de frames qui grossit, sélection type Context-as-Memory/TetherCache) et la contribution revendiquée est : *une mémoire fast-weight à état constant peut-elle égaler une banque qui grossit, sur le revisit non signalé ?* Courbe état-vs-fidélité (nombre de sujets mémorisés avant saturation des ~28M×3), comparaison au retrieval à budget mémoire égal. C'est la réponse au reviewer "pourquoi pas du retrieval ?" et le vrai test de la question capacité. À faire comme SECOND papier/section si l'idée 1 donne un signal ; seul, il risque la conclusion négative d'Echo-Memory (le raw context gagne).

### Idée 4 — (mention, déconseillée seule) Reward self-référentiel DINO/ArcFace sur crops
Reward = similarité des crops tardifs au sujet du premier chunk. Déjà à moitié essayé chez vous (id_loss existant, e-series : l'injection l'écrase, et c'est le terrain de la triade Goodhart gel/flicker) ; préempté-adjacent par IPRO/Identity-GRPO (référence externe) ; un scalaire épars contre une loss dense, en co-training ça re-perd. Utilisable uniquement en RÉGULARISEUR au-dessus de l'idée 1, jamais en signal principal.

### Ordre d'exécution recommandé (coût croissant)
1. **0$ :** probe re-ID à 10 paires × 2 seeds sur le ckpt c3 EMA@1000 ET l1 (les deux mémoires) ; + l'ablation recache (1.2.5). Tant que ça n'est pas fait, personne ne sait si la mémoire actuelle aide ou nuit au revisit — les 2 paires existantes se contredisent.
2. **~1 j :** oracle-splice en inférence seule : générer A→B→A avec recache-depuis-A au switch-back → borne SUP de ce que la mémoire pourrait apprendre (si l'oracle lui-même est mauvais, l'idée 1 tombe, autant le savoir avant de coder le training).
3. **Ensuite :** idée 1 (RSD), avec l'idée 2 en run parallèle à poids faible si les GPUs le permettent.

---

## §4. Le cas "mur data-free" (à inclure dans le papier quoi qu'il arrive)

**L'argument formel, version courte.** Soit T une transformation qui remplace l'instance du sujet par une autre instance plausible sous le même prompt, appliquée au-delà de l'horizon {fenêtre locale, sink, prompt}. La loss windowed-DMD est (approximativement) invariante sous T : le juge évalue une distribution sur 27 frames conditionnée au texte, et T préserve cette distribution. Donc ∂L/∂(identité longue-portée) ≈ 0, et le contenu de la mémoire n'affecte la loss que par son effet sur la plausibilité LOCALE — d'où les trois attracteurs observés : inerte (AF, le sink fait le travail), nuisible (LongLive, l'activité non guidée coûte de l'identité), dégénéré (gel/flicker, quand des garde-fous créent des gradients exploitables). Huit runs + la confirmation tierce d'Echo-Memory (compression < raw context sur revisit). **Sous cet objectif, aucune idée ne peut marcher — ce n'est pas une question de stabilisation, de gate, ou de règle d'écriture.**

**Pourquoi ce n'est pas la fin du data-free.** L'invariance est une propriété de l'OBJECTIF, pas du régime data-free : les rollouts du modèle contiennent l'instance exacte (le A original), et tout teacher à contexte privilégié (splice/retrieval sur ses propres frames) brise T sans vidéo externe. C'est le contenu de l'idée 1. Si l'idée 1 échoue à son tour ALORS que l'oracle-splice (étape 2 ci-dessus) était bon, la cause résiduelle est la capacité/bande passante du canal mémoire (3 injections gated, cap 0.5×RMS) — et là le verdict honnête devient : *sur un substrat distillé gelé/LoRA, une mémoire fast-weight à état constant ne peut pas battre le retrieval sur l'identité longue-portée* — un négatif audité publiable, avec l'argument d'invariance en théorème central et le probe re-ID en protocole. Attention : la partie "les métriques divergent" du papier de repli est déjà en partie couverte par Echo-Memory (2606.09803) — le repli a perdu de la fraîcheur, raison de plus pour jouer l'idée 1 vite.

---

## Sources littérature
[VideoSSM](https://arxiv.org/abs/2512.04519) · [MemGen/MAG](https://arxiv.org/abs/2512.18741) · [Memento](https://arxiv.org/abs/2606.14667) · [AHN](https://arxiv.org/abs/2510.07318) ([GitHub](https://github.com/ByteDance-Seed/AHN)) · [RELIC](https://arxiv.org/pdf/2512.04040) · [Context-as-Memory](https://arxiv.org/pdf/2506.03141) · [CaR](https://arxiv.org/pdf/2606.23105) · [Retrieve What's Missing](https://arxiv.org/pdf/2606.02479) · [Echo-Memory](https://arxiv.org/pdf/2606.09803) · [IAMFlow](https://arxiv.org/pdf/2605.18733) · [TetherCache](https://arxiv.org/pdf/2606.13035) · [DySink](https://arxiv.org/pdf/2605.21028) · [AdaState](https://arxiv.org/pdf/2605.30349) · [IPRO](https://arxiv.org/html/2510.14255v3) · [Identity-GRPO](https://arxiv.org/html/2510.14256v3) · [VCD](https://arxiv.org/pdf/2510.19193) · [TTT-Video](https://arxiv.org/pdf/2504.05298) · [LaCT](https://arxiv.org/abs/2505.23884) · [FSM/Elastic TTT](https://arxiv.org/abs/2604.07350) · [FFN-TTT (understanding)](https://arxiv.org/abs/2606.26515) · [Head Forcing](https://arxiv.org/pdf/2605.14487) · [Self-Forcing](https://arxiv.org/abs/2506.08009) · [LongLive](https://arxiv.org/abs/2509.22622)
