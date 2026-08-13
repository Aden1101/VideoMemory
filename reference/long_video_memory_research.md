# Vidéo longue + mémoire: recherche de fond (code + littérature), 2026-06-28

7 agents (6 lentilles + synthèse), run wf_0e95bae1-c06. (1 lentille a échoué sur le schema, 5+synthèse OK.)

## Les points les plus interessants (non-evidents)

- **Le champ a deja publie notre resultat negatif.** "A comparison of memory mechanisms in world models" (arXiv:2512.06983) benchmarke exactement notre classe de design : la memoire fast-weight (Titans) **sous-performe le baseline sans memoire** ("reconstruction collapse / training instability"), le cache explicite gagne, le state-space (Mamba) est un milieu de gamme. MAG n'est pas en territoire inconnu cote negatif ; on reproduit un echec connu de la memoire online.

- **Le tueur est l'horizon du signal, pas l'architecture.** La perte DMD est per-chunk sur <=21 frames latentes avec gradient uniquement sur les frames nouvellement generees (`model/streaming_training.py:615-618, 682-689`), notee par un teacher memory-less qui ne voit que ce chunk (`model/dmd.py:106-159`). La fenetre KV = `local_attn_size=12 + sink_size=3` (`configs/ttt_memory.yaml:77-78`). Donc **l'horizon supervise <= l'horizon d'attention** : il n'y a litteralement aucun gradient sur un span que l'attention locale ne couvre pas deja. Le seul differenciateur de la memoire (persistance cross-chunk) n'est jamais dans la perte.

- **Le DMD rend l'optimum de la memoire = "ne rien faire".** Le DMD aligne la marginale du student sur celle d'un teacher **sans memoire et court-horizon** ; toute contribution distinguable de la memoire s'eloigne de cette marginale et est penalisee. La perte **ne peut pas recompenser un rappel d'identite que le teacher lui-meme ne possede pas**. La degradation **pire au horizon d'entrainement ~6s, qui s'attenue aux longueurs** est la signature de ce mecanisme, pas un probleme de tuning.

- **Notre regle interne (negative-dot LaCT) est correcte, mais c'est la mauvaise regle pour les prompt-switch.** Verifie contre LaCT (arXiv:2505.23884, Eq.7) : `swiglu_fast_weight_gradient` (`video_memory.py:129-155`) ascende `<f_W(k),v>`, identique a LaCT. Mais elle **accumule sans terme d'erreur** : elle ne peut jamais "desapprendre" une identite perimee. Pour le cas A->B->A, la **delta rule** error-correcting (DeltaNet arXiv:2406.06484, Gated DeltaNet-2 arXiv:2605.22791, KDA arXiv:2510.26692) est superieure parce qu'elle minimise `||Sk-v||^2`.

- **La metrique est aveugle au seul job que la memoire a en propre.** VBench subject_consistency = `0.5*(cos(d1,dt)+cos(d_{t-1},dt))` sur features DINO globales (arXiv:2311.17982). La **moitie du score est le terme frame-adjacent ~0.99** ; le 0.95@240s est domine par le lissage local, pas l'identite. Pire, DINO global mesure le fond (verrouille par le sink), pas le sujet ; pour des humains il faut ArcFace/ReID sur le crop du sujet.

- **La memoire est blessee la ou elle ne peut prouvablement pas aider.** Dans les world models, WorldMem (arXiv:2504.12369) ne gagne ~rien dans la fenetre (24.11->25.98 PSNR) et beaucoup hors fenetre (17.32->23.98). Notre gap DINO est **uniforme, y compris dans la fenetre 3s**. Une memoire qui blesse la ou le KV a deja la reponse n'est pas sous-entrainee, elle est **mal cablee** : le read tire la ou il ne sert a rien.

- **La porte non bornee + double-norm = nuisance a RMS fixe.** Le read = `gate*(a_norm(x)*(1+scale)+shift)` (`video_memory.py:355`) avec `gate` = sortie Linear brute, **sans tanh/sigmoid**, biais init 0.1 (`:271`) ayant derive a 1.26 ; ajoute directement au residu (`causal_model.py:601`). Le double-norm (`o_norm` `:265` puis `a_norm` `:244`) emet toujours un vecteur unit-RMS : **la memoire ne peut pas se taire quand elle n'a rien d'utile**. L'ablation de WorldMem (retrieval aleatoire : PSNR 23.98->18.32) est l'analogue exact de ce read decouple de la pertinence.

- **Toutes les memoires qui marchent en video sont keyees par geometrie, qu'on n'a pas.** WorldMem (pose+FOV), spatial-memory (arXiv:2506.05284, point clouds), Context-as-Memory (arXiv:2506.03141, FOV-overlap), VMem (ICCV 2025). Notre pipeline ne porte que `switch_frame_index/switch_prompt` (`pipeline/streaming_switch_training.py`). **Sans cle spatiale, il n'y a rien vers quoi "retrieve"** ; la memoire ne peut injecter qu'une moyenne non conditionnee, exactement le add a RMS fixe mesure.

## Une memoire a-t-elle un vrai job ici ? (verdict honnete)

**Non, pas dans la configuration actuelle, et pas dans le regime mesure.** Convergence des 5 rapports (confiance 0.78-0.82) :

1. **Le job est deja pris.** LongLive (arXiv:2509.22622) atteint 240s sur le **meme** Wan-1.3B avec **9 fenetre + 3 sink + KV-recache, zero memoire apprise** ; Self-Forcing++ (arXiv:2510.02283) atteint 4m15s avec **DMD + rolling KV** et notre recette exacte moins la memoire. Anchor-Forcing a deja 12+3. Rolling Forcing (arXiv:2509.25161) prouve que le sink seul porte le long-terme (ablation : drift 0.01->4.63).

2. **Notre signal ne peut pas l'apprendre.** DMD vs teacher memory-less court-horizon = zero pression gradient pour le rappel long-range (horizon supervise <= fenetre KV) ; backbone gele (`lr:0.0`, `configs/ttt_memory.yaml:42`, `trainer/distillation.py:367-374`) ne co-adapte jamais au read.

3. **Notre metrique ne peut pas le voir.** subject_consistency saturee ~0.95, dominee par l'adjacence ; et l'axe vivant aux ces longueurs est le **Dynamic Degree**, pas l'identite (MoC arXiv:2508.21058 : +0.4pt identite mais +0.10 dynamique ; Relax Forcing arXiv:2603.21366 : identite **-0.35pt**, +8.1 dynamique). Une memoire qui injecte un vecteur a RMS fixe par frame va surtout **nuire au Dynamic Degree**, cohrent avec le gap uniforme observe.

**La premisse n'est pas "fast-weight impossible".** TTT-Video (arXiv:2504.05298, +34 Elo vs sliding-window/Mamba2/GatedDeltaNet) prouve que ca marche, mais sous **les conditions exactes qu'on viole** : perte de reconstruction sur vraies videos (identite dans la cible), backbone co-entraine, porte bornee `tanh(alpha)` alpha init 0.1, horizons croissants 3s->63s. Notre combo (memoire + DMD + backbone gele + porte non bornee + transplant Scal3R inference-only) est **genuinement nouveau et non supporte par aucun blueprint**.

## Ou et quand le KV-cache+sink echoue vraiment (regime cible)

Deux regimes ou la litterature montre un echec **prouve** du window+sink, et **notre eval n'en cree aucun** (rolling forward + prompt-switch frames 21/39, sans revisit) :

- **Revisit / loop-closure (world models).** Camera/sujet part et revient apres que le contenu a quitte la fenetre et n'est pas le sink : "severely forgets" (spatial-memory arXiv:2506.05284, WorldMem, VMem, Memory Forcing arXiv:2510.03198). Mais la memoire y est **geometrie-grounded, retrieval par pose 3D**, pas un fast-weight DMD. WorldMem random-retrieval (23.98->18.32) montre que la pertinence doit etre imposee en amont.

- **Re-identification apres eviction (interactif T2V).** Un sujet hors fenetre > `local_attn_size`+sink puis qui revient, ou un switch A->B->A ou le sujet original doit **reapparaitre**. Self-Forcing++ nomme explicitement ce gap que rolling-KV+sink ne ferme pas : "occluded objects change after being blocked". C'est le **seul job structurellement unique** d'une memoire associative : le sink (3 frames fixes) est un ancrage non content-addressable ; meme la History-KV selectionnee (Relax Forcing) est bornee par son budget.

Point cle : "la memoire degrade tout" **ne refute pas** un job ici ; ca refute un job dans le regime monotone qui a ete mesure. Le job (s'il existe) est dans le regime interactif/revisit, jamais teste.

## Comment l'entrainer si on y va (signal de stockage, porte de TTT-Video/LaCT)

Condition necessaire transversale a toute la litterature : **la perte EXTERNE doit decroitre quand la bonne chose est rappelee.** Next-token (Titans/Mamba2) et denoising-reconstruction (TTT-Video/LaCT) la satisfont ; le DMD-vs-teacher-memory-less **ne la satisfait pas**. Donc la reconstruction n'est pas strictement requise (Mamba2 arXiv:2405.21060 n'a aucune cible interne, appris par BPTT ; la memoire RL apprise par reward, Fast-and-Forgetful arXiv:2310.04128), mais il faut un **signal externe dependant du rappel**. Si on y va, porter le recipe TTT-Video/LaCT integralement :

- **(Signal de stockage)** Remplacer/augmenter le DMD par une cible long-horizon avec **identite dans la cible** : soit reconstruction/denoising teacher-forced sur un chunk futur tenu hors fenetre (>12 frames latentes), ce qui **contourne le BPTT cross-chunk juge infaisable** (un seul forward window suffit) ; soit alignement de features type **REPA** (arXiv:2410.06940) du read au bloc 5 (dans la bande optimale early-mid de REPA) vers DINOv2 sur frames decodees sur long contexte clair. Le DMD est self-referentiel (`model/dmd.py:146-153` ne reference jamais frame 0) : rallonger l'horizon DMD seul ne suffirait pas, il faut un **ancrage absolu**.

- **(Porte)** Remplacer la porte non bornee par **`tanh(alpha)*read + x`, alpha init ~0.1** (TTT-Video). Le `gate_extra`/agreement (`video_memory.py:277-310`) est le bon instinct mais il est **ajoute** a une porte deja non bornee au lieu de la remplacer ; le cabler comme la porte bornee pilotee par l'agreement memoire<->prompt.

- **(Backbone)** **Degeler** la self-attention (LoRA suffit) pour qu'elle apprenne a consommer le read. Alternative frozen-backbone qui marche : **DiT-Mem** (arXiv:2511.19229) concatene des tokens memoire dans la self-attention (pertinence geree par l'attention), au lieu d'un add residuel.

- **(Regle d'ecriture)** Secondaire (downstream du fix de cible) : passer du negative-dot a la **delta rule** error-correcting pour permettre l'overwrite aux switches ; ajouter momentum + forget gate appris (Titans arXiv:2501.00663). Correction a notre diagnostic : le `weight_norm_update` renorm-to-init (`video_memory.py:185-187) **n'est PAS le tueur principal** ; LaCT et TTT-Video L2-normalisent aussi (l'info est dans la direction des lignes). Les vrais tueurs sont la cible court-horizon + backbone gele + porte non bornee.

## Les alternatives moins cheres (et quand elles gagnent)

Par ROI croissant en risque :

1. **Mesurer d'abord (cout ~0).** Re-scorer les videos existantes avec `cos(d1,dt)` seul (sans le terme adjacent), ArcFace/ReID sur crop, et une sonde occlusion/re-entry. Knob deja cable : `MEM_ZERO_COLD_FRAMES` (`causal_model.py:597-600`) isole read-harm vs write-harm sans reentrainement.

2. **Matcher train/inference horizon (LongLive, arXiv:2509.22622).** Ablation LongLive : le sink ne devient effectif qu'apres streaming-long-tuning (train-long-test-long). Notre DMD est per-chunk ~6s : **fixer ce mismatch est plus cheap et prouve qu'ajouter un module**.

3. **DySink (arXiv:2605.21028), training-free.** Remplacer le sink statique par top-k keyframes retrieved (encodeur gele 0.02B), drop-in dans le path KV existant. Attaque directement le cas "mauvaise frame d'ancrage" qu'un fast-weight vise.

4. **Retrieval sur frame-bank (Context-as-Memory arXiv:2506.03141 ; MoC arXiv:2508.21058 ; FramePack arXiv:2504.12626).** MoC = router top-k parameter-free sur le KV retenu, co-entraine ; masque j>=i pour **interdire les loop-closures** (sinon "stalled motion / repeated frames"). Context-as-Memory bat le sliding-window sur revisit (PSNR 20.22 vs 15.70).

**Quand le fast-weight gagne quand meme :** un seul niche, **video interactive infinie temps-reel** ou un etat de taille O(1) constant bat un frame-bank croissant et ou la latence interdit le retrieval. Et seulement avec le recipe TTT-Video complet. Sinon le retrieval egale ou bat le fast-weight a une fraction du risque (verdict independant de 2512.06983 : retrieval > fast-weight).

## L'experience decisive a lancer en premier

**Deux etapes, cheap-first, qui gate tout le reste.**

**ETAPE 0 (mesure, ~zero training) :** construire une sonde **revisit/re-identification** que la metrique actuelle ne voit pas. Schedule A->B->A (ou objet qui sort du cadre et revient apres > `local_attn_size`+sink frames, donc evince de la fenetre 12 sauf les 3 sinks). Scorer la **recuperation d'identite du sujet original au retour vs sa premiere apparition** (DINO/ArcFace cosine), PAS le cos frame-adjacent. Lancer **le baseline KV+sink seul** :
- Si le baseline recupere deja l'identite au revisit (KV sink + re-conditionnement texte suffisent) : **la memoire n'a aucun job a ces longueurs, on stoppe.**
- S'il echoue : le niche existe, on continue.

**ETAPE 1 (si gap reel) :** A/B matche sur la meme sonde re-id :
- (i) baseline KV+sink ;
- (ii) **fix cheap d'abord** : sink plus gros / History-KV selectionnee (Relax/MoC-style) ou DySink, a compute egal ;
- (iii) MAG actuel sous DMD ;
- (iv) MAG reentraine recipe-litterature : cible reconstruction/identite sur chunk futur hors-fenetre teacher-forced, porte `tanh`, backbone degele (LoRA), delta rule.

**Sorties falsifiables :** si (ii) egale (iv) -> le KV/retrieval gagne, **droppe la memoire**. Si (iv) bat clairement (ii) sur re-id alors que (iii) non -> le job est reel ET l'echec etait le signal d'entrainement, pas l'architecture (seul resultat qui justifie de continuer). Prediction basee sur les preuves : **le retrieval matche ou bat le fast-weight a risque bien moindre**.

## Recommandation

**Ne pas reinvestir dans la config MAG actuelle (DMD + backbone gele + porte non bornee + transplant Scal3R).** Elle est non-saine par construction : l'horizon supervise <= la fenetre KV, donc aucun gradient n'existe sur le long-range que la memoire est censee fixer, et l'eval le montre correctement echouer.

Ordre d'action :
1. **Reparer la mesure d'abord** (ETAPE 0 : sonde revisit/re-id avec ArcFace+`cos(d1,dt)`). C'est le pre-requis qui rend tout interpretable. Sans gap mesurable, **on stoppe la memoire** -- c'est l'issue la plus probable vu que LongLive/Self-Forcing++ atteignent 240s/4min sans memoire sur le meme backbone.
2. **Si un gap revisit existe : essayer le cheap avant l'apprise** -- horizon train/inference matche (LongLive), puis DySink/retrieval (training-free, reutilise le path KV).
3. **Le fast-weight appris en dernier recours uniquement** s'il bat a la fois bigger-KV et retrieval sur la sonde re-id, et seulement reentraine avec le recipe TTT-Video complet. Son unique justification structurelle est le cout O(1) pour video interactive infinie, pas la qualite.

Fichiers de reference cles : `wan/modules/video_memory.py:129-155,167-187,344-355`, `wan/modules/causal_model.py:597-601`, `model/streaming_training.py:615-618,682-689`, `model/dmd.py:106-159`, `configs/ttt_memory.yaml:42,77-78`, `trainer/distillation.py:367-374`.
