# Review adversariale du plan MLP fast-weight (panel 0/5), 2026-06-28

Workflow wf_9b9c0dfe-011, 6 agents (5 experts indep + chair).

## Verdict du panel (combien valident, et a quelles conditions)

0/5 valident le plan tel qu'ecrit. Aucun expert ne donne de feu vert. Le niveau d'accord avec le plan est faible et homogene (agreement_level 0.25 a 0.35, confiance des reviewers 0.72 a 0.82).

La position majoritaire n'est pas "abandonner", c'est "repivoter avant de depenser du compute" : garder le diagnostic et le benchmark, sortir le levier vedette (windowed-DMD + truncated-BPTT) du chemin critique, et faire de la supervision d'identite et/ou de l'ablation substrat-controlee la contribution principale. Trois reviewers (3, 4, 5) iraient jusqu'a retrograder le MLP au rang d'ablation et faire de la matrice gated-delta le pari principal. Deux reviewers (1, 2) accepteraient une version drastiquement reduite, mesure-d'abord, eventuellement avec une porte d'oubli ajoutee au MLP.

Conditions minimales citees par plusieurs panelistes pour un eventuel feu vert partiel : (a) construire le probe re-ID A->B->A et tester la baseline KV+sink AVANT tout reentrainement ; (b) ajouter une porte d'oubli/decay a la recurrence quel que soit le substrat ; (c) faire l'ablation MLP-vs-matrice a une variable a la fois, within-chunk, sans BPTT cross-chunk.

## Ce sur quoi les experts sont d'accord (le plan tient)

Accord unanime ou quasi-unanime sur quatre points :

- **Le diagnostic du code actuel est juste et bien ancre.** Backbone gele (`ttt_memory.yaml:42` lr 0.0), write historiquement sous no_grad / read-only, detach d'etat par chunk (`streaming_training.py:294-298`), horizon supervise <= fenetre KV (local_attn 12 + sink 3, `config:77-79`), erreur de gradient bf16 du Newton-Schulz Muon (`video_memory.py:103`), gate non bornee. Reels et correctement identifies.

- **Les corrections d'hygiene (2, 3, 5, 6) sont saines et necessaires quel que soit le cadrage** : Muon off + expose via model_kwargs (aligne LaCT-video), gate tanh bornee g_init=0 avec init propre, unfreeze de l'hote (tous LaCT/TTT-Video/VideoSSM co-entrainent le backbone), signal d'identite explicite (text-dropout a la MAG, puisque le teacher DMD est sans memoire).

- **Le design d'evaluation est la meilleure partie du plan.** Rejeter VBench subject_consistency (DINO inter-frames adjacentes, aveugle a l'adjacence, recompense le statique) au profit du Dynamic Degree + une metrique re-ID A->B->A explicite est bien motive, discriminant, et reutilisable quel que soit le substrat gagnant. Independamment valide par la these d'EntityBench ("echec a rendre" vs "echec a memoriser").

- **La plomberie est correcte et substrat-agnostique** : TTTBlockCache comme leaf pytree FSDP-safe, threading du ttt_cache, fix C3 du detach cross-chunk. Cet investissement se porte sur une memoire matricielle a peu de frais.

- **L'honnetete intellectuelle du plan est saluee** : il enonce explicitement le risque central (truncated-BPTT sur un MLP non-gate est non prouve, il faut une recurrence preservante que le MLP n'a pas). Le reproche n'est pas le manque de candeur, c'est que le plan sous-pondere a quel point ce risque est central plutot que lateral.

## Les desaccords et objections les plus fortes (avec dissidence explicite)

Quatre objections sont partagees par la majorite, et reviennent comme potentiellement fatales :

1. **Le levier vedette ne peut pas entrainer ce qu'il cible (chemin de gradient coupe).** Argument repete par les reviewers 1, 2, 4 et 5. Le sujet a re-identifier est ECRIT dans le chunk A, qui vit dans le warmup no_grad. L'etat entrant dans la fenetre BPTT est une constante detachee. Le gradient n'atteint donc que les writes situes DANS la fenetre W ; tout write plus vieux que W (tout le warmup, ou l'identite long-range a ete stockee) recoit ZERO gradient. Pour mettre le write du chunk A dans le chemin, il faut un BPTT plein de la fin jusqu'a A, c'est-a-dire exactement l'OOM que le plan pretend contourner. Conclusion : windowed-DMD entraine un bon read et un write court-horizon (<= W), mais est structurellement incapable d'entrainer le stockage d'identite long-range.

2. **La revendication VRAM contredit vos propres mesures.** Reviewers 1, 2, 3, 5. "Checkpoint seulement l'inner write" n'economise que la branche SwiGLU d'une seule couche (~1-2GB/chunk). L'OOM vient du maintien en vie de W graphes forward DiT 1.3B complets pour un seul backward, que Muon-off et ckpt_write ne touchent PAS. Votre note `crosschunk-bptt-infeasible` mesure : no-checkpoint a l'echelle minimale (1 couche, K=2) OOM des le step 0 (46.6GiB) ; le full-block checkpointing crashe sur le mismatch checkpoint-recompute vs rolling-KV-cache. Donc W>=2 OOM ou re-crashe sans le refactor cache-snapshot ; a W=1 il n'y a aucun gradient cross-chunk, le levier se reduit a "entrainer le read sur un chunk lointain".

3. **Le substrat n'a pas la primitive de retention requise.** Reviewers 1, 2, 3, 4, 5. `weight_norm_update` (`video_memory.py:167-188`) renormalise chaque ligne a sa norme initiale a chaque write : le fast-weight ne fait que tourner dans une coquille a norme fixe, sans decay, sans slot protege. Ecrire B fait necessairement tourner le poids hors du contenu de A -> rappel A->B->A structurellement fragile. Or le truncated-BPTT n'assigne du credit long-range qu'a travers une recurrence preservante (resultat classique Williams & Peng ; raison d'etre des portes LSTM/GRU et du decay SSM). Le plan concede que le MLP "manque d'une porte d'oubli" puis avance quand meme : la machinerie est construite et le gradient de stockage n'a nulle part ou atterrir.

4. **La seule novelty revendiquee est mince, confondue, et partiellement scoopee.** Reviewers 1, 3, 4, 5. L'evidence MLP>matrice (TTT-Video, +3 Elo a 18s, dans le bruit) est en teacher-forced full-BPTT avec backbone co-entraine, sans metrique d'identite ; son transfert au regime DMD self-rollout + BPTT tronque est non prouve, le plan l'appelle "le pari". Sous le regime DMD-AR reel, l'evidence publiee va dans l'AUTRE sens : VideoSSM (meme Wan2.1-T2V-1.3B, Self-Forcing+DMD) montre que la matrice gated-delta bat LongLive ET Self-Forcing. La re-ID est de la recuperation associative cle->valeur, terrain canonique des memoires lineaires/delta gates ; "la non-linearite aide la re-ID" est asserte, pas argumente, et possiblement a l'envers. De plus les quatre ingredients (substrat LaCT, training = Long Self-Rollout + Windowed Teacher Correction de VideoSSM, signal d'identite = text-dropout p=0.6 de MAG, benchmark re-ID = MAG-Bench/EntityBench) sont deja publies.

**Dissidence explicite a signaler :**

- **Sur le diagnostic d'horizon lui-meme, le panel est divise.** Le reviewer 1 affirme que le diagnostic est DEJA FALSIFIE par vos propres donnees : l'eval e2_write3L montre la memoire la PIRE a l'horizon d'entrainement (6s, subject_consistency -0.163) et MOINS mauvaise quand la video s'allonge (60s -0.084, 120s -0.070). Un probleme horizon-limite serait bon en-distribution et se degraderait avec la longueur ; la signature observee est l'inverse, soit de la TOXICITE du read per-frame (mode B/D), pas un credit long-range manquant. Les reviewers 2, 3, 4, 5 acceptent au contraire le mismatch d'horizon comme reel et bien diagnostique. C'est un desaccord de fond, non reconcilie : si le reviewer 1 a raison, tout le levier BPTT vise un probleme inexistant et il faut d'abord trancher empiriquement (freeze-state, MEM_ZERO_COLD_FRAMES, probe de toxicite du read) avant quoi que ce soit.

- **Sur "publiable de toute facon".** Majorite (3, 4, 5) : un negatif n'est publiable que s'il est propre, or il y a au moins quatre confounds simultanes (pas de porte vs la matrice, recette truncated-BPTT non prouvee, Muon off donc plus le substrat "publie", backbone gele->LoRA) ; un reviewer lit "le MLP a perdu" comme "vous l'avez mal entraine". Minorite (reviewer 1) : pre-enregistrer le negatif comme livrable PRIMAIRE est plus defendable et plus basse-variance que parier sur MLP>matrice, a condition de le rendre rigoureux.

- **Sur le sort du MLP.** Majorite (3, 4, 5) : retrograder le MLP en ablation, matrice gated-delta en pari principal. Minorite (1, 2) : garder le MLP est defendable mais SEULEMENT en lui ajoutant une porte de decay (S_t = diag(a_t) ⊙ S_{t-1} + update), ce qui preserve le APPLY non-lineaire (la novelty d'expressivite) tout en rendant la retention architecturale.

## VideoSSM : schema de gradient et ecart avec notre version cheap

Constat verifie par le panel (reviewer 3 le documente, 1, 2, 4, 5 le confirment) : **VideoSSM ne fait PAS de BPTT plein sur 60s.** Son schema d'entrainement est :

- un **Long Self-Rollout** jusqu'a ~60s sur un cache roulant KV + memoire ;
- puis le DMD applique uniquement sur une **fenetre courte echantillonnee uniformement, ~5s** ("Windowed Teacher Correction").

Le point critique : sa retention long-range vient du **gated delta-rule avec porte de decay explicite** (gate cumulatif negatif, exp(g_t) ⊙ M_{t-1}), c'est-a-dire de l'ARCHITECTURE, pas du gradient qui remonterait loin. Le gradient tronque ~5s suffit a la matrice PRECISEMENT parce que la retention ne depend pas du gradient. VideoSSM prouve donc l'INVERSE de l'hypothese du plan : un gradient fenetre/tronque est suffisant pour une memoire matricielle parce que la retention est une propriete de la dynamique.

**Ce que cela implique pour notre version cheap :** notre MLP fast-weight n'a aucune porte (renorm-to-init, `video_memory.py:185-187`). Sous un gradient tronque il n'a donc NI route gradient (tronquee a ~1 chunk) NI route architecturale (pas de porte). Pire, votre `chunk=21 latent ~= 5s` fait que le W=1 truncated-BPTT existant donne deja a peu pres l'horizon de gradient de VideoSSM (~1 chunk). Pousser W de 1 a 2-3 "pour echantillonner loin" n'achete qu'un gain marginal d'horizon et c'est exactement ce qui OOM. **Le vrai ecart avec VideoSSM est architectural (la porte de decay), pas la longueur de la fenetre de gradient.** Le plan depense donc son ingenierie la plus dure sur l'axe qui compte le moins.

Corollaire (reviewer 4, ancre dans `dmd.py:213-226`) : le DMD est un objectif distributionnel/marginal (KL teacher-sans-memoire vs critic, conditionne sur le prompt courant). Il recompense "ce chunk est un echantillon plausible pour son prompt", jamais "ce sujet = le sujet du chunk A". Etendre le BPTT ne propage que ce signal distributionnel plus loin. Les gains de VideoSSM sont du drift/Dynamic-Degree (distributionnel) et il ne teste PAS la re-ID d'instance. Le seul vrai levier d'identite est l'anchor feature-distillation du fix (4), aujourd'hui sous-specifie, qui devrait etre le titre et non un item lateral.

## Les ameliorations concretes a faire au plan

Convergence forte des cinq reviewers sur l'ordre et le contenu :

1. **Inverser l'ordre : mesure avant machinerie.** Construire le probe re-ID/revisit A->B->A et scorer la RECUPERATION d'identite (ArcFace/DINO sur le crop sujet, cos(frame1, frame_t), PAS VBench subject_consistency) sur la baseline KV+sink seule, plus une variante bigger-sink et une variante retrieval cheap. Si la baseline recupere deja l'identite, la memoire n'a pas de travail -> STOP/pivot. Test decisif et quasi-gratuit qui conditionne tout le reste.

2. **Faire d'abord le Step 0 de falsification gratuite** : gate=0 doit reproduire la baseline AF bit-pour-bit (attrape les fuites init/etat comme le bug `mag-init-weights-clobbers-gate`) ; freeze-state-never-write isole read-transform vs etat porte ; MEM_ZERO_COLD_FRAMES isole le cold-start. Ceci tranche aussi la dissidence horizon-vs-toxicite.

3. **Sortir windowed-DMD + truncated-BPTT du chemin critique.** Faire la science du substrat en within-chunk (W=1, ~5s, qui matche deja l'horizon de gradient VideoSSM). Ne revenir au BPTT cross-chunk que si le probe montre specifiquement un drift graduel horizon-limite.

4. **Ajouter une porte d'oubli/decay a la recurrence, quel que soit le substrat.** Prerequis, pas option. Sans recurrence preservante-et-oublieuse, ni le stockage long-range ni le credit truncated-BPTT ne sont mecaniquement possibles.

5. **Lancer l'ablation appariee MLP-vs-matrice tot et cheap, une variable a la fois.** Appliquer des corrections de regime IDENTIQUES a un MLP fast-weight et a une matrice gated-delta closed-form (VideoSSM Eq.8), meme pipeline/gate/cadence/identity-loss, une couche. La grille propre est le 2x2 (MLP/matrice) x (gate/sans-gate). Si vous changez substrat et regime ensemble (comme le "premier run a lancer"), un null est ininterpretable.

6. **Ajouter une baseline retrieval/KV identity-anchor** (Context-as-Memory : stocker quelques latents keyframe du premier prompt, re-injecter a la demande). Pour l'identite, stocker sans perte peut battre la compression dans un fast-weight oublieux, avec bien moins d'entrainement ; c'est la baseline qui pourrait rendre toute la ligne "memoire apprise" inutile, donc elle doit etre mesuree.

7. **Hygiene confirmee** : Newton-Schulz en fp32 avant tout BPTT-through-write (sinon l'erreur 2-4%/step se compose) ; supprimer la double-normalisation (`o_norm` + `a_norm`) qui decouple magnitude du read et pertinence ; verifier que frames_since_switch est borne a l'inference (flag 0..960 dans l'eval e2) ; instrumenter la gate (dbg_read/out_rms) pour MONTRER si le read pilote vraiment l'identite a distance.

8. **Comparer au vrai SOTA et sur les benchmarks publics** : VideoSSM (pas seulement LongLive/Self-Forcing), et reporter sur EntityBench / MAG-Bench, sinon une metrique re-ID privee sera lue comme cherry-pickee.

## Le plan alternatif le plus credible propose (et quand il bat le notre)

Les cinq alternatives convergent vers un meme plan, le plus complet etant celui du reviewer 5 (supervision-led, matrice-primary, benchmark-spined) :

- **Phase 0 (2 sem, risque recherche quasi-nul) :** livrer le benchmark A->B->A + la metrique drift-vs-distance. Le projet ne peut etre evalue sans, et c'est une contribution workshop autonome.
- **Phase 1 (la colonne vertebrale) :** un bake-off de memoires apparie sous DMD distillation identique et horizon d'entrainement cheap identique, sur {no-memory ; matrice gated-delta/SSM (PARI PRINCIPAL) ; le LaCT-MLP existant (ablation) ; une banque retrieval/KV identity-anchor}, avec pour TITRE un objectif de supervision d'identite long-range (coherence re-ID self-supervisee entre frames distantes du meme sujet + teacher-feature distillation vers un anchor lointain) applique a tous les substrats.
- **Variante minoritaire (reviewers 1, 2) si on tient au MLP :** garder le APPLY non-lineaire mais ajouter une porte de decay par canal (init a~0.99), puis 2x2 gate x substrat ; le titre honnete devient "gated >> ungated, et MLP-gated vs matrice-gated est le test d'expressivite, retention tenue fixe".

**Quand cet alternatif bat le notre :** quasi-toujours, pour trois raisons. (1) Contribution garantie : benchmark + comparaison equitable + recette de supervision sont publiables gagne-ou-perd, au lieu d'un pari binaire sur "MLP>matrice" que deux papiers publies enserrent deja. (2) Risque bien moindre : la matrice est prouvee sous exactement ce reglage DMD, a la primitive de decay que la tache re-ID exige, et s'entraine avec un gradient chunkwise-parallel exact au lieu du truncated-BPTT non prouve, bf16-lossy, Muon-off que le plan doit inventer. (3) Effort comparable ou moindre : meme infra de distillation, ~80% de la plomberie TTTBlockCache se porte, et on arrete d'empiler cinq fixes non prouves sur un substrat oublieux. Le meilleur resultat honnete du plan MLP (un negatif rigoureux) est deja un sous-ensemble de la Phase 1 de cet alternatif, obtenu avec moins de risque et un upside positif plus fort.

Le seul cas ou notre plan resterait defendable : si le probe Phase 0 montre que la baseline KV+sink echoue ET que le drift est specifiquement horizon-limite (degrade avec la longueur), ce que l'eval e2 actuelle ne soutient PAS. Tant que ce n'est pas demontre, le levier BPTT vise une cible non confirmee.

## Recommandation finale honnete

Ne pas lancer le plan tel quel. Le diagnostic, la plomberie, les fixes d'hygiene et surtout l'eval re-ID sont solides et reutilisables ; mais le contributeur unique de novelty ("MLP expressif bat une matrice closed-form sous distillation") est mince, possiblement a l'envers, partiellement scoope (VideoSSM, MAG, EntityBench), et son levier vedette (no_grad-warmup + truncated-BPTT loin) est mecaniquement incapable d'entrainer le stockage long-range (chemin de gradient coupe) et contredit vos propres mesures VRAM. Le constat VideoSSM est decisif : la retention long-range y est ARCHITECTURALE (porte de decay), pas obtenue par un gradient qui remonte loin ; notre MLP sans porte n'a ni route gradient ni route architecturale.

Action concrete recommandee, par ordre :
1. Step 0 de falsification + probe re-ID sur la baseline KV+sink (avant tout reentrainement). Ceci tranche aussi la dissidence horizon-vs-toxicite, qui est le desaccord de fond non resolu du panel.
2. Si la memoire a un travail : ajouter une porte de decay (prerequis), puis l'ablation appariee within-chunk MLP-gated vs matrice-gated vs retrieval, une variable a la fois, contre VideoSSM, sur EntityBench/MAG-Bench.
3. Faire de la supervision d'identite (DMD ne peut PAS superviser la re-ID d'instance, prouvable depuis `dmd.py:213-226`) le titre, pas un item lateral.
4. Ne revenir au cross-chunk BPTT que si, et seulement si, le drift est empiriquement montre horizon-limite.

Position honnete sur la publication : preparer le negatif rigoureux comme livrable principal probable, mais seulement s'il est propre (substrat appariE gate-vs-gate, une variable a la fois) ; "publiable de toute facon" est faux tant que les confounds (pas de porte, Muon off, backbone change, BPTT non prouve) coexistent. En l'etat, parier le papier sur MLP>matrice via truncated-BPTT sur un MLP non-gate est precisement le PIRE cas pour le MLP.

Fichiers de reference cites par le panel pour ancrer la suite : `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/video_memory.py` (renorm-to-init lignes 167-188, erreur bf16 ligne 103, double-norm 244/265), `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/causal_model.py` (read injection ligne 601, MEM_ZERO_COLD_FRAMES 597-600), `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/model/dmd.py` (objectif distributionnel 213-227), `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/trainer/distillation.py` (fenetre BPTT 1117-1137), `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/configs/ttt_memory.yaml` (lr 0.0 ligne 42, KV window 77-79).
