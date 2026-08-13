# Plan de recherche agréé (synthèse session), 2026-06-29

Workflow wf_bfe9dbf7-b12, 6 agents (5 angles + consensus).

## L'angle retenu (une phrase)

Une memoire apprise adressable par contenu a write NON-LINEAIRE (SwiGLU fast-weight) entrainee par self-rollout causal, evaluee sur le seul axe non sature : la re-identification de sujet a travers les revisites A->B->A en streaming interactif prompt-switch ; on construit et valide d'abord le benchmark re-ID, on rend le read non-toxique (decay+gate borne, prerequis), puis on escalade les leviers de stockage du moins cher au plus cher (sonde diagnostique -> recall supervise -> RL-write facon REFINE), de sorte que le RL soit un upside conditionne, jamais le pari fondateur.

## La nouveaute revendiquee + le venue vise

Trois nouveautes empilees, verifiees non revendiquees cette session :
1. Le premier benchmark de re-IDENTIFICATION de sujet (revisite A->B->A) pour une memoire recurrente apprise en video streaming : aucune memoire recurrente apprise n'a jamais ete mesuree sur la re-ID, l'unique axe non sature (MAG-Bench PSNR ~19-21, EntityMem face-acc 0.41, loin du plafond).
2. Le bake-off apparie compresse-appris vs lineaire (gated-delta) vs retrieval sur cet axe, avec l'edge non-lineaire comme seule arete defendable apres la deflation TTT=linear-attention (2602.21204) et la preemption du gated-delta generique par VideoSSM (2512.04519).
3. Le RL-write porte de REFINE (2602.16704, language-only) vers la VIDEO avec une reward d'identite (ArcFace/DINO returned-vs-first) : il dissout nos deux murs (la cross-chunk BPTT infaisable, et l'incapacite structurelle de DMD a superviser la re-ID).

Venue : workshop est le plancher realiste et garanti (benchmark valide + resultat do-no-harm + diagnostic, publiables meme si l'entrainement echoue). Main-conf est conditionne a battre le retrieval (structurellement avantage) sur la revisite SCENE-CHANGEE, tout en EGALANT (pas en battant) la SOTA training-free de dynamic_degree (bande 57-86). Renommer le module interne "MAG" pour eviter le clash avec MAG publie (2512.18741).

## Le plan par phases

### Phase 0 -- MESURER : construire ET valider le benchmark re-ID + mesurer toutes les baselines (inference seule, ~1 semaine, zero entrainement)
- Objectif : prouver qu'il existe un headroom re-ID que le retrieval ne ferme pas deja, et livrer le harness de mesure que toutes les phases suivantes scorent. C'est a la fois le prerequis du RL (on ne peut recompenser ce qu'on ne mesure pas) et une contribution publiable en soi.
- Experience : 30-50 triples A->B->A (visages + animaux/objets distinctifs) ou le sujet S apparait en A, sort de la fenetre d'attention locale ET du sink pendant N latents (eviction PROUVEE via les indices de cache dans causal_model, pas une simple decroissance), puis revient en A' ; moitie meme-scene, moitie scene-CHANGEE/pose differente (le regime ou le retrieval pixel ne peut pas relighter). Etendre reference/e2_eval_parallel.py (pool gpu1/2/7). Scorer re-ID : ArcFace-cos (visages) + DINOv2-cos (general, crop via GroundingDINO+SAM sur le nom commun), normalise a un plafond oracle de self-similarite intra-A ; PSNR/SSIM/LPIPS facon MAG-Bench sur la region revisitee. Tourner sur : AF-vanilla (anchor re-cache ON), AF sans re-cache, retrieval (DySink/Context-as-Memory), MAG compresse publie, et la memoire 3L toxique actuelle (ckpt 1200). Tout sur wandb.
- Baseline+metrique : la table re-ID elle-meme (ArcFace/DINOv2-cos returned-vs-first, PSNR/LPIPS) ; retrieval = reference superieure structurellement avantagee ; AF-vanilla = plancher. VALIDATION du metrique comme gate dur : (a) il doit separer "le sujet ne part jamais" (borne sup) de "sujet different substitue" (borne inf) par >3 sigma (bootstrap CI) ; (b) Spearman >0.6 vs un classement humain sur ~30 clips.
- Gate go/no-go : GO si le metrique valide (>3 sigma + Spearman >0.6) ET la meilleure baseline (attendu : anchor re-cache ou retrieval) reste <= 0.8x du plafond oracle sur le split scene-CHANGEE. NO-GO/pivot si le retrieval est deja >= 0.9x du plafond (re-ID resolu par retrieval) ou si rien ne se separe du bruit (axe sature/mal pose) : on bascule alors vers un papier benchmark+analyse pur ou un autre axe.

### Phase 1 -- PREREQUIS : rendre le read non-toxique (decay+gate borne, DMD intra-chunk, pas cher, pas de BPTT)
- Objectif : transformer le read uniformement toxique en do-no-harm a TOUS les horizons. Necessaire (Titans/ARL2/VideoSSM) mais explicitement PAS la contribution.
- Experience : plomber les primitives deja implementees (use_decay/decay_init/bounded_gate dans video_memory.py, gated-delta dans weight_norm_update) a travers causal_model.py (lignes ~498-500, actuellement droppees) + configs/ttt_memory.yaml. Reentrainer la memoire avec decay + tanh borne via le lever DMD self-rollout existant (backbone gele, groupe lr_mem), intra-chunk uniquement, 1L puis 3L [5,16,27], quelques centaines de pas. Logger live cos(w_k,w_0) (doit decroitre = oubli), norme par colonne, valeur de gate, dbg_out_rms/dbg_read_rms.
- Baseline+metrique : memoire 3L toxique actuelle (e2_write3L/ckpt 1200) et AF-vanilla. Metrique = ecart re-ID par frame (doit cesser d'etre uniformement negatif) + qualite VBench globale (dims hors la cible de victoire) a 5s/60s/120s, et etat qui oublie reellement (cos(w_k,w_0) decroit, gate borne).
- Gate go/no-go : GO si le decay+gate atteint le do-no-harm (ecart re-ID par frame ramene dans le bruit a 5s ET 60s) avec un etat qui oublie. NO-GO si le read reste toxique meme avec decay+gate borne : l'injection (mode B/D) est cassee structurellement ; arreter et rediagnostiquer le site d'injection, OU livrer le benchmark + un resultat negatif propre ("une fast-weight memory ne peut etre rendue non-toxique sous DMD 4-step streaming").

### Phase 2 -- DIAGNOSTIC : la re-ID est-elle READ-limitee ou STORAGE-limitee ? (sonde cheap qui decide la fourche couteuse)
- Objectif : savoir si l'identite du sujet SURVIT dans l'etat compresse a travers le gap. Cette experience cheap decide si le RL est meme necessaire, et c'est le crux contre faire du RL le pari fondateur.
- Experience : sur les clips Phase-0, geler la memoire Phase-1 et entrainer une petite sonde lineaire/MLP a decoder l'embedding ArcFace/DINO du sujet original depuis l'etat de la memoire a trois points (in-window, juste-evicte, apres tout le segment B). Comparer au decodage depuis le cache retrieval (borne sup lossless). Inclut la sonde "oracle-write" : ecrire a la main les features de A dans le fast-weight, lire en A', est-ce que le read seul reproduit l'identite ?
- Baseline+metrique : retrieval-accuracy / R^2 de l'embedding depuis l'etat appris vs depuis le cache retrieval (borne sup) vs chance (plancher), en fonction de la longueur du gap.
- Gate go/no-go : BRANCHE (pas kill). Decodable depuis l'etat post-gap -> READ-LIMITE -> Phase 3a (supervise cheap) suffit. Indecodable -> STORAGE-LIMITE -> Phase 3b (RL) justifie. Decodable depuis retrieval mais jamais depuis aucun etat appris -> la memoire compressee est structurellement dominee par le retrieval -> reporter "le retrieval bat la memoire apprise sur la re-ID" (publiable), ne poursuivre 3b que pour chasser la victoire etat-compresse. KILL global si indecodable depuis l'etat appris ET le cache retrieval -> l'info est genuinement perdue, livrer benchmark + finding "le streaming perd fondamentalement l'identite a la revisite".

### Phase 3a -- LEVIER SUPERVISE (cheap ; pris si READ-LIMITE)
- Objectif : entrainer le read+gate (et le write court-horizon) a faire remonter le sujet stocke a la revisite, avec une cible differentiable, en evitant BPTT cross-chunk ET RL.
- Experience : loss auxiliaire dans model/dmd.py::generator_loss (hooke le scaled_generator_loss.backward existant ; groupe LR memoire deja isole) : (1) associative-recall intra-fenetre (ecrire le sujet, le relire), (2) distillation de feature ou la cible = etat cache du sujet in-window ou le read retrieval, supervisant le read a la revisite. Optionnellement degeler a bas lr les couches backbone adjacentes a la memoire.
- Baseline+metrique : Phase-1 (DMD-only, decay, sans recall) et AF anchor re-cache, sur le re-ID held-out (ArcFace/DINO-cos, PSNR), guardrail dynamic_degree dans la bande 57-86.
- Gate go/no-go : SUCCES si la memoire apprise ferme >=50% du gap AF-vanilla -> retrieval sur le re-ID held-out en tenant le do-no-harm et la bande dynamic_degree. Sinon (plateau) escalader vers 3b.

### Phase 3b -- RL-WRITE facon REFINE (couteux, haute nouveaute ; pris si STORAGE-LIMITE ou 3a plateau) : LE pari main-conf
- Objectif : entrainer le write non-lineaire avec une reward sequence-level de re-ID, SANS backprop a travers le rollout / les inner updates (reward scalaire = policy gradient ; c'est exactement ce qui esquive l'OOM cross-chunk BPTT et fournit le signal de stockage que DMD ne peut pas donner).
- Experience : rendre le write stochastique (les lr par token lr0/lr1/lr2 de _qkv_lr deviennent une policy gaussienne softplus-mean + log-std appris ; le lr echantillonne EST l'action). GRPO dans trainer/distillation.py : G rollouts par triple A->B->A, avantage group-normalise (pas de value net), KL vers la reference (la policy Phase-1/3a gelee). Le gradient ne passe QUE par le log-prob de la policy lrs/decay, jamais par les inner updates ni la diffusion (_detach reste). Reward COMPOSITE = identite ArcFace/DINO (returned-vs-first) MOINS penalite de degenerescence (frame-difference / floor dynamic_degree / LPIPS-copy anti-freeze) ; encoder d'EVALUATION different de l'encoder de reward. Smoke d'abord : overfit 1-2 triples, G=4, 1 couche, horizon le plus court contenant une revisite (2-3 chunks <=21f). Puis pleine echelle sur le train split. Budget GPU fixe d'avance (N GPU-jours gpu1/2/7). Tout sur wandb live.
- Baseline+metrique : le write apparie NON-RL (3a / Phase-1, architecture identique) en tete-a-tete, ET AF anchor re-cache. Metrique = re-ID held-out (encoder != reward) avec bootstrap CI, ponderation split scene-CHANGEE ; guardrail = dynamic_degree dans 57-86, decomposition de reward (terme identite vs penalite) pour detecter le hacking.
- Gate go/no-go : Smoke GO si la reward monte sur l'overfit en ~200 pas avec gradients stables (non-NaN). Echelle GO/main-conf si le RL bat le write non-RL apparie sur le re-ID held-out par une marge CI-separee SANS sortir de la bande dynamic_degree, dans le budget. KILL si reward plate apres ~200 pas (credit assignment mort -> fallback teacher-forced identity loss style TTT-Video), OU gains uniquement avec violation des guardrails (reward hacking irreductible), OU gains qui disparaissent sur le split scene-CHANGEE (scene-copy, pas de vraie memoire).

### Phase 4 -- BAKE-OFF APPARIE + ECHELLE + ABLATIONS + DECISION DE VENUE
- Objectif : isoler l'edge non-lineaire (exigence du panel), montrer la tenue a l'echelle minute, attribuer la victoire, decider workshop vs main-conf.
- Experience : a entrainement/site d'injection/reward/compute fixes, swapper SEULEMENT le module : (a) SwiGLU non-lineaire (notre), (b) gated-delta lineaire/matriciel (VideoSSM-style, le cas preempte + deflation TTT), (c) retrieval (DySink/Context-as-Memory, relighte les pixels exacts, structurellement avantage), (d) AF anchor re-cache. Leaderboard sur le benchmark Phase-0 vs VideoSSM/ARL2/MAG publie. Echelle re-ID + dynamic_degree a 120s/240s (sweep e2_eval_parallel.py). Ablations : non-lineaire vs lineaire, RL vs DMD-only, decay on/off, termes de reward, gate_extra (agreement/frames_since_switch/denoising_step), placement des couches memoire. Interpretabilite : mem_out feature-maps + scalaire dbg_agreement autour de l'instant de revisite.
- Baseline+metrique : les quatre variantes + les publies sur le re-ID held-out (split same-scene ET scene-CHANGEE) + dynamic_degree ; comparaison-cle : non-lineaire (a) vs lineaire (b), et (a) vs retrieval (c,d) sur le split scene-CHANGEE ; delta re-ID par ablation avec CIs.
- Gate go/no-go : MAIN-CONF si (a) bat (b) sur le re-ID ET egale/bat le retrieval sur le split scene-CHANGEE tout en egalant dynamic_degree 57-86, ET une ablation attribue proprement le gain au write non-lineaire entraine (pas seulement au decay gate). WORKSHOP si (a) bat seulement (b)+anchor-recache mais perd contre le retrieval (lacher "bat le retrieval", garder "premier benchmark re-ID d'une memoire apprise + premier RL-write video"). KILL de la nouveaute non-lineaire si (a) n'egale pas (b) (deflation linear-attention realisee) ; KILL de la nouveaute write si tout le gain est explique par le decay gate (resultat Titans/ARL2 non-novel).

## Les baselines a battre + les metriques (jamais subject_consistency)

Baselines (sur le benchmark re-ID Phase-0, protocole apparie) :
- AF-vanilla (anchor re-cache ON) = plancher de reference.
- AF anchor re-cache = la baseline retrieval native du stack (re-cache d'anchor).
- Retrieval externe : DySink / Context-as-Memory (relighte les pixels exacts, structurellement avantage sur la re-ID).
- MAG publie (2512.18741, leader re-ID compresse), VideoSSM (2512.04519, gated-delta SOTA), ARL2.
- Notre lineaire gated-delta apparie (controle de deflation) et notre write non-RL apparie (tete-a-tete du RL).

Metriques primaires :
- re-ID cosine returned-vs-first : ArcFace (visages), DINOv2 (general), crop GroundingDINO+SAM, normalise au plafond oracle intra-A ; report SEPARE same-scene vs scene-CHANGEE (le scene-CHANGEE est le metrique-cle main-conf).
- Revisite PSNR/SSIM/LPIPS facon MAG-Bench sur la region du sujet revisite.
Metriques guardrail (a EGALER, pas a battre) :
- dynamic_degree : doit rester dans la bande SOTA training-free 57-86 (MATCH, jamais regresser).
- Qualite VBench globale (dims hors la cible de victoire) + CLIP-interactif (prompt-follow au switch) comme guardrails de non-degradation.
- Cout d'inference : etat O(1) (memoire apprise) vs cache O(t) (retrieval) comme axe de parite/avantage.

## Les risques + les criteres d'abandon (kill criteria)

Risques majeurs :
- Headroom : le retrieval (re-lecture pixel exacte) peut deja resoudre la re-ID same-scene ; le seul terrain defendable est le split scene-CHANGEE, qui peut etre trop etroit pour une main-conf.
- Construction du metrique : ArcFace ne couvre que les visages ; DINOv2 depend du crop (echecs SAM contaminent le score) ; la scene-CHANGEE confond derive d'identite et changement legitime d'apparence. Si variance > derive baseline, la Phase 0 ne gate rien.
- Credit-assignment RL : une reward scalaire sur un rollout diffusion de 60-240s avec G samples peut etre trop bruitee/faible (meme raison structurelle que l'echec de DMD) ; REFINE marchait sur de courtes fenetres LM.
- Reward hacking : la reward d'identite est triviallement gamee en figeant la scene / copiant le sujet ; le guardrail KL+dynamic_degree+LPIPS peut manger tout le gain.
- Backbone gele (lr=0) : l'info d'identite doit transiter par un block gele ; capacite possiblement insuffisante, plafonnant le gain re-ID quelle que soit la qualite du write.
- Compute/contention : seuls gpu1/2/7, chunk cape a 21f ; GRPO multiplie le cout du rollout par G ; historique de blocage GPU.
- Preemption mid-projet : un papier VideoSSM/REFINE-adjacent peut lander le RL-write video en premier, ou le benchmark re-ID en premier.

Kill criteria :
- Phase 0 : metrique non valide (<3 sigma ou Spearman <0.6) -> axe inmesurable, stop ; OU retrieval >= 0.9x oracle sur scene-CHANGEE -> pas de white space, pivot.
- Phase 1 : decay+gate borne n'atteint pas le do-no-harm -> injection cassee, rediagnostiquer ou livrer benchmark + resultat negatif.
- Phase 2 : embedding sujet indecodable depuis l'etat appris ET le cache retrieval apres le gap -> info perdue, aucun levier ne la recupere, livrer benchmark + finding negatif.
- Phase 3a : ne ferme pas >=50% du gap AF->retrieval en tenant do-no-harm -> arreter le supervise, decider 3b vs publier.
- Phase 3b : reward plate sur overfit apres ~200 pas (credit-assignment mort), OU gains seulement avec violation guardrail (hacking), OU gains nuls sur scene-CHANGEE, OU write-non-RL egale write-RL -> kill RL, fallback teacher-forced identity loss + benchmark workshop.
- Phase 4 : (a) n'egale pas (b) -> kill nouveaute non-lineaire ; gain entierement explique par decay gate -> kill nouveaute write ; perd contre retrieval ET lineaire sur tout split -> pas de claim main-conf.
- Transversal : toute tentative de cross-chunk BPTT comme "fix" -> kill immediat (prouve infaisable : chunk cape 21f / seq_len 32760, crash checkpoint-recompute, OOM sans checkpoint).

## Ce qu'on NE fait PAS

- Pas de cross-chunk BPTT (prouve infaisable) ; le RL existe precisement pour l'eviter.
- Pas de course sur les axes satures/encombres : VBench Total 5s, FPS, max-length, RoPE, flickering, motion, et subject/background-consistency comme cible de victoire ; on ne BAT pas dynamic_degree (on l'EGALE, bande 57-86).
- Pas de claim "gated-decay memory generique via DMD self-rollout sur Wan-1.3B" (preempte par VideoSSM) ni "linear-TTT novel" (deflate par TTT=linear-attention) ; le decay gate est cadre strictement comme prerequis, jamais comme contribution.
- Pas de RL comme pari fondateur : il est conditionne par le diagnostic Phase 2 (et la non-toxicite Phase 1) ; on depense le compute RL haute-variance seulement quand une sonde cheap prouve qu'il est le goulot.
- Pas d'entrainement avant que le metrique re-ID soit valide (gate dur Phase 0).
- Pas de backbone degele dans le chemin par defaut (casse la story cheap-first ; seulement un degel a bas lr optionnel des couches adjacentes en 3a si necessaire).
- Pas de runs sans wandb ; pas de mem_breakdown:true en entrainement reel (artefact de leak connu).

## La toute premiere action (cette semaine)

Executer la Phase 0, cout zero entrainement, en trois pas :
1. Renommer le module interne "MAG" pour eviter le clash avec MAG publie (2512.18741) avant toute trace ecrite.
2. Auteur le suite de triples A->B->A (commencer a 10, visages + objets distinctifs, moitie same-scene moitie scene-CHANGEE) via la machinerie multi-switch existante (streaming_switch_training _recache_after_switch / generate_video_with_switch), avec assertion d'eviction window+sink via les indices de cache dans causal_model.
3. Etendre reference/e2_eval_parallel.py en harness re-ID (ArcFace-cos + DINOv2-cos avec crop GroundingDINO+SAM + PSNR/SSIM/LPIPS revisite), puis passer d'abord le GATE DE VALIDATION DU METRIQUE (separation "ne part jamais" vs "sujet substitue" >3 sigma + Spearman >0.6 vs humain sur ~30 clips), et tourner read-only sur AF-vanilla (re-cache ON/OFF), retrieval, MAG publie et la memoire 3L ckpt 1200, sur gpu1/2/7, tout logge wandb. Ne toucher a aucun code d'entrainement tant que le metrique n'a pas valide et que le headroom (<= 0.8x oracle sur scene-CHANGEE) n'est pas confirme.

Fichiers concernes (chemins absolus) : /shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/e2_eval_parallel.py, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/video_memory.py, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/causal_model.py, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/configs/ttt_memory.yaml, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/trainer/distillation.py, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/model/dmd.py, /shared/ssd_14T/home/adenelmi/Anchor-Forcing/pipeline/streaming_switch_training.py.
