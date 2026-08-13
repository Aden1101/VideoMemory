# Faire marcher un MLP/TTT fast-weight pour la vidéo AR (DMD/Self-Forcing): plan, 2026-06-28

Workflow wf_675680b7-c08 (6 lentilles + synthèse ; 1 lentille novelty a échoué sur le format).

## Verdict viabilite de l'angle (honnete)

Viable mais a haut risque, et la novelty est etroite. Le constat partage par les 5 investigations est sans ambiguite: notre substrat MLP/SwiGLU n'est PAS le bug. `wan/modules/video_memory.py:129-155` (swiglu_fast_weight_gradient, montee du dot `<f_W(k),v>`) est un portage ligne-a-ligne de LaCT (`ar_lact_swa_repeat.py`, loss `L=-f_W(k)^T v`, arXiv:2505.23884), sur le meme Wan2.1-T2V-1.3B, avec les memes chunks 3 frames (`configs/ttt_memory.yaml:60` vs LaCT `ar_window_size:3`). LaCT prouve que ce MLP s'entraine. Donc "objectif Hebbien vs reconstruction" et "taille de chunk" sont des fausses pistes pour la parite.

La cellule reellement vide et novel: un fast-weight MLP a gradient interne, EXPRESSIF, entraine A TRAVERS une auto-rollout DMD/Self-Forcing. LaCT et TTT-Video (2504.05298) sont teacher-forced sur latents reels (flow/v-pred, pas de DMD, pas de chaine de rollout). VideoSSM (2512.04519), le seul en DMD+self-rollout, abandonne volontairement le MLP pour une matrice gated-delta close-form (`M_t=exp(g_t)M_{t-1}+K V_new^T`) precisement pour rendre la BPTT longue tractable. Notre note `[[crosschunk-bptt-infeasible]]` documente exactement cette intractabilite.

Verdict: poursuivre, mais conditionner tout l'effort a une ablation appariee MLP-vs-matrice sous DMD. La novelty ("on entraine un write par rollout") est preemptee par VideoSSM; la SEULE contribution defendable est "un memoire non-lineaire expressif bat une matrice close-form SOUS DISTILLATION, sur une tache ou la non-linearite compte (re-identification A->B->A)". Si le MLP n'y bat pas la matrice a budget egal, l'angle s'effondre vers VideoSSM et il faut publier le resultat negatif/egal ou pivoter.

## Pourquoi leur MLP fast-weight marche et pas le notre

Le delta n'est pas le MLP ni l'objectif interne (on a deja celui de LaCT). C'est le REGIME autour, par ordre d'importance:

1. Le signal ne demande jamais a la memoire de stocker. DMD est applique LOCALEMENT a un chunk de 21 frames (`model/dmd.py:161-227`), le teacher real_score est le Wan bidirectionnel sans memoire qui note "ce chunk matche-t-il le prompt" et non "matche-t-il le sujet etabli il y a 50s" (`model/dmd.py:106-120`). L'etat TTT est detache a chaque frontiere de chunk (`model/streaming_training.py:294-298`, `bptt_window=1` par defaut `trainer/distillation.py:1117`). L'horizon supervise (<=21 frames) est du meme ordre que la fenetre KV locale (`local_attn_size=12`, sink 3). Donc aucun gradient ne recompense la retention au-dela de ce que l'attention voit deja: l'optimum est gate->0 ou bruit residuel. VideoSSM fait l'inverse: windowed-DMD echantillonne une fenetre 5s LOIN le long d'une rollout 60s, gradients propages a travers les pas dans la memoire.

2. Backbone gele, contrairement aux 3 recettes. `configs/ttt_memory.yaml:42` `lr:0.0`; seuls les params "memory" degelent (`trainer/distillation.py:283-285,367-374`). LaCT finetune Wan (`attn_only:true`, lr 2e-5, lr_multiplier 10), TTT-Video finetune tout le modele en curriculum 3->63s, VideoSSM entraine end-to-end. On injecte `x = x + mem_out` (`causal_model.py:601`) dans un flux residuel qui n'a jamais appris a lire la memoire.

3. Le write n'est meme pas entraine dans la config livree. `train_write` defaut False (`pipeline/streaming_training.py:69,84`): le write tourne sous `no_grad`. Seuls qkv/proj/lrs/gate et le seed w0/w1/w2 (init_state) recoivent du gradient. L'expressivite du WRITE -- la raison de garder un MLP -- n'est pas apprise.

4. Muon-x5 inconditionnel dans le write differencie. `video_memory.py:175-177` (defaut 5 `:214`), en bf16 avec son propre FIXME "2%-4% relative gradient error through checkpoint" (`video_memory.py:103`). LaCT video: `use_moun:false`. C'est la piece qui rend la BPTT inter-chunk couteuse et bruitee.

5. Gate non-borne + init bruite. Gate AdaLN non-borne `gate*(a_norm(x)*(1+scale)+shift)` (`video_memory.py:350-355`), w0/w1/w2 init randn/sqrt(fan_in) (`video_memory.py:255-257`). TTT-Video: gate borne `tanh(alpha)`, alpha init 0.1. LaCT: gate silu zero-init + init "clean"/identite. Une injection non-bornee d'un read bruite dans un flux gele = degradation uniforme d'identite (coherent avec `[[mag-init-weights-clobbers-gate]]`).

## Le verrou central: rendre la BPTT du MLP-TTT tractable (la solution)

VideoSSM evite la BPTT a gradient interne via la close-form; nous devons la rendre faisable SANS perdre le MLP. La solution combine trois leviers qui se multiplient:

1. Windowed-DMD: dissocier un long forward d'un court backward (LE mouvement make-or-break). Derouler K_warmup chunks via `generate_next_chunk(requires_grad=False, detach_state=True)` -- le write tourne TOUJOURS sous no_grad et construit un vrai etat long-horizon -- puis activer le grad seulement sur les W derniers chunks (ex. 1-2), echantillonnes LOIN (au-dela de `local_attn_size+sink`), sommer leur DMD et un seul backward. La profondeur memoire devient W chunks, pas K. La plomberie existe deja: boucle `bptt_window` (`trainer/distillation.py:1117-1137`), detach au debut de fenetre (`model/streaming_training.py:294-298`), args requires_grad/detach_state (`model/streaming_training.py:483,207-213`), `train_write=True` (`pipeline/streaming_training.py:69-77`). C'est un re-ecriture d'environ 25 lignes, pas un refactor.

2. Muon OFF (`muon_update_steps=0`, deja supporte `video_memory.py:214`; saute Newton-Schulz `:175-177`, garde la renorm `:184-187`). Supprime l'erreur bf16 2-4% (`:103`) et la partie la plus profonde/chere du backward par write. Aligne sur LaCT video. A exposer dans `model_kwargs` car `causal_model.py:498` ne le passe pas aujourd'hui (hardcode 5).

3. Checkpoint UNIQUEMENT le `_write` interne (KV-independant). `ckpt_write=true` (`configs/ttt_memory.yaml:81` -> `video_memory.py:219,381-383`). Point cle souvent mal diagnostique: la region checkpointee est `swiglu_fast_weight_gradient+weight_norm_update` avec entrees (k,v,w0,w1,w2,lr), elle ne touche JAMAIS le kv_cache; donc le conflit "checkpoint vs rolling-KV" de `[[crosschunk-bptt-infeasible]]` concernait les blocs entiers, pas le write. Avec Muon off le recompute du write devient EXACT et bon marche.

4. Reduire la profondeur sequentielle de writes. La taille de chunk (3f) n'est PAS le probleme (LaCT aussi), mais 7 writes sequentiels differencies par chunk de 21 frames (`streaming_switch_training.py:254-270`) le sont. Accumuler k,v sur ~6 frames et faire UN write par ~2 blocs ramene la chaine BPTT de ~7K a ~K. Garder l'apply-first par bloc pour la causalite AR.

Net: 1+2+3 transforment la BPTT full-window qui OOM en backward a memoire bornee, ET donnent enfin un gradient au vrai job cross-chunk. La diagnostic OOM precedent etait fait AVEC Muon-x5 et checkpoint de blocs; le retirer change materiellement le tableau VRAM (baseline `[[vram-readwrite-2gpu-result]]`: 2x48GiB, fenetre ~24 latents).

## La recette d'entrainement portee au MLP (signal long-range + backbone + gate)

1. Signal long-range (necessaire, pas seulement la tractabilite). Meme une BPTT parfaitement cheap entraine le write a ne rien faire si la fenetre est trop proche. Echantillonner la fenetre DMD a un offset > fenetre KV, et idealement un curriculum d'horizon (3->63s, code commente `model/streaming_training.py:393-415`). Important: le teacher DMD memoryless ne sait pas superviser l'identite a 50s; ajouter une cible explicite -- distillation de features teacher-ancrees sur [latents ancre chunk0 + fenetre lointaine] (le teacher est bidirectionnel, il voit l'ancre), OU text-dropout p~0.6 sur la fenetre lointaine pour que le SEUL chemin vers le sujet passe par la memoire (cf. Memorize-and-Generate). Sans cela, "un gradient de stockage existe" n'est pas "stockage de la BONNE chose (identite)".

2. Degeler l'hote (le delta #1 commun a LaCT/TTT-Video/VideoSSM). Donner un lr>0 au groupe LoRA/other_params (`configs/ttt_memory.yaml:42`, `trainer/distillation.py:372`), ex. 1e-5, version la moins risquee: LoRA sur le self_attn/o-proj/norms du bloc 5 seulement (`memory_layers:[5]`). Garder `lr_mem` ~10x (mirror LaCT lr_multiplier:10). Conserver une variante gate-only gelee pour l'ablation propre.

3. Gate borne + init inerte. Remplacer la gate non-bornee (`video_memory.py:350-355`) par `tanh(alpha)*read`, alpha/g_init=0.0 (do-no-harm reel, `configs/ttt_memory.yaml:80`; le commentaire `video_memory.py:217` confirme que 0.0 est le warm start inerte). Init "clean"/identite des w0/w1/w2 (`video_memory.py:255-257`) pour que le cold read soit une quasi-identite et non du bruit. Baisser base_lr de 1e-2 (`video_memory.py:261`) vers ~1e-3 (LaCT). Garder `reset_parameters()` pour eviter le clobber global (`[[mag-init-weights-clobbers-gate]]`).

4. Ordre: poser d'abord 2 (degel) + 1 (horizon), qui sont method-agnostic et diront en un retrain si UN memoire parametrique a un job ici; puis 3 (gate/init) qui n'a de sens qu'une fois le job present (sinon gate reste inerte ~6e-4 comme deja observe).

Note honnete: la regle interne delta/reconstruction (TTT-Video) au lieu du dot-ascent est une ablation optionnelle (point fixe `f(k)=v`, plus stable sous BPTT profonde), PAS un correctif: LaCT prouve que le Hebbien marche. A tester apres 1-4, jamais avant.

## Exploiter l'expressivite du MLP (ou il bat la matrice, et le test qui le montre)

C'est le coeur scientifique. La seule preuve publiee "MLP>matrice pour la video" est TTT-Video (TTT-MLP > TTT-Linear, +34 Elo) -- mais en regime teacher-forced, backbone finetune, v-pred sur video reelle. Dans VOTRE regime DMD-AR exact, la matrice gated-delta de VideoSSM bat deja LongLive ET Self-Forcing (Dynamic Degree 60s 50.5 vs 37.5 vs 35.0; subject-consistency 92.5 vs 91.1 vs 88.3). Donc aujourd'hui zero evidence que l'expressivite MLP batte la matrice sous DMD, et la matrice est bien moins chere a entrainer.

Ou la non-linearite peut gagner: le rappel non-Markovien, compositionnel, multi-sujets -- re-identification A->B->A, ou une recurrence lineaire/matricielle (capacite finie, rappel lineaire) plafonne. Le DMD marginal-matching ne fera PAS emerger cela sans perte d'identite explicite.

Le test decisif (gate tout l'effort dessus): implementer une variante matrice close-form (VideoSSM Eq.8) a cote de `VideoMemoryLayer`, reutilisant l'API `init_state/step/forward_sequence` (`video_memory.py:357-410`), sous le MEME pipeline DMD, MEME gate, MEME cadence, MEME windowed-DMD, MEME perte d'identite. Comparer head-to-head sur derive 60-240s et surtout Dynamic Degree (PAS subject_consistency, adjacency-blind, aveugle a la vraie derive) + une tache A->B->A explicite (machinerie de switch deja la: `switch_choices [21,39]` `configs:103-106`, `switch_global_frame` plombe `causal_model.py:578-587`, previous_frames `model/streaming_training.py:633`). Si MLP>matrice a budget egal: c'est LE papier. Sinon: resultat negatif/egal honnete -> la close-form est le bon outil pour le self-rollout, l'expressivite MLP appartient au regime teacher-forced.

## Le plan concret (etapes) + la premiere experience

Etape 0 -- Falsification gratuite (inference seule, <1 jour, knobs deja la). (a) Forcer gate=0 (`causal_model.py:601`) -> doit reproduire l'AF baseline bit-a-bit (meme base+LoRA+seed); sinon fuite d'init/etat partage. (b) FREEZE-STATE-NEVER-WRITE: lire init_state chaque chunk, do_update off (`video_memory.py:357-395`); si l'identite chute encore, c'est le read-transform, pas l'etat porte. (c) Probe `MEM_ZERO_COLD_FRAMES` (`causal_model.py:597-600`) pour isoler le cold-start. Surveiller dbg_read_rms/dbg_out_rms (`video_memory.py:393-394`). Cela ordonne les retrains et donne l'interpretabilite mem_out demandee par le superviseur.

Etape 1 -- Parite de substrat (config + ~1 jour code, sans retrain lourd): Muon off (`muon_update_steps=0`), gate `tanh` g_init=0, init clean, base_lr 1e-3, `ckpt_write=true`. Tout dans `video_memory.py` + `model_kwargs`.

Etape 2 -- Le verrou tractabilite: re-ecrire la boucle BPTT (`trainer/distillation.py:1117-1137`) en windowed-DMD asymetrique (K_warmup no_grad qui ECRIT + W chunks avec grad), `train_write=true`, `bptt_window`>1, fenetre echantillonnee loin. Asserter que le grad des w-params remonte d'une perte tail-only.

Etape 3 -- Donner un job: degel LoRA bloc 5 (lr 1e-5) + perte d'identite (text-dropout d'abord, ~gratuit; distillation teacher-ancree ensuite). Curriculum horizon 5->60s. Eval 60/120/240s.

Etape 4 -- Le verdict scientifique: ablation appariee MLP-vs-matrice (VideoSSM Eq.8) sous pipeline identique, Dynamic Degree + A->B->A.

Premiere experience a lancer (apres Etape 0): un retrain unique combinant Etape 1 + Etape 2 minimal (W=1 fenetre lointaine, Muon off, ckpt_write) + Etape 3 leger (degel LoRA bloc 5 lr 1e-5, gate tanh zero-init, text-dropout p=0.5 sur la fenetre). C'est le plus petit changement qui cree pour la premiere fois un gradient de stockage cross-chunk traversant des writes differencies, dans un hote tolerant. Logger live sur wandb (`[[always-wandb-for-training]]`), `mem_breakdown:false` (`[[write-training-vram-leak]]`). Critere de succes: a 60s+, Dynamic Degree et derive d'identite s'ameliorent vs la baseline read-only AF, gate non-effondree, pas de degradation uniforme. Si oui, enchainer l'ablation matrice (Etape 4); si la gate s'effondre ou l'identite chute encore, c'est le diagnostic que sans cible long-range supervisee aucun memoire parametrique n'a de job -> escalader la perte d'identite avant de blamer le MLP.

## Risques et chevauchement avec le deja-publie

- Chevauchement substrat: notre module EST LaCT (portage verifie ligne-a-ligne sur Wan-1.3B). "MLP fast-weight pour video AR" est publie. Non defendable seul.
- Chevauchement methode: "entrainer un memoire sous DMD/Self-Forcing self-rollout" est VideoSSM, qui bat deja LongLive et Self-Forcing. "On a entraine un write par rollout" n'est PAS une contribution.
- Risque de convergence: pour rendre le MLP entrainable on adopte la recette LaCT (no Muon, gate tanh, backbone entraine) + le lens VideoSSM (windowed-DMD); on devient alors "LaCT avec une perte externe DMD", delta modeste sur LaCT et VideoSSM.
- Risque scientifique central: les fixes de regime (objectif horizon + degel) peuvent rendre util N'IMPORTE quel memoire (MLP OU matrice); la novelty MLP n'existe PAS tant que l'ablation appariee (Etape 4) ne montre pas un ecart sur une tache ou la non-linearite compte. L'evidence MLP>Linear est uniquement teacher-forced; son transfert au DMD est non prouve, c'est le pari.
- Risque tractabilite residuel: la fenetre BPTT reste bornee VRAM (~24 latents, `[[vram-readwrite-2gpu-result]]`) et le chunk est cape a 21f (seq_len 32760); le windowed-DMD lointain contourne cela mais le valider est obligatoire.
- Recommandation finale: poursuivre comme contribution etroite et a haute variance, cadree "memoire non-lineaire expressive > matrice close-form SOUS DISTILLATION sur le rappel", PAS comme "nouvelle astuce d'entrainement du write". Gater tout l'effort sur l'ablation MLP-vs-matrice; etre pret a reporter honnetement l'egalite. Voisins 2026 (Attend Locally Remember Linearly 2605.16579, OmniMem 2605.30519, A2RD 2605.06924) sont tous lineaires/retrieval -- la cellule MLP-expressif reste vide, ce qui confirme `[[novelty-fast-weight-write-self-rollout]]` mais rappelle que vide != avantageux.

Fichiers cles: `wan/modules/video_memory.py` (objectif `:129-155`, Muon `:175-177,:214,:103`, gate `:350-355`, init `:255-257`, base_lr `:261`, checkpoint write `:219,:381-383`, API `:357-410`), `wan/modules/causal_model.py:498,556-604,597-600`, `pipeline/streaming_training.py:69,79-84`, `pipeline/streaming_switch_training.py:254-270`, `model/streaming_training.py:294-298,393-415,483,650-689`, `trainer/distillation.py:283-285,367-374,1117-1137`, `model/dmd.py:106-120,161-227`, `configs/ttt_memory.yaml:42,48,60,77-82,107-108`.
