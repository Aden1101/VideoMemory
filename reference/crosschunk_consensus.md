# Cross-chunk write-training: consensus multi-agents (2026-06-28)

6 agents (5 critiques sous angles distincts + 1 synthese), tous ancres dans le code, ont relu et
critique `reference/crosschunk_solution_analysis.md`. Verdict et marche a suivre agreee ci-dessous.
Workflow run id: wf_d1ea9f96-29e (398k tokens, 100 tool calls).

## Verdict sur la conclusion (la critique tient-elle ?)

Globalement OUI, avec trois corrections. Le coeur technique du doc est verifie dans le code : `memory.step`
(video_memory.py:366-395) ne lit jamais le KV cache (il ne prend que `x=h` et l'etat `w` prealable ; le
write = `swiglu_fast_weight_gradient` + `weight_norm_update`, k/v/lr derives de h seul), et `mem_out` est
deja detache sur la passe write (causal_model.py:593). Le gradient cross-chunk porte sur l'etat fast-weight
est donc bien cache-free, et le snapshot du cache est a juste titre abandonne.

Mais trois affirmations du doc sont fausses ou trompeuses et changent l'ordre :
1. "Curriculum sur la longueur in-context" presente comme route cheap AVANT le cross-chunk BPTT : incoherent.
   Le forward est plafonne a 21 latent frames (`assert seq_lens.max() <= seq_len`, causal_model.py:1145). On
   ne peut pas allonger la sequence d'un seul forward ; le seul curriculum disponible est de ramper K (nombre
   de chunks portes sous un backward), ce qui EST le cross-chunk BPTT. Le curriculum n'est pas une alternative
   moins chere : c'est le planning qui tourne SUR l'infra cross-chunk.
2. "Detacher h ne perd ZERO gradient car backbone gele (lr=0)" : surestime. lr=0 bloque le pas optimiseur,
   pas l'autograd a travers le backbone. Un gradient remonte vers les params memoire (trainables) via la route
   read `mem_out` -> `denoised_pred` -> `context_noisy=add_noise(...)` (switch_training:240-248) -> write h.
   Detacher h coupe ce credit read-via-write. Route tres attenuee (re-noising + ~25 couches gelees),
   probablement negligeable, mais a mesurer, pas a postuler.
3. Mode d'echec C "l'etat decroit trop vite = un knob renorm scalaire (185-187)" : mecaniquement faux.
   `weight_norm_update` repine chaque colonne a sa norme d'origine (167, 185-187) : la norme est CONSERVEE, il
   n'y a aucun terme de decay. L'oubli est purement directionnel (ecrasement), pilote par `base_lr` (defaut
   1e-2, video_memory.py:213, non expose dans le yaml) x Muon. Il n'existe pas de knob de decay a tourner ; il
   faut en AJOUTER un.

## Points d'accord (les 5 critiques convergent)

- `memory.step` est cache-free au niveau fonction ; c'est le seul chemin cross-chunk trainable vivant (KV
  cache = backbone gele + detache).
- EVAL FIRST avant toute construction.
- Snapshot du cache infaisable (~90 GiB K=2), abandonne.
- Le cross-chunk BPTT est une REECRITURE, pas un flip de flag : la boucle dormante (distillation.py:1118-1137 :
  somme de K pertes, un seul backward) crashe car le checkpoint au niveau bloc (causal_model.py:1217) enveloppe
  `self_attn` ET la memoire, donc le recompute relit un cache deja avance.
- Si BPTT : Muon/Newton-Schulz en fp32 ou desactive (LaCT video : `use_moun=false`) ; sinon l'erreur bf16
  2-4%/pas (video_memory.py:103) se compose sur K.
- Teacher-distillation AVANT tout BPTT.

## Desaccords et resolution (qui gagne et pourquoi)

- Ordre du doc (curriculum avant BPTT) : gagnent les critiques 2/3/5. Le plafond 21f rend le curriculum-sur-
  longueur identique au cross-chunk ; on le reclasse comme planning K, pas comme route separee.
- "Detach h = zero gradient perdu" : gagne la critique 1. Correct = zero perdu VERS LE BACKBONE ; la route
  read-via-write h est larguee. On le mesure (ratio grad-norm avec/sans detach-h) plutot que de l'affirmer.
- Mode C "knob renorm" : gagnent 2/3/5. La retention reelle = `base_lr` + Muon ; le vrai knob manquant est un
  decay/forget explicite a ajouter.
- Teacher-distill "aussi efficace que le cross-chunk" : gagnent 3/4/5. C'est une cible STATIONNAIRE within-chunk
  (proxy), qui n'entraine PAS le credit cross-chunk ; on le garde pour ce qu'il repare vraiment (le write du
  dernier bloc, sous no_grad switch_training:254, qui ne recoit aucun gradient) et la qualite de feature,
  reutilisant le forward `real_score` deja lance (dmd.py:448), past+current only.
- Config : gagne la critique qui pointe le mismatch. Le yaml committe entraine `memory_layers:[5]` (1 couche,
  ttt_memory.yaml:79), pas [5,16,27]. Avec `gate_init 0.1` + `lr_mem 1e-4`, le bottleneck le plus probable est
  gate/capacite/oubli, pas l'horizon. On EPINGLE la config cible avant tout chiffre.
  (NB Aden : notre run e2_write3L a bien tourne en [5,16,27] via /tmp/_write3L_real.yaml, et l'eval pointe le
  bon checkpoint avec layers [5,16,27]. Le mismatch concerne le yaml committe, pas notre run.)

## Idees nouvelles retenues

- Diagnostics zero-training (decident A vs C/D gratuitement) : (a) freeze-state probe (a l'inference, arreter
  le write apres le chunk k ; si l'identite s'ameliore -> ecrasement/capacite, si elle se degrade -> read/gate)
  ; (b) log `cos(w_k, w_0)` + norme par colonne (doit etre plate par construction) + gate vs frame ; (c) ratio
  grad-norm vers params memoire avec/sans detach-h (transforme la claim contestee en nombre).
- Test "crash = eviction" : monter `slice_last_frames` pour qu'aucune eviction ne tombe dans la fenetre, lancer
  K=2 ; si le CheckpointError disparait, le crash est un artefact d'offset (causal_model.py:309), pas un
  couplage cache fondamental. Validation quasi gratuite avant tout refactor.
- Ajouter un vrai terme de decay/forget (scalaire ou par canal) dans `weight_norm_update` : a la fois knob
  d'eval cheap ET le mecanisme que le champ utilise pour le long horizon. Absent aujourd'hui.
- Controle quasi gratuit non-memoire : agrandir `sink_size` (yaml:78, actuellement 3). Si un sink plus large
  ferme l'ecart 60/120/240s, tout l'effort memoire est discutable.
- Si le BPTT est instable/couteux : memoire fast-weight LINEAIRE (Gated DeltaNet) -> backward = scan transpose,
  cache-free par construction, forget gate gratuit ; OU variante RTRL/Jacobien analytique (le state-update est
  ferme : `swiglu_fast_weight_gradient` + `weight_norm_update`) qui ne re-rentre jamais dans le backbone ->
  cache-free par construction plutot que par detach soigneux.
- Garde-fou : `assert` que tous les groupes de params non-memoire ont lr==0 quand detach-h est actif ; et
  detacher UNIQUEMENT la copie h de la memoire (h=norm1(x_in) est partagee avec self_attn, causal_model.py:
  540/583 ; detacher la copie partagee tuerait le gradient de self_attn).

## Marche a suivre (ordonnee, cout/effort + gain attendu)

1. EPINGLER la config cible (1L [5] du disque vs 3L [5,16,27]) et lancer l'eval : VBench subject-consistency a
   60/120/240s + score identite par longueur. Cout S. Gain : la decision de tout le reste.
2. EN PARALLELE, instrumenter l'eval pour diagnostiquer la RETENTION, pas seulement VBench-par-longueur : log
   `cos(w_k,w_0)`, norme par colonne, gate/`dbg_out_rms`/`dbg_read_rms` (video_memory.py:393-394) vs frame ; +
   freeze-state probe ; + ratio grad-norm detach-h. Cout S (read-only/probe). Gain : distingue A (horizon) vs B
   (read/gate) vs C (oubli) vs D (gate~0).
3. Si NON horizon-limite (mode B/C/D, le plus probable vu gate_init 0.1 / lr_mem 1e-4) : sweeps cheap sur les
   BONS axes : `lr_mem` 1e-4 -> 3e-4/1e-3 et/ou warmup de gate (D) ; `base_lr` (code, 1e-2) + ajout decay/forget
   (C) ; `sink_size` comme baseline (control). Cout S-M. Gain : peut fermer l'ecart sans aucun graphe
   cross-chunk. STOP si ca marche.
4. Si encore court : teacher feature-distillation, hooke sur le forward `real_score` deja lance (dmd.py:448),
   past+current only, avec un projecteur appris (teacher bidirectionnel, espace/timestep differents). Repare le
   write de bord non-entraine (switch_training:254). Cout M. Gain : feature-quality + boundary-write, BPTT-free.
5. SEULEMENT si horizon-limite confirme : cross-chunk BPTT cache-free-state, traite comme reecriture : (a)
   remplacer la boucle somme/un-backward (distillation.py:1118-1137) par backward immediat par chunk + injection
   du grad d'etat via leaf `s_k = w_{k-1}.detach().requires_grad_()` ; (b) detacher le KV cache a chaque
   frontiere de fenetre (porter `_clear_cache_gradients` dans la boucle generateur ; aujourd'hui seulement dans
   compute_critic_loss) ; (c) write-pass backbone sous no_grad, seul `memory.step` update sous enable_grad ; (d)
   Newton-Schulz en fp32 ou Muon off ; (e) valider d'abord K=2 avec eviction neutralisee (etape diagnostic
   ci-dessus) ; (f) garde-fou lr==0 + detach sur la copie h memoire seule. Le curriculum = ramper K (2->4->...)
   PAR-DESSUS cette infra, pas avant. Cout L. Gain : le seul chemin qui entraine reellement le carry 60-240s.
6. Si le BPTT cale (cout/stabilite) : basculer la fast-weight SwiGLU vers une fast-weight lineaire (Gated
   DeltaNet) pour forward+backward chunk-paralleles et forget gate par construction. Cout L. Gain : tractabilite
   + oubli structurel.

## Ce que l'eval en cours (60/120/240s) fera basculer

- Gate plat ~0 (`dbg_out_rms` faible) -> mode D : monter `lr_mem`/warmup gate, la memoire ne contribue rien
  (etape 3). Pas d'horizon.
- Gate ouvert mais `cos(w_k,w_0)` chute vite / read-RMS plat a distance -> mode C/B : ajouter decay/forget +
  tourner `base_lr` (etape 3). Pas d'horizon.
- Freeze-state ameliore l'identite tardive -> ecrasement/capacite : decay + two-timescale, pas du BPTT.
- Etat STABLE, gate ouvert, identite qui derive quand meme, ET freeze-state degrade -> mode A vrai horizon :
  passer teacher-distill (4) puis cross-chunk BPTT (5).
- Si le test "crash=eviction" supprime le CheckpointError a K=2 : le crash etait un artefact d'offset, le
  cross-chunk devient nettement moins risque ; sinon, prevoir le refactor complet (5a-f) avant tout K>1.

## Fichiers de reference (chemins absolus)

- wan/modules/video_memory.py : step 366-395, renorm 167/185-187, base_lr 213, Muon bf16 FIXME 103, dbg_*_rms 393-394
- wan/modules/causal_model.py : memoire 556-596, mem_out detach 593, h partage 540, in-place cache 1036-1037, assert seq_len 1145, checkpoint bloc 1217, offset recompute 309
- pipeline/streaming_switch_training.py : write-context + no_grad dernier bloc 240-256
- trainer/distillation.py : boucle bptt_window 1112-1137, _clear_cache_gradients (compute_critic_loss only)
- configs/ttt_memory.yaml : lr 42, lr_mem 48, memory_layers [5] 79, gate_init 80, ckpt_write 81, slice_last_frames 86, streaming_chunk_size 107
