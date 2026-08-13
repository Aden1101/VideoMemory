# Pourquoi NOTRE memoire degrade (vs TTT-Video/LaCT/Scal3R) -- brainstorm critique 2026-06-28

6 agents (5 lentilles critiques code+papiers + consensus), run wf_8bddda58-557.

## Le verdict en une phrase

Notre `VideoMemoryLayer` ne mesure jamais l'identite pendant l'entrainement (loss DMD par-chunk/par-frame, sans terme cross-frame, model/dmd.py:146,151-153,222-226) et injecte un read dont la *magnitude est decouplee de la pertinence* (double normalisation o_norm/a_norm + gate non borne, video_memory.py:344-355): SGD apprend donc un read qui baisse le DMD marginal en ajoutant un vecteur de RMS ~constant a chaque frame sur 3 blocs (causal_model.py:601), ce qui ecrase l'identite *uniformement*; TTT-Video/LaCT marchent parce que leur read est supervise par une loi de reconstruction sur de vraies videos longues, ou l'identite EST dans la cible.

## Pourquoi notre read est toxique (causes classees, preuve code)

Les 5 audits convergent fortement. Classement par probabilite, avec reconciliation des desaccords:

**#1 -- L'objectif ne contient AUCUN signal d'identite (cause racine, consensus 5/5).**
Seule la memoire s'entraine (configs/ttt_memory.yaml:42 `lr:0.0`, :48 `lr_mem:1e-4`; trainer/distillation.py:283-285, 367-374). Son unique gradient est le DMD par-chunk: `grad = pred_fake - pred_real` (model/dmd.py:146), normalise *par frame* sur les dims [1,2,3,4] (model/dmd.py:151-153, verifie), puis `0.5*MSE` masque sur les nouvelles frames (model/dmd.py:222-226; model/streaming_training.py:682-689). C'est une realisme/adherence-prompt *marginale par frame*, evaluee contre un teacher bidirectionnel *sans memoire*. Rien ne mesure la coherence sujet entre frames ou entre chunks. Consequence logique: SGD utilise le read comme capacite supplementaire pour grignoter le score par-frame; le DMD sain (~0.12) et la chute DINO coexistent precisement parce que le DMD est aveugle (ici anti-correle) a ce que VBench/DINO mesurent. Le fait que le gate se soit ouvert (0 -> 1.26) *prouve* que le read aide le DMD; la chute d'identite prouve que cet objectif est le mauvais proxy.

**#2 -- La magnitude du read est decouplee de la pertinence (mecanisme de l'uniformite, consensus 5/5).**
`_project_output` (video_memory.py:344-355, verifie) fait: `x = o_norm(x)` (RMSNorm, :265) puis `gate * (a_norm(x)*(1+scale)+shift)` ou `a_norm` est une LayerNorm *sans affine* (:244). Les deux normalisations detruisent la magnitude du read: il ne reste que sa *direction*. La taille injectee est donc fixee uniquement par le gate, qui est la 3e tranche brute de `proj(x)`, *non borne* (pas de tanh/sigmoid). Ouvert a 1.26, on additionne un vecteur de RMS ~1.26 directement dans le residu (`x = x + mem_out`, causal_model.py:601, verifie), a chaque frame, sur les blocs [5,16,27]. Il n'existe AUCUN chemin pour que le read se taise quand la memoire n'a rien de pertinent. C'est exactement une perturbation quasi-stationnaire de magnitude fixe -> chute DINO ~-0.25 *plate*, ni front-loaded ni croissante avec l'horizon.

**#3 -- Le read est dimensionne par frame, donc la perturbation est differente a chaque frame.**
Le query `q = qkv(norm1(x_in))` (video_memory.py:322-325; causal_model.py:585) varie par frame, donc la *direction* injectee change chaque frame, alors que sa norme reste ~constante. Un kick O(1) de direction variable a chaque frame est precisement un brouilleur de coherence frame-a-frame: il rabat la consistance sur un plancher (~0.68) immediatement et l'y maintient. Cela reconcilie les 4 mesures: (a) uniforme = injection additive par frame; (b) plancher fixe = ni cold-start ni horizon; (c) gap qui *retrecit* avec la longueur = la baseline derive lentement (0.973 -> 0.932 ...) vers ce meme plancher, donc les deux convergent; (d) zeroing du read sur 12 frames -> recuperation locale = retirer l'injection retire la corruption.

**#4 -- L'apply-first depuis un etat generique prouve que c'est le READ-TRANSFORM, pas la memoire accumulee.**
`step()` lit avant d'ecrire (video_memory.py:374) et `init_state` renvoie les slow-weights globaux w0/w1/w2 (:357-360, init `randn/sqrt(fan_in)` :255-257). Sur le 1er chunk le read ne porte AUCUNE info sujet accumulee, et pourtant le 1er bin DINO chute deja 0.973 -> 0.746. Preuve code que le mal est la transformation de lecture elle-meme (un injecteur de features par frame), pas le carry/BPTT (correctement exclu par tes mesures).

**#5 -- Le backbone gele ne peut jamais apprendre a consommer ou attenuer le read.**
`lr:0.0` sur tout sauf la memoire (trainer/distillation.py:372-374). Le read est un vecteur OOD ajoute a des couches aval figees, jamais entrainees a le recevoir. La memoire ne peut baisser le DMD qu'en *ajoutant* (jamais en raffinant l'identite en place), et l'ajout le moins cher contre un critic qui s'adapte est un push realisme/detail par frame qui ecrase l'identite specifique au seed. Contraste direct avec TTT-Video qui *fine-tune backbone + self-attention + gates*.

**#6 -- L'etat fast-weight ne peut PAS stocker une identite stable.**
`weight_norm_update` memorise la norme de chaque colonne avant update et y renormalise apres l'ascension (video_memory.py:167-169,185-187): le fast-weight ne peut que *tourner* dans une coquille de norme fixe, jamais accumuler une association haute-SNR persistante. L'objectif interne ascend `<f_W(k),v>` (swiglu_fast_weight_gradient, :129-155), une auto-association generique des activations du modele, sans cible "stocke le sujet". Le read est donc une projection dominee par la geometrie d'init aleatoire plus une rotation du chunk recent: content-following mais identity-agnostic.

**#7 -- Detail aggravant: mismatch read/write.** Le write prend k,v du chunk de contexte ~propre (`ttt_update=True`, pipeline/streaming_switch_training.py:243-270), le read prend q des latents *bruites* aux pas de denoising (`ttt_update=False`, :184,223). On interroge la memoire hors de la distribution ou elle a ete ecrite. Secondaire mais decorrele encore le read du contenu reel.

## Les divergences critiques code-vs-papiers (TTT-Video / LaCT / Scal3R)

| Axe | TTT-Video (2504.05298) | LaCT (2505.23884) | Scal3R (2604.08542) | NOUS |
|---|---|---|---|---|
| **Objectif qui supervise le read** | denoising/v-pred sur *vraies* videos, curriculum 3->9->18->30->63s; la cible EST une video coherente | reconstruction/flow sur donnees reelles (NVS = vues tenues, la loss EST la coherence cross-vue) | recon 3D *a l'inference seulement*, cible = scene geometrique reelle | **DMD par-frame vers teacher sans memoire** (model/dmd.py:146,222-226). Aucune cible de reconstruction, aucun terme identite |
| **Gate** | `tanh(alpha)*TTT(X)+X`, alpha init ~0.1, *borne* (-1,1), do-no-harm warm start | scale par-tete content-adaptive `silu(scale_proj(x))` (peut tendre vers 0) | g_init~0.1, sans terme generatif qui tire le read hors identite | **non borne**, `gate*(...)`, ouvert a **1.26** (video_memory.py:350-355) |
| **Magnitude du read** | bloc TTT residuel `f(x)=x+LN(MLP(x))` | `l2_norm(q,k)` + `silu(v)` + o_norm + scale adaptatif | RMS-norm (inoffensif car le read est *toujours* significatif: ecrit depuis de vrais patches) | **double-normalise** o_norm + a_norm => magnitude = gate seul, decouplee de la pertinence |
| **Objectif interne** | L2 reconstruction `||f(theta_K x)-theta_V x||^2`, residuel+LN | ascension du dot + L2-norm (reglarisateur) | accumulation renorm-to-init | ascension `<f_W(k),v>`, *pas* de point fixe associatif |
| **Update** | LR internes petites apprises | grands chunks (faible variance); **video config: `use_moun=false`** | renorm-to-init, fixe (OK pour inference bornee) | **Muon Newton-Schulz steps=5** + renorm-to-init, ecriture par bloc de 3 frames (~6 rotations/chunk) |
| **Backbone** | fine-tune (incl. self-attn) | slow-weights entraines par la tache | jamais entraine | **gele** (lr=0) |

**Reconciliation d'un desaccord important entre audits.** L'audit #4 fait remarquer, a juste titre, que notre *equation* d'update (ascension du dot + L2-renorm) **correspond a LaCT**, donc la regle d'update n'est pas le bug en soi. La nuance correcte: (a) la *direction* d'update est LaCT-compatible; (b) MAIS LaCT video met `use_moun=false`, alors que nous tournons Muon steps=5 (orthogonalisation plein-spectre = ecriture large-bande qui destabilise la video), et (c) LaCT laisse les poids *accumuler* via L2-norm comme regularisateur dans un regime grands-chunks faible-variance, tandis que notre renorm-to-init *epinglee* + petits chunks de 3 frames empeche l'accumulation et maximise la variance. Le verdict: la regle interne n'est pas fausse, c'est son *regime* (Muon ON, micro-chunks, renorm dure depuis un init aleatoire) plus l'*absence d'ancrage par une loss de reconstruction* qui la rend instable. Donc on ne range pas "mauvaise equation d'update" en cause racine; on range "Muon-on-video + micro-chunks" en cause secondaire (#4 du classement papiers), et l'objectif en cause racine.

**Le contresens Scal3R (suspect confirme).** Scal3R est de la reconstruction 3D *a l'inference*: le fast-weight est ajuste a UNE scene statique sur un ensemble borne de vues, le read tire *toutes* les vues vers un etat global partage (correct pour une geometrie statique). Ses choix (renorm-to-init, LR interne fixe, read RMS-normalise, apply-first depuis init generique, *aucun* gate de pertinence/oubli) sont des choix de *stabilite-pour-inference-bornee* ou le read est *toujours* significatif (etat ecrit depuis de vrais patches observes), donc la normalisation est inoffensive. Transplantes dans une memoire-identite *entrainee* pour de la video AR generative: (i) "tirer chaque frame vers un attracteur global partage" aplatit l'identite par-frame; (ii) RMS-normaliser le read force la magnitude pleine meme quand la memoire est non pertinente; (iii) renorm-to-init bloque tout stockage persistant; (iv) l'auto-association n'a aucune cible identite. C'est la categorie d'erreur centrale.

## Ce qui est fondamentalement mal pose dans notre design

Trois pathologies *structurelles* (pas des hyperparametres):

1. **On optimise un proxy desaligne.** Le read est juge purement sur le DMD par-frame; or sur un student AF deja converge vers le teacher, le read globalement correct pour cette loss est ~0 (gate -> 0). Que le gate soit monte a 1.26 signifie que le read a trouve une *reduction locale* du DMD via de la capacite texture/detail que le score par-frame tolere mais qui detruit l'identite. L'objectif recompense activement le comportement nuisible.

2. **La magnitude est decouplee de la pertinence.** o_norm + a_norm + gate non borne => `||mem_out|| ~ gate*sqrt(dim)`, independant du contenu de la memoire. Pas de chemin "rester silencieux si rien d'utile". C'est *pourquoi* le mal est uniforme.

3. **Le module ne peut pas stocker une identite, meme en principe.** renorm-to-init (rotation dans une coquille de norme fixe) + init aleatoire dominant + auto-association sans cible identite. Meme avec le bon objectif, l'etat actuel ne peut pas faire grossir un slot d'identite haute-SNR.

## Le changement recommande (concret, code + entrainement)

Ordre de priorite = ordre de causalite. Ne PAS commencer par les hyperparametres.

**(A) Corriger l'OBJECTIF (cause racine).** Le read doit etre supervise par quelque chose qui mesure l'identite cross-frame. Deux voies, par ordre de force:
- *Forte (recette TTT-Video/LaCT)*: pre-entrainer la memoire teacher-forced avec une loss de denoising/flow sur de vraies videos longues (ou sur les latents *propres* du long-rollout du teacher), curriculum d'horizon croissant, avant tout DMD on-policy. La cible EST une video temporellement coherente.
- *Minimale (additif au DMD existant)*: ajouter un terme auxiliaire de consistance cross-frame/cross-chunk dans `compute_generator_loss` (model/streaming_training.py:680-689), p.ex. cosine/feature-consistency DINO ou VAE entre la region sujet des nouvelles frames et des frames anterieures, sur la fenetre roulante. Le DMD pur ne peut pas, par construction, enseigner l'identite.

**(B) Decoupler magnitude et pertinence + rendre le read inerte au depart.** Dans `_project_output` (video_memory.py:344-355):
- supprimer la double-normalisation qui detruit la magnitude (retirer le `a_norm`, ou ne pas RMS-normaliser le read en sortie), pour qu'une memoire faible/non pertinente puisse rester silencieuse;
- remplacer le gate brut non borne par un `tanh(alpha)` par-canal init a 0 (do-no-harm), ou par un scale content-adaptif `silu(scale_proj)` facon LaCT qui peut tendre vers 0; mettre `gate_init=0.0` (pas 0.1, configs/ttt_memory.yaml:80; le commentaire video_memory.py:217 dit que l'inerte vrai est 0.0);
- mettre `gate_extra` (le gate "agreement" zero-init, reste marginal ~0.06) sur la *pertinence memoire* (norme du read / accord etat-query), pas un self-gate derive du read lui-meme;
- normaliser `mem_out` au RMS du residu avant `x = x + mem_out` (causal_model.py:601) pour qu'un bloc ne puisse jamais injecter un vecteur O(1) par frame.

**(C) Permettre le stockage et stabiliser l'update (corriger les imports Scal3R/Muon).**
- remplacer la renorm-to-init *dure* (video_memory.py:185-187) par une decroissance douce vers l'init / weight-decay, pour que l'identite s'accumule;
- passer l'objectif interne d'ascension du dot a une regression associative `||v - f_W(k)||^2` avec residuel+LN dans `swiglu_apply` (video_memory.py:122-126), pour un *rappel* fidele;
- `muon_update_steps=0` (Muon OFF) comme la config video LaCT; ou a minima Newton-Schulz en fp32 (video_memory.py:103 signale 2-4% d'erreur bf16);
- ecrire par chunk de 21 frames (un write/chunk) plutot que par bloc de 3 (pipeline/streaming_switch_training.py:254-270), regime grands-chunks faible-variance LaCT;
- reduire/apprendre-a-la-baisse `base_lr` (1e-2 hardcode, video_memory.py:213) pour que l'etat se stabilise au lieu de tourner pleinement chaque chunk.

**(D) Laisser le backbone co-adapter.** Donner un lr non nul a la self-attention (ou au groupe LoRA existant) en plus de la memoire (trainer/distillation.py:367-374), pour que les couches aval apprennent a consommer le read, comme le schedule etage de TTT-Video. Sinon tout read reste une perturbation OOD.

## Comment le verifier vite (avant un gros retrain)

Falsifications *sans reentrainement* d'abord, pour confirmer le ranking avant d'investir:

1. **Test magnitude/gate (confirme #2/#3).** A l'inference, clamper le gate a ~0.1 ET scaler `mem_out` a <=10% du RMS du residu (causal_model.py:601). Relancer le profil DINO par-frame matched-seed. Si le gap uniforme -0.25 s'effondre, #2/#3 sont confirmes comme contributeurs dominants du mecanisme d'uniformite.

2. **Test "le read sans memoire suffit a nuire" (confirme #4, deja a moitie fait).** Tu l'as deja: apply-first sur chunk 0 chute deja a 0.746. A renforcer: figer l'etat a `init_state` (jamais de write) et lire quand meme; si la chute persiste, c'est le read-transform, pas le carry.

3. **Test objectif (isole #1).** Le diagnostic le plus net: ajouter *seulement* le terme de consistance cross-frame (B-minimale) sur 1 bloc ([5]), gate clampe petit, court run. Si le gate ne s'ouvre que quand la consistance est dans la loss et que la chute DINO disparait, l'hypothese objectif (#1) est confirmee comme racine. Si le read degrade encore l'identite a consistance-active, alors le design d'injection residuelle (B/C) est le levier suivant.

4. **Sanity gate-a-zero.** Forcer gate=0 partout doit reproduire *exactement* la baseline AF (meme base.pt + LoRA + seed). Si ce n'est pas le cas, il y a une fuite (init_weights, etat partage) avant meme le read -- a verifier vu l'historique du bug init_weights qui clobberait le gate (cf. ta memoire MAG).

**Pari attendu:** (1) effondre une grande part du gap (mecanisme d'uniformite), (3) montre que le gap residuel ne disparait vraiment qu'avec un terme d'identite dans la loss (cause racine). Le couple gagnant a corriger en premier est donc **objectif (A) + decouplage magnitude/gate (B)**; renorm/Muon/backbone (C/D) sont necessaires pour que la memoire *aide* reellement une fois le mal retire, mais ne sont pas la cause premiere de la degradation uniforme.
