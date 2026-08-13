# RED-TEAM du plan MÉMOIRE 2.0 (write-replay) — 2026-07-27

Cible : `reference/memory2_design.md` (mandat `memory2_brief.md`). Lecture seule, rien lancé.
Code lu : `wan/modules/video_memory.py` (write_only/replay_writes/AnchorFrameDecoder),
`reference/runs/frame_dec/{selftest_replay,collect_pairs,train_decoder}.py`,
`reference/runs/reid_probe_l1/{run_reid_probe.py,probe_l1.yaml}`,
`pipeline/switch_causal_inference.py` (MEM_DECODE L406-443), `wan/modules/causal_model.py` L620-705.

**VERDICT : GO-AVEC-PATCHS**, mais **3 gates à 0-3 GPU-h passent AVANT la collecte** (F1.1a, F5.2,
F5.3a) et **une décision de collecte change au J2** (paires même-prompt/seed différent, F1.1b).
Le plan est techniquement solide sur ce qu'il a audité (le replay, la parité bitwise, le budget
mémoire) et **structurellement aveugle sur son objectif** : tel qu'écrit, l'InfoNCE est résoluble
par le PROMPT (le corpus est sélectionné pour que le texte contienne l'attribut d'instance), et
rien dans le plan ne mesure l'attribution état-vs-prompt. Le run peut « réussir » (+0.16) sans
que l'écrivain ait rien stocké : c'est un faux positif sur la thèse centrale du papier.

Sévérités : **[TUE]** = le run ne produit rien d'exploitable / se plante ; **[BIAISE]** = le run
tourne mais le verdict n'est pas interprétable ; **[COSM]** = bruit d'exécution.

---

## FRONT 1 — La loss InfoNCE anti-régression-de-catégorie

### F1.1 **[TUE LE VERDICT]** L'InfoNCE est résoluble par le TEXTE : la banque ne teste pas l'état
Le corpus n'est pas un corpus quelconque : `collect_pairs.is_reidentifiable` (L159-180) ne garde
un prompt **que si une regex `a/an/the [adj]{0,2} <ATTR> <noun>` matche**, c'est-à-dire **que si le
texte spécifie explicitement l'attribut distinctif du sujet** (« a red fox with soft fur », « an
orange house », « a white horse with a black mane »). Et `cond_emb` = T5 poolé de CE prompt, unique
par vidéo (135 prompts distincts) → **cond est à la fois un identifiant de vidéo et le porteur de
l'attribut d'instance**. Conséquences :
- la tâche « retrouve TA cible parmi 5-11k » est solvable à ~100% par le chemin FiLM seul ;
- `inst/acc` et `inst/acc_same_cat` monteront joliment **avec un gradient ≈ 0 vers l'écrivain** ;
- pire, à l'inférence le bras `memdec20` peut GAGNER (+0.16) pour la mauvaise raison : un décodeur
  devenu bon rendu-texte-vers-frame produit une ancre fidèle au prompt sans mémoire. Le papier
  claimerait « l'écrivain a appris à stocker l'instance » sur une preuve nulle.
- Note : ceci contredit frontalement le slogan maison « l'identité d'instance est sous-spécifiée
  par le texte » — **sur CE corpus elle est sur-spécifiée**, par construction du filtre.
Indice à charge déjà en main : au test discriminant (cheval, B=A), le prompt dit littéralement
« white horse with a black mane » et l'ancre v1.2 sort BRUNE ⇒ aujourd'hui le chemin cond est
sous-exploité **parce que la MSE ne le récompense pas** ; l'InfoNCE est précisément la première
loss qui le récompensera. C'est le chemin de moindre résistance, il sera pris.

**Patchs (à appliquer ensemble, sinon inutiles) :**
1. **(collecte, J2) paires même-prompt / seed différent** : collecter K prompts × 2 seeds. cond
   identique ⇒ **discriminabilité textuelle nulle** ⇒ l'InfoNCE restreint à ces négatifs est un
   test d'instance PUR (et ce sont aussi les négatifs même-scène qui manquent, cf. F1.2).
   Reco : ~45 prompts × 2 seeds + ~45 prompts × 1 seed (diversité conservée).
2. **(loss, J3) cond-dropout p=0.5** (et/ou variante « cond = tête nominale seule », extraite par
   la regex existante) : la moitié des steps exigent l'état.
3. **(banque) négatifs tirés parmi les plus proches voisins en espace cond** (T5) : cond cesse
   d'être discriminant, seul l'état peut trancher.
4. **(métrique de gating, OBLIGATOIRE) marge causale** : `inst/acc(état vrai) − inst/acc(état
   mélangé/init, même cond)` et l'équivalent sur la loss (`ctrl/state_shuffled_loss`). **Si la marge
   ≈ 0, la phase est inerte quel que soit l'absolu.** Ce compteur est ABSENT du §3.6 et c'est le
   seul qui répond à la question du chantier.

### F1.1a **[LE PROBE À FAIRE EN PREMIER, ~3 GPU-h]** bras `memdec_cond` sur le décodeur v1.2 EXISTANT
Patch minimal : dans `switch_causal_inference.py` L417-421, env `MEM_DECODE_STATE=init|shuffle` →
`_sts` remplacé par `layer.init_state(...)` (ou les états d'une autre cellule), tout le reste
identique ; ~10 lignes, inerte si non défini (règle additive respectée). Passer les 24 cellules.
**Lecture pré-enregistrée** : si delta(memdec_cond − memzero) ≥ +0.09, **le plateau +0.127 est
majoritairement un générateur prompt→frame**, l'état contribue ~0, et : (a) la baseline de 2.0
n'est pas memdec mais memdec_cond, (b) la cible « +0.16 » ne veut plus rien dire, (c) le diagnostic
du brief (« le goulot est l'état ») reste vrai mais le chiffre de référence change. Si delta ≤
+0.04, le chemin état porte bien le gain et le reste du plan se lit normalement.
**Insertion : J0, avant toute ligne de code 2.0.** C'est la mesure manquante la moins chère et la
plus structurante de tout le dossier.

### F1.2 **[BIAISE]** Le contrastif est satisfait par la SCÈNE, et il l'est peut-être déjà à l'init
Les négatifs inter-vidéos diffèrent en décor, palette, cadrage, style. DINO-CLS (même en crop60,
même en ViT-S) est dominé par la statistique de scène. Or le diagnostic v1 dit exactement :
« scène correcte, sujet infidèle ». Donc **le décodeur v1.2 gagne probablement déjà le retrieval**
→ softmax saturé → gradient ≈ 0 **précisément sur l'axe visé**. L'argument du design (« (b) émerge
de (a) + grande banque + τ bas ») suppose que les négatifs durs sont « même catégorie » ; ils sont
en réalité « même catégorie ET autre scène », ce qui reste résoluble par la scène. Le corpus ne
contient AUCUN négatif à scène appariée : c'est un trou de données, pas un trou de température.
- **Test à ~30 min (données déjà sur disque, aucune génération)** : construire la banque sur les
  cibles des shards switch2 existants, décoder les 16 échantillons held-out avec le décodeur v1.2,
  mesurer le top-1 retrieval crop60. **Si ≥ 80% : l'InfoNCE tel qu'écrit est déjà satisfait** et
  n'apportera pas le gradient manquant → il FAUT les négatifs même-prompt (F1.1b) avant tout run.
- Patch : F1.1b (même prompt/seed différent = même scène, autre instance) + logger `inst/acc`
  restreint aux négatifs même-prompt (le seul chiffre qui compte).

### F1.3 **[BIAISE]** Le masquage intra-vidéo jette les négatifs les PLUS pertinents
Le design masque « les entrées de la banque provenant de la MÊME vidéo ». Or pour une position
switch (cible = dernière frame de A), **les frames de B de la même vidéo sont de VRAIS négatifs**,
et ce sont exactement ceux de la pathologie (ressusciter B, CORRECTIF v1). Les masquer supprime la
seule pression anti-contamination gratuite du dispositif.
Patch : masquer uniquement le MÊME segment dans une fenêtre de ±k chunks autour de la cible ;
garder les frames de l'autre segment comme négatifs. Logger `inst/acc_vs_B` = fraction des ancres
plus proches de la cible A que de toute frame B de leur propre vidéo. **Gratuit, et c'est le
diagnostic le plus informatif de tout le run.**

### F1.4 **[BIAISE]** crop60 quand le sujet n'est pas centré
`center_crop(frac=0.6)` = 288×499 au centre de 480×832, en pixel. Si le sujet est décentré, ancre
ET négatifs sont dominés par le fond → retour au raccourci scène ; et comme la métrique d'éval
utilise le MÊME crop, l'entraînement hérite de l'angle mort de la métrique (Goodhart aligné).
Patchs : multi-crop (center60 + 2 crops aléatoires 50%) moyenné, ou moyenne des patch-tokens DINO
sur le crop plutôt que le CLS ; reporter full ET crop60 séparément (déjà prévu côté harness).
Sanity gratuite sur les latents déjà stockés : fraction de la masse de |frame − médiane temporelle|
tombant dans crop60 (proxy « sujet centré ? »).

### F1.5 **[BIAISE→TUE selon λ]** Banque FIXE + VAE différentiable = générateur d'exemples adverses
Le gradient d'un objectif DINO à travers un décodeur VAE gelé, vers un latent 16×60×104 LIBRE,
c'est la recette exacte d'un exemple adverse ; et comme la banque est figée, l'ancre n'a qu'à
battre 11k points fixes : elle peut viser une direction de faible densité au lieu de reproduire la
cible. Rappel décisif : la sonde `dose` a montré que le mécanisme tolère σ=0.6 de bruit (89% du
gain) ⇒ **la fidélité pixel n'est pas ce que le mécanisme consomme** ; l'ancre sert de latent de
conditionnement. Optimiser une perceptuelle pixel peut donc produire un latent joli-après-VAE et
hors-distribution comme conditionnement.
Patchs : garder le ratio de gradient contrast/reco ≤ 0.5 (prévu) ; tripwires OBLIGATOIRES
`anchor/latent_rms`, `anchor/rms_ratio_vs_target`, `anchor/kurtosis` ; **un embedder d'audit jamais
optimisé** (cf. F1.7) pour `inst/cos_audit` ; hard-negative mining (top-K par cosinus, rafraîchi
tous les N steps : 11k×384, coût nul) sinon la banque devient stale et le terme meurt tout seul.

### F1.6 **[BIAISE]** Les positions single sont TRIVIALES et enseignent la pathologie
Cible single = frame évincée **1 write plus tôt** = « décode ta dernière écriture » : zéro exigence
de mémoire, et c'est littéralement le biais de récence que le CORRECTIF v1 a identifié comme la
pathologie du revisit. Elles domineront `inst/acc` et masqueront l'échec sur les switch.
Patchs : (i) TOUT splitter par mode dans wandb ; (ii) remplacer la cible single par « frame évincée
il y a A writes », A ∈ {5,10,20} tiré au sort — **gratuit au replay** (tous les latents sont
stockés) et c'est aussi le signal de stockage LONG qui manque (cf. F3.2) ; (iii) plafonner la part
single à ≤20% de la loss.

### F1.7 **[COSM→BIAISE]** « embedder train ≠ embedder éval » : mitigation quasi vide
L'éval score en **DINOv2-vitb14** (run_reid_probe L132). Le train propose **DINOv2 ViT-S/14** :
même famille, même objectif, ViT-S distillé du même enseignant ⇒ représentations très corrélées.
Ce n'est pas une garde anti-Goodhart, c'est un changement de taille.
Patch : garder ViT-S pour la loss, mais **ajouter un embedder d'audit d'une autre famille**
(CLIP ViT-B/32 ou SigLIP) pour la métrique reportée, + l'œil d'Aden (règle maison) comme arbitre.

---

## FRONT 2 — Le curriculum (toy → écrivain seul / décodeur v1.2 gelé → joint)

### F2.1 **[TUE LA PHASE A — faux négatif quasi certain]** Le décodeur gelé est un jeu d'ADRESSES
`AnchorFrameDecoder` lit l'état par `swiglu_apply(q, w0, w1, w2)` avec **3×1560 requêtes apprises
(7.2M params)** : ce sont des **adresses dans l'espace de clés défini par `qkv` + `k_norm` de l1**.
Or θ_w **EST** `qkv`/`k_norm`. Une simple ROTATION de l'espace de clés (préservant parfaitement
l'information, et rendue peu coûteuse par l'orthogonalisation Muon) **détruit l'adressage**. Donc :
- l'optimiseur est confiné au minuscule sous-espace de changements qui préservent les adresses de
  l1 → phase A quasi inerte **même si l'écrivain apprend bien** (exactement le faux négatif visé
  par la question) ;
- pire, ce faux négatif est **indiscernable** du risque n°2 du design (« le décodeur est le
  plafond ») et du risque n°3 (« l'info n'est pas dans h ») : la phase A telle qu'écrite **ne peut
  pas diagnostiquer son propre échec**.
**Protocole correcteur SANS perdre le test causal** (c'est la question posée) :
1. **Dégeler l'ADRESSAGE seul** en phase A : `queries`, `pos_row/pos_col`, `q_norms` — et GELER le
   chemin de CONTENU (`fuse1`, `fuse2`, `out_norm`, `head`) **et la FiLM cond** (qui, elle, porte de
   l'information textuelle : la dégeler rouvrirait F1.1). Re-pointer des adresses ne peut pas
   inventer du détail d'instance ; ça ne fait que suivre l'espace de clés. Le test causal « le
   goulot est-il l'état ? » est préservé, l'artefact de base arbitraire disparaît.
2. **+ contrôle à budget apparié** (F2.2), qui est la vraie réponse rigoureuse.
Tripwires à ajouter : `dec/read_rms`, `dec/read_cos_vs_step0` (dérive de ce que le décodeur lit).

### F2.2 **[BIAISE LE VERDICT]** L'attribution `memdec20w` est structurellement injuste
`memdec20w` = écrivain 2.0 **+ décodeur v1.2** : c'est-à-dire un nouvel écrivain lu par un ancien
lecteur mal-adressé (F2.1). Ce bras est biaisé CONTRE l'écrivain, et c'est justement le bras censé
prouver la thèse. Idem pour le go/no-go « val ≥ +5% vs step 0 ».
**Protocole de remplacement (standard des probes de représentation, et peu cher) :** geler
l'écrivain ∈ {l1, 2.0}, **ré-entraîner un décodeur FRAIS** (même archi, même seed, mêmes steps,
mêmes données, warm-start v1.2 identique) sur les états de CHACUN, puis comparer val ET harness.
Coût : un entraînement décodeur par écrivain (pas de BPTT, ~heures). Ça élimine d'un coup le
confondant OOD (F2.1) et le confondant de co-adaptation de la phase B (F4.1).
Redéfinir : `memdec20w` := « écrivain 2.0 + décodeur re-fit sur états 2.0 » ; baseline := « écrivain
l1 + décodeur re-fit sur états l1 (memzero) ». **Insertion : J4 et J6.**

### F2.3 **[BIAISE]** Warm-start : ce doit être l'EMA, et il faut l'asserter
La collecte et v1.2 utilisent l1@2000-**EMA** (build_pipeline L210-225). Si θ_w est warm-starté sur
les poids bruts, le step 0 n'est pas le régime v1.2 et la « baseline plateau recalculée » est
fausse. Patch : charger via l'overlay EMA existant + assert `writer/delta_l1 == 0.0` au step 0 +
logger la provenance. (Le §4 dit d'ailleurs « 0.0 = suspect » pour le ckpt d'inférence : ici c'est
l'inverse, 0.0 est REQUIS au step 0.)

### F2.4 **[BIAISE]** Deux décalages de distribution empilés
memon→memzero (leur §1.2) ET l1-writer→2.0-writer. Le probe §1.2 doit tourner AVANT la phase A, et
le décodeur gelé doit être **re-fit quelques centaines de steps sur des états memzero** avant de
servir de base de lecture, sinon le step-0 de la phase A est déjà handicapé et le « +5% relatif »
mesure surtout la correction de ce handicap.

---

## FRONT 3 — Le gradient à travers les writes

### F3.1 **[LE TOY-OVERFIT EST UN GATE VIDE]**
2 vidéos, écrivain 21.3M + décodeur 33M : la chaîne n'a besoin d'encoder qu'**~1 bit** (quelle
vidéo) dans l'état, le décodeur peut stocker les deux frames cibles dans ses propres poids. **La
loss va s'écraser quoi qu'il arrive**, y compris si aucun stockage long n'existe. Il prouve le flux
de gradient — que `selftest_replay.py` test 2 prouve déjà (grads non nuls sur qkv/lrs/k_norm/decay).
**Critère de PASS redéfini (même coût, 0.5 GPU-h) :** ≥8 vidéos, cibles à **≥10 writes d'âge**, et
le toy ne passe que si :
(a) la loss s'écrase sur ces cibles âgées, **ET**
(b) le MÊME toy avec `n_bptt_writes=1` **NE** s'écrase **PAS** (sinon on n'apprend qu'un encodeur de
dernière écriture), **ET**
(c) un contrôle « h-shuffle » (permuter l'ordre de la séquence h) dégrade nettement la loss (sinon
l'ordre — donc la récurrence — ne sert à rien).

### F3.2 **[LE MODE D'ÉCHEC QUE LE TOY NE VERRA PAS]** « tricher court » par saturation
L'écrivain peut monter `softplus(lrs)` / l'échelle de `qkv` jusqu'à ce que **chaque write sature le
cap de norme par ligne** (`weight_norm_update` L207-209) : l'état devient une fonction des 1-3
derniers writes (oubli catastrophique), ce qui minimise les pertes de récence (majoritaires,
cf. F1.6) et **ressemble à de l'apprentissage**. `state/rowcap_frac` et `writer/decay` (prévus) le
suggèrent mais ne le prouvent pas.
**Tripwire proposé (sans gradient, coût négligeable) : `mem/influence_decay@k`.** Sur une vidéo de
val fixe, rejouer la chaîne puis la rejouer avec `h_{T−k}` remplacé par sa moyenne par token ;
reporter `||Δétat||/||état||` **et** Δloss décodée pour k ∈ {1,2,5,10,20,T}. Un écrivain
« tricheur court » a une courbe qui s'effondre après k=2 ; un écrivain qui stocke garde une
influence non triviale à k=20. C'est LE compteur qui distingue les deux hypothèses.
Compléments : `train/loss_by_age` et `val/loss_by_age` (buckets d'âge de cible) ; et un **run de
contrôle `n_bptt_writes=1` de 300 steps** : si la val égale le BPTT complet, l'écrivain n'est qu'un
encodeur de dernière écriture.

### F3.3 **[TUE LE RUN, SILENCIEUX, 1 LIGNE]** `AnchorFrameDecoder(detach_states=True)` par défaut
`video_memory.py` L671 et L755-756 : **par défaut le décodeur DÉTACHE w0/w1/w2**. Si le trainer
l'instancie avec les défauts (comme `train_decoder.py` le fait aujourd'hui, L287), **la loss
n'atteint JAMAIS l'écrivain** : la phase A n'entraîne rien (courbe plate, interprétée comme « le
décodeur gelé est le plafond » = risque n°2 du design !) et la phase B n'entraîne que le décodeur.
S1 ne couvre PAS ce chemin (son test 2 rétropropage une loss factice directement sur l'état).
Patch : `detach_states=False` **+ assert de démarrage** (un step, puis `qkv.weight.grad` non nul et
fini) à ajouter dans S6. Coût : 2 lignes. Sans ça, c'est 2 jours de GPU pour une courbe plate.

### F3.4 **[COSM→GÊNANT]** Contrat de dtype incompatible avec le plan de parité
`replay_writes` asserte `h.dtype == layer.qkv.weight.dtype` (L576). Le §3.5 prévoit « master fp32 +
cast explicite param→bf16 » : avec un module fp32 et un cast fonctionnel, l'assert saute (ou la
parité 1.4c est perdue). Patch : argument explicite `compute_dtype` / chemin `functional_call` en
bf16, assert contre ce dtype-là.

### F3.5 **[BIAISE]** Évanouissement par le decay, et son optimum dégénéré
`sigmoid(decay_logit) ≈ 0.95` ⇒ contribution d'un write à 25 pas : ×0.28 ; à 45 pas : ×0.10. Or
l'éval décode à ~49 writes de profondeur (F5.1). La réponse la moins chère de l'optimiseur est
decay→1, qui sature alors le cap sur toutes les lignes (retour à F3.2). Pré-enregistrer comme
condition d'arrêt : `writer/decay > 0.999` **et** `state/rowcap_frac → 1`.

---

## FRONT 4 — Généralisation

### F4.1 **[BIAISE]** La taille d'échantillon effective est ~135-270, pas ~5 000
Les positions d'une même vidéo partagent scène, sujet, seed et 90% de la trajectoire d'état : elles
ne sont PAS indépendantes. n_eff ≈ nombre de segments-vidéo (135-270) face à 21.3M (écrivain) +
33M (décodeur) = 54M paramètres entraînés en phase B. C'est une machine à mémoriser.
Patchs : (i) **split de val PAR VIDÉO, ≥20 vidéos** réservées (et le dire dans le format v3) ;
(ii) rendre le proximal `||θ_w − θ_l1||²` **obligatoire** en phase A (pas « en option si dérive
sauvage »), λ calé sur val ; (iii) early-stop sur la val-par-vidéo ; (iv) logger l'écart train−val
par mode.
**Constat à charge sur v1 (à ne pas rejouer et à ne pas citer dans le papier)** : `train_decoder.py`
+ `holdout_meta.txt` montrent que la val v1.0/v1.1/v1.2 = **16 échantillons = 8 vidéos × 2
positions**, et que la val de v1.0 (shards single) n'est **pas le même ensemble** que celle de
v1.1/v1.2 (shards switch). La comparaison « val 0.778 → 0.867 » est donc à la fois très bruitée et
inter-ensembles : elle ne supporte pas une conclusion.

### F4.2 **[BIAISE LE VERDICT — statistique]** Le harness sur-compte l'indépendance, et est sous-puissant
`sign_test_p` traite 24 cellules comme indépendantes ; ce sont **8 clusters (paires) × 3 seeds**.
Les p sont optimistes (sur moyennes par paire, n=8 : 7/8 ⇒ p≈0.07, 8/8 ⇒ p≈0.008). Et surtout :
distinguer +0.16 de +0.127 demande le contraste **apparié** `memdec20 − memdec` sur les mêmes
cellules ; avec 8 paires, seul un écart ≥ ~+0.05 est détectable.
Patchs ($0, recalculables sur les JSON existants) : reporter les moyennes par paire (n=8) + un
bootstrap apparié **clusterisé par paire** ; **re-pré-enregistrer la cible** comme
« delta apparié (memdec20 − memdec) ≥ +0.05 avec ≥7/8 paires positives », et non « ≥ +0.16 absolu ».
Ajouter 2 seeds (40 cellules) si l'effet attendu est petit.

### F4.3 **[BIAISE]** Le held-out n'exclut que par sous-chaîne exacte
`_hits_held_out` est un test de sous-chaîne : les prompts de même CATÉGORIE (horse, dog, cat, man,
woman, girl, parrot, bicycle) restent dans le corpus. Acceptable pour du re-ID d'instance, mais le
papier ne peut pas claimer de généralisation catégorielle. Patch gratuit : compter et logger les
prompts du corpus partageant la tête nominale d'une paire held-out, et **scinder le résultat du
harness en « catégorie vue / non vue »**.

### F4.4 **[BIAISE]** 8 paires held-out, c'est aussi la métrique qu'on optimise
Entraîner en espace DINO-crop60 et juger en DINO-crop60 sur 8 paires réutilisées à chaque campagne
= érosion progressive du held-out. Garder 2-3 paires **jamais utilisées** jusqu'au run final
(la règle « 8 paires FRAÎCHES à l'éval finale » du plan_final : l'appliquer réellement ici).

---

## FRONT 5 — Attaque libre

### F5.1 **[BIAISE→TUE]** Décalage de PROFONDEUR train/éval, jamais mentionné dans le design
Entraînement : vidéos 120f = 40 chunks ≈ **35 writes**, **UN** switch. Éval (défauts du harness) :
240 frames, switch1=81, switch2=162 ⇒ le décodage a lieu au chunk 54, soit **~49 writes** de
profondeur, après un schéma **A→B→A** (A ≈ 22 writes, B ≈ 22 writes). Une mémoire à capacité fixe
avec decay 0.95 et cap de norme est exactement le genre de système où 35 → 49 n'extrapole pas.
Patch peu cher (J2) : collecter **10-12 vidéos VAL-ONLY à 240 frames avec le schéma exact du probe**
(≈3.2GB/vidéo ⇒ ~35GB) et logger `val/loss_deep`. Si `val/loss_deep` se dégrade pendant que
`val/loss` s'améliore, le verdict du J7 est déjà connu et on économise la campagne.

### F5.2 **[TUE LE CHANTIER SI FAUX — et ce n'est PAS testé]** S1 ne teste pas l'invariance de h en θ_w
S1 (déjà VERT) prouve « replay ≡ in-situ **avec le MÊME écrivain** ». La prémisse porteuse est plus
forte : **h doit être invariant à θ_w**. Elle est vraie par lecture du code… *tant que le chemin de
lecture reste fini* : `causal_model.py` L653-658 implémente `MEM_ZERO_ALL` par
`mem_out = mem_out * 0.0` (une **multiplication**, pas un skip structurel). Un seul Inf/NaN dans le
read (saturation d'état → overflow bf16 dans `swiglu_apply`) donne `0*NaN = NaN`, qui contamine x,
les latents, **et donc h** — et un écrivain 2.0 est précisément un écrivain dont on a bougé
l'échelle d'écriture.
**Patch : selftest S1b, ~10 min.** Deux rollouts même seed/prompt sous MEM_ZERO_ALL=1, l'un avec
θ_w perturbé (qkv ×1.5, decay_logit +2) : asserter que les h capturés **et** les latents sont
**byte-identiques**. + `assert torch.isfinite(mem_out).all()` sous MEM_ZERO_ALL (ou passer au skip
structurel, comme le plan_final l'exigeait pour le sink virtuel). C'est le vrai gate fondateur du
chantier et il coûte 10 minutes.

### F5.3 **[LA BASELINE MANQUANTE QUI DÉCIDE DE TOUT — J2.5, quelques heures, aucun modèle vidéo]**
Puisque h est une DONNÉE et les cibles sont des DONNÉES, **l'information d'instance stockable par
n'importe quel écrivain est bornée par ce que contient le flux h** (inégalité de traitement de
l'information). Le design range ça en risque n°3 avec « seul le run 2.0 le dira » : **c'est faux,
ça se mesure avant**, hors BPTT :
- **(a) h-oracle** : entraîner un décodeur sur (h_i → cible_i), sans état ni récurrence.
  **Borne supérieure de tout le chantier.** Si l'ancre reste un cheval brun, mémoire 2.0 est morte
  telle que conçue et aucun entraînement d'écrivain ne la sauvera → on l'apprend au J2 au lieu du J7.
- **(b) compresseur trivial** : état := moyenne (ou EWMA) des h du segment A → même décodeur. Si ça
  égale l'écrivain TTT entraîné, la machinerie Muon/fast-weight est décorative — **un reviewer
  posera cette question**, autant avoir la courbe.
**Insertion : J2.5, gate de la phase A.**

### F5.4 **[TUE UNE JOURNÉE — OOM]** Le coût VRAM du terme d'instance est sous-estimé
« +2-4GB pour 6 VAE-decodes différentiables » est optimiste : un décode VAE **avec graphe** vers
480×832 matérialise des cartes de features de ~76MB pièce (480×832×96×2o) sur plusieurs niveaux ⇒
plusieurs GB **par ancre**. Le budget annoncé (16-22GB) est déjà proche de la règle maison
(marge >25GB sur 49GB). Le compte par-write (85MB/write ⇒ 6.4GB/échantillon ⇒ 12.8GB à B=2) est
juste, mais n'inclut ni le pic de recompute ni le décodeur ni l'InfoNCE.
Patch : **micro-benchmark du backward VAE au J1** (20 lignes, aucune donnée requise) ; défaut K=1
ancre avec gradient (pas 3) tant que ce n'est pas mesuré ; `torch.utils.checkpoint` sur les blocs du
VAE ; fallback latent prêt (déjà prévu).

### F5.5 **[OPS]** IO : 3GB/step sur un SSD à 98% partagé
~200-300 MB/s soutenus en lecture quasi-aléatoire sur un working set de 193GB non cacheable. Patch :
streaming shard-local (réutiliser le pattern `ShardStream`), et amortir en faisant 2-4 sous-steps
par vidéo chargée (en notant le coût de corrélation de batch). fp8 halve l'IO — mais F5.6/§2.4.

### F5.6 **[DÉCISION FACTUELLEMENT FAUSSE]** Le reclaim de 127GB n'existe pas
Mesuré aujourd'hui : `frame_dec_data` = **32GB** (pas 127GB), `frame_dec_data_switch` = 127GB,
`frame_dec_data_switch2` = 191GB ; **384GB libres** (pas 355GB). La décision n°3 du §6 doit être
corrigée : le seul reclaim significatif serait switch/switch2, **qui sont les données nécessaires aux
contrôles F2.2 et aux probes F5.3** → ne rien supprimer avant que F1.1a/F5.3 soient faits ; caler la
collecte à ≤190GB.

### F5.7 **[BIAISE]** Le cut n'est pas traité par 2.0 : co-primaire obligatoire
L'ancre est splicée en position de récence : le cut ×13.5 et le rewind du décor sont inchangés par
design (axe « rejoin » = v2). Une meilleure ancre peut monter le re-ID **et** dégrader la vidéo.
Le DD segmenté est prévu au J7 : en faire un **co-primaire avec un seuil no-worse pré-enregistré**,
sinon « victoire » borgne (et c'est exactement le triptyque Goodhart déjà documenté maison).

### F5.8 **[COSM, mais ça mord au J7]** Deux bugs du harness à corriger en même temps
`summarize()` L403-405 : la liste `all_arms` **ne contient pas** `memdec` (ni memdec3/10/12) → leurs
moyennes absolues ne sont jamais imprimées (seuls les deltas appariés L419 le sont). Et `MAG_ARMS`
liste `memdec10` alors que `ARM_ENV` n'a **pas** d'entrée `memdec10` → KeyError si on le demande.
Ajouter `memdec20`/`memdec20w`/`memdec_cond` **aux deux** listes et à `ARM_ENV`.

### F5.9 **[PAPIER]** Ce que 2.0 est réellement, et les ablations que ça impose
Sous MEM_ZERO_ALL=1, la « mémoire » n'est plus dans la boucle : 2.0 entraîne **un encodeur récurrent
offline de features gelées + un décodeur**, lu une fois au revisit. Les ablations que des reviewers
demanderont sont exactement F5.3(b) (compresseur trivial sur le même h), la **frame-bank**
(déjà identifiée dans plan_final, toujours pas implémentée) et Echo-Forcing. Le claim défendable est
« état O(1), adressé par contenu, synthétisant une ancre conditionnée au contexte » — et il exige
que le bras frame-bank tourne vraiment.

---

## SYNTHÈSE : ce qui bouge dans le planning

| Quand | Action | Coût | Bloque quoi |
|---|---|---|---|
| **J0** | **F1.1a** bras `memdec_cond` (état=init/shuffle) sur v1.2, 24 cellules | ~3 GPU-h + 10 lignes | La baseline et la cible du chantier entier |
| **J0** | **F1.2** retrieval top-1 du décodeur v1.2 contre la banque (données sur disque) | ~30 min | La pertinence même de l'InfoNCE |
| **J1** | **F5.2** selftest S1b (invariance de h à θ_w, θ_w perturbé) + assert isfinite | ~10 min | La prémisse fondatrice |
| **J1** | **F5.4** micro-benchmark backward VAE | ~20 min | K et B du terme d'instance |
| **J2** | **F1.1b** collecte : ~45 prompts × 2 seeds (négatifs même-prompt) | décision de collecte | La validité de la loss |
| **J2** | **F5.1** +10-12 vidéos VAL-ONLY 240f schéma probe (~35GB) | ~1 GPU-h | La détection du décalage de profondeur |
| **J2** | **F4.1** split de val PAR VIDÉO (≥20), inscrit dans le format v3 | 0 | Toute mesure de généralisation |
| **J2.5** | **F5.3** h-oracle + compresseur trivial (bornes sup. et baseline) | quelques h | GO/NO-GO de la phase A |
| **J3** | **F3.3** `detach_states=False` + assert de grad ; **F3.1** toy redéfini (8 vidéos, cibles âgées, contrôles n_bptt=1 et h-shuffle) | 0 / même coût | Le gate n°2 |
| **J3** | **F1.3/F1.6** masquage par segment, positions single vieillies, splits par mode ; **F3.2** `mem/influence_decay@k`, `loss_by_age` ; **F1.1.4** marge causale | 0 | L'interprétabilité de tout le run |
| **J4** | **F2.1** phase A = adressage du décodeur dégelé / contenu + FiLM gelés ; **F2.3** warm-start EMA asserté | 0 | Le faux négatif de la phase A |
| **J4/J6** | **F2.2** contrôles à budget apparié (décodeur frais par écrivain) | ~heures ×2 | L'attribution |
| **J7** | **F4.2** stats clusterisées par paire + cible ré-enregistrée en delta apparié ; **F5.7** DD co-primaire ; **F5.8** bugs harness ; bras `memdec_cond` re-joué | 0 | La lisibilité du verdict |

---

# RÉSOLUTION DU DÉBAT (1 tour, 2026-07-27) — ACTÉE

1. **F1.1 rétrogradée** : memdec10 (v1.0 sans conditionnement) +0.113 p=0.0015 prouve que le
   cœur v1 vient de l'état. F1.1 menace le signal d'entraînement 2.0 (InfoNCE + décodeur
   conditionné), pas le résultat v1. Bras `memdec_cond` = quantification du plancher texte
   (utile papier), plus un gate existentiel. RESTE DIE-ON : marge causale
   `acc(état vrai) − acc(état shuffle)` loggée dès step 0 + négatifs cond-non-discriminants.
2. **Collecte v3 GARDÉE + supplément jumeaux** : `--twin-offset 100000` implémenté dans
   collect_pairs.py (30 prompts single × 2 seeds, mêmes réglages/MEM_ZERO_ALL=1, out
   `mem2_h_v3_twins`, ~33GB). Conditions : jumeaux du MÊME côté du split train/val (clé
   corpus_idx), échantillonnage InfoNCE FORCE le jumeau dans chaque softmax (sinon noyé).
3. **Disque : clos** (artefact de chronologie). Ne pas supprimer switch/switch2 avant
   F5.3/F2.2 (contrôles à budget apparié).
4. **DIE-ON avant train** (au-delà des 3 gates F1.1a/F5.2/F5.3a) :
   F3.3 detach_states=False + assert grad qkv ; F3.1 toy redéfini (≥8 vidéos, cibles ≥10
   writes, PASS ssi n_bptt_writes=1 ÉCHOUE et h-shuffle dégrade) ; F1.1(4) marge causale
   step 0 ; F1.6 cibles single vieillies 5/10/20 + métriques par mode ; F4.1 val par vidéo
   ≥20 ; F2.3 warm-start EMA + assert delta_l1==0. Quasi-DIE-ON jour du train : F5.4 (mesure
   VRAM backward VAE avant de fixer K,B), F3.2 (influence_decay@k + loss_by_age).
