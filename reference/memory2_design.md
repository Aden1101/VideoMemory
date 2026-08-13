# MÉMOIRE 2.0 : PLAN D'EXÉCUTION COMPLET du write-replay (architecte, 2026-07-27)

Mandat : reference/memory2_brief.md. Code audité : wan/modules/video_memory.py,
wan/modules/causal_model.py (L586, L628-701, L1342-1385), reference/runs/frame_dec/
(collect_pairs.py, train_decoder.py, dump_anchor.py), pipeline/switch_causal_inference.py
(MEM_DECODE L406-443), reference/runs/reid_probe_l1/run_reid_probe.py (build_pipeline L158-222,
arm memdec L111 + L256), reference/runs/l1_longlive/l1_longlive.yaml, plan_final_sink_virtuel.md.

---

## 1. VÉRIFICATION DE L'INSIGHT CENTRAL contre le code : **VALIDÉ, avec 3 réserves chiffrables**

### 1.1 h est-il une DONNÉE sous MEM_ZERO_ALL=1 ? OUI, prouvé par lecture du code.
La chaîne exacte (causal_model.py) :
- `h = self.norm1(x_in)` (L586) est calculé AVANT la branche mémoire, depuis le flux
  attention seul.
- Le fifo de l'eviction-feed est rempli avec ce `h.detach()` (L640), poppé au niveau modèle
  (L1371-1374) et passé comme `mem_x_write` au write.
- L'UNIQUE rétroaction de la mémoire sur le flux est `x = x + mem_out` (L698) ; sous
  MEM_ZERO_ALL=1, `mem_out = mem_out * 0.0` (L653-658) sur les DEUX passes (read et write),
  et le garde-fou EMA (L679) est explicitement bypassé (`_mz_all == 0` requis).
- L'état écrit (`ttt_cache.state = new_state`, L701) ne nourrit RIEN d'autre que `mem_out`.
- Donc sous MEM_ZERO_ALL=1 : les latents du rollout == rollout memzero pur, et les h de
  chaque bloc [5,16,27] sont des fonctions du backbone GELÉ de (seed, prompt, cache) seuls.
  **h est indépendant de (qkv, lrs, decay_logit, w0/w1/w2) : c'est une donnée. Le
  write-replay est mathématiquement exact, pas une approximation.**
- Détail utile : le fifo n'est rempli QUE sur les passes write (`ttt_update=True`, L628),
  donc h = le flux norm1 de la passe write (timestep = context_noise). Aucune ambiguïté de
  denoising-step : le collecteur capture exactement ce tenseur, le replay le rejoue tel quel.

### 1.2 Le mismatch du collecteur v1 : réel, et la collecte 2.0 DOIT passer en MEM_ZERO_ALL=1.
- collect_pairs.py L299-301 : "memory injection LIVE = the state we snapshot is the one the
  trained model actually uses" ; `_OFF` remet MEM_ZERO_ALL=0. Le collecteur v1 tournait
  injection ACTIVE.
- MAIS l'arm d'éval memdec tourne MEM_ZERO_ALL=1 (run_reid_probe.py L111). Le selftest du
  harness prouve que MEM_ZERO_ALL change les latents (L358 : assert d > 0). Donc le décodeur
  v1 a été entraîné sur des états issus de rollouts memon et évalué sur des états issus de
  rollouts memzero : les h divergent dès le bloc 5, la divergence se compose sur 25-75
  writes. Ordre de grandeur attendu : FAIBLE (l1 memon vs memzero au probe re-ID = −0.02/
  −0.03, l'injection est quasi inerte), donc contributeur de second ordre au plateau +0.12,
  pas la cause principale : mais c'est gratuit à corriger et OBLIGATOIRE pour le replay.
- **Décision : collecte 2.0 = MEM_ZERO_ALL=1.** Double bénéfice : (a) h devient une donnée
  (condition sine qua non du replay, cf. 1.1 : injection active, h dépendrait de θ_write et
  le replay serait faux dès l'epoch 1) ; (b) alignement train == éval memdec (les états vus
  par le décodeur à l'inférence viennent de rollouts memzero, exactement comme au training).
- **Probe de quantification (J1, ~1 GPU-h, chiffre pour le papier)** : 20 vidéos switch
  collectées MEM_ZERO_ALL=1 (états seuls, format v2), passer le décodeur v1.2 GELÉ dessus :
  val-recon vs le 0.81 mesuré sur états memon. Le delta = le coût du mismatch v1.

### 1.3 Différentiabilité de la chaîne write en (qkv, lrs, decay_logit) : OUI telle qu'écrite.
Audit op par op (video_memory.py) :
- `qkv` Linear + `k_norm` RMSNorm (poids appris, dans le chemin write via k) : standard.
  `q_norm` ne touche que q, jeté par le write : hors périmètre θ_w.
- `lrs` : calculé en fp32 avec cast per-call des poids (L377-386) : le cast est
  différentiable, le softplus aussi.
- `swiglu_fast_weight_gradient` : gradient interne ÉCRIT À LA MAIN mais ce ne sont que des
  matmuls + silu + silu_backprop (élémentaire) : autograd le traverse sans état caché.
- `zeropower_via_newtonschulz5` : `X = G.bfloat16()` (L115) : cast différentiable ; le
  FIXME du code annonce 2-4% d'erreur RELATIVE de gradient (bf16 à travers checkpoint).
  Polynôme de degré 5^5 : différentiable, mais mal conditionné (cf. 1.4).
- `weight_norm_update` : les normes PRÉ-update sont détachées (L179-181 `w0.detach().norm`) :
  la cible de renorm ne porte pas de gradient, propre. Branche gated-delta (l1 a
  `use_decay: true`) : `decay = sigmoid(decay_logit)` différentiable ; `init_norms` détachées
  du paramètre w-init ; le cap `clamp(min=1.0)` a un gradient NUL quand la ligne est capée
  (point dur : cf. tripwire `state/rowcap_frac` en 3.6).
- `detach_state` : `_detach()` renvoie True en eval : **le replay doit instancier la cellule
  avec `detach_state=False`** (sinon le graphe est coupé à chaque write, silencieusement).
- `checkpoint_write` (L479-483) existe déjà : le checkpointing par write est fourni.

### 1.4 Points durs du BPTT à travers 15-39 writes successifs (le vrai risque technique) :
1. **Erreur de gradient NS bf16 composée** : 2-4% par write, ~25 writes : le gradient reste
   un estimateur biaisé/bruité. Probablement tolérable (LaCT/Scal3R entraînent à travers,
   e2_write3L l'a fait within-chunk), mais à vérifier par le toy-overfit J3 avant tout run.
2. **Mémoire d'activations** : sans checkpoint, >30GB/vidéo. Avec checkpoint par write :
   sauvés par write ≈ k+v (27MB) + états frontière (57MB) ≈ 84MB × B. À B=2, 25 writes,
   3 blocs ≈ 12.6GB : passe sur un GPU 49GB avec marge. B=4 (25GB) : seulement via
   checkpointing par GROUPES de 5 writes (frontières seules ≈ 3.4GB) ou
   `torch.autograd.graph.save_on_cpu`. **Plan : B=2 + checkpoint par write (le flag
   existant), grad-accum ×3 ; optimisation groupes seulement si besoin.**
3. **Jacobien du renorm/cap** : le renorm-to-init-norm projette la composante radiale du
   gradient ; le decay géométrique (sigmoid(decay_logit)≈0.95 init) atténue les
   contributions anciennes : c'est plutôt STABILISANT (pas de croissance exponentielle),
   le danger est l'ÉVANOUISSEMENT vers les writes anciens + le blocage par cap saturé.
   Mitigation prête : BPTT TRONQUÉ à T writes (stop-grad au-delà des T=8-12 derniers ;
   le forward est INCHANGÉ donc la parité bitwise avec l'inférence est préservée).
4. **Parité bitwise replay vs in-situ** : atteignable car aucun RNG dans le write, mêmes
   shapes/dtypes/kernels. Conditions STRICTES : (a) réutiliser `_qkv_lr` tel quel (le qkv
   est UN matmul fusionné 1536→4608 : le refaire en 3 matmuls séparés change l'arrondi) ;
   (b) module en bf16 comme à l'inférence, pas d'autocast fantôme ; (c) `lrs` : l'inférence
   fait `.float()` depuis des poids bf16 : le trainer doit caster fp32-master → bf16 → fp32
   (round-trip explicite), sinon la fonction d'entraînement ≠ la fonction déployée ;
   (d) même GPU-modèle pour le selftest. Export des ckpts writer en bf16 (le cast du master
   = les octets exacts que l'inférence utilisera).

**VERDICT insight : GO.** L'entraînement 100% offline de l'écrivain est exact par
construction sous MEM_ZERO_ALL=1 ; le seul inconnu est la QUALITÉ du gradient sur 25 writes,
gaté par un toy-overfit à 0.5 GPU-h avant d'engager quoi que ce soit.

---

## 2. COLLECTE h : dimensionnement, précision, format v3

### 2.1 État du disque (mesuré au design) : ATTENTION, plus serré que le brief.
`/shared/ssd_14T_project` : **355GB libres** (98% plein). Données v1 déjà posées :
frame_dec_data 127GB + frame_dec_data_switch 127GB + frame_dec_data_switch2 191GB = 445GB.
Le plafond 200GB pour la 2.0 tient dans les 355GB mais laisse 155GB de marge disque globale :
- garde-fou de projection après la 1re vidéo (porté du collecteur v1, L754-774) + `df` dans
  la commande de lancement ;
- **candidat de reclaim à proposer à Aden** (pas de suppression sans son GO) : frame_dec_data
  (v1.0 single, 127GB) est superseded par v1.1/v1.2 ; libérerait de quoi faire une 2e vague.

### 2.2 Arithmétique (F=3, law=12, sink=3 → wc=4, sc=1, writes aux chunks c≥5, fsl=1560) :
- h par write : (1, 4680, 1536) × 3 blocs = **43.1MB en 2 octets** (fp16/bf16).
- Vidéo single 90f (30 chunks) : 25 writes = **1.08GB** + latents complets fp16 (18MB) + cond.
- Vidéo switch 120f (40 chunks, switch ∈ {45,60,75} = chunks {15,20,25}) : 35 writes =
  **1.51GB** + latents (24MB). Positions switch supervisées (A totalement évincé,
  c ≥ switch_chunk+5, la règle min_gap du v1) : 10 à 20 par vidéo selon le switch.
- **Une vidéo switch sert TRIPLE** : (a) paires switch (cible = dernière frame de A, cond =
  A, états post-écritures-B : la tâche exacte du revisit) ; (b) paires single pré-switch
  (chunks 5..switch-1, cible = frame évincée, cond = A) ; (c) paires single post-switch
  (cond = B). D'où un mix 85/15 switch/single (le single pur garde une population de
  contrôle longue-cohérence).

### 2.3 Budget retenu (plafond 200GB) :
- **Baseline fp16 : 110 vidéos switch 120f (166GB) + 25 vidéos single 90f (27GB) = 193GB,
  135 vidéos, ~4 500 writes stockés, ~1 700 positions switch + ~3 500 positions single.**
- Si probe fp8 GO (2.4) : h en fp8, latents/cond restent fp16 → ~0.78GB/vidéo switch →
  **~250-270 vidéos sous le même plafond** (la diversité, le point faible v1.2).
- Sous-échantillonnage de tokens : NON (brief : change la sémantique du write). Confirmé :
  le write consomme les 4680 tokens du chunk, tronquer = un autre opérateur.

### 2.4 fp8 : l'idée, les risques, le protocole de test (J1, AVANT la collecte).
Stockage e4m3 (dequant → bf16 au compute ; on ne calcule JAMAIS en fp8) :
- Risque 1 : précision e4m3 ≈ 2^-3..2^-4 relatif ; h est RMS-normé (norm1), O(1), pas de
  problème de range, mais le bruit entre dans k, v ET lrs, et se compose sur 25-39 writes.
- Risque 2, LE vrai : **mismatch train/inférence** : à l'inférence memdec2.0 les writes
  voient des h bf16 LIVE, pas quantifiés. Un écrivain entraîné sur h-fp8 subit un shift de
  distribution exactement sur son tenseur d'entrée. (La tolérance énorme prouvée côté
  ancres ne se transfère pas automatiquement : elle a été mesurée sur l'ENTRÉE du
  débruiteur, pas sur l'entrée d'une chaîne d'écriture récurrente.)
- Risque 3 : sous-normaux/zéros dans les queues de h (compteur au moment de la quantif).
- **Protocole (0 collecte engagée avant le verdict)** : 4 vidéos collectées h-bf16 ;
  replay du writer l1 GELÉ sur h-bf16 vs h-fp8-roundtrip ; comparer : erreur relative
  d'état final, cos par ligne (seuil pré-enregistré > 0.995), delta de décodage v1.2
  (MSE latente << variance inter-positions) + œil sur les 2 ancres VAE-décodées.
  GO → fp8 (corpus ×2) ; DOUTE → fp16, point final.

### 2.5 Format de shard v3 (extension du v2, PAR VIDÉO et non par paire) :
```python
{
  "format_version": 3, "kind": "h_replay",
  "memory_layers": [5, 16, 27],
  "h_dtype": "float16" | "float8_e4m3fn", "h_scale": {...} si fp8 (échelles par tenseur),
  "mem_zero_all": True,          # ASSERTÉ True par le trainer (refus de shards memon)
  "writer_ckpt": {"lora_ckpt": ..., "use_ema": True},   # provenance l1@2000-EMA
  "cond_dim": 4096, "cond_pooling": <idem v2, masked mean>,
  "videos": [{
    "h_seq": {"5": [T(1,4680,1536), ...], "16": [...], "27": [...]},
        # h_seq[b][i] = le mem_x_write consommé par la passe write du chunk write_chunks[i]
    "write_chunks": [5, 6, ..., 39],   # ordre du replay, exhaustif (TOUS les writes)
    "evicted_chunks": [1, ..., 35],    # = write_chunks - wc (bookkeeping)
    "latents": T(F_total, 16, 60, 104) fp16,   # le rollout ENTIER : toutes cibles possibles
    "cond_emb_a": T(4096,) fp16, "cond_emb_b": T(4096,) fp16 | None,
    "prompt", "prompt_b", "corpus_idx", "corpus_idx_b", "seed", "mode",
    "switch_frame", "switch_chunk", "window_chunks", "sink_chunks",
    "check_states": [(chunk, (w0,w1,w2) bf16), ...] | None,
        # 4 premières vidéos du shard 0 seulement : cross-check bitwise trainer (S5)
  }, ...],
  "meta": {config, ckpt, frames, created, ...}   # idem v1/v2
}
```
Sharding : ~8 vidéos/part (~12GB), flush atomique + manifest + resume, tout le pattern v1
réutilisé tel quel. Différence clé vs v1/v2 : **on ne stocke PLUS les états** (le replay les
recompute) ; on stocke la donnée amont (h) + toutes les cibles → chaque vidéo fournit des
dizaines de points de loss au lieu de 8 états figés.

### 2.6 Collecteur : `collect_h.py` (extension de collect_pairs.py, additif).
- Même boucle de rollout, mêmes seeds/formules (stream 500000/700000 + corpus_idx), même
  sélection de corpus (is_reidentifiable + exclusion held-out, grep-selftest intégré),
  corpus_seed FRAIS (diversité vs v1), MÊME held-out (les 16 prompts probe exclus).
- `_OFF` remplacé par `_OFF + MEM_ZERO_ALL=1` (le point 1.2).
- Capture de h SANS toucher au code coeur : juste AVANT la passe write du chunk c, peek
  `ttt_cache[bi].fifo[0]` (l'élément qui va être poppé) ; assert son chunk_idx == c-wc ;
  `.to("cpu", fp16)`. C'est exactement le tenseur que le modèle passera en mem_x_write
  (pop pré-append, L1371). Vérif croisée par MEM_EVICT_DEBUG=1 sur la 1re vidéo.
- Assert de range fp16 : max|h| < 65504 et comptage des sous-normaux (<6.1e-5) ; le
  round-trip bf16→fp16→bf16 est EXACT pour les valeurs in-range (fp16 a 10 bits de
  mantisse > 7 bits bf16) : selftest S4.

---

## 3. SCHÉMA D'ENTRAÎNEMENT

### 3.1 Nouveau code (additif) :
- `VideoMemoryLayer.write_only(x_write, state)` dans video_memory.py : factorise
  EXACTEMENT le chemin write de step() (appel `_qkv_lr(x_write)` fusionné, q calculé et
  jeté comme dans step() L472, puis `_write` avec decay/init_norms) ; flag
  checkpoint par write réutilisé. Aucune modif de step() : selftest byte-identical trivial.
- `reference/runs/frame_dec/train_writer.py` (extension de train_decoder.py : réutilise
  norm_mse/subject_weight, ShardStream adapté aux shards v3, VAE viz, wandb, ckpt).
- Cellule replay : instanciée avec les kwargs EXACTS de l1_longlive.yaml (use_decay=true,
  muon_update_steps=5, num_heads=1, decay_init, etc.), `detach_state=False`,
  chargée depuis `generator_memory` du lora ckpt l1@2000-EMA (la logique de
  run_reid_probe.build_pipeline L193-202 + overlay EMA L210-222, réutilisée).

### 3.2 Paramètres entraînés (θ_w, par bloc ×3) :
- Phase A/B : `qkv` (7.08M), `k_norm.weight`, `lrs` (4.6K), `decay_logit` (1) ≈ **21.3M**.
- `w0/w1/w2` init (28.3M/bloc) : GELÉS en phase A/B (c'est le prior d'état l1 ; les
  init_norms du cap en dépendent, détachées donc sans piège). Dégel = option phase C.
- `q_norm`, `proj`, `o_norm`, etc. (chemin read) : hors périmètre, l'injection est morte.
- **Warm-start θ_w = valeurs l1** (jamais random : les états l1 portent déjà le +0.13).

### 3.3 Curriculum (répond à la question ouverte du brief) :
- **Phase 0 (gate)** : toy-overfit 2 vidéos, 500 steps : le gradient traverse-t-il la
  chaîne ? (verdict risque n°1, 0.5 GPU-h).
- **Phase A : ÉCRIVAIN SEUL, décodeur v1.2 GELÉ (warm-start, PAS from-scratch).**
  Argument : le décodeur gelé définit la base de lecture ; tout gain de val = de
  l'information NOUVELLE stockée dans le format déjà lisible = le test causal propre de la
  thèse du brief ("le goulot est l'état, pas le lecteur"). Baseline automatique = val à
  step 0 (writer l1 + décodeur v1.2) : c'est le plateau v1.2 recalculé dans le cadre replay.
  ~2 500 steps. Go/no-go pré-enregistré : val switch (subject-weighted) doit battre le
  step-0 de ≥ 5% relatif, sinon le décodeur gelé est le plafond → phase B directe.
  Le terme d'instance (3.4bis) est ACTIF dès la phase A (rampe λ steps 300-600) : c'est
  lui qui interdit à l'écrivain la moyenne de catégorie, le verdict discriminant l'exige.
- **Phase B : JOINT** : dégel du décodeur, lr writer 1e-4 / décodeur 1e-4, EMA des deux
  pour l'éval. C'est ici qu'on prend le "extra mile" de co-adaptation (le code privé
  writer↔decoder est acceptable : seule la frame décodée compte au splice).
- Phase C (si plateau, options ordonnées) : dégel w-inits ; écriture à deux clés FiLM
  (= 2.1, pas 2.0, conformément au brief) ; élargissement de corpus (fp8 ou 2e vague) ;
  suréchantillonnage des négatifs même-catégorie dans la banque (3.4bis).

### 3.4 Loss :
- Principale : `norm_mse(dec(state_c, cond), target_c, subject_weight=3)` (la loss v1.2),
  moyennée sur K positions échantillonnées par vidéo et par passage : 3 positions switch
  (cible = dernière frame de A, cond = A) + 1 position single (cible = frame évincée du
  chunk c-wc, cond = prompt courant) + toujours la DERNIÈRE position éligible.
  Les positions sont GRATUITES au replay (tous les états intermédiaires existent) : c'est
  l'avantage structurel n°1 sur v1 (8 états figés/vidéo).
- **Terme d'instance : OBLIGATOIRE** (verdict du test discriminant, 2026-07-27 : le cheval
  single-prompt, état 100% A sans aucun B, donne ENCORE une ancre brune → l'échec est une
  RÉGRESSION DE CATÉGORIE, pas une contamination de B ; la reconstruction pondérée seule
  laisse l'écrivain s'en tirer avec la moyenne de catégorie). Dimensionnement en 3.4bis.
- Régularisation : `||θ_w − θ_l1||²` légère (1e-4) EN OPTION si le toy-overfit montre de
  la dérive sauvage ; sinon rien (le warm-start + lr modeste suffisent en général).

### 3.4bis Le terme d'instance : arbitrage, banque de négatifs, coût, curriculum.
**Arbitrage des 3 candidats :**
- (c) perceptuel DINO simple (cosine à la cible) : c'est une loss de reconstruction
  perceptuelle : elle a le MÊME mode d'échec que la MSE (la moyenne de catégorie est
  proche aussi en DINO, le gradient vers le détail discriminant reste faible). Rejeté
  comme terme unique ; mais il est INCLUS gratuitement comme numérateur de l'InfoNCE.
- (a) contrastif inter-vidéos in-batch : la bonne PRESSION (matcher SA cible mieux que
  celles des autres), mais à B=2 vidéos il y a ~1 négatif par step : mort-né en in-batch
  pur. → il faut la banque (ci-dessous).
- (b) contrastif restreint même-catégorie : c'est la pression idéale (cheval vs cheval
  force la robe), mais le tagging explicite est fragile et le gating par tags jette des
  négatifs utiles. **Observation clé : (b) émerge de (a) + GRANDE banque + température
  basse** : dans un softmax à ~5-10k négatifs, seuls les plus durs (= même catégorie)
  portent le gradient ; les négatifs cross-catégorie sont résolus dès le step 1 et
  s'éteignent d'eux-mêmes. Pas besoin de tags pour la loss.
**Choix recommandé : (a) InfoNCE en espace DINO-crop60 contre une banque PRÉCALCULÉE de
toutes les cibles du corpus, température basse (le mécanisme de (b) sans tagging),
numérateur = cosine DINO à sa propre cible (= (c) inclus).**
- **La banque (la solution au problème des petits batches)** : les négatifs sont des
  CIBLES = des DONNÉES fixes (frames latentes du corpus) → leurs embeddings se précalculent
  UNE FOIS, hors gradient (MoCo sans momentum : rien ne bouge). Script `embed_bank.py`
  (J2, post-collecte) : pour chaque frame-cible éligible du corpus (~40/vidéo × 135-270
  vidéos ≈ 5.4-11k), VAE-decode → crop60 central pixel → resize 224 → DINOv2 ViT-S/14 →
  L2-norm → banque (11k × 384 fp32 ≈ 17MB, résidente GPU). Coût one-shot ≈ 10-15 min GPU.
- **La loss au step** : aux K=3 positions switch (là où l'instance se joue) : ancre =
  frame décodée → VAE-decode DIFFÉRENTIABLE (VAE gelé, le gradient traverse jusqu'à
  l'écrivain) → crop60 → DINO (gelé) → z_i ; InfoNCE : −log exp(z_i·z_tgt/τ) /
  (exp(z_i·z_tgt/τ) + Σ_bank exp(z_i·z_neg/τ)), τ = 0.07 ; les entrées de la banque
  provenant de la MÊME vidéo sont masquées (pas de faux négatifs intra-vidéo).
- **Coût** : 6 ancres/step (K=3 × B=2) : 6 VAE-decodes 1-frame différentiables + 6 DINO
  ViT-S. Estimation : +1-2 s/step et +2-4GB de VRAM (activations du décodeur VAE ;
  mitigations si besoin : checkpoint sur les blocs du VAE, ou n'appliquer le terme qu'à
  2 ancres tirées au sort par step). Budget VRAM révisé : ~16-22GB, toujours sous la
  marge >25GB. Softmax banque : négligeable (11k × 384).
- **Pondération et curriculum** : warm-up reconstruction D'ABORD : λ_inst rampé 0 → cible
  sur les steps 300-600 de la phase A (le signal de reco est dense et stable, le
  contrastif est plus bruité ; à step 0 les décodes warm-start sont déjà sémantiques donc
  la rampe peut être courte). Cible de λ calibrée comme λ_RSD maison : ratio de normes de
  gradient contrast/reco ∈ [0.3, 0.5] mesuré à la fin de la rampe ; point de départ
  λ = 0.3. Actif dans les DEUX phases (A et B).
- **Garde-fou Goodhart (obligatoire au papier)** : la métrique re-ID du harness est une
  similarité DINO : entraîner en espace DINO entraîne partiellement la métrique d'éval.
  Mitigations : (i) embedder de TRAIN ≠ embedder d'ÉVAL (ViT-S au train si le probe score
  en ViT-B/L : à vérifier au moment de l'implémentation, et le documenter) ; (ii) arbitre
  final = paires held-out + l'œil d'Aden (le cheval PÂLE) ; (iii) reporter full ET crop60.
- **Fallback économique** (si le backward VAE coûte trop au toy-run) : même InfoNCE en
  espace LATENT crop60 (RMS-normé) : plus faible (le layout de scène domine, négatifs trop
  faciles), documenté comme dégradé ; flag `--inst_space {dino,latent}`.
- **Tags de catégorie** : PAS dans la loss ; extraction grossière du nom-tête (la regex
  de is_reidentifiable existe déjà) utilisée uniquement pour (i) le diagnostic wandb
  `inst/acc_same_cat` (précision contrastive restreinte aux négatifs même-catégorie : LE
  compteur qui dit si la robe du cheval rentre) et (ii) option 2.0.1 : suréchantillonner
  les négatifs même-catégorie dans la banque si l'acc sature.

### 3.5 Optim / budget :
- AdamW β=(0.9, 0.95) ; wd 0.01 décodeur, **wd 0 sur θ_w** (le weight decay sur qkv
  dériverait l'échelle d'écriture sans signal) ; lr writer 1e-4 cosine → 5e-6, warmup 200 ;
  clip 1.0 (normes loggées séparément writer/décodeur).
- Compute : ~1.6 TFLOP/write/bloc (qkv 66G + gradient interne ~530G + NS ~975G) ;
  vidéo switch ≈ 0.17 PFLOP forward, ~0.7 PFLOP avec backward + recompute. B=2 + accum 3 :
  ~10-14 s/step effectif + 1-2 s pour le terme d'instance (6 VAE-decodes différentiables,
  3.4bis) sur un GPU 49GB ; VRAM ~16-22GB (règle marge >25GB respectée, co-run GPU 0
  possible). Phase A 2 500 steps ≈ 9-12h ; phase B idem.
  **Total ~2-3 GPU-jours, 1 seul GPU.** IO : ~3GB/step, prefetch thread + page cache, OK.
- Précision : master fp32, forward avec cast explicite param→bf16 (parité déploiement,
  cf. 1.4d), h dequant → bf16, loss fp32.

### 3.6 Métriques wandb (projet anchor-forcing-memory, run mem2_writer_<phase>) :
train/loss, train/loss_switch, train/loss_single, inst/loss + inst/acc (top-1 contre la
banque) + **inst/acc_same_cat** (négatifs même-catégorie seuls : le compteur du détail
discriminant) + inst/grad_ratio (calibration de λ),
val/loss + val/loss_switch (vidéos held-out REJOUÉES, jamais vues), writer/grad_norm,
dec/grad_norm, writer/delta_l1 (||θ−θ_l1||/||θ_l1||), writer/lr_softplus_mean,
writer/decay (sigmoid(decay_logit)), state/rms, **state/rowcap_frac** (fraction de lignes
saturées au cap : le tripwire du gradient bloqué, cf. 1.3), profil de gradient par
profondeur de write (t=1/mi/dernier, tous les 200 steps : le tripwire de l'évanouissement),
viz/target_vs_recon (grille VAE, comme v1), samples/s, epoch.

### 3.7 Selftests OBLIGATOIRES (tous verts avant tout run long) :
- **S1 REPLAY-BITWISE (LE test qui valide l'approche)** : rollout court in-situ (45f,
  MEM_ZERO_ALL=1) en capturant h (bf16, en RAM) + l'état après CHAQUE write ; replay
  offline `write_only` avec les params l1 en bf16 ; `torch.equal` à chaque write, 3 blocs.
  Même GPU-modèle, pas d'autocast, qkv fusionné. ÉCHEC = stop chantier, débug avant tout.
- S2 parité collecteur : latents du rollout collecteur == latents de l'arm memzero du
  harness, byte-identical, même (prompt, seed). (Le v1 matchait memon ; le 2.0 doit
  matcher memzero : c'est la preuve de l'alignement train/éval.)
- S3 held-out grep : 0 hit des 16 prompts probe dans le corpus sélectionné (porté de v1).
- S4 round-trip fp16 : h_bf16 → fp16 → bf16 identité exacte + asserts range/sous-normaux.
- S5 trainer vs collecteur : replay des 4 vidéos à check_states → états bitwise identiques
  aux états snapshotés in-situ par le collecteur.
- S6 inertie des flags : subject_weight=1 et λ_inst=0 ⇒ loss == norm_mse nue,
  byte-identical (le terme d'instance est obligatoire au RUN mais son flag doit être
  inerte à zéro pour l'ablation) ; hash de la banque loggé ; chemin ancre→embed
  déterministe ; masquage intra-vidéo asserté (0 faux négatif de sa propre vidéo).
- S7 MEM_WRITER identité (cf. 4) ; S8 py_compile + wandb live (règles maison).

---

## 4. INFÉRENCE 2.0 : le hook MEM_WRITER

- **Sémantique** : `MEM_WRITER=/path/writer_step*.pt` → charger les params d'écriture 2.0
  dans les cellules AVANT le rollout (les writes commencent au chunk 5 : contrairement à
  MEM_DECODE qui n'agit qu'au revisit, le writer doit être en place au BUILD).
- **Où** : `run_reid_probe.py::build_pipeline`, juste APRÈS l'overlay EMA (L210-222), donc
  ordre de chargement : base → LoRA → generator_memory → EMA → **MEM_WRITER par-dessus**.
  Zéro code coeur touché (le harness est le seul consommateur memdec) ; si un autre
  pipeline en a besoin plus tard, le même bloc env-gated se copie tel quel.
- **Format ckpt** (écrit par train_writer.py) : `{"writer": {"5": sd, "16": sd, "27": sd},
  "step": int, "args": {...}}` avec sd ⊂ {qkv.*, lrs.*, k_norm.weight, decay_logit
  [, w0, w1, w2 si phase C]}, **poids en bf16** (les octets exacts du forward
  d'entraînement, cf. 1.4d).
- **Comment** : pour chaque bi : `blocks[bi].memory.load_state_dict(sd, strict=False)` ;
  assert unexpected == [] ; assert que les clés chargées ⊆ l'ensemble attendu ; print de
  `||θ_new − θ_l1||/||θ_l1||` par bloc (0.0 = ckpt identité, suspect ; log obligatoire).
- **Plomberie harness** : `cfg._memwriter_ckpt` injecté en env dans run_arm (miroir exact
  de `_memdec_ckpt`, L256) ; nouveaux arms :
  - `memdec20`  = MEM_ZERO_ALL=1 + MEM_WRITER(2.0) + MEM_DECODE(décodeur 2.0),
  - `memdec20w` = MEM_ZERO_ALL=1 + MEM_WRITER(2.0) + MEM_DECODE(v1.2 gelé) : **l'arm
    d'attribution** (la part du gain qui vient de l'écrivain seul).
- **Selftests** : S7 = arm memdec avec MEM_WRITER pointant un ckpt "identité" (les params
  l1 repackagés au format writer) → latents byte-identical à memdec sans MEM_WRITER.
  Régression 2.0 : latents PRÉ-splice de memdec20 == memzero byte-identical (structurel :
  l'état ne rétroagit pas sous MEM_ZERO_ALL=1, mais on le vérifie quand même, règle brief).

---

## 5. RISQUES CLASSÉS + détection précoce, et PLANNING

### Risques (ordre décroissant) :
1. **Pathologie du gradient sur 15-39 writes chaînés** (erreur NS bf16 composée,
   évanouissement à travers renorm/decay, blocage par cap saturé) → l'écrivain n'apprend
   pas ou dérive. Détection J3 : toy-overfit 2 vidéos (500 steps, la loss DOIT s'écraser) +
   profil de gradient par profondeur + rowcap_frac. Mitigations prêtes : BPTT tronqué
   (forward inchangé = parité préservée), lr réduit, groupes de checkpoint.
2. **Plafond du décodeur gelé** (phase A inerte : la base de lecture v1.2 ne peut pas
   exprimer le détail d'instance quel que soit l'état). Détection : val phase A plate dès
   ~800 steps. Réponse : phase B jointe immédiate ; l'attribution se fait alors via l'arm
   memdec20w a posteriori.
3. **Le vrai mur** : l'info d'instance n'est PAS dans les h des blocs [5,16,27] (le flux
   norm1 du backbone gelé la sous-représente). Le test discriminant (2026-07-27) a tranché
   le mode d'échec v1 (régression de catégorie sur état pur-A, pas de contamination B) :
   compatible avec la thèse "l'écrivain l1 stocke la catégorie", mais il ne PROUVE pas
   encore que h contient l'instance : seul le run 2.0 avec terme d'instance le dira.
   Détection : inst/acc_same_cat qui ne décolle pas en phase A/B, puis phase B qui
   replafonne à ~+0.12 au harness. Réponse : narratif papier "co-entraîner l'écrivain ne
   suffit pas à backbone gelé", leviers suivants = blocs supplémentaires ou co-train
   backbone (hors 2.0).
4. **fp8 : shift train/inférence** (h quantifié au train, bf16 live à l'inférence).
   Détection : probe J1 (protocole 2.4), AVANT d'engager la collecte. Défaut = fp16.
5. **Famine de diversité** (135 vidéos fp16). Mitigé par la supervision multi-positions ;
   surveillance du gap train/val ; leviers : fp8 GO (×2), reclaim 127GB v1.0 (GO Aden).
6. **Débordement disque** (355GB libres, on en pose 200) : garde-fou de projection +
   df dans la commande + plafond dur.
7. **Goodhart sur le terme d'instance DINO** (la métrique re-ID du harness est
   DINO-based : on entraîne un cousin de la métrique d'éval). Détection : divergence
   entre inst/acc qui monte et l'œil humain qui ne voit pas le cheval pâlir. Mitigations
   pré-enregistrées : embedder train ≠ embedder éval, paires held-out, verdict œil
   obligatoire avant toute déclaration de victoire (règle maison).
8. **Overfit/co-adaptation phase B** : val sur vidéos held-out rejouées + arbitre final =
   harness 24 cellules sur les paires held-out ; EMA à l'éval.
9. Ops habituels : GPUs volatils (pré-check DANS la commande, marge >25GB, 4 GPUs max),
   wandb obligatoire, mem_breakdown:false (non applicable ici mais règle rappelée).

### Planning (jours, séquencé sur les gates) :
- **J1** : `write_only` + **S1 REPLAY-BITWISE** (gate n°1 de tout le chantier) ; probe fp8
  (2.4) ; probe quantification mismatch-v1 (1.2, chiffre papier). [1 GPU, quelques heures]
- **J2** : `collect_h.py` v3 + S2/S3/S4 ; lancement collecte 135-270 vidéos
  (90-120f ≈ 2-4 min/vidéo → 7-9 GPU·h : 2-4 GPUs co-run, fini dans la journée) ;
  `embed_bank.py` en post-pass (banque DINO des cibles, ~15 min, 3.4bis).
- **J3** : `train_writer.py` + S5/S6 ; **toy-overfit** (gate n°2 : le verdict risque n°1),
  avec mesure du surcoût VAE-backward du terme d'instance (bascule fallback latent si
  prohibitif).
- **J4-J5** : Phase A (écrivain seul, décodeur v1.2 gelé, terme d'instance rampé dès
  step 300), ~2 500 steps, val held-out ; go/no-go décodeur gelé.
- **J5-J6** : Phase B jointe (mêmes losses, décodeur dégelé).
- **J6** : hook MEM_WRITER + S7 ; arms memdec20/memdec20w.
- **J7** : harness 24 cellules : memdec20, memdec20w, memdec (v1.2, re-run mêmes seeds),
  memzero, oracle1, stockoff ; DD segmenté ; côte-à-côte pour l'œil d'Aden (cheval PÂLE ?).
  Verdict vs cible pré-enregistrée : **memdec20 ≥ +0.16** (battre le plateau +0.12/+0.147),
  et memdec20w > memdec (l'écrivain seul apporte du gain = la thèse du brief).
**Total : ~7 jours jusqu'au verdict, ~3-4 GPU-jours cumulés.**

---

## 6. Décisions demandées à Aden (bloquantes ou à fenêtre)
1. GO sur MEM_ZERO_ALL=1 pour la collecte 2.0 (recommandé OUI, argument 1.2 : requis par
   l'insight ET aligne train/éval).
2. fp16 vs fp8 : décision après le probe J1 (recommandation par défaut : fp16 ; fp8
   seulement si le probe est net).
3. Reclaim éventuel de frame_dec_data v1.0 (127GB) pour une 2e vague de collecte.
4. Corpus_seed frais (diversité) vs reprise des prompts v1 (comparabilité) : recommandé
   FRAIS, la comparabilité passe par le harness held-out, pas par le corpus de train.
5. Rappel règle maison : tout le code probes/frame_dec est NON-COMMITÉ (commit Aden).
