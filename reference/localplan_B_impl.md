# Plan B : implémentation fichier-par-fichier du sink-virtuel (volet (b) de l'ultraplan)

Rédigé le 2026-07-23, après lecture intégrale de : `wan/modules/video_memory.py`,
`wan/modules/causal_model.py`, `utils/wan_wrapper.py`, `pipeline/streaming_training.py`,
`pipeline/streaming_switch_training.py`, `pipeline/switch_causal_inference.py`,
`model/streaming_training.py`, `model/dmd_switch.py`, `trainer/distillation.py`,
`utils/distributed.py` (EMA_FSDP), `wan/modules/attention.py`, configs `ttt_memory.yaml` /
`probe_l1.yaml`, et `reference/runs/reid_probe_l1/results/summary.json`.

Données nouvelles intégrées :
* **Le plancher de compression est tombé** : oracle1 (UNE frame latente de A dans la
  fenêtre) = +0.224/+0.241 ≈ oracle3 ≈ oracle full (+0.220/+0.240), signes 24+/0−.
  Conséquence : le sink virtuel peut viser **m ≤ 1560 tokens par bloc mémoire** (1 frame
  latente), et on testera moins (m ∈ {390, 780, 1560}). Capacité multi-sujet plausible :
  1 frame/sujet ⇒ 3-4 sujets ≈ 3-4 frames d'info dans l'état.
* **Extension MULTI-SWITCH (décision Aden 2026-07-23, brief §2)** : la cible est
  A→B→C→A→D... (usage natif LongLive), pas seulement A→B→A. Impacts intégrés ci-dessous :
  curriculum à schedules de revisit variés (étape 3), teacher oracle à registre de queues
  par segment (étape 4), **écriture/lecture à deux clés FiLM par prompt-embedding**
  (étape 6, passe d'optionnel à probablement nécessaire : l'écriture de B/C/D ne doit pas
  écraser la trace de A), driver d'éval N-switch (étape 7), decay appris exposé/loggé en
  config (étapes 0 et 8, interludes de plusieurs minutes à tolérer).

## 0. TL;DR des décisions d'architecture

1. **Insertion cache : concaténation au moment de l'attention** (tenseurs séparés `mem_kv`
   passés en kwarg), PAS de slots réservés dans `kv_cache`. Précédent maison : le
   `prompt_cache` fait exactement ça (`k_cat = [k_sink, k_prompt_cache, k_local]`,
   causal_model.py:398). Neutralisation : flag OFF (m=0) = codepath intact = byte-identical ;
   `MEM_ZERO_ALL=1` = `mem_kv=None` = ablation EXACTE (pas de concat du tout).
2. **Décodeur état→k/v : m requêtes apprises → `swiglu_apply(q, w0,w1,w2)` → 2 projections
   linéaires k/v par bloc mémoire**, k RMS-normé (comme les vraies clés via `norm_k`),
   **v zero-init + tanh-gate scalaire zero-init** (contenu injecté strictement nul à l'init).
   Sous-modules créés dans `VideoMemoryLayer` (préfixe de nom `memory.*` conservé ⇒
   FSDP-ignored, groupe d'optim `lr_mem`, all-reduce manuel, EMA trainable-only et
   sauvegarde ckpt héritent GRATUITEMENT de toute la plomberie existante).
3. **Décodage HORS de la région checkpointée** : `_forward_inference` décode une fois par
   bloc mémoire et passe `mem_kv` comme entrée explicite du checkpoint, exactement le
   mécanisme déjà validé pour `ttt_state` / `mem_ema_prev` / `mem_x_write`
   (causal_model.py:1397-1408). Le recompute backward relit les mêmes tenseurs, l'état lu
   est le snapshot explicite : piège maison n°1 respecté par construction.
4. **Adressage par prompt (multi-switch) : FiLM léger zero-init sur les clés d'ÉCRITURE
   et sur les requêtes du DÉCODEUR**, conditionné par le prompt-embedding poolé (le même
   pooling `context.mean(dim=1)` que le gate_extra "agreement" existant,
   video_memory.py:343). Zero-init ⇒ FiLM = identité ⇒ byte-identical flag ON à l'init ;
   flag OFF ⇒ sous-modules non créés, state_dict inchangé. C'est la parade structurelle à
   l'interférence entre traces (écrire B/C/D sans écraser A) : chaque segment écrit sur
   son propre manifold de clés, le retour de A ré-adresse le sien.

## 1. État des lieux du code (les faits qui contraignent le design)

* `CausalWanSelfAttention.forward` (causal_model.py:108-472) : deux branches avec cache.
  Branche `sink_tokens > 0` : construit `k_cat/v_cat = [k_sink, (k_prompt_cache), k_local]`
  puis `attention(roped_query, k_cat, v_cat)` (l.351-437). Branche `sink_tokens == 0`
  (cache-dropout c2) : `k_win/v_win` directs (l.438-462). `attention()`
  (wan/modules/attention.py) = flash-attn SANS masque ni bias sur le kv fourni : l'ordre
  dans la concat est indifférent, et un biais de logit n'est PAS disponible (contraint le
  gating, cf. §3.3).
* Le cache est un buffer muté in-place par `_apply_cache_updates` (l.1140-1213) avec une
  arithmétique d'index contiguë `[sink | fenêtre roulante]` (`local_end_index`,
  `write_start_index`, `num_evicted_tokens`, protection sink au recompute, cas
  `sink_recache_after_switch`). Toute réservation de slots après le sink décale TOUTE
  cette arithmétique + le recache + le chemin `is_recompute` : blast radius maximal,
  rejeté (§2).
* Site mémoire actuel : `CausalWanAttentionBlock.forward` (l.544-723), lecture/écriture
  APRÈS self-attn, injection résiduelle `x = x + mem_out`. L'écriture eviction-feed
  (fifo append au niveau bloc l.628-643, pop au niveau modèle l.1350-1385) est saine et
  CONSERVÉE telle quelle.
* `VideoMemoryLayer.step()` (video_memory.py:453-493) : apply-first (`swiglu_apply`),
  write Muon (`swiglu_fast_weight_gradient` + `weight_norm_update`), état threadé par
  `TTTBlockCache` (objet-feuille pytree, jamais un dict). `swiglu_apply` est déjà isolé :
  le décodeur le réutilise tel quel.
* Trainer : mem params = `"memory" in name` partout (optim group distillation.py:415-422,
  all-reduce manuel l.1474-1477 et 1542-1545, clip séparé `_clip_mem_grads` l.1388,
  gather ckpt `_gather_memory_state_dict` l.1905, FSDP-ignore par
  `type(m).__name__ == "VideoMemoryLayer"` l.320-332). EMA = `EMA_FSDP` trainable-only
  (utils/distributed.py:93). Snapshot/restore ttt autour du backward :
  `_snapshot_ttt_state/_restore_ttt_state` (l.1869-1903). Twin mem-OFF e3 avec RNG et
  caches snapshotés : `_twin_snapshot/_twin_restore` (model/streaming_training.py:559-622),
  selftest `E3_TWIN_SELFTEST`. **Le teacher ORACLE du RSD est une généralisation directe
  de ce twin** (§5).
* Switch : entraînement = UN switch A→B max (`has_switched`,
  model/streaming_training.py:216-245 ; `switch_choices` [21,39] dans ttt_memory.yaml,
  [1000] = jamais dans la campagne l1). Inference A→B→A + ORACLE_SPLICE +
  ORACLE_SPLICE_FRAMES + MEM_EPISODIC : validés dans
  `pipeline/switch_causal_inference.py:226-276` avec `_recache_after_switch(...,
  frames_override=...)`. Le pipeline d'ENTRAÎNEMENT
  (`streaming_switch_training.py:_recache_after_switch`, l.304+) n'a PAS `frames_override` :
  à ajouter (étape 4).
* Config runtime l1 : dim 1536, 12 têtes × 128, `local_attn_size=12`, `sink_size=3`
  (train) / 1 (probe), `rope_enhance=true`, `memory_layers=[5,16,27]`, chunk 21 frames,
  `frame_seqlen=1560`, `num_frame_per_block=3`, eviction_feed, use_decay, mem_stream_scale
  0.5, LoRA rank 256, EMA 0.99 trainable-only.

## 2. Décision : insertion des m tokens virtuels dans l'attention

**Verdict : tenseurs séparés concaténés au moment de l'attention.** Nouveau kwarg
`mem_kv: Optional[Tuple[k_mem, v_mem]]` avec `k_mem/v_mem: (B, m, 12, 128)`.

Pourquoi pas des slots réservés après le sink :
* l'arithmétique d'index du cache (roll, recache, recompute, `sink_recache_after_switch`)
  suppose un layout contigu ; chaque `local_end_index` devrait être décalé de m, dans
  ~15 endroits, en cassant le byte-identical par défaut ;
* le gradient RSD doit traverser k/v virtuels → décodeur → état ; le kv_cache est un
  buffer de stockage détaché muté in-place, la concat au moment de l'attention garde le
  graphe vivant sans rien changer au cache ;
* le précédent `prompt_cache` prouve que la concat marche dans les deux modes rope.

Détails d'implémentation :
* Concat dans **LES DEUX branches** : `sink>0` (`k_cat = torch.cat([k_sink, k_mem, ...])`)
  ET `sink==0` (`k_cat = torch.cat([k_mem, k_win])`). La branche sink=0 + rope_enhance
  (fix récent, l.442-457) est précisément le scénario cache-dropout où la mémoire est le
  seul porteur d'ancre : elle DOIT lire les tokens virtuels.
* **Pas de RoPE sur les k virtuels** : ce sont des registres libres de position (pratique
  AdaState/MAG). m n'est pas contraint au grid (f,h,w), donc `causal_rope_apply` est
  inapplicable de toute façon ; le décodeur apprend ses clés directement dans l'espace
  post-rope que lisent les requêtes. Position dans la concat indifférente (pas de masque).
* **MEM_ZERO_ALL** : géré au niveau `_forward_inference` : si
  `int(os.environ.get("MEM_ZERO_ALL", "0")) != 0`, `mem_kv=None` pour tous les blocs
  (print-annonce une fois, comme l'existant l.653-658). C'est une neutralisation EXACTE
  (aucun token concaténé), plus propre que le `* 0.0` résiduel. Le zéroing résiduel
  existant reste pour le mode legacy.
* Nouveau garde-fou anti-« switch inference bypassait la mémoire » : si le modèle a
  `vsink_tokens > 0` et `mem_readout == "vsink"` mais `ttt_cache is None` dans
  `_forward_inference`, **hard error** (l'oubli historique devient impossible à
  reproduire silencieusement).

## 3. Le décodeur état→k/v (video_memory.py)

### 3.1 Forme minimale retenue

Sous-modules de `VideoMemoryLayer`, créés **UNIQUEMENT si `vsink_tokens > 0`** (sinon le
state_dict est inchangé et les vieux ckpts chargent sans clés manquantes ; l'assert
`still_missing_mem` du resume, distillation.py:269-270, ne claque pas) :

```python
# __init__ (si vsink_tokens > 0), dim=1536, m = vsink_tokens, nh=1 -> head_dim = dim
self.vsink_queries = nn.Parameter(torch.randn(m, dim) / math.sqrt(dim))  # m requêtes apprises
self.vsink_qnorm   = RMSNorm(dim, eps=1e-5)          # stabilité de la lecture (comme q_norm)
self.vsink_to_k    = nn.Linear(dim, dim)             # init normal(std=0.02)
self.vsink_knorm   = RMSNorm(dim, eps=1e-6, elementwise_affine=True)  # clés in-distribution
self.vsink_to_v    = nn.Linear(dim, dim)             # ZERO-INIT (poids ET biais)
self.vsink_vgate   = nn.Parameter(torch.zeros(1))    # tanh-gate scalaire (double zéro)

def decode_kv(self, state, B, attn_heads=12, attn_head_dim=128):
    """État fast-weight -> m tokens k/v virtuels prêts pour l'attention softmax du backbone.
    state peut être None (début de séquence) -> init_state (lecture du seed w0/w1/w2)."""
    if state is None:
        state = self.init_state(B, self.vsink_queries.device, self.vsink_queries.dtype)
    w0, w1, w2 = state
    q = self.vsink_qnorm(self.vsink_queries)[None].expand(B, -1, -1)   # (B, m, dim)
    r = swiglu_apply(q, w0, w1, w2)                                     # lecture: (B, m, dim)
    k = self.vsink_knorm(self.vsink_to_k(r)).view(B, -1, attn_heads, attn_head_dim)
    v = (torch.tanh(self.vsink_vgate) * self.vsink_to_v(r)).view(B, -1, attn_heads, attn_head_dim)
    # diagnostics 0-dim (pattern dbg_* existant, .item() au log seulement)
    self.dbg_vsink_krms = k.detach().float().pow(2).mean().sqrt()
    self.dbg_vsink_vrms = v.detach().float().pow(2).mean().sqrt()
    return k, v
```

Et une écriture découplée de la lecture résiduelle (la lecture vsink n'a plus besoin de
`_project_output`) :

```python
def write_only(self, x_write, state, allreduce_fn=None):
    """La moitié do_update de step(): qkv/lr sur x_write, gradient inner-loop, Muon,
    weight_norm_update (decay/init_norms inchangés). Ni lecture q du chunk courant,
    ni _project_output. Retourne new_state. checkpoint_write respecté."""
```

`step()` reste intact octet pour octet (mode legacy `mem_readout: residual`).

### 3.2 Comptage de paramètres et VRAM

Par bloc mémoire (m = 780 par défaut, ½ frame latente ; le plancher oracle1 justifie
m ≤ 1560) : queries m×1536 = 1.20M ; to_k 1536² + 1536 = 2.36M ; to_v idem = 2.36M ;
norms ≈ 3k ; gate 1. **≈ 5.9M params/bloc, ×3 blocs ≈ 17.8M** (m=1560 : ≈ 21.4M ;
m=390 : ≈ 16.0M). En bf16 : ~36MB poids + ~36MB grads + ~72MB états AdamW, répliqués par
rank (FSDP-ignored) : **< 0.15GB/GPU, négligeable.** Coût attention : +m tokens sur ~23k
(sink 4680 + fenêtre 18720) dans 3 blocs sur 30 : **< +7% du temps attention de 3 blocs,
< +1% du forward.**

### 3.3 Inertie initiale : ce que le zero-init garantit et ne garantit pas

* Garanti : **contenu injecté nul** (v ≡ 0 par double zéro to_v + tanh(0)). Aucun k/v
  synthétique « empoisonné » ne peut entrer dans le flux à l'init : le premier gradient
  RSD doit OUVRIR le gate, jamais le refermer en urgence (leçon c1/c2).
* Non garanti : la masse softmax détournée vers des valeurs nulles atténue l'output
  d'attention d'un facteur (1−ε(q)), ε ≈ m·e^{l̄_v}/Σe^{l_réel}. Avec flash-attn sans
  bias, on NE PEUT PAS pousser les logits virtuels à −∞. Ordre de grandeur attendu
  ε ~ m/23k à quelques %, sur 3 blocs/30.
* Mesure obligatoire au selftest S2 (§6) : rollout m>0-init vs m=0 sur 4 prompts, 60f.
  Critère GO : delta DINO dans le bruit de seed connu (0.18). Si NO-GO : (a) réduire m ;
  (b) **variante V2** : dans les 3 blocs mémoire seulement, remplacer l'appel `attention()`
  par `F.scaled_dot_product_attention` avec `attn_mask` additif portant un biais scalaire
  appris `vsink_logit_bias` (init −8 ⇒ masse e^{-8}≈0 EXACTE à l'init, et gate de phase
  possible plus tard). V2 coûte le delta numérique sdpa-vs-flash sur l'arm ON uniquement
  (l'arm OFF reste flash, byte-identical) ; on ne la déclenche que sur échec de S2.
* Règle maison respectée : le knorm est une normalisation de synthèse de clé (toutes les
  vraies clés passent `norm_k`), PAS un renorm-après-soustraction ; les VALEURS ne sont
  jamais renormées (pas de dose fixe : la leçon e2 s'applique au canal valeur, on la
  respecte en laissant v libre et borné par tanh-gate).
* Kill-switch probe $0 : env `MEM_VSINK_VSCALE` (float, défaut 1.0) multipliant v_mem au
  niveau `_forward_inference` (analogue de MEM_STREAM_SCALE, pour capper/annuler
  l'injection sur un ckpt entraîné sans retrain).

### 3.4 Piège init_weights

`CausalWanModel.init_weights()` xavier-init TOUTES les nn.Linear puis
`memory.reset_parameters()` répare le zéro-init (causal_model.py:912-918). **Étendre
`VideoMemoryLayer.reset_parameters()`** : re-zéroer `vsink_to_v` (poids+biais) et
`vsink_vgate`, re-init normal(0.02) `vsink_to_k`, re-init queries. Oubli = gate ouvert
dès le step 0 = exactement le bug d'init déjà vécu (f0908ee).

## 4. Plan d'édition fichier par fichier, dans l'ordre

Chaque étape est mergeable seule, flag OFF par défaut, avec son selftest. Règle maison :
flag OFF ⇒ comportement identique à l'octet près.

### Étape 0 : plomberie des flags (0.25 j)
* `utils/wan_wrapper.py::WanDiffusionWrapper.__init__` : nouveaux kwargs
  `vsink_tokens=0`, `mem_readout="residual"`, passés à `CausalWanModel.from_pretrained`.
* `wan/modules/causal_model.py::CausalWanModel.__init__` +
  `CausalWanAttentionBlock.__init__` : mêmes params, stockés (`self.vsink_tokens`,
  `self.mem_readout`), transmis à `VideoMemoryLayer(...)`.
* `model/dmd_switch.py` / `model/base.py` : rien (model_kwargs passe tout).
* Config : `configs/ttt_vsink.yaml` (copie de ttt_memory.yaml + `vsink_tokens: 780`,
  `mem_readout: vsink`, curriculum §5, `rsd_weight`, wandb project).
* **Selftest S0** : `vsink_tokens: 0` absent/présent ⇒ state_dict identique (diff de
  clés) + hash bitwise d'un rollout 3 chunks vs HEAD (GPU 0, env longlive).

### Étape 1 : le décodeur (1 j)
* `wan/modules/video_memory.py::VideoMemoryLayer` : `__init__` (sous-modules ssi m>0,
  §3.1), `decode_kv()`, `write_only()`, `reset_parameters()` étendu (§3.4). Aucun
  changement à `step()`, `swiglu_apply`, `weight_norm_update`.
* **Selftest S1** (unitaire, GPU 0) : m=0 ⇒ state_dict inchangé ; m>0 ⇒ shapes
  (B,m,12,128), `v.abs().max()==0` à l'init, decode déterministe à état fixé, grads
  non-nuls vers w0/w1/w2 ET vers to_k/queries via `decode_kv(...).sum().backward()`.

### Étape 2 : insertion attention + câblage bloc/modèle (1.5 j)
* `causal_model.py::CausalWanSelfAttention.forward` : kwarg `mem_kv=None` ; branche
  sink>0 : `k_cat = torch.cat([k_sink, k_mem] + ...)` (et v idem) juste après la
  construction de k_sink ; branche sink==0 : concat avant `attention(roped_query, ...)`.
  `mem_kv=None` ⇒ pas une seule ligne exécutée en plus.
* `causal_model.py::CausalWanAttentionBlock.forward` : kwarg `mem_kv=None`, transmis au
  self_attn. Si `mem_readout == "vsink"` et memory présent : (a) NE PLUS calculer/ajouter
  `mem_out` sur les passes de lecture (tout le §« parallel memory branch » lecture est
  sauté) ; (b) sur les passes d'écriture (`ttt_update=True`) : conserver l'append fifo
  eviction-feed puis `new_state = self.memory.write_only(mem_x_write, state_in)` et
  `ttt_cache.state = new_state` (mutation d'attribut objet, mécanisme inchangé).
  `mem_readout == "residual"` ⇒ chemin actuel intact.
* `causal_model.py::CausalWanModel._forward_inference` : avant la boucle blocs, si vsink
  actif : pour chaque bloc mémoire, lire l'état hors checkpoint (même pattern que
  `ttt_state` l.1399-1403), `mem_kv[bi] = block.memory.decode_kv(state, B)` ; forcer
  `None` si `MEM_ZERO_ALL` ; kwargs `"mem_kv": _mem_kv.get(block_index)` dans les DEUX
  branches (checkpointée et non). Hard-assert `ttt_cache is not None` si vsink actif.
  Note : le decode vit HORS checkpoint ⇒ ses activations (m×1536, minuscules) sont
  conservées, le recompute du bloc relit les mêmes tenseurs ⇒ bit-reproductible.
* `pipeline/switch_causal_inference.py` / `pipeline/streaming_training.py` : RIEN à
  changer pour la lecture (ttt_cache déjà threadé partout, `_snapshot_ttt/_restore_ttt`
  couvrent l'état que lit le décodeur ; MEM_EPISODIC devient automatiquement un arm
  vsink-compatible). Vérifier seulement que le harness re-ID accepte un arm `vsink`
  (config-driven, aucun code).
* **Selftest S2** : (i) m=0 : rollout 3 chunks bitwise vs HEAD ; (ii) m>0 +
  `MEM_ZERO_ALL=1` == m=0 bitwise (l'ablation zère AUSSI les tokens virtuels, exigence du
  brief) ; (iii) m>0 à l'init : mesure de ε (§3.3), critère GO/NO-GO → sinon V2 ;
  (iv) `E3_TWIN_SELFTEST=1` toujours 0.0 (le twin snapshot couvre déjà tout ce dont
  dépend decode_kv : state + RNG + caches).

### Étape 3 : curriculum MULTI-SWITCH à l'entraînement (1.5 j)
Cible : schedules mélangés (brief §2 : single + multi, positions aléatoires), dont le
revisit A→B→A est le cas N=2. Design en LISTE D'ÉVÉNEMENTS dès le départ, pas un
« second switch » spécial.
* `trainer/distillation.py` : nouvelle méthode `_get_switch_schedule(max_length)` :
  tirage rank-synced (rank 0 + broadcast, pattern l.771-774) d'un schedule
  `[(frame_idx, cond_key), ...]` piloté par `config.switch_schedule` :
  `n_switches_choices` (ex [1,2,3]), `revisit_prob` (prob qu'un événement re-tire un
  cond DÉJÀ VU au lieu d'un neuf), positions dans `switch_position_choices`, espacement
  min 1 chunk. `switch_schedule` absent ⇒ chemin actuel `_get_switch_frame_index`
  inchangé = byte-identical. Prompts C/D... : encodés depuis des lignes supplémentaires
  du batch prompt-file (le text encoder est déjà sur GPU au setup ; 1 passe T5 par prompt
  extra, coût négligeable), stockés dans `conditional_info["conds"] = {key: cond_dict}`.
* `model/streaming_training.py` :
  - `setup_sequence(...)` : + `switch_schedule` dans `switch_info` (remplace en interne
    le couple switch_frame_index/switch_conditional_dict, qui reste accepté et converti
    en schedule à 1 événement = compat) ;
  - état : `switch_events_done` (compteur), `segment_tails` = **registre
    {cond_key: (tail_latents 12f détachées, pooled_prompt_emb)}** alimenté à CHAQUE
    switch sortant depuis `state["previous_frames"][:, -local_attn_size:]` (remplace le
    `a_tail_latents` unique du design A→B→A) ;
  - `_should_switch_prompt` généralisé : prochain événement du schedule dans la fenêtre
    du chunk ; `_get_current_conditional_dict` : cond du dernier événement passé ;
  - `_generate_chunk` : au chunk d'un événement, passer `switch_frame_index` relatif +
    `switch_conditional_dict = conds[key]` + `switch_recache_frames` (queue du segment
    sortant, comportement student standard) ; si `key` est un REVISIT (déjà dans
    `segment_tails`), marquer `chunk_info["is_resurrection"]=True` +
    `chunk_info["revisit_key"]=key` (et optionnellement le chunk suivant,
    `config.rsd_extra_chunks`). Cap `config.rsd_max_teachers_per_seq` (défaut 1-2) pour
    borner le coût teacher.
* `pipeline/streaming_switch_training.py::generate_chunk_with_cache` : déjà générique
  (un switch par appel de chunk) : N switches = N appels avec kwargs ; contrainte
  documentée : max 1 événement par chunk (le schedule generator l'impose). Seul ajout :
  `_recache_after_switch(..., frames_override=None)` (miroir exact du hook inference,
  switch_causal_inference.py:80-93) pour l'étape 4.
* **Selftest S3** : `switch_schedule` absent + `rsd_weight: 0` ⇒ 5 steps de train, pertes
  bitwise vs HEAD (seed fixe, 1 GPU). Schedule ON ⇒ assert de la séquence des conds
  générée (ex A,B,C,A) vs le schedule tiré, et du contenu du registre `segment_tails`
  (clés + shapes) à chaque événement.

### Étape 4 : le teacher ORACLE_SPLICE (1 j)
* `model/streaming_training.py::generate_next_chunk` : nouveau paramètre
  `make_teacher: bool` (posé par le trainer quand le chunk est une résurrection et
  `rsd_weight > 0`). Généralisation du twin e3, MÊME machinerie :
  1. `_snap = self._twin_snapshot()` (couvre kv/crossattn/flow/prompt caches, ttt state
     + ema + fifo, RNG cpu/cuda/py, has_switched) ;
  2. recache oracle : `pipeline._recache_after_switch(output_B_tail,
     current_start_frame, cond_A, frames_override=a_tail_latents)` : le contenu de la
     queue de A aux positions courantes (le splice validé par la probe, +0.22) ;
  3. rollout du chunk sous `torch.no_grad()` + `MEM_ZERO_ALL=1` (env posé/restauré comme
     le twin, l.716-733) ⇒ **teacher = pur oracle KV, zéro mémoire, zéro fuite de B**
     (la fuite connue du `_recache_after_switch` en injection active est neutralisée par
     MEM_ZERO_ALL qui englobe le recache) ;
  4. `finally: self._twin_restore(_snap)` ⇒ le rollout student repart d'un état bitwise
     identique, même RNG ⇒ même bruit ⇒ paire teacher/student exacte, variance nulle.
  5. `chunk_info["teacher_frames"] = teacher.detach()`.
  Coût : un forward chunk no_grad + un aller-retour CPU du snapshot (~9.2GB RAM hôte :
  30 blocs × k+v × 51480×12×128 bf16 ; VRAM nette ≈ 0, les buffers sont restaurés par
  `copy_` en place).
* **Selftest S4** : `RSD_TEACHER_SELFTEST=1` ⇒ teacher SANS splice (frames_override=None
  ⇒ recache standard identique au student) et SANS MEM_ZERO_ALL ⇒
  `max|student − teacher| == 0.0` bitwise (valide la complétude du snapshot dans le
  nouveau contexte 2e-switch, même contrat que E3_TWIN_SELFTEST).

### Étape 5 : la loss RSD (0.5 j)
* `model/streaming_training.py::compute_generator_loss` : après le bloc DMD,
  si `chunk_info.get("teacher_frames") is not None` :
  `rsd = F.huber_loss(chunk_masked, teacher_masked)` sur `gradient_mask` (frames neuves
  uniquement), `loss += config.rsd_weight * rsd`. Logs wandb systématiques :
  `rsd_loss`, `rsd_l2`, `rsd_teacher_student_gap`, + `dbg_vsink_vrms/krms` par bloc dans
  `_log_memory_stats` (pattern dbg_* existant, distillation.py:1774-1797).
* `trainer/distillation.py` : grad probe steps 0/5/10... étendu à
  `vsink_to_v / vsink_to_k / vsink_queries` (l.1256-1267) ; c'est le détecteur précoce
  « gate=0 ⇒ grads=0 » (leçon starvation : ici FAUX car v est linéaire en to_v, le grad
  RSD arrive dès le step 0 : l'assert doit le prouver).
* Routage du gradient, phase 1 : `detach_state` par chunk (défaut) ⇒ le RSD entraîne
  décodeur + (via state du chunk courant si train_write) l'écriture locale. Phase 2 si
  plafond : élargir `bptt_window`/`grad_tail_W`/`keep_last_write_grad` (mécanisme
  EXISTANT, distillation.py:1178-1247) pour que la fenêtre couvre du dernier chunk de A
  jusqu'à la résurrection ⇒ le grad RSD atteint les WRITES de la queue de A. À chiffrer
  au profile_vram avant activation.
* **Selftest S5** : `rsd_weight: 0` ⇒ bitwise vs étape 4 ; `rsd_weight > 0` ⇒ probe :
  grads non-nuls sur to_v au step 0, nuls sur to_v si `MEM_ZERO_ALL=1` (sanité).

### Étape 6 : intégration ops + profil VRAM (0.75 j)
* Resume/warm-start : `trainer/distillation.py:269-270` : l'assert `still_missing_mem`
  claque si on charge un ckpt pré-vsink dans un modèle vsink. Ajouter
  `config.allow_missing_vsink_keys: true` qui exclut les clés `vsink_` de l'assert
  (elles gardent alors leur init inerte : le warm-start voulu).
* Vérifs automatiques au setup (asserts + prints rank0) : chaque param `vsink_*` est
  (a) dans `self.mem_params` (groupe lr_mem), (b) requires_grad, (c) dans le shadow
  `EMA_FSDP` (trainable-only ⇒ automatique, mais on le PROUVE par un print de comptage),
  (d) présent dans `_gather_memory_state_dict` au premier save.
* `profile_vram` (hook existant, l.1368-1386) sur la config réelle 4 GPU, curriculum ON,
  teacher ON, `mem_breakdown: false` (piège maison), EMA CPU-offload comme e4.
* **Selftest S6** : dry-run 20 steps 4×GPU (fenêtre libre, jamais 5 GPUs) : VRAM pic
  < 44GB/rank, drift guard OK (`assert_memory_drift`), save→load→resume roundtrip, wandb
  actif.

## 5. Budget VRAM et temps (contrainte : 4×48GB, RSD double le forward)

| Poste | VRAM/GPU | Notes |
|---|---|---|
| Base l1 write-training (mesuré, campagnes c3/l1) | ~20-38GB selon phase | tenait déjà sur 4×48 avec empty_cache seed-step + EMA CPU-offload |
| Décodeur (params+grads+AdamW, répliqués) | +0.15GB | FSDP-ignored |
| Concat attention (+m ≤ 1560 tokens, 3 blocs) | +<0.3GB activations | < +1% forward |
| Teacher ORACLE (no_grad, snapshot CPU) | ~+1-2GB transitoire | +9.2GB RAM HÔTE ; VRAM restaurée en place |
| Curriculum A→B→A (séquence ~96-111 frames) | +0GB | chunks no_grad en plus ⇒ TEMPS, pas VRAM (KV taille fixe) |
| Phase 2 BPTT élargi (optionnel) | à profiler | seule ligne à risque ; gated par profile_vram |

Le « double forward » teacher+student est un coût TEMPS (+30-50% sur les steps générateur
avec résurrection, ~1 chunk inference + 2 copies host de ~9GB), pas un doublement VRAM,
parce que le teacher est no_grad et réutilise les buffers via snapshot/restore.

Temps de dev total : **~6 jours** (0.25 + 1 + 1.5 + 1 + 1 + 0.5 + 0.75), chaque étape
livrée avec son selftest byte-identical, exécutables sur GPU 0 sans toucher aux runs.

## 6. Récapitulatif des pièges maison et où le plan les traite

* Checkpointing = entrées explicites : decode hors checkpoint, `mem_kv` en kwarg explicite
  (étape 2), même contrat que ttt_state/mem_ema_prev/mem_x_write.
* Params mémoire FSDP-ignored + all-reduce manuel : décodeur DANS VideoMemoryLayer ⇒
  hérite du module-ignore (type-check l.322) et du name-match `"memory"` (all-reduce,
  optim, clip, gather) ; prouvé par asserts à l'étape 6.
* Jamais de renorm-après-soustraction : aucun chemin soustractif ; knorm = norme de
  synthèse de clé (distribution des vraies clés), valeurs jamais renormées.
* EMA trainable-only : automatique (requires_grad), vérifié par comptage étape 6.
* MEM_ZERO_ALL doit zéroer AUSSI les tokens virtuels : neutralisation exacte par
  non-concat, selftest S2-ii.
* Switch/interactive inference qui bypassait la mémoire : hard-assert ttt_cache si vsink
  actif (étape 2).
* use_ema silencieux : les configs d'éval vsink DOIVENT poser `use_ema` explicitement
  (rappel dans ttt_vsink.yaml, commentaire en tête).
* mem_breakdown VRAM : `mem_breakdown: false` sur tout run réel (étape 6).
* Fuite de B dans `_recache_after_switch` en injection active : le teacher tourne sous
  MEM_ZERO_ALL qui englobe le recache (étape 4) ; pour le student c'est le comportement
  voulu (la mémoire assiste le recache du retour-A).
* id_loss Goodhart : la loss RSD est PAIRÉE (même bruit, même histoire) et dense en
  latents ; pas de terme identité auto-référent ; le DMD reste le garde on-manifold.
