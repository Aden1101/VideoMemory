# Tour des diffs, commit par commit, hunk par hunk

Découpage de l'arbre de travail en **3 commits au niveau fichier** (chaque fichier va entièrement dans un seul commit, pas besoin de `git add -p`). Committer dans l'ordre 1 → 2 → 3 (le commit 3 référence des clés de log produites par le commit 2).

- **Commit 1** : module mémoire MAG + intégration + fix de fuite VRAM
- **Commit 2** : modèle d'entraînement streaming : objectif d'identité E1 (+ detach BPTT cross-chunk, text-dropout)
- **Commit 3** : trainer : BPTT fenêtré, dégel LoRA sélectif, logging wandb, diagnostics mémoire

Légende des tags : **[mem]** = la feature mémoire fast-weight, **[leak]** = fix de fuite VRAM / OOM, **[E1]** = l'objectif d'identité, **[infra]** = logging / plomberie.

---

# COMMIT 1 : module mémoire + intégration + fix de fuite VRAM

Fichiers : `wan/modules/video_memory.py`, `wan/modules/causal_model.py`, `utils/wan_wrapper.py`, `configs/ttt_memory.yaml`, `pipeline/streaming_training.py`, `pipeline/streaming_switch_training.py`, `utils/distributed.py`

## `wan/modules/video_memory.py` : la mémoire fast-weight elle-même

1. **Signature + docstring de `weight_norm_update`** : ajoute `decay=None, init_norms=None`. **[mem]** Sépare la mise à jour du fast-weight en deux modes et les documente : `decay=None` = le renorm-to-init original de Scal3R (conserve la norme, pas d'oubli) ; mode gated-delta = `w ← decay*w + muon(grad)` puis cape chaque ligne à sa norme d'INIT (ajoute une primitive d'oubli/rétention).
2. **Corps de `weight_norm_update`** : le `if decay is None: ... return / else gated-delta ...`. **[mem]** Implémente les deux branches. Le chemin par défaut est un comportement identique au byte près ; la branche gated-delta n'est prise que si `use_decay=True`.
3. **Helper `_sinusoid_embed`** : embedding sinusoïdal sans paramètre. **[mem]** Encode les signaux scalaires de `gate_extra` (denoising step, frames-since-switch) en vecteurs.
4. **Params de `VideoMemoryLayer.__init__`** : ajoute `use_decay`, `decay_init`, `bounded_gate`, `single_norm`, `gate_extra`. **[mem]** Les nouvelles options mémoire (toutes OFF par défaut = comportement inchangé). `single_norm` est le flag d'hygiène de lecture ajouté cette session.
5. **Corps de `__init__` : stocke les flags + `decay_logit`** : stocke les flags ; si `use_decay`, crée un forget-gate scalaire appris. **[mem]**
6. **`__init__` : build gate_extra + changement d'init du fast-weight** : appelle `_build_gate_extra` si activé ; **et change l'init de w0/w1/w2 de `sqrt(2)/sqrt(fan_in)` à `1/sqrt(fan_in)`** (fidèle à LaCT/Scal3R, retire le gain sqrt(2) parasite). **[mem]** (Note : ce changement d'init affecte tout run mémoire.)
7. **`_build_gate_extra` + `_gate_extra_vec`** : la machinerie gate_extra. **[mem]** Construit un `proj_extra` zéro-init (do-no-harm à l'init) et le vecteur de features pour les 3 entrées de gate optionnelles (agreement idée-6, denoising_step idée-2, frames_since_switch idée-5). Pose aussi `dbg_agreement` pour le logging.
8. **`reset_parameters` : re-zéro `proj_extra`** : garde gate_extra do-no-harm après que l'init globale du modèle se re-lance. **[mem]**
9. **`_project_output`** : ajoute `single_norm` (skip le `o_norm` redondant pour que la magnitude brute de la lecture atteigne proj), l'addition gate_extra, et `bounded_gate` (tanh sur la gate). **[mem]** L'hygiène du canal de lecture.
10. **Helper `_decay_and_norms`** : renvoie `(decay, init_norms)` si `use_decay`, sinon `(None, None)`. **[mem]** Alimente la branche gated-delta de `weight_norm_update`.
11. **`step(...)`** : passe `prompt_emb/frames_since_switch/denoising_step` à `_project_output`, utilise `_decay_and_norms`, et enregistre `dbg_read_rms/dbg_out_rms` (0-dim, pas de sync hôte). **[mem]** Le chemin AR / inférence.
12. **`forward_sequence`** : passe decay/init_norms à l'écriture parallèle. **[mem]**

## `wan/modules/causal_model.py` : greffe la mémoire dans le transformer

1. **`import os`** : pour le hook d'ablation env-gated. **[mem]**
2. **Params de `CausalWanAttentionBlock.__init__`** : ajoute `gate_extra/use_decay/decay_init/bounded_gate/single_norm/muon_update_steps`, forwardés à `VideoMemoryLayer`. **[mem]** Plomberie des flags (niveau bloc).
3. **Signature de `forward`** : ajoute `denoising_step`, `switch_global_frame`. **[mem]** Entrées pour les signaux gate_extra.
4. **Branche mémoire dans `forward`** : construit les signaux par-token `mem_ds`/`mem_fss` (denoising_step, frames_since_switch calculé depuis l'index de frame global vs le switch), appelle `memory.step(...)` avec + le contexte prompt (agreement). **[mem]**
5. **Injection + ablation** : `x = x + (mem_out.detach() if ttt_update else mem_out)` (le **fix C3** : detach sur la passe d'écriture pour que le backbone gelé ne construise pas un graphe backward gâché), et le hook d'ablation env-gated `MEM_ZERO_ALL` (zéro mem_out pour l'attribution de l'éval, OFF par défaut). **[leak]/[mem]**
6. **Params de `CausalWanModel.__init__` + construction des blocs** : même plomberie de flags, niveau modèle. **[mem]**
7. **kwargs de `_forward_inference`** : passe `switch_global_frame` et `denoising_step` (= timestep `t`) aux blocs, dans les chemins checkpointé et non-checkpointé. **[mem]**

## `utils/wan_wrapper.py` : expose les flags au config
1. **Params + forward de `__init__` du wrapper** : ajoute les kwargs mémoire et les forwarde à `CausalWanModel.from_pretrained`. **[mem]** Le tuyau config → modèle.

## `configs/ttt_memory.yaml`
1. **`gate_extra: []`** : documente l'option (OFF par défaut). **[mem]**

## `pipeline/streaming_training.py` : fix de fuite VRAM
1. **`_write_context(requires_grad=True)`** : ne construit le graphe d'écriture que si le chunk sera backpropé (sur les steps critic le chunk est detaché → construire le graphe enable_grad là était du pur gâchis + la fuite dominante). **[leak]**
2. **Write du dernier bloc en no_grad** : `_write_grad = requires_grad and (block_index < last)` : le write du dernier bloc n'est jamais lu dans le chunk et est detaché en C3 cross-chunk, donc son graphe n'était jamais backpropé ni libéré (+1 write-forward fuité par step générateur). **[leak]**

## `pipeline/streaming_switch_training.py` : même fix + plomberie switch
1. **`switch_global_frame` calculé** : index de frame global du switch (pour le signal gate frames_since_switch). **[mem]**
2. **Threadé dans les appels denoising + write** : passe `switch_global_frame` au modèle. **[mem]**
3. **Même fix no_grad du write dernier-bloc** que ci-dessus. **[leak]**

## `utils/distributed.py`
1. **Override env `PG_TIMEOUT_MIN`** : le timeout du process-group NCCL (hardcodé à 30 min avant) est maintenant réglable par env ; les scripts de lancement mettent `PG_TIMEOUT_MIN=20`. **[infra]**

---

# COMMIT 2 : modèle d'entraînement streaming : objectif E1 (+ BPTT cross-chunk, text-dropout)

Fichier : `model/streaming_training.py`

1. **imports `os`, `random`** : pour le chemin du hub DINO (os) et le tirage du text-dropout (random). **[E1]**
2. **`__init__` : text-dropout + knobs E1** : `text_dropout_prob` (coupe le prompt pour forcer la mémoire) et les poids de l'objectif E1 `id_loss_weight / id_decode_frames / tv_loss_weight / tv_motion_floor`, + `_dino` paresseux et `_id_anchor` par-séquence. Tous à 0 par défaut = comportement inchangé. **[E1]**
3. **`_generate_chunk` reçoit `detach_state`** : nouvel argument pour contrôler le detach du BPTT cross-chunk (voir hunk 5). **[leak]**
4. **Bloc text-dropout** : avec proba `text_dropout_prob` (et seulement pour chunk_start_frame>0), GÉNÈRE le chunk conditionné sur du texte NULL ; la cible DMD + le critic gardent le vrai prompt, donc le student doit reproduire le sujet depuis la mémoire. **[E1]**
5. **Detach BPTT cross-chunk C3** : detache l'état TTT porté entrant dans le chunk quand `detach_state=True` (le graphe d'écriture de chaque chunk part d'une feuille → pas de crash "backward une 2e fois") ; quand `detach_state=False` (chunks 2..K d'une fenêtre BPTT) il garde le grad_fn pour que la loss backprop dans le chunk précédent. **[leak]** (C'est la plomberie BPTT du write-training qu'utilise le chemin fenêtré du commit 3.)
6. **`setup_sequence` : reset `_id_anchor`** : efface l'ancre d'identité au début de chaque nouvelle séquence. **[E1]**
7. **`generate_next_chunk` reçoit `detach_state`** : le forwarde à `_generate_chunk`. **[leak]**
8. **Le rollout cache l'ancre d'identité** : sur le 1er chunk (`previous_frames is None`), appelle `_set_id_anchor(full_chunk)` **dans le rollout** (Fix 1), pour que l'ancre soit posée que ce soit un step générateur ou critic qui roule frame 0. **[E1]**
9. **Méthodes helper E1** : `_get_dino` (DINOv2 gelé offline), `_dino_embed_pixels` (embeddings CLS DINO), `_set_id_anchor` (cache l'ancre chunk-0, no grad), `_compute_identity_tv_loss` (la loss cosine d'identité sur les chunks suivants + le plancher anti-freeze de variance temporelle ; renvoie des tensors de log 0-dim cohérents entre chunks pour que `merge_dict_list` les garde). **[E1]**
10. **Intégration dans `compute_generator_loss`** : `dmd_loss += id_loss_weight*id_loss + tv_loss_weight*tv_loss`, enveloppé dans un garde-OOM pour qu'un OOM de décodage/DINO saute le terme au lieu de tuer le run. **[E1]**

---

# COMMIT 3 : trainer : BPTT fenêtré, dégel LoRA, logging, diagnostics

Fichier : `trainer/distillation.py`

1. **Réordonnancement du SummaryWriter** : déplace `wandb.init(...)` AVANT `SummaryWriter(...)` pour que `sync_tensorboard` patche le writer TB (sinon nos courbes `add_scalar` n'atteignent pas wandb). **[infra]**
2. **Dégel LoRA sélectif** : `lora_unfreeze_from_block` : gèle le LoRA des blocs sous K (retire leur grad + état AdamW → économise la VRAM qui faisait OOM le full-unfreeze) ; les params mémoire toujours entraînables, le LoRA du critic intact. `None` = ancien comportement. **[infra]**
3. **`fake_score_cpu_offload`** : offload CPU optionnel des params/grads/optim FSDP du critic (~2.6GB/rank), OFF par défaut. **[leak]**
4. **BPTT fenêtré cross-chunk** : le chemin `K_bptt = bptt_window` : roule K chunks (les K−W premiers en no_grad pour construire un état profond au-delà de l'éviction KV, les W derniers avec grad), somme les losses, un seul backward. Inclut le **force-fresh** (gaté sur `id_loss_weight>0` pour n'affecter qu'E1) et la **sonde de gradient** (steps {0,5,10,15,20,30} : imprime les normes de grad READ qkv/proj + WRITE lrs/decay pour prouver que le gradient atteint read+write). **[E1]/[leak]**
5. **`empty_cache_each_step`** : `torch.cuda.empty_cache()` optionnel par step pour récupérer la VRAM cachée bloquée, OFF par défaut. **[leak]**
6. **Lignes whitespace-only** : trims d'espaces en fin de ligne autour des steps d'optim (cosmétique). **[infra]**
7. **Garde `dmdtrain_gradient_norm` + logging E1** : garde l'add_scalar de `dmdtrain_gradient_norm` (absent dans le chemin fenêtré → c'était un crash KeyError) et logue les termes E1 `e1_id_cos/e1_id_loss/e1_latent_motion/e1_tv_loss` vers wandb. **[E1]/[infra]**
8. **Expansion de `_log_memory_stats`** : diagnostics mémoire par-couche vers wandb : gate bias, norme du gate-weight, usage gate_extra `gx_*`, deltas de poids qkv/proj depuis l'init, RMS `memout/read/agree` (depuis les `dbg_*` de video_memory.py), et `drift` du fast-weight + `ratio statenorm/colnorm` (le ratio expose le state-collapse muon=0 que le drift ne voit pas). Ce sont les courbes qu'on a lues toute la session (gatew/memout/drift). **[infra]**

---

# Les commandes de commit (à lancer par toi)

```bash
# Commit 1 : module mémoire + intégration + fix de fuite VRAM
git add wan/modules/video_memory.py wan/modules/causal_model.py utils/wan_wrapper.py \
        configs/ttt_memory.yaml pipeline/streaming_training.py \
        pipeline/streaming_switch_training.py utils/distributed.py
git commit -m "feat: MAG fast-weight TTT memory (SwiGLU read, Muon write, gated-AdaLN inject) + VRAM leak fix"

# Commit 2 : modèle streaming : objectif E1 (+ detach BPTT, text-dropout)
git add model/streaming_training.py
git commit -m "feat: E1 identity anchor + anti-freeze temporal-variance objective; cross-chunk BPTT detach; text-dropout"

# Commit 3 : support trainer
git add trainer/distillation.py
git commit -m "feat: training-loop support -- windowed BPTT, selective LoRA unfreeze, E1/memory wandb logging"
```
