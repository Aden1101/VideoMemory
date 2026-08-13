# MAG memory : vue d'ensemble des changements (diffs commentés)

Ce qu'on a modifié dans Anchor-Forcing pour greffer la mémoire TTT/fast-weight ("MAG") et l'entraîner. Une section par fichier : le **vrai diff vs HEAD** (lignes ajoutées/retirées) suivi de son explication en français. État de l'arbre de travail au 2026-06-18 (10 fichiers, +117/-19 lignes).

> Note : tout est piloté par la **config** (`model_kwargs` + clés top-level), **plus aucune variable d'environnement**. Les knobs mémoire (`memory_layers`, `gate_init`, `ckpt_write`) passent par `model_kwargs`, comme `local_attn_size`, donc ils sont enregistrés dans le `config.json` du checkpoint et dans la config wandb (reproductibilité).

> Emplacement : ce doc vit dans `reference/` (gitignoré, privé). Pour en faire un `CHANGES.md` committé, dis-le et je le déplace.

---

## Surface de contrôle (les knobs ajoutés)

Tout est opt-in. Avec les défauts, la mémoire est à la couche 5 avec un gate 0.1 (Scal3R) ; mettre `gate_init: 0.0` redonne un warm start inerte (== baseline AF au step 0).

| Knob | Où | Défaut | Effet |
|---|---|---|---|
| `memory_layers` | `model_kwargs` | `[5]` | quels blocs transformer reçoivent une branche mémoire, ex `[5, 11, 17, 23, 29]` |
| `gate_init` | `model_kwargs` | `0.1` | init du biais de gate (0.1 façon Scal3R casse la zone morte gate~0 ; `0.0` = inerte) |
| `ckpt_write` | `model_kwargs` | `false` | gradient-checkpoint de l'écriture (recalcul Hebbian + Newton-Schulz au backward) pour réduire la VRAM |
| `train_write` | config | `false` | `true` = entraîne le write-path (écriture sous `enable_grad`, graphe d'état gardé) ; `false` = read-only |
| `profile_vram` / `profile_vram_warmup` | config | `0` / `3` | profiler VRAM : reset du pic après warmup, report pic alloc/reserved, hard-exit |
| `use_wandb` / `wandb_project` | config | `false` | miroir des scalaires TensorBoard vers wandb |

---

## `.gitignore` (+3)

```diff
 # --- Figures/plots and other unecessary files ---
 reference/
+
+# --- wandb local cache ---
+wandb/
```

**Explication :** on ignore `wandb/` (cache local de wandb) pour ne jamais le committer. `reference/` était déjà ignoré.

---

## `configs/ttt_memory.yaml` (+6)

```diff
 auto_resume: False
 logdir: /shared/ssd_14T_project/adenelmi/ttt_memory_train
 
+use_wandb: true
+wandb_project: anchor-forcing-memory
+
 ema_weight: 0.99
```

```diff
 model_kwargs:
   timestep_shift: 5.0
   local_attn_size: 12
   sink_size: 3
+  memory_layers: [5]    # which transformer blocks get a memory branch
+  gate_init: 0.1        # gate-bias init (Scal3R-style 0.1; 0.0 = inert do-no-harm)
+  ckpt_write: false     # gradient-checkpoint the write update (cuts VRAM in write-training)
```

**Explication :** active wandb (le trainer fera `wandb.init(sync_tensorboard=True)`), et déclare les 3 knobs mémoire dans `model_kwargs` (donc captés dans la config + le checkpoint). Ici : mémoire à la couche 5, gate 0.1. Le watcher surcharge `memory_layers` par run.

---

## `wan/modules/video_memory.py` (+15) — le module mémoire

La mémoire MAG : une couche fast-weight (TTT) SwiGLU alignée Scal3R/LaCT. Lecture APPLY-FIRST `o = f_W(q)`, puis (si écriture) update Hebbian + Muon (Newton-Schulz) + weight-norm des fast weights `w0/w1/w2`.

**Import du checkpoint :**

```diff
 from torch import Tensor, nn
+from torch.utils.checkpoint import checkpoint
```

**Nouveau paramètre de constructeur :**

```diff
         g_init: float = 0.0,       # 0.0 -> inert warm start (identical to baseline at step 0; Scal3R: 0.1)
         detach_state: Optional[bool] = None,  # None -> detach in eval, keep graph in train
+        checkpoint_write: bool = False,  # gradient-checkpoint the write update (recompute in backward) to cut VRAM
     ):
```

```diff
         self.use_modulation = use_modulation
         self.detach_state = detach_state
+        self.checkpoint_write = checkpoint_write
```

**Explication :** `checkpoint_write` est désormais un **paramètre** (passé depuis `model_kwargs.ckpt_write`), pas une variable d'env. À True, l'écriture est recalculée au backward au lieu d'être stockée (réduit les activations du write-graph, utile en write-training).

**L'écriture refactorée en closure + checkpointée :**

```diff
         if do_update:                                          # write the clean chunk
-            w0_grad, w1_grad, w2_grad = swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2)
-            w0n, w1n, w2n = weight_norm_update(
-                w0, w1, w2, w0_grad, w1_grad, w2_grad, self.muon_update_steps, allreduce_fn)
+            def _write(k, v, w0, w1, w2, lr0, lr1, lr2):
+                g0, g1, g2 = swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2)
+                return weight_norm_update(
+                    w0, w1, w2, g0, g1, g2, self.muon_update_steps, allreduce_fn)
+            if self.checkpoint_write and torch.is_grad_enabled():
+                w0n, w1n, w2n = checkpoint(_write, k, v, w0, w1, w2, lr0, lr1, lr2,
+                                           use_reentrant=False)
+            else:
+                w0n, w1n, w2n = _write(k, v, w0, w1, w2, lr0, lr1, lr2)
```

**Explication :** les 2 lignes retirées calculaient le gradient Hebbian puis l'update weight-norm. On les met dans une closure `_write(...)` (calcul identique). Si `checkpoint_write` est actif **et** qu'on est en mode grad (write-training), on wrappe dans `torch.utils.checkpoint(..., use_reentrant=False)` (recalcul au backward) ; sinon appel normal. Comportement par défaut **inchangé**, vérifié bit-exact (`grad_equiv_ckpt_write.py`, relL2=0).

---

## `wan/modules/causal_model.py` (+24/-9) — câblage de la branche mémoire

**Plus d'`import os`** (on ne lit plus l'env) :

```diff
 import matplotlib.pyplot as plt
-import os
```

**Le bloc d'attention reçoit `gate_init` / `ckpt_write` :**

```diff
                  eps=1e-6,
-                 use_memory=False):
+                 use_memory=False,
+                 gate_init=0.1,
+                 ckpt_write=False):
```

```diff
-        # parallel memory branch (None = unchanged baseline block)
+        # parallel memory branch (None = unchanged baseline block); gate_init / ckpt_write come
+        # from model_kwargs (gate_init 0.1 = Scal3R-style, 0.0 = inert do-no-harm).
         self.memory = VideoMemoryLayer(dim=dim, num_heads=1, inter_multi=4,
-                                       g_init=0.0) if use_memory else None
+                                       g_init=gate_init, checkpoint_write=ckpt_write) if use_memory else None
```

**Le modèle reçoit `memory_layers` / `gate_init` / `ckpt_write` (enregistrés via `@register_to_config`) :**

```diff
                  qk_norm=True,
                  cross_attn_norm=True,
-                 eps=1e-6):
+                 eps=1e-6,
+                 memory_layers=(5,),
+                 gate_init=0.1,
+                 ckpt_write=False):
```

```diff
         cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
-        MEMORY_LAYERS = {5}  # inject memory at L5 to start; later e.g. {5, 10, 15, 20}
+        # which blocks get a memory branch (model_kwargs.memory_layers, default {5}); gate_init and
+        # ckpt_write are forwarded to those branches.
+        MEMORY_LAYERS = {int(x) for x in memory_layers}
         self.blocks = nn.ModuleList([
             CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                     local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
-                                    use_memory=(i in MEMORY_LAYERS))
+                                    use_memory=(i in MEMORY_LAYERS),
+                                    gate_init=gate_init, ckpt_write=ckpt_write)
             for i in range(num_layers)
         ])
```

**Explication :** avant, les couches mémoire (`{5}`) et l'init du gate (`0.0`) étaient codées en dur. Maintenant elles viennent des paramètres `memory_layers` / `gate_init` / `ckpt_write`, qui descendent depuis `model_kwargs` (cf. `wan_wrapper.py`). Comme `CausalWanModel.__init__` est décoré `@register_to_config`, ces valeurs sont **auto-enregistrées dans le `config.json`** sauvé avec le checkpoint. (Le commentaire de la branche dans `forward` a aussi été raccourci à `# parallel memory branch`.)

---

## `utils/wan_wrapper.py` (+6) — passage de `model_kwargs` au modèle

```diff
             local_attn_size=-1,
             sink_size=0,
-            use_gan=False
+            use_gan=False,
+            memory_layers=(5,),
+            gate_init=0.1,
+            ckpt_write=False
     ):
```

```diff
             self.model = CausalWanModel.from_pretrained(
                 model_name, local_attn_size=local_attn_size, sink_size=sink_size,
+                memory_layers=memory_layers, gate_init=gate_init, ckpt_write=ckpt_write,
                 low_cpu_mem_usage=False, device_map=None)
```

**Explication :** c'est le maillon qui relie la config au modèle. `WanDiffusionWrapper` est construit avec `**model_kwargs` (cf. `model/base.py`), donc il accepte maintenant `memory_layers` / `gate_init` / `ckpt_write` et les passe à `CausalWanModel.from_pretrained(...)`, exactement comme `local_attn_size` / `sink_size`. C'est le pattern standard du codebase (au lieu des variables d'env qu'on avait au début).

---

## `pipeline/streaming_training.py` (+16) — contrôle du gradient sur le write-path

Le rollout streaming qui génère bloc par bloc et écrit la mémoire une fois par bloc.

```diff
         self.rope_enhance = kwargs.get("rope_enhance", False)
         print(f"[rope_enhance policy] {self.rope_enhance}")
 
+        # train_write=True: write under enable_grad + keep state graph so the write-path trains.
+        self.train_write = kwargs.get("train_write", False)
+        if self.train_write:
+            from wan.modules.video_memory import VideoMemoryLayer
+            model = getattr(self.generator, "model", None)
+            if model is not None:
+                for _, m in model.named_modules():
+                    if isinstance(m, VideoMemoryLayer):
+                        m.detach_state = False
+            print("[train_write] ENABLED")
+
+    def _write_context(self):
+        return torch.enable_grad() if getattr(self, "train_write", False) else torch.no_grad()
```

```diff
             with torch.no_grad():
+            with self._write_context():
                 self.generator(
                     noisy_image_or_video=context_noisy,
                     conditional_dict=conditional_dict,
```

**Explication :** `train_write` (défaut False) décide si on entraîne le write-path. Quand True : on met `detach_state = False` sur chaque `VideoMemoryLayer` pour **garder le graphe d'état** des fast weights d'un bloc à l'autre (sinon l'écriture n'est pas entraînable). `_write_context()` renvoie `enable_grad()` en write-training, sinon `no_grad()`. La bascule remplace le `torch.no_grad()` qui entourait l'écriture (re-run propre du bloc).

---

## `pipeline/streaming_switch_training.py` (1 ligne) — même bascule, pipeline de switch

```diff
             with torch.no_grad():
+            with self._write_context():
                 self.generator(
                     noisy_image_or_video=context_noisy,
                     conditional_dict=cond_in_use,
```

**Explication :** exactement le même changement d'une ligne sur la variante "prompt-switch" du rollout. `_write_context()` est hérité de `StreamingTrainingPipeline`.

---

## `model/dmd_switch.py` (+2/-1) — propagation du flag

```diff
             global_sink=getattr(self.args, "global_sink", False),
             rope_enhance=getattr(self.args, "rope_enhance", False),
-        )
+            train_write=getattr(self.args, "train_write", False),
+        )
```

**Explication :** transmet `train_write` depuis les args du config jusqu'au constructeur du pipeline (pour atteindre `streaming_training`). (Newline final ajouté que git signalait.)

---

## `trainer/distillation.py` (+59) — le plus gros, 4 ajouts indépendants

**1) wandb (rank-0) :**

```diff
         if self.is_main_process:
             self.writer = SummaryWriter(log_dir=config.logdir)
+            if getattr(config, "use_wandb", False):
+                import wandb
+                run_name = os.path.basename(config.logdir.rstrip("/"))
+                wandb.init(
+                    project=getattr(config, "wandb_project", "anchor-forcing-memory"),
+                    name=run_name,
+                    id=run_name,
+                    resume="allow",
+                    dir=config.logdir,
+                    config=OmegaConf.to_container(config, resolve=True),
+                    sync_tensorboard=True,  # mirror every SummaryWriter add_scalar to wandb
+                )
```

**Explication :** si `use_wandb`, le rank 0 init un run dont le nom/id = basename du logdir (reprendre un training continue la même courbe via `resume="allow"`). `sync_tensorboard=True` recopie automatiquement vers wandb chaque `add_scalar` existant (losses, grad norms, `memory/*`), donc pas besoin de logger à la main. `config=...` enregistre les hyperparams (dont `memory_layers`/`gate_init`) pour la repro, et `dir=config.logdir` garde le cache wandb hors du disque home (presque plein).

**2) Exclure la mémoire de FSDP :**

```diff
+        # The TTT memory's fast weights (w0/w1/w2) are compute tensors; FSDP sharding breaks the
+        # SwiGLU on >1 GPU. Exclude them (tiny, replicated per rank). FSDP does not move ignored
+        # modules, so place them on device/dtype first.
+        mem_ignored = []
+        if self.world_size > 1:
+            mem_ignored = [m for m in self.model.generator.modules()
+                           if type(m).__name__ == "VideoMemoryLayer"]
+            for m in mem_ignored:
+                m.to(device=self.device, dtype=self.dtype)
+
         self.model.generator = fsdp_wrap(
             self.model.generator,
             sharding_strategy=config.sharding_strategy,
             mixed_precision=config.mixed_precision,
-            wrap_strategy=config.generator_fsdp_wrap_strategy
+            wrap_strategy=config.generator_fsdp_wrap_strategy,
+            ignored_modules=mem_ignored or None
         )
```

**Explication :** les fast weights `w0/w1/w2` sont des tenseurs de **calcul** ; FSDP les sharderait et casserait le SwiGLU à >1 GPU. On les passe en `ignored_modules` (minuscules, répliqués par rank). FSDP ne déplaçant pas les modules ignorés, on les met d'abord sur le bon device/dtype. En mono-GPU, rien ne change.

**3) Le hook profiler VRAM :**

```diff
+    def _vram_profile_tick(self):
+        """VRAM profiler hook (off unless config.profile_vram > 0): reset the peak after a warmup,
+        then report peak allocated/reserved over profile_vram steps and hard-exit."""
+        n = getattr(self.config, "profile_vram", 0)
+        if not n:
+            return
+        warmup = getattr(self.config, "profile_vram_warmup", 3)
+        if self.step == warmup:
+            torch.cuda.reset_peak_memory_stats()
+        elif self.step >= warmup + n:
+            torch.cuda.synchronize()
+            alloc = torch.cuda.max_memory_allocated() / 2**30
+            reserved = torch.cuda.max_memory_reserved() / 2**30
+            if not dist.is_initialized() or dist.get_rank() == 0:
+                print(f"[VRAM_PROFILE] layers={getattr(self.config, 'model_kwargs', {}).get('memory_layers', [5])} "
+                      f"train_write={getattr(self.config, 'train_write', False)} "
+                      f"alloc_GiB={alloc:.2f} reserved_GiB={reserved:.2f}", flush=True)
+            if dist.is_initialized():
+                dist.barrier()
+            os._exit(0)
```

**Explication :** désactivé sauf si `profile_vram > 0`. Reset du pic après `profile_vram_warmup` steps, puis après `profile_vram` steps de plus il imprime le pic alloc/reserved (étiqueté avec les couches lues depuis `config.model_kwargs.memory_layers`) et fait `os._exit(0)`. C'est l'outil des tables VRAM.

**3bis) Appel du hook à chaque step :**

```diff
                     # Increase step count
                     self.step += 1
-                    
+                    self._vram_profile_tick()
+
```

**4) Logging mémoire vers TensorBoard (donc wandb) :**

```diff
                         third = b.numel() // 3
-                        print(f"mem stats step={self.step} gate={b[2*third:].abs().mean().item():.3e} "
-                              f"shift+scale={b[:2*third].abs().mean().item():.3e}")
+                        gate = b[2*third:].abs().mean().item()
+                        shift_scale = b[:2*third].abs().mean().item()
+                        print(f"mem stats step={self.step} gate={gate:.3e} shift+scale={shift_scale:.3e}")
+                        if hasattr(self, "writer"):
+                            self.writer.add_scalar("memory/gate", gate, self.step)
+                            self.writer.add_scalar("memory/shift_scale", shift_scale, self.step)
                         break
```

```diff
             if not found and do_log:
                 print(f"mem stats step={self.step} ttt allocated but no state")
+            if do_log and found and hasattr(self, "writer"):
+                self.writer.add_scalar("memory/drift_max", max_drift, self.step)
```

**Explication :** on extrait `gate` / `shift+scale` (même print) et on loggue `memory/gate`, `memory/shift_scale`, `memory/drift_max` : ça permet de **suivre dans wandb** si le gate s'ouvre et si l'état fast-weight vit (la drift).

---

## `utils/distributed.py` (+3) — paramètre `ignored_modules`

```diff
-def fsdp_wrap(module, sharding_strategy="full", ..., cpu_offload=False):
+def fsdp_wrap(module, sharding_strategy="full", ..., cpu_offload=False, ignored_modules=None):
```

```diff
         cpu_offload=CPUOffload(offload_params=cpu_offload),
+        ignored_modules=ignored_modules,  # excluded from sharding (params kept whole per rank)
         sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
```

**Explication :** `fsdp_wrap` gagne un paramètre `ignored_modules=None` transmis au constructeur FSDP. C'est ce qui permet l'exclusion de la mémoire faite dans `distillation.py`.

---

## Configs non trackés (à toi de décider)

| Fichier | Statut | Rôle |
|---|---|---|
| `configs/baseline.yaml` | untracked | A/B inférence : AF pur ; `memory_layers: []` (pas de branche mémoire, sinon le défaut `gate_init 0.1` réveillerait une mémoire non entraînée) |

> `configs/e0_cmp_memory750.yaml` a été supprimé (il pointait vers le ckpt pré-fix invalide). On créera `configs/memory.yaml` (symétrique de `baseline.yaml`, avec le `memory_layers` du run et le ckpt post-fix) quand le training aura produit un checkpoint. `tea_debug.log` (vide, déposé par une dépendance au runtime) est désormais gitignoré.

> Rappel : `reference/` (watcher, profiler, probe, ce doc) est privé/gitignoré, hors commit. Le watcher `gpu_ro_autorun.py` a été mis à jour pour écrire `model_kwargs.memory_layers` dans le YAML généré (au lieu de poser des variables d'env).
