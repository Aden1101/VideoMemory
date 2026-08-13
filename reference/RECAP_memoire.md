# Récap — mémoire TTT sur Anchor-Forcing (au 2026-06-08)

État de l'intégration d'une mémoire test-time-training (TTT / fast-weight, alignée sur Scal3R/LaCT)
sur le backbone Anchor-Forcing (Wan2.1-T2V-1.3B causal), et ce qu'il reste à faire.

---

## 1. Ce qui est FAIT

### Code mémoire (`wan/modules/video_memory.py`)
- Réécrit pour que les maths fast-weight soient **verbatim Scal3R** (`zju3dv/Scal3R`, LaCT) ;
  helpers restaurés à l'identique (`inv_softplus`, `silu_backprop`, `zeropower_via_newtonschulz5`,
  `CompatRMSNorm`), noms de variables de Scal3R.
- **6 déviations assumées** vs Scal3R : (1) inference-only→training, (2) machinerie VGGT/patch retirée,
  (3) threading causal `step`+`TTTBlockCache`, (4) `g_init=0.0` (warm-start inerte ; Scal3R 0.1),
  (5) `num_heads=1, inter_multi=4` (Scal3R 2/1), (6) fix dtype bf16 du chemin `lrs`.
- **Commité** par l'utilisateur : `a45cb9a "feat : finished the model of memory"`. L'en-tête verbeux
  des 6 points a été retiré au commit (préférence « moins de commentaires ») ; l'attribution Scal3R
  par fonction est conservée (10 mentions inline).

### Validation (verte de bout en bout)
- `tests/test_grad.py` : **2 passed** sur la version commitée.
- **Smoke FSDP complet** (`configs/ttt_check.yaml`, 52 steps, GPU unique) : gate monte `0 → 3.0e-4`,
  drift `‖w−w_init‖` accumule/reset par séquence, `assert_memory_drift` ne saute pas.
- **Checkpoint vérifié** : `generator_lora(600)` + `critic_lora(600)` + `generator_memory(12)`,
  **0 tenseur non-fini**, gate entraîné (biais max 1.07e-3 depuis init 0). Round-trip OK.
- À retenir : lancer le smoke avec `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (sinon OOM au
  step ~5). Les warnings `clone() OOM` au save sont **inoffensifs** (tenseurs backbone non sauvegardés
  en mode LoRA).

### Teacher 14B
- `Wan-AI/Wan2.1-T2V-14B` téléchargé (transformer seul, **57.15 Go**, intégrité vérifiée :
  6 shards + index + config, 1095 tenseurs).
- Stocké sur `/shared/ssd_14T_project/adenelmi/wan_models/Wan2.1-T2V-14B`, symlinké dans
  `wan_models/Wan2.1-T2V-14B`. (Le home `/shared/ssd_14T` est plein à 100 %.)
- ⚠️ **Ne tient pas sur 1 GPU** (cf. §4). Réservé au run multi-GPU.

### Config Phase 1 (`configs/ttt_phase1.yaml`) — créé, PAS encore lancé
- Mémoire-seule sur AF entraîné : `lora_ckpt: ckpt/final.pt`, `resume_step: False` (sinon démarre au
  step 3000 et sort), `lr: 0.0` (LoRA AF gelé), `lr_mem: 1e-4` (seule la mémoire apprend),
  `lr_critic: 2e-6`, teacher **1.3B**, logdir sur le disque projet.
- Validé (parse + clés correctes). Lancement (à faire) :
  ```
  CUDA_VISIBLE_DEVICES=GPU-<uuid> WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    <longlive>/python -m torch.distributed.run --nnodes 1 --nproc_per_node=1 --master_port 29520 \
    train.py --config_path configs/ttt_phase1.yaml --no-auto-resume --no-one-logger
  ```

---

## 2. Architecture : ce que Phase 1 entraîne (et n'entraîne PAS)

Le rollout alterne : passes de débruitage **bruitées = lecture seule** (`ttt_update=False`, avec
gradient) ; re-run **propre = écriture** (`ttt_update=True`, **sous `torch.no_grad()`** —
`pipeline/streaming_switch_training.py:244`).

| Entraîné en Phase 1 (`lr_mem`) | Gelé à l'init (write sous no_grad) |
|---|---|
| `proj` (gate + shift/scale + sortie) → **ouvre le gate** | `lrs` (learning-rate du write) |
| `o_norm`, partie `q` de `qkv`, `q_norm` | projections **k, v** ; dynamique d'update |
| fast-weights initiaux `w0/w1/w2` | |

→ **Phase 1 = on apprend à LIRE** une mémoire qui s'écrit avec une règle TTT fixe (projections k/v à
l'init). **Phase 2 = rendre le write différentiable** (apprendre `lrs`, k/v) — c'est là qu'est le vrai
gain, mais c'est du second ordre (BPTT à travers les updates Muon).

---

## 3. Où se trouve le « vrai » gain (jugement)

La mémoire ne peut améliorer qu'**une chose** : la cohérence/identité **sur long horizon** (là où l'AR
dérive). Classement :
1. **Phase 2 (write appris)** — cœur de la méthode TTT ; gain le plus probable. *(confiance haute)*
2. **Phase 1 (lecture seule)** — test de plomberie pas cher et sans risque (`g_init=0` → ne peut pas
   régresser). Go/no-go pour Phase 2. *(gain marquant peu probable seul)*
3. **Teacher 1.3B → 14B** — orthogonal : joue sur la qualité absolue, pas sur le delta mémoire. Même
   teacher pour comparer AF vs AF+mémoire.

Deux leviers ≥ au choix de phase :
- **Nombre de couches** : actuellement `MEMORY_LAYERS = {5}` = **1 bloc sur 30**. Multi-couches
  (`{5,11,17,23,29}`) sans doute le levier le plus visible.
- **Horizon d'éval** : évaluer **bien au-delà** de la fenêtre d'entraînement, sinon la baseline n'a pas
  encore dérivé et on mesure « aucun gain » à tort.

**Le vrai run de référence = write appris × multi-couches × horizon long × (teacher 14B).**

---

## 4. Plan de capacité (VRAM / GPU)

Ancré sur **1 point mesuré** (Phase 1, 1 couche, teacher 1.3B, streaming 60 = **47 Go** sur RTX 6000
Ada 48 Go) + **probe d'activation mesuré** (1.26 Go bf16 / chunk / couche pour le BPTT du write).

Rappel modèle mémoire FSDP : **les activations ne shardent PAS** (seuls poids/grads/optim shardent).
Pour des modèles 1.3B (params petits), **les ~47 Go sont ~75 % d'activations** → ajouter des cartes
48 Go ne réduit pas le plancher d'activations. C'est la taille de carte (48 vs 80 Go), pas le nombre,
qui décide.

### Phase 1 (lecture seule)
| Config | VRAM | GPU |
|---|---|---|
| 1 couche, teacher 1.3B *(mesuré)* | **47 Go** | **1× 48 Go** ✅ |
| 5 couches, teacher 1.3B | ~52 Go | 2× 48 Go (FSDP) ou 1× 80 Go |

### Phase 2 (write différentiable) — **dominé par la profondeur de BPTT**
BPTT plein = profondeur ~20 (streaming 60 / `num_frame_per_block` 3) → **impraticable** :
- 1 couche : ~**72 Go**  · 5 couches : ~**173 Go**

→ **Phase 2 oblige un BPTT TRONQUÉ.** À profondeur 2 :
| Config (BPTT tronqué prof. 2) | VRAM | GPU |
|---|---|---|
| 1 couche, teacher 1.3B | ~**50 Go** | 1× 80 Go (ou 2× 48 + coupes) |
| 5 couches, teacher 1.3B | ~**60 Go** | 1× 80 Go (ou 2× 48 + coupes) |
| 5 couches + 14B + horizon long (streaming 240) | classe `train_long` | **8× 80 Go** |

**RAM : jamais le goulot** — la machine a **1 To** (même le 14B cpu-offloadé à 57 Go passe large).

---

## 5. Contraintes d'infra (au moment du run)

8× RTX 6000 Ada (48 Go). État observé :
- GPU 0,1 : noemosca · GPU 2-5 : yl535 (100 % util) · GPU 7 : jeremymoebel (41 Go libre, 93 %).
- **GPU 6** : seul « libre » mais **inutilisable** — `torch.cuda` renvoie « No CUDA GPUs available »
  (util `[N/A]` = alloué à un autre namespace/container). Le gotcha noté.
- `import wan` **force l'init CUDA à l'import** (`wan/modules/t5.py:478`) → besoin d'un GPU utilisable
  même pour de petits scripts. Utiliser un GPU **par UUID** (les index échouent sur certaines cartes).

→ **Aucun GPU réellement disponible pour nous en ce moment** pour lancer la Phase 1.

---

## 6. Ce qu'il RESTE à faire (TODO priorisé)

1. **Lancer Phase 1 mono-couche** (`configs/ttt_phase1.yaml`) — **bloqué sur un GPU utilisable**.
   Dès qu'une carte se libère (ou avec accord pour partager une carte), lancer ~1000-3000 steps
   (~35 s/step ; save tous les 500). Surveiller gate ↑ / drift / `assert_memory_drift`.
2. **Évaluer long-horizon** : AF vanilla vs AF+mémoire sur identité/cohérence, **bien au-delà** de
   60 frames, même teacher.
3. **Rendre `MEMORY_LAYERS` pilotable depuis le YAML** (actuellement codé en dur `causal_model.py:717`)
   pour ablater 1 vs ~5 couches sans toucher au code.
4. **Phase 2 (write appris)** — implémenter le write différentiable avec **BPTT tronqué (prof. ~1-2)** ;
   gérer les caches KV/flow in-place (la troncature évite le BPTT profond qui casserait l'autograd).
   Besoin de **≥ 1× 80 Go** (ou 2× 48 Go + coupes). Le run de référence (5 couches + 14B + horizon
   long) = **nœud 8× 80 Go**.
5. **(Optionnel)** doc superviseur `reference/MEMORY_vs_scal3r.md` (6 justifications + pointeurs
   ligne-à-ligne Scal3R) ; committer `configs/ttt_phase1.yaml` (l'utilisateur committe lui-même).

---

## 7. Décisions ouvertes
- GPU 6 réellement utilisable un jour, ou définitivement réservé ? (sinon il faut attendre/négocier des
  GPU pour tout run).
- Premier run single-GPU : **1 couche + fenêtre pleine** (signal propre) **ou** ~3-5 couches + fenêtre
  rognée (plus de capacité, horizon plus court) ?
- Profondeur de troncature BPTT pour Phase 2 (1, 2, ou plus) — fixe directement la VRAM.
