# Un gating de sortie adapté à la vidéo pour la mémoire TTT

## 1. Contexte et motivation

Notre mémoire TTT injecte sa lecture dans le backbone via un **gate AdaLN** emprunté tel quel à
la lignée Scal3R / VGGT :

```text
o   = f_W(q)                      # lecture SwiGLU, par token, head_dim = dim = d = 1536
ô   = RMSNorm(o)                  # o_norm
[s, c, g] = ô · W_p + b_p         # projection AdaLN, W_p dans R^{d x 3d}  (_project_output)
mem_out   = g ⊙ ( LN(ô) ⊙ (1 + c) + s )      # LN = a_norm (LayerNorm, sans affine)
x        := x + mem_out
```

avec le warm start do-no-harm `W_p = 0, b_p = 0` (gate bias `g_init = 0`), donc `g ≡ 0` et
`mem_out ≡ 0` au step 0.

Ce design est **malin pour le problème pour lequel il a été conçu** (Scal3R / VGGT = reconstruction
3D bidirectionnelle) : l'affine scale/shift, normalisé par LN, donne une modulation géométrique
riche et stable, auto-conditionnée sur les features rappelées. LaCT, lui, n'utilise qu'un gate
scalaire per-head SiLU `silu(ttt_scale_proj(x)) · o_norm(f_W(q))`, sans scale/shift.

**Le mismatch pour nous.** `(s, c, g)` ne dépendent que de `ô` au token courant. Aucune notion de
(i) **position temporelle** `t`, (ii) **mouvement vs stable**, ni (iii) **dérive**. C'est l'héritage
VGGT : la reconstruction 3D est symétrique entre les vues, il n'y a pas d'axe temps causal. Notre
problème est l'inverse : la **dérive d'identité qui croît avec `t`** en génération causale AR. Le
gating est conditionné sur la mauvaise chose.

Le but de cette note : transposer l'idée "maline pour leur problème" sur **notre** axe (temps /
dérive), pour obtenir un gating vidéo-natif. Il est **slow-weight uniquement** (outer loop, comme
le gate actuel), donc totalement découplé du problème d'entraînement du write-path (problem B) : il
peut être entraîné et ablaté sans aucun des soucis de write-graph / régression VRAM.

## 2. Pourquoi un gate croissant dans le temps devrait aider (accumulation d'erreur AR)

Modélisons l'erreur d'identité par frame comme `e_t`. En génération AR, chaque chunk se conditionne
sur les chunks précédents (déjà imparfaits), donc les erreurs se composent :

```text
e_{t+1} = (1 + ρ) e_t - κ·g_t·e_t + η_t
```

où `ρ ≥ 0` est le taux de composition du backbone non corrigé (la dérive), `g_t` la force du gate
(la correction de la mémoire), `κ > 0` l'efficacité de la correction, `η_t` le bruit d'injection.

- Un gate **constant** `g_t = g` injecte la même correction à chaque `t`. Tôt (petit `e_t`) il
  sur-corrige (coût / bruit pour peu d'erreur) ; tard (grand `e_t`) il sous-corrige, donc l'erreur
  peut quand même diverger si `(1 + ρ - κg) > 1` tôt, tant que `g` est gardé petit pour éviter la
  sur-correction initiale. Un seul `g` global ne peut pas être à la fois doux tôt et fort tard.
- Un gate **croissant dans le temps** `g(t)` aligne la correction sur l'erreur accumulée : petit
  quand `e_t` est petit, grand quand `e_t` est grand. C'est l'intuition du contrôle proportionnel
  (correction ∝ erreur). Ça garde `e_t` borné sans payer de sur-correction tôt.

C'est exactement le degré de liberté qu'un gate 3D bidirectionnel n'a aucune raison d'avoir, et dont
un gate vidéo causal a besoin.

## 3. Formalisation

Soit `t` = index de frame absolu du token (disponible dans le code via
`current_start_frame = current_start // frame_seqlen`). Soit `φ(t) ∈ R^{d_t}` un embedding
temporel sinusoïdal (même famille que l'embedding de timestep de diffusion déjà construit par le
modèle).

### 3.1 Version minimale : AdaLN temporel (do-no-harm préservé)

```text
ψ_t      = [ ô ‖ φ(t) ]                 # concat, dans R^{d + d_t}
[s, c, g] = ψ_t · W_p' + b_p'           # W_p' dans R^{(d + d_t) x 3d}
mem_out   = g ⊙ ( LN(ô) ⊙ (1 + c) + s )
```

Le do-no-harm est intact : on zéro-init `W_p'` sur **les deux** blocs (le bloc `ô` ET le bloc
`φ(t)`), gate bias `= 0` -> `g ≡ 0` au step 0. L'entraînement est ensuite libre d'apprendre une
rampe croissante `g(t)`. C'est l'analogue direct du scale/shift de Scal3R, conditionné sur **notre**
axe (le temps).

### 3.2 Extension B : sélectivité au mouvement / à la stabilité

La mémoire devrait ancrer le contenu stable (identité) et ne pas combattre le mouvement légitime.
Un signal cheap et calculable est le changement temporel local de la query :

```text
δ_{t,p} = q_{t,p} - q_{t-1,p}            # p = index de token spatial
m_{t,p} = MLP_petit(δ_{t,p}) ∈ R^{d_m}
ψ_{t,p} = [ ô_{t,p} ‖ φ(t) ‖ m_{t,p} ]
```

Nécessite de cacher la dernière frame de `q` du chunk précédent (un petit buffer, comme le KV
cache). Là où `‖δ‖` est grand (mouvement) le réseau peut apprendre à fermer le gate ; là où c'est
stable (sujet) à l'ouvrir. Ça rejoint le masque sujet-sélectif existant (prompt-switch `‖ΔK‖`
bottom 25%).

### 3.3 Extension C : feedback de dérive, gate borné (optionnel, plus risqué)

Injecter plus là où le rappel est fort (la mémoire "n'est pas d'accord" avec le contenu courant).
On rend le gate croissant dans la magnitude du rappel, et borné :

```text
g_{t,p} = g_max · σ( g_raw_{t,p} + α · ‖ô_{t,p}‖_2 ),   α ≥ 0
```

Avantage vs le gate linéaire actuel : borné (pas d'explosion). Init do-no-harm : gate bias -> −10
(`σ(−10) ≈ 0`) et `α = 0`. Caveat : `α > 0` est une boucle fermée (injecte -> change le contenu ->
…) ; à introduire doucement et à clipper. À garder en option.

### 3.4 Forme unifiée

```text
mem_out_{t,p} = g_{t,p} ⊙ ( LN(ô_{t,p}) ⊙ (1 + c_{t,p}) + s_{t,p} )
[s, c, g_raw] = [ ô ‖ φ(t) ‖ m ] · W_p' + b_p'
g            = g_max · σ( g_raw + α‖ô‖ )
```

Mettre `α = 0` et `d_m = 0` redonne la version temporelle minimale (§3.1) ; mettre aussi `d_t = 0`
redonne le gate baseline actuel.

## 4. Points de branchement dans le code

| Pièce | Fichier / endroit | Changement |
| --- | --- | --- |
| passer `t` à la mémoire | `causal_model.py` (a `current_start_frame`, L217) -> `memory.step(x, t=…)` | threader le scalaire `t` |
| `φ(t)` | `video_memory.py` `__init__` | embedding sinusoïdal (réutiliser celui de diffusion) |
| concat + projection | `_project_output` (L294-296) | `proj` : `(d+d_t+d_m) → 3d`, entrée = ψ |
| `δ` mouvement (opt. B) | `step()` | cacher la dernière frame `q` du chunk précédent |
| init do-no-harm | `_init_modulation_params` + `reset_parameters` | zéro sur les nouvelles colonnes (gate bias 0, ou −10 pour le σ-gate) |

## 5. Coût et risque

- **Slow-weight uniquement** (outer loop, comme le gate actuel). Aucune interaction avec le
  write-path / problem B / la régression VRAM. Entraînable et ablatable sur le setup read-only
  existant.
- Params ajoutés : `~(d_t + d_m) × 3d`, quelques M, négligeable.
- Le risque principal est dans l'extension C (boucle de feedback) ; §3.1 et §3.2 sont à faible
  risque.

## 6. Plan d'ablation

Même protocole d'entraînement, on ne fait varier que le gate :

1. **baseline** -- gate actuel (pas de `t`, pas de mouvement).
2. **+ temporel** (§3.1) -- une rampe apprise `g(t)` réduit-elle la dérive à long horizon ?
3. **+ mouvement** (§3.2) -- la sélectivité de stabilité aide-t-elle identité vs mouvement ?
4. **+ feedback** (§3.3) -- optionnel, seulement si 2-3 sont positifs.

Métrique : cohérence d'identité (ArcFace) au fil de la vidéo à 120-240s, plus les diagnostics de
dérive KV-probe existants. La courbe apprise `g(t)` est elle-même un résultat interprétable
(monte-t-elle ?).

## 7. Relation avec l'état de l'art

- **Scal3R / VGGT** -- origine de l'AdaLN scale/shift/gate ; conditionné sur les features rappelées,
  sans axe temps (3D bidirectionnel). On garde la forme affine, on ajoute le conditionnement
  temps/mouvement.
- **LaCT** ("Test-Time Training Done Right") -- gate scalaire per-head SiLU seulement, pas de
  scale/shift ; pas non plus de conditionnement temporel du gate.
- **Nous (proposé)** -- AdaLN affine comme Scal3R, mais conditionné sur `(ô, t, mouvement)` pour
  cibler la dérive d'identité en vidéo causale plutôt que la géométrie 3D.
