# Gating adapté vidéo : exploration exhaustive + autocritique

But : avant de coder quoi que ce soit, explorer largement comment rendre le gating de sortie de
la mémoire TTT vraiment adapté à la vidéo causale, en partant de la gate temporelle puis en
ouvrant le champ, et en s'autocritiquant en boucle (proposition -> critique -> raffinement).

Statut : document de discussion (avec Aden d'abord, puis la superviseuse). Tout est slow-weight
(outer loop), donc découplé de problem B / du write-training.

---

## 0. Recadrage : le gate n'est qu'UN levier parmi trois

Avant de raffiner "sur quoi conditionner le gate", il faut voir que le gate pose **trois questions
orthogonales**, et que la littérature (et nous) confond souvent la première avec les deux autres :

- **Q1 -- Conditionnement** : de quoi dépend `g` ? (`ô` seul aujourd'hui ; temps, mouvement, ...)
- **Q2 -- Activation** : est-ce que `g` apprend à être non-nul tout court ?
- **Q3 -- Localisation** : le gate (la **sortie**) est-il seulement le bon endroit où intervenir,
  vs le **read** (q / rappel) ou le **write** (k,v / état) ?

**Fait empirique qui change la priorité :** après 805 steps, `gate ~6e-4` (il ne décolle pas),
alors que `drift_max` 57->27 (l'état fast-weight vit). Donc **le write écrit, mais le gate étouffe
la sortie**. Conclusion forte et contre-intuitive :

> Toute idée de conditionnement riche (Q1) est **second-ordre** tant que Q2 (activation) n'est pas
> réglée. Un gate conditionné finement sur le temps/mouvement/fréquence ne sert à rien s'il reste
> collé à zéro. **Q2 domine Q1.**

On garde ça en tête tout le long : on classera les idées en "ça aide le gate à s'ouvrir" vs "ça
conditionne mieux un gate déjà ouvert".

---

## 1. La gate temporelle (point de départ) : proposition -> critique -> raffinement

**Proposition (idée A).** Concaténer un embedding temporel `φ(t)` (t = index de frame absolu) à
`ô` avant `W_proj`. Le réseau apprend une rampe `g(t)` croissante : la mémoire monte en puissance
à mesure que la dérive s'installe. Do-no-harm préservé (zéro-init des nouvelles colonnes).

**Critique 1 (la plus sérieuse) -- les clips d'entraînement sont trop courts pour enseigner une
rampe long-horizon.** On entraîne sur des clips de ~10-15s (fenêtre `streaming_max_length`), mais
la dérive qu'on veut corriger apparaît à **120-240s**. Sur un clip court, le gate ne voit qu'une
**petite fenêtre de t** -> il ne peut pas apprendre une rampe qui ne se manifeste qu'à long
horizon. Une gate `g(t)` naïve risque de ne jamais voir le régime où elle devrait être grande.
- *Mitigation :* `t` doit être l'**absolu** (depuis le début de la vidéo), et l'entraînement doit
  **échantillonner des positions de départ variées** (clips pris à t=0, t=50, t=200...). Sinon la
  rampe est non-identifiable.

**Critique 2 -- t est un SURROGATE de "le contenu a dérivé".** Le réseau pourrait apprendre la
rampe non pas via `t` mais via le fait que `ô` *ressemble* à du contenu dégradé (les features
tardives sont différentes). Auquel cas conditionner sur `t` est redondant avec conditionner sur le
contenu. Plus profond : **ce qu'on veut, ce n'est pas "plus de mémoire quand t est grand", c'est
"plus de mémoire quand ça a dérivé".** `t` n'est qu'un proxy de la dérive.
- *Raffinement :* remplacer (ou doubler) `φ(t)` par un **proxy de dérive calculé en ligne**, qui
  marche même sur clips courts (voir idée C, version sûre). La gate temporelle devient un cas
  particulier d'une gate "drift-aware".

**Critique 3 -- la dérive n'est pas monotone.** Elle est souvent **par à-coups** (changements de
plan, mouvements rapides, switch de prompt), pas une montée lisse en `t`. Une rampe pure rate ces
événements.
- *Raffinement :* `g` devrait pouvoir **sauter** sur événement, pas seulement ramper. -> motiver
  un signal événementiel (mouvement, switch) en plus de `t`.

**Verdict gate temporelle :** idée juste mais **fragile seule** (clips courts) et **indirecte**
(t = proxy de dérive). À garder comme composante, mais le coeur devrait être un signal de dérive
calculable en ligne.

---

## 2. Catalogue élargi (chaque idée : mécanisme -> critique -> verdict)

### B. Sélectivité au mouvement -- concat `m = MLP(q_t - q_{t-1})`
- *Idée :* ancrer le stable (identité), lâcher le mouvement légitime.
- *Critique :* `Δq` **confond** mouvement caméra, mouvement objet et dérive d'identité -- le signe
  est ambigu (un grand `Δq` peut être du mouvement OU de la dérive). Et **le sujet bouge aussi**
  (un visage qui parle) -> "peu de mouvement = identité" est faux. Aux coupures de plan `Δq`
  explose sans que ce soit "du mouvement à laisser passer".
- *Verdict :* signal trop ambigu pour piloter directement le gate. **Rétrograder** ; éventuellement
  utile comme signal **spatial** (où est le mouvement) couplé au masque sujet, pas comme scalaire.

### C. Feedback de dérive -- `g = g_max·σ(g_raw + α‖ô‖)`
- *Idée :* injecter plus là où la mémoire "n'est pas d'accord".
- *Critique (grave) :* `‖ô‖` = magnitude du rappel, **pas** le désaccord. Et c'est une **boucle de
  feedback positive** : si la mémoire se trompe, on injecte PLUS de l'erreur -> divergence. Risqué.
- *Raffinement (version sûre) :* le bon signal de dérive n'est pas `‖ô‖` mais un **désaccord
  mesuré** entre le rappel et une **ancre d'identité** stable. Concrètement : garder une **ancre**
  (ex. moyenne EMA du read des premières frames, ou un "canonical read" sur une query d'identité),
  et conditionner le gate sur `‖ô - ancre‖` ou plutôt l'**accord** (gate up quand le rappel est
  *cohérent* avec l'ancre, gate down sinon -- l'inverse du feedback positif). Ça donne un proxy de
  dérive calculable **en ligne et sur clips courts** (résout la critique 1 de la gate temporelle).
- *Verdict :* la version `‖ô‖` est dangereuse ; la version **"accord avec une ancre d'identité"**
  est le **coeur prometteur** -- c'est la vraie "drift-awareness" que `t` ne fait qu'approximer.

### D. Prior fréquentiel -- l'identité vit en basse fréquence (diagnostic DCT)
- *Idée initiale :* gate différent par bande de fréquence.
- *Critique :* gater "par fréquence" demande une transformée spatiale (DCT/FFT) du grid de tokens
  dans la couche puis inverse -- ajout architectural réel, pas un simple concat, et casse la
  structure per-token.
- *Raffinement (plus propre) :* au lieu de gater par fréquence, **passe-bas sur `mem_out`** : une
  conv depthwise spatiale légère qui ne laisse passer que la composante **lisse** de la
  contribution mémoire. Prior structurel : *la mémoire corrige l'identité basse-fréquence, pas la
  texture/le détail haute-fréquence.* Cheap, branché direct sur ton diagnostic (identité = DC).
- *Verdict :* **promouvoir** la version passe-bas. Novel, image/vidéo-native, orthogonale au reste.

### E. Conditionnement sur le pas de débruitage (DMD 4-step) -- le plus diffusion-natif
- *Idée :* `g` dépend du **pas de denoising** (le générateur est un DMD 4-step). La mémoire lit sur
  les passes bruitées et écrit sur la passe propre ; l'identité/structure se forme à des niveaux de
  bruit différents. Conditionner `g` sur le timestep -> injecter au **bon stade** (ex. plus fort
  aux pas peu bruités où les détails d'identité se fixent).
- *Critique :* on ignore empiriquement si l'identité se forme tôt (layout grossier) ou tard
  (détails). À tester. Mais le coût est nul (le modèle a déjà l'embedding de timestep).
- *Verdict :* **promouvoir** -- c'est le levier le plus **novel** et le plus **vidéo-diffusion-
  natif** (orthogonal à l'axe dérive : ici on adapte au *process de génération*, pas au temps de la
  vidéo). Personne ne le fait (VGGT/LaCT ne sont pas des diffusions 4-step distillées).

### F. Gate spatial / région-sujet
- *Idée :* injecter sur la région du sujet (identité), pas le fond.
- *Critique :* demande un masque sujet **en ligne** ; le nôtre (ΔK prompt-switch) est offline. Et le
  gate est **déjà per-token**, donc il peut déjà être spatialement sélectif via `ô`.
- *Verdict :* largement **subsumé** par le gate per-token actuel. Rétrograder.

### G. Gate "post-switch" (frames-depuis-switch)
- *Idée :* re-stabiliser après un changement de prompt.
- *Critique :* niche (événement rare) ; ambigu (préserver l'identité vs laisser le nouveau prompt
  s'installer). Mais le switch est connu, et "frames-since-switch" est un signal événementiel propre
  qui complète `t`.
- *Verdict :* **secondaire** ; intéressant comme cas du signal événementiel de la critique 3.

### H. Gate piloté par l'incertitude du rappel
- *Critique :* la fast-weight n'a pas d'incertitude naturelle ; il faudrait une tête de variance /
  un ensemble -> machinerie spéculative.
- *Verdict :* **abandonner** pour l'instant.

---

## 3. Boucle d'autocritique transversale (au-dessus des idées individuelles)

**AC-1 : on optimise peut-être le mauvais levier (Q3).** Le gate ne contrôle que *combien* du
rappel on injecte. Si le **read** rappelle la mauvaise chose (mauvaise identité), aucun gate
n'aide. Or on n'a **jamais vérifié** que le read recalle l'identité (on sait juste que l'état
*dérive* = quelque chose est écrit). Donc : **avant de raffiner le gate, sonder le read** (les
feature-maps `mem_out` / le read pré-gate -- pile la directive "feature maps" de la superviseuse).
Si le read est inutile, tout ce doc est prématuré.

**AC-2 : Q2 (activation) reste la priorité 1.** Même avec le meilleur conditionnement, si le gate
n'apprend pas à s'ouvrir, rien ne change. Leviers d'activation (orthogonaux au conditionnement) :
- `g_init` petit positif (0.05) façon Scal3R -- force la mémoire dans la boucle dès le step 0
  (risque : dégradation initiale par le read aléatoire ; borné car normalisé).
- **gate-floor schedulé à l'entraînement** : un plancher `g_floor` qui rampe de 0 vers une cible
  sur les steps de training -> force l'usage de la mémoire, le reste s'adapte, puis on relâche le
  plancher. Do-no-harm au step 0, mais pas d'extinction.
- entraîner le **write** (problem B) pour que le rappel soit *utile*, rendant le gate "worth
  opening". (Hors scope slow-weight, mais c'est peut-être LA vraie cause.)

**AC-3 : risque d'over-engineering avec un signal faible.** Empiler t + mouvement + freq + denoise
+ ancre, c'est beaucoup de params nourris par un signal d'apprentissage déjà faible (gate ~0). Le
réseau pourrait ne rien en faire, ou pire, sur-ajuster le bruit. **Principe de parcimonie :** une
idée à la fois, ablation propre, garder ce qui bouge la dérive ArcFace.

**AC-4 : identifiabilité en self-rollout.** En self-rollout, `t`, le contenu dégradé, et la dérive
sont **tous corrélés**. Le réseau peut apprendre n'importe lequel comme proxy des autres -> on ne
saura pas *pourquoi* ça marche. Pour l'interprétabilité (et la directive feature-maps), préférer un
signal **explicite et mesurable** (l'ancre de dérive AC-1/C) plutôt qu'un `t` opaque.

---

## 4. Synthèse raffinée (ce qui survit, dans l'ordre)

Après la boucle, la priorité n'est PAS "ajouter `φ(t)`". C'est :

1. **(Q3) Sonder le read d'abord.** Feature-maps de `mem_out` / read pré-gate (réutilise kv_probe).
   Le read recalle-t-il l'identité ? Si non, on corrige le read/write avant le gate. *Préalable.*

2. **(Q2) Régler l'activation.** Tester `g_init=0.05` et/ou un **gate-floor schedulé**. Sans ça,
   tout conditionnement est moot. *Priorité 1.*

3. **(Q1) Conditionnement, les 3 survivants, par ordre de promesse/propreté :**
   - **Drift-aware via ancre d'identité** (C raffiné) : `g` fonction de l'**accord** entre le read
     courant et une ancre EMA des premières frames. C'est la vraie cible que `φ(t)` approxime, et
     ça marche **sur clips courts**. -> le coeur.
   - **Conditionnement sur le pas de denoising** (E) : le plus novel, coût nul, diffusion-natif.
   - **Prior passe-bas sur `mem_out`** (D raffiné) : structurel, branché sur le diagnostic DCT.
   - *(gate temporelle `φ(t)` (A) : la garder comme entrée auxiliaire bon-marché, mais PAS comme
     mécanisme principal -- elle est dominée par l'ancre de dérive.)*

4. **Rétrogradés/abandonnés :** mouvement scalaire (B, ambigu), feedback `‖ô‖` (C naïf, instable),
   gate spatial (F, déjà subsumé), incertitude (H, spéculatif).

**La phrase à retenir pour la superviseuse :** Scal3R a été malin en conditionnant le gate sur ce
qui compte pour LEUR problème (la géométrie 3D). Pour NOTRE problème (dérive d'identité causale),
le bon conditionnement n'est ni `ô` seul (VGGT) ni même le temps `t`, c'est **le désaccord du
rappel avec une ancre d'identité** -- avec deux compléments propres et nouveaux : le **pas de
denoising** (diffusion-natif) et un **prior passe-bas** (l'identité est basse-fréquence). Mais tout
ça est second-ordre tant qu'on n'a pas (a) vérifié que le read est utile et (b) fait s'ouvrir le
gate.

---

## 5. Plan de test (do-no-harm, slow-weight, ablation propre)

| # | Test | Ce que ça répond | Coût |
|---|------|------------------|------|
| 0 | feature-maps read pré-gate (kv_probe) | le read recalle-t-il l'identité ? (Q3) | nul (offline) |
| 1 | `g_init=0.05` vs `0.0` | le gate s'ouvre-t-il / la dérive baisse-t-elle ? (Q2) | 1 run |
| 2 | gate-floor schedulé | forcer l'usage aide-t-il ? (Q2) | 1 run |
| 3 | drift-aware (ancre EMA) vs baseline | la vraie drift-awareness aide-t-elle ? (Q1) | 1 run |
| 4 | + conditionnement denoising-step | le levier diffusion-natif aide-t-il ? (Q1) | 1 run |
| 5 | + prior passe-bas mem_out | le prior basse-fréquence aide-t-il ? (Q1) | 1 run |

Métrique commune : cohérence d'identité (ArcFace) à 120-240s + diagnostics KV-probe. Chaque test
est do-no-harm à l'init et n'a aucun lien avec problem B.

## 6. Questions ouvertes pour la superviseuse

1. Sonde-t-on le read d'abord (AC-1), ou assume-t-on qu'il est utile et on attaque le gate ?
2. Accepte-t-on de casser le do-no-harm pour l'activation (g_init>0 / gate-floor), avec le risque
   de dégradation initiale ?
3. L'axe le plus intéressant pour elle : l'ancre de dérive (le coeur), ou le pas de denoising (le
   plus novel) ?

---
---

# Partie 2 -- approfondissement (2e passe critique)

## 7. Théorie : à quoi ressemble le gate optimal ?

Prenons le modèle d'erreur AR de la §2 et rendons-le décisionnel. Erreur d'identité `e_t`, gate
`g_t`, et un point qu'on avait négligé : **injecter coûte du bruit**. Le rappel n'est pas parfait,
donc `mem_out = g_t·(recall)` ajoute un bruit `∝ g_t·σ_r` (σ_r = imperfection du rappel) :

```
e_{t+1} = (1+ρ) e_t - κ g_t e_t + g_t σ_r ξ_t          (ξ_t bruit unitaire)
```

On veut minimiser l'erreur long-horizon `E[e_t^2]`. Deux forces opposées dans `g_t` :
- **réduction de biais** : le terme `-κ g_t e_t` corrige la dérive (veut `g` grand),
- **ajout de variance** : le terme `g_t σ_r ξ_t` injecte du bruit (veut `g` petit).

Minimiser `E[e_{t+1}^2]` en `g_t` donne (en première approx, espérance sur ξ) :

```
g_t*  ≈  κ e_t / (κ² e_t² + σ_r²) · (qqch)   ~  proportionnel à e_t quand e_t petit,
                                                puis saturant/décroissant quand e_t grand.
```

Trois enseignements concrets, pas évidents :

1. **`g*` doit suivre l'erreur `e_t`** (contrôle proportionnel). Un gate **constant** est sous-
   optimal par construction. -> justifie théoriquement le drift-aware, PAS la rampe temporelle (qui
   suit `t`, pas `e`).
2. **`g*` ne doit PAS exploser quand l'erreur est grande** : à cause de la variance `σ_r`, au-delà
   d'un seuil, injecter plus AJOUTE plus de bruit que ça ne corrige. -> le gate doit **saturer**
   (d'où la version bornée `σ`, §3-C raffiné), et ce d'autant plus que le rappel est imparfait
   (`σ_r` grand). C'est exactement notre cas (mémoire à peine entraînée -> `σ_r` énorme).
3. **Condition de stabilité** : pour empêcher `e` de diverger, il faut `κ g_t > ρ` dans le régime
   linéaire. Si le rappel est mauvais (`κ` petit, `σ_r` grand), **aucun `g` ne stabilise** -> on
   ne peut pas gater notre sortie d'un mauvais read. Re-converge vers Q3 : régler le read d'abord.

Caveat honnête : modèle-jouet scalaire, la réalité est haute-dim et non-linéaire. C'est un guide
d'intuition (le gate doit tracker l'erreur et saturer), pas une formule à implémenter.

## 8. L'ancre d'identité, précisément (et ses pièges)

Le coeur = `g` fonction du désaccord `d = ‖ô - ancre‖`. Mais "ancre" est sous-spécifié. Options :

- **EMA du read des premières frames** : simple, mais les premières frames sont elles-mêmes
  imparfaites (la génération n'est pas parfaite dès la frame 1) -> ancre bruitée.
- **Query d'identité canonique** : un `q_id` appris, fixe, dont le read `f_W(q_id)` = "ce que la
  mémoire pense être l'identité". Plus propre, mais une tête de plus à apprendre.
- **Registre d'identité dédié** : un slot séparé que la mémoire maintient (pas la fast-weight
  générale). Plus expressif, plus lourd.
- **Embedding du prompt texte** : décrit l'identité ("a woman with red scarf"), mais **trop
  grossier** (ne capture pas l'identité d'instance, juste la catégorie).

**Le piège majeur (et son lien avec le switch) :** si le sujet **change légitimement** (switch de
prompt vers un autre sujet, coupure de plan), l'ancre devient **périmée** et le gate **combat le
changement légitime**. -> il faut **réinitialiser l'ancre sur événement** (switch, scene-cut). Ça
**relie l'idée "frames-since-switch" (E) au coeur** : le switch n'est pas un signal de gate, c'est
le moment où on **reset l'ancre**. Joli : E n'était pas un mécanisme, c'est le déclencheur du reset.

**Autres pièges :** vidéos **multi-sujets** ou **sans sujet clair** (paysage) -> l'ancre est mal
définie ; le gate drift-aware doit alors se réduire au gate de base (ne pas nuire). Et **chicken-
egg** : ancrer l'identité suppose de connaître l'identité, qui suppose de bonnes premières frames.

## 9. Q3 vraiment creusé : OÙ naît la dérive ? (4 hypothèses)

On a dit "sonder le read". Plus précisément, la dérive peut naître à **quatre** endroits, et le
gate n'en répare qu'**un** :

- **(a) Read** : la mémoire rappelle la mauvaise chose (le `q` ne retrouve pas l'identité stockée).
- **(b) Write** : la mémoire stocke la mauvaise chose (k/v n'encodent pas l'identité, ou l'état se
  fait écraser). On sait que l'état *bouge* (drift 57->27), pas qu'il stocke l'**identité**.
- **(c) Gate** : la mémoire rappelle la bonne chose mais le gate ne l'injecte pas. **Seul cas que
  notre travail répare.**
- **(d) Backbone / éviction KV** : la dérive naît dans le backbone lui-même -- le cache roulant
  (sink 3 + local 9) **oublie** l'identité des premières frames. C'est précisément le trou que la
  mémoire est censée combler.

**Diagnostic décisif (offline, gratuit) :** les feature-maps du **read pré-gate** `ô`.
- Si `ô` contient déjà de la structure d'identité (le sujet apparaît) -> on est en **(c)**, le gate
  est le bon levier, tout ce doc s'applique.
- Si `ô` est du bruit -> on est en **(a)/(b)**, et raffiner le gate est **prématuré**.

**Hypothèse (d), sous-estimée et importante :** même si read+gate sont bons, la mémoire est au
**Layer 5**, en parallèle de la self-attn, **ajoutée au résidu**. La dégradation d'identité se
**recompose à travers les couches 6->30**. Une correction unique au L5 peut être **lavée** en
aval. Ça motive :
- une mémoire en **couche tardive** (proche de la sortie, plus dure à laver), ou
- **multi-couche** (corriger à plusieurs profondeurs), ou
- injecter dans un endroit **persistant** (les sink tokens, §11) plutôt que le résidu transitoire.

## 10. Le placement en couche (où ancrer l'identité ?)

- Couches **précoces** : sémantique grossière, layout. Couches **tardives** : détails fins,
  texture, identité-de-détail.
- Le diagnostic DCT (identité = basse-fréquence) suggère identité = **grossière** -> couche
  précoce/moyenne + passe-bas. Cohérent avec le prior passe-bas.
- **MAIS tension :** "c'est un visage" est basse-fréquence ; "c'est CE visage précis" (identité
  d'instance) est peut-être plus **haute-fréquence** -> couche plus tardive. À **sonder par
  couche** (le profiling VRAM nous donnera déjà le coût 1/2/3 couches).
- Donc le choix de couche n'est pas qu'une question de VRAM, c'est **où vit l'identité**.

## 11. Remettre en cause le gate additif lui-même (mécanismes d'injection)

On a tout raisonné en supposant "mem_out **ajouté** au résidu, **gaté**". Challengeons la prémisse :

- **(i) Additif-gaté** (actuel) : simple, transitoire (lavé en aval, cf. (d)).
- **(ii) FiLM** : moduler les features du backbone (le scale/shift AdaLN le fait déjà un peu).
- **(iii) Cross-attention vers une banque mémoire** : les tokens du backbone *attendent* la
  mémoire. Plus expressif, mais gros changement archi.
- **(iv) Mémoire-augmente-le-SINK** (novel, et ça colle à notre archi) : au lieu d'ajouter au
  résidu, **écrire l'identité dans les sink tokens** (les 3 tokens toujours-attendus du cache
  roulant). Le sink est l'ancre permanente que toutes les frames futures voient -> injection
  **durable** par construction, contrairement au résidu transitoire. Ça attaque directement
  l'hypothèse (d). À mon sens l'idée la plus intéressante de cette 2e passe.
- **(v) Guidance-style (CFG-like)** : utiliser la mémoire pour former une **direction de guidage**
  (pousser la génération vers l'identité rappelée), comme la classifier-free guidance. Diffusion-
  natif, et ça sépare "de combien on pousse" (échelle de guidage = un gate scalaire) de "vers quoi"
  (la mémoire).

-> Le gate (combien injecter) reste pertinent dans **tous** ces mécanismes, mais **où** et
**comment** on injecte est un axe de conception qu'on n'avait pas ouvert. Le **sink-augmenté (iv)**
mérite son propre creusage.

## 12. Positionnement (pour situer face à la littérature)

- L'AdaLN scale/shift = **FiLM** (feature-wise linear modulation, Perez 2018).
- Le drift-aware = forme de **self-conditioning / anchored generation**.
- Le pas-de-denoising = **conditionnement par niveau de bruit** (standard en diffusion, mais pas
  pour une *mémoire* TTT).
- La vue contrôle = la mémoire comme **contrôleur par feedback** stabilisant l'identité (g* ∝ erreur
  = contrôle proportionnel, §7).
- Le sink-augmenté = proche des **register tokens** / **attention sinks** (Xiao 2023) mais utilisés
  comme **store d'identité écrit par la mémoire**, ce qui semble neuf.

## 13. Ordre des expériences sous pénurie GPU (réaliste)

1. **Probe read pré-gate (offline, 0 GPU-train)** : tranche Q3 (a/b/c) -- LE préalable, et c'est la
   directive feature-maps. À faire dès qu'un GPU d'inférence se libère.
2. **Tests d'activation config-only** (`g_init`, gate-floor) : pas de nouveau code, 1 run chacun.
3. **Drift-aware (ancre EMA + reset au switch)** : petit code, 1 run.
4. **Pas-de-denoising** + **passe-bas** : petits, 1 run chacun.
5. **Sink-augmenté (iv)** : creusage séparé, plus gros -- seulement si la probe dit que (d) domine.

Tout est do-no-harm / slow-weight (sauf éventuellement le sink) -> aucun lien problem B.

## 14. Autocritique de cette 2e passe

- **Sur-théorisation ?** Le modèle d'erreur scalaire est un jouet ; les conclusions (tracker
  l'erreur, saturer) sont des intuitions, pas des prédictions. Assumé.
- **Assume-t-on que la dérive d'identité EST le problème ?** Peut-être que le vrai échec à 240s est
  autre (incohérence de mouvement, dérive de prompt-following, sur-lissage). -> à vérifier sur les
  **vidéos baseline 240s** qu'Aden génère + la probe, avant de tout miser sur l'identité.
- **L'ancre a un chicken-egg** non résolu (bonnes premières frames requises). Robustesse à creuser.
- **J'ai peut-être trop élargi** (sink, guidance, multi-couche) alors que Q2 (activation) reste le
  blocage n°1. Discipline : la 2e passe ouvre des options, mais l'ordre §13 garde Q2/Q3 devant.

## 15. Ce qui a vraiment changé entre passe 1 et passe 2

- Passe 1 : "le bon conditionnement = drift-aware, pas temps". 
- Passe 2 ajoute : (a) **la théorie** dit que g doit *tracker l'erreur ET saturer* (le rappel
  imparfait borne le gain utile) ; (b) **l'ancre se reset au switch** (E devient le déclencheur,
  pas un mécanisme) ; (c) **la dérive peut naître ailleurs que dans le gate** (hyp. d = éviction
  KV), ce qui motive le **placement en couche** et surtout le **sink-augmenté** ; (d) le **mécanisme
  d'injection** (pas que "combien", mais "où") est un axe neuf, avec le sink comme candidat fort.
- La hiérarchie finale : **Q3 (où naît la dérive, via la probe) -> Q2 (activation) -> Q1
  (drift-aware + denoising + passe-bas) -> mécanisme (sink-augmenté si (d) domine)**.
