# Le module mémoire MAG — petit cours

*Une mémoire à poids rapides (fast weights / TTT) greffée sur Wan2.1-T2V, au bloc 5, baseline Anchor Forcing.*

L'idée tient en une phrase, et tout le reste n'est que cette phrase habillée :

> **La mémoire, c'est un petit réseau de neurones (un MLP *gaté*). Ses poids `(w0, w1, w2)` *sont* le contenu mémorisé. Lire = faire un forward pass dedans. Écrire = bouger ses poids d'un cran pour qu'il réponde `v` quand on lui présente `k`.**

On va construire ça en trois temps : d'abord le squelette (mémoire associative linéaire), puis le module réel qui enrichit chaque pièce du squelette, puis la dimension temporelle (l'ordre des opérations et l'entraînement).

---

## Partie I — Le squelette : une mémoire associative

### I.1 La mémoire associative la plus simple

Oublie le SwiGLU une minute. Imagine une **seule** matrice $W \in \mathbb{R}^{d\times d}$ qui sert de mémoire. On définit deux opérations :

- **Lire** une question $q$ : $\quad o = W q$.
- **Écrire** l'association « $k \to v$ » : $\quad W \leftarrow W + v\,k^\top$.

Le terme $v\,k^\top$ est un **produit extérieur** (une matrice de rang 1). C'est la **règle de Hebb** : « corréler l'entrée $k$ avec la sortie désirée $v$ ».

### I.2 Pourquoi ça marche (toy à $d=2$)

Partons de $W = 0$. Rangeons « $k_1=\binom{1}{0} \to v_1=\binom{0}{1}$ » :

$$W \leftarrow 0 + v_1 k_1^\top = \begin{pmatrix}0&0\\1&0\end{pmatrix}.$$

Relisons avec $q = k_1$ :

$$W q = \begin{pmatrix}0&0\\1&0\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}0\\1\end{pmatrix} = v_1.\;\checkmark$$

La mémoire a **ressorti $v_1$**. Rangeons maintenant « $k_2=\binom{0}{1}\to v_2=\binom{1}{1}$ » :

$$W \leftarrow \begin{pmatrix}0&0\\1&0\end{pmatrix} + v_2 k_2^\top = \begin{pmatrix}0&1\\1&1\end{pmatrix}.$$

Relire $k_2$ donne $v_2$, et relire $k_1$ donne **toujours** $v_1$. Pas d'interférence ici, parce que $k_1 \perp k_2$ : en général $W q = \sum_i v_i (k_i^\top q)$, donc la lecture est propre quand les clés sont orthogonales, et il y a du *crosstalk* quand elles se ressemblent. (D'où l'intérêt du `q_norm`/`k_norm` plus loin : garder les clés bien réparties.)

### I.3 Le pont vers TTT : écrire = monter un gradient

Voici l'observation qui fait tout basculer. Définissons un **score d'alignement**

$$s = \langle Wk,\, v\rangle = v^\top W k.$$

Son gradient par rapport à $W$ est exactement

$$\nabla_W\, s = v\,k^\top.$$

Autrement dit : **la règle de Hebb $W \mathrel{+}= v k^\top$ est *un pas de montée de gradient* sur l'alignement $\langle Wk, v\rangle$.** Écrire « $k\to v$ », c'est pousser $W$ dans la direction qui rend « la réponse à $k$ » la plus alignée possible avec $v$.

C'est ça, le **test-time training** (TTT) : au lieu d'une règle d'écriture inventée à la main, on pose un *objectif* (« lire $k$ doit donner $v$ ») et on fait **un pas de gradient** dessus, pendant le forward, sur les poids de la mémoire. Le module ne fait rien d'autre — il généralise juste chaque pièce de ce squelette.

---

## Partie II — Le module réel = le squelette, enrichi

Trois pièces du squelette sont remplacées par des versions plus musclées. Dimensions concrètes : $d = 1536$, couche cachée $h = 4d = 6144$, et les **trois** matrices de mémoire

$$w_0, w_2 \in \mathbb{R}^{d\times h}, \qquad w_1 \in \mathbb{R}^{h\times d}.$$

D'abord, les projections (communes à la lecture et à l'écriture), pour chaque token de $x$ :

$$[q, k, v] = x\,W_{qkv} + b, \qquad lr = \operatorname{softplus}(x\,W_{lr} + c)\ \in \mathbb{R}^3.$$

Puis $q$ et $k$ passent **chacun par une `RMSNorm` dédiée** (les `q_norm` / `k_norm` annoncés en I.2 — garder questions et clés à une échelle stable) ; $v$, lui, n'est **pas** normalisé :

$$q \leftarrow \operatorname{RMSNorm}(q)\ \ (\texttt{q\_norm}), \qquad k \leftarrow \operatorname{RMSNorm}(k)\ \ (\texttt{k\_norm}).$$

C'est cette $q$ (resp. $k$) **normalisée** qui entre ensuite dans la lecture $f_W(q)$ (resp. dans l'écriture). *(La 3ᵉ normalisation, `o_norm`, agit plus loin — sur la **sortie** $o$, voir II.2.)*

Rôles : $q$ = la **question**, $k$ = l'**étiquette** sous laquelle on range, $v$ = le **contenu** à ranger, et $lr_0,lr_1,lr_2$ = **la force d'écriture** (un scalaire par token, propre à chaque matrice). Le terme constant $c = \operatorname{inv\_softplus}(\text{base\_lr})$ est **ajouté avant le softplus** (ce n'est pas un biais entraîné qu'on ajuste, c'est une constante fixe). Du coup $lr \approx \text{base\_lr}=10^{-2}$ au départ — *exactement* `base_lr` uniquement là où la part linéaire $x W_{lr}$ s'annule, ce qui n'est pas le cas à l'init (poids aléatoires).

### II.1 La mémoire est un petit réseau *gaté* (SwiGLU)

`(w0, w1, w2)` sont les **couches d'un petit réseau de neurones**, et lire la mémoire = un **forward pass** dedans :

$$o = f_W(q) = \big(\,\operatorname{silu}(q\,w_0) \odot (q\,w_2)\,\big)\,w_1.$$

**Attention au piège : ce n'est pas un MLP « normal ».** Un MLP standard est à **flux unique** — une montée, une activation, une descente, **deux** matrices :

$$y = \operatorname{act}(x\,W_1)\,W_2 \qquad \text{(2 matrices, 1 flux)}.$$

Ici on a un MLP **gaté**, de la famille **GLU** (et SwiGLU = la variante avec SiLU). Sa signature, ce sont **deux flux parallèles** fusionnés par un produit terme à terme — d'où **trois** matrices :

$$y = \big(\operatorname{silu}(x\,w_0) \odot (x\,w_2)\big)\,w_1 \qquad \text{(3 matrices, 2 flux)}.$$

Les rôles, terme par terme :

- $a = q\,w_0 \in \mathbb{R}^h$ : projection vers le haut « **porte** » ($w_0$), passée dans `silu`.
- $m = q\,w_2 \in \mathbb{R}^h$ : projection vers le haut « **contenu** » ($w_2$).
- $\operatorname{silu}(a) \odot m$ : la porte **multiplie** le contenu — « laisse passer $m$ seulement sur les dimensions que $a$ ouvre » (silu ≈ 0 quand $a<0$, ≈ $a$ quand $a$ grand). C'est cette **multiplication entre deux branches** qui distingue le GLU du MLP plain, où l'activation agit sur l'unique flux, sans seconde branche.
- $\cdot\,w_1$ : descente, reprojection vers $\mathbb{R}^d$ ($w_1$).

Ce n'est pas exotique : c'est **exactement le bloc feed-forward des transformers modernes** (le FFN de LLaMA, PaLM… est un SwiGLU à 3 matrices). La seule chose inhabituelle ici, ce n'est pas la structure du réseau, c'est qu'on traite ses poids comme une **mémoire mutable** — réécrite à chaque chunk (cf. II.3) — au lieu de poids figés.

Pourquoi gaté plutôt que linéaire ($Wq$) ou plain ? Une mémoire linéaire sature vite et ne stocke que des associations linéaires. Le produit $\odot$ donne une interaction **multiplicative dépendante de l'entrée** : la mémoire peut **router/filtrer** son contenu selon $q$ — une liaison clé→valeur bien plus riche, à paramètres comparables. Mais conceptuellement, $o$ reste **« ce que la mémoire ressort pour la question $q$ »**.

### II.2 Sortie : normalisation, gate AdaLN, démarrage inerte

La lecture $o$ n'est pas injectée brute. Elle passe par une normalisation puis une **modulation gatée** (AdaLN) :

$$\hat o = \operatorname{RMSNorm}(o), \qquad (\text{shift}, \text{scale}, \text{gate}) = \hat o\,W_{proj} + b_{proj},$$
$$\boxed{\;\text{mem\_out} = \text{gate} \odot \big(\operatorname{LN}(\hat o)\,(1+\text{scale}) + \text{shift}\big)\;}$$

Détail crucial : le biais du gate est initialisé à **0** (`g_init=0.0`). Donc au pas 0, $\text{gate}=0 \Rightarrow \text{mem\_out}=0$ : **démarrage strictement identique au baseline**. La mémoire ne peut que *s'ajouter* à mesure qu'elle apprend — elle ne peut pas casser le modèle au départ. (« Sortie nulle » ≠ « gelé » : $\partial L/\partial\,\text{gate} = g\odot(\dots)\neq 0$ ouvre la porte, puis le reste s'entraîne.)

### II.2b Où `mem_out` entre : le câblage exact au bloc 5

`mem_out` est ajouté au **flux résiduel à la profondeur 5**, mais *où exactement* est tout l'enjeu — la vraie question est **qui voit la mémoire ?**. Le déroulé réel du bloc (`causal_model.py`) :

```
h = norm1(x_in)                  # pré-norm du bloc
y = self_attn(h modulé)
x = x_in + e2 · y                # résidu self-attn
mem_out, W' = memory.step(h, W)  # lit le MÊME h, en parallèle
x = x + mem_out                  # résidu mémoire
x = cross_attn_ffn(x)            # cross-attn + FFN voient x + mem_out
→ blocs 6, 7, … voient x + mem_out
```

Quatre points structurels :

1. **Elle lit $h = \operatorname{norm1}(x_{\text{in}})$, pas $x_{\text{in}}$** — et surtout elle **réutilise la pré-norm de la self-attention**, elle n'a pas de norm propre ($h$ est calculé une fois, mis en cache, partagé). Mémoire et self-attn partent donc de la **même pré-norm $h$** — à une nuance près : la self-attn module ensuite $h$ par $e$ ($h(1+e_1)+e_0$) avant l'attention, tandis que la mémoire prend $h$ **brut** dans son propre `qkv`.
2. **Parallèle à la self-attention**, pas un pré-traitement de l'entrée du bloc : les deux branches partent de $h$ indépendamment.
3. **Auto-gatée.** Son gate est **interne** (l'AdaLN zéro-init, dans `mem_out`). Contrairement à la self-attn (gate $e_2$) et au FFN (gate $e_5$), la mémoire **n'est pas** modulée par le vecteur $e$ partagé du bloc. Vraie différence de câblage : les autres sous-couches sont gatées par la modulation commune du bloc, la mémoire **se gate elle-même**.
4. **Vue par** : la cross-attention + le FFN du bloc 5, et **tous les blocs 6+** — **pas** la self-attention du bloc 5 (elle a déjà lu $h$ en parallèle).

D'où la formule juste, à la profondeur 5 :

$$x = x_{\text{in}} + \underbrace{e_2\cdot \text{self\_attn}(h)}_{\text{parallèle}} + \underbrace{\text{mem\_out}(h,\,W)}_{\text{parallèle}}, \qquad h = \operatorname{norm1}(x_{\text{in}}),$$

puis `cross_attn_ffn` par-dessus.

*Caveat* : c'est le câblage de **cette** intégration (greffe parallèle à la self-attn, sur la pré-norm partagée) — un **choix**, pas une nécessité. On pourrait poser la mémoire en sous-couche propre (sa norm, sa place séquentielle). À garder en tête pour la portabilité de la méthode (LongLive, etc.).

### II.3 Écriture : la règle de Hebb, mais à travers le MLP

Même objectif que le squelette — après le pas, lire $k$ doit donner $v$ — mais $f_W$ est non-linéaire, donc on dérive l'alignement par la **règle de la chaîne**. Objectif (qu'on fait **monter**) :

$$s = \langle f_W(k),\ v\rangle.$$

**But.** Pour chaque matrice $w_0, w_1, w_2$, trouver la direction qui fait **monter** $s$. On note $\bar{x} = \partial s / \partial x$ — le « signal qui remonte » vers $x$.

**Le forward** (un token ; $k\in\mathbb{R}^d$, $w_0,w_2\in\mathbb{R}^{d\times h}$, $w_1\in\mathbb{R}^{h\times d}$, $v\in\mathbb{R}^d$) :

$$a = k\,w_0, \quad m = k\,w_2, \quad g = \operatorname{silu}(a), \quad \text{hidden} = g\odot m,$$
$$y = \text{hidden}\,w_1 = f_W(k), \quad s = y\cdot v.$$

**Le calcul, directement en forme matricielle.** Méthode des différentielles : on calcule $ds$ en ne faisant varier qu'**une** matrice à la fois, et on met $ds$ sous la forme $\langle \nabla, dw_i\rangle$ ; le facteur devant $dw_i$ **est** le gradient. (Produit scalaire : $\langle A,B\rangle = \operatorname{tr}(A^\top B)$.) Deux règles suffisent :

- **Linéaire** : $\langle X W, c\rangle = \langle X, c\,W^\top\rangle$ et $\langle K X, c\rangle = \langle X, K^\top c\rangle$ — on transpose la matrice de l'autre côté.
- **Hadamard** : $\langle a\odot b, c\rangle = \langle a, b\odot c\rangle$ — le $\odot$ se déplace librement.

Pour $w_0$ (on varie $w_0$, donc $dm = 0$), on déroule $ds$ :

$$ds = \langle dy,\, v\rangle = \langle d\text{hidden}\,w_1,\, v\rangle = \langle d\text{hidden},\, v\,w_1^\top\rangle$$
$$= \langle dg\odot m,\, v\,w_1^\top\rangle = \langle dg,\, m\odot(v\,w_1^\top)\rangle$$
$$= \langle \operatorname{silu}'(a)\odot da,\, m\odot(v\,w_1^\top)\rangle = \langle da,\, \operatorname{silu}'(a)\odot m\odot(v\,w_1^\top)\rangle$$
$$= \langle k\,dw_0,\, \bar a\rangle = \langle dw_0,\, k^\top\bar a\rangle, \quad \bar a := \operatorname{silu}'(a)\odot m\odot(v\,w_1^\top).$$

On **lit** le gradient devant $dw_0$ : $\nabla w_0 = k^\top\bar a$. Les deux autres tombent du **même** déroulé (en variant $w_1$, resp. $w_2$, avec $\bar m := \operatorname{silu}(a)\odot(v\,w_1^\top)$) :

$$\nabla w_1 = \text{hidden}^\top v, \qquad \nabla w_2 = k^\top\bar m.$$

(Correspondance avec les noms du code : $v\,w_1^\top$ = `dhidden`, $\bar m$ = `dhide_before_mul`, $\bar a$ = `dgate_before_act` — à ne pas confondre avec `dgate` $=(v\,w_1^\top)\odot m$, l'intermédiaire *avant* le passage par $\operatorname{silu}'$.)

Le motif est toujours le même : pousser $ds$ vers la matrice voulue en **transposant** les matmuls et en **commutant** les $\odot$, jusqu'à isoler $dw_i$. **Les trois gradients**, avec le $lr$ par token replié (il scale simplement) :

$$\nabla w_0 = (k\odot lr_0)^\top\bar a, \quad \nabla w_1 = (\text{hidden}\odot lr_1)^\top v, \quad \nabla w_2 = (k\odot lr_2)^\top\bar m.$$

**Le sens.** Les trois $\nabla w_i$ ont la **même forme** : $(\text{entrée})^\top \otimes (\text{signal arrière})$ — un **produit extérieur**, donc la règle de Hebb. Ce qui les distingue, c'est *quel* signal arrière est branché :

| Matrice | entrée | signal arrière | rôle |
|---|---|---|---|
| $w_1$ | hidden | $v$ (cible brute) | descente |
| $w_0$ | $k$ | $\bar{a}$ (cible → cachée → porte) | porte |
| $w_2$ | $k$ | $\bar{m}$ (cible → cachée → contenu) | contenu |

Toute la chaîne (`dhidden`, `dm`, `dg`) ne sert qu'à **transporter la cible $v$** depuis la sortie jusqu'à chaque branche (via $w_1^\top$, le gate, la pente de silu) ; une fois le signal arrivé devant une matrice, le gradient est le **produit extérieur** entrée ⊗ signal.

Trois remarques :

- **Pas de signe moins.** On *monte* l'alignement (objectif TTT) ; Muon vient ensuite.
- $\operatorname{silu}'(a) = \sigma(a)\big(1 + a(1-\sigma(a))\big)$ (c'est `silu_backprop`).
- Sur un chunk de $L$ tokens, le $(\cdots)^\top(\cdots)$ est une **somme de $L$ produits extérieurs** (le matmul sur l'axe $L$) : un chunk = **une écriture combinée** de toutes ses associations.

### II.4 Le pas réel : `+=` → Muon + weight-norm

On n'applique **pas** bêtement $w \mathrel{+}= \nabla w$. Deux corrections, dans cet ordre :

$$n_i = \lVert w_i \rVert \ \text{(par colonne, AVANT)}; \quad \nabla w_i \leftarrow \text{NewtonSchulz}_5(\nabla w_i); \quad w_i \leftarrow w_i + \nabla w_i; \quad w_i \leftarrow \frac{w_i}{\lVert w_i\rVert}\, n_i.$$

**Qui fait quoi — les 4 bouts de cette ligne = 3 opérations distinctes.** Le piège classique : *« Muon » et « Newton-Schulz » sont la **même** étape* (Newton-Schulz est *comment* Muon s'exécute), pas deux.

| Bout de la formule | Nom | Rôle |
|---|---|---|
| $n_i = \lVert w_i\rVert$ *(par colonne, AVANT)* | **weight-norm** — moitié 1 : *sauvegarde* | mémoriser la longueur de chaque colonne, pour y revenir à la fin |
| $\nabla w_i \leftarrow \text{NewtonSchulz}_5(\nabla w_i)$ | **Muon** $=$ Newton-Schulz | orthogonaliser le gradient (équilibrer ses directions) |
| $w_i \leftarrow w_i + \nabla w_i$ | **le pas de gradient** (la montée) | appliquer *enfin* le gradient — déjà orthogonalisé — au poids |
| $w_i \leftarrow \frac{w_i}{\lVert w_i\rVert}\,n_i$ | **weight-norm** — moitié 2 : *restaure* | redonner à la colonne sa longueur $n_i$ d'avant |

L'ordre est donc un **sandwich** : *sauver $n_i$* → *orthogonaliser (Muon/NS)* → *$w \mathrel{+}= \nabla w$* → *restaurer $n_i$*. Le `+=` du milieu est le **vrai** pas ; Muon le prépare avant, le weight-norm répare l'échelle après (et l'englobe : sauve avant / restaure après). *(Code : `weight_norm_update`, exactement cet ordre.)*

**Muon (= Newton-Schulz, 5 itérations) — pourquoi orthogonaliser *avant* le pas ?** Le gradient brut est un produit extérieur, donc **rang-1** (une seule direction porte toute l'énergie, zéro ailleurs) et très mal conditionné. Appliqué tel quel, il pose un **dilemme insoluble** sur le pas : trop grand → tu **exploses** sur cette unique direction ; trop petit → tu **n'apprends rien** sur les autres. Newton-Schulz remplace $\nabla w = U S V^\top$ par $\approx U V^\top$ : il **met toutes les valeurs singulières à 1** → un pas de **même taille sur toutes les directions à la fois**, ce qu'aucun choix de learning rate ne peut faire seul. *(Et normaliser juste la norme ne suffit pas : un axe à $+1000$ et le reste à $0$, divisé par $1000$, donne $+1$ et le reste **toujours** à $0$ — seule l'orthogonalisation rééquilibre **entre** les directions.)* C'est l'optimiseur Muon — son cœur, ici **sans momentum** — appliqué au poids rapide.

**Weight-norm — pourquoi renormaliser ?** Sur des dizaines de chunks, sans contrainte les poids exploseraient ou s'effondreraient. On remet chaque colonne à sa norme d'origine : on change **ce qui** est stocké (la *direction* des colonnes) sans toucher à **combien d'énergie** il y a (leur *norme*). C'est ce qui rend la mémoire stable sur une vidéo longue.

Résultat : $W' = (w_0', w_1', w_2')$ = la mémoire **après** avoir absorbé ce chunk.

### II.5 Correspondance squelette ↔ module

| Squelette (Partie I) | Module réel | Pourquoi |
|---|---|---|
| Lire $Wq$ | $f_W(q)$ SwiGLU | expressivité, lecture sélective |
| Écrire $v k^\top$ (Hebb) | les 3 $\nabla w_i$ par chaîne | même Hebb, à travers le MLP |
| $W=0$ au départ | init aléatoire + gate de sortie à 0 | démarrage inerte = baseline |
| $W \mathrel{+}= \cdots$ | Muon + weight-norm | conditionnement + stabilité long terme |

---

## Partie III — Le temps : ordre et entraînement

### III.1 Apply-first et causalité

Pour chaque chunk, l'ordre est **lire d'abord, écrire ensuite** :

1. $o = f_W(q)$ avec l'état $(w_0,w_1,w_2)$ **reçu du chunk précédent** ;
2. puis l'écriture $W \to W'$ avec $(k,v,lr)$ de ce chunk.

Conséquence : un token ne lit **que le passé** (les chunks $< t$), et son présent n'est gravé que *pour les futurs*. C'est ce qui rend la mémoire compatible avec la génération autorégressive (pas de fuite du futur).

`do_update=False` saute l'écriture : on **lit sans graver** (utile pour un chunk qu'on ne veut pas mémoriser, p.ex. bruité ou généré plutôt que propre).

### III.2 Les deux échelles de temps

Il y a **deux** sortes de « poids », à ne pas confondre :

- **Poids lents** (les vraies `nn.Parameter`, entraînés par Adam sur le dataset, boucle externe) : `W_qkv`, `q_norm`, `k_norm`, `lrs`, `o_norm`, `proj`, **et la valeur initiale** de $w_0/w_1/w_2$.
- **Poids rapides** (l'**état**) : la triade $(w_0,w_1,w_2)$ courante, qui **monte le gradient au moment du test**, chunk après chunk. Elle n'est pas dans l'optimiseur ; elle est recalculée à chaque forward, en repartant de l'init.

Le piège : $w_0/w_1/w_2$ joue **les deux rôles** — son *point de départ* est un poids lent entraîné, sa *valeur courante* est un poids rapide qui évolue à l'inférence.

### III.3 Phases d'entraînement (E1 vs E3)

Le gradient externe ne traverse que ce qui est **dans le graphe**. En **Phase 1 (E1)**, le pipeline met l'écriture sous `torch.no_grad()`. Donc le gradient ne passe **que par la lecture** :

| Paramètre | Sert à | E1 | E3 |
|---|---|:--:|:--:|
| `qkv` (part $q$), `q_norm` | lire | ✅ | ✅ |
| `o_norm`, `proj` (gate) | sortie | ✅ | ✅ |
| $w_0/w_1/w_2$ | mémoire | ✅ (init, via lecture) | ✅ + via la dynamique d'écriture |
| `qkv` (part $k/v$), `k_norm` | quoi écrire | ❌ | ✅ |
| `lrs` | force d'écriture | ❌ | ✅ |

Pourquoi les ❌ : `lrs`, `k_norm`, `k/v` n'apparaissent **que** dans l'écriture. Comme l'écriture est `no_grad` en E1, aucun gradient ne les atteint → ils restent à l'init. C'est exactement pourquoi `lrs.bias = [0,0,0]` dans le checkpoint E1.

**E3 (write training)** = enlever ce `no_grad` et rétropropager **à travers le temps** (les chunks, BPTT). Ça réveille toute la colonne de droite. Mais il faut garder en mémoire les activations de plusieurs chunks → c'est lourd en VRAM → d'où le **BPTT tronqué**.

### III.4 La loss externe et les deux gradients (et pourquoi le *write* est dur à entraîner)

Jusqu'ici « entraîner » est resté flou. Précisons : **quelle perte**, **quel gradient va où**, et **pourquoi certaines parties coincent**.

**La loss : on n'invente PAS de loss mémoire.** La mémoire est greffée dans le générateur Anchor-Forcing et entraînée **de bout en bout par la même perte que lui** — **DMD** (*Distribution Matching Distillation*). Aucun objectif auxiliaire « mémoire ».

Le principe de DMD, en bref : on n'a **pas** de vidéo-cible à recopier. À la place, on déroule une vidéo « fake » avec le générateur et on pousse sa **distribution** vers celle du professeur, à l'aide de **deux réseaux de score** :
- `real_score` = le **professeur** (ici Wan 1.3B **figé**) — sait à quoi ressemble une vraie vidéo ;
- `fake_score` = le **critic** (1.3B + LoRA) — suit la distribution *actuelle* du générateur, entraîné en parallèle par une perte de débruitage (flow), à raison de `dfake_gen_update_ratio = 5` pas de critic par pas de générateur.

Schématiquement, le gradient DMD qui tire le générateur ≈ **`score_fake − score_real`** sur une version bruitée des frames générées (eq. 7 du papier DMD). Dans le code il est injecté via une perte-surrogate `½·‖x − (x − grad).detach()‖²` dont le gradient par rapport à `x` **est** exactement `grad` (`compute_distribution_matching_loss`, `model/dmd.py`). *(La perte ne porte que sur les frames fraîchement générées — le `gradient_mask`.)*

Ce qui compte pour nous : **`mem_out` modifie les frames générées → modifie la loss DMD → la mémoire reçoit un gradient.** Elle est entraînée *uniquement* en rendant la vidéo générée meilleure, rien d'autre.

**Deux gradients, à ne jamais confondre.** C'est le pendant des deux échelles de temps (III.2) :

| | Gradient **interne** (l'écriture) | Gradient **externe** (l'apprentissage) |
|---|---|---|
| C'est quoi | la mise à jour des poids **rapides** (II.3–II.4) | la backprop de la loss **DMD** |
| Sur quel objectif | l'alignement TTT $\langle f_W(k),v\rangle$ | la distribution des vidéos (DMD) |
| Calculé comment | **à la main**, formule fermée | **autograd**, appliqué par Adam |
| Met à jour quoi | l'**état** $w_0/w_1/w_2$ (rapide) | les **poids lents** (qkv, norms, proj, lrs, *init* de $w_0/w_1/w_2$) |
| Quand | pendant le forward, **à chaque chunk** (même à l'inférence) | à chaque **step d'entraînement**, sur le dataset |

$w_0/w_1/w_2$ est au carrefour : son **point de départ** bouge par le gradient externe, sa **valeur courante** par le gradient interne.

**Pourquoi certaines parties coincent.** Le gradient externe n'atteint un poids lent **que s'il est dans le graphe autograd de la loss**. Deux cas :
- **Côté lecture** (`q` de qkv, `q_norm`, `o_norm`, `proj`, et $w_0/w_1/w_2$ comme *état initial*) : la lecture est un simple forward différentiable (SwiGLU + AdaLN). Le gradient DMD les atteint **directement et pas cher** → entraînables **dès E1**.
- **Côté écriture** (`k/v` de qkv, `k_norm`, `lrs`) : ils n'apparaissent **que dans l'écriture**. Pour que le gradient externe les atteigne, il doit **traverser l'écriture** — et c'est là que ça coince, pour **deux raisons** :
  1. L'écriture contient **Muon (Newton-Schulz, 5 matmuls) + weight-norm** : rétropropager à travers ça est **lourd et numériquement délicat** (le NS5 tourne en bf16 → ~2-4 % d'erreur relative de gradient, *cf.* le `FIXME` du code).
  2. L'état passe de chunk en chunk : un gradient vers l'écriture doit donc remonter **à travers les chunks = backprop dans le temps (BPTT)** → il faut **garder les activations de tous les chunks** → la VRAM explose.

  → D'où le choix pragmatique d'**E1** : mettre l'écriture sous `torch.no_grad()`. Elle **tourne** quand même (la mémoire se remplit et se lit), mais elle est **coupée du graphe** → aucun gradient externe n'atteint le côté écriture → il **reste à l'init**. C'est *pourquoi* `lrs.bias=[0,0,0]` et `k_norm` non entraîné (cf. III.3).

**Ce n'est pas « impossible », c'est cher.** Entraîner l'écriture est juste **coûteux** (VRAM du BPTT) et **délicat** (gradient à travers Muon/NS). On le **diffère** donc à **E3**, qui enlève le `no_grad` et fait un **BPTT *tronqué*** (fenêtre limitée de chunks, pas toute la vidéo) pour rendre la VRAM gérable. La stratégie : **E1 apprend à bien *lire*** une mémoire qui écrit avec sa dynamique d'init (gaussienne, non entraînée mais raisonnable) ; **E3 raffine ensuite *comment* elle écrit**.

### III.5 Le gradient externe en détail : par où il passe (et par où il ne passe pas)

En II.3 on a calculé **à la main** le gradient *interne* (l'écriture). Faisons le pendant pour le gradient *externe* — la backprop de la loss DMD $L$ — pour voir **concrètement** pourquoi il atteint la lecture pour pas cher et l'écriture à grand prix.

**Le graphe d'un chunk $t$** (état entrant $W_t=(w_0,w_1,w_2)$) :

$$q_t = \operatorname{q\_norm}(h_t W_{qkv}^q),\quad o_t = f_{W_t}(q_t),\quad \text{mem\_out}_t = \text{gate}\big(\operatorname{o\_norm}(o_t)\big)\!\cdots \ \to\ x \mathrel{+}= \text{mem\_out}_t \ \to\ \text{vidéo}\ \to\ L,$$

$$\underbrace{W_{t+1} = U(W_t,\,k_t,\,v_t,\,lr_t)}_{\text{écriture : } U=\text{weight-norm}\,\circ\,\text{Muon}\,\circ\,\nabla},\qquad k_t=\operatorname{k\_norm}(h_t W_{qkv}^k),\ \ v_t=h_t W_{qkv}^v.$$

La loss ne « voit » la mémoire **qu'à travers les $\text{mem\_out}_t$**. Or $\text{mem\_out}_t$ dépend de **deux** choses : la lecture au chunk $t$ (via $q_t$) **et** l'état $W_t$ (l'accumulation de toutes les écritures passées). D'où les deux chemins.

**Chemin lecture — bon marché (dès E1).**

$$\frac{\partial L}{\partial \theta_{\text{read}}} = \sum_t \frac{\partial L}{\partial\,\text{mem\_out}_t}\cdot\frac{\partial\,\text{mem\_out}_t}{\partial \theta_{\text{read}}},\qquad \theta_{\text{read}}\in\{W_{qkv}^q,\ \operatorname{q\_norm},\ \operatorname{o\_norm},\ \text{proj}\}.$$

- `proj`, `o_norm` : **direct** (c'est l'AdaLN de sortie).
- `q_norm`, $W_{qkv}^q$ : via $\partial\,\text{mem\_out}_t/\partial o_t \cdot \partial o_t/\partial q_t \cdot \partial q_t/\partial\theta$, où $o_t=f_{W_t}(q_t)$ est le SwiGLU — **différentiable** en $q_t$ (même machinerie `silu_backprop`, mais cette fois pour le gradient *externe* par rapport à $q$, pas l'interne par rapport à $W$).

Point clé : chaque lecture utilise $W_t$ comme une **constante donnée** (en E1 les écritures sont détachées) → **aucun chaînage entre chunks**. C'est une backprop **simple, par chunk, en un pas**. Pas cher, pas de dimension temporelle.

**$w_0/w_1/w_2$ (l'init) — cas à part.** $o_t=f_{W_t}(q_t)$ dépend *aussi* des poids $W_t$. Au **premier** chunk, $W_1=$ la `nn.Parameter` d'init → $\partial o_1/\partial(w_0,w_1,w_2)$ entraîne l'init. Pour $t>1$, $W_t$ est le résultat (détaché en E1) des écritures → pas de gradient vers l'init. Donc l'init de $w_0/w_1/w_2$ est entraîné, **mais seulement via la première lecture** (l'apply-first sur l'état frais). C'est le « ✅ (init, via lecture) » de III.3.

**Chemin écriture — cher (→ coupé en E1).** Les $\theta_{\text{write}}\in\{W_{qkv}^{k},\,W_{qkv}^{v},\,\operatorname{k\_norm},\,W_{lr}\}$ n'apparaissent **que** dans $U$. Ils n'influencent $L$ qu'en changeant $W_{t+1}$, donc **toutes les lectures futures** $o_{t+1},o_{t+2},\dots$ :

$$\frac{\partial L}{\partial \theta_{\text{write}}} = \sum_t \sum_{s>t} \frac{\partial L}{\partial o_s}\,\frac{\partial o_s}{\partial W_s}\,\underbrace{\Big[\textstyle\prod_{r=t+1}^{s-1}\tfrac{\partial W_{r+1}}{\partial W_r}\Big]}_{\text{chaîne dans le temps}}\,\frac{\partial W_{t+1}}{\partial \theta_{\text{write}}}.$$

Deux ingrédients lourds :
1. $\dfrac{\partial W_{t+1}}{\partial \theta_{\text{write}}}$ **traverse $U=$ weight-norm $\circ$ Muon $\circ\,\nabla$** : il faut dériver à travers **Newton-Schulz (5 itérations de matmul)** et le weight-norm. Lourd (5 « couches » dont il faut garder les itérés) + délicat (NS5 en bf16 → ~2-4 % d'erreur de gradient, *cf.* le `FIXME`).
2. Le produit $\prod \partial W_{r+1}/\partial W_r$ **chaîne à travers les chunks = BPTT** : le gradient d'une écriture au chunk $t$ doit remonter par **chaque** mise à jour d'état jusqu'au chunk $s$ → il faut **retenir les activations de tous les chunks** → VRAM $\propto$ nombre de chunks.

**Le `no_grad` d'E1 = couper net.** Poser $W_{t+1}=\operatorname{detach}\big(U(\dots)\big)$ rend $\partial W_{t+1}/\partial\,\cdot \equiv 0$ dans le graphe → toute la double somme **s'annule** pour $\theta_{\text{write}}$ → $k/v$, `k_norm`, `lrs` reçoivent **exactement zéro** gradient → figés à l'init. L'écriture **tourne** quand même numériquement (la mémoire se remplit et se lit), elle est juste **invisible à l'autograd**.

**E3 = rebrancher, mais tronqué.** On enlève le `no_grad` et on **borne le BPTT à une fenêtre des $K$ derniers chunks** (BPTT *tronqué*) : le gradient d'une écriture ne remonte que $K$ pas, pas toute la vidéo → VRAM bornée → l'écriture se réveille, à coût maîtrisé.

### III.6 Un exemple chiffré : gradient *read* vs *write*

Des vrais chiffres (vérifiés en `torch`) valent mille formules. Jouet minimal : `head_dim=2`, un token $x=[1,\,0.5]$, une projection $W_q$ (**read**), deux projections $W_k,W_v$ (**write**/valeur), des poids rapides $w_0,w_1,w_2$ ($2\times2$). Deux chunks, pour voir l'état se propager.

**Chunk 1 — lire puis écrire (apply-first) :**
- $q_1=x\,W_q=[0.85,\,0.25]$ → lecture $o_1=f_W(q_1)=[0.0281,\,-0.0169]$.
- écriture : on monte $\ell=\langle f_W(k_1),v_1\rangle=0.0185$ → le **gradient interne** (le write) :
$$\nabla w_0=\begin{pmatrix}0.0190 & -0.0017\\ 0.0380 & -0.0034\end{pmatrix}\quad(\text{idem }\nabla w_1,\nabla w_2),\qquad W_1=W_0+\eta\,(\dots)$$

**Chunk 2 — lire avec l'état mis à jour :** $o_2=f_{W_1}(q_2)=[0.0288,\,-0.0167]$, perte $L=\tfrac12\lVert o_2-\text{cible}\rVert^2=0.0181$.

**Le gradient externe (training) pour un param read ($W_q$) vs un param write ($W_k$) :**

| Cas | $\partial L/\partial W_q$ (**read**) | $\partial L/\partial W_k$ (**write**) |
|---|---|---|
| écriture **dans le graphe** | `[[-0.0130, -0.0055], [-0.0065, -0.0028]]` | `[[-0.0003, -0.0002], [-0.0001, -0.0001]]` |
| écriture en **`no_grad`** (E1) | **identique** (inchangé) | **exactement 0** (aucun chemin) |

Trois choses sautent aux yeux :
1. **Le read ($W_q$) reçoit le même gradient**, que l'écriture soit dans le graphe ou non → il ne passe **pas** par l'écriture : entraîné proprement et **pas cher**.
2. **Le write ($W_k$) ne reçoit un gradient que si l'écriture est dans le graphe** — et il est **~40× plus petit** ($\sim\!3\text{e-}4$ vs $\sim\!1.3\text{e-}2$), car c'est un effet de **second ordre** (il traverse la mise à jour ⇒ la Hessienne de III.5).
3. **Sous `no_grad` (notre E1), $\partial L/\partial W_k=0$ exactement** → $W_k$ et tous les params write restent **figés à l'init**.

C'est *exactement* la table E1/E3 de III.3, démontrée **numériquement** : read entraîné bon marché, write soit second-ordre cher, soit zéro.

---

## Partie IV — Trace sur 2 chunks

Mettons les pièces en mouvement (qualitativement). État initial $W^{(0)} = $ l'init entraînée.

**Chunk 1** (tokens $x_1$, supposé propre donc `do_update=True`) :
1. Projections → $q_1, k_1, v_1, lr_1$.
2. **Lecture** : $o_1 = f_{W^{(0)}}(q_1)$. La mémoire est « neuve » (juste l'init) ; comme le gate de sortie démarre quasi-fermé, $\text{mem\_out}_1 \approx 0$, le bloc se comporte ~comme le baseline.
3. **Écriture** : on grave $k_1 \to v_1$. $\nabla w_i = (k_1\odot lr)^\top(\cdots)$, Muon + renorm → $W^{(1)}$. La mémoire « sait » maintenant répondre à $k_1$.

**Chunk 2** (tokens $x_2$) :
1. Projections → $q_2, k_2, v_2, lr_2$.
2. **Lecture** : $o_2 = f_{W^{(1)}}(q_2)$ — et c'est **ici** que tout se joue. Si $q_2$ ressemble à $k_1$ (le sujet réapparaît), alors $f_{W^{(1)}}(q_2)$ ressort une composante $\approx v_1$ : **le chunk 2 se souvient du chunk 1**. C'est le contraste avec le chunk 1, où la lecture ne voyait encore rien.
3. **Écriture** : on grave $k_2 \to v_2$ par-dessus → $W^{(2)}$, transmis au chunk 3.

La différence $o_2 - o_1$ — le fait que la lecture du second chunk soit informée par le premier — est **toute la valeur ajoutée** du module par rapport à un bloc sans mémoire.

---

## Récapitulatif (cheat-sheet)

- **Mémoire = petit réseau *gaté* SwiGLU** (GLU, 3 matrices / 2 flux — pas un MLP plain) ; ses poids $(w_0,w_1,w_2)$ sont le contenu.
- **Lire** : $o = f_W(q) = (\operatorname{silu}(q w_0)\odot(q w_2))w_1$ — un forward pass (récupération).
- **Écrire** : monter $\langle f_W(k), v\rangle$ d'un pas → 3 produits extérieurs (Hebb à travers le MLP), puis **Muon** (orthogonalise) + **weight-norm** (stabilise) — stockage de l'association $k\to v$.
- **$q$ lit, $k/v$ écrivent, $lr$ règle la force.**
- **Apply-first** : lire le passé, puis graver le présent → causalité.
- **Deux échelles** : poids lents (Adam, dataset) vs poids rapides (l'état, gradient au test). $w_0/w_1/w_2$ = init lente *et* état rapide.
- **Entraînement** : pas de loss mémoire — elle ride la perte **DMD** d'Anchor-Forcing. **Deux gradients** : *interne* (l'écriture, à la main, met à jour les poids rapides) vs *externe* (DMD via autograd, met à jour les poids lents). `mem_out` change la vidéo → change la loss → la mémoire apprend.
- **E1** : écriture en `no_grad` → seule la lecture est entraînée (`lrs/k_norm/k,v` dorment). **E3** : BPTT (tronqué) à travers les chunks → réveille l'écriture. *Pourquoi différer le write : backprop à travers Muon/NS = délicat, et à travers les chunks = BPTT = VRAM.*
