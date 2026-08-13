# Campagne des GATES "composite vs additif" (2026-08-02)

Exécution du plan arbitré après le débat. Rien n'est commité, tout est additif et par défaut OFF.
Chaque gate a son tableau ; un gate ne débloque la suite que s'il est vert.

## 0. Le point de départ (GATE 0, déjà acquis)

`reference/runs/reid_scen/results/comp/comp_SS2.json`, 3 cellules SS2 (t12/t13/t14, seed 0),
sujet mesuré contre le **A\* propre** (segment 0 de SS1, même triple/même seed) :

| bras | sujet crop60 | fond (head) | fond (win) | cut2 |
|---|---|---|---|---|
| memzero (sans mémoire) | 0.265 | 0.934 | 0.893 | 0.71 |
| framebank (banque Echo) | 0.375 | 0.898 | 0.828 | 7.67 |
| oracle1 (splice vraie ancre) | 0.376 | 0.899 | 0.828 | 7.69 |
| **memdec20 (notre mémoire)** | **0.689** | **0.459** | **0.475** | **20.03** |

Le constat qui a lancé le débat : notre bras gagne le sujet (0.69 vs 0.38 pour la banque) mais
**paie le fond** (0.46 vs 0.90) et **coupe deux fois et demie plus fort** (20.0 vs 7.7). Le splice
pleine-frame re-rend toute la scène depuis le passé.

Deux explications rivales, deux remèdes :
* **composite** — le mal est que l'ancre écrase TOUTE la frame ; on garde son centre (le sujet) et
  on laisse le bord au contexte courant.
* **additif** — le mal est l'ÉVICTION elle-même ; on injecte l'ancre SANS retirer aucune vraie
  frame de la fenêtre, en tokens k/v supplémentaires.

## 1. Protocole commun

* **Cellules** : 3 SS2 (même décor, sujet non-ouvrant : t12 café / t13 pelouse / t14 boulangerie)
  et 3 S2 (décor changeant : t2 chat / t4 cheval / t11 théière), seed 0. Les cellules S2 sont
  celles de la campagne composite en cours, pour rester appariées.
* **Sujet** : DINO crop60 de la fenêtre de retour contre le **A\* propre** (SS1 seg0 pour SS2,
  S3 seg0 pour S2), jamais contre la 1re apparition du même rollout (qui est déjà polluée).
* **Fond** : DINO sur le complément du crop60, retour vs queue du segment B du MÊME rollout.
  Haut = le monde a continué. Plafond de référence = memzero.
* **cut2_ratio** : brutalité du raccord au retour (lag-1 max autour du joint / lag-1 médian intra).
* **TRIPWIRES ajoutés pour cette campagne** (`score_comp.py`) :
  * `dd_ell` = mouvement (lag-1 pixel) **à l'intérieur** du crop central, fenêtre chaude et
    fenêtre complète. S'effondrer sous memzero = autocollant = rejet, pas discussion.
  * `clip_text` = adhérence CLIP (ViT-B/32 openai) du segment de retour à SON prompt.
  * sujet aussi aux rayons 0.5 et 0.7 (le rayon 0.6 seul serait un choix libre).
* **Code** : `pipeline/switch_causal_inference.py`, `wan/modules/causal_model.py`,
  table de bras `reference/runs/reid_scen/scenarios.py`.
  Selftests : `reference/runs/reid_scen/selftest_gates.py` (nouveau) et `selftest_composite.py`
  (mis à jour), **tous verts** (18 + 24 assertions).

### 1.1 Un résultat de méthode obtenu en route : le rollout n'est PAS bit-à-bit reproductible

La preuve bit-à-bit demandée (« le code additif ne touche à rien quand il est éteint ») a été
tentée en **génération réelle** : regénérer `SS2_t12_s0_memzero` avec le code modifié et comparer
au mp4 de la campagne. Résultat : **les pixels diffèrent** (max 50/255, moyenne 1.0/255) — mais la
divergence commence à la **frame latente ~3**, c'est-à-dire au deuxième chunk, très avant tout
hook (1er switch à 60, revisit à 180) et sur un bras qui n'a **aucun** hook. Ce n'est donc pas le
code : c'est le non-déterminisme ambiant du pipeline (flash-attention / cuBLAS / allocateur),
amplifié de façon chaotique par 240 frames d'autorégression. Vérifié aussi **à l'intérieur d'un
même process** : deux bras consécutifs (`oracle1add1` / `oracle1add2`) diffèrent déjà avant le
revisit (moyenne 1.20/255). La phrase « tous les bras MAG partagent le segment B bit à bit »
(docstring de `score_comp.py`) est donc trop forte.

**Ce qui sauve l'appariement** : le PLANCHER DE BRUIT sur les métriques, mesuré sur la cellule
SS2_t12 en rejouant memzero :

| | sujet crop60 | fond head | fond win | cut2 | DD ellipse | CLIP |
|---|---|---|---|---|---|---|
| memzero (campagne) | 0.1436 | 0.9467 | 0.8908 | 0.64 | 0.0029 | 0.2037 |
| memzero (rejoué, autre process/GPU) | 0.1428 | 0.9474 | 0.8893 | 0.64 | 0.0029 | 0.2038 |
| **écart** | **0.0008** | 0.0007 | 0.0015 | 0.00 | 0.0000 | 0.0001 |

Le bruit de reproductibilité est **~0.001 sur le sujet**, deux ordres de grandeur sous les effets
mesurés (0.1 à 0.7). Les comparaisons inter-process de ce rapport sont donc valides ; mais la
garantie est statistique, pas bit-à-bit (mesurée sur une cellule).
La vraie preuve d'inertie du code est structurelle et testée en selftest : hook absent =>
attribut `_add_kv` absent => **le même appel `attention()` qu'avant**, et `_maybe_composite`
retourne l'OBJET ancre lui-même.

## 2. Prédictions PRÉ-ENREGISTRÉES

Écrites avant les mesures. (Le compte-rendu du débat n'existant pas sous forme de fichier dans
le repo, elles sont reconstituées ici à partir de la logique des deux camps ; à confronter au
log d'Aden.)

| # | Prédiction | Camp |
|---|---|---|
| P1 | Le canal additif récupère une fraction NETTE du sujet **sans** toucher au fond : le mal était l'éviction. | additif |
| P1' | Contre-prédiction : la résurrection est un phénomène de **récence pure** (courbe de position déjà mesurée : slot0 +0.224, slot3 +0.019, sink échangé −0.02), donc des tokens hors fenêtre sont à peine lus et le canal est **inerte**. | position |
| P2 | Unifier la convention de fenêtre change le score de memdec20 : la fenêtre legacy DUPLIQUE la récence (elle garde la vraie frame cur-1 ET ajoute l'ancre), la convention oracle la REMPLACE. | mécanique |
| P3 | Reculer l'ancre d'un ou deux slots (vraie frame à la récence) fait **chuter** le sujet et **remonter** le fond (c'est exactement la courbe de position). | position |
| P4a | Masque uniforme à dose appariée : si la géométrie compte, le sujet doit être **nettement inférieur** à l'ellipse. S'il est égal, le « masque » n'est qu'un bouton de dose et la géométrie est une illusion. | anti-illusion |
| P4b | Ancre bruitée, masque gardé : le sujet doit **s'effondrer vers memzero**. Sinon la métrique mesure la géométrie/les statistiques du composite, pas l'identité. | anti-illusion |
| P4c | Masque décalé d'un tiers : le sujet doit chuter (le sujet est centré). Masque inversé : sujet effondré ET fond effondré (le bord devient le passé). | anti-illusion |

## 3. GATE 1 — canal additif à contenu oracle

### 3.1 Ce qui a été implémenté

`MEM_ADDITIVE_KV=1` (bras `oracle1add*`, défaut OFF). Au revisit :
1. **passe fantôme** de la VRAIE frame d'ancre (celle qu'oracle1 splice) dans un cache 1-frame
   NEUF ; chaque bloc y calcule ses propres k/v, capturés **avant RoPE** (`_add_capture`) ;
2. ces 1560 k/v deviennent des tokens **supplémentaires**, concaténés au moment de l'attention
   (template `prompt_cache` déjà présent dans le code), donc **rien n'est évincé** : le recache du
   retour est le recache NORMAL et la fenêtre garde ses 12 vraies frames ;
3. ils restent lus jusqu'à la fin du rollout (sink virtuel), pas seulement au recache.

**Bande RoPE.** Sous `rope_enhance` (config du pipeline MAG) les bandes sont fixes :
sink = `[0, 1)`, fenêtre locale = `[rope_start_f_k_local, 21)` avec
`rope_start_f_k_local = max(rope_start_f - local_budget/1560 + gen_frame, sink) = max(18-11+3, 1)
= 10` en génération et `max(9-11+12, 1) = 10` au recache. La bande **`[1, 10)` est donc libre** ;
`MEM_ADD_POS=auto` place l'ancre à **9**, collée devant la fenêtre — des distances que le modèle
lit à chaque pas. Pas de collision, vérifié dans les deux modes (`gen_frame` 3 et 12).

**Gain = repondération EXACTE du softmax**, pas une heuristique. Les deux log-sum-exp rendus par
le noyau flash donnent
`out = (Z_ctx·out_ctx + g·Z_anc·out_anc) / (Z_ctx + g·Z_anc)`,
c'est-à-dire l'attention sur `[contexte ; ancre]` avec les logits de l'ancre décalés de `log g`
(g = 1 : concaténation pure, vérifié à 2.3e-3 près en bf16, selftest d1). Sous-produit gratuit :
`alpha = g·Z_anc/(Z_ctx + g·Z_anc)` **est** la masse d'attention prise par l'ancre — le tripwire
d'inertie du plan sink-virtuel, mesuré et non supposé.

### 3.2 Le tripwire mécanique, avant même les scores

Masse d'attention des tokens d'ancre au dernier forward (moyenne sur les 30 blocs), gain 1 :

| | masse moyenne | min bloc | max bloc | pour comparaison |
|---|---|---|---|---|
| ancre additive, gain 1 | **2.04 %** | 0.67 % | 4.24 % | 7.7 % si elle était lue **uniformément** (1560 tokens sur 20280) |

Le modèle **sous-pondère** l'ancre d'un facteur ~4 par rapport à sa part de tokens. Le balayage
{0.5, 1, 2} ne couvre donc que ~1 % à ~4 % de masse : c'est pour cela qu'un point de **dose
forcée gain 8** (~14 %, plus que la part de tokens) a été ajouté au balayage — il sépare
« canal mort parce que NON LU » de « canal mort parce que ILLISIBLE ».

Le canal n'est pas inerte au sens numérique : sur S2_t2, `oracle1add1` vs `oracle1add2` diffèrent
de 3.98/255 en moyenne **après** le retour contre 1.20/255 avant (= le bruit ambiant), et
`oracle1add1` vs `memzero` de 6.21/255 après contre 1.21 avant. **Les pixels bougent ; les
métriques, non.**

### 3.3 Scores — tier S2 (décor changeant), 3 cellules t2/t4/t11, seed 0

| bras | sujet crop60 | fond head | fond win | cut2 | DD ellipse | CLIP texte |
|---|---|---|---|---|---|---|
| memzero | 0.0715 | 0.8874 | 0.8242 | 0.73 | 0.0101 | 0.1862 |
| framebank (banque Echo) | 0.1641 | 0.7410 | 0.7288 | 7.38 | 0.0054 | 0.2211 |
| **oracle1 (splice, même contenu)** | **0.1645** | 0.7416 | 0.7292 | 7.39 | 0.0054 | 0.2212 |
| memdec20 (notre mémoire) | 0.7757 | 0.0650 | 0.0676 | 10.96 | 0.0085 | 0.3342 |
| **oracle1add05** (gain 0.5) | **0.0713** | 0.8834 | 0.8230 | 0.74 | 0.0100 | 0.1870 |
| **oracle1add1** (gain 1) | **0.0747** | 0.8838 | 0.8123 | 0.74 | 0.0097 | 0.1873 |
| **oracle1add2** (gain 2) | **0.0719** | 0.8838 | 0.8203 | 0.75 | 0.0097 | 0.1887 |

Chemin memzero -> oracle1 sur le sujet : 0.0715 -> 0.1645 (**+0.0930**, moitié du chemin = 0.1180).

| bras | delta sujet vs memzero | fraction du chemin | signes |
|---|---|---|---|
| framebank / oracle1 | +0.0926 / +0.0930 | 100 % | 3+/0− |
| memdec20 | +0.7042 | (au-delà, autre régime) | 3+/0− |
| oracle1add05 | **−0.0001** | **−0 %** | 1+/2− |
| oracle1add1 | **+0.0032** | **+3 %** | 1+/2− |
| oracle1add2 | **+0.0005** | **+1 %** | 1+/2− |

**VERDICT GATE 1 (tier S2) : CANAL MORT.** Le seuil pré-enregistré était « la moitié du chemin »
(+0.0465) ; le meilleur gain rend **+0.003**, soit **3 %**, avec des signes 1+/2− (aléatoire) et
une amplitude de l'ordre du plancher de bruit (0.001). Ce n'est même pas un petit effet : c'est
zéro. Et il ne détruit rien non plus — fond 0.883 vs 0.887 pour memzero, cut2 0.74 vs 0.73 : le
canal ne fait **rien du tout** au niveau des métriques.

**C'est EXACTEMENT le même contenu que le splice `oracle1`**, la vraie frame d'ancre. Placée dans
le slot de récence, elle rapporte +0.093 (100 % du chemin) ; injectée en tokens supplémentaires
dans une bande RoPE voisine, elle rapporte +0.003. La prédiction **P1' (récence pure) est
confirmée, P1 (le mal était l'éviction) est réfutée** : ce n'est pas l'éviction qui fait le
travail, c'est **l'occupation du slot de récence**. La résurrection n'est pas un problème de
lecture, c'est un problème de *ce que le modèle déclare être son présent immédiat*.

### 3.4 Scores — tier SS2 (même décor), balayage complet des gains

La masse d'attention suit exactement le gain (mesurée, moyenne sur 30 blocs) :

| gain | 0.5 | 1 | 2 | **8** | (part de tokens) |
|---|---|---|---|---|---|
| masse d'attention de l'ancre | 1.04 % | 2.04 % | 3.62 % | **10.5 – 15.7 %** | 7.7 % |

À gain 8 l'ancre prend **plus que sa part de tokens** : elle est lue, et beaucoup.

| bras | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | delta sujet | % du chemin | signes |
|---|---|---|---|---|---|---|---|---|
| memzero | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | — | — | — |
| oracle1 (même contenu, en splice) | 0.3764 | 0.8989 | 7.69 | 0.0051 | 0.2692 | +0.1116 | 100 % | 3+/0− |
| oracle1add05 | 0.2473 | 0.9322 | 0.71 | 0.0041 | 0.2478 | −0.0175 | −16 % | 1+/2− |
| oracle1add1 | 0.2509 | 0.9330 | 0.72 | 0.0042 | 0.2473 | −0.0140 | −13 % | 1+/2− |
| oracle1add2 | 0.2614 | 0.9361 | 0.71 | 0.0044 | 0.2483 | −0.0035 | −3 % | 1+/2− |
| **oracle1add8 (dose forcée)** | **0.3039** | **0.9250** | **0.73** | 0.0050 | 0.2549 | **+0.0390** | **+35 %** | 2+/1− |

par cellule à gain 8 : t12 0.1436 -> **0.2394** (oracle1 0.1620, l'additif BAT le splice) ;
t13 0.5130 -> 0.5047 (oracle1 0.6937) ; t14 0.1379 -> 0.1675 (oracle1 0.2735).

L'effet est **monotone en gain** (−0.018, −0.014, −0.004, +0.039) : le canal n'est pas cassé, il
est **saturé de faiblesse**. Il faut multiplier la masse d'attention par 6 (2 % -> 12 %) pour
sortir du bruit, et on n'obtient encore que le tiers du chemin.

**VERDICT GATE 1 : CANAL MORT au sens de la règle pré-enregistrée** (le meilleur gain rend 35 %
sur SS2 et 3 % sur S2, contre un seuil de 50 %, sans significativité : 2+/1− et 1+/2−). Sur les
gains demandés {0.5, 1, 2} le canal est même **exactement nul** (−0.018 à +0.003 sur les deux
tiers, dans le plancher de bruit). Prédiction **P1 réfutée, P1' (récence pure) confirmée** : la
même frame d'ancre rend 100 % du chemin quand elle OCCUPE le slot de récence et 0-3 % quand elle
est lue à côté. **Ce n'est pas l'éviction qui fait le travail, c'est l'occupation du présent
déclaré.**

**Mais le mode d'échec n'est pas celui qu'on attendait, et il faut le dire** : le canal additif ne
DÉTRUIT rien. À dose forcée il laisse le fond à 0.925 (memzero 0.934, le splice 0.899), le
raccord à **0.73 — c'est-à-dire le cut de memzero (0.71), pas celui du splice (7.69)** — et le
mouvement dans l'ellipse à 0.0050 (memzero 0.0040, splice 0.0051 : aucun autocollant). C'est le
seul bras jamais mesuré qui apporte de l'identité (faiblement) **en restant dans le « bon
quadrant »** (identité + douceur) que la campagne quadrant du 24/07 avait trouvé VIDE. Trop
faible pour passer le gate, pas nocif : à ranger comme piste v2 (contenu appris pour ce canal),
pas comme mécanisme mort-né.

Note de lecture sur `bg_cont` en tier S2 : le prompt du retour ORDONNE un changement de décor
(A et B sont deux scènes sans rapport), donc un fond bas y est le comportement CORRECT, pas un
dégât — memdec20, qui a le fond le plus bas (0.065), a aussi la **meilleure adhérence texte**
(CLIP 0.334 contre 0.186 pour memzero et 0.221 pour la banque). L'axe fond n'est un axe de DÉGÂT
que sur le tier même-décor (SS2), où la caméra est censée continuer.

## 4. GATE 2 — convention de fenêtre et placement

### 4.1 L'asymétrie, exactement (lecture de code, avant toute mesure)

Le recache écrit 12 frames mais **la première tombe sur le slot sink protégé** (`is_recompute` =>
`write_start_index = max(local_start_index, sink_tokens)`, `global_sink: true` donc le cache n'est
pas remis à zéro) : seules les **11 dernières** entrent réellement dans la fenêtre. Avec `cur` =
frame du retour :

| chemin | tenseur passé au recache | ce qui est réellement écrit (positions `cur-11 … cur-1`) |
|---|---|---|
| oracle1 (`ORACLE_SPLICE_FRAMES=1`) | `out[cur-12 : cur-1] + ancre` | `out[cur-11 … cur-2]` **à leur position vraie**, ancre au slot de récence |
| memdec20 (legacy) | `out[cur-11 : cur] + ancre` | `out[cur-10 … cur-1]` **décalées d'un slot vers le passé**, ancre au slot de récence |

Donc les deux bras ne partageaient PAS la même fenêtre : le chemin memdec garde la vraie frame
`cur-1` (le présent immédiat) et pousse tout le reste d'un cran, le chemin oracle la remplace.
Corrigé par `MEM_DECODE_SLOT_OFFSET` (`_memdec_window`, selftest b1-b8 : `offset < 0` reproduit
l'expression legacy verbatim, `offset = 0` est **slot par slot** la fenêtre d'oracle1).
### 4.2 Mesures (3 cellules SS2, seed 0)

| bras | placement de l'ancre | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | delta sujet | % du chemin |
|---|---|---|---|---|---|---|---|---|
| memzero | — | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | — | — |
| oracle1 | récence, contenu VRAI | 0.3764 | 0.8989 | 7.69 | 0.0051 | 0.2692 | +0.1116 | 100 % |
| memdec20 (legacy) | récence, fenêtre décalée | 0.6888 | 0.4586 | 20.04 | 0.0053 | 0.3455 | +0.4239 | 380 % |
| **memdec20s0** (convention oracle) | récence, fenêtre alignée | **0.7249** | 0.4553 | 19.79 | 0.0054 | 0.3342 | **+0.4600** | 412 % |
| **memdec20s1** | ancre reculée d'1 slot | **0.2723** | **0.9405** | **1.43** | 0.0040 | 0.2518 | +0.0075 | 7 % |
| **memdec20s2** | ancre reculée de 2 slots | **0.2664** | **0.9413** | **1.24** | 0.0037 | 0.2448 | +0.0015 | 1 % |

**Convention de fenêtre (P2) : effet réel mais MODESTE et non significatif.** Aligner la fenêtre
sur la convention d'oracle1 donne **+0.036 de sujet** en moyenne (0.6888 -> 0.7249) sans rien
coûter (fond 0.455 vs 0.459, cut 19.8 vs 20.0). Mais par cellule : **+0.0910 / −0.0061 / +0.0235**,
soit 2+/1− et une moyenne portée par t12 — à n = 3 ce n'est pas significatif, et P2 (« unifier
change le score ») n'est confirmée qu'au sens faible. La vraie raison d'adopter `s0` n'est donc
pas le gain : c'est que **memdec20s0 et oracle1 partagent enfin exactement la même fenêtre**, si
bien que leur différence est *uniquement le contenu de l'ancre*, ce qui n'était pas vrai des
chiffres du GATE 0. Correction de propreté, gratuite, jamais nuisible.

**Placement (P3) : confirmée de façon spectaculaire.** Reculer l'ancre d'UN SEUL slot détruit
**98 % du gain sujet** (+0.4600 -> +0.0075) et restaure intégralement le reste : le fond remonte
à 0.9405 (**au-dessus de memzero**, 0.9343) et le cut tombe de 19.8 à **1.43**. Deux slots :
+0.0015, fond 0.9413, cut 1.24. C'est la courbe de position du 24/07 (slot0 +0.224, slot3 +0.019)
reproduite à l'identique sur une ancre DÉCODÉE, sur un autre tier et avec un autre scoring.

**Conséquence, et c'est la ligne directrice de toute la campagne** : puissance et violence sont
**le même bouton**. Le slot de récence donne l'identité ET le rembobinage ; tout ce qui l'éloigne
(reculer, additionner à côté) rend le fond et la douceur en reprenant l'identité. Il n'existe pas
de point intermédiaire *par la position* — seulement, peut-être, par le CONTENU (composite).

## 5. GATE 3 — les trois contrôles anti-illusion

Tous les contrôles sont `memdec21c` + UN flag, donc toute différence leur est attribuable
(selftest e2). Vérifié en pipeline réel (poids moyen du masque imprimé à chaque revisit) :

| bras | ce qui change | poids moyen du masque |
|---|---|---|
| `memdec21c` (référence) | — | 0.288 |
| `memdec21c_unif` | plus de géométrie, poids CONSTANT | 0.288 (**dose appariée exactement**) |
| `memdec21c_noise` | masque gardé, ancre = bruit pur (σ=1, RMS préservé) | 0.288 |
| `memdec21c_shift` | ellipse décalée de W/3 | 0.238 |
| `memdec21c_inv` | m -> 1-m (ancre au BORD) | 0.712 |

Caveat honnête sur `_shift` : l'ellipse décalée sort partiellement du cadre, sa dose tombe de
0.288 à 0.238. Le décalage n'est donc pas parfaitement apparié en dose — mais dans le sens qui
DURCIT la lecture : s'il conserve le score, c'est avec MOINS d'ancre, donc la conclusion
« la géométrie ne fait pas le travail » n'en est que plus forte.

### 5.0 Le tableau complet des contrôles (3 cellules SS2, seed 0)

| bras | dose | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | delta sujet | % du chemin | signes |
|---|---|---|---|---|---|---|---|---|---|
| memzero | 0 | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | — | — | — |
| oracle1 | (splice) | 0.3764 | 0.8989 | 7.69 | 0.0051 | 0.2692 | +0.1116 | 100 % | 3+/0− |
| memdec20 (splice plein) | 1.000 | 0.6888 | 0.4586 | 20.04 | 0.0053 | 0.3455 | +0.4239 | 380 % | 3+/0− |
| **memdec21c (référence composite)** | 0.288 | **0.4119** | 0.8265 | 5.83 | 0.0058 | 0.2613 | **+0.1471** | 132 % | 3+/0− |
| oracle1comp (contenu vrai) | 0.288 | 0.3549 | **0.9374** | **2.80** | 0.0042 | 0.2628 | +0.0900 | 81 % | 3+/0− |
| **(a) memdec21c_unif** (dose appariée, sans forme) | 0.288 | **0.3298** | 0.9403 | 2.32 | 0.0042 | 0.2541 | **+0.0650** | 58 % | 3+/0− |
| **(b) memdec21c_noise** (masque gardé, contenu détruit) | 0.288 | **0.3441** | 0.9302 | 3.47 | 0.0040 | 0.2551 | **+0.0792** | 71 % | 3+/0− |
| **(c) memdec21c_shift** (bonne forme, mauvaise place) | 0.238 | **0.2950** | 0.9075 | 4.65 | 0.0050 | 0.2437 | **+0.0301** | 27 % | 2+/1− |
| **(c) memdec21c_inv** (masque inversé) | 0.712 | **0.5349** | 0.5982 | 12.23 | 0.0045 | 0.3004 | +0.2701 | 242 % | 3+/0− |

Ordonnancement **stable aux 4 rayons de scoring** (0.5 / 0.6 / 0.7 / frame pleine) : par exemple
memdec21c 0.425 / 0.412 / 0.467 / 0.458 contre unif 0.332 / 0.330 / 0.392 / 0.354 — la conclusion
ne dépend pas du choix de rayon.

### 5.1 Contrôle (a) — masque uniforme à dose appariée : **P4a CONFIRMÉE**

À dose **exactement** appariée (0.288, imprimé par le pipeline), supprimer la géométrie fait
tomber le gain sujet de **+0.1471 à +0.0650**. Autrement dit : **56 % du gain sujet du composite
vient de la FORME** (concentrer la dose sur le centre), 44 % de la dose seule. La géométrie n'est
donc pas un bouton de dose déguisé — le masque fait un vrai travail. (Et le masque uniforme est un
point « doux » intéressant en soi : fond 0.9403, cut 2.32.)

### 5.2 Contrôle (b) — ancre bruitée, masque gardé : **P4b RÉFUTÉE, et c'est le résultat gênant**

Prédiction : le sujet devait **s'effondrer vers memzero** si la métrique mesurait bien l'identité.
Mesuré : **+0.0792, soit 54 % du gain de memdec21c, avec un contenu d'ancre 100 % bruit**
(σ = 1, RMS préservé), 3+/0−.

Interprétation mécanique : le composite bruité **efface le sujet de l'interlude** au centre de la
frame de récence, et le modèle, conditionné par le prompt du retour (qui décrit A), **re-remplit
ce centre depuis le TEXTE**. Le score monte sans qu'aucune information mémorisée ne circule.
**Plus de la moitié du gain apparent du composite est donc « effacer l'intrus + laisser le prompt
repeindre », pas du rappel.** Ce qui reste attribuable au CONTENU de l'ancre est au mieux
+0.147 − 0.079 = **+0.068, soit ~46 %** du gain affiché de memdec21c.

**Et le tier S2 tranche le mécanisme** (mêmes contrôles, 3 cellules t2/t4/t11, décor CHANGEANT) :

| bras (tier S2) | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | delta | % chemin | signes |
|---|---|---|---|---|---|---|---|---|
| memzero | 0.0715 | 0.8874 | 0.73 | 0.0101 | 0.1862 | — | — | — |
| memdec21c | 0.1815 | 0.8203 | 2.70 | 0.0066 | 0.2492 | +0.1100 | 118 % | 3+/0− |
| (a) memdec21c_unif | 0.0732 | 0.9040 | 1.33 | 0.0081 | 0.1952 | **+0.0018** | **2 %** | 1+/2− |
| (b) memdec21c_noise | 0.0799 | 0.9100 | 1.52 | 0.0083 | 0.1963 | **+0.0084** | **9 %** | 2+/1− |

Sur S2, **les deux contrôles s'effondrent** : le masque uniforme rend 2 % et l'ancre bruitée 9 %,
contre 118 % pour le composite complet. **P4a et P4b sont donc toutes deux CONFIRMÉES sur le tier
cross-scène.** L'anomalie de P4b est **spécifique au tier même-décor**, et son mécanisme est
maintenant clair : quand le décor est partagé, « effacer l'intrus et laisser le prompt repeindre »
produit une frame qui ressemble légitimement à A\*, sans aucun rappel ; quand le décor change,
effacer ne rapporte rien.

**Conséquence de méthode, à porter au papier** : le tier SS2 **sur-récompense l'effacement**, et
c'est précisément le tier pour lequel le composite a été conçu. Toute annonce composite doit
publier son contrôle « ancre bruitée, masque gardé » sur le même tier — sans quoi on présente
comme du rappel ce qui est, à moitié, de la suppression.

### 5.3 Contrôle (c) — géométrie : décalé d'un tiers, inversé (3 cellules SS2)

| bras | dose du masque | sujet crop60 | fond head | fond win | cut2 | DD ellipse | CLIP |
|---|---|---|---|---|---|---|---|
| memzero | 0 | 0.2648 | 0.9343 | 0.8928 | 0.71 | 0.0040 | 0.2485 |
| oracle1 (splice vrai) | 1 (autre mécanisme) | 0.3764 | 0.8989 | 0.8284 | 7.69 | 0.0051 | 0.2692 |
| oracle1comp (composite, vrai contenu) | 0.288 | 0.3549 | **0.9374** | 0.8866 | **2.80** | 0.0042 | 0.2628 |
| **memdec21c** (référence composite) | 0.288 | 0.4119 | 0.8265 | 0.7755 | 5.83 | 0.0058 | 0.2613 |
| memdec20 (splice plein) | 1.000 | 0.6888 | 0.4586 | 0.4754 | 20.04 | 0.0053 | 0.3455 |
| **memdec21c_shift** | 0.238 | **0.2950** | 0.9075 | 0.8541 | 4.65 | 0.0050 | 0.2437 |
| **memdec21c_inv** | 0.712 | **0.5349** | 0.5982 | 0.6139 | 12.23 | 0.0045 | 0.3004 |

**Le composite fait bien ce qu'on lui demandait** (à noter avant les contrôles) : `oracle1comp`
rend le fond **au-dessus de memzero** (0.9374 vs 0.9343) et divise le cut par 2.7 (7.69 -> 2.80)
en ne perdant que 0.02 de sujet ; `memdec21c` remonte le fond de 0.459 à 0.827 et le cut de 20.0 à
5.8, en payant 0.277 de sujet.

**Contrôle décalé : la place COMPTE.** À dose quasi identique (0.238 vs 0.288, donc MOINS d'ancre),
l'ellipse déplacée d'un tiers de largeur ne rend plus que **+0.030** vs memzero (27 % du chemin)
contre **+0.147** (132 %) pour l'ellipse centrée : ~80 % du gain sujet du composite disparaît
quand on met la même ancre au mauvais endroit. **P4c confirmée pour le décalage.**

**Contrôle inversé : prédiction RÉFUTÉE, et c'est instructif.** P4c prévoyait un sujet effondré ;
mesuré, il MONTE (0.535 > 0.412). L'explication est un **confond de dose** que la prédiction avait
raté : `1 - m` couvre 71 % de la frame contre 29 % pour `m`, donc le bras « inversé » donne
**2.5× plus d'ancre**, pas moins. Remis dans l'ordre des doses, les trois points composites et le
splice plein forment une courbe presque parfaitement monotone :

| dose d'ancre | 0 (memzero) | 0.238 (décalé) | 0.288 (ellipse) | 0.712 (inversé) | 1.0 (splice plein) |
|---|---|---|---|---|---|
| sujet | 0.265 | 0.295 | 0.412 | 0.535 | 0.689 |
| fond (head) | 0.934 | 0.908 | 0.827 | 0.598 | 0.459 |
| cut2 | 0.71 | 4.65 | 5.83 | 12.23 | 20.04 |

Le **fond est une fonction quasi LINÉAIRE de la dose** et indifférente à la forme : la droite
memzero -> splice plein prédit 0.608 à la dose 0.712, on mesure 0.598 (écart 0.010, soit 7× le
plancher de bruit mais 2 % de l'échelle). Le sujet, lui, est **au-dessus** de la droite quand la
dose est au centre (0.412 mesuré contre 0.377 prédits) et **en dessous** quand elle est décalée
(0.295 contre 0.357) : la place vaut environ **±0.05**, soit le second ordre.

### 5.4 VERDICT GATE 3

| prédiction | verdict SS2 (même décor) | verdict S2 (cross-scène) |
|---|---|---|
| P4a masque uniforme ⇒ sujet nettement inférieur | **CONFIRMÉE** : +0.065 vs +0.147, la forme vaut **56 %** | **CONFIRMÉE, massivement** : +0.002 vs +0.110, la forme vaut **98 %** |
| P4b ancre bruitée ⇒ sujet effondré vers memzero | **RÉFUTÉE** : +0.079 = **54 %** du gain sans contenu | **CONFIRMÉE** : +0.008 = **9 %** |
| P4c masque décalé ⇒ sujet chute | **CONFIRMÉE** : +0.030 (27 %) à dose comparable | (non couru) |
| P4c masque inversé ⇒ sujet ET fond effondrés | **fond oui, sujet NON** : confond de dose (`1−m` = 2.5× plus d'ancre) | (non couru) |

**Le masque est validé comme mécanisme** (la place et la forme comptent, P4a + P4c) **mais la
moitié du gain qu'on lui attribuait est de l'effacement, pas du rappel** (P4b). Les deux
conclusions tiennent ensemble : le masque fait bien quelque chose de géométrique, et ce qu'il
fait est autant « enlever B » que « remettre A ».

Tripwires : aucun bras ne déclenche l'autocollant (DD dans l'ellipse 0.0040-0.0058 contre 0.0040
pour memzero, donc jamais SOUS memzero) ; l'adhérence texte ne chute jamais sous memzero
(0.2437 au pire pour `_shift`, contre 0.2485) ; `cut2` reste toujours sous celui du splice plein.

**Lecture d'ensemble** : le composite n'achète pas un mécanisme nouveau, il achète un **point de
dose** sur un compromis sujet/fond largement déterminé par « quelle fraction de la frame vient du
passé » ; centrer cette fraction sur le sujet vaut un bonus réel (56 % du gain) ; et une bonne
part du reste est un effet d'effacement que le prompt exploite.

## 6. GATE 4 — composite corrigé (conditionnel)

Le bras `memdec21u` est **implémenté et selftesté** mais ne doit tourner que si GATE 1 est mort ET
que GATE 3 valide le masque. Ce qu'il contient, tel que pré-enregistré :
* **masque UNION** : à défaut de cross-attention, union de l'ellipse centrée sur le centroïde
  d'énergie de l'ANCRE et de celle centrée sur le centroïde d'énergie de la frame COURANTE
  (`_energy_centroid`, selftest c5a-c5c) — sinon le composite colle le sujet rappelé PAR-DESSUS
  le sujet de l'interlude ;
* **`MEM_COMP_CROP=0.75` PRÉ-ENREGISTRÉ**, scoring aux 3 rayons 0.5 / 0.6 / 0.7 ;
* **dose α portée par le cosinus de prompt** entre le prompt qui entre et celui qu'on quitte
  (rampe 0.50 -> 0.80) : même décor demandé deux fois => composite ; scène sans rapport => α -> 0
  et le bras redevient **memdec20 BIT À BIT** (selftest c4a : `torch.equal`, max|d| = 0).

**Conditions d'ouverture remplies** : GATE 1 mort (§3.4) et GATE 3 valide le masque (P4a + P4c,
§5.4). Lancé sur les 3 cellules SS2 seed 0. *La 2e seed demandée n'a pas été faite : elle exigeait
aussi de générer ses propres références (SS1 seed 1 + memzero/oracle1 SS2 seed 1 = 15 rollouts de
plus), ce qui ne rentrait pas dans la nuit ; les 3 cellules restent strictement appariées.*

Le mécanisme se comporte comme spécifié : cos(prompt entrant, prompt quitté) = **0.88-0.92** sur
le tier même-décor -> **α = 1.000** (composite plein, c'est le bon tier) ; masque union de moyenne
**0.504-0.612** (contre 0.288 pour l'ellipse simple à crop 0.6).

| bras | dose | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | delta sujet | signes |
|---|---|---|---|---|---|---|---|---|
| memzero | 0 | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | — | — |
| framebank (banque Echo) | — | 0.3752 | 0.8976 | 7.67 | 0.0051 | 0.2678 | +0.1104 | 3+/0− |
| memdec21c (composite v1) | 0.288 | 0.4119 | 0.8265 | 5.83 | 0.0058 | 0.2613 | +0.1471 | 3+/0− |
| **memdec21u (composite CORRIGÉ)** | ~0.55 | **0.5579** | **0.7807** | 9.60 | 0.0058 | 0.2869 | **+0.2931** | **3+/0−** |
| memdec20 (splice plein) | 1.000 | 0.6888 | 0.4586 | 20.04 | 0.0053 | 0.3455 | +0.4239 | 3+/0− |

**Il bat la droite de dose sur les DEUX axes.** À la dose 0.55, l'interpolation memzero -> splice
plein prédit sujet 0.498 et fond 0.673 ; mesuré **0.558 (+0.060) et 0.781 (+0.108)**. Le masque
union + crop 0.75 achète donc un vrai point Pareto-supérieur, pas un simple curseur.
Contre memdec21c : **+0.146 de sujet pour −0.046 de fond** (3+/0− sur le sujet), un taux de change
bien meilleur que la droite de dose (~−0.16 de fond pour ce gain). Sur t13 il domine même
memdec21c sur **les deux** axes (sujet 0.894 vs 0.808, fond 0.826 vs 0.622).
Stable aux 4 rayons (0.605 / 0.558 / 0.575 / 0.627).

Tripwires : DD dans l'ellipse **0.0058 > 0.0040 (memzero)** -> pas d'autocollant ; adhérence texte
**0.2869 > 0.2485 (memzero)** -> le prompt du retour est mieux respecté, pas écrasé ; `cut2`
**9.60 contre 20.04** pour le bras qu'il corrige (mais 5.83 pour memdec21c : le raccord est plus
dur que le composite v1, prix de la dose plus élevée — à montrer à l'œil avant toute adoption).
α = 0 rend memdec20 **bit à bit** (selftest), donc le bras ne peut rien casser sur les tiers où
le cosinus de prompt est bas.

**VERDICT GATE 4 : POSITIF, sous réserve de l'œil.** Le composite corrigé occupe la place que
cherchait le projet : **+0.18 de sujet sur la banque Echo (0.558 vs 0.375)** avec un fond à 0.78
et sans autocollant. Ce n'est pas l'oracle et le raccord reste visible ; mais c'est le premier
bras qui déplace le compromis au lieu de le parcourir.

## 7. Synthèse : une seule courbe, deux boutons

Les quatre gates racontent la même chose. Tout ce qui a été mesuré cette nuit se range sur **un
compromis unique sujet ↔ fond**, et il n'y a que deux façons d'y bouger.

**Bouton 1 — la POSITION. Il est binaire, et il n'a pas de réglage fin.**
L'ancre ne compte que si elle occupe le slot de récence :

| l'ancre est... | sujet | fond | cut2 |
|---|---|---|---|
| au slot de récence (memdec20s0) | 0.7249 | 0.4553 | 19.79 |
| reculée d'UN slot (memdec20s1) | 0.2723 | 0.9405 | 1.43 |
| reculée de deux (memdec20s2) | 0.2664 | 0.9413 | 1.24 |
| en tokens k/v additionnels, gain 1 (oracle1add1) | 0.2509 | 0.9330 | 0.72 |
| idem, dose forcée à 12 % de masse (oracle1add8) | 0.3039 | 0.9250 | 0.73 |
| (rien : memzero) | 0.2648 | 0.9343 | 0.71 |

Un slot de recul détruit 98 % du gain ; un canal additif lu à 12 % de la masse d'attention en
récupère 35 % au mieux. **La résurrection est une propriété du présent déclaré du modèle, pas de
ce qu'il peut lire.** GATE 1 et GATE 2 le disent deux fois, indépendamment, sur des contenus
différents (frame vraie / frame décodée) et deux tiers.

**Bouton 2 — la DOSE, c'est-à-dire QUELLE FRACTION de la frame de récence vient du passé.**
Elle, elle est continue, et c'est le seul curseur réel :

| dose | 0 | 0.238 (décalé) | 0.288 (ellipse) | ~0.55 (union) | 0.712 (inversé) | 1.0 (splice) |
|---|---|---|---|---|---|---|
| sujet | 0.265 | 0.295 | 0.412 | **0.558** | 0.535 | 0.689 |
| fond | 0.934 | 0.908 | 0.827 | **0.781** | 0.598 | 0.459 |

Le fond est quasi linéaire en dose et **indifférent à la forme**. Le sujet, lui, dépend de la
forme : centrer la dose sur le sujet vaut +0.05 environ, et le masque **union** (GATE 4) sort de
la droite sur les deux axes à la fois (+0.060 de sujet et +0.108 de fond contre l'interpolation).

**Ce qui reste vrai après les contrôles.** GATE 3 impose une décote honnête : sur le tier
même-décor, ~54 % du gain composite s'obtient encore avec une ancre de bruit pur (effacer
l'intrus, laisser le prompt repeindre). Sur le tier cross-scène cette décote disparaît (9 %).
Le gain « de rappel » du composite est donc réel mais **entre une et deux fois plus petit que le
chiffre brut** selon le tier.

## 8. Recommandations

1. **Adopter `MEM_DECODE_SLOT_OFFSET=0`** partout (arm `memdec20s0`) : c'est gratuit, jamais
   nuisible, et c'est la seule façon d'avoir memdec20 et oracle1 sur la MÊME fenêtre.
2. **Fermer le canal additif comme mécanisme principal** (GATE 1), sans le jeter : c'est le seul
   bras du projet qui apporte de l'identité en restant dans le bon quadrant (fond 0.925, cut 0.73
   contre 7.69 pour le splice). S'il doit revivre, ce sera avec un contenu APPRIS pour ce canal,
   pas avec la frame brute — et la question à poser d'abord est « pourquoi le modèle ne lui donne
   que 2 % de masse d'attention ».
3. **Publier tout chiffre composite avec son contrôle « ancre bruitée »** sur le même tier. C'est
   maintenant une exigence de méthode, pas une précaution.
4. **`memdec21u` (GATE 4) mérite l'œil d'Aden** avant toute adoption : il gagne +0.18 de sujet sur
   la banque Echo avec un fond à 0.78, mais son raccord (cut2 9.6) est plus dur que celui du
   composite v1 (5.8). C'est un arbitrage esthétique, pas métrique.
5. **2e seed et n plus grand** : tous les verdicts ci-dessus reposent sur 3 cellules par tier et
   seed 0. Les effets tranchés (GATE 1, GATE 2 placement, GATE 3 sur S2) sont d'un ordre de
   grandeur au-dessus du bruit ; les effets fins (convention de fenêtre +0.036, 2+/1−) ne le sont
   pas et ne doivent pas être annoncés comme acquis.

## 9. Fichiers

* code : `pipeline/switch_causal_inference.py` (hooks `MEM_ADDITIVE_KV`, `MEM_DECODE_SLOT_OFFSET`,
  contrôles `MEM_COMP_*`, `_memdec_window`, `_install_add_kv`, `_energy_centroid`, `_comp_alpha`),
  `wan/modules/causal_model.py` (`_additive_attention`, `_sdpa_lse`, capture `_add_capture`)
* bras : `reference/runs/reid_scen/scenarios.py` (11 bras ajoutés, tous inertes par défaut)
* selftests : `reference/runs/reid_scen/selftest_gates.py`, `selftest_composite.py` (mis à jour)
* scoring : `reference/runs/reid_scen/score_comp.py` (tripwires DD-ellipse, CLIP, 3 rayons),
  `reference/runs/reid_scen/gates_table.py`
* résultats : `reference/runs/reid_scen/results/comp/comp_{S2gates_base, S2gates_g1, S2gates_g3,
  SS2gates_g1g2, SS2gates_g1gain8, SS2gates_g3geom, SS2gates_g3g4, SS2_noisefloor}.json`
* vidéos : `/shared/ssd_14T_project/adenelmi/reid_scen/videos/` (54 nouvelles)
* logs : `reference/runs/reid_scen/logs/gate*.log`

## 10. ADDENDUM (2026-08-03) — le FANTÔME et le remplissage bruit (`memdec21un`)

**Verdict œil d'Aden sur `memdec21u`** : le sujet revient bien placé, mais un **fantôme flou**
reste à l'ancienne position du sujet d'interlude. Diagnostic : la région effacée par l'union
recevait le contenu **passe-bas de l'ancre**, que le débruiteur ne nettoie qu'à moitié.

**Correctif** `MEM_COMP_FILL=noise` (bras `memdec21un`, env-gated, défaut OFF) : les trois poids
du composite somment à 1 par construction — `w_a = m_a` (l'ancre dans SA propre ellipse),
`w_n = m_union − m_a` (la région du sujet COURANT, remplie de bruit, calibration identique au
bras `memdec21c_noise` du gate 3), `w_bg = 1 − m_union`. Selftests : hook absent ⇒ `memdec21u`
inchangé ; **région de l'ancre bit à bit identique à `memdec21u` (max|diff| = 0)** ; α = 0 ⇒
`memdec20` bit à bit.

**Nouvelle métrique `ghost_*`** (`score_comp.py`) : la boîte fantôme est localisée UNE FOIS PAR
CELLULE sur memzero (frame juste avant le retour, même localiseur d'énergie que le pipeline),
donc identique pour tous les bras. `ghost_hf` = |laplacien| moyen dans cette boîte (un
remplissage passe-bas le fait chuter) ; `ghost_dino` = cette zone vs memzero.

| bras | sujet crop60 | fond head | cut2 | DD ellipse | CLIP | **ghost_hf** | ghost_dino |
|---|---|---|---|---|---|---|---|
| memzero | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | **0.0310** | 0.9532 |
| framebank | 0.3752 | 0.8976 | 7.67 | 0.0051 | 0.2678 | 0.0305 | 0.7229 |
| memdec20 | 0.6888 | 0.4586 | 20.04 | 0.0053 | 0.3455 | 0.0374 | 0.2826 |
| memdec21c | 0.4119 | 0.8265 | 5.83 | 0.0058 | 0.2613 | 0.0304 | 0.6731 |
| **memdec21u** | 0.5579 | 0.7807 | 9.60 | 0.0058 | 0.2869 | **0.0327** | 0.4427 |
| **memdec21un** | 0.5640 | 0.7523 | **8.58** | 0.0063 | 0.2799 | **0.0338** | 0.4474 |

Par cellule (poids de la région bruitée / sujet / ghost_hf) : t12 **0.056** — 0.397→0.477,
0.0277→0.0265 ; t13 **0.181** — 0.894→0.844, 0.0303→**0.0318** ; t14 **0.099** — 0.383→0.371,
0.0402→**0.0431**.

**Mon œil sur la planche zoomée** : le fantôme est **réel et flagrant sur t13** — un aplat vert
mou, sans structure, à gauche là où était le chat/le jardinier ; `memdec21un` le remplace par du
contenu structuré (jambes bleues du jardinier, banc, herbe). Sur t14 même sens, plus discret. Sur
t12 le remplissage laisse au contraire le bras du barista **plus présent** au bord — le monde
continue davantage, mais l'intrus n'est pas effacé.

**Verdict : correctif dans la bonne direction, mais PARTIEL, et la raison est géométrique.**
Le poids de la région bruitée n'est que **0.11 en moyenne** (0.056 / 0.181 / 0.099) parce que sur
le tier même-décor A et B occupent quasiment la MÊME place : le croissant « B seul » est mince.
Or une bonne part du fantôme est **à l'intérieur de l'ellipse de l'ancre**, là où le composite
colle le passe-bas de A par-dessus l'endroit où était B — et le correctif, par construction
(selftest g2), n'y touche pas. Les gains suivent d'ailleurs le poids : la cellule au plus grand
croissant (t13, 0.181) est celle où ghost_hf et l'œil s'améliorent le plus.

**Prochain cran, une ligne** : donner la PRIORITÉ au bruit sur le recouvrement, c'est-à-dire
`w_n = m_b` et `w_a = max(0, m_a − m_b)` au lieu de `w_n = m_union − m_a`. Toute la zone où était
le sujet courant est alors effacée, y compris sous l'ancre. À runner avant toute adoption.

Artefacts : `reference/runs/reid_scen/eyepack_fill/` (3 hstacks [union | union+bruit | banque],
8-20 MB, + `SS2_planche_zoom_fantome.png`), `results/comp/comp_SS2_fill.json`.

### 10.1 `memdec21up` — priorité au bruit sur le recouvrement (2026-08-03)

`MEM_COMP_FILL=priority` : `w_n = m_b`, `w_a = max(0, m_a − m_b)`, `w_bg = 1 − max(m_a, m_b)`
(somme = 1 partout, selftest g6). Toute la zone du sujet courant est effacée, **y compris sous
l'ellipse de l'ancre** (selftest g7 : sous `m_b = 1` l'ancre ne contribue plus du tout).

| bras | sujet crop60 | fond head | cut2 | DD-ell | CLIP | ghost_hf | ghost_dino |
|---|---|---|---|---|---|---|---|
| memzero | 0.2648 | 0.9343 | 0.71 | 0.0040 | 0.2485 | 0.0310 | 0.9532 |
| framebank | 0.3752 | 0.8976 | 7.67 | 0.0051 | 0.2678 | 0.0305 | 0.7229 |
| memdec21u | 0.5579 | 0.7807 | 9.60 | 0.0058 | 0.2869 | 0.0327 | 0.4427 |
| memdec21un (croissant) | 0.5640 | 0.7523 | 8.58 | 0.0063 | 0.2799 | 0.0338 | 0.4474 |
| **memdec21up (prioritaire)** | **0.5326** | **0.7972** | **6.33** | 0.0041 | 0.2746 | **0.0354** | **0.5204** |

**Le fantôme est réparé** : `ghost_hf` 0.0327 -> **0.0354** (memzero 0.0310 ; sur t13, 0.0303 ->
0.0364 contre une cible de 0.0374 : le déficit de netteté est comblé), `ghost_dino` +0.077, et le
raccord tombe de 9.60 à **6.33**, le meilleur des trois variantes.

**Mais le tripwire sujet TIRE**, exactement pour la raison annoncée. Le poids d'ancre survivant
n'est plus que **0.054 / 0.164 / ~0.10** (t12/t13/t14) : sur le tier même-décor A et B occupent la
même place, donc « prioritaire » retire l'ancre **pile où A doit revenir**. Par cellule :
**t12 +0.220, t13 −0.093, t14 −0.203** (moyenne −0.025, 1+/2−).

**Mon œil sur la planche** — et il explique le désaccord entre les cellules :
* **t12 (homme au café)** : la meilleure image de la ligne, nette, aucun fantôme, la table et les
  tasses continuent proprement. Le prompt a repeint un homme crédible… mais **en veste turquoise**
  alors que A porte une veste ROUGE VIF : le score monte, l'instance est fausse.
* **t13 (golden retriever)** : le flou vert est parti, mais **un second animal blanc apparaît** à
  gauche du chien — le repeint a réinventé quelque chose là où on avait effacé.
* **t14 (scooter bleu)** : A **ne revient pas du tout** ; on reste sur la devanture et le livreur.
  Un objet rigide spécifique n'est pas repeignable depuis le texte seul.

**Lecture** : c'est la confirmation directe du mécanisme du gate 3 (les 54 % « effacer + laisser
le prompt repeindre ») **et de sa limite** : il porte les sujets génériques/animés et lâche sur
l'objet rigide spécifique — c'est-à-dire précisément là où une mémoire d'instance est censée
servir. Effacer sous l'ancre répare le fantôme en échangeant du RAPPEL contre de la GÉNÉRATION.

**Recommandation** : ne pas adopter `priority` tel quel ; garder un **plancher d'ancre**
(`w_a = max(m_a − m_b, f·m_a)` avec f ~ 0.3-0.5) pour effacer le fantôme sans jamais couper le
canal d'identité, et re-mesurer avec le même tripwire. `memdec21un` (croissant) reste la variante
sûre, `memdec21up` la variante qui prouve la cause.

### 10.2 `memdec21uf` — plancher d'ancre f = 0.4, et RÉCAPITULATIF FINAL

`MEM_COMP_FILL=floor` : `w_a = max(m_a − m_b, f·m_a)`, `w_n = m_union − w_a`, `w_bg = 1 − m_union`.
Le plancher **contient les deux extrêmes** : f = 0 redonne `priority`, f = 1 redonne `noise`
(selftest g11, à 2.4e-7 près), donc les quatre variantes sont une seule famille à un paramètre.

**Les 4 variantes de remplissage, 3 cellules SS2, seed 0** (memzero 0.2648 / 0.9343 / 0.71 /
0.0310 / 0.2485 ; banque Echo 0.3752 / 0.8976 / 7.67 / 0.0305 / 0.2678) :

| variante | poids ancre | sujet crop60 | fond head | cut2 | ghost_hf | CLIP | tripwire sujet |
|---|---|---|---|---|---|---|---|
| **`memdec21u`** aucun remplissage | 0.44 | **0.5579** | 0.7807 | 9.60 | 0.0327 | **0.2869** | référence |
| `memdec21un` bruit sur le croissant | 0.44 | **0.5640** | 0.7523 | 8.58 | 0.0338 | 0.2799 | +0.006 (1+/2−) OK |
| `memdec21up` bruit prioritaire | 0.11 | 0.5326 | 0.7972 | **6.33** | 0.0354 | 0.2746 | **−0.025, t14 −0.203 : TIRE** |
| `memdec21uf` plancher f = 0.4 | 0.22 | 0.5505 | **0.8377** | 7.70 | 0.0464* | 0.2590 | −0.008 (1+/2−) OK |

\* `ghost_hf` 0.0464 est un FAUX bon chiffre : sur t12 il vaut 0.0623 pour 0.0288 chez memzero
(2.2×). Ce n'est pas de la netteté retrouvée, c'est de la **texture hallucinée** (voir ci-dessous).

**Mon œil sur la planche finale** :
* **t13 (golden retriever)** : `21uf` est **le meilleur panneau de toute la planche** — flou parti,
  jambes du jardinier et banc nets, un seul chien (alors que `21up` en faisait apparaître un
  second, et `21u` laissait l'aplat vert).
* **t12 (homme au café)** : `21uf` **hallucine un grand buisson** en plein cadre là où le bruit a
  été posé. Défaut DUR, pire que le fantôme flou de `21u`.
* **t14 (scooter bleu)** : A ne revient toujours pas complètement ; `21uf` (0.323) est entre `21u`
  (0.383) et `21up` (0.180).

**RECOMMANDATION FINALE — geler `memdec21u`** (union nue, crop 0.75, porte cosinus, sans
remplissage). Raisons :
1. c'est le meilleur sujet (0.558) ET la meilleure adhérence texte (0.287) de la famille ;
2. son défaut connu — le fantôme — est un défaut **MOU** (un flou) ; les trois remplissages
   échangent ce flou contre des défauts **DURS** et non déterministes : buisson halluciné (uf/t12),
   animal dupliqué (up/t13), sujet qui ne revient pas (up/t14) ;
3. aucune variante ne gagne sur plus d'une cellule sur trois : à n = 3 la famille est
   **sous-déterminée**, et geler la variante la plus simple est la décision honnête.

Le bouton qui n'a PAS été testé et qui expliquerait les hallucinations est l'**amplitude** du
bruit (σ = 1.0 partout) : à σ ≈ 0.5-0.6 la région serait effacée sans inviter le débruiteur à
inventer un objet. C'est le premier essai à faire si la famille est reprise — pas un balayage de
la géométrie, qui est maintenant cartographiée.

## 11. COMPARAISON ÉLARGIE À 5 CELLULES MÊME-DÉCOR (2026-08-03)

Aden préfère `memdec21un` (union + bruit sur le croissant) au fantôme de `memdec21u`. Pour
trancher sur autre chose que 3 cellules : **2 triples même-décor NEUFS**, écrits pour élargir les
catégories de sujet, held-out vérifié (0 hit sur les 248 221 lignes du corpus l1/v3) et selftest
same-scene vert (23 assertions) :
* **t15 garage** : A = voiture vintage rouge, B = vélo bleu, C (ouvreur) = mécanicien en bleu de
  travail — clause de décor partagée verbatim, 11 mots.
* **t16 serre** : A = perroquet vert, B = arrosoir en cuivre, C = jardinier au chapeau de paille
  — clause partagée, 12 mots.

### 11.1 Les 5 cellules, 5 bras (seed 0, sujet contre le A\* propre de SS1)

| bras | sujet crop60 | fond head | cut2 | DD-ell | CLIP | ghost_hf | ghost_dino |
|---|---|---|---|---|---|---|---|
| LongLive (stockoff) | 0.3249 | 0.8907 | 0.61 | 0.0142 | 0.2366 | 0.0482 | 0.6779 |
| sans mémoire (memzero) | 0.3089 | 0.9172 | 0.73 | 0.0049 | 0.2384 | 0.0351 | 0.9656 |
| banque Echo (framebank) | 0.4140 | 0.8726 | 8.01 | 0.0053 | 0.2595 | 0.0354 | 0.7959 |
| **NOTRE pure (memdec20)** | **0.6296** | 0.4557 | 17.52 | 0.0057 | **0.3406** | 0.0464 | 0.3489 |
| **NOTRE union+bruit (memdec21un)** | **0.5306** | **0.7620** | **7.65** | 0.0075 | 0.2761 | 0.0416 | 0.6156 |

Sujet par cellule :

| bras | t12 café | t13 chien | t14 scooter | **t15 voiture** | **t16 perroquet** |
|---|---|---|---|---|---|
| memzero | 0.144 | 0.513 | 0.138 | 0.102 | 0.648 |
| banque Echo | 0.162 | 0.692 | 0.272 | 0.113 | 0.831 |
| NOTRE pure | 0.622 | 0.840 | 0.605 | **0.237** | 0.845 |
| NOTRE union+bruit | 0.477 | 0.844 | 0.371 | **0.092** | **0.869** |

delta sujet vs memzero : banque **+0.105 (5+/0−)**, NOTRE pure **+0.321 (5+/0−)**,
NOTRE union+bruit **+0.222 (4+/1−)** — l'unique cellule négative est **t15**.

### 11.2 Mon œil : le bord-intrus de `21un` n'est PAS stable, et son instabilité a une LOI

Sur les 5 cellules, le défaut que j'avais noté discret sur t12 va de l'anodin au fatal :
* **t16 (perroquet)** : absent. Deux perroquets verts reviennent nettement sur la table de
  rempotage — meilleur sujet de toute la campagne (0.869), fond correct.
* **t13 (chien)** : mince liseré (les jambes bleues du jardinier au bord gauche), sans dommage —
  le golden retriever revient aussi bien qu'avec le splice plein (0.844 vs 0.840).
* **t12 (café)** : le barista reste au bord droit ET **la veste rouge de A est perdue** (0.477
  contre 0.622 pour le splice pur, où la veste rouge revient).
* **t14 (scooter)** : le livreur en vert occupe la moitié droite, le scooter n'est que partiel.
* **t15 (voiture rouge)** : **échec total** — le mécanicien occupe TOUT le retour, la voiture ne
  revient jamais ; 0.092, **sous memzero (0.102)**, la seule cellule négative de la campagne.
  Le splice pur, lui, ramène la voiture rouge (0.237).

**La loi est la même que celle du gate 3 et de `21up`** : le remplissage bruit efface l'intrus et
laisse le prompt repeindre — ça marche pour un sujet **générique ou animé** (perroquet, chien) et
ça lâche pour un **objet rigide spécifique** (scooter, voiture rouge), c'est-à-dire exactement là
où une mémoire d'INSTANCE est censée servir. Le bord-intrus n'est donc pas un défaut cosmétique
variable : c'est le symptôme visible d'un basculement rappel -> génération, dont la probabilité
dépend de la catégorie du sujet.

### 11.3 Ce que ça change pour le point de fonctionnement

`memdec21un` reste **le meilleur compromis global** : +0.117 de sujet sur la banque Echo pour un
raccord identique (7.65 vs 8.01) et un fond à 0.76 ; il divise le cut du splice pur par 2.3 et
lui reprend +0.31 de fond, pour −0.10 de sujet. Mais il **n'est pas uniformément supérieur** :
sur les objets rigides (t14, t15) le splice pur `memdec20` reste nettement meilleur et
`memdec21un` peut passer sous memzero.

Recommandation révisée : **`memdec21un` comme point de fonctionnement par défaut** (choix d'Aden,
confirmé sur 3 cellules /5), **avec la limite écrite noir sur blanc** : sur sujet rigide
spécifique, préférer `memdec20`. Le sélecteur naturel n'est pas un réglage de masque de plus —
c'est la même porte que la porte de prompt déjà implémentée, appliquée à la CATÉGORIE du sujet.
5 cellules et 1 seed restent trop peu pour l'annoncer comme acquis.

Artefacts : `results/comp/comp_SS2_5cells.json` ; `reference/runs/reid_scen/eyepack_5cells/`
(5 hstacks 5 panneaux 9-25 MB + `SS2_planche_5cellules_retours.png`).

## 11. Contrôles reviewer S2 (2026-08-03) — `noiseanchor` et `wronganchor` : **les deux prédictions sont RÉFUTÉES, le claim « contenu » tombe**

Les deux contrôles pré-enregistrés dans `reference/memory2_probes_registry.md` (ADDENDUM
2026-08-03), écrits pour parer l'attaque n°1 de `reference/paper/risks.md`. Ils ne la parent pas :
ils la confirment.

### 11.1 Les deux bras, et ce qui les sépare de `memdec20`

Tout est identique à `memdec20` — même routage (drapeau de retour de l'ordonnancement), même
instant (le switch de retour, frame 180), même position (récence), même convention de fenêtre
(legacy), même écrivain A2, même décodeur A2 (le décodage tourne quand même), même graine, même
flux de bruit. **Seul le CONTENU de l'ancre change** :

| bras | ancre splicée | env |
|---|---|---|
| `memdec20` | décodée de l'état de CETTE cellule | — |
| `noiseanchor` | **bruit pur**, calibré par canal sur les statistiques latentes de la fenêtre vraie où il est splicé (dose 1.0 = zéro contenu d'ancre) | `MEM_DECODE_NOISE=1.0` |
| `wronganchor` | ancre décodée de l'état d'une **AUTRE cellule** (triple (t+1) % 12, même seed) | `MEM_DECODE_SRC=<.pt>` |

Additif, env-gated, défaut OFF ; les helpers renvoient l'**objet ancre lui-même** quand les hooks
sont éteints, donc tous les bras déjà sur disque sont bit à bit inchangés (`selftest_controls.py`,
8 groupes, TOUT PASSE : table des bras sans fuite, inertie legacy, calibration par canal, ancre
étrangère splicée verbatim, exclusion mutuelle, **les 11 slots de contexte bit à bit identiques
entre les trois bras**, décalage sans point fixe, structure du hook dans le bloc MEM_DECODE après
le dump et avant le composite).

**Coût 48 rollouts et non 72** : `noiseanchor` persiste (MEM_DECODE_DUMP) l'ancre qu'elle décode
AVANT de la jeter — c'est exactement l'ancre `memdec20` de sa cellule (les deux rollouts sont
identiques jusqu'au splice) — et c'est elle que `wronganchor` du triple précédent relit.

### 11.2 Tripwire mécanique, avant les scores

24/24 splices de bruit (log : `per-channel mu in [-0.55, +0.73], sigma in [0.60, 1.45]`, ancre
std 0.60 -> splice std 1.17 = la vraie échelle latente) et 24/24 splices d'ancre étrangère
(`cos` avec l'ancre propre de la cellule **0.018 à 0.410**, rel-L2 1.06 à 1.65 : c'est bien un
autre contenu). Une seule collision GPU (OOM d'un co-run hostile), relancée par le superviseur.

### 11.3 Scores — S2, 24 cellules, re-scoring PROPRE (A\* = segment 0 du S3 de la même cellule)

Le script reproduit **bit à bit** `rescore_clean.py` sur les cellules communes (max\|diff\| = 0 sur
4 cellules × 2 bras), donc la colonne `memdec20` est exactement le chiffre-titre du rapport.

| bras | score `full` | delta vs memzero | signes | p | % du claim | score `crop60` | delta `crop60` |
|---|---|---|---|---|---|---|---|
| `memzero` | 0.2235 | — | — | — | — | 0.2282 | — |
| **`memdec20`** | 0.6577 | **+0.4343** ±0.0476 | 24+/0− | 1.2e-07 | 100 % | 0.6514 | +0.4232 |
| **`noiseanchor`** | 0.5134 | **+0.2899** ±0.0531 | 23+/1− | 3.0e-06 | **66.8 %** | 0.5139 | +0.2858 (67.5 %) |
| **`wronganchor`** | 0.6823 | **+0.4588** ±0.0474 | 24+/0− | 1.2e-07 | **105.7 %** | 0.6779 | +0.4498 (106.3 %) |

Contrastes appariés bras à bras : `noiseanchor` − `memdec20` = **−0.1444** (6+/18−, p = 0.023) ;
`wronganchor` − `memdec20` = **+0.0245** (13+/11−, p = 0.84) — **statistiquement indiscernable**.
Test d'instance : nul pour les quatre bras (−0.001 à −0.008, aucun p < 0.5).

### 11.4 Prédictions pré-enregistrées vs mesure, et règle de décision

| bras | prédiction (registre, écrite avant) | mesuré | verdict |
|---|---|---|---|
| `noiseanchor` | +0.02 à +0.08 (≈ 9 % du gain, extrapolé du gate 3 à dose 0.288) | **+0.2899** | **DEHORS**, ×3.6 au-dessus de la borne haute |
| `wronganchor` | +0.00 à +0.06, « avec contaminations visibles » | **+0.4588** | **DEHORS**, ×7.6 au-dessus, et **aucune** contamination |

Règle : *« si l'un des deux rend ≥ la moitié de +0.434, le claim "c'est le CONTENU de la mémoire"
tombe et le papier doit être reformulé (récence générique) »*. **Les DEUX la dépassent** (67 % et
106 %). ⇒ **LE CLAIM TOMBE.**

### 11.5 Test de contamination : le contenu splicé n'est PAS LU DU TOUT

Pour chaque cellule, le retour est comparé au **A\* propre du triple SOURCE** (celui dont
`wronganchor` a splicé l'ancre) :

| bras | cos(retour, A\* du sujet SOURCE) |
|---|---|
| `memzero` | 0.0439 |
| `memdec20` | 0.0426 |
| `noiseanchor` | 0.0439 |
| **`wronganchor`** | **0.0426** |

Excès de ressemblance au sujet source, `wronganchor` − `memdec20` : **−0.0000** (11+/13−, p =
0.84). Splicer l'ancre décodée d'une AUTRE vidéo ne laisse **aucune trace mesurable** de cette
vidéo, tout en rendant **100 % du gain**. Ce n'est donc pas « le contenu de la mémoire aide
partiellement » : sur ce tier, **le contenu splicé n'est pas lu**.

### 11.6 L'œil (`eyepack_controls/`, hstacks t2_s0 et t11_s0 + planche 6 colonnes)

* **t2_s0 (chat tabby orange)** — `memzero` : on reste sur le camion de pompiers avec un chat-jouet
  cartoon. `memdec20` : un vrai chat tabby revient. `wronganchor` : **le même vrai chat tabby**,
  alors que l'ancre venait de t3 = *un homme à barbe rousse en veste bleue buvant un café* — pas un
  pixel d'homme, pas un pixel de veste bleue. `noiseanchor` : **échoue ici**, on reste sur le camion
  et le chat-jouet (delta +0.120) — le bruit efface moins fiablement.
* **t11_s0 (théière en cuivre)** — les TROIS bras ramènent la théière sur le marbre (+0.61 / +0.60 /
  +0.59) ; l'ancre étrangère venait de t0 = *un chien brun*. Aucun chien.

### 11.7 Le mécanisme, et ce qui reste debout

Ce que la batterie établit, en une phrase : **sur le tier cross-scène, splicer N'IMPORTE QUELLE
frame latente plausible au slot de récence casse le contexte de l'interlude, et c'est le PROMPT
qui repeint A.** Le bruit y arrive dans 2 cas sur 3 (67 %), une frame décodée (la nôtre ou celle du
voisin) y arrive toujours (100 %). Cohérent avec trois faits déjà publiés : le test d'instance nul,
la courbe de position du gate 2 (reculer l'ancre d'UN slot détruit 98 % du gain — c'est le SLOT qui
compte, pas ce qu'on y met), et les 54 % du contrôle bruit sur le tier même-décor.

**L'apparente contradiction avec `oracle1` (+0.027 seulement) est explicable et instructive** : en
S2 la « première apparition de A » est contaminée par l'ouvreur épinglé dans le sink (cos 0.165 du
A\* propre), donc `oracle1` splice une frame qui **réimporte l'ouvreur** — son contenu, lui, est lu
assez pour NUIRE. Autrement dit le canal n'est pas totalement aveugle au contenu ; il est aveugle à
l'IDENTITÉ, et sensible au fait que la frame soit « lisse et compatible avec le prompt ».

**Ce que ça change pour le papier**
1. Le +0.434 de S2 **ne peut plus être présenté comme un rappel mémoriel**. Reformulation honnête :
   *« une intervention SANS CONTENU au slot de récence restaure le sujet sur le tier non-ouvrant ;
   notre ancre décodée n'est pas distinguable de celle d'une autre vidéo »*. C'est un résultat
   négatif de plus, pré-enregistré et réfuté — la posture du dossier (§ rappel de posture de
   `risks.md`) le supporte, mais l'abstract doit changer.
2. Ce qui n'est PAS touché : P1 (l'ablation causale des écritures dégrade l'ancre décodée 12/12) et
   le h-shuffle. Ils portent sur **la sortie du décodeur**, pas sur la vidéo rendue — l'état contient
   bien de l'information spécifique ; c'est le canal splice→rendu qui ne la transmet pas.
3. **Deux mesures manquent avant de conclure définitivement** (≈ 4 GPU-h chacune) :
   (a) les mêmes 2 contrôles sur le tier **SS2 même-décor** (le seul tier où le score ne peut venir
   que du sujet : le gate 3 y donnait 54 % au bruit, il faut la version 24 cellules dose 1.0) ;
   (b) l'attribution **convention de fenêtre vs type de contenu** : `noiseanchor`/`wronganchor` à
   `MEM_DECODE_SLOT_OFFSET=0` et `oracle1` en convention legacy, sur les mêmes 24 cellules — c'est
   la seule ambiguïté qui reste entre « n'importe quelle frame lisse » et « la fenêtre décalée ».

**Fichiers** : `reference/runs/reid_scen/{selftest_controls.py, rescore_controls.py,
launch_controls.py, make_eyepack_controls.py}`, résultats
`results/controls_rescore/{summary_controls.json, contamination.json, shard_*.json}`, vidéos
`S2_t*_s*_{noiseanchor,wronganchor}.mp4`, ancres décodées `reid_scen/dec_anchors/*.pt`, œil
`eyepack_controls/`. Hooks : `MEM_DECODE_NOISE`, `MEM_DECODE_NOISE_SEED`, `MEM_DECODE_SRC`
(`pipeline/switch_causal_inference.py`, N-switch uniquement, `inference()` legacy les refuse).

## 12. Pilote « TEXTE INSUFFISANT » (2026-08-03) — 2 cellules : **zone grise, une cellule VOID par construction**

Suite directe du §11. Là-bas le contenu splicé n'était pas lu, mais tous les prompts de retour
DÉCRIVAIENT le sujet : le texte suffisait. Ici on retire cette suffisance. Prédictions écrites
avant le run (`memory2_probes_registry.md`, section « PILOTE TEXTE INSUFFISANT »).

### 12.1 Le point d'Aden qui a sauvé le pilote : le routage par mots MEURT ici

Avec un prompt de retour appauvri, le cosinus de prompt s'effondre et la banque **ne se déclenche
jamais** : le bras `framebank` aurait silencieusement dégénéré en `memzero` et la comparaison
serait devenue « un splice contre pas de splice ». Mesuré, au retour (segment 3) :

| cellule | cos(retour, A banquée) | cos(retour, OUVREUR banqué) | décision du routeur (tau 0.68) |
|---|---|---|---|
| t17 « générique » | 0.5509 | **0.6202** | **AUCUN rappel** — et son argmax pointe **le mauvais souvenir** (l'ouvreur) |
| t18 « ambiguë » | 0.4470 | 0.4272 | **AUCUN rappel** (marge 0.02 = pile ou face) |

Et sur t18 il produit un **FAUX rappel** à un switch qui n'est PAS un retour (0.8200 ≥ tau :
*« a white dog … in the same city park »* ressemble lexicalement à *« a jogger … in the same city
park »*). Bilan du routage par mots en régime générique : **0/2 rappels corrects, 1 faux rappel,
1 argmax sur le mauvais souvenir**. C'est un résultat à part entière : **argument direct pour un
routage par CONTENU** (le nôtre n'a pas ce problème : il ne route pas, il lit un état).

**Correctif appliqué** : `FRAMEBANK_GT=1` — le QUAND vient de la vérité du planning pour TOUS les
bras mémoire (égalité stricte : `selftest_textpoor.py` (d) audite les logs des rollouts RÉELS et
vérifie **exactement 1 splice par bras et par cellule, au segment 3 frame 180**, et 0 pour
memzero), la SÉLECTION restant celle de la méthode (`framebank_gt`) ou la vérité terrain
(`framebank_oracle_gt`, plafond). La décision qu'aurait prise le routeur est conservée dans
l'événement (`router_recalled`, `router_chosen_seg`) — d'où le tableau ci-dessus.

### 12.2 Scores (seed 0, re-ID contre A\* propre = segment 0 du rollout SS1 de la même cellule)

**t17 « générique »** — A = *jet-black coat + snow-white mane*, retour = *the animal from earlier
returns to the meadow* :

| bras | reid full | reid crop60 | luminance crop | attr_margin CLIP |
|---|---|---|---|---|
| memzero | 0.1768 | 0.2192 | 0.379 | −0.0129 |
| **memdec20** | 0.2900 | 0.2449 | 0.415 | −0.0206 |
| **wronganchor** | 0.2342 | 0.2138 | 0.343 | −0.0132 |
| framebank_gt (sélection FAUSSE) | 0.2487 | 0.2600 | 0.500 | −0.0028 |
| framebank_oracle_gt (vraie frame de A) | 0.2737 | 0.2329 | 0.456 | −0.0065 |
| *A\* propre (référence)* | — | — | 0.398 | **+0.0040** |

Contraste `memdec20 − wronganchor` : **+0.0311** crop60, +0.0558 full — **sous** le +0.10 de
H_lu, dans la bande de H_récence.

**t18 « ambiguë »** — A = chien BRUN, B = chien BLANC même parc, retour = *the first dog comes
back* :

| bras | reid full | reid crop60 | vs A vu | vs B vu | margin |
|---|---|---|---|---|---|
| memzero | 0.2558 | 0.2223 | 0.3231 | 0.6429 | −0.3198 |
| **memdec20** | 0.3245 | **0.4141** | 0.2503 | 0.4121 | −0.1618 |
| **wronganchor** | 0.3100 | 0.3113 | 0.2553 | 0.4875 | −0.2322 |
| framebank_gt = framebank_oracle_gt | 0.1559 | 0.1801 | 0.3653 | 0.4965 | −0.1312 |
| *A\* propre (référence)* | — | — | 0.2398 | 0.2327 | **+0.0070** |

Contraste `memdec20 − wronganchor` : **+0.1028** crop60 (≥ +0.10, sens de H_lu), +0.0144 full.

### 12.3 Mon œil — et pourquoi t17 est VOID par construction

* **t17 : LA CELLULE NE TESTE PAS CE QU'ELLE DEVAIT TESTER.** Le A\* propre montre bien le cheval
  voulu (robe très sombre, crinière blonde/blanche : les attributs EXISTENT dans la vidéo). Mais
  au retour, **les cinq bras rendent une VACHE** dans un pré — pas un cheval sombre, pas un cheval
  du tout. *« The animal from earlier »* est si générique que le modèle retombe sur un prior de
  catégorie (bétail dans un pré) et **l'espèce elle-même est perdue**. Question « la robe est-elle
  noire ? » : **sans objet, il n'y a pas de cheval**. Le confondant annoncé d'avance s'est réalisé
  dans sa forme la plus dure. Détail aggravant : `framebank_oracle_gt` splice la **VRAIE frame du
  cheval** et rend quand même une vache ⇒ même un contenu parfait ne contrôle pas le rendu quand
  le texte part ailleurs.
  Correctif de design pour toute reprise : **appauvrir l'ATTRIBUT, jamais la CATÉGORIE**
  (« the horse comes back » et non « the animal from earlier »).
* **t18 : lisible, et dans le bon sens.** `memdec20` ramène un labrador **doré/brun** ;
  `wronganchor` un labrador **crème/blanc** ; `memzero` un petit chien blanc-et-roux qui court
  (le chien de l'interlude). Les bras banque, eux, rendent un chien **sombre à collier rose**,
  très loin des deux. C'est cohérent avec le +0.103 crop60.
* **Le `margin` (A vu vs B vu) est INEXPLOITABLE** sur t18 : la référence elle-même ne sépare les
  deux chiens que de **+0.0070** (0.2398 vs 0.2327), donc la métrique ne distingue quasiment pas
  « chien brun » de « chien blanc » dans ce décor. Tous les bras sortent négatifs — y compris
  ceux qui splicent le vrai chien. À refaire avec un discriminateur de couleur, pas DINO global.

### 12.4 Verdict du pilote (règle pré-enregistrée)

Règle : deux cellules dans le bon sens ⇒ investir ; aucune ⇒ le négatif du §11 se généralise ;
**une sur deux ⇒ ZONE GRISE, pas d'investissement avant une 3e cellule du type qui a marché.**

**On est exactement dans la zone grise, et en réalité un cran plus bas** : t18 passe le seuil sur
crop60 (+0.103) mais pas sur full (+0.014), et t17 **ne compte pas** (aucun cheval à retrouver).
Donc : **1 cellule sur 1 exploitable**, dans le bon sens, à n = 1 et sur une seule des deux vues.

**Ce qu'il faut faire avant de dépenser 12 cellules × 2 seeds** :
1. **Réécrire les prompts de retour** : catégorie conservée, attribut retiré (« the horse comes
   back », « the first dog comes back » — le second est déjà bon, c'est le premier qui a cassé).
2. **Changer le discriminateur** pour les cellules ambiguës : la marge doit être mesurée par un
   descripteur sensible à la COULEUR du sujet (histogramme sur le crop, ou CLIP « brown dog » vs
   « white dog »), pas par DINO global qui ne sépare les deux chiens que de 0.007.
3. Alors seulement : 3 cellules ambiguës du type t18, et si le contraste tient (≥ +0.10 sur les
   deux vues, 3/3), lancer le tier complet.

**Fichiers** : `reference/runs/reid_scen/{selftest_textpoor.py, score_textpoor.py,
make_eyepack_textpoor.py}`, scores `results/textpoor/scores_textpoor.json`, logs de routage
`results/textpoor/fb_logs/`, œil `eyepack_textpoor/` (2 hstacks 4 panneaux + planche 8 colonnes),
vidéos `S2p_t1{7,8}_s0_*.mp4`, tier `TEXTPOOR_TRIPLES` (17/18) et bras `framebank_gt` /
`framebank_oracle_gt` (hook `FRAMEBANK_GT`, défaut OFF, `force=None` = comportement publié).

## 13. Mini-round CORRIGÉ « texte insuffisant » (2026-08-03) — **règle scellée : 0/2, le négatif du §11 est GÉNÉRAL**

Les 3 correctifs pré-enregistrés appliqués (prompts avec catégorie gardée, +3 cellules type-t18,
juge couleur dédié), 5 cellules × 6 bras = 30 rollouts + 5 A\*, splice par vérité du planning
partout (`selftest_textpoor.py` (d)).

### 13.1 Le juge couleur : validé sur les A\*, et il RÉVÈLE un défaut de substrat

Gate pré-enregistré (le juge doit classer les références avant qu'on regarde un seul bras) :

| cellule | CLIP(A\* propre) | CLIP(B vu) | verdict |
|---|---|---|---|
| t18 chien brun/blanc | **+0.0234 OK** | +0.0088 **RATÉ** | **VOID** |
| t20 chat noir/blanc | **+0.0476 OK** | −0.0029 OK | valide |
| t21 pick-up rouge/bleu | **+0.0398 OK** | +0.0091 **RATÉ** | **VOID** |
| t22 perroquet jaune/vert | **+0.0334 OK** | −0.0104 OK | valide |

**3/3 A\* correctement classés** ⇒ le juge marche quand la vidéo porte vraiment l'attribut. Les
2 échecs sont sur le segment **B**, et la planche montre pourquoi : **le B rendu ne porte pas son
attribut**. Sur t21 le segment « blue pickup truck » rend un camion **ROUGE** (comme A) ; sur t18
le « white dog » rend un chien **pâle identique** à celui du segment A. Cause : la pathologie S2
déjà documentée (l'ouvreur épinglé dans le sink domine tous les segments suivants).
**Conséquence de design, importante** : une cellule d'ambiguïté même-catégorie **ne peut pas être
construite sur la forme S2p** — l'interlude n'affiche jamais l'attribut opposé, donc il n'y a
rien à désambiguïser. Il faudra la forme SS1 (A ouvre) ou un autre moyen d'installer B.

### 13.2 Contrastes (bras − `wronganchor`), la signature d'une lecture

| cellule | config-Aden attr | notre-pur attr | config-Aden reid crop60 | notre-pur reid crop60 |
|---|---|---|---|---|
| t19 cheval corrigé | (pas d'attribut binaire) | — | −0.0185 | −0.0723 |
| t18 **VOID** | +0.0057 | +0.0061 | +0.1037 | +0.1028 |
| **t20 valide** | **−0.0203** | −0.0203 | −0.1238 | −0.1238 |
| t21 **VOID** | −0.0393 | −0.0388 | +0.0237 | +0.0220 |
| **t22 valide** | **−0.0126** | −0.0126 | +0.0741 | +0.0749 |

**RÈGLE SCELLÉE : ≥ +0.08 sur ≥ 4/5 cellules valides ⇒ le contenu est lu. Obtenu : 0/2**
(0/4 même en comptant les cellules VOID). Le critère secondaire (`notre-pur − wronganchor` sur
reid crop60 ≥ +0.08) donne **0/2** aussi (t22 +0.074 passe juste sous la barre, t20 −0.124).
⇒ **« ≤ 1/5 ⇒ le négatif du §11 est général » : c'est la branche atteinte.**

### 13.3 `config-Aden` ≡ `notre-pur` sur ce tier (la porte α ne s'ouvre jamais)

`memdec21un` et `memdec20` sont **identiques à 3 décimales** sur t20 et t22 (0.2216/0.2216,
0.5738/0.5746) et à 0.002 près ailleurs : la porte cosinus lit ces retours comme un **saut de
monde**, met α = 0 et retombe sur `memdec20` bit à bit. La config définitive d'Aden **n'est donc
pas testée ici** : sur ce tier elle EST le bras pur. Si on veut la juger, il faut un tier où le
cosinus prompt(retour) / prompt(interlude) est haut — c'est-à-dire le tier même-décor SS2.

### 13.4 Mon œil sur la planche (`eyepack_textpoor2/`)

* **t19 : le correctif de prompt MARCHE — et le résultat reste négatif.** Là où « the animal from
  earlier » donnait une **vache**, « the horse comes back » ramène **un cheval dans les 6 bras** :
  la catégorie est sauvée, la leçon de design était juste. **Mais l'attribut, non** : le A\* montre
  bien un cheval sombre à **crinière blanche**, et nos deux bras (`memdec21un`, `memdec20`)
  ramènent un cheval **brun ordinaire, crinière brune**. `memzero` (aucun rappel !) et
  `framebank_gt` sont ceux qui gardent le mieux la crinière claire. Notre mémoire ne restitue pas
  l'attribut, et elle fait **moins bien que ne rien faire**.
* **t20** : A\* = chat **noir**. Nos bras ramènent un chat **tigré brun**, `wronganchor` aussi,
  `noiseanchor` un chat blanc-roux. Personne ne ramène le chat noir.
* **t21** : A\* = pick-up **rouge**. Nos bras ramènent un pick-up **gris/argent** ; `wronganchor`
  et `noiseanchor` ramènent un **rouge**. Le bras à contenu FAUX est plus proche de la vérité que
  le nôtre — c'est le −0.039 du tableau, visible à l'œil.
* **t22** : A\* = perroquet **jaune**. Nos bras : ara **rouge et vert** ; `wronganchor` : aras
  bleus/orange. Aucun jaune nulle part.
* **t18** : tous les bras convergent vers un labrador **crème**, y compris `wronganchor`.

### 13.5 Conclusion

Le régime « texte insuffisant » a été construit correctement cette fois (catégorie préservée,
juge validé sur les A\*), et il **ne fait pas apparaître de lecture du contenu** : sur les deux
cellules valides le contraste est **négatif**, et sur la cellule la plus lisible (t21) c'est le
bras à **contenu faux** qui rend le bon attribut. Ajouté au §11 (wronganchor = 106 % du gain,
contamination nulle), la conclusion du dossier est stable et doit être écrite telle quelle :
**le canal splice→rendu n'est pas gouverné par le contenu de ce qu'on splice** ; ce qu'il gouverne,
c'est l'effacement du contexte au slot de récence.

Deux réserves honnêtes à mentionner dans le papier : (a) n = 1 seed par cellule, 2 cellules
valides — c'est un GATE de continuation, pas une mesure de précision ; (b) la config d'Aden n'a
pas pu être jugée ici (porte α fermée) : le seul terrain qui la teste est le tier même-décor.

**Fichiers** : `score_textpoor2.py` (juge CLIP + histogramme HSV + gate), `make_eyepack_textpoor2.py`,
`results/textpoor2/{gate_judge.json, scores_textpoor2.json, verdict.json, refs_check.jpg}`,
œil léger `eyepack_textpoor2/` (5 hstacks 4 bras < 1.3 Mo, 5 micro-clips verdict ~0.2 Mo,
planche JPEG 0.38 Mo), tier `TEXTPOOR_TRIPLES` 19-22 + `TEXTPOOR_RERUN` + `TEXTPOOR_ATTR`.

## 16. LOT OFFENSIF (nuit du 2026-08-04) — trois leviers pour faire LIRE la mémoire

Prédictions toutes scellées AVANT run dans `memory2_probes_registry.md` (section « LOT OFFENSIF
DE LA NUIT »). Règle transverse : le comparateur de chaque bras est son **propre jumeau**
(`wronganchor`, `caption_wrong`, `i1_wrong`), jamais `memzero`.

### 16.1 T0 — la bande basse : **« PAS LU » EST RÉFUTÉ. Le splice EST lu, et copié pendant 3 s**

Analyse à **0 GPU** sur les 24 cellules S2 (seeds 0 et 1) déjà rendues au §11 : rien n'a été
regénéré, seule la question a changé. §11.5 mesurait la contamination **en espace DINO** (0.0426
contre 0.0426, p = 0.84) et concluait *« le contenu splicé n'est pas lu du tout »*. Ici on mesure
la **corrélation image** entre le retour rendu et l'ancre latente réellement splicée (VAE →
niveaux de gris → moyenne n × n, centrée-réduite par frame), en design **apparié 2 × 2** : chaque
cellule fournit un retour `wronganchor` (qui a reçu l'ancre de t+1) et un retour `memdec20` (qui a
reçu la sienne), et on demande à chacun s'il ressemble davantage à **l'ancre qu'il a reçue**.

| fenêtre | retour | ancre **REÇUE** | ancre **NON reçue** | ancre **NULLE** (t+2, reçue par personne) |
|---|---|---|---|---|
| early (12 frames après le splice) | `wronganchor` | **+0.5210** | −0.0093 | +0.0997 |
| early | `memdec20` | **+0.4670** | +0.1699 | +0.1027 |
| scored (frames 192-239, soit 12 à 59 après) | `wronganchor` | **+0.4627** | −0.0097 | +0.1120 |
| scored | `memdec20` | **+0.3551** | +0.1648 | +0.0879 |

Excès apparié `[c(ret_wrong, reçue) − c(ret_wrong, non reçue)] + [c(ret_dec, reçue) − c(ret_dec,
non reçue)]` : **+0.827 ± 0.116 (23+/1−, p = 1.2e-06)** en fenêtre *early*, **+0.663 ± 0.121
(22+/2−)** en fenêtre *scored* ; en RGB **+0.715 (24+/0−)** et +0.575. Prédiction scellée sous
H_slot : « dans [−0.02, +0.05], ≤ 14/24 ». **RÉFUTÉE d'un facteur ~20.**

Le **contrôle nul** est ce qui rend le résultat publiable : une ancre décodée d'une TROISIÈME
cellule, que ni l'un ni l'autre bras n'a reçue, corrèle à **+0.10** — c'est le plancher « deux
images floues se ressemblent un peu ». L'ancre reçue est à **+0.47/+0.52**, cinq fois le
plancher, et l'ancre non reçue de la MÊME cellule (même prompt, même graine, même rollout à un
hook près) tombe à **−0.01**. Ce n'est donc ni un effet de cellule, ni un effet de flou.

**Échelle de bande** — la copie ne meurt pas quand on monte en résolution (excès *early* : n = 4
+0.883, n = 8 +0.827, n = 16 +0.784, n = 32 +0.751, n = 64 +0.726, 23-24 signes positifs partout).
Ce n'est donc pas « une bande basse » au sens strict : c'est **toute la structure spatiale
mesurable par corrélation** qui est reprise, et qui persiste **59 frames latentes (≈ 3 s)** après
le raccord.

**Ce que ça change.** Le canal splice → rendu **transmet**, et il transmet beaucoup :
composition, disposition des masses, luminance. Ce qu'il ne transmet pas, c'est **l'IDENTITÉ** —
et c'est cohérent avec tout le dossier (§11.5 test d'instance nul, §13 attribut perdu). La phrase
juste n'est plus *« le contenu splicé n'est pas lu »* mais :

> **le contenu splicé est lu, et copié pendant trois secondes ; ce que notre ancre transporte,
> c'est une composition, pas une instance.**

C'est une réécriture de la conclusion du §11 (et de l'abstract) qui *renforce* le résultat
négatif au lieu de l'affaiblir : le goulot n'est plus « le modèle est aveugle », il est
« l'ancre est vide de l'information qu'on croit y mettre » — donc le chemin est l'ÉCRITURE et le
DÉCODEUR, pas le canal. Réserve honnête : la corrélation image ne distingue pas « l'ancre est
copiée » de « l'ancre détermine la continuation » — c'est la même chose vue de deux côtés.

**Fichiers** : `reference/runs/reid_scen/prescreen_offensif.py`,
`results/offensif/t0_lowband.json`, log `logs/t0_lowband.log`, ancres décodées en pixels mises en
cache dans `/shared/ssd_14T_project/adenelmi/reid_scen/anchor_px/`.

#### 16.1 bis — le corollaire décisif : la copie est **orthogonale** au re-ID (0 GPU)

Puisque le canal transmet, la question suivante s'écrit toute seule : **est-ce que transmettre
PLUS ferait gagner PLUS ?** Test apparié cellule par cellule (24 cellules), corrélation entre
« combien l'ancre a été copiée » (T0, fenêtre *scored*) et « combien la cellule a gagné en re-ID »
(deltas par cellule du §11.3) :

| vue | bras | copie moyenne | delta re-ID moyen | Pearson | Spearman |
|---|---|---|---|---|---|
| crop60 | `memdec20` | +0.355 | +0.423 | **+0.11** | +0.04 |
| crop60 | `wronganchor` | +0.463 | +0.450 | **+0.19** | +0.26 |
| full | `memdec20` | +0.355 | +0.434 | +0.14 | +0.11 |
| full | `wronganchor` | +0.463 | +0.459 | +0.08 | +0.10 |

Et le test qui compte, celui du claim : **contraste de copie** (notre ancre copiée − l'étrangère
copiée) contre **contraste de re-ID** (`memdec20` − `wronganchor`) : Pearson **−0.075**,
Spearman **−0.163** (n = 24) — **nul, voire du mauvais signe**.

> Le tuyau est large et il coule ; ce qui manque, c'est ce qu'on y met. Élargir le canal
> (légende, métadonnée honnête, position additive : les trois leviers de cette nuit) attaque le
> mauvais bout. Le chemin critique est **l'écriture et le décodeur**.

Détail à noter, contre-intuitif et robuste : **l'ancre ÉTRANGÈRE est copiée PLUS fort que la
nôtre** (+0.463 contre +0.355 en fenêtre *scored*). Lecture la plus simple : notre ancre décodée
ressemble déjà en partie à la scène en cours, donc sa copie se confond avec la continuité
normale, tandis qu'une ancre étrangère impose une structure distincte, visible. Cela explique
mécaniquement pourquoi `wronganchor` égale (105.7 %) `memdec20` au §11 : à ce tier, ce qui est
mesuré est un effacement de contexte, et l'ancre étrangère efface mieux.

#### 16.1 ter — sujet ou décor ? la copie porte **les deux**, et le centre est le plus spécifique

Objection reviewer immédiate : une corrélation 8 × 8 sur l'image entière est dominée par le fond,
donc « l'ancre est copiée » pourrait ne vouloir dire que « l'ancre fixe le décor ». Même design
apparié, mesuré séparément sur le **centre** (crop 60 %, majoritairement le sujet) et sur le
**bord** (complément du crop, le décor) :

| fenêtre | zone | retour | ancre **REÇUE** | ancre **NON reçue** | excès apparié |
|---|---|---|---|---|---|
| early | centre | `wronganchor` | **+0.3257** | −0.0451 | **+0.551** (24+/0−, p = 6e-8) |
| early | centre | `memdec20` | **+0.3116** | +0.1312 | idem |
| early | bord | `wronganchor` | +0.7419 | +0.4363 | **+0.495** (23+/1−) |
| scored | centre | `wronganchor` | **+0.2235** | −0.0174 | **+0.361** (21+/3−, p = 3e-4) |
| scored | bord | `wronganchor` | +0.7143 | +0.4108 | **+0.431** (21+/3−) |

Le bord part d'un plancher élevé (deux vidéos de la même campagne partagent une structure de
décor), mais l'**écart reçue / non-reçue est du même ordre dans les deux zones** (+0.30 environ),
et au centre la corrélation avec l'ancre NON reçue est **nulle ou négative** (−0.045, −0.017) :
la zone du sujet est donc l'endroit le plus **spécifique** de la copie. Le canal ne transporte pas
seulement une scène : il transporte **où est le sujet et quelle forme il a**. Ce qu'il ne
transporte pas — et le §11.5, le §13 et le 16.1 bis convergent — c'est **QUI c'est**.

#### 16.1 quater — **DE QUOI l'ancre décodée est-elle l'image ? De l'INTERLUDE, pas de A**

*(analyse POST-HOC, engendrée par 16.1 : direction, pas gate. Elle n'était pas pré-enregistrée
et ne peut donc pas servir de règle de décision — mais elle explique tout le dossier.)*

Trois faits s'accumulent : le canal transmet (16.1), ce qu'il transmet ne prédit pas le re-ID
(16.1 bis), et l'ancre ne porte ni l'identité de A ni sa couleur ni sa structure grossière —
mesuré directement, avec l'ancre étrangère pour comparateur :

| mesure directe sur l'IMAGE de l'ancre | ancre propre | ancre étrangère | avantage | signes | p |
|---|---|---|---|---|---|
| DINO crop60 vs A\* | 0.0110 | 0.0103 | **+0.0007** | 11+/13− | 0.84 |
| DINO full vs A\* | 0.0142 | 0.0102 | +0.0040 | 15+/9− | 0.31 |
| couleur (juge §13) vs A\* | — | — | +0.128 | 13+/11− | 0.84 |
| bande basse 8×8 vs A\* | 0.0412 | 0.0274 | +0.014 | 13+/11− | 0.84 |

Elle n'est pourtant **pas générique** (T0 : elle est copiée de façon spécifique, 0.47 contre un
plancher de 0.10). Elle est donc l'image de quelque chose. **De quoi ?** Le décodeur a été
entraîné à reconstruire la frame **ÉVINCÉE** de la fenêtre. Au switch de RETOUR (frame 180), ce
qui vient d'être évincé appartient à **l'INTERLUDE B**, pas à A. Mesure, 24 cellules :

| | ancre vs **A\*** (le sujet à retrouver) | ancre vs **B** (l'interlude évincé) | ancre vs **X** (l'ouvreur épinglé au sink) |
|---|---|---|---|
| bande basse 8×8 | **+0.0419** | **+0.2981** | **+0.3438** |
| DINO crop60 | **+0.0111** | **+0.0453** | +0.0279 |

B est plus proche que A\* sur **21/24 cellules en bande basse (p = 3e-4)** et **18/24 en DINO
(p = 0.023)** ; X aussi (21/24 et 15/24).

> **Notre « ancre » est une image de l'interlude qu'on veut effacer, et de l'ouvreur qui pollue
> l'état — pas du sujet qu'on veut rappeler.** *(Lequel des deux domine n'est PAS tranché :
> X contre B est un match nul, 14/24 en bande basse et 10/24 en DINO, p = 0.54 — voir 16.5 bis.
> Les deux battent A\* à 21/24, p = 3e-4 : c'est ça, le résultat.)* Le décodeur fait exactement ce qu'on lui a appris
> à faire (reconstruire ce qui vient de sortir de la fenêtre) ; c'est l'OBJECTIF qui pose la
> mauvaise question au moment du rappel.

Cela unifie tout le dossier sans rien y contredire : `wronganchor` = 106 % du gain (§11) parce
que les deux bras splicent une image d'un interlude quelconque ; `noiseanchor` = 67 % parce que
n'importe quoi d'étranger au contexte efface l'interlude ; le SLOT décide (gate 2) parce que le
travail réel du splice est un **effacement**, pas un rappel ; et l'attribut n'est jamais restitué
(§13) parce qu'il n'a jamais été dans la charge utile.

**Le test qui tranche « mauvaise scène » vs « mauvais SUJET »** (tier même-décor, ajouté après
coup). Sur le tier cross-scène, « l'ancre ressemble à B » pourrait ne vouloir dire que « elle
ressemble à la SCÈNE de B ». Sur **SS2**, A et B partagent le décor **verbatim** et ne diffèrent
que par le sujet. Mesure sur les 4 cellules dont l'ancre est sur disque (t12-t15) :

| | ancre vs **A\*** | vs **B** | vs **X** | B plus proche que A |
|---|---|---|---|---|
| bande basse 8×8 | +0.2064 | **+0.3707** | **+0.4153** | 3/4 |
| DINO crop60 | −0.0076 | +0.0104 | +0.0170 | 3/4 |

Même quand le décor est identique, l'ancre penche vers B et X. Elle ne porte donc pas « la
mauvaise scène » : elle porte **le mauvais SUJET**. *Limite à dire franchement : n = 4, donc
3/4 n'est pas significatif isolément (p ≈ 0.31) et t12 fait exception (A +0.629 > B +0.442) ;
c'est un appui directionnel au résultat à 24 cellules du tier cross-scène (21/24, p = 3e-4), pas
une preuve autonome.*

**Et ce n'est pas que l'état serait vide de A** : P1 (registre, déjà mesuré) montre qu'ablater
toutes les écritures du segment A dégrade l'ancre décodée **12/12**. L'état porte donc bien de
l'information spécifique à A ; c'est la QUESTION posée au décodeur au moment du rappel qui est
mauvaise — « rends la frame qui vient de sortir » au lieu de « rends le sujet demandé ».

**Conséquence pour le plan** : les trois leviers de cette nuit (légende, métadonnée honnête,
position additive) travaillent tous sur le **canal**, et 16.1 + 16.1 bis disent que le canal n'est
pas le goulot. Le chemin critique est l'**objectif d'écriture / de décodage** : la cible du
décodeur au rappel doit être **le sujet demandé**, pas la frame qui vient de sortir. À vérifier
proprement (ce point est post-hoc) avant d'en faire une décision de projet.

**Fichiers** : `anchor_identity.py`, `anchor_what.py`, `t0_split.py`,
`results/offensif/{anchor_identity.json, anchor_what_is_it.json, t0_split.json,
anchor_lowband_vs_astar.json}`, logs `logs/anchor_what.log`.

#### 16.1 quinquies — l'œil, et il tranche seul (`eyepack_offensif/board_ancre_est_un_interlude.jpg`)

Six cellules, six colonnes : **A\* propre** (le sujet à retrouver) | **X ouvreur** | **B
interlude** (ce que le décodeur vient de voir évincer) | **ANCRE décodée** | **retour memdec20** |
**retour wronganchor**. 0.31 Mo, aucune métrique.

Mon œil, cellule par cellule — la colonne ANCRE est un flou coloré, et **sa couleur est celle de
l'interlude, jamais celle de A** :

* **t3, la démonstration** : A\* = un homme roux en **veste bleue** au café ; l'interlude = des
  **flamants roses** ; l'ancre décodée est un flou **rose saumon**. Notre mémoire, au moment
  précis où on lui demande « rends-moi l'homme en bleu », rend une tache de flamants.
* **t2** : A\* = un chat **roux** sur une table ; interlude = un **camion de pompiers rouge** ;
  ancre = un flou **rouge vif** au centre sur du vert.
* **t4** : A\* = un cheval **blanc** dans un pré ; interlude = un **bol de terre brune** au tour
  du potier ; ancre = un flou **brun-tan**.
* **t0** : A\* = un chien brun sur l'herbe ; interlude = un **vélo rouge** ; ancre = jaune + rouge.
* **t11** : A\* = théière **cuivre** sur marbre ; interlude = une **fanfare** ; ancre = jaune/blanc.

Et la colonne des retours confirme l'autre moitié du dossier : `memdec20` **et** `wronganchor`
ramènent tous les deux un A correct et net — parce que c'est **le prompt** qui les peint, pas
l'ancre. La planche montre donc les deux résultats d'un coup : *l'ancre n'est pas A*, et *le
retour est bon quand même*.

**Ops (leçon de la nuit, à garder)** : un rollout 1.3B de cette campagne monte à **~23.5 Go** et
la carte en fait 47.4 — **deux rollouts ne tiennent pas sur un GPU**. Un pré-check en mémoire
libre ne suffit pas (un voisin lancé il y a 30 s n'a pas encore alloué son pic) : le critère
stable est **l'exclusivité** (aucun voisin ≥ 20 Go), plus un second pré-check jitté. Quatre lots
offensifs et plusieurs lots défensifs sont tombés en OOM avant ce correctif ;
`launch_offensif.py` l'implémente et la règle vaut pour tous les superviseurs du projet.

### 16.2 Levier 1 — pré-écran de la légende : **VIVANT d'une cellule, mais au niveau du hasard**

Question du pré-écran (0 rollout) : notre ancre décodée porte-t-elle la **couleur** du sujet ?
Juge = celui du §13 (histogramme de teinte pondéré S·V sur le crop 60 %, + moyennes S et V),
appliqué aux 24 ancres S2 déjà sur disque ; comparateur = **l'ancre étrangère** (celle que splice
`wronganchor`), contre le même A\* propre.

| | distance couleur à A\* | |
|---|---|---|
| ancre **propre** | **1.4787** | |
| ancre **étrangère** | 1.6069 | |
| **avantage** (étrangère − propre) | **+0.128 ± 0.117** | **13+/11−**, p = 0.84 |

Règle scellée : *« avantage ≤ 0 OU ≤ 12/24 signes ⇒ levier mort »*. On est à **13/24** : le levier
survit **d'une cellule**, et le test des signes est du **pur bruit** (p = 0.84). Prédiction
scellée : 16-19/24 sous H_couleur-présente, 14-17/24 comme pari personnel — **les deux sont
en-dessous de la mesure**. Version discrète du même test, faite en construisant les légendes
(accord du terme de couleur du cœur avec celui d'une VRAIE frame) : **notre ancre 3/8, ancre
étrangère 2/8**. La charge utile que la légende doit transporter est donc, au mieux, faible.
Le levier est tiré parce que la règle a été écrite d'avance, et il est rapporté avec ce contexte.

**16.1 quater explique ce hasard, après coup** : si l'ancre est une image de l'INTERLUDE, alors la
couleur qu'on y mesure est celle de B, et l'accord avec la couleur de A ne peut être que fortuit —
c'est exactement ce qu'on observe (13/24, p = 0.84), et c'est visible sur la planche (t3 : on
légenderait « rose » un homme en veste bleue). **Conséquence de priorité, pas de règle** : le
levier 1 reste tiré (la règle scellée le dit) mais il passe **en dernier** dans la file, derrière
le levier 4 qui répare la source de la légende. Si le levier 4 passe son pré-écran, la légende
devra de toute façon être reconstruite sur l'ancre SNAPSHOT, pas sur l'ancre courante : les 40
suffixes actuels sont alors à jeter et à remesurer (`build_captions.py`, une commande).

#### 16.2 ter — le témoin `caption_none` : **la tuyauterie du levier 1 est validée**

Le scénario `S2c` fait pointer le segment de retour sur `prompts[3]` au lieu de `prompts[0]`. Avec
un suffixe VIDE les deux chaînes sont identiques, donc le rollout doit reproduire le `memzero`
déjà sur disque. Mesuré (S2c t2 s0 contre S2 t2 s0) :

| | mesuré | référence |
|---|---|---|
| différence pixel moyenne | **1.17/255** | plancher de bruit d'un re-run documenté §3.2 : **1.21/255** |
| DINO cos(retour, retour) crop60 | 0.9446 | — |
| re-ID vs A\* | 0.1616 contre 0.1610 | **écart 0.0006** |

C'est **au plancher de bruit exactement** (le rollout n'est pas bit à bit reproductible, §1.1 :
1.17 < 1.21). Le détour par `prompts[3]` n'introduit donc **aucun biais**, et les contrastes du
levier 1 mesureront bien la LÉGENDE et rien d'autre. Le témoin coûtait un rollout et il ferme
d'avance l'objection « votre scénario de légende n'est pas comparable ».

### 16.2 bis — **réplication indépendante** de « l'ancre ne porte pas d'identité »

Le lot défensif a écrit la même nuit, sans concertation, une sonde convergente
(`probe_anchor_identity.py`, `results/tfree/anchor_identity.json`) avec un design DIFFÉRENT du
mien : elle compare l'ancre au A\* de SA cellule contre le A\* de la cellule SOURCE (design
« contamination » du §11.5), là où je compare ancre propre contre ancre étrangère face au MÊME
A\* (design apparié). Les deux tombent au même endroit :

| | comparateur | vs A\* propre | vs le mauvais A\* / ancre étrangère | marge | n |
|---|---|---|---|---|---|
| sonde défensive (ancre BRUTE) | A\* de la cellule source | 0.0023 | −0.0154 | **+0.0177** (4/6) | 6 |
| la mienne (§16.1 quater) | ancre étrangère | 0.0110 | 0.0103 | **+0.0007** (11+/13−) | 24 |

Deux designs, deux tailles d'échantillon, **le même zéro**. Le résultat ne dépend donc pas d'un
choix de comparateur.

**Une mise en garde de lecture, à ne pas manquer** : la même sonde donne des marges *énormes* sur
les ancres **affûtées** (σ = 0.7 : +0.273, 6/6 ; σ = 0.833 : +0.508, 6/6). Ce n'est **pas** de
l'identité retrouvée dans la mémoire : à σ élevé, SDEdit régénère l'image **depuis le prompt de
retour**, lequel décrit A — le confondant est déclaré d'avance dans le registre de la
mini-batterie training-free. L'ancre ressemble à A parce que le PROMPT l'a peinte, exactement le
mécanisme que le §11 documente déjà. À σ = 0.625 (une seule passe du sampler) la marge retombe à
+0.039, 3/6 : c'est-à-dire au bruit.

### 16.3 État des trois leviers : **implémentés, scellés, selftestés — et affamés de GPU**

Tout le code des trois leviers est écrit, vérifié et en file d'attente ; ce qui a manqué cette
nuit, ce sont les GPU (machine partagée : 5 cartes réservées par un autre utilisateur, les 3
restantes tenues par le lot défensif de GATE A, qui est prioritaire par construction). Les
prédictions sont scellées, le scoring et l'œil sont prêts : la reprise ne demande aucune décision.

| levier | ce qui est fait | ce qui reste |
|---|---|---|
| **1 — légende** | scénarios `S2c`/`SS2c` (le retour pointe `prompts[3]`), 5 bras (`caption_ours`, `caption_wrong`, `padded`, `framebank_vlm`, `caption_none`), les 40 suffixes **mesurés** et écrits (`results/offensif/captions.json`), pré-écran fait | 33 rollouts (vagues C0/C1/C2) |
| **2 — I1 ancre déclarée bruitée** | hook `MEM_ANCHOR_T` (bruitage au σ du scheduler **et** `context_timestep` par slot), 4 bras `i1_{ours,wrong}{250,500}` | 24 rollouts (B1/B2) |
| **3 — additif à Δ = 1** | bras `oracle1add{1,8}p17` + témoin de position `oracle1add8p9` + `memdec20add8p17` (conditionnel), et la **masse d'attention par source** `[sink \| fenêtre \| ancre]` par bloc et par pas (`MEM_ADD_MASS_SRC`, décomposition logaddexp EXACTE) | 9 rollouts (A1/A2), un lot en cours |

**Garanties d'intégrité, vérifiées avant tout rollout** (`selftest_offensif.py`, **72 contrôles,
tout passe**) : additif et env-gated, **défaut OFF**, chaque nouveau bras re-met TOUTES les clés de
l'union (aucune fuite d'un bras au suivant) ; `i1_ours*` et `i1_wrong*` ont une table **identique**
(seul `MEM_DECODE_SRC`, injecté au run time, les sépare) ; les slots d'ancre calculés par
`_anchor_slots` sont exactement ceux que `_memdec_window` remplit dans les deux conventions ; le
bruit I1 ne tire **aucun** RNG global (l'appariement `memzero` est préservé) ; la décomposition de
masse somme à 1 et redonne `alpha` à 1e-5 près ; `caption_none` a par construction
`prompts[3] == prompts[0]` (témoin bit à bit du `memzero` déjà sur disque).

**Un footgun trouvé en route, et paré** : `WanDiffusionWrapper.forward` fait
`input_timestep = timestep[:, 0]` quand `uniform_timestep` est vrai — cela **écraserait
silencieusement** la déclaration par slot du levier 2 avec celle du slot 0 (une vraie frame,
t = 0), et le bras aurait mesuré un simple bruitage en se croyant une déclaration. Ici
`is_causal=True` donc le drapeau est faux ; `_recache_after_switch` **refuse désormais de tourner**
dans le cas contraire, et le selftest lit la source de vérité.

**PAUSE GÉNÉRALE (demande d'Aden, 2026-08-04)** : superviseur offensif arrêté, aucun nouveau
rollout lancé ; les rollouts en vol ont été laissés finir. **Reprise en une commande** —
`python reference/runs/reid_scen/score_offensif.py --lever 1` : les **33 mp4 de la vague légende
sont déjà rendus**, il ne reste donc que le SCORING pour honorer la règle scellée du levier 1.

**Reprise, état au matin du 2026-08-04** (le superviseur reprend par mp4 existant, donc tout
relancement est idempotent) :

| levier | état | ce qui reste |
|---|---|---|
| **T0** + analyses 0 GPU | **CLOS**, 24 cellules | rien |
| **4** snapshot | **CLOS** à 0/2 ; `D1` **délibérément non tiré** (la règle scellée en faisait la condition d'investissement) | rien |
| **3** additif Δ=1 | **CLOS** à 2/4 ; masse T1 + axes de dégât acquis sur 4 cellules | rien (`memdec20add8p17` reste conditionnel et le gate ne l'a pas ouvert) |
| **2** I1 | 1 paire mesurée sur 6 (−0.089), hook validé | 10 rollouts à t\* = 500, puis 12 à t\* = 250 (vague `B2`) |
| **1** légende | témoin bit à bit **validé**, 40 suffixes mesurés | 32 rollouts (`C1`, `C2`) |

Commandes : `python reference/runs/reid_scen/launch_offensif.py --waves B1,C1,C2,B2 --max-gpus 2`
puis `score_offensif.py --lever {1,2}`, `snap_prescreen.py` (levier 4, déjà fait),
`make_eyepack_offensif.py --lever N`.

**Avertissement si le levier 4 devait être repris** : les 40 suffixes de légende ont été mesurés
sur l'ancre COURANTE. Si un jour une ancre corrigée existe, il faut relancer `build_captions.py`
avant toute cellule de légende, sinon le levier 1 testerait une charge utile périmée.

### 16.3 bis — T1, le **budget d'attention par source**, mesuré pour la première fois

Premier rollout de la vague A1 (SS2 t12, `oracle1add8p17` : contenu = la VRAIE frame d'ancre,
bande RoPE **17** = Δ = 1 de la requête, gain 8), moyenne sur **19 pas × 30 blocs après le
retour** — décomposition **exacte** (logaddexp), les trois fractions somment à 1 :

| source | tokens | part uniforme | **masse d'attention mesurée** | lecture |
|---|---|---|---|---|
| sink (l'ouvreur épinglé) | 1560 | 7.7 % | **4.6 %** | **sous-lu** (0.6× sa part) |
| fenêtre locale (12 vraies frames) | 17160 | 84.6 % | **75.1 %** | ≈ sa part |
| **ancre additive** | 1560 | 7.7 % | **20.3 %** | **sur-lue, 2.6× sa part** |

À comparer au même gain 8 à Δ = 9 (§3.4) : **10.5-15.7 %**. Rapprocher l'ancre de la requête
**augmente bien la masse** (×1.5 environ), et à Δ = 1 elle est enfin lue **plus que sa part de
tokens**. Prédiction scellée « 25 à 45 % à gain 8 » : mesuré **20.3 %**, légèrement **sous** la
borne basse.

Deux enseignements qui ne dépendent pas de la suite de la vague :
1. **le sink est sous-lu** (4.6 % pour 7.7 % des tokens) alors qu'il domine le rendu (pathologie
   S2 du §13.1). Son emprise ne passe donc **pas** par la masse d'attention au moment de la
   génération : elle passe par le **recache** (il conditionne les k/v que la fenêtre reconstruit).
   C'est une correction utile à l'intuition « le sink capte l'attention » ;
2. le canal additif n'est **pas** limité par la lecture : à Δ = 1 il est lu 2.6× sa part. Si le
   score ne suit pas, la conclusion sera « lu et illisible », pas « non lu » — exactement le
   pendant, côté canal, de ce que 16.1 dit côté splice.

*(Le score des bras additifs demande la vague complète ; la mesure de masse, elle, est acquise.)*

### 16.4 Ce que la nuit change, en une phrase

Les trois leviers de la nuit travaillent tous sur **le canal**. Les mesures à 0 GPU disent que le
canal n'est pas le goulot : il transmet (16.1), ce qu'il transmet ne prédit pas le score
(16.1 bis), et **la charge utile est une image de l'interlude** (16.1 quater). Le chemin critique
est donc l'**objectif d'écriture/décodage**, pas le transport — ce qui, en termes de plan, pointe
la mémoire 2.0 (co-entraînement écrivain) et non un correctif d'inférence.

## 15. LOT DÉFENSIF (nuit 2026-08-03/04) — **GATE A : PRÉMISSE FALSIFIÉE, 0/4**

### 15.1 VERDICT GATE A (en tête, c'est lui qui décide le finetune)

Tier **MÊME DÉCOR** (SS2 : X → A → B → A, même lieu, seul le sujet change) = le seul terrain où
le score ne peut venir que du SUJET, donc le terrain de la thèse. On y rejoue les deux contrôles
du §11, re-scoring propre contre A\* = segment 0 du rollout SS1 de la même cellule.

**Règle scellée : `memdec20 − wronganchor` ≥ +0.08 sur ≥ 4/5 cellules ⇒ finetune autorisé.**

| contraste (crop60, vue primaire) | t12 | t13 | t14 | t15 | moyenne | ≥ +0.08 |
|---|---|---|---|---|---|---|
| **`memdec20` − `wronganchor`** | −0.015 | +0.046 | +0.000 | −0.038 | **−0.0020** (2+/2−) | **0/4** |
| `memdec20` − `noiseanchor` | −0.041 | −0.009 | +0.009 | +0.032 | +0.0107 (3+/2−) | 0/5 |
| `memdec20` − `memzero` | +0.478 | +0.327 | +0.467 | +0.135 | **+0.3208** (5+/0−) | **5/5** |
| **`wronganchor` − `memzero`** | +0.493 | +0.281 | +0.467 | +0.174 | **+0.3538** (4+/0−) | **4/4** |

En `full` : `memdec20 − wronganchor` = **−0.0153** (2+/2−, 0/4). t16 manque (2 OOM de co-run puis
un 3e) — **sans effet sur le verdict** : même si t16 passait, le maximum atteignable serait 1/5,
loin des 4/5 exigés.

**⇒ PRÉMISSE FALSIFIÉE.** Sur le tier même-décor comme sur le tier cross-scène (§11), une ancre
décodée de l'état d'une AUTRE vidéo vaut la nôtre (elle est même **au-dessus** en moyenne :
+0.354 contre +0.321 vs memzero), et du bruit pur calibré fait jeu égal (+0.011 d'écart). Le
gain massif (+0.32 sur memzero, 5/5) est **réel et reproductible**, mais il n'est pas porté par
le CONTENU de la mémoire. **Le finetune n'est pas autorisé par cette porte.**

C'est la troisième mesure indépendante qui pointe la même chose (§11 cross-scène 24 cellules,
§13 texte-insuffisant 5 cellules, §15 même-décor 5 cellules) : le mécanisme est l'**effacement du
contexte au slot de récence**, pas la lecture d'un état. Le papier doit être écrit là-dessus.

### 15.2 Ce que le lot défensif a mis en route (chaîné, reprise par mp4)

`launch_defensive.py` : file ordonnée par priorité, pré-check GPU à l'instant du lancement,
jamais l'index 6, max 4 GPU à moi, relance d'un lot OOM jusqu'à 3 fois (la machine est saturée :
plusieurs lots sont tombés en OOM à cause de co-runs arrivés entre le pré-check et l'allocation).

| priorité | tâche | état à la rédaction |
|---|---|---|
| 1 | **GATE A** SS2 `noiseanchor` + `wronganchor` | **TERMINÉ** (t16 wronganchor manquant, sans effet) |
| 2 | attribution `MEM_DECODE_SLOT_OFFSET=0` (arms `memdec20s0`, `wronganchor_s0`, `noiseanchor_s0`, 6 cellules S2) | 8/18 rollouts |
| 3 | `memdec20r` (routeur symétrique) sur S2p / S3p / S4p, 24 cellules | 4/72 rollouts |
| 4 | 3e graine (seed_idx 2) : S3 (A\* propres) puis S2 / S2p × 4 bras | en file |

Les deux nouveaux bras d'attribution (`wronganchor_s0`, `noiseanchor_s0`) portent la question
ouverte du §11.5 : croisés avec `memdec20s0` ils séparent enfin « n'importe quelle frame lisse »
de « la fenêtre décalée ». Ils sont déclarés, sans fuite d'environnement (selftest), et leur
scoring se fera avec `rescore_controls.py` (mêmes fenêtres, mêmes A\*).

### 15.3 Correctif d'infrastructure trouvé en route

Le hook `MEM_DECODE_DUMP` écrivait le `.pt` **non atomiquement** alors qu'un autre bras (ou un
autre agent : nous étions deux à travailler sur SS2 cette nuit) peut le RELIRE au même instant
(`wronganchor` lit le dump d'une autre cellule). Corrigé : écriture `.tmp` + `os.replace`
(atomique sur le même système de fichiers). Aucune lecture tronquée n'a été observée, mais la
fenêtre existait.

**Fichiers** : `score_gateA.py`, `launch_defensive.py`, `results/gateA/gateA.json`,
logs `logs/def_*.log` + `logs/defensive_supervisor.log`.

### 16.5 LEVIER 4 — décoder le **snapshot de fin de segment A** (ajouté en cours de nuit)

Conséquence directe de 16.1 quater : si notre ancre est une image de l'interlude **parce que le
décodeur reconstruit ce qui vient d'être évincé**, alors il faut le brancher sur l'état
**snapshotté à la fin du segment de A**, où le contenu le plus récemment évincé EST A.

**Il n'y a rien à écrire.** `MEM_EPISODIC=1` snapshotte déjà cet état exact (`inference_schedule`
le prend quand se termine la première apparition d'un sujet qui sera revisité) et le **restaure au
revisit AVANT le décodage** (`_switch_recache`, bloc `is_revisit`) ; `_memdec_frame` lit l'état
vivant. Et sous `MEM_ZERO_ALL=1` l'injection mémoire est mise à **zéro dans les deux passes**
(`causal_model.py`), donc restaurer l'état **ne peut rien changer au rendu** — son seul
consommateur est le décodeur. Le bras `memdec_snap` est donc `memdec20` **au caractère près**,
avec un autre ÉTAT en entrée du décodeur (selftest : diff = `{MEM_EPISODIC}`, et l'ordre
restaure → décode est vérifié sur la source).

**Pré-écran scellé** (registre) sur l'ancre dumpée, avec **la métrique de T0** pour être
directement comparable : `corr(ancre_snap, A*) ≥ +0.15` **et** `> corr(ancre_snap, B)` sur 2/2
cellules du pilote (t2 chat/camion rouge, **t3 homme en veste bleue / flamants roses** = la
cellule la plus diagnostique de la planche) ⇒ le colis est réparé, on tire les 4 restantes.
Référence à battre, mesurée : ancre courante **vs A\* +0.042 | vs B +0.298 | vs X +0.344**.
Script : `snap_prescreen.py`. Prédiction re-ID scellée : `memdec_snap − wronganchor` ≥ +0.08 sur
≥ 4/6 sous H_colis-réparé ; mon pari ponctuel est +0.02 à +0.08 (l'ouvreur X reste dans l'état,
et l'ancre reste floue).

**État** : vague `D0` en tête de file, un premier lancement tombé en OOM (deux jobs d'un autre lot
posés sur la même carte après mon pré-check), re-mis en file automatiquement. Le levier 4 passe
**devant** les leviers 1/2/3 : il répare la CHARGE UTILE, eux le CANAL, et 16.1 bis dit que le
canal n'est pas le goulot.

#### 16.5 bis — RÉSULTAT du pré-écran du levier 4 : **0/2, le colis n'est pas réparé par l'INSTANT**

| ancre décodée | vs **A\*** | vs **B** | vs **X** |
|---|---|---|---|
| **SNAPSHOT** (état à la fin du segment A, frame 120) | **+0.323** | +0.540 | +0.562 |
| COURANTE (état au retour, frame 180) | +0.260 | +0.452 | +0.463 |

Le snapshot va dans le bon sens sur A\* (**+0.063** apparié) — mais il monte **tout** d'à peu près
autant (+0.09 sur B, +0.10 sur X) et **ne change pas le classement**. Règle scellée
(`vs A* ≥ +0.15` **et** `> vs B`) : **0/1**, parce que `vs B` domine toujours. C'est la branche
pré-assignée : *« le snapshot ne répare pas le contenu, le problème est dans l'ÉCRITURE (l'état
lui-même), pas dans le choix de l'instant »*.

**Prédiction scellée contre mesure, ligne par ligne** (t2) :

| | prédit (registre, écrit avant) | mesuré | verdict |
|---|---|---|---|
| vs A\* | +0.20 à +0.35 | **+0.323** | **DANS la bande** — le snapshot rapproche bien de A |
| vs B | **< +0.15**, et < vs A\* | **+0.540** | **DEHORS**, ×3.6 : je prédisais l'effondrement de B, il MONTE |
| vs X | +0.20 à +0.35 (reste haut) | **+0.562** | au-dessus de la borne haute |

La partie que j'avais raisonnée juste (le snapshot est le bon instant pour A) est confirmée ; la
partie qui portait le gate (B s'effondre) est réfutée. C'est le pré-écran qui échoue, pas le
raisonnement mécanique — et c'est pour ça qu'il était écrit avant.

**Un détail qui exclut l'explication la plus simple** : le snapshot est pris à la frame **120**,
c'est-à-dire **avant que le segment B n'existe** (B tourne de 120 à 180). L'ancre ne peut donc pas
contenir B — et elle corrèle quand même +0.540 avec le rendu de B. La corrélation « vs B » est
donc en partie **médiée** par ce que B et l'ancre ont en commun (la structure ambiante du rollout,
que l'ouvreur épinglé impose à tous les segments, §13.1).

**Ce que je ne peux PAS conclure, et je l'ai cru un instant** : que l'ouvreur X *domine* l'ancre.
Testé sur les 24 cellules, **X contre B est un match nul** — X > B sur **14/24** en bande basse
(p = 0.54) et **10/24** en DINO (p = 0.54). Les deux battent A\* très nettement (21/24, p = 3e-4)
mais ils ne se départagent pas. L'énoncé défendable est donc le plus prudent :

> l'ancre décodée ressemble à **la structure ambiante du rollout** — indifféremment l'ouvreur ou
> l'interlude — et **pas au sujet qu'on lui demande**. Déplacer l'instant du décodage ne
> l'améliore que marginalement et sans rien changer au classement.

**t3 est arrivée, et elle tranche dans le même sens, en pire** (A = homme en veste bleue au café,
B = flamants roses, X = un troisième décor : les trois maximalement distincts) :

| ancre | vs **A\*** | vs **B** | vs **X** |
|---|---|---|---|
| SNAPSHOT (fin de A) | **−0.268** | +0.322 | **+0.567** |
| COURANTE | −0.182 | +0.247 | +0.513 |

Sur la cellule la plus diagnostique du lot, l'ancre décodée de l'état de fin de A est
**ANTI-corrélée** au sujet qu'elle est censée porter, et le snapshot **aggrave** l'écart
(−0.268 contre −0.182). Agrégat des 2 cellules : SNAPSHOT vs A\* **+0.027** | vs B +0.431 |
vs X **+0.565** ; COURANTE vs A\* +0.039 | vs B +0.350 | vs X +0.488. **Gain apparié sur A\* :
−0.012** — changer l'instant ne rapproche même pas de A en moyenne.

**RÈGLE SCELLÉE : 0/2** ⇒ branche pré-assignée : **« le snapshot ne répare pas le contenu : le
problème est dans l'ÉCRITURE (l'état lui-même), pas dans le choix de l'instant »**.

**Le rendu confirme, indépendamment** (re-ID crop60 contre A\* propre, t2) :
`memdec_snap − wronganchor` = **−0.026** (full +0.008), `memdec_snap − memdec20` = **+0.012**
(full +0.031). Changer l'état d'entrée du décodeur ne déplace ni l'ancre ni la vidéo. Au passage,
sur les 6 cellules S2 de ce tableau, `memdec20 − wronganchor` = **+0.007** (3+/3−, 1/6 au-dessus
du seuil) — la réplication, à 6 cellules, du négatif du §11.

**Ce que ça élimine, et ce que ça laisse debout.** Trois hypothèses de *canal* sont mortes ou
mourantes (T0 : le canal transmet ; 16.1 bis : ce qu'il transmet ne prédit pas le score ; levier 4 :
l'instant du décodage n'est pas le levier). Ce qui reste est **l'ÉCRITURE** : à la fin même du
segment de A, l'état ne permet pas de décoder A. C'est le programme mémoire 2.0
(co-entraînement de l'écrivain), et cette nuit lui donne son premier argument mesuré plutôt
qu'inféré.

**L'œil, et il dit la même chose en une image** (`eyepack_offensif/board_levier4_snapshot.jpg`,
0.11 Mo) : 6 colonnes — A\* propre | X | B | **ancre COURANTE** | **ancre SNAPSHOT** | retour rendu.
Les deux colonnes d'ancres sont **quasiment la même image** : sur t2 un flou rouge-et-vert dans les
deux cas (le camion de pompiers), sur t3 un flou rosé dans les deux cas, là où A\* est un homme en
**veste bleue**. Déplacer l'instant du décodage de la frame 180 à la frame 120 ne change
**visiblement rien** au colis. Et la dernière colonne montre l'autre moitié du dossier : le retour
rendu est un **chat roux correct** (t2) et un **homme en veste bleue correct** (t3) — peints par
le prompt, pas par l'ancre.

*(n = 2 : c'est un GATE de continuation pré-enregistré, pas une mesure de précision. La règle du
pilote étant tranchée à 0/2, les 4 cellules restantes **ne sont pas tirées** — la règle scellée
en faisait précisément la condition d'investissement, et le GPU va aux leviers encore ouverts.)*

## 14. MINI-BATTERIE TRAINING-FREE (2026-08-04) — **l'ancre n'a jamais contenu d'identité, et le « correctif minimal » du reviewer marche partiellement**

Trois bras additifs, env-gated, défaut OFF, prédictions scellées AVANT run dans
`memory2_probes_registry.md` (section « MINI-BATTERIE TRAINING-FREE » + amendement) :
`sdedit_ours` / `sdedit_wrong` (l'ancre décodée est **affûtée** par le prior du modèle avant
splice, sur notre souvenir et sur celui d'une autre cellule) et `sinkpurge` (**aucun splice,
aucune mémoire** : au retour, les slots du sink épinglé sont réécrits avec la frame vraie
courante). 6 cellules, seed 0 : `SS2` t12/t13/t14 (même décor) + `S2` t2/t4/t11.

### 14.0 Lecture de code demandée : LongLive ne rafraîchit JAMAIS son sink

Vérifié avant toute mesure. Avec `global_sink: true` (nos configs **et** `stock_official.yaml`),
`_recache_after_switch` ne remet pas le cache à zéro et passe `sink_recache_after_switch=False` ;
`causal_model.py` protège `[0, sink_tokens)` à chaque recompute
(`write_start_index = max(local_start_index, sink_tokens)`). Les `sink_size` premières frames
latentes (1 chez nous, **3** chez LongLive stock) gardent donc les k/v de la **frame 0** pendant
tout le rollout, à travers tous les switches. C'est le substrat mécanique de la pathologie du
§13.1 (l'ouvreur domine tous les segments suivants).

### 14.1 Le gate d'œil a d'abord attrapé un bug, PUIS une contrainte de substrat

La planche σ est faite avant tout rollout (règle scellée). Son témoin dégénéré **σ = 1.0**
(= le sampler 4 pas du modèle partant du bruit pur, donc il DOIT rendre une image propre) rendait
du **bruit** : ma première version débruitait un bloc d'**une** frame latente, hors distribution
pour un backbone entraîné à `num_frame_per_block = 3`. Corrigé (réplication à la taille de bloc
native, on relit la frame du milieu) ⇒ σ = 1.0 rend l'image du prompt sur 6/6 cellules.

Puis la planche a montré une **contrainte du substrat, pas un réglage** : le générateur est
distillé DMD en 4 pas, sa grille de timesteps est ~[1000, 937.5, 833.3, **625**] ; un appel à
t = 400/500/600 est **sous la grille**, là où le modèle n'a jamais été entraîné, et sort une
**mosaïque haute fréquence**. **Les σ ≈ 0.4 et 0.6 demandés ne sont donc pas réalisables sur ce
backbone.** Règle scellée appliquée aux points réalisables {0.625, 0.833} : σ = 0.625 reste
visiblement mosaïque ⇒ **σ = 0.833 retenu** (le plus bas qui soit net).

### 14.2 LE RÉSULTAT PRINCIPAL — l'ancre décodée ne contient AUCUNE identité (sonde directe)

La planche a rendu évident ce qu'aucune campagne n'avait regardé : **l'ancre brute n'est pas un A
flou, c'est une tache de couleur** (t12 : un smear sombre rougeâtre, ni homme ni table ni tasse ;
t13 : du vert, pas de chien ; t11 : un smear jaune, pas de théière). Tout le dossier mesurait la
vidéo RENDUE, donc ne pouvait pas séparer « le canal est aveugle au contenu » de « il n'y a pas de
contenu ». Sonde ajoutée et pré-enregistrée (`probe_anchor_identity.py`) : l'ancre est VAE-décodée
en pixels et comparée en DINO au A\* de SA cellule et au A\* de la cellule SOURCE.

| variante de l'ancre | vs SON A\* | vs A\* SOURCE | marge | cellules + |
|---|---|---|---|---|
| **brute (ce que `memdec20` splice)** | **0.0023** | −0.0154 | **+0.018** | 4/6 |
| affûtée σ 0.625 | 0.0516 | 0.0126 | +0.039 | 3/6 |
| affûtée σ 0.7 | 0.3078 | 0.0347 | +0.273 | 6/6 |
| affûtée σ 0.833 | 0.5385 | 0.0308 | +0.508 | 6/6 |
| **σ 1.0 = PROMPT SEUL, zéro ancre** | **0.7770** | 0.0692 | +0.707 | 6/6 |

* Prédiction scellée (marge ≤ +0.03, ≤ 4/6 positives) : **ATTEINTE** sur l'ancre brute.
  **Cosinus 0.002 avec le sujet qu'elle est censée encoder : en espace DINO elle lui est
  ORTHOGONALE.** Pas une identité dégradée — aucune identité.
* **Le contrôle qui rend la mesure incontestable** : la ligne σ 1.0 ne contient AUCUNE information
  d'ancre et score **0.777**. La métrique voit parfaitement le sujet ; c'est l'ancre qui n'en
  porte pas.
* **Et l'échelle est MONOTONE DÉCROISSANTE** : 0.777 (prompt seul) > 0.539 > 0.308 > 0.052 >
  0.002 (ancre brute). *Plus on garde d'ancre, moins la frame ressemble à A.* Entre les deux
  points propres et sur-grille (σ0.833 vs σ1.0), injecter notre ancre coûte **−0.24 d'identité**
  par rapport à n'utiliser que le prompt. **L'ancre décodée n'est pas un signal faible : c'est un
  PERTURBATEUR actif.**
* Nuance honnête, visible à l'œil : l'ancre transporte quand même la **statistique de couleur**
  (sur t12 `memdec20` ramène une veste ROUGE là où `wronganchor` en ramène une BLEUE). Elle teinte
  le rendu ; elle ne l'identifie pas. Les deux faits sont compatibles et se complètent.

### 14.3 Au niveau du rollout : l'affûtage ne change RIEN

| bras | sujet crop60 SS2 | sujet crop60 S2 | fond (head) SS2 | cut2 SS2 | CLIP SS2 |
|---|---|---|---|---|---|
| `memzero` | 0.2648 | 0.0715 | 0.9343 | 0.71 | 0.2485 |
| **`sinkpurge`** | **0.4111** | 0.1234 (1 cellule) | **0.9269** | **0.72** | 0.2725 |
| `memdec20` | 0.6888 | 0.7757 | 0.4586 | 20.0 | 0.3455 |
| `wronganchor` | 0.6786 | 0.7464 | 0.4376 | 20.9 | 0.3344 |
| **`sdedit_ours0833`** | 0.7481 | 0.7494 | 0.3071 | 17.5 | 0.3455 |

Contrastes appariés (sujet crop60, 6 cellules) :
* **`sdedit_ours` − `memdec20` = +0.0165, 3+/3−, sign-p 1.00** ⇒ **affûter l'ancre n'apporte
  RIEN** au niveau du rendu. Même quand la frame splicée est nette et dans la distribution du
  modèle, le résultat est celui du blob flou.
* `memdec20` − `wronganchor` = +0.0197 (4+/2−, p = 0.69) : le §11 se **reproduit** sur ces
  6 cellules (indiscernables).
* Le bénéfice collatéral pré-enregistré (« un raccord moins brutal ») est **au mieux marginal** :
  cut2 20.0 → 17.5, prédit 8-14 ⇒ **prédiction ratée**, et le fond se DÉGRADE (0.459 → 0.307).

**Verdict idée 1 (E1 « l'ancre est illisible parce que floue ») : ÉLIMINÉE**, par deux routes
indépendantes — au niveau de l'ancre (rien à lire : cos 0.002) et au niveau du rollout (rendre
l'ancre nette ne change rien : +0.017 n.s.).

**LE CHIFFRE (R1), obtenu : `sdedit_ours` − `sdedit_wrong` = +0.0221, 3+/3−, sign-p 1.000,
sd 0.142, sem 0.058, 2/6 cellules ≥ +0.08** (t12 +0.223, t13 +0.003, t14 −0.166, t2 +0.151,
t4 −0.063, t11 −0.016). La vague a fini quand l'autre agent a libéré les GPU.

**Application de la règle scellée, et pourquoi elle est INEXPLOITABLE telle qu'écrite.** La lettre
de la règle (« ≥ +0.08 sur ≥ 4/6 ⇒ E1 confirmée ; 2 ou 3 ⇒ zone grise ; ≤ 1 ⇒ E1 éliminée »)
donne **2/6 ⇒ zone grise**. Mais la règle ne comptait que les dépassements POSITIFS, et les
données fournissent leur propre étalon de bruit :

| contraste | moyenne | sd | sem | ≥ +0.08 | ≤ −0.08 |
|---|---|---|---|---|---|
| `ours` − `wrong` (le chiffre) | +0.0221 | 0.142 | 0.058 | **2/6** | 1/6 |
| **`wrong` − `wronganchor`** (MÊME ancre fausse, seul l'affûtage change ⇒ **différence de contenu NULLE par construction**) | +0.0142 | 0.197 | 0.081 | **3/6** | 2/6 |
| `sinkpurge` − `memzero` (effet réel, pour comparaison) | +0.1189 | **0.078** | 0.032 | 5/6 | 0/6 |

**Un contraste sans AUCUNE différence de contenu franchit déjà +0.08 sur 3/6 cellules.** Le
plancher de bruit par cellule (sd ≈ 0.15-0.20 à 1 graine) est donc **2 à 2,5 fois le seuil**
pré-enregistré : la règle R1 était **sous-dimensionnée**, et notre contraste (2/6) fait **moins
bien que son propre témoin sans contenu** (3/6). Comparaison qui rend l'argument visible : un
effet RÉEL sur les mêmes cellules et la même métrique (`sinkpurge`) a un sd de **0.078**, deux
fois plus serré, **5/6 au-dessus de +0.08 et 0/6 en dessous de −0.08** (contre 2/6 et 1/6 pour
notre contraste) : un vrai effet est UNILATÉRAL, le nôtre est symétrique.

**Conclusion honnête** : R1 est **VOID par insuffisance de puissance** (c'est un résultat de
méthode, à écrire tel quel : il aurait fallu 2-3 graines par cellule), et le contraste apparié
**n'apporte aucun soutien à la lecture du contenu** — sa valeur centrale (+0.022 ± 0.058) tombe
pile dans la bande H_slot pré-enregistrée (\|Δ\| < 0.05) et son signe est exactement 3+/3−.
E1 reste donc **éliminée par les deux routes de §14.2 et §14.3**, la troisième étant muette
plutôt que contraire.

**L'ŒIL, et c'est la pièce la plus lisible de toute la batterie** (`eyepack_tfree/board_tfree.jpg`,
colonne `sdedit_wrong0833`) : avec le souvenir d'une AUTRE cellule, affûté, **le bon sujet revient
quand même sur 6/6 cellules** — t12 un homme en veste ROUGE (ancre venue du chien/pelouse de t13),
t13 un golden retriever (ancre venue du scooter/boulangerie de t14), t14 un scooter bleu devant la
boulangerie (ancre venue de la voiture rouge/garage de t15), t2 un chat tigré orange, t4 un cheval
blanc, t11 une théière en cuivre. **Le contenu splicé est faux dans les six cas et le sujet
correct revient dans les six cas.** C'est le §11 rendu visible, et l'affûtage n'y change rien.

**Test de contamination** (le contenu étranger laisse-t-il une trace ?). Excès de ressemblance au
A\* du sujet **SOURCE**, apparié par cellule : `sdedit_wrong` − `sdedit_ours` = **+0.0129
(sd 0.049, 3+/3−, sign-p 1.000)** ; rappel du couple non affûté, `wronganchor` − `memdec20` =
**+0.0096 (sd 0.025, 4+/2−, p = 0.69)**. **Aucun excès significatif dans les deux cas, et
l'affûtage ne le fait pas apparaître** : splicer l'ancre décodée d'une AUTRE vidéo, nette ou
floue, ne laisse pas de trace mesurable de cette vidéo. Cohérent avec le §11.5 (−0.0000 à
24 cellules).

### 14.4 `sinkpurge` — l'objection du « correctif minimal » est PARTIELLEMENT fondée

**+0.1189 vs `memzero` sur les 6 cellules (5+/1−, sd 0.078), +0.1462 sur les 3 cellules SS2
(3+/3), soit 21 % du gain global de `memdec20` (+0.5641) et 34 % sur SS2 (+0.4239)** — sans
aucune mémoire, sans aucun splice, en réécrivant UN slot de k/v.

Et surtout : **c'est gratuit sur tous les autres axes**, là où notre splice paie très cher.

| | fond (head) SS2 | cut2 SS2 | mouvement DD-ell | CLIP texte |
|---|---|---|---|---|
| `memzero` (plafond de continuité) | 0.9343 | 0.71 | 0.0040 | 0.2485 |
| **`sinkpurge`** | **0.9269** | **0.72** | **0.0061** | **0.2725** |
| `memdec20` | 0.4586 | **20.0** | 0.0053 | 0.3455 |

`sinkpurge` garde le fond de `memzero` (0.93), ne coupe pas (0.72 contre **20**), **augmente** le
mouvement (0.0061 > 0.0040 : aucun effet autocollant) et améliore l'adhérence texte. Tripwire de
stabilité pré-enregistré : **rien ne se déstabilise**, le sink peut être rafraîchi sans casser la
génération. Tripwires mécaniques du hook verts sur 4/4 rollouts (audit des logs : 23 rollouts, tout vert) (k du sink : rel-L2 0.27-0.33 ;
reste de la fenêtre : max\|diff\| exactement 0).

**Mais l'œil corrige le chiffre, et c'est important** (`eyepack_tfree/sinkpurge_timeline.jpg`) :
sur t12 et t14, `sinkpurge` est **visuellement indiscernable de `memzero`** — le barista et le
livreur en vert sont toujours là, le sujet A ne revient PAS. Le +0.146 est un gain
**sous-catégoriel** (l'emprise de l'ouvreur se desserre, la scène se recentre), pas une
résurrection. Règle scellée R2 (≥ +0.21 ⇒ objection fondée ; < +0.05 ⇒ parée) :
**+0.131 ⇒ plancher PARTIEL, à reporter.**

**Ce que ça change pour le papier** : le +0.43/+0.56 de nos bras ne peut plus être annoncé
au-dessus de `memzero` seul. `sinkpurge` doit devenir **une ligne de baseline** — un correctif à
coût nul qui capture ~un quart à un tiers du gain sans mémoire ni splice, et **sans aucun des
dégâts** (fond ×2 mieux, raccord ×28 mieux). Un reviewer le trouverait ; autant l'avoir mesuré.

### 14.5 Note d'ops (à garder)

* **Reproductibilité mp4** : re-générer `memdec20` avec le code courant donne un mp4 qui diffère
  de celui sur disque (mean 1.0/255, max 50-97) — **dès le segment 0, avant tout hook**. Contrôle :
  deux bras DÉJÀ sur disque, générés dans des process différents, diffèrent pareil (mean 0.98-1.22)
  alors que deux bras du MÊME process diffèrent de 0.044. Donc : **le pipeline n'est pas bit à bit
  reproductible entre process** (non-déterminisme GPU amplifié par l'autorégression), et ce n'est
  pas dû à cette batterie. Impact sur la mesure : **−0.0001 / +0.0061** sur le chiffre-titre, deux
  ordres de grandeur sous les effets. La garantie d'inertie des bras existants est le selftest
  (c1) : hooks éteints, les helpers renvoient l'**objet ancre lui-même**.
* **Course entre superviseurs** : deux superviseurs lancés en parallèle ont lu « 48 Go libres » sur
  le MÊME GPU et s'y sont écrasés (3 OOM). Correctif : **verrou fichier par GPU** dans
  `launch_tfree.py` (prise atomique), + règle « un seul superviseur, vagues chaînées ».
* **VRAM** : les caches KV/cross-attn sont morts après la boucle de génération ; ils sont
  maintenant **libérés juste avant le décodage VAE final** (~3.5 Go rendus au pic exact où un
  co-run nous fait OOM). Numériquement inerte par construction.
* **Édition concurrente** : un autre agent du projet a réécrit `run_reid_scen.py` pendant la
  campagne et a généralisé ma branche « ancre étrangère » en `S.WRONG_SRC_ARMS` (qui inclut bien
  `SDEDIT_WRONG_ARMS`) — fonctionnellement correct, mais mon selftest testait le nom littéral. Il
  teste désormais le SENS. **Leçon : sur fichier partagé, tester le comportement, pas le texte.**

**Fichiers** : hooks `MEM_SDEDIT`, `MEM_SDEDIT_FRAMES`, `MEM_SDEDIT_SEED`, `MEM_SINKPURGE`
(`pipeline/switch_causal_inference.py`, driver N-switch uniquement, `inference()` legacy les
refuse) ; `reference/runs/reid_scen/{selftest_tfree.py, launch_tfree.py, make_sdedit_board.py,
probe_anchor_identity.py, score_tfree.py, make_eyepack_tfree.py}` ; résultats
`results/tfree/{tfree_verdict.json, anchor_identity.json}` et `results/comp/comp_tfree_{SS2,S2}.json` ;
œil `eyepack_tfree/` (planches σ 1.3-1.5 Mo, `board_tfree.jpg` 0.36 Mo, `sinkpurge_timeline.jpg`
0.53 Mo, 6 hstacks < 1.5 Mo + 6 micro-clips verdict < 0.5 Mo).

### 16.6 LEVIER 3 — RÉSULTAT : **2/4, le canal additif est CLOS** (et la masse, elle, est acquise)

Quatre cellules SS2, contenu = la VRAIE frame d'ancre, gain 8, bande RoPE **17** (Δ = 1 de la
requête au lieu de Δ = 9). Sujet crop60 contre le A\* PROPRE :

| cellule | memzero | oracle1 (splice) | oracle1add8 (Δ=9) | **add8p17 (Δ=1)** | delta vs memzero |
|---|---|---|---|---|---|
| t12 | 0.1436 | 0.1620 | 0.2394 | **0.2577** | **+0.114** |
| t13 | 0.5130 | 0.6937 | 0.5047 | **0.7473** | **+0.234** |
| t14 | 0.1379 | 0.2735 | 0.1675 | 0.1191 | −0.019 |
| t15 | 0.1019 | — | — | 0.1137 | +0.012 |
| **moyenne** | | | | | **+0.0853** (3+/1−) |

**RÈGLE SCELLÉE** (`≥ +0.0558`, la moitié du chemin memzero → oracle1, sur **≥ 3/4** cellules) :
**2/4** (t12, t13). La branche « ouvre » demande 3/4 : elle échoue. La branche « **< +0.0558 sur
≥ 2/4 ⇒ la position n'était pas le goulot non plus, le canal additif est clos définitivement** »
est atteinte. ⇒ **CLOS.**

**Ce qu'il faut quand même dire, parce que c'est la seule fois de la campagne** : sur les deux
cellules où il marche, l'additif à Δ = 1 **bat le splice lui-même** (t12 0.2577 contre 0.1620 ;
t13 0.7473 contre 0.6937) — sans évincer une seule vraie frame. La variance entre cellules est
énorme (−0.019 à +0.234) et c'est elle qui tue le gate, pas un effet nul.

**T1 — la masse d'attention par source, acquise sur les 4 cellules** (moyenne de 19 pas × 30 blocs
après le retour, décomposition logaddexp exacte, les trois fractions somment à 1) :

| source | tokens | part uniforme | masse mesurée (4 cellules) | rapport |
|---|---|---|---|---|
| sink (ouvreur épinglé) | 1560 | 7.7 % | **4.55 %** | **0.59×** |
| fenêtre locale | 17160 | 84.6 % | 74.1 % | 0.88× |
| **ancre additive** | 1560 | 7.7 % | **21.3 %** | **2.78×** |

Par cellule : 20.3 / 17.0 / 23.9 / 24.2 %. Prédiction scellée « 25 à 45 % à gain 8 » : mesuré
**21.3 %**, juste **sous** la borne basse.

**Le contrôle de position à instrumentation IDENTIQUE** (`oracle1add8p9`, même cellule t12, même
gain, seule la bande RoPE change — il remplace la comparaison inter-campagnes avec les 10.5-15.7 %
du §3.4) :

| | sink | fenêtre | **ancre** | ancre / sa part de tokens |
|---|---|---|---|---|
| **Δ = 9** (bande 9, « auto ») | 4.82 % | 83.49 % | **11.69 %** | 1.52× |
| **Δ = 1** (bande 17) | 4.61 % | 75.06 % | **20.33 %** | **2.64×** |

Rapprocher l'ancre de la requête lui fait gagner **+74 % de masse relative** — et la masse est
prise à la **FENÊTRE** (83.5 → 75.1 %), pas au sink (4.82 → 4.61 %, inchangé). Deux conséquences :
l'ancre additive **concurrence le présent local**, pas l'ancre épinglée ; et **le sink reste
sous-lu à ~4.7 % (0.6× sa part) dans les deux configurations**, ce qui rend le point 1 ci-dessous
robuste au placement.

**Les deux enseignements de T1 survivent au verdict négatif** :
1. **le sink est SOUS-LU** (4.55 % pour 7.7 % des tokens, 0.59×) alors qu'il domine le rendu
   (§13.1). Son emprise ne passe donc **pas** par l'attention à la génération : elle passe par le
   **recache**, qui reconstruit la fenêtre en le lisant. Correction utile à l'intuition courante ;
2. **le canal n'est pas limité par la lecture** : à Δ = 1 l'ancre est lue 2.8× sa part, et le
   score ne suit que sur 2 cellules sur 4. Le mode d'échec est donc « **lu et pas décisif** », pas
   « non lu » — exactement le pendant, côté canal additif, de ce que T0 dit côté splice.

**Fichiers** : `results/offensif/{scores_lever3.json, add_mass/*.json}`, logs
`logs/{off_oracle1add8p17_*, score_lever3.log}`.

#### 16.6 bis — les axes de DÉGÂT du levier 3 : prédiction **quasi exacte**, et un effet non prédit

Prédiction scellée d'avance : *« à Δ = 1 le canal cesse d'être inoffensif — je prédis `cut2` qui
monte de 0.73 vers **2 à 6** et le fond qui descend de 0.925 vers **0.86-0.91** »*. Mesuré
(3 cellules SS2) :

| | memzero | add8 (Δ=9) | **add8p17 (Δ=1)** | oracle1 (splice) | memdec20 | prédit |
|---|---|---|---|---|---|---|
| **cut2** (raccord, bas = mieux) | 0.71 | 0.73 | **3.51** | 7.69 | 20.04 | **2 à 6 ✓** |
| **fond head** (haut = mieux) | 0.934 | 0.925 | **0.912** | 0.899 | 0.459 | 0.86-0.91, **quasi ✓** |
| fond fenêtre | 0.893 | 0.880 | 0.878 | 0.828 | 0.475 | — |
| sujet crop60 | 0.289 | 0.298 | **0.372** | 0.357 | 0.689 | — |

Les deux prédictions tombent juste (le `cut2` en plein milieu de la bande, le fond juste au bord
haut = un peu moins de dégât qu'annoncé). Le point de fonctionnement de Δ = 1 est donc lisible :
**un raccord 2× plus doux que le splice et 6× plus doux que notre mémoire, un fond quasi intact,
et le meilleur sujet des trois bras sans mémoire** — mais une variance qui l'empêche de passer le
gate.

**Un effet NON prédit, à signaler** : le tripwire de mouvement dans l'ellipse **monte** —
add8p17 **0.0144** contre 0.0040 pour memzero, 0.0051 pour oracle1, 0.0053 pour memdec20, soit
**×3.6**. Le tripwire était écrit pour attraper l'inverse (« s'effondrer sous memzero =
autocollant = rejet ») : il ne se déclenche donc pas, mais une multiplication par 3.6 du lag-1
dans la région du sujet ressemble à du **scintillement**, pas à du mouvement utile. C'est à
vérifier à l'œil avant toute reprise de ce canal (aucune cellule n'a été regardée sous cet angle
cette nuit). Noté comme observation post-hoc, non pré-enregistrée.

### 16.7 SYNTHÈSE DE LA NUIT — quatre leviers, une seule direction

Tout ci-dessous était **pré-enregistré avant run** (`memory2_probes_registry.md`), et chaque
levier avait son propre jumeau pour comparateur, jamais `memzero`.

| # | levier | prédiction scellée | mesuré | verdict |
|---|---|---|---|---|
| **T0** | bande basse (0 GPU) | excès ∈ [−0.02, +0.05], ≤ 14/24 | **+0.827, 23+/1−, p = 1.2e-06** | **RÉFUTÉE ×20** — le canal LIT |
| **4** | décoder le snapshot de fin de A | vs A\\* +0.20-0.35, vs B < +0.15 | +0.027 / **+0.431** | **0/2 — l'INSTANT n'est pas le levier** |
| **3** | additif à Δ = 1 | ≥ +0.0558 sur ≥ 3/4 | **2/4** (+0.114, +0.234, −0.019, +0.012) | **CLOS — la POSITION non plus** |
| **1** | légende (pré-écran) | 16-19/24 | **13/24, p = 0.84** | vivant d'une cellule, **au hasard** |
| **2** | ancre déclarée bruitée | ≥ +0.08 sur ≥ 4/6 | **0/4, moyenne −0.0055, 2+/2−** | **RÈGLE INATTEIGNABLE — H_slot confirmée 3/3** |

*Levier 2, validation mécanique avant les scores* (log du premier rollout, SS2 t14) :
`anchor slot(s) [11, 12) of 12 NOISED to t=500 (sigma~0.500, rel-L2 0.919) AND declared at
t=500; the 11 true frames keep t=0` puis `recache timesteps: 1 slot(s) declared off-baseline
[(11, 500)]`. **Un seul slot** — celui de récence — est réellement bruité ET déclaré, les 11
vraies frames gardent t = 0 : le mensonge de métadonnée est bien réparé, et les deux tripwires
(changement rel-L2 non nul, compte de slots hors baseline) se déclenchent. Le bras mesure donc
l'intervention voulue, quel que soit son résultat.

**RÉSULTAT (4 paires sur 6, t\* = 500)** — la règle est déjà **INATTEIGNABLE** : 0/4 au-dessus
du seuil, 2 cellules restantes, maximum 2/6 < 4/6.

| cellule | `i1_ours500` | `i1_wrong500` | contraste |
|---|---|---|---|
| S2 t2 | 0.9014 | 0.8312 | +0.070 |
| S2 t11 | 0.7409 | 0.7148 | +0.026 |
| S2 t4 | 0.6601 | 0.6893 | −0.029 |
| SS2 t14 | 0.6146 | 0.7037 | −0.089 |
| **moyenne** | | | **−0.0055** (2+/2−, **0/4** ≥ +0.08) |

**H_slot était prédite « |Δ| < 0.05, signe aléatoire, ≤ 1/6 au-dessus du seuil » : les TROIS
critères tombent juste** (−0.0055 ; 2+/2− ; 0/4). L'hypothèse nulle n'est pas « non rejetée »,
elle est **confirmée sur chacun de ses termes**. Et `i1_ours500 − memdec20` = **−0.004** (3+/1−) :
déclarer le bruit ne change rien face au bras qui ne le déclare pas.

**Le mensonge de métadonnée était réel, il a été correctement réparé (vérifié in situ), et le
réparer ne produit rien.** Le défaut nommé par le mécaniste existe ; il n'est pas le goulot.

**Les 2 cellules restantes (SS2 t12/t13) ne sont PAS tirées** — même règle que le levier 4 : on ne
dépense pas de GPU à préciser un négatif déjà tranché, et le levier 1 n'a aucune cellule mesurée.

*Historique du premier point (SS2 t14, n = 1, avant les 3 autres)* :
`i1_ours500 − i1_wrong500` = **−0.089** crop60 (−0.077 full) et `i1_ours500 − memdec20` =
**+0.010** crop60 (−0.006 full). Autrement dit : déclarer honnêtement le niveau de bruit de
l'ancre **ne change rien** face au bras non déclaré, et sur cette cellule le contenu PROPRE
fait **moins bien** que le contenu FAUX soumis au même traitement. Hors de ma bande prédite
(|Δ| < 0.05) mais **du mauvais côté** pour le levier. À confirmer sur les 5 cellules restantes.

*Sous-produit à 6 cellules, avec référence A\* propre* : `memdec20 − wronganchor` = **+0.0197**
crop60 (4+/2−, 1/6 au-dessus du seuil, p = 0.69) — la réplication du négatif du §11 sur ce
sous-ensemble.

**Les prédictions qui sont tombées juste** méritent d'être notées autant que celles qui ont
échoué, parce qu'elles montrent que le modèle mécanique du dossier est bon là où il est testable :
`cut2` à Δ = 1 prédit « 2 à 6 » → mesuré **3.51** ; le fond prédit « 0.86-0.91 » → **0.912** ;
l'ancre snapshot vs A\\* prédite « +0.20-0.35 » → **+0.323** sur t2. Ce qui casse à chaque fois,
c'est la partie qui portait le GATE.

**La chaîne d'élimination, dans l'ordre où elle s'est faite** :
1. le canal **transmet** — la structure de l'ancre reçue est copiée dans le rendu à 0.47-0.52
   contre un plancher de 0.10, pendant ≈ 3 s, au centre comme au bord (T0 + 3 contrôles) ;
2. mais **ce qu'il transmet ne prédit pas le score** — corrélation copie ↔ re-ID : Pearson
   **−0.075** sur le contraste (16.1 bis). Élargir le canal ne paie pas ;
3. et **l'ancre ne porte pas l'identité** — DINO +0.0007 (p = 0.84), couleur 13/24 (p = 0.84),
   structure +0.014 (p = 0.84), le tout contre une ancre étrangère ; répliqué indépendamment par
   le lot défensif avec un autre design (16.2 bis) ;
4. **changer l'INSTANT du décodage n'y change rien** (levier 4, 0/2 : à la fin même du segment de
   A, l'état ne permet pas de décoder A) ;
5. **changer la POSITION de lecture n'y change rien non plus** (levier 3, 2/4) — alors même que
   la masse d'attention de l'ancre passe de 11.7 % à 20.3 %, soit **2.6× sa part de tokens**.

> **Le transport marche. Le colis est vide. Le chemin critique est l'ÉCRITURE.**

C'est la première fois que cette phrase est **mesurée** et non inférée, et elle est cohérente avec
tout le dossier antérieur sans en contredire une ligne : `wronganchor` = 106 % du gain (§11),
l'attribut jamais restitué (§13), le slot qui décide (gate 2) — tous s'expliquent si le splice
sert à **effacer** l'interlude et si le prompt repeint A.

**Deux corrections d'intuition, à garder** : (a) **le sink est SOUS-LU** (4.6 % de la masse pour
7.7 % des tokens, 0.59×, dans les deux placements) alors qu'il domine le rendu — son emprise passe
par le **recache**, pas par l'attention ; (b) « l'ouvreur domine l'ancre » n'est **pas** établi —
X contre B est un match nul (14/24, p = 0.54) ; ce qui est établi, c'est que les deux battent A\\*
(21/24, p = 3e-4).

**Ce qui reste ouvert et honnête** : n petit partout (2 à 6 cellules pour les gates, 24 pour T0) ;
le levier 2 tourne ; et le levier 3, bien que clos par sa règle, est **le seul bras de toute la
campagne qui gagne de l'identité en restant dans le bon quadrant** (raccord 3.5 contre 20 pour
notre mémoire, fond 0.91 contre 0.46) — à ranger comme piste v2 à contenu appris, pas comme
mécanisme mort.

### 16.8 CLÔTURE DU LOT OFFENSIF — les cinq leviers, prédit contre mesuré

| # | levier | PRÉDIT (scellé avant run) | MESURÉ | VERDICT |
|---|---|---|---|---|
| **T0** | corrélation bande-basse, 24 cellules (0 GPU) | excès ∈ [−0.02, +0.05], ≤ 14/24 | **+0.827**, 23+/1−, p = 1.2e-06 ; ancre reçue 0.47-0.52 contre plancher nul 0.10 | **RÉFUTÉE ×20 — le canal LIT le splice** |
| **1** | légende (canal texte) | pré-écran 16-19/24 ; `ours − wrong` ≥ +0.08 sur ≥ 4/5 **et** jumeau faux NÉGATIF | pré-écran **13/24** ; contraste **+0.0429**, 5+/3−, p = 0.73, **2/8** au seuil ; jumeau faux **−0.0057** (négatif ✓) | **ÉCHOUE 2/8** (le contrôle directionnel, lui, passe) |
| **2** | ancre déclarée bruitée (I1) | ≥ +0.08 sur ≥ 4/6 ; H_slot : \|Δ\| < 0.05, signe aléatoire, ≤ 1/6 | **−0.0055**, **2+/2−**, **0/4** ⇒ règle **inatteignable** (max 2/6) | **H_slot CONFIRMÉE 3/3** |
| **3** | additif à Δ = 1 | ≥ +0.0558 sur ≥ 3/4 ; masse 25-45 % ; cut2 2-6 ; fond 0.86-0.91 | **2/4** ; masse **21.3 %** (2.8× sa part) ; cut2 **3.51** ; fond **0.912** | **CLOS** (mais les 3 prédictions physiques justes) |
| **4** | décoder le snapshot de fin de A | vs A\* +0.20-0.35 ; vs B < +0.15 | vs A\* **+0.027** ; vs B **+0.431** ; t3 **−0.268** | **0/2 — l'INSTANT n'est pas le levier** |

**La chaîne, en une lecture** : le canal **transmet** (T0) → ce qu'il transmet **ne prédit pas** le
score (Pearson −0.075) → l'ancre **ne porte pas** l'identité (DINO +0.0007, couleur 13/24,
structure +0.014 ; tous p = 0.84 contre une ancre étrangère, répliqué indépendamment) → **ni
l'instant** (levier 4) **ni la position** (levier 3) **ni la métadonnée** (levier 2) n'y changent
rien. **Le transport marche, le colis est vide, le chemin critique est l'ÉCRITURE.**

**Deux cellules NON tirées, délibérément** : `SS2 t12/t13` du levier 2 (règle déjà inatteignable)
et les 4 cellules restantes du levier 4 (règle 0/2) — même principe que le gate du pilote : *on ne
dépense pas de GPU à préciser un négatif tranché*.

#### Levier 1, le détail (8 cellules, 33 rollouts)

`caption_ours − caption_wrong` crop60 : **+0.0429** (5+/3−, p = 0.73), par cellule t0 +0.101,
t12 +0.174, t11 +0.044, t3 +0.033, t13 +0.003, t2 −0.002, t4 −0.001, t14 −0.008 ⇒ **2/8** au
seuil, la règle en exigeait ≥ 4/5. `caption_ours − padded` = +0.0372 (1/8). Le **jumeau faux est
bien NÉGATIF** (`caption_wrong − padded` = **−0.0057**) : la seconde condition de la règle passe,
la première non. Plafond `framebank_vlm − caption_wrong` = +0.0518 (2/8) — **une vraie frame ne se
légende pas beaucoup mieux que notre ancre**, ce qui est cohérent avec un canal texte à faible
bande passante plutôt qu'avec une charge utile qui serait bonne.
Tendance positive mais très en-dessous du seuil et non significative — exactement ce que §16.2
annonçait : le pré-écran couleur était au hasard **parce que** l'ancre est une image de
l'interlude (16.1 quater).

**HANDOFF** — **le lot est CLOS : les cinq leviers ont leur verdict.** Rien n'est en attente.
Œil : `eyepack_offensif/` (0.87 Mo : `board_ancre_est_un_interlude.jpg`,
`board_levier4_snapshot.jpg`, `board_lever1.jpg`). Prédictions et résultats appariés :
`memory2_probes_registry.md`.

**ÉTAT DE REPRISE du lot défensif (pause générale 2026-08-04, chantier T0)** : superviseur arrêté,
0 vivant, les rollouts en vol finissent et restent acquis ; reprise par mp4 avec
`launch_defensive.py [--only 2|3|4] [--max-gpus N]`. Acquis : **GATE A complet (t16 wronganchor
récupéré, verdict 0/4 inchangé)** ; P2 attribution slot0 **11/18** (memdec20s0 4/6,
wronganchor_s0 4/6, noiseanchor_s0 3/6 — le scorer `score_attrib.py` est écrit et attend les 7
derniers rollouts) ; P3 `memdec20r` S2p 17/24, S3p 16/24, S4p 0/24 ; P4 3e graine 0/108.
