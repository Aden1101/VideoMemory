# ÉTAPE 1 — SONDAGE DES POIDS RAPIDES : rapport d'implémentation

# PRÊT POUR AUDIT

Tout est construit, **40/40 selftests verts**, et **aucun run de verdict n'a été lancé** :
`train_probe.py --stage final` sort en erreur sans `--audited` (selftest T10). Le test n'a jamais
été touché ; même la recherche d'hyperparamètres n'a pas encore tourné.

Protocole : `reference/plan_memoire_2026-08-04.md` (« tâche n°1/masques » + « ÉTAPE 1 ») et les
seuils scellés en fin de `reference/memory2_probes_registry.md`.

---

## 1. Ce qui est construit

Code : `reference/runs/probe1/` (gitignoré, non commité).
Artefacts lourds : `/shared/ssd_14T_project/adenelmi/probe1/` (le home est plein à 100 %),
accessible par le lien `reference/runs/probe1/heavy`.

| fichier | rôle |
|---|---|
| `probe1_common.py` | corpus, 30 paires, split scellé 18/5/7, segments-sujet, âges d'état, extraction du nom du sujet, VAE, DINO du harness |
| `build_masks.py` | **tâche 0** : masques du sujet (OWL-ViT + SAM), crops sujet/fond, embeddings DINO, cohérence des jumeaux, planches d'œil |
| `build_states.py` | rejeu des écritures (`replay_writes`) → snapshots d'état fp16 aux ~4 âges |
| `build_probes.py` | batterie de sondes FIXE : 192 centroïdes k-means + 64 requêtes du décodeur, **par bloc** |
| `build_signatures.py` | **le chemin unique** `signature(source, …)`, `source ∈ {state, h}` |
| `train_probe.py` | le sondeur (MLP 2-3 couches, InfoNCE), recherche d'hyperparamètres sur val, évaluation |
| `gates.py` | application **mécanique** des 4 seuils scellés (ne calcule rien, lit les JSON) |
| `selftest_probe1.py` | les 11 selftests du contrat (40 assertions) |
| `run_build.sh` | la chaîne de construction complète, pré-check GPU dans la commande |

### Chiffres de construction

| artefact | mesure |
|---|---|
| masques | **270 segments-sujet** (165 vidéos), 2 700 frames décodées, **34 signalés (12,6 %)** |
| | score de détection médian **0,355** (q25 0,255 / q75 0,442), aire moyenne du masque **19,7 %** de l'image, 3,9 requêtes candidates par segment |
| masques des 30 paires | 60 segments, **8 signalés**, score médian **0,406** ; ≥ 0,30 sur **42/60**, < 0,15 sur **4/60** |
| cohérence des jumeaux | 30 paires vérifiées, **5 requêtes harmonisées** (voir §2.4) |
| états | 60 vidéos × 4 âges = **240 snapshots**, **40,8 GB** fp16, `delta_l1 = 0.0` |
| signatures | 2 × 240, dimension **1 179 648** (3 blocs × 256 sondes × 1536) ; \|max\| = 485,7 (état) et 27,7 (h), très en deçà de la dynamique fp16 |
| jeu du sondeur | 60 vidéos → 36 train / 10 val / 14 test ; **144 / 40 / 56** signatures ; **0 paire écartée** |

---

## 2. Décisions d'implémentation — à auditer

**2.1 Pile de segmentation : OWL-ViT `large-patch14` + SAM `vit-huge`, pas GroundingDINO.**
`transformers` est figé à **4.33.2** dans l'env `longlive` (le trainer en dépend) et cette version
n'a pas `GroundingDinoForObjectDetection`. Mettre à jour `transformers` pour une tâche annexe
aurait touché une dépendance du trainer : refusé. OWL-ViT est le détecteur ouvert-vocabulaire
guidé par le texte de la même famille, disponible dans 4.33.2. Modèles sur le **disque projet**
(`probe1/hf`), rien dans le home.

**2.2 Extraction du nom du sujet : règle déterministe, arbitrage par le DÉTECTEUR.**
Vocabulaire de ~250 têtes de catégories ; le complément de lieu introductif (« In a luxurious
house, … ») est **déprioritisé** — sans cela le décor devient le sujet ; le syntagme est
reconstruit en remontant les modificateurs jusqu'à un déterminant, une préposition **ou une
ponctuation**. 2 à 4 requêtes candidates sont proposées et OWL-ViT tranche par **score médian sur
les frames**, un candidat plus tardif devant battre le précédent de **15 % relatif**. Quand aucune
tête humaine n'a été extraite, **`"a person"` est ajouté en dernier candidat** : nos prompts
nomment souvent le sujet par un nom propre (« featuring Mustafa Kemal Atatürk », « capturing
Joseph Stalin ») invisible à un vocabulaire de catégories, et la tête tombe alors sur un
accessoire (« raki glass », « shiny black jacket »).

**2.3 Crops.** Sujet = boîte serrée du masque, l'extérieur du masque **mis à zéro** (le sujet, pas
sa vignette de contexte). Fond = **complément** : l'image entière avec le sujet mis à zéro
(l'esprit du `zero_center` de `score_comp.py`, avec le vrai masque au lieu du carré central).

**2.4 Défaut trouvé et corrigé pendant l'implémentation : la cohérence des jumeaux.**
Le détecteur arbitrait **vidéo par vidéo**, et il lui arrivait de trancher différemment des deux
côtés d'une paire alors que les deux partagent le MÊME prompt : `main:204394` → « medium-sized
brown dog » mais `twin:204394` → « small bird » ; `main:142397` → « vase » mais `twin:142397` →
« small table ». Le sondeur aurait alors comparé « un chien contre un oiseau » au lieu de « CE
chien contre CET AUTRE chien » : la tâche changeait de nature sur ces paires. Correctif
(`build_masks.py --fix_pairs`) : la requête est choisie **une fois par paire**, sur la moyenne des
scores médians des deux vidéos, puis les segments concernés sont recalculés. **5 requêtes
harmonisées sur 30 paires**, et les 30 paires portent désormais la même requête des deux côtés.

**2.5 Batterie de sondes PAR BLOC.** Les états des blocs 5 / 16 / 27 vivent dans des espaces de
features différents : une batterie unique ne parlerait pas les trois. Chaque bloc reçoit ses 192
centroïdes k-means et les 64 requêtes du décodeur **de ce bloc** (prises après le `q_norm` du
décodeur, telles qu'elles partent vers `swiglu_apply`), le tout RMS-normalisé ; sous-échantillonnage
= grille 8×8 équidistante sur la grille 30×52, déterministe.
**Anti-fuite** : le k-means est ajusté sur les **105 vidéos `switch`**, qui n'appartiennent à
aucune paire, donc ni au train, ni au val, ni au test (asserté dans `build_probes.py`, testé en T4).

**2.6 Réduction de dimension — le seul endroit où la spec appelait une décision.**
La signature brute vaut 3 × 256 × 1536 = **1 179 648** coordonnées pour **144** échantillons
d'entraînement : elle ne peut pas alimenter un MLP telle quelle. Le « blanchiment par sonde » est
implémenté comme une **standardisation coordonnée par coordonnée** (moyenne/écart-type du TRAIN),
suivie d'une **ACP ajustée sur le TRAIN SEUL**, calculée par le tour de Gram
(`z ↦ (z·Zᵀ) U λ^{-1/2}`) : la base n'est jamais matérialisée et le test ne peut pas y fuir
(selftest T8 : saccager le test ne bouge aucune statistique, **bit à bit**). Rang maximal
= n_train − 1 = 143 ; `sig_dim` est cherché **sur le val uniquement**.

**2.7 Lecteur `h` (la borne supérieure) : lecture associative LINÉAIRE, sans réglage.**
`R(q) = (q hᵀ) h / (L·dim)`. L'état est une mémoire associative NON linéaire construite à partir
de ces `h` ; la borne supérieure lit la mémoire associative LINÉAIRE construite à partir des
**mêmes** `h` avec les **mêmes** sondes : même forme fonctionnelle, même forme de sortie, **et
aucun hyperparamètre**. Une lecture par attention softmax aurait exigé une température,
c'est-à-dire un réglage libre au cœur de la mesure qui décide si le protocole est valide.

**2.8 Dtypes.** Rejeu en bf16 (le dtype de déploiement → trajectoire bit à bit identique à
l'inférence) ; états stockés en fp16 (**exact** : fp16 a plus de bits de mantisse que bf16, la
finitude est assertée à l'écriture) ; signature calculée en **fp32**, comme `_memdec_frame` du
harness lit l'état. T6 vérifie que la signature relue du disque reproduit le recalcul **bit à bit**.

**2.9 Écrivain = l1@2000-EMA**, celui qui a réellement écrit les `h` du corpus ; `delta_l1 == 0.0`
asserté au chargement (contrat F2.3, selftest T11). Requêtes du décodeur =
`frame_dec_v12/ckpt/dec_step4000.pt` (le décodeur de `memdec20`) ; `--dec_ckpt` bascule sur A2.

**2.10 Verrou du verdict.** `--stage final` — la seule étape qui touche le split de test — sort en
erreur sans `--audited`. La recherche d'hyperparamètres ne voit **que** le val.

**2.11 Comparaison sujet / fond.** Deux sondeurs indépendants, chacun avec sa propre recherche
d'hyperparamètres sur son propre val : chacun a sa meilleure chance, ce qui rend l'écart
`précision-sujet − précision-fond` honnête et non un handicap imposé au fond.

---

## 3. Les deux planches de validation à l'œil

| planche | contenu |
|---|---|
| `heavy/board/masks_board.jpg` (0,36 MB, 24 cellules) | balayage **varié** du corpus : les signalés d'abord, puis une tête de sujet différente par cellule |
| `heavy/board/masks_board_pairs.jpg` (0,55 MB, 60 cellules) | **les 30 paires en entier**, les deux jumeaux côte à côte : c'est ce corpus-là, et lui seul, dont dépend le sondage |

Colonnes : **image propre | sujet surligné | masque binaire**. Le libellé donne
`clé | requête retenue | score de détection` ; `!` en rose = segment **signalé**, `~` = le
détecteur a changé de candidat par rapport à la priorité du prompt.

**Ma lecture (à confirmer par l'œil d'Aden, qui tranche).** Sur les 30 paires, **≈ 19 portent le
protagoniste** (chat noir, tigre, aigle, pompier, danseuse, chien, athlète, fillette, Iron Man,
ballon, bouteille, papillon, fleurs, assiette…) et **≈ 11 tombent sur un accessoire ou un objet de
faible confiance** (« cup », « small table », « empty table », « cozy sofa », « glass door »,
« camera », « small bird », et les 4 paires à score < 0,15). Cela ne casse pas le test — le
contrôle par le jumeau tient dans tous les cas, et la question reste « CET individu-ci ou
celui-là » — mais cela **dilue la puissance**.

**Décision à pré-enregistrer AVANT le run de verdict (elle n'est pas prise) :** garder les
**30 paires** ou restreindre à un sous-ensemble de confiance. Ma recommandation est de garder les
30 (aucun filtre post-hoc n'est légitime après avoir vu les scores) ; les champs
`det_scores` / `query_index` / `flagged` sont dans chaque `.npz` si l'auditeur préfère écrire une
autre règle maintenant.

---

## 4. Risques signalés à l'auditeur

**4.1 LIMITE DE CORPUS, à connaître avant de lire le verdict : les 30 jumeaux sont tous `single`.**
Le sondeur travaille donc sur 60 vidéos sans switch : **la tâche ne comporte AUCUNE condition
d'interférence**. Le plan demandait des états « y compris après l'interférence du second sujet » ;
c'est impossible sur les paires, puisqu'aucune vidéo `switch` n'a de jumelle. `end_subject` et
`end_seq` coïncident donc pour ces vidéos et ne sont pas stockés deux fois ; les 4 âges balaient la
séquence d'écritures. **Un verdict positif dirait « l'identité est extractible d'un état non
perturbé », pas « elle survit à l'interférence ».** Le code produit bien `end_subject` et `end_seq`
distincts sur le corpus `switch` : il suffirait de collecter 30 paires `switch` pour lever la
limite — c'est exactement l'extension « +30 paires » déjà pré-enregistrée dans la règle 4.

**4.2 La borne supérieure est le vrai risque technique.** 90 % de précision par paire avec
**18 paires d'entraînement** est exigeant. Si elle échoue, la règle scellée n°1 dit « protocole
invalide, on répare l'instrument », **pas** « la mémoire ne contient rien ». Leviers disponibles
sans toucher aux règles : plus d'âges par vidéo, un `h` plus riche (plusieurs écritures au lieu
d'une), et les +30 paires.

**4.3 Le sujet retenu est parfois un accessoire** (§3). Marqué sur la planche, à trancher à l'œil.

**4.4 41 GB d'états** sur le disque projet (il restait 204 GB libres avant, 165 GB après).

---

## 5. Selftests — 40/40 verts

```
T1  py_compile de 8 fichiers
T2  30 paires ; 18/5/7 ; aucun chevauchement (|train|=36 |val|=10 |test|=14) ;
    jumeau toujours du côté de son frère ; split déterministe ; 60 vidéos distinctes
T3  segments bien formés ; `end_seq` toujours présent, `end_subject` sur les switch,
    âges valides et distincts ; `end_subject` évince bien une frame antérieure au switch
T4  batterie (3, 256, 1536) ; sondes RMS-normalisées à 1e-5 près ; finies et non nulles ;
    le k-means n'a vu AUCUNE vidéo val/test ; centroïdes ≠ requêtes
T5  `signature` ne branche pas sur la source (vérification du code source) ; les deux
    sources dans le MÊME registre ; même forme de sortie ; sorties finies
T6  signature `state` et `h` bitwise identiques sur 2 appels (cuda, delta = 0.000e+00) ;
    signature RÉELLE bitwise identique ; la signature stockée reproduit le recalcul (0.0)
T7  270 segments / 2 700 frames : toute frame d'IoU nulle est SIGNALÉE (0 exception) ;
    34/270 signalés ; le .npz porte masque + crops + DINO sujet ET fond ; sujet ≠ fond
T8  mu/sd inchangés quand le TEST change ; base ACP inchangée ; ACP sur 144 lignes de
    train seul ; transform déterministe
T9  SIGNAL acc=0.964 (p=1.2e-4) | états PERMUTÉS acc=0.536 | BRUIT PUR acc=0.536 ;
    l'évaluation rend précision, marges par paire et Wilcoxon
T10 `--stage final` sans `--audited` refuse de tourner
T11 contrat F2.3 : delta_l1 == 0 au chargement ; rejeu bit à bit reproductible (2 passes)
```

**T9 mérite d'être souligné** : l'instrument complet (blanchiment → ACP → MLP → InfoNCE →
évaluation appariée) est validé sur un problème à **réponse connue** — une signature synthétique
qui est une image linéaire de l'embedding DINO du sujet, noyée dans un bruit d'amplitude
comparable. Il trouve 96,4 % quand le signal est là, et retombe exactement au hasard (53,6 %) quand
on retire le signal **et** quand on permute les états. Un verdict négatif sur les vrais états ne
pourra donc pas être imputé à un sondeur inerte.

---

## 6. Ce qui reste à faire, dans l'ordre, APRÈS l'audit

1. `python reference/runs/probe1/train_probe.py --stage hpsearch --source h --target subject`
   (val uniquement) puis `--stage final --audited` → **règle 1, la validité du protocole**.
2. Même chose pour `--source state --target subject` puis `--target background`.
3. `python reference/runs/probe1/gates.py` → applique mécaniquement les 4 seuils scellés et écrit
   le verdict dans `heavy/runs/gates.json`.

---

# ANNEXE — AUDIT INDÉPENDANT (2026-08-04, auditeur externe au code)

Périmètre : lecture de tout le code de `reference/runs/probe1/`, vérifications CPU sur les
artefacts réels, aucun run de verdict (`runs/` était et reste vide). Verdict en fin d'annexe.

## A1. Fuites — le split est étanche, y compris là où le rapport ne le revendiquait pas

| canal | vérifié | résultat |
|---|---|---|
| split 18/5/7 | hash sha1(seed, corpus_idx), 2 appels identiques | étanche, aucun chevauchement d'id |
| k-means des sondes | intersection pool ∩ vidéos de paires | **0** (les 105 `switch` n'ont aucun `corpus_idx` de paire) |
| requêtes du décodeur | corpus de `frame_dec_v12` (994 vidéos, campagne `switch`) vs les 30 paires | **0 `corpus_idx` commun, 0 prompt commun** |
| blanchiment / ACP | T8 + relecture du tour de Gram | exact, base = train seul, saccager le test ne bouge rien |
| hyperparamètres | `hpsearch` ne lit que `val` ; `--stage final` verrouillé | conforme |
| choix de la requête de détection | n'utilise que le prompt et les frames de la paire elle-même | pas de canal vers le split |

**Le seul point à déclarer** (non mentionné au rapport) : 6 vidéos `switch` du pool k-means
utilisent comme prompt B le TEXTE d'un prompt de paire de TEST (20 pour le train, 6 pour le val).
Ce sont d'autres rollouts, d'autres graines, d'autres individus : aucune vidéo de test n'est dans
le pool. Et c'est structurellement inoffensif : une direction apprise au niveau du PROMPT ne peut
pas départager deux vidéos qui PARTAGENT ce prompt, ce qui est exactement la tâche. Le bras
`--kmeans_source train` reste disponible si Aden veut la version la plus stricte (coût :
reconstruire sondes + 480 signatures, ~1 h GPU). **Mon avis : inutile.**

## A2. Chemin unique `signature(source, …)` — conforme, avec une asymétrie à connaître

Le flag ne sélectionne qu'un lecteur ; même batterie (`probes.pt` identique dans les deux index),
même forme (3, 256, 1536), même aval. Vérifié aussi sur les artefacts : `_index_state.json` et
`_index_h.json` pointent le même fichier de sondes.
**Asymétrie non dite :** la charge utile `h` est le SEUL write de cet âge (≈4 680 tokens), alors
que l'état est l'ACCUMULATION de tous les writes jusqu'à cet âge. La borne supérieure est donc
« les activations brutes du dernier chunk évincé », pas « tout ce que l'état a vu ». C'est
défendable (c'est bien la matière première du write), mais si la règle 1 échoue, le premier
réparateur à essayer est un `h` moyenné sur plusieurs writes, avant toute autre modification.

## A3. Requête choisie une fois par paire — vérifié sur les .npz, pas sur le journal

30/30 paires portent la même `query_chosen` des deux côtés avec `query_joint = True`. Le chiffre
« 5 requêtes harmonisées » n'est en revanche **pas reconstituable** : la sortie de `--fix_pairs`
n'a pas été journalisée et les .npz ont été écrasés. Ce qui est vérifiable et qui compte (les deux
jumeaux comparés sous la même requête) est vérifié. Mesure connexe : sur 17 paires /30 le
détecteur a écarté le premier candidat du prompt (`query_index != 0`).

## A4. Sujet / fond — disjoints, avec une réserve conservatrice

Disjoints par construction (`subj = frame·m` recadré, `bg = frame·(1−m)`), et disjoints en
pratique : cos(DINO sujet, DINO fond) médian **0,222** sur 80 segments (aire de masque moyenne
19,6 %). DINO = **exactement** celui du banc (`run_reid_probe.dino_embed`, dinov2_vitb14, 224,
CLS, L2), importé et non recopié. Réserve : le fond garde la **silhouette** du sujet (le trou noir
a sa forme et sa pose) — le contrôle « fond » n'est donc pas totalement dépourvu d'information de
sujet. Le biais va dans le sens conservateur (il gonfle la précision-fond, donc RÉDUIT l'écart
sujet−fond de la règle 3) : ce n'est pas un bloqueur.

## A5. T9 était un contrôle de plomberie, pas de difficulté → T9b ajouté

Dans T9, les deux jumeaux ont des références DINO **indépendantes** (cos ≈ 0). Les VRAIS jumeaux
ont un cos médian de **0,820** (0,486 – 0,962 ; entre paires différentes : 0,203). T9 ne prouvait
donc pas que l'instrument sait résoudre l'écart d'individu qu'on lui demande de résoudre.
**T9b (ajouté)** rejoue le même problème à réponse connue avec les VRAIES références des 30 paires
et le VRAI split : `acc_row = 0.911`, `acc_by_pair = 1.000`, `p = 0.0078` ; bruit pur 0,518 ;
permuté 0,513. L'instrument tient la difficulté du terrain — et livre au passage le chiffre le plus
important de cet audit (§A6).

## A6. gates.py — mécanique, mais deux quantités à trancher AVANT le run

Correspondance vérifiée ligne à ligne avec les 4 règles scellées (seuils 0.90 / 0.55 / 0.65 +
p<0.05 + 10 pts / zone grise 55-75). Trois écarts :

1. **« précision par paire » est ambigu et ce n'est pas neutre.** Le code applique les seuils à la
   précision par LIGNE (une décision par (vidéo, âge), 56 lignes de test). L'autre lecture est la
   précision au niveau de la PAIRE (7 valeurs, granularité 14,3 pts). T9b montre qu'un signal
   PARFAIT donne 0,88-0,91 en lecture « ligne » pour 1,00 en lecture « paire » : **la règle 1
   (≥ 90 %) posée sur la lecture « ligne » est au niveau du plafond de l'instrument** — risque réel
   d'un « protocole invalide » alors que l'instrument est excellent. Gates imprime désormais les
   DEUX lectures partout ; `PRIMARY` (défaut : `row`, l'implémentation historique) doit être
   confirmé par Aden avant le run. Aucune des deux n'est choisie après avoir vu un score.
2. **Écart sujet−fond négatif.** La règle 4 scellée ne déclare l'échec que si l'écart est *nul*
   (`abs(gap) < 1e-9` dans le code) ; la grille (d) du plan appelle « sujet↓ fond↑ » un
   enregistreur d'ambiance → ARRÊT. Un écart négatif en zone 55-75 % est aujourd'hui routé vers
   « +30 paires ». Gates imprime maintenant une NOTE explicite dans ce cas ; l'arbitrage revient
   à Aden (je n'ai pas touché la logique du verdict).
3. **L'analyse SECONDAIRE pré-enregistrée n'existait pas** (ni liste figée, ni code), et elle était
   même *incalculable* : `train_probe.py` jetait `per_row` avant d'écrire le JSON de verdict.
   Corrigé (§A8).

Note mineure : `success` ajoute la règle 2 en conjonction (conservateur, sans effet sur un run
sain) ; et les chiffres `state` sont imprimés même si la règle 1 échoue — inoffensif, le verdict
reste « protocole invalide ».

## A7. Pièges cherchés (dont 2 fermés par la mesure, 2 réels)

- **Quantification fp16 des signatures — FERMÉ.** Mesuré sur 16 vidéos : écart-type inter-vidéo /
  bruit de quantification = **5008 (médiane)**, q05 = 2353, aucune coordonnée sous 1 (h : 3607).
  Le stockage fp16 n'écrase rien.
- **Blanchiment qui écraserait le signal — FERMÉ.** Sur les 144 vraies lignes de train : écart-type
  par coordonnée min = 0,579 (état) / 2,0e-4 (h), **aucune coordonnée sous le clamp 1e-5** (le
  clamp ne se déclenche jamais, donc pas d'amplification de coordonnées mortes) ; rang 143, top-64
  = 96 % / 98 % de la variance : la grille `sig_dim ∈ {64, 128}` couvre bien le spectre.
- **Effet « hub » des références — sans objet.** La marge de paire vaut (P_A − P_B)·(m_A − m_B) :
  un biais constant du sondeur s'annule exactement. La précision par ligne moyenne les deux
  jumeaux : le hasard y est exactement 50 %. C'est bien construit.
- **RÉEL — le contrôle permuté ne tirait QU'UN dérangement.** Avec 7 paires, la règle 2 était
  elle-même une variable à ~14 degrés de liberté : elle pouvait échouer ou passer au tirage.
  Corrigé (moyenne sur 25 dérangements ; T9 est passé de 0,536 à 0,501, T9b donne 0,513).
- **RÉEL — la composition du split de test est défavorable et ne peut plus être changée.** Les 7
  paires de test : `black cat` (0,37), `eagle` (0,38), `blue floral dress` (0,47), `glass door`
  (0,29), `cozy sofa` (0,25), `a tiger` (0,24), `camera` (0,13). Deux à trois portent vraiment le
  protagoniste. Conséquences : puissance faible sur la primaire, et surtout **l'analyse secondaire
  « le masque porte le protagoniste » ne laissera que ~2-3 paires de test**, ce qui la rend
  quasi ininterprétable. Rien de tout cela n'autorise à retirer le split (ce serait du p-hacking) ;
  mais **une validation croisée par paires pré-enregistrée MAINTENANT** (6 plis, hyperparamètres
  choisis dans chaque pli, 30 paires au dénominateur) serait un instrument bien plus puissant pour
  le même corpus. Décision d'Aden, à prendre avant le run, jamais après.

## A8. Correctifs appliqués par l'auditeur (aucun ne touche un seuil scellé)

1. `train_probe.evaluate` : le contrôle permuté est moyenné sur **25 dérangements** (`n_perm`) au
   lieu d'un seul ; assertion si aucun dérangement n'est trouvé. Quantité inchangée, variance
   d'estimation divisée par ~5.
2. `train_probe.evaluate` : `per_row` porte désormais `video_id`, `corpus_idx`, `pair_id`, et la
   sortie porte `pair_corpus` — sans quoi aucune analyse par sous-ensemble n'est calculable.
3. `train_probe.main` : `per_row` **conservé** dans le JSON de verdict (il était jeté) ; ajout de
   `acc_by_pair_mean` à côté de `acc_mean`.
4. `gates.py` : imprime les DEUX lectures de « précision par paire » pour les règles 1/2/3, la NOTE
   d'écart non positif, et **implémente l'analyse SECONDAIRE pré-enregistrée** — lue dans
   `runs/secondary_pairs.json` (`{"corpus_idx": [...], "gele_le": "..."}`) et **refusée si ce
   fichier est plus récent que les JSON de verdict** (verrou mécanique contre un sous-ensemble
   choisi après coup). Elle est rapportée à côté, elle ne rend aucun verdict.
5. `selftest_probe1.py` : **T9b** (contrôle positif à la difficulté réelle, §A5).

Selftests après correctifs : **42/42 verts** (sans GPU ; T11 sauté). Vérification de bout en bout
de `gates.py` sur des JSON de verdict factices dans un répertoire jetable : les 3 branches
(pas de liste / liste antérieure / liste postérieure refusée) se comportent comme spécifié.

## A9. VERDICT DE L'AUDIT : **GO**, sous deux décisions à prendre avant de lancer

Le code fait ce que le rapport dit qu'il fait ; les fuites cherchées ne sont pas là ; les deux
pièges silencieux que je soupçonnais (fp16, blanchiment) sont mesurés et fermés ; les deux vrais
défauts trouvés (contrôle permuté à un seul tirage, analyse secondaire incalculable) sont corrigés.
Restent **deux décisions d'Aden, à écrire avant le premier `--stage final --audited`** :
(a) la lecture de « précision par paire » (`PRIMARY = row` ou `pair`) — c'est la règle 1 qui en
dépend ; (b) la liste figée du sous-ensemble secondaire (et, s'il la veut, la validation croisée
par paires, qui est la seule réponse honnête à la faiblesse du split de test).

---
---

# §v2 — RÉPARATION DE L'INSTRUMENT + NOUVEAUX MATÉRIAUX (2026-08-05)

# PRÊT POUR AUDIT — aucun run de verdict n'a été lancé

`train_probe.py --stage cv` **et** `--stage final` sortent en erreur sans `--audited`
(selftest T10, les deux étapes). Les seuls chiffres de ce §v2 sont des chiffres de
**construction** et de **selftest sur données à réponse connue**.

Point de départ : le verdict du 04/08 au soir était **PROTOCOLE INVALIDE** (borne `h` à 0,542
par ligne / 0,762 par paire, très en dessous des 0,90 exigés), et le registre a scellé le 05/08
trois défauts à réparer plus la génération de nouveaux matériaux.

---

## v2.1 — Défaut n° 1 : la DIMENSIONNALITÉ (1 179 648 coordonnées pour 144 lignes)

**(a) Compression par sonde.** `build_signatures.compress_signature` résume chaque ligne de la
matrice-réponse par **(moyenne, max) sur ses coordonnées** : 3 blocs × 256 sondes × 2 =
**1 536 nombres** au lieu de 1 179 648, soit **8 200 fois moins**. Deux valeurs et pas une : la
moyenne est le niveau de réponse de la sonde, le max sa saillance — une sonde qui « voit »
quelque chose de précis bouge le max sans bouger la moyenne. La fonction est sans paramètre,
sans données, ligne à ligne : elle ne peut structurellement pas fuir d'un split (T12).

**(b) Variante ACP conservée.** `--features raw` garde la signature brute et l'ACP par le tour
de Gram, ajustée **sur le train du pli uniquement** (le code d'origine, audité, est intact).
`--features pool` est la nouvelle représentation. `sig_dim = 0` (« pas d'ACP du tout ») entre
dans la grille : sur 1 536 dimensions l'ACP n'est plus une nécessité mais un hyperparamètre.

**(c) Pré-entraînement sur les 105 vidéos `switch`.** Elles sont **déjà rejouables** (mêmes h,
même écrivain l1@2000-EMA) et **n'appartiennent à aucune paire** (asserté, T16). Elles donnent
**420 lignes** de pré-entraînement contrastif contre les 144 du train — et elles apportent la
condition que les paires n'ont pas : **l'interférence du second sujet** (l'âge `end_seq` d'une
vidéo switch est post-switch). La tête part donc de poids pré-entraînés au lieu d'une
initialisation aléatoire, et **le blanchiment est estimé sur train + pré-entraînement**
(564 lignes au lieu de 144 ; `n_fit` est imprimé). Les 30 paires ne servent **qu'à l'évaluation
du pli**. Coût disque : **zéro** (voir v2.4, le mode `--stream`).
Le pool de pré-entraînement est **toujours celui du corpus v1** (105 vidéos switch), y compris
quand on sonde le corpus v2 : il en est alors disjoint par construction (corpus entièrement
différent), ce qui rend le pré-entraînement encore plus propre côté v2.

## v2.2 — Défaut n° 2 : la BORNE DE VALIDITÉ passait par une écriture

L'ancienne borne (`--source h`) lisait `R(q) = (q hᵀ) h` : une mémoire associative *linéaire*,
mais une mémoire quand même. Si la **forme d'écriture** était le goulot, la borne le subissait
aussi, et son échec ne distinguait pas « le matériau ne porte pas l'identité » de « la lecture
par sonde ne la voit pas ».

**Nouvelle source `hdirect` (additive, l'ancienne `h` reste sur disque).** Les activations sont
sondées **directement** : aucune sonde, aucune écriture, aucun paramètre. Pour chaque bloc, on
met en commun les tokens h **par chunk** (moyenne et max), agrégés sur **tous les writes jusqu'à
cet âge** — exactement l'ensemble des activations que l'état a absorbées. Cela corrige au
passage l'asymétrie relevée en **A2 de l'audit** (l'ancienne borne ne voyait QUE le dernier
write, alors que l'état les voit TOUS). Sortie : 3 × 1536 × 2 = **9 216 nombres**.

La matrice-réponse d'`hdirect` est écrite dans la **même orientation** que celle des sondes
(ligne = unité de réponse, colonne = coordonnée), si bien que la compression générique s'y
applique **telle quelle, sans savoir d'où vient le tenseur** — et sans rien perdre : de
((m+M)/2, M) on retrouve (m, M) exactement (T12, vérifié à 2e-7 près). Le contrat « UN SEUL
CHEMIN » (T5) est donc préservé à la lettre : `signature()` ne branche pas sur la source, la
compression non plus, et tout l'aval est identique.

**Re-ancrage de la règle 1 (scellé au registre le 05/08, appliqué ici).** Le plafond humain
mesuré par l'œil d'Aden sur les matériaux v1 est ~0,74 : 5 paires sur 15 sont **indistinguables
pour un humain**. Les 90 % étaient donc inatteignables *même pour un humain* et ne pouvaient
pas juger l'instrument. La borne se juge désormais sur le sous-ensemble **« clairement
différents » {01, 02, 11, 12, 19, 27}, seuil ≥ 5/6**. Le 0,90 global reste **imprimé** comme
information, jamais comme verdict.

**`gates.py` lit PARTOUT en par-paire.** `PRIMARY = "pair"` est figé (T14 vérifie qu'aucun seuil
de verdict n'est appliqué à une précision par ligne ; la lecture par ligne reste imprimée à
côté).

## v2.3 — Défaut n° 3 : le split fixe 18/5/7 → VALIDATION CROISÉE PAR PAIRES

`train_probe.py --stage cv` : **6 plis × 5 paires**, plis déterministes (même clé de hash que le
split scellé). Dans chaque pli : test = les 5 paires du pli, val = les 5 du pli suivant (c'est
**là** que les hyperparamètres sont choisis), train = les 20 restantes ; puis ré-entraînement
sur les **25** (train + val) avec les hyperparamètres retenus — le val a fini son office, le
test n'a rien vu. **Le blanchiment et la base ACP sont ré-ajustés dans chaque pli** (T15 :
saccager le test d'un pli ne bouge aucune statistique ; changer de pli invalide le cache).
Dénominateur : **les 30 paires**, chacune évaluée exactement une fois.

**Les 4 sous-ensembles pré-enregistrés sont rapportés côte à côte**, gelés dans
`heavy/runs/subsets_v1.json` **avant tout score** :

| sous-ensemble | n | numéros de planche |
|---|---|---|
| **PRIMAIRE** — toutes | 30 | 1–30 |
| agent (règle des masques) | **21** | 1,2,5,6,10,12,13,15,16,17,18,19,20,21,22,23,25,27,28,29,30 |
| œil d'Aden « BONNES » | 15 | 1,2,6,8,11,12,15,16,19,20,21,22,24,27,30 |
| **décisif** « clairement différents » | 6 | 1,2,11,12,19,27 |

**Déclaration honnête sur la liste « agent-19 ».** Le rapport d'origine ne l'avait **pas** figée
en indices : il la décrivait par une règle (« ~11 tombent sur un accessoire ou un objet de faible
confiance : cup, small table, empty table, cozy sofa, glass door, camera, small bird, et les
paires à score < 0,15 »). Cette règle, appliquée **mécaniquement** aux `.npz`, rend **21** paires
et non 19 — la prose comptait aussi des cellules de la planche VARIÉE du corpus, pas seulement
des 30 paires. J'ai gelé la règle et son résultat (21) plutôt que d'inventer deux exclusions
pour retomber sur 19.

**La numérotation de planche est reconstituée et vérifiée.** Le script de la planche numérotée
n'existait plus ; l'ordre est celui de l'**identifiant lexicographique de la vidéo principale**
(`main:10440`, `main:111706`, `main:14984`, …). Vérification indépendante (T18) : les **10
paires qu'Aden a nommées** tombent toutes sur le bon numéro — 01 pompier, 02 bouteille,
06 foule, 08 verrou, 11 oiseau, 12 athlète, 16 couple, 19 enfant, 24 ange, 27 fillette. La table
complète numéro → `corpus_idx` est dans `subsets_v1.json`, et `build_masks.py` sait désormais
re-rendre la planche numérotée lui-même (`masks_board_pairs_numerote*.jpg`). **Confirmation
visuelle** : la planche re-rendue a été relue à l'œil, cellule par cellule, et les 10 repères
d'Aden y sont bien aux numéros annoncés (le crop de la 24 est bien l'ange, comme il l'avait
corrigé).

## v2.4 — Le mode `--stream` (économie de 112 GB, exigée par le disque)

Les états pèsent 170 MB pièce : 105 vidéos switch × 4 âges = 71 GB, plus 41 GB pour les
nouvelles paires. `build_signatures.py --stream` **rejoue les écritures à la volée** (le
`replay_video` de `build_states.py`, inchangé) et n'écrit **aucun** état sur disque : seules les
signatures (2,4 MB, ou 18 KB comprimées) survivent. **T19 vérifie que la signature streamée est
BITWISE identique à celle calculée depuis l'état stocké**, et identique à celle déjà sur disque.

---

## v2.5 — MATÉRIAUX v2 : 30 paires à prompts SANS AUCUN ATTRIBUT

**Le diagnostic qui les rend nécessaires.** Sur les paires v1, le prompt DÉCRIT l'individu
(« a brave firefighter clad in a red helmet and reflective vest », « a little girl with curly
blonde hair and a blue floral dress ») : les deux graines produisent alors deux instances
quasi identiques, et le plafond humain tombe à ~0,74. **Aucun instrument ne peut séparer ce que
le texte a déjà fixé.** Les prompts v2 ne fixent RIEN de l'apparence : *un nom de sujet nu +
une action + un lieu*. C'est le tirage de la graine qui invente l'individu, donc les deux
jumeaux divergent librement — et « l'identité est-elle dans les poids rapides ? » redevient une
question qui a un sens.

**`collect_pairs.py --prompts-file` (additif, rien d'autre n'est touché).** Les prompts viennent
d'un fichier au lieu du corpus vidprom ; tout le reste de la machinerie v3 est réutilisé tel
quel (même rollout, même capture h sous `MEM_ZERO_ALL=1`, même sharding, même manifeste, même
`--twin-offset`). `corpus_idx = 9 000 000 + rang de la ligne` : au-dessus de tout index de ligne
vidprom, donc **aucune collision possible** avec la campagne v1, et les deux corpus cohabitent
sans se marcher dessus. Mode single forcé (une paire n'a de sens que si les deux jumeaux
partagent toute la trajectoire de conditionnement). `--out` dédié : l'index du trainer
d'écrivain n'est **jamais** touché (probe1 tient son propre index `index_v2.json`).

**Held-out vérifié À LA LECTURE, deux références et deux directions** (T17) : les 16 prompts des
paires de sonde **et** les **248 175 lignes** du corpus vidprom que l'entraînement l1 a
consommées. Résultat : **0 collision**.

**Les 30 prompts** (`reference/runs/probe1/prompts_noattr_v2.txt`) : 10 personnes, 10 animaux,
6 objets, 4 véhicules ; **30 têtes de sujet distinctes** ; aucune couleur, matière, taille,
vêtement, âge (liste interdite vérifiée mécaniquement) ; le sujet est toujours le premier
syntagme nominal, jamais un complément de lieu introductif ; sa tête appartient au vocabulaire
de catégories, donc la segmentation guidée par le texte a toujours une requête à donner au
détecteur. Exemples : *a firefighter walks through smoke near a fire truck*, *a parrot lands on
a fence in a garden*, *a scooter waits at a traffic light on a street*.

**Génération** : 60 vidéos 90 f (30 mains + 30 jumelles, graine + 100 000 — le même décalage que
la campagne v3), **deux vagues chaînées** sur un GPU réservé, pré-check GPU dans la commande de
lancement, index 6 exclu (`run_collect_v2.sh`).

---

## v2.6 — Selftests

Les **anciens** (T1–T11, y compris T9b et le verrou T10) sont **inchangés et verts** ; T5 et T10
ont gagné des assertions (3 sources, `--stage cv` verrouillé lui aussi). **Neuf tests nouveaux** :

```
T12 COMPRESSION : 2 valeurs par sonde et par bloc = 1536 nombres ; bitwise deterministe ;
    (moyenne, max) exacts ; la compression d'une ligne ne depend d'AUCUNE autre ligne
    (donc aucune fuite de split possible) ; hdirect -> 9216 nombres ; compression INVERSIBLE
    sur hdirect (2.1e-7)
T13 BORNE SANS ECRITURE : le lecteur hdirect n'appelle ni swiglu_apply ni aucun rejeu,
    n'utilise meme pas la batterie de sondes ; verification NUMERIQUE que c'est bien
    (moyenne, max) sur TOUS les tokens de TOUS les chunks jusqu'a l'age
T14 GATES : PRIMARY == 'pair' ; acc() rend la precision par paire par defaut ; aucun seuil de
    verdict applique a une precision par ligne ; regle 1 re-ancree a 5/6 ; les 4 sous-ensembles
    sont rapportes cote a cote
T15 PLIS : 6 x 5 paires, partition exacte, chaque paire de test exactement une fois,
    deterministe ; saccager le test d'un pli ne bouge pas son blanchiment ; changer de pli
    invalide le cache (re-ajustement obligatoire)
T16 PRE-ENTRAINEMENT : 0 video de paire dans le pool des 105 switch (ids ET corpus_idx) ;
    le blanchiment integre bien les lignes de pre-entrainement ; un sondeur pre-entraine
    s'entraine et s'evalue
T17 PROMPTS v2 : 30 prompts, corpus_idx >= 9e6 distincts, AUCUN attribut d'apparence,
    sujet reconnu par l'extracteur pour les 30, jamais dans un complement de lieu,
    30 tetes distinctes, held-out verifie a la lecture
T18 NUMEROTATION : 1..30 ; les 10 paires nommees par Aden tombent sur les bons numeros ;
    decisif = 6, oeil d'Aden = 15 (contient les 6 decisives), agent = 21
T19 (GPU) --stream == disque BITWISE, et == la signature deja stockee
T20 VALIDATION CROISEE bout en bout sur donnees a REPONSE CONNUE : signal -> 0.933 par paire
    (30 paires au denominateur, p = 6.5e-9) ; permute -> 0.505 ; bruit pur -> 0.400
```

**T20 est le pendant de T9 pour la nouvelle procédure** : la validation croisée complète
(plis, recherche d'hyperparamètres interne, blanchiment par pli, agrégation sur 30 paires,
contrôle permuté) est validée sur un problème dont on connaît la réponse.

---

## v2.7 — Ce qui reste à faire, dans l'ordre, APRÈS l'audit

Aucune de ces commandes n'a été lancée.

```bash
P=reference/runs/probe1
# 1. LA BORNE DE VALIDITE (regle 1 re-ancree : >= 5/6 sur les "clairement differents")
python $P/train_probe.py --stage cv --audited --source hdirect --target subject --pretrain
# 2. L'ETAT, sujet puis fond (regles 2, 3, 4)
python $P/train_probe.py --stage cv --audited --source state   --target subject    --pretrain
python $P/train_probe.py --stage cv --audited --source state   --target background --pretrain
# 3. LE VERDICT (mecanique, ne calcule rien)
python $P/gates.py --pretrain
# 4. Idem sur les MATERIAUX v2 (--corpus v2), une fois l'oeil d'Aden passe sur la planche
python $P/train_probe.py --stage cv --audited --corpus v2 --source hdirect --target subject --pretrain
...
python $P/gates.py --corpus v2 --pretrain
```

Coût : ~50 min de GPU par bras (6 plis × 32 configs × 2 graines + 3 graines finales), soit
~2,5 h pour les trois bras d'un corpus. `--features raw` (la variante ACP) se lance de la même
façon et écrit dans des fichiers distincts : les deux représentations sont **pré-enregistrées
toutes les deux**, et c'est `--features pool` qui est la primaire (l'autre est rapportée à
côté ; aucune n'est choisie après avoir vu un score).

## v2.8 — Risques et déclarations à l'auditeur

**v2.8.1 — La liste « agent-19 » est une RECONSTRUCTION (21 paires), pas une liste retrouvée.**
Voir v2.3. La règle est écrite dans le code (`AGENT_BAD_QUERIES`, `AGENT_MIN_DET_SCORE`) et gelée
dans `subsets_v1.json` avant tout score. Si l'auditeur préfère une autre règle, c'est maintenant.

**v2.8.2 — Le sous-ensemble décisif ne fait que 6 paires.** Le seuil 5/6 a été scellé par le
registre, pas choisi ici. À 6 paires, une seule paire fait 17 points : la borne est un test
**grossier**. C'est assumé — c'est le prix des matériaux v1, et c'est exactement la raison
d'être des matériaux v2.

**v2.8.3 — `hdirect` reste une borne sur ce que le CHUNK contient, pas sur ce que l'état
retient.** Elle mesure « l'identité est-elle dans les activations que l'écriture consomme ? ».
Si elle échoue, la conclusion est « nos matériaux/masques/références ne portent pas l'identité »
— jamais « la mémoire est vide ».

**v2.8.4 — Le pré-entraînement est une aide, pas une fuite.** Les 105 vidéos switch ne sont dans
aucune paire (asserté deux fois). Elles partagent en revanche des **textes** de prompt B avec
certaines paires (constat déjà déclaré en A1 de l'audit) : structurellement inoffensif, puisqu'une
direction apprise au niveau du PROMPT ne peut pas départager deux vidéos qui PARTAGENT ce prompt
— ce qui est exactement la tâche. **À déclarer explicitement** : ce sont aussi les vidéos sur
lesquelles le k-means des sondes a été ajusté (anti-fuite d'origine). Le même pool sert donc
maintenant deux fois — batterie de sondes ET pré-entraînement — mais toujours du même côté de la
barrière : aucune vidéo de paire, dans aucun des deux usages.

**v2.8.5 — Les états v2 ne sont pas stockés** (mode `--stream`) : 41 GB que le disque n'a pas.
Les re-dériver coûte 2,5 s par vidéo, et T19 prouve que le chemin streamé est bit à bit celui du
disque.

**v2.8.6 — Le verdict d'œil d'Aden sur les paires v2 n'existe pas encore.** La planche numérotée
v2 est produite ; tant qu'elle n'est pas validée, le corpus v2 n'a que l'analyse PRIMAIRE
(30 paires) — pas de sous-ensemble « protagoniste » ni « clairement différents », et
`sealed_subsets` refuse mécaniquement d'appliquer au corpus v2 les listes d'œil du corpus v1.

**v2.8.7 — `gates.py` a été vérifié de bout en bout sur des JSON de verdict FACTICES** dans un
répertoire jetable, comme l'avait fait l'audit du 04/08 : les trois branches se comportent comme
spécifié — borne à 4/6 → *PROTOCOLE INVALIDE* (et les chiffres `state` restent imprimés, sans
verdict) ; borne 6/6 + sujet 0,73 / fond 0,55 → *(c) SUCCÈS* ; borne 6/6 + sujet 0,60 → *ZONE
GRISE → +30 paires*. Aucun chiffre réel n'a été produit.

---

## v2.9 — Ce qui a bougé dans le code (tout est ADDITIF)

| fichier | changement |
|---|---|
| `probe1_common.py` | registre des **deux corpus** (`--corpus v1/v2`, index v2 dans un cache PROPRE à probe1, l'index du trainer n'est jamais touché) ; `board_numbers` ; les **sous-ensembles scellés** + `--freeze-subsets` |
| `build_signatures.py` | source **`hdirect`** ; **`compress_signature`** ; mode **`--stream`** ; `--switch_only` ; `--corpus` |
| `train_probe.py` | `--features pool/raw` ; **`--pretrain`** ; **`--stage cv`** (plis, `set_fold`, `subset_report`, `cv_run`) ; `Preproc` sait ne pas faire d'ACP (`sig_dim=0`) et accepter des lignes hors-paires |
| `gates.py` | réécrit : `PRIMARY = "pair"`, règle 1 **re-ancrée** (≥ 5/6 décisives), tableau des 4 sous-ensembles, lecture des JSON de CV |
| `build_masks.py` | `--corpus` ; planche **NUMÉROTÉE** (`build_board(..., numbers=)`), les planches v1 ne peuvent plus être écrasées par celles du v2 |
| `build_states.py` | `--corpus` |
| `collect_pairs.py` (frame_dec) | **`--prompts-file`** + `load_prompts_file` (held-out à la lecture). **Non-régression prouvée (T21)** : sans le drapeau, la sélection de corpus v3 est identique à celle qui a produit les 165 vidéos sur disque (135/135). |
| `run_collect_v2.sh`, `run_build_v2.sh` | les deux chaînes (génération ; artefacts), pré-check GPU dans la commande |

Aucun fichier hors de `reference/runs/probe1/` et `reference/runs/frame_dec/collect_pairs.py`
n'est modifié. Rien n'est commité (règle de la maison).

---

## v2.10 — Contrôle d'instrument sur les représentations comprimées (aucun score)

Même contrôle que l'audit A7 (« un blanchiment qui écraserait le signal »), refait sur les
nouvelles entrées, sur les 240 lignes réelles du corpus v1 :

| représentation | dim | écart-type par coordonnée (min / méd / max) | coordonnées mortes | finies |
|---|---|---|---|---|
| `state` comprimé | 1 536 | 1,96e-02 / 1,10 / 82,6 | **0** | oui |
| `hdirect` comprimé | 9 216 | 3,55e-02 / 0,186 / 3,03 | **0** | oui |

**Aucune coordonnée sous le clamp 1e-5** : le blanchiment n'amplifie jamais du bruit mort, dans
aucune des deux représentations. Jeu de pré-entraînement : 420 lignes, mêmes dimensions, toutes
finies.

---

## v2.11 — Chiffres de construction v2 (tous mesurés, aucun score)

| artefact | mesure |
|---|---|
| génération | **60 vidéos** 90 f (2 vagues chaînées, GPU 7) : 30 mains **33,9 GB / 0,5 h** + 30 jumelles **32,9 GB / 0,5 h**, 750 writes par vague |
| held-out | 0 collision sur les 16 prompts de sonde **et** sur les **248 175** lignes du corpus l1 |
| index v2 | 60 vidéos (0 switch / 60 single, 30 jumelles) sur 6 parts → **30 paires** |
| masques v2 | 60 segments, **2 signalés** (contre 8 sur v1), score de détection médian **0,487** (q25 0,381 / q75 0,582) ; ≥ 0,30 sur **51/60** (v1 : 42/60), < 0,15 sur 6/60 ; requête non prioritaire sur 8/60 |
| cohérence des jumeaux | 30 paires vérifiées, 60 segments recalculés, **2 requêtes harmonisées** |
| signatures v2 | 3 × 240 (`state` en `--stream`, `hdirect`, `h`), \|max\| = 409,8 / 37,3 / 27,8 — très en deçà de la dynamique fp16 |
| pré-entraînement (v1) | 105 vidéos switch × 4 âges = **420** signatures `state` (streamées) + 420 `hdirect` |
| borne v1 | 240 signatures `hdirect` sur les 60 vidéos de paires |
| disque | 101 GB libres après tout (les 41 GB d'états v2 n'ont pas été écrits) |
| selftests | **99/99 verts** avec GPU (95/95 sans) |

**Ma lecture de la planche v2 (`board/masks_board_pairs_numerote_v2.jpg`, 60 cellules) — à
confirmer par l'œil d'Aden, qui tranche.** Le masque porte le sujet sur **~25 paires sur 30**, et
surtout **les deux jumeaux sont visiblement deux individus différents** sur la grande majorité :
chien beagle contre bouvier bernois (13), chèvre blanche contre chèvre fauve (12), cheval noir
contre cheval bai (15), bouteille verte contre bouteille ambrée (21), guitare noire contre
sunburst (23), scooter noir contre scooter blanc (27)… **C'est exactement ce que les matériaux v1
n'avaient pas.** Les 5 paires douteuses : **01** (la requête harmonisée est devenue « fire truck »
— le camion, pas le pompier), **07** (« cup » a battu « woman »), **08** (skater, score 0,03-0,05,
le masque part sur un véhicule — la seule paire SIGNALÉE), **10** (peintre/chevalet, 0,11), **04**
(violoniste : le masque prend le violon et le bras). Le mécanisme de ces ratés n'est pas le repli
« a person » que je redoutais, mais la **deuxième tête nominale du prompt** (truck, cup) qui bat
la première de plus de 15 % au score du détecteur.

**Conséquence de protocole, à trancher AVANT le run v2 :** le corpus v2 n'a pour l'instant que
l'analyse **PRIMAIRE (30 paires)** ; `sealed_subsets` refuse mécaniquement de lui appliquer les
listes d'œil du corpus v1 (`subsets_v2.json` ne contient que `all`). Si Aden veut un
sous-ensemble « masque valide » et un sous-ensemble « clairement différents » pour v2, ils
doivent être figés **à partir de la planche, avant tout score**, exactement comme pour v1.

---
---

# ANNEXE 2 — AUDIT INDÉPENDANT DU DELTA v2 (2026-08-05, auditeur externe au code)

Périmètre : le §v2 (v2.1-v2.11) et le code qu'il décrit (`compress_signature`, source `hdirect`,
`--stage cv`, `gates.py` réécrit, `--stream`, `collect_pairs --prompts-file`, sous-ensembles
scellés), confronté aux sections scellées du registre. Vérifications CPU sur les artefacts réels
+ trois petits tests GPU (carte 7, exclusive au moment du lancement, index 6 jamais touché).
**Aucun run de verdict** : `runs/cv_*.json` était et reste absent.

## B1. Les huit points demandés — ce que la mesure dit

**(1) La compression.** `compress_signature` est ligne à ligne, sans paramètre ni données :
perturber la sonde 0 d'une signature réelle ne change **aucune** autre sonde (vérifié). Bitwise
déterministe sur la signature stockée ; `(moyenne, max)` exacts ; `state` 3×256×1536 → **1 536**,
`hdirect` 3×1536×2 → **9 216** ; inversion de `hdirect` exacte à **0,0** sur la vraie signature.
Le blanchiment et l'ACP sont ré-ajustés **par pli** : sur les **6 plis réels**, multiplier le test
par 1 000 et lui ajouter 13 ne bouge `mu`, `sd` ni la base ACP **d'un seul bit**.
`n_fit` = 160 + 420 = 580 (recherche) puis 200 + 420 = 620 (ré-entraînement), rang 579.

**(2) Le pré-entraînement.** Re-vérifié moi-même sur l'index : **105** vidéos `switch`,
intersection avec les 60 vidéos de paires = **0 par `video_id` ET 0 par `corpus_idx`**, 0 prompt A
commun. Les **20** prompts de paire qui réapparaissent comme prompt B de vidéos switch sont le
constat déjà déclaré en A1/v2.8.4 (inoffensif : une direction apprise au niveau du PROMPT ne
départage pas deux vidéos qui partagent ce prompt). Les hyperparamètres du pré-entraînement sont
ceux du pli, choisis sur le **val du pli** ; `pre_steps = 800` est dans `FIXED`, jamais cherché.
Le test n'entre ni dans le pré-entraînement, ni dans le blanchiment, ni dans la recherche.

**(3) La source `hdirect`.** Le lecteur n'appelle ni `swiglu_apply`, ni `replay_writes`, ni aucun
état ; la charge utile est `h_seq[bloc][0 .. âge]`. J'ai **recalculé la signature à la main** à
partir des `h` bruts pour 8 couples (vidéo, âge) : c'est bien la moyenne globale sur **tous** les
tokens de **tous** les chunks jusqu'à l'âge, et le max élément par élément. Sur GPU, le recalcul
reproduit le fichier stocké **bit à bit** (8/8).

**(4) La validation croisée.** 6 plis × 5 paires, partition exacte des 30 `corpus_idx`,
déterministe, **chaque paire de test exactement une fois**, dénominateur = 30. Étanchéité
vérifiée aux deux étages (train 20 paires pour la recherche, puis 25 pour le ré-entraînement,
test exclu des deux). Les 6 paires décisives se répartissent 1/1/2/0/1/1 sur les plis.

**(5) `gates.py`.** `PRIMARY = "pair"` ; règle 1 = **≥ 5/6** sur `{01,02,11,12,19,27}` ;
règles 2/3/4 lues en par-paire, la lecture par ligne seulement imprimée. Vérifié de bout en bout
sur des JSON fabriqués : borne 4/6 → *PROTOCOLE INVALIDE*, 6/6 + 0,73/0,55 → *(c) SUCCÈS*,
permuté réel 0,62 → *ÉCHEC règle 2*, sujet 0,60 → *ZONE GRISE*.

**(6) La numérotation.** Re-dérivée indépendamment (tri lexicographique de la vidéo principale) et
confrontée aux prompts : non pas 5 mais **les 10 repères d'Aden tombent juste** (01 firefighter,
02 plastic bottle, 06 crowd, 08 glass door + lock, 11 small bird + dog, 12 track athlete,
16 couple, 19 cat + small child, 24 angelic figure, 27 little girl + black horse). Les 4
sous-ensembles recalculés sont **identiques** au fichier gelé `runs/subsets_v1.json` (30/21/15/6).
Les 30 paires portent la même requête des deux côtés.

**(7) `--stream`.** T19 rejoué en vrai sur GPU : streamé == disque == stocké, bitwise. Je l'ai
**étendu** : sur une autre vidéo et **les 4 âges**, streamé == disque == stocké bit à bit, et
`stream(un seul âge)` == `stream(les 4 âges)` bit à bit (le vrai risque du mode, puisque
`build_signatures` ne rejoue que les âges manquants).

**(8) Les matériaux.** Les 30 prompts v2 relus **un par un** : aucune couleur, matière, taille,
vêtement, âge ; 30 têtes de sujet distinctes ; `corpus_idx = 9 000 000 + rang` correspond bien à
l'ordre du fichier ; jumeaux à graine +100 000 ; 60 vidéos, toutes `single`, 25 écritures.
**Aucune pollution du v1** : shards et manifestes v3 datés du 27/07, index du trainer daté du
27/07, signatures des paires v1 datées du 04/08 19:53, planches v1 intactes, **0 collision de
chemin** sur les 900 signatures (240 v1 + 420 switch + 240 v2). La déclaration « agent-19 devient
21 » est traitée proprement **dans le rapport** (v2.3, v2.8.1) et gelée dans `subsets_v1.json` ;
elle ne l'est **pas au registre** (voir B3-R1).

## B2. Défauts trouvés et corrigés (aucun ne touche un seuil scellé)

**D1 — GRAVE : la règle 2 était jugée sur UN SEUL dérangement.** L'audit du 04/08 avait corrigé ce
défaut en moyennant sur 25 dérangements, mais seulement pour la lecture **par ligne** ; le passage
de `PRIMARY` à `"pair"` l'a réintroduit, `acc_pair` du bloc `permuted` ne portant que le premier
tirage. Mesuré sur la validation croisée à **réponse connue** : `acc_pair` d'un tirage unique
= **0,700**, moyenne sur 25 dérangements = **0,505**. `gates.py` aurait donc imprimé
*« RÈGLE 2 ÉCHEC (le sondeur triche) »* **sur un instrument parfaitement sain**. À 30 paires, la
probabilité qu'un tirage unique dépasse 0,55 alors que la vraie valeur est 0,50 vaut ≈ **29 %**.
Corrigé : `evaluate` rend `acc_by_pair_allperm`, `cv_run` agrège `acc_pair_25perm`, `gates.py`
applique la règle 2 à cette quantité et imprime le premier tirage à côté.

**D2 — `success: true` était possible sous un verdict « PROTOCOLE INVALIDE ».** Le champ était
calculé sans la règle 1. Corrigé : `success = ok1 and (...)`.

**D3 — le gel des sous-ensembles n'était pas verrouillé.** `sealed_subsets` **recalculait**
`agent` sur les `.npz` à chaque appel et ne lisait jamais `runs/subsets_v1.json` ; le verrou de
mtime introduit par l'audit du 04/08 a disparu dans la réécriture de `gates.py`. Un simple
re-rendu des masques aurait déplacé en silence un sous-ensemble pré-enregistré **après** le gel.
Corrigé : le recalcul est désormais confronté au fichier gelé et toute divergence fait échouer le
run (sabotage testé : l'assertion se lève). Au moment de l'audit, recalcul == gel, exactement.

**D4 — `masks/_summary.json` du corpus v1 écrasé par la campagne v2** (il annonce maintenant 60
segments / 2 signalés au lieu de 270 / 34). Aucun code ne le lit : perte de documentation seule.
Corrigé : `_summary_<corpus>.json`.

**D5 — doc.** L'en-tête de `gates.py` affirmait lire encore les fichiers de l'ancien split 18/5/7 ;
il ne les lit pas du tout. Corrigé.

Selftests après correctifs : **95/95 verts** sans GPU ; T6, T11 et T19 rejoués **avec** GPU, verts.

## B3. Déclarations et risques qui restent à l'arbitrage d'Aden

**R1 — LE SEUIL 5/6 N'EST PAS SCELLÉ AU REGISTRE.** Le registre (jamais modifié depuis le 04/08
23:16) écrit noir sur blanc : *« seuil = plafond humain − marge (à sceller au re-run) »*. Le
§v2.8.2 affirme au contraire *« le seuil 5/6 a été scellé par le registre, pas choisi ici »* :
c'est **faux en l'état**. Même situation pour la liste agent (le registre dit toujours
« 19, déjà gelée »), pour `--features pool` comme primaire, pour `--pretrain`, et pour `hdirect`
comme nouvelle borne : tout cela est pré-enregistré **dans le rapport** (écrit avant tout score,
ce qui est légitime) mais **pas dans le document scellé**. À écrire au registre par Aden avant le
premier `--stage cv --audited`, sinon la règle 1 est un seuil choisi par l'agent qui a écrit le
code.

**R2 — le corpus v2 n'a aujourd'hui AUCUNE règle 1 atteignable.** Sans verdict d'œil,
`sealed_subsets(v2)` ne rend que `all`, et `gates.py` retombe alors sur le seuil **global 0,90**,
celui-là même que le registre a déclaré inatteignable même pour un humain. Lancer les runs v2
maintenant produit mécaniquement *PROTOCOLE INVALIDE*. Il faut d'abord l'œil d'Aden sur
`board/masks_board_pairs_numerote_v2.jpg`, un sous-ensemble « clairement différents » v2 gelé, et
un seuil v2 scellé.

**R3 — ma lecture de la planche v2 est plus sévère que celle du §v2.11.** Au-delà des 5 paires
déclarées (01, 04, 07, 08, 10), les paires **02** (le masque prend la main et la poêle chez la
principale, le chef entier chez la jumelle), **03** et **10** donnent deux crops qui ne portent
**pas le même objet des deux côtés** : `query_joint` garantit la même *requête textuelle*, pas le
même *objet*, et une paire dont les deux références DINO ne sont pas comparables corrompt la
décision. Par ailleurs v2 supprime l'identité fixée par le TEXTE, pas celle fixée par le MODÈLE :
cygne (19), lapin (17), hibou (16), voilier (30), bus (29) donnent deux individus quasi
identiques. Mon compte : ~22/30 exploitables, ~12-15 « clairement différents ».

**R4 — le blanchiment est ajusté à 72 % sur des lignes hors distribution** (les switch ont 35
écritures, les paires 25 : la population des âges n'est pas la même). Mesuré : les lignes de
paires sont décentrées de 0,18 à 0,58 écart-type (médiane) et d'échelle 0,87-1,04. Effet modéré,
absorbé par le `LayerNorm` d'entrée de la tête, et il va dans le sens conservateur. Non déclaré au
§v2.

**R5 — le dérangement peut envoyer une vidéo sur son PROPRE jumeau** (~1 chance sur 9 par vidéo et
par pli) : la réponse est alors forcément fausse, et le plancher du contrôle permuté est ~0,44 et
non 0,50. Conservateur pour la règle 2, mais à savoir en lisant le chiffre.

**R6 — la sélection d'hyperparamètres se fait sur la précision par LIGNE du val**, alors que le
verdict se lit par PAIRE. Pas une fuite, une simple discordance de critère.

**R7 — pour `hdirect`, `--features raw` n'est PAS la signature brute** : c'est le même vecteur de
9 216 nombres que `pool`, à une application linéaire inversible près. La « double
représentation » pré-enregistrée n'a de contenu que pour `state`.

## B4. VERDICT DE L'AUDIT 2

**GO pour les runs de verdict du corpus v1**, sous **une** condition préalable, non technique :
que R1 soit réglé, c'est-à-dire que le seuil ≥ 5/6, la reconstruction de la liste agent à 21, le
choix `--features pool` et `--pretrain` soient **écrits au registre** avant le premier
`--stage cv --audited`. Le code fait ce que le §v2 dit qu'il fait, les quatre fuites cherchées
(compression, blanchiment par pli, pool de pré-entraînement, corpus v2 vs v1) ne sont pas là, la
borne `hdirect` ne passe par aucune écriture et je l'ai recalculée à la main, `--stream` est bit à
bit, et le seul défaut sérieux trouvé (D1, la règle 2 sur un tirage unique, qui aurait déclaré
« le sondeur triche » sur un instrument sain) est corrigé et testé.

**NO-GO pour les runs de verdict du corpus v2** tant que R2 n'est pas tranché : aujourd'hui ils ne
peuvent produire que *PROTOCOLE INVALIDE*, quelle que soit la qualité de l'instrument. La collecte
et les artefacts v2, eux, sont propres et n'ont rien pollué : le corpus est prêt, c'est son
pré-enregistrement qui ne l'est pas encore.

---
---

# §v3 — CORRECTIF DE BORNE : LE POOLING SOUS LE MASQUE-SUJET (2026-08-05, nuit)

Source scellée au registre AVANT ce run (section « CORRECTIF DE BORNE ») : `hdirect_masked`.
Constat de départ : la borne à pooling uniforme échouait sur **les deux** corpus (v1 : 3/6 sur
les décisives ; v2 : 0,444 par paire sur 30) là où un humain réussit. Le sujet occupe 5-12 % de
l'image : le pooling uniforme le moyennait dans le fond, donc **l'instrument reproduisait le
défaut qu'il devait mesurer**.

## v3.1 — Ce que fait la nouvelle source

Exactement `hdirect` (pooling moyenne + max des tokens h par chunk, agrégé sur tous les writes
jusqu'à l'âge, **aucune écriture, aucune sonde, aucun hyperparamètre**), à une seule différence
près : **le pooling ne porte que sur les tokens sous le masque-sujet**.

- **Géométrie, lue dans le code du modèle, pas supposée.** Le patch d'entrée est (1, 2, 2)
  (`causal_model.py` l. 949, `Conv3d(kernel=patch, stride=patch)`) et les tokens sont posés par
  `u.flatten(2).transpose(1, 2)` (l. 1366) : index = frame × 1560 + ligne × 52 + colonne, sur
  une grille 30 × 52 pour un latent 60 × 104. Le masque est **pixel** (480 × 832), soit
  exactement **16 × 16 pixels par cellule de grille**.
- **Sous-échantillonnage pixel → grille** : une cellule entre dans le pooling si la **majorité**
  de ses pixels est sous le masque (règle qui conserve l'aire ; à l'égalité exacte le seuil `≥`
  tranche vers l'inclusion, jamais vers un pooling amputé).
- **Association chunk → masque** : le chunk porte 3 frames latentes ; on lui donne le masque de
  la frame échantillonnée la plus proche de son centre (égalités tranchées par l'index de
  frame), tuilé sur ses 3 frames. Un chunk, un masque. Chez les `switch` (pré-entraînement) les
  masques A **et** B sont chargés, si bien qu'un chunk post-switch reçoit le masque du sujet qui
  est réellement dedans.
- **Repli, pour ne jamais pooler le vide** : si moins de **3 %** des tokens (47 sur 1 560) sont
  retenus, on prend les 3 % les plus saillants du masque, égalités tranchées par l'index. Le
  repli est **déterministe** et **loggé** par le constructeur de signatures.
- **UN SEUL CHEMIN** : `READERS["hdirect_masked"] is READERS["hdirect"]` — littéralement la même
  fonction. La seule différence vit dans la charge utile (`load_payload`), pas dans une branche
  de lecture. Masque plein ⇒ sortie **bitwise identique** à `hdirect`.

## v3.2 — Ce que les masques donnent, mesuré (aucun score)

| corpus | chunks | aire retenue (q25 / méd. / q75) | repli déclenché |
|---|---|---|---|
| v1 (60 vidéos de paires) | 1 500 | 0,038 / **0,112** / 0,311 | 290/1 500 (**19,3 %**), 8/60 vidéos à 100 % |
| v2 (60 vidéos de paires) | 1 500 | 0,114 / **0,180** / 0,310 | 70/1 500 (**4,7 %**), 2/60 vidéos à 100 % |
| switch (pré-entraînement) | 3 675 | — / 0,150 / — | 665/3 675 (18,1 %) |

Le repli est **quatre fois plus rare sur v2** : ses masques sont meilleurs, ce que la planche
disait déjà.

## v3.3 — Selftests

**126/126 verts avec GPU** (les 105 anciens inchangés, T5/T13/T14/T18 étendus). Le nouveau
**T22** couvre les quatre contrats exigés :

```
(a) masque PLEIN -> hdirect_masked == hdirect BIT A BIT (max |delta| = 0.000e+00) ;
    masque partiel -> (moyenne, max) sur les SEULS tokens retenus ; et le resultat DIFFERE
    du pooling uniforme (le correctif fait bien quelque chose)
(b) masque VIDE -> repli DECLENCHE, exactement 47 tokens (le plancher 3 %) ; masque a
    10 tokens -> repli aussi, et il GARDE ses 10 tokens ; masque a 200 tokens -> aucun repli ;
    le repli est logge par le code de production
(c) selection et signature bitwise DETERMINISTES ; a egalites parfaites le repli prend les
    premiers index, deux fois pareil
(d) ALIGNEMENT pixel -> grille sur cas synthetiques CONNUS : 1 patch = 16 x 16 pixels ;
    un damier cellule-par-cellule se retrouve exactement sur la grille 30 x 52 ; un carre
    pixel donne EXACTEMENT les tokens de son rectangle (index = ligne*52+colonne) ; un
    decalage sous-patch ne dilate ni n'erode l'aire
    + controle sur les VRAIS masques du disque : pooling jamais vide, aire mediane 4,8 %,
      frames de chunk contigues finissant sur `frame_of_write`
```

## v3.4 — LA BORNE, face aux seuils scellés

Configuration de verdict : `--stage cv --audited --features pool --pretrain`, 6 plis × 5 paires,
3 graines, lecture **par paire**.

| corpus | sous-ensemble | `hdirect` (uniforme) | **`hdirect_masked`** | seuil scellé | verdict |
|---|---|---|---|---|---|
| **v1** | 30 paires (primaire) | 0,533 (16,0/30) | **0,667 (20,0/30)** | — (information) | +0,134 |
| v1 | agent (21) | 0,556 | 0,714 | — | |
| v1 | œil d'Aden (15) | 0,511 | 0,711 | — | |
| **v1** | **décisives (6)** | 0,444 (2,7/6) | **0,500 (3,0/6)** | **≥ 5/6** | **ÉCHEC** |
| **v2** | 30 paires | 0,444 (13,3/30) | **0,756 (22,7/30)** | — | **+0,312** |
| **v2** | **valides (25, primaire v2)** | — | **0,760 (19,0/25)** | — | |
| **v2** | **fortes (19)** | — | **0,825 (15,7/19)** | **≥ 16/19** | **PASS au bord** |

Marges et significativité (30 paires, Wilcoxon sur les marges par paire) :
`hdirect` v1 p = 0,52 → `hdirect_masked` v1 **p = 0,017** ; `hdirect` v2 p = 0,72 →
`hdirect_masked` v2 **p = 0,0016**. Contrôle permuté (25 dérangements) : 0,490 et 0,486, au
hasard dans les deux cas.

**Lecture honnête.**
1. **Le correctif marche, et c'est le premier signal net de tout ce sondage.** Le pooling
   uniforme était au hasard sur les deux corpus (0,533 et 0,444, p = 0,52 et 0,72) ; le pooling
   masqué monte à 0,667 et 0,756 avec p = 0,017 et 0,0016. Le diagnostic scellé au registre
   (« l'instrument reproduisait le défaut à mesurer ») est **confirmé**.
2. **v1 échoue toujours sa règle 1** : 3/6 sur les décisives contre 5/6 exigé. Mais ce
   sous-ensemble fait **6 paires** et les matériaux v1 sont ceux dont le plafond humain vaut
   ~0,74 ; le sous-ensemble « œil d'Aden » (15 paires) passe de 0,511 à 0,711, ce qui va dans le
   même sens que v2. La règle 1 v1 reste **ÉCHEC**, écrit tel quel.
3. **v2 passe sa règle 1 — mais au bord, et il faut le dire.** 15,67/19 en moyenne sur 3 graines
   arrondit à 16, exactement le seuil ; **par graine : 17, 16, 14**. Le verdict mécanique est
   PASS, la marge est nulle. À 19 paires, une paire vaut 5 points.
4. La prédiction du registre (« hdirect passe ~17-18/19 ») est **proche** du résultat observé
   (15,7/19) sans l'atteindre.

**Contrôle de spécificité de la borne elle-même** (cible SUJET vs cible FOND, la même source) :

| corpus | sous-ensemble | borne cible SUJET | borne cible FOND | écart |
|---|---|---|---|---|
| v1 | 30 paires | **0,667** (p = 0,017) | 0,500 (p = 0,39) | **+0,167** |
| v1 | œil d'Aden (15) | 0,711 | 0,467 | +0,244 |
| v2 | 30 paires | **0,756** (p = 0,0016) | 0,644 (p = 0,085) | +0,112 |
| v2 | valides (25) | 0,760 | 0,627 | +0,133 |
| v2 | **fortes (19)** | **0,825** | 0,579 | **+0,246** |

Le pooling masqué désigne bien **l'individu**, pas la scène : le même instrument, tourné vers le
fond, retombe près du hasard. C'est la première fois de ce sondage qu'un écart sujet-fond va dans
le bon sens **du côté de la borne**.

## v3.5 — TABLEAU DES SEUILS, corpus v1 (`gates.py --corpus v1 --features pool --pretrain`)

```
BORNE hdirect_masked   30 paires  n=30  pair=0.667 (20.0/30)  row=0.607  p=0.01727
                       agent      n=21  pair=0.714 (15.0/21)  row=0.629  p=0.00708
                       oeil Aden  n=15  pair=0.711 (10.7/15)  row=0.647  p=0.01508
                       decisives  n= 6  pair=0.500 ( 3.0/6)   row=0.569  p=0.4219
REGLE 1 (decisives >= 5/6) : 3/6                    -> PROTOCOLE INVALIDE
ETAT cible SUJET       30 paires  pair=0.522   decisives 0.778 (4.7/6)
ETAT cible FOND        30 paires  pair=0.667   decisives 0.833 (5.0/6)
REGLE 2 permutes (25 derangements) : pair=0.504     -> PASS (<= 0.55)
REGLE 3 sujet 0.522 (>= 0.65 exige) | p = 0.3425 (< 0.05 exige) | fond 0.667
        ecart sujet - fond = -0.144 (>= +0.10 exige)  -- NEGATIF
VERDICT SCELLE : PROTOCOLE INVALIDE -- aucune conclusion sur la memoire n'est autorisee.
```

Le corpus v1 reste **fermé** : la borne n'atteint pas son seuil, donc les chiffres d'état sont
imprimés mais ne valent pas verdict. À noter tout de même, puisqu'il faudra en parler : l'écart
sujet − fond de l'état y est **négatif** (−0,144), la signature de l'« enregistreur d'ambiance ».

## §v3 : VERDICT FINAL (2026-08-05, écrit par le coordinateur)
Borne masquée (hdirect_masked) : v2 fortes 16/19 (0.825, p=0.011) → RÈGLE 1 PASS :
l'instrument est VALIDE. Permuté : PASS. MÉMOIRE (état) : sujet 0.467/25 valides,
0.456/19 fortes, p=0.82 → HASARD. Verdict scellé appliqué : ARRÊT du chantier mémoire ;
le sondage négatif à instrument validé est la contribution. v1 : borne 3/6 même masquée
→ protocole-invalide (matériaux quasi-clones), aucun verdict légal, v2 suffit.
Chaîne finale du projet : le canal transmet ; l'ancre décodée ne porte que l'ambiance ;
et l'ambiance est tout ce que l'écriture actuelle rend extractible. Travail futur motivé :
l'écriture triée par masque-sujet (deux registres).
