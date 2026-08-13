# LES 5 ATTAQUES REVIEWER LES PLUS PROBABLES, ET CE QUI LES PARE

Format : l'attaque telle qu'elle sera écrite dans la review → ce qu'on a DÉJÀ → ce qui MANQUE
(expérience à lancer, coût) → la phrase de rebuttal. Classées par dégât potentiel.

## Attaque 1 (LA plus dangereuse) : "vous ne montrez pas que c'est de la MÉMOIRE"

> "Your +0.434 is not recall. You overwrite the recency slot with a low-pass image, which erases
> the interlude subject; the text prompt then repaints A. Your own noise-anchor control shows 54 %
> of a composite gain survives with a pure-noise anchor, and your instance test is negative. A
> noise splice would probably score the same."

**Ce qu'on a déjà**
- Le contrôle bruit sur le tier **cross-scène** (S2, le tier du résultat principal) rend
  **+0.0084 = 9 %** du gain, et le masque uniforme **+0.0018 = 2 %**, contre 118 % pour le
  composite complet. C'est exactement la réponse, mais à **n = 3 cellules, seed 0, et à la dose
  0.288**, pas à la dose 1.0 du bras principal. (`composite_gates_report.md` §5.2)
- **P1, ablation causale** : neutraliser les écritures du segment A dégrade l'ancre décodée de
  **+0.429 de loss, 12/12 vidéos**, contre une ligne rouge pré-enregistrée à 0.02 (ratée ×21) ; le
  contrôle "h d'une autre vidéo" dégrade moins (+0.184), donc l'écrivain dépend du contenu réel.
- **Cheval 3-bras** : l'ancre n'est pâle qu'avec l'écrivain entraîné (décodeur v1.2 et contrôle :
  brunes) ⇒ l'instance entre dans l'état.
- **h-shuffle du toy** : rejouer la séquence h d'une autre vidéo laisse la loss au niveau du
  prédicteur nul (0.9395 vs 0.4704) ⇒ l'état porte de l'information spécifique à SA vidéo.
- Le contre-exemple qui prouve les limites du "repaint" : **le scooter bleu ne revient jamais**
  quand on efface sans ancre (objet rigide non repeignable depuis le texte).

**CE QUI MANQUE (à lancer en priorité absolue)**
1. **Bras `memnoise20`** : splice à la récence, dose 1.0, d'une ancre de **bruit pur** (σ calibré,
   RMS préservé) sur **S2, 24 cellules**. Prédiction à pré-enregistrer AVANT : si le bruit rend
   > 50 % du +0.434, le claim principal tombe ; on attend < 20 % au vu du tier cross-scène.
2. **Bras `memdec20twin`** : splice de l'ancre décodée d'une **AUTRE cellule** (même catégorie de
   prompt, autre instance), S2, 24 cellules. Sépare "le décodage de CET état" de "une frame
   plausible quelconque". (Le hook existe déjà : `ORACLE_ANCHOR_DUMP` / `ORACLE_ANCHOR_SRC`,
   `memory2_probes_report.md`.)
3. Optionnel mais peu cher : **`memdec20` avec cond FiLM d'un autre prompt** (le décodeur lit
   l'état mais le conditionnement ment) : sépare la contribution état / prompt dans le décodeur.
Coût estimé : 24 cellules × 2-3 bras ≈ 48-72 rollouts de 240 frames ≈ **12-20 GPU-h**, offline
scoring gratuit (les A\* sont déjà sur disque).

**Rebuttal** : "Erasure alone is measurable and we measured it: on the cross-scene tier a pure-noise
anchor recovers 9 % and a shape-free mask 2 %, while ablating the writes of segment A degrades the
decoded anchor 12/12 (Δloss +0.429 against a pre-registered red line of 0.02). We also report where
erasure DOES pay (same-background tier, 54 %) instead of hiding it."

## Attaque 2 : "un seul substrat, un seul modèle"

> "Everything is measured on a single 1.3B checkpoint. Nothing shows this is a property of
> streaming video models rather than of your LoRA."

**Ce qu'on a déjà**
- Le phénomène attaqué est **structurel, pas propre à notre checkpoint** : le sink qui épingle la
  frame 0 est mesuré sur **LongLive officiel non modifié** (`stockoff`), qui est **négatif** sur
  S2 (−0.096, 7+/17−) alors qu'il est +0.16 à +0.25 quand A ouvre. La "chance positionnelle" est
  démontrée sur le modèle publié, pas sur le nôtre.
- La baseline Echo-Forcing est elle-même annoncée training-free et portable ; son échec sur S2
  est un échec de **substrat de stockage** (du rendu), pas d'implémentation.
- Le backbone est **strictement gelé** et prouvé byte-identique à l'upstream (audit de lignée
  déjà fait : `reference/RESEARCH_STATUS.md`, ckpts sha256-match HF).

**CE QUI MANQUE**
- Un **transfert minimal**, par ordre de coût croissant : (a) `stockoff` + `framebank` + `oracle1`
  sur S2 avec un **autre générateur streaming** (Self-Forcing ou Wan-1.3B + CausVid) : cela
  suffit à montrer que le RÉGIME généralise, même sans y porter notre mémoire (≈ 1 jour de dev,
  6-10 GPU-h) ; (b) porter l'écrivain + décodeur (coûteux : nouvelle collecte h, ~2 semaines) :
  **à ne pas tenter avant la deadline**.
- Recommandation : faire (a), l'annoncer comme "the failure mode generalizes; our fix is
  demonstrated on one substrate", et mettre (b) en future work explicite.

**Rebuttal** : "The failure we attack is a property of window+sink streaming decoding, and we show
it on the unmodified published model. Our remedy is demonstrated on one substrate; we say so."

## Attaque 3 : "24 cellules, et 3 pour la moitié de vos figures"

> "n = 24 paired videos, and the composite study is n = 3 with a single seed. Effect sizes of
> ±0.03 are reported as results."

**Ce qu'on a déjà**
- Test **des signes exact** (non paramétrique) et appariement strict, pas de t-test abusif.
  Le chiffre-titre est **24+/0−, p = 1.2e-7** : à 24 cellules c'est le maximum d'information que
  le test peut donner, et l'effet (+0.434, sd 0.233) est **d ≈ 1.9**.
- **Plancher de bruit mesuré** (0.0008 sur le sujet), donc les effets sont 2 ordres de grandeur
  au-dessus du bruit de reproductibilité.
- La distribution des deltas est **décrite** (mélange à queue : 7 cellules ~0, 12 à 0.03-0.18,
  5 jackpots ≥ 0.35) : on ne vend pas une moyenne comme un effet uniforme.
- Les rapports internes distinguent déjà explicitement les effets "un ordre de grandeur au-dessus
  du bruit" des effets fins "qui ne doivent pas être annoncés comme acquis"
  (`composite_gates_report.md` §8.5).

**CE QUI MANQUE**
1. **2 seeds de plus sur S2/S2p** → 48 à 72 cellules sur le chiffre-titre. ~24-36 rollouts ×
   4 bras : **20-30 GPU-h**, à faire tourner pendant la rédaction.
2. **Décision D3** : ne mettre AUCUN chiffre n=3 dans la table principale ; l'étude composite
   devient une section "operating points" explicitement étiquetée `n = 3, seed 0, exploratory`.
3. Retirer du papier tout delta ±0.03 présenté comme un résultat (les garder comme "non
   significatif" avec leur p).

**Rebuttal** : "Our headline is a sign test at 24/24 (p = 1.2e-7) with a measured reproducibility
floor two orders of magnitude below the effect. Everything below that bar is reported with its
sign count and its p-value, and the exploratory operating-point study is labeled n = 3."

## Attaque 4 : "l'instance n'est pas résolue, donc votre mémoire ne mémorise pas"

> "You admit the returned subject is no closer to its own A than to another seed's A. Then what
> is stored? A prompt-conditioned category prior would produce the same numbers."

**Ce qu'on a déjà**
- La réponse est **quantitative et pré-existante** : le canal plafonne l'instance **quel que soit
  le contenu écrit**. Une frame **parfaite** écrite puis relue immédiatement (`thruwriter`) vaut
  **+0.131**, contre +0.126 pour notre système complet et +0.224 pour l'oracle. Donc ce n'est pas
  "notre mémoire ne stocke rien", c'est "le canal écriture→lecture ne restitue pas les détails
  fins" : deux affirmations très différentes, et la seconde est mesurée.
- Le **diagnostic de rendu** renforce : au niveau de fidélité atteint, **même la vérité terrain
  floutée échoue** au gate d'instance DINO (gap −0.127 contre −0.190 pour le décodé) ⇒ le
  bottleneck est le rendu, pas la sélection du contenu.
- L'information EST là en amont : **h-oracle val 0.0004** (contre 0.812-0.867 pour les décodeurs)
  ⇒ le goulot est l'ÉTAT et sa lecture, pas la donnée.
- Et le test négatif est **calibré** : la métrique résout l'instance à +0.1335 (12/12), donc c'est
  un vrai négatif, pas un manque de puissance. Le dire nous-mêmes désamorce l'attaque.
- Contre-argument au "category prior" : si c'était un prior de catégorie conditionné au prompt,
  **P1 ne dégraderait pas 12/12** et le **h-shuffle** ne renverrait pas la loss au niveau du
  prédicteur nul.

**CE QUI MANQUE**
- Le contrôle "prompt seul" de l'attaque 1 (bras `memnoise20`) sert **aussi** ici : c'est
  littéralement l'implémentation du "category prior" du reviewer. Même run, double usage.
- Facultatif, gratuit : reporter la **distribution** du delta d'instance par cellule (déjà dans le
  JSON) pour montrer qu'aucune cellule n'atteint la marge de calibration.

**Rebuttal** : "We do not claim instance recall; we bound it. The ceiling is the write→read channel,
not the stored content: a perfect frame written and immediately read back is worth +0.131 versus
our +0.126 and an oracle's +0.224. And the information is upstream (h-oracle 0.0004), which is why
we name the state→frame decoder as the next bottleneck rather than the objective."

## Attaque 5 : "Echo-Forcing vous bat / votre baseline est un homme de paille"

> "On four of your six scenarios the frame bank matches or beats you, and by your own admission it
> IS the oracle in verbatim. You also did not implement their KV compression, their relative RoPE,
> or their decay: you re-implemented a weaker method and called it Echo-Forcing."

**Ce qu'on a déjà**
- **On le dit avant eux** : la banque gagne le terrain ouvrant (+0.20 à +0.30 contre nos +0.11 à
  +0.13) et en verbatim elle **atteint la borne oracle** (prouvé bitwise `framebank == oracle1`
  sur A→B→A). Le papier ne revendique pas ce terrain (décision D1/D3).
- Le port est **doublé de ses plafonds** : `framebank_oracle` (sélection vérité-terrain) et
  `oracle1` (vraie frame parfaite). **Aucun verdict ne dépend de la qualité du port** : sur S2,
  `oracle1` lui-même est nul (+0.027, p = 0.84), donc même une implémentation PARFAITE d'Echo,
  ou n'importe quelle méthode qui splice du rendu réel, échoue. C'est l'argument imparable, et il
  doit être écrit noir sur blanc dans le papier, pas seulement en rebuttal.
- Les différences non portées (fusion spatiale multi-blocs, RoPE relatif, decay différentiel) sont
  des **optimisations de fidélité de scène**, pas le mécanisme de rappel : à justifier
  explicitement en annexe, avec la citation des sections du papier d'origine.
- Coûts structurels de la banque, mesurés : mémoire **O(#scènes)** contre notre **O(1)** ; routage
  par cosinus de prompt qui **rate 17 %** des rappels en paraphrase (83.3 % de rappel, 6 faux
  rappels) et pour lequel **aucun seuil unique** ne sépare paraphrase (0.665) de distracteur
  (0.7218).

**CE QUI MANQUE**
- Une phrase de cadrage dans l'abstract et l'intro : *"we do not claim to beat frame/KV banks in
  the regime they were designed for; we identify the regime where they are provably null"*.
- `memdec20r` (notre bras routé par le MÊME routeur que la banque) sur la matrice : supprime la
  dernière asymétrie (notre bras reçoit du planning un signal oracle "ceci est un retour").
  **Déjà codé et selftesté** (routage identique à `framebank` sur 4 formes), ~11 GPU-h.
- Idéalement : un point de composabilité (banque **+** notre ancre) pour montrer que ce n'est pas
  un remplacement mais un complément. À ne faire que si le temps le permet.

**Rebuttal** : "Our port is bracketed by two upper bounds we also run: ground-truth routing and the
real past frame. On the non-opening tier the real past frame itself is null (+0.027, p = 0.84), so
no implementation of a render-storing memory can pass this tier. The comparison is therefore about
the storage substrate, not about our port's fidelity."

## Attaques secondaires (à préparer, une phrase chacune)

- **"Vos scénarios sont synthétiques."** 12 triples couvrant véhicule / personne / animal / objet
  rigide, 2 seeds, **48 prompts vérifiés absents du corpus d'entraînement** (selftest, sous-chaîne
  dans les deux sens) ; les formes S1/S3/S4 sont la généralisation directe du protocole "scene
  recall" d'Echo-Forcing (plans 4-6 vs plans 1-3), et S2 n'est pas un piège mais **le cas
  générique** (un sujet qui n'ouvre pas la vidéo). MANQUE : au moins **un second scénario
  non-ouvrant** (ex. `X → A → B → C → A`) pour prouver que le résultat n'est pas lié à une forme
  particulière (~12-15 GPU-h), et une session longue non scriptée en supplementary.
- **"Pourquoi ne pas co-entraîner (VideoSSM/MemGen) ?"** Contrainte assumée (backbone gelé,
  attribution causale possible, coût) + le write-replay est exactement ce qui rend l'entraînement
  possible sous cette contrainte. On ne prétend pas battre une co-adaptation e2e ; on ne la
  compare pas non plus, et il faut le dire au lieu de laisser croire à une comparaison.
- **"crop60 est un rayon choisi."** Ordonnancement **stable aux 4 rayons** (0.5 / 0.6 / 0.7 /
  frame pleine), chiffres publiés en annexe.
- **"Vos rollouts ne sont pas reproductibles."** Vrai, mesuré, et publié : divergence
  flash-attn/cuBLAS dès la frame latente ~3, **avant tout hook et sur un bras sans hook** ; le
  plancher de bruit sur les métriques est **0.0008**, deux ordres de grandeur sous les effets.
- **"Vous avez choisi la variante composite à l'œil."** Assumé : la famille de remplissage est
  **sous-déterminée à n = 3**, on gèle la variante la plus simple (`memdec21u`) et on publie les
  trois autres avec leurs défauts durs (buisson halluciné, animal dupliqué, sujet absent).

## Rappel de posture pour le rebuttal

Le dossier a une force rare : **trois prédictions pré-enregistrées ont été réfutées et sont
publiées** (P1 "le mal était l'éviction", P4b "l'ancre bruitée doit s'effondrer", P4c "masque
inversé"), un **verdict interne a été inversé par nos propres contrôles** (−0.378 → +0.434), un
**faux diagnostic a été refusé** au lieu d'être exécuté, et un **gate a bloqué une phase entière**
(le terme d'instance DINO, B2 non lancé). C'est l'argument à mettre en avant quand un reviewer
soupçonne du cherry-picking : le papier montre ses propres échecs de prédiction avec leurs dates.
