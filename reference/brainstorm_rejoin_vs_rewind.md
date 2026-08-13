# Brainstorm : RESSUSCITER vs REMBOBINER (rejoin vs rewind)

Rédigé le 2026-07-24, lecture seule (plan_final_sink_virtuel.md, localplan_D_pivot.md,
pipeline/switch_causal_inference.py l.56-128 et l.226-299, run_reid_probe.py + summary.json,
model/streaming_training.py l.975-1044 pour la pile e4). Question posée par Aden : au retour
A->B->A dans la MÊME scène (robe rouge -> robe bleue -> robe rouge, la caméra continue), le
splice replaque une frame du passé ; risque de téléportation caméra/décor. oracle1 (+0.224,
24/24) prouve le canal identité, mais la cohérence de transition n'a JAMAIS été mesurée, et
reid_resA_origA récompenserait un rembobinage brutal.

## 0. Cadrage : deux axes orthogonaux, un quadrant, deux angles morts

Le succès au retour a DEUX composantes indépendantes :
1. **identité restaurée** (le sujet A : visage, robe rouge d'origine), mesurée (reid_resA_origA) ;
2. **monde continué** (plage, angle caméra, lumière, mouvement en cours), JAMAIS mesurée.

Quadrant des issues : (id oui, monde oui) = **rejoin** (le but) ; (id oui, monde non) =
**rewind** (le risque du splice) ; (id non, monde oui) = perte d'identité (le statu quo
memzero) ; (id non, monde non) = collapse. Notre métrique actuelle ne voit que l'axe 1 : sur
les 8 paires cross-scène du harness, res-A DOIT ressembler à orig-A décor compris, donc un
rewind pur score au maximum. Deux conséquences :

- **Angle mort n°1 : le rewind n'est un échec QUE sur le tier même-scène.** Sur les paires
  actuelles (chien au parc / voiture en ville), revenir au monde de A est le comportement
  désiré. Le problème d'Aden n'existe pas dans le harness : il faut fabriquer des paires
  même-scène-attribut-switch pour le rendre mesurable.
- **Angle mort n°2 : sans mouvement caméra CONTINU et explicite dans le prompt, la mesure est
  nulle.** Si la caméra est quasi statique (nos DD ~0.67 le suggèrent), monde@retour ~
  monde@départ et rejoin/rewind sont indistinguables. Les paires du probe DOIVENT contenir
  "camera slowly panning/orbiting" pour garantir que l'état du monde a avancé pendant B.

Point stratégique : le rejoin n'est pas un bug à corriger, c'est LE claim. La baseline
frame-bank (plan_final §frame bank) replaque une frame figée = rewind par construction ;
AdaState replay un buffer. "Synthèse conditionnée au contexte courant" (différenciateur n°3
du plan_final) n'a de contenu empirique QUE si on mesure la transition. Ce brainstorm
opérationnalise le différenciateur.

## 1. Mécanismes candidats (classés simplicité x chance de marcher)

### M1. Statu quo v1 : insertion minimale 1-slot, position P1 (dernier slot)
Ce qui joue POUR : 11/12 frames du recache sont le présent ; l'attention du recache est
PLEINE (localplan_D §1.1) donc les k/v du slot d'ancre absorbent aussi le contexte présent
(l'ancre est "contextualisée" au moment de l'encodage) ; en P1 le slot s'évince en ~3 chunks
(dose limitée dans le temps). C'est plausiblement pour ça qu'oracle1 marche si bien : c'est
déjà un rejoin partiel par dilution.
Mode de défaillance identifié : **la position est une revendication temporelle.** En P1
l'ancre est ropée côté récence = le modèle la lit comme "la frame d'il y a un instant" = une
vérité géométrique immédiate, exactement le régime qui tire vers le cut/rewind. Trois sorties
possibles jamais départagées : (a) rejoin doux (l'identité se recopie sur la géométrie
courante), (b) cut-rewind (la caméra saute à la pose de l'ancre), (c) ghosting/morph (double
exposition pendant quelques frames). Verdict : peut suffire, mais on ne le saura qu'avec les
métriques du §3 ; à mesurer AVANT de complexifier.

### M2. Position/pondération du slot : l'ancre au sink (P2) + atténuation k
P2 (frame décodée en TÊTE de frames_override, gratuit, localplan_D §1.2) rope l'ancre dans la
bande sink = revendication "passé lointain", sémantiquement correcte pour un souvenir.
Hypothèse forte : même re-ID, moins de traction rewind, parce que le backbone traite les
sinks comme référence d'apparence, pas comme prédécesseur géométrique. Le probe
oracle-position-sink déjà planifié départage P1/P2 : il suffit de le scorer AUSSI avec les
métriques de transition. Variante pondération : après le recache, multiplier k (et/ou v) du
slot d'ancre par alpha dans le cache (chirurgie post-recache ~10 lignes, même pattern que le
snapshot oracle3b l.268-277) = dose-réponse d'influence de l'ancre, un "gate manuel" à 0$.

### M3. Ancre à bruit partiel (évidence molle)
Splicer la frame de rappel bruitée à sigma > context_noise (bruit ajouté au SEUL slot d'ancre
avant le recache). L'identité (basses fréquences, sémantique) survit au bruit, la géométrie
exacte (la vérité pixel de l'ancien angle caméra) se dégrade en premier : l'ancre devient une
référence d'apparence, pas une contrainte de pose. C'est LITTÉRALEMENT le hook oracle-dose
déjà planifié (localplan_D §5 risque n°1), relu comme BOUTON de rejoin et plus seulement
comme courbe de tolérance du décodeur : chercher sigma* où re-ID tient ET le saut frontière
chute. Si sigma* existe, le mécanisme v1 est gratuit (bruiter la frame décodée au splice).
Double dividende : la même campagne répond au risque décodeur-imparfait ET au rejoin.

### M4. Décodage conditionné au contexte courant (le vrai v2)
Entrées de conditionnement, dans l'ordre de valeur : (1) les frames de la fenêtre courante
(cross-attention des 1560 queries du décodeur sur la queue B, c'est ce qui porte la géométrie
présente) ; (2) le prompt du segment retour (FiLM sur les queries, déjà prévu au plan
multi-switch pour l'adressage). Le prompt seul ne suffit pas : il ne contient pas l'angle
caméra courant. Le problème dur n'est pas l'architecture, c'est la CIBLE : il n'existe pas de
vérité terrain "sujet A dans le monde tel qu'il a continué". Réponses data-free au §4. Sans
cible dédiée, le décodeur conditionné apprendra à ignorer le contexte (la reconstruction de
l'évincé n'en a pas besoin) : conditionner sans superviser = poids morts.

### M5. Désenchevêtrement sujet/décor : compositing latent (rejoin par construction)
Le décodeur sort une frame + un masque doux ; frame splicée = masque*frame_A_décodée +
(1-masque)*dernière_frame_présente. La frame injectée est alors "le présent avec le sujet
échangé" : le rewind devient impossible par construction, le décor injecté EST le décor
courant. Les latents sont spatialement alignés (16x60x104) : un composite crop-centre
(région sujet ~ crop60 depuis A, bord depuis le présent) se fabrique en 5 lignes. Version v1
sans masque appris : masque fixe centre-60% ; v2 : masque prédit (tête sigmoïde du décodeur,
supervisée par la cible composite du §4). Risque : couture latente ; mitigé par le recache à
context_noise + l'attention pleine qui lisse. LE mécanisme à plus fort a priori de rejoin,
et il a une version oracle à 0$ (§2, probe n°3).

### M6. Non envisagés jusqu'ici
- **Two-pass propose-then-correct** : générer le chunk de retour SANS ancre (le monde
  continue), puis re-recacher fenêtre+ancre et régénérer le même chunk avec le même bruit
  (l'identité se corrige dans un contexte déjà rejoint). Coût 2x sur un chunk, inférence
  seule. En réserve si M2/M3/M5 échouent.
- **Contrôle par le texte** : prompt retour explicite "the same beach, the camera keeps
  panning, the woman in the red dress". Bras contrôle obligatoire : mesure ce que le canal
  texte donne gratuitement, et c'est la version "paraphrasée" du curriculum (plan_final
  §frame bank, revisits implicites).
- **Fade-in multi-chunk** : alpha croissant sur l'ancre sur 2-3 recaches successifs. Demande
  la persistance P3/hybride, pas v1.

Classement final (simplicité x chance) : **M2 (P2 sink) > M3 (bruit) > M1 (statu quo mesuré)
> M5 (composite) > M4 (conditionnement v2) > M6**. M2, M3, M5-oracle sont mesurables à 0$
cette semaine ; M4/M5-appris sont des décisions d'architecture à prendre APRÈS ces mesures.

## 2. Probes 0$ réalisables MAINTENANT (hooks existants)

### Probe 0 (gratuit, aucune génération) : instrumenter et rescorer l'existant
Les 24x plusieurs mp4 et les latents de la campagne l1 sont sur disque. Ajouter au scorer les
métriques du §3 et rescorer TOUTES les cellules existantes (memzero, oracle, oracle1,
oracle3b, stock) aux DEUX joints (switch1, switch2). But : calibrer le détecteur de cut (le
saut au switch1 = le saut "naturel" de changement de scène, la référence interne), et
vérifier une prédiction : oracle (12 frames de A) doit montrer un boundary-jump PLUS PETIT
qu'oracle1 au switch2 sur paires cross-scène (le monde rembobiné est COHÉRENT avec le
contexte splicé), tandis qu'oracle1 doit montrer le jump le plus proche de memzero. Si le
détecteur ne sépare même pas oracle de memzero, il est inutilisable, on le sait avant de
générer quoi que ce soit.

### Probe 1 (LE probe) : campagne SAMESCENE, arms position/dose
8 paires même-scène-attribut-switch, caméra mobile OBLIGATOIRE, attribut re-identifiable et
sous-spécifiable. Proposition :
```
A = "a woman in a long red dress walking along a sunny beach, camera slowly panning right"
B = "a woman in a long blue dress walking along a sunny beach, camera slowly panning right"
```
et 7 variantes du même moule : (chien, bandana rouge/bleu, parc, dolly avant), (homme, veste
jaune/noire, rue, travelling), (voilier, voile rouge/blanche, mer, orbite), (fille, ballon
rouge/vert, aire de jeux, pan), (cycliste, maillot orange/violet, route, follow), (chat,
collier rouge/bleu, salon, pan lent), (parasol rouge/bleu, plage, orbite). Retour = prompt A
verbatim (le tier paraphrasé viendra après). Mêmes réglages que l1 : 240 frames, sw 81/162,
3 seeds, seed apparié par cellule.

Arms (tous sur hooks existants ou chirurgie <20 lignes) :
- `memzero` : baseline. Prédiction : l'ATTRIBUT (robe rouge) revient par le texte, le VISAGE
  et les détails non-textuels dérivent ; le monde continue trivialement (pas de splice). Fixe
  le plancher de transition et le plafond "texte gratuit".
- `oracle` (12f de A) : rewind maximal assumé = contrôle positif du détecteur de téléportation.
- `oracle1` (P1, dernier slot) : le statu quo v1. LE chiffre que le design attend.
- `oracle1sink` (P2) : réordonner frames_override, cat([A_frame, B_tail 11f]) au lieu de
  cat([B_tail 11f, A_frame]). ~5 lignes dans le hook ORACLE_SPLICE (une env var
  ORACLE_SPLICE_POS=sink). C'est le probe oracle-position-sink du plan_final, scoré transition.
- `oracle1-dose-sigma{0.2,0.4,0.6}` : bruiter le slot d'ancre (add_noise du scheduler au
  niveau sigma) avant recache. Fusionne avec la campagne oracle-dose déjà décidée : mêmes
  runs, un axe de lecture en plus.
- `oracle1-composite` : chimère latente, centre-60% de la frame A + bord de la dernière frame
  présente, splicée en dernier slot. ~10 lignes. Teste M5 sans décodeur.
- `stock` : la référence externe habituelle (3-way, protocole standard).

Lecture pré-enregistrée (à écrire AVANT de lancer, comme oracle3b) : sur le quadrant du §3,
(i) si oracle1 est déjà dans la case rejoin (id > memzero + 0.05, rejoin_index > 0, cut_score
< 2x switch1) : M1 suffit, le design v1 est validé tel quel, on arrête de sur-designer ;
(ii) si oracle1 rewind mais oracle1sink ou dose-sigma rejoint à re-ID comparable : adopter
P2/bruit comme défaut v1 (coût nul) ; (iii) si seul composite rejoint : M5 entre au design du
décodeur (tête masque) ; (iv) si TOUT rewind y compris composite : le backbone traite tout
splice comme un cut, le rejoin exige le conditionnement v2 (M4) et le tier curriculum dédié,
re-chiffrer. Coût : ~8 arms x 24 cellules, mêmes ~40 min/arm que les campagnes précédentes.

### Probe 2 (après premier ckpt décodeur) : rejouer Probe 1 avec la frame DÉCODÉE
Inchangé vs le plan (risque n°3 de localplan_D), mais scoré avec les métriques de transition :
la contamination A/B de l'état pourrait paradoxalement AIDER le rejoin (mélange = ancre plus
molle), à vérifier plutôt qu'à supposer.

## 3. Métriques de cohérence de transition (à intégrer au scorer du harness)

1. **cut_score (téléportation)** : saut lag-1 au joint, normalisé deux fois.
   `jump = mean|x[t*] - x[t*-1]|` sur latents (dispo via return_latents, gratuit) ET pixels ;
   `cut_score = jump / median(lag-1 intra-segment)` (l'échelle de mouvement propre de la
   vidéo) ; rapporter aussi `cut_score(sw2) / cut_score(sw1)` (le switch1 = saut naturel de
   référence : en même-scène il devrait être petit, bon contrôle de santé). Compléter par
   l'impulsion D2 au joint : la formule per-site de streaming_training.py l.1024-1030 se
   réutilise verbatim (un cut = impulsion d'une frame dans la dérivée seconde, le mouvement
   lisse y contribue ~0). Seuil indicatif littérature shot-boundary : cut si > ~3x.
2. **rejoin_index (monde continué vs rembobiné)** : DINO sur le COMPLÉMENT du crop60
   (l'anneau de bord = le décor, le sujet vit au centre par construction des prompts).
   `rejoin_index = cos(bord(resA_early), bord(B_tail)) - cos(bord(resA_early), bord(origA_late))`
   avec resA_early = ~8 frames pixel après sw2, B_tail = ~8 avant, origA_late = fin du
   segment A. Positif = le monde a continué ; négatif = rembobiné. N'a de sens que sur le
   tier même-scène à caméra mobile (sur cross-scène il mesure autre chose : le retour de
   scène, qui est désiré).
3. **id_restored (l'axe existant, restreint au sujet)** : reid_resA_origA en crop60 (déjà
   là). Le couple (id_restored, rejoin_index) donne le quadrant du §0 : publier les 4 cases
   par arm, jamais une moyenne unique, c'est précisément l'agrégation qui a caché le problème.
4. **Flux optique au joint (confirmation, pas défaut)** : Farneback OpenCV sur 3 frames
   autour du joint, magnitude vs médiane intra-segment. Plus cher, à ne sortir que si
   lag-1/D2 et l'oeil humain divergent. L'oeil humain reste le juge final (protocole
   côte-à-côte existant) : un morph laid peut avoir un cut_score doux.
Intégration : score_video(path, frames, sw1, sw2) a déjà tout le nécessaire en signature ;
ajouter les fenêtres boundary et les deux transforms (crop60 / bord). Rétro-compatible, on
rescorera les campagnes passées (Probe 0).

## 4. Implications curriculum (v1 puis v2)

- **Ajouter le tier même-scène-attribut à la liste d'événements** (chantier B3 inchangé) :
  ~30% des séquences revisit, prompts à caméra mobile obligatoires, retour verbatim ET
  paraphrasé (le tier paraphrasé était déjà exigé par la baseline frame-bank, plan_final).
  Sans ce tier, le modèle apprend "switch retour = cut de scène" comme politique générale et
  le rejoin ne peut PAS émerger : c'est un choix de distribution, pas d'architecture.
- **Le teacher oracle rembobine par construction : toxicité SEGMENTÉE.** Sur cross-scène,
  rewind = comportement correct, la RSD y est saine. Sur même-scène, distiller
  memzero+splice-de-A apprendrait la téléportation. En v1 ce n'est PAS un problème : la v1
  tourne SANS RSD (loss = DMD + L_rec, décision localplan_D). Le danger est en v2 : y
  restreindre la RSD aux événements cross-scène (un masque par type d'événement dans la
  liste B3, trivial) et couvrir le tier même-scène autrement (deux options ci-dessous).
- **Cible rejoin data-free pour le DÉCODEUR (v1, gratuite)** : sur le tier même-scène, la
  cible de reconstruction devient le COMPOSITE auto-généré : bord = dernière frame présente
  (dispo au moment du splice), centre/sujet = frame évincée de A (déjà stockée pour L_rec).
  Le décodeur apprend "décor courant + sujet mémorisé" par construction, sans rollout ni
  teacher. C'est M5 au niveau de la cible plutôt que du mécanisme, compatible masque fixe v1
  et masque appris v2.
- **Teacher rejoin data-free pour la RSD (v2), deux constructions** :
  (a) **twin no-switch même RNG** : la machinerie twin e3/RSD (snapshot -> rollout no_grad ->
  restore, même bruit) sait générer "la vidéo si A avait continué sans interruption". Au même
  index temporel global, le twin fournit le monde continué de référence : superviser
  bord/basses fréquences sur le twin-même-temps et l'identité sur l'évincé de A. Coût = un
  rollout twin par séquence même-scène, la machinerie existe.
  (b) **pile e4 comme loss rejoin sans teacher dense** : c'est frappant à quel point les
  outils existent déjà : terme identité DINO vs ancre A (l.1033-1043) tire l'identité,
  mm_rel/lag-2 + tripwire D2 vs le twin no-splice (l.987-1032) interdisent le saut au joint.
  La pile anti-Goodhart construite pour contraindre l'injection devient la définition
  opérationnelle du rejoin : identité vers A, dynamique do-no-harm à la frontière.
- **Piège Goodhart symétrique à surveiller** : une fois cut_score dans la loss ou la
  sélection, le mode de triche évident est le MORPH lent (interpoler linéairement vers la
  frame rembobinée en 10 frames : cut_score doux, monde quand même rembobiné). Le
  rejoin_index (bord vs B_tail) le voit, le cut_score seul non : toujours les deux, plus
  l'oeil humain.

## 5. Synthèse actionnable

| Priorité | Action | Coût | Décision qu'elle débloque |
|---|---|---|---|
| 1 | Probe 0 : métriques §3 dans le scorer + rescoring des campagnes existantes | 0 génération | le détecteur de cut est-il calibrable ? |
| 2 | Probe 1 : campagne SAMESCENE (memzero, oracle, oracle1, oracle1sink, dose, composite, stock) | ~8 arms x 40 min | M1 suffit-il ? sinon P2/bruit/composite ? |
| 3 | Tier même-scène dans B3 + cible composite du décodeur | ~0 (design) | le rejoin devient entraînable en v1 |
| 4 | v2 : RSD segmentée par type d'événement + teacher twin no-switch | v2 | rejoin distillable sans rembobiner |

Le pari le moins cher et le plus informatif : P2-sink et l'ancre bruitée transforment tous
deux l'ancre de "vérité géométrique" en "prior d'identité" SANS une ligne d'entraînement, et
les deux se mesurent dans la même campagne de 40 minutes que des probes déjà décidés
(oracle-position-sink, oracle-dose). Le design v1 n'a probablement pas besoin de changer,
seulement d'être mesuré sur l'axe qui manquait.
