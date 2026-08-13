# Programme d'entraînement complet : faire revenir le sujet à l'écran

> **RÉVISION 2026-08-08 (autonomie donnée par Aden : "je veux juste une mémoire qui
> marche, pas trop de temps").** Le plan ci-dessous garde toute sa valeur, mais il est
> précédé d'une étape 0 qui décide s'il vaut la peine d'être exécuté. Motif : on
> s'apprêtait à construire un décodeur sans connaître la FIDÉLITÉ qu'il doit atteindre,
> alors que les maillons d'aval (insertion → adoption par le générateur → œil d'Aden)
> n'ont jamais été validés par son œil, même avec une image parfaite.
>
> ## ÉTAPE 0 : L'ÉCHELLE DE FIDÉLITÉ (1 jour, protocole scellé ci-dessous)
> Scénario de revisite (sujet A → autre scène → retour de A). À l'instant du retour, on
> insère dans le contexte une image de fidélité CONTRÔLÉE, et Aden juge en aveugle.
> BARREAUX (du meilleur au pire) :
>  1. `oracle` : la vraie image du sujet A (borne haute absolue) ;
>  2. `flou_leger` / `flou_fort` : la vraie image dégradée (ce que produit un décodeur
>     imparfait) ;
>  3. `meme_categorie` : le sujet du JUMEAU (même prompt, autre individu) = exactement le
>     niveau "voisinage" que notre mémoire sait fournir aujourd'hui ;
>  4. `autre_categorie` : un sujet d'une autre catégorie (contrôle négatif) ;
>  5. `rien` : aucune insertion (référence).
> PROTOCOLE : ~10 scénarios × 6 barreaux, mêmes prompts et mêmes graines, fichiers
> ANONYMISÉS, table de correspondance scellée avant l'envoi. Aden répond à UNE question
> par vidéo : "le sujet est-il bien revenu ?" (oui / à peu près / non).
> CE QUE ÇA DÉCIDE, écrit d'avance :
>  - si `oracle` ne convainc pas son œil (≤ 6/10 "oui") : AUCUN décodeur ne peut réussir,
>    le programme d'entraînement est ANNULÉ et on écrit le papier. C'est le verdict le
>    moins cher possible.
>  - si `meme_categorie` convainc autant que `oracle` (écart ≤ 2 scénarios) : le niveau
>    VOISINAGE suffit → notre mémoire est déjà au bon niveau, le décodeur devient un
>    problème facile, GO franc pour les étapes 1-4.
>  - si seul `oracle` (et éventuellement `flou_leger`) convainc : la fidélité d'INSTANCE
>    est requise → nos mesures disent que l'état ne la porte pas → on écrit le papier, et
>    l'échelle de fidélité devient elle-même un résultat du papier (personne n'a mesuré
>    quelle fidélité d'identité un canal d'insertion exige).
> Cette étape produit des vidéos réelles regardables par Aden dès le lendemain, et elle
> est utile au papier dans les trois cas.

Objectif final, en une phrase : produire des vidéos où un sujet apparaît, disparaît, puis
revient, et où Aden, en aveugle, désigne systématiquement celles qui utilisent notre
mémoire. Tout le reste est un moyen.

---

## 1. La leçon des trois échecs passés, qui dicte la conception

1. **La cible d'entraînement était fausse.** L'ancien décodeur apprenait à reconstruire
   l'image ENTIÈRE : la scène occupe 80-90 % des pixels, donc la perte était dominée par
   le décor et le sujet ne pesait rien. Résultat mesuré : il rendait la CATÉGORIE
   ("un cheval") jamais l'INSTANCE ("le cheval blanc").
2. **La référence d'identité ne contient pas l'identité.** Mesuré le 07/08 : même la
   lecture directe des activations donne au JUMEAU le même rang médian (9/60) qu'au vrai
   sujet. Un embedding visuel générique calculé sur une vignette encode le voisinage
   (catégorie, apparence grossière), pas l'individu. ⇒ **INTERDIT** d'en faire la cible
   principale d'un entraînement : on ré-obtiendrait la catégorie.
3. **L'état ne contenait rien d'exploitable.** Corrigé depuis : la règle d'écriture
   correctrice (delta) fait 13.5/19 de médiane contre 7.0 pour l'écriture actuelle.

**Décision de conception qui découle des trois** : la cible d'entraînement du décodeur
est la **reconstruction en PIXELS de la région du sujet uniquement** (masque), pas
l'image entière, pas un embedding. Les pixels du sujet portent l'instance par
construction (couleur de robe, motif, morphologie) sans dépendre d'un encodeur tiers.
L'embedding ne sert plus que de métrique secondaire, jamais de cible.

---

## 2. Les bras comparés (3 mémoires, 1 décodeur par mémoire, budget identique)

| bras | règle d'écriture | statut du code |
|---|---|---|
| **M0** | notre écriture actuelle (associative, MLP) | native, aucune modif |
| **M1** | MLP correcteur (résidu + séquentiel 512 + Muon) = A3 | native (grad_fn), déjà validée |
| **M2** | règle delta linéaire (état matriciel) | **à implémenter dans le chemin d'inférence** |

M2 est celle qui marche le mieux au sondage mais elle n'existe aujourd'hui qu'en rejeu
hors ligne : il faut l'ajouter comme type de mémoire dans le modèle, avec un test
d'équivalence bit-à-bit contre le rejeu. C'est le seul vrai développement du programme.

---

## 3. Le jeu de données (le point où l'on s'est planté trois fois)

**Scénario** : A → B → A (le sujet part, autre chose se passe, le sujet revient). C'est le
seul régime où une mémoire a un travail que le contexte local ne fait pas déjà.

**Diversité des sujets** : leçon des corpus v3/v4. On évite (a) les humains de même
métier (indistinguables pour toute métrique disponible), (b) les espèces étroites
(éléphant, girafe : le générateur en fait des clones). On prend des catégories à forte
variabilité intra-classe : chien, oiseau, voiture, fleur, poisson, bateau, moto, sac,
chapeau, meuble. Chaque prompt sans attribut d'apparence.

**Taille** : 600 vidéos d'entraînement, 120 de validation, 120 de test (sujets
DISJOINTS entre les trois : on teste la généralisation à des sujets jamais vus, pas à des
graines jamais vues). Génération ~1 min/vidéo/GPU : une nuit sur 4 cartes.

**Stockage (contrainte réelle, ~90 GB libres)** : on ne garde PAS les activations brutes
(1.1 GB/vidéo = impossible). On garde, par vidéo : l'état de mémoire aux 2 âges utiles
(fin du segment A, retour de A) + le masque du sujet + les images cibles. Pour la mémoire
linéaire l'état pèse ~14 MB/âge, pour le MLP ~85 MB/âge. Budget : ~45 GB pour M2,
~60 GB pour M0/M1 en ne gardant qu'un âge. On génère et on traite bras par bras pour ne
jamais dépasser.

---

## 4. Le décodeur (architecture et objectif)

**Entrée** : l'état de mémoire, lu par un jeu de requêtes apprises (mécanisme déjà
validé : c'est ainsi que l'ancien décodeur lisait).
**Sortie** : une image latente insérable dans le contexte de génération (le canal
d'insertion existe et transmet, mesuré +0.83).
**Taille** : ~33 M paramètres (l'ancienne, qui n'était pas le goulot).

**Objectif d'entraînement, dans l'ordre d'importance** :
1. **Reconstruction pixel PONDÉRÉE PAR LE MASQUE** du sujet (poids ~10× sur la région du
   sujet, 1× ailleurs). C'est la correction n°1.
2. **Terme contrastif entre jumeaux** : la sortie doit être plus proche du sujet de SA
   vidéo que de celui de la vidéo jumelle (même prompt, autre graine). Empêche le
   repli sur la catégorie.
3. **Régularisation** : reconstruction pleine image à faible poids, pour que l'image
   insérée reste plausible et ne casse pas la génération.

**Ce qu'on ne fait PAS** : conditionner le décodeur sur le texte du prompt (c'était la
cause racine identifiée : le texte donnait gratuitement les attributs, la mémoire n'avait
jamais besoin de les stocker).

---

## 5. Protocole d'évaluation (décidé AVANT d'entraîner)

**Quantitatif** (indicatif seulement) : similarité de la région du sujet après le retour,
contre le sujet d'origine ; comparée entre bras.

**Décisif : l'œil d'Aden, en aveugle.** Pour 20 scénarios de test (sujets jamais vus) :
4 vidéos par scénario, mêmes prompts, mêmes graines : (1) notre mémoire, (2) **mémoire
permutée** (le souvenir d'une AUTRE vidéo), (3) sans mémoire, (4) LongLive de base.
Fichiers anonymisés, table de correspondance scellée à l'avance, Aden classe sans savoir.
**Critère de succès pré-enregistré** : Aden désigne le bras (1) comme "le sujet est bien
revenu" strictement plus souvent que le bras (2), sur au moins 14 scénarios sur 20
(binomial p < 0.06). Le bras (2) est le juge : il neutralise le "le prompt repeint le
sujet", qui nous a trompés deux fois.

---

## 6. Calendrier et risques

| étape | durée | risque |
|---|---|---|
| P0 implémentation mémoire linéaire native + test bit-à-bit | 0.5 j | faible |
| P1 génération du corpus (840 vidéos, 4 GPU) | 1 nuit | disque |
| P2 extraction masques + états + cibles | 0.5 j | faible |
| P3 entraînement des 3 décodeurs (budget identique) | 1.5 j | **le vrai risque** |
| P4 campagne vidéo + planche aveugle pour Aden | 0.5 j | faible |
| **total** | **~4 jours pleins** | |

**Risque principal** : rien ne garantit qu'un décodeur entraîné sur un état dont la
discrimination est de 13.5/19 produise une image reconnaissable. Notre propre mesure dit
que l'état porte surtout du VOISINAGE. Il est donc plausible que le résultat soit
"le sujet revient dans la bonne catégorie et la bonne teinte, mais ce n'est pas le même
individu" : ce serait déjà mieux qu'aujourd'hui, mais il faut le dire d'avance et ne pas
le vendre comme une identité retrouvée.

**Critère d'arrêt intermédiaire (P3)** : si, sur les données de validation, la
reconstruction du sujet ne bat pas nettement le décodeur alimenté par une mémoire
permutée, on arrête là et on écrit : inutile de générer une campagne vidéo dont le
résultat est déjà connu.

**Coût d'opportunité, à dire franchement** : 4 jours sur les 8 avant le 15/08. Le papier
d'analyse est déjà solide et peut être écrit en 3-4 jours. Faire les deux est possible
mais serré, et la campagne vidéo peut échouer.
