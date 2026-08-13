# Plan : déterminer si les poids rapides contiennent l'identité du sujet, puis l'en extraire

*(Écrit sans noms de code internes ; chaque terme spécifique est défini à sa première
apparition. Dès la sortie du mode plan : écrire la note mémoire "zéro jargon interne avec
Aden" : demande explicite du 04/08.)*

## Où on en est, en langage clair
Notre mémoire = des petits modules à "poids rapides" (mis à jour en ligne pendant la
génération, à partir des activations des images qui sortent de la fenêtre d'attention :
c'est du test-time training). Au retour d'un sujet, on décode depuis ces poids une
image-souvenir qu'on insère dans le contexte du modèle. Les expériences de cette semaine
(toutes pré-enregistrées, chacune avec son contrôle) ont établi :
1. Le modèle INCORPORE l'image insérée (sa structure spatiale grossière se retrouve dans
   les images suivantes pendant ~3 s) : le canal de transmission fonctionne.
2. Mais notre image-souvenir décodée ne contient RIEN du sujet (corrélation ~0.002 avec
   lui) : elle reflète l'ambiance générale de la vidéo. Résultat robuste : quel que soit
   le moment du décodage et le lieu d'insertion.
3. Preuve par contrôle : insérer l'image-souvenir d'une AUTRE vidéo (le "contrôle à
   mémoire permutée") donne les mêmes scores que la nôtre, partout.
4. MAIS : en sur-apprentissage extrême (8 vidéos), les couleurs spécifiques des sujets
   DEVIENNENT stockables et décodables : la capacité des poids rapides n'est pas le
   verrou ; c'est l'extraction généralisable qui échoue.
5. Deux causes racines identifiées : (a) pendant l'entraînement de la mémoire, le
   décodeur recevait le prompt : or nos prompts décrivent les sujets : tout attribut
   textuel était "gratuit" pour la loss, la mémoire n'a jamais eu besoin de le stocker ;
   (b) la loss de reconstruction moyenne favorise le contenu commun (la scène) au
   détriment du contenu spécifique (l'individu, petite fraction des pixels).
6. La revue de code indépendante d'hier a bloqué in extremis un ré-entraînement mal ciblé :
   la métrique d'entraînement prévue (corrélation basse-fréquence en NIVEAUX DE GRIS) ne
   distingue pas les sujets entre vidéos : même une vraie image du sujet A corrèle plus
   avec le reste de sa propre vidéo qu'avec un rendu propre de A. Et les paires de
   contraste prévues (deux vidéos du même prompt, graines différentes) manquaient à la
   banque d'entraînement.

## La question centrale, jamais posée directement en 3 semaines
Toutes nos tentatives ont essayé de DÉCODER une image depuis les poids rapides. Personne
n'a fait le test le plus élémentaire : **l'information d'identité est-elle seulement
EXTRACTIBLE des poids rapides ?** La ré-identification (dire QUI c'était) est bien plus
facile que la reconstruction (redessiner). Si même elle échoue, tout décodeur est
condamné et on a la preuve d'arrêt la plus forte possible. Si elle réussit, on sait
exactement quoi construire. C'est la méthodologie standard de "probing" en
interprétabilité, appliquée (à notre connaissance pour la première fois) à une mémoire
test-time-training vidéo : publiable quel que soit le verdict.

## Quatre corrections issues de l'auto-audit du 04/08 (relecture de nos propres preuves)
1. Le "cos 0.002, l'ancre ne contient rien du sujet" est partiellement un ARTEFACT :
   cosinus DINO sur image floue = mesure de dégradation, pas d'identité (on l'avait
   nous-mêmes prouvé). La question "l'état contient-il l'individu" est plus ouverte que
   dit : confiance du verdict positif de l'étape 1 remontée à ~50-55%. DOCTRINE gravée :
   plus jamais de cosinus d'embedding sur image hors-distribution comme preuve ; les
   décodes flous se jugent par sondeur entraîné ou contraste intra-vidéo.
2. Les MASQUES DU SUJET (crops de l'étape 1, cible de l'étape 2) n'existent pas encore :
   TÂCHE n°1 : segmentation guidée par le nom du sujet du prompt (GroundingDINO+SAM) sur
   les images propres du corpus, une fois, avec planche de validation à l'œil AVANT tout.
3. Le "plafond du canal +0.131" mélangeait perte d'écriture et perte de l'ANCIENNE
   lecture : ce n'est pas une loi, c'est la mesure d'un système daté ; il peut bouger.
4. Forme de projet retenue : le papier-anatomie (canal + sondage + méthode hors-ligne +
   échelle d'interventions contrôlées) comme SQUELETTE ; l'étape 2 comme couronne si
   elle marche : fort sans elle, complet avec.

## ÉTAPE 1 : le sondage des poids rapides (~1 jour, décisif)
- Données : les états des poids rapides, reconstitués hors-ligne en rejouant les
  écritures sur les activations déjà stockées sur disque (aucune génération vidéo) :
  165 vidéos, dont 30 PAIRES "même prompt / graines différentes" : deux vidéos décrites
  identiquement mais aux individus visuellement distincts : le texte ne peut pas aider,
  c'est le juge d'identité parfait.
- Lecture de l'état : FONCTIONNELLE, pas directe : on passe une batterie fixe
  d'entrées-sondes à travers le module à poids rapides et on recueille ses sorties (sa
  "signature comportementale" : c'est ainsi que le décodeur le lit, par requêtes) ;
  variante de contrôle : statistiques des écarts aux poids initiaux. États échantillonnés
  à plusieurs âges d'écriture, y compris après l'interférence du second sujet (la
  condition opérationnelle réelle).
- Tâche : une petite tête (MLP) apprend à projeter cette signature vers un espace
  d'embedding où elle est plus proche de l'embedding visuel (DINO) du crop du sujet de
  SA vidéo que de celui de la vidéo jumelle du même prompt, et des autres vidéos.
- Le piège identifié et son contrôle : deux vidéos de même prompt diffèrent AUSSI par le
  fond : le sondeur pourrait matcher l'ambiance. On mesure donc la précision contre des
  références "crop du sujet" ET contre des références "fond seul" : l'écart entre les
  deux = ce qui est spécifiquement extractible du sujet. Autres contrôles : états
  permutés entre vidéos (la précision doit s'effondrer), split strict par vidéo,
  calibration de puissance, seuils fixés par écrit AVANT le run.
- Renforts issus du dernier tour de réflexion (04/08 soir) :
  (a) BORNE SUPÉRIEURE obligatoire : le même pipeline appliqué aux activations brutes
  (qui contiennent tout) doit être quasi parfait : sinon c'est le protocole qui est
  cassé, pas la mémoire : on construit sur ce cas facile d'abord, puis on ne change que
  l'entrée. (b) Sondes ALIGNÉES SUR LES DONNÉES (centroïdes k-means des activations du
  corpus + sous-échantillon des requêtes apprises du décodeur), pas de vecteurs
  aléatoires (recouvrement quasi nul avec les directions sujet en dim 1536).
  (c) Statistique : marges continues par paire (moyennées sur âges et images), test
  apparié ; règle d'extension pré-enregistrée : résultat en zone grise (55-75%) → +30
  paires générées (une nuit, ~40GB) AVANT toute conclusion. (d) Grille d'interprétation
  scellée : sujet↑fond↓ = feu vert ; les deux ↑ = décidé par la marge sujet−fond
  (seuil écrit d'avance) ; sujet↓fond↑ = enregistreur d'ambiance, ARRÊT ; les deux ≈
  hasard = ARRÊT. (e) Logistique : modèles de segmentation téléchargés sur disque
  projet (home plein).
- Verdicts possibles (règles écrites avant le run) : (a) ré-identification ≈ hasard →
  l'identité n'est pas extractible des poids rapides tels qu'écrits : ARRÊT net du
  chantier mémoire, avec le résultat de sondage comme contribution ; (b) au-dessus du
  hasard mais AUTANT par le fond que par le sujet → mémoire holistique de scène : ARRÊT
  aussi (une mémoire de scène, la banque concurrente la fait mieux et sans entraînement) ;
  (c) nettement au-dessus, tirée spécifiquement par le SUJET → feu vert pour les étapes
  2-3, et la tête de sondage devient elle-même un outil.

## ÉTAPE 2 (si extractible) : d'abord la LECTURE seule, l'écriture ensuite
Principe d'escalade (nouveau, issu de la réflexion) : si l'identité est extractible des
poids rapides TELS QU'ÉCRITS, alors on entraîne D'ABORD uniquement la nouvelle lecture
(requêtes+têtes, écrivain gelé) : moins cher, et ça sépare proprement "problème de
lecture" de "problème d'écriture". On ne touche à l'écriture (étape 2b) que si la
lecture seule plafonne nettement sous ce que le sondage promettait.

### Étape 2b : LA REFONTE D'ARCHITECTURE (autorisée par Aden 04/08) : mémoire à 2 registres
Le défaut architectural que toutes les mesures désignent : l'écriture est INDIFFÉRENCIÉE
(tous les tokens à égalité → le fond majoritaire écrase l'individu minoritaire à chaque
mise à jour ; le sur-apprentissage contourne par force brute, la règle générale jamais).
La refonte : garder le registre actuel pour la scène, ajouter un PETIT registre SUJET qui
ne reçoit que les tokens pondérés par "est-ce le sujet" : LE TRI SE FAIT À L'ÉCRITURE,
pas à la lecture. Signal de tri : à l'entraînement, les masques de segmentation (tâche
n°1) transférés aux tokens (alignés patches) ; à l'inférence, la carte de cross-attention
du générateur vers le nom du sujet (existe gratuitement dans chaque forward, jamais
exploitée ; repli : mini-tête de masque sur le latent évincé ; entraîner avec masques
bruités pour couvrir l'écart). Lecture du souvenir : depuis le registre sujet (image ou
tokens additifs). Positionnement : factorisation côté ÉCRITURE (Memento factorise côté
lecture : complémentaire ; vérifier dans leur papier). Coût : +1-1.5 j vs 2b naïf,
machinerie de rejeu inchangée. Affinements (dernier tour) : le registre sujet peut être
PETIT (contenu compact → moins d'interférence, plus rapide) ; et il N'ÉCRIT PAS quand le
chunk ne contient pas de sujet (masse du masque < seuil → écriture sautée : l'intuition
d'Aden "ne pas réécrire l'inutile", au seul endroit où elle a un levier mesurable).

## Calendrier vérifié (si GO au 05/08)
Masques+planche œil (0.5 j) → sondage avec audit pré-run (verdict ~06/08) → lecture seule
(hors-ligne ~08-09/08) → campagne vidéo (~10/08) → registre-sujet si nécessaire (campagne
~12-13/08) → 2 jours de marge avant le kill du 15/08. Écriture de l'article : reprend au
verdict du sondage quoi qu'il dise.
- D'abord 30 min de lecture (par moi) du papier concurrent le plus proche (Memento, code
  public : leur objectif est "reconstruire le sujet depuis la mémoire seule" avec deux
  jeux de requêtes) : adopter ce qui est meilleur, citer honnêtement ; notre différence :
  backbone gelé + entraînement hors-ligne par rejeu des écritures.
- Le décodeur gagne un second jeu de requêtes dédié au SUJET : entraîné SANS aucun
  conditionnement texte (correction de la cause racine 5a), avec deux têtes : une
  reconstruction dense du crop du sujet des vraies images (les masques du sujet sont
  précalculés une fois sur les images propres du corpus), et une régression vers
  l'embedding visuel du sujet (utile pour les gates et l'étape 3). Pénalité contrastive
  auxiliaire avec les paires même-prompt. La reconstruction pleine image actuelle reste
  comme régulariseur. Recette d'entraînement déjà validée (gradient à travers toutes les
  écritures, démarrage depuis les poids existants, ~2 h par itération).
- Validation en cascade, seuils écrits avant chaque run : sur-apprentissage 8 vidéos
  (la marge sujet doit s'ouvrir), puis vidéos held-out, puis SEULEMENT une campagne
  vidéo : 6 cellules, notre mémoire réparée × deux voies d'insertion (remplacer l'image
  la plus récente du contexte ; ajouter des tokens supplémentaires à proximité) CONTRE
  le contrôle à mémoire permutée : le verdict = l'écart avec ce contrôle (aujourd'hui
  nul), plus l'œil d'Aden (la couleur revient-elle).

## (Il n'y a PAS d'étape 3 : décision Aden 04/08)
La "livraison par les mots" (réduire le souvenir à des attributs injectés dans le prompt)
n'est pas une étape : c'est UNE LIGNE du tableau final de l'étape 2 : la baseline
dégradée que l'image-souvenir doit dépasser pour justifier son existence. Elle ne se
mesure qu'au moment de la campagne finale, comme bras de comparaison, jamais avant.
Le plan = DEUX étapes et un tableau final : (1) le sondage ; (2) l'image-souvenir
fidèle, avec campagne finale à bras multiples : notre mémoire-pixels, la réduction-en-
mots, la mémoire permutée, le système actuel, sans mémoire.

## Rôles et méthode (directives d'Aden)
Réflexion, design, lectures : moi, en direct. Agents : uniquement (a) implémenter sur
spécification serrée, (b) relire le code avant chaque run GPU, (c) exécuter les
campagnes. Chaque expérience : prédictions et seuils écrits avant, contrôles intégrés,
compte-rendu avec vidéos légères consultables sur téléphone. Échéance dure du chantier
mémoire : 15/08 ; l'écriture de l'article reprend au premier verdict de l'étape 1
(qui est une section de l'article quoi qu'il dise) ; plus aucune exploration après
le 01/09.

## Niveaux de confiance (calibrés, demandés par Aden)
- Étape 1 s'exécute proprement (technique) : ~90%. Design étanche (contrôle fond/sujet) : ~80%.
- Verdict central "l'identité est extractible, tirée par le sujet" : **~45%**
  (pour : le sur-apprentissage stockait les couleurs ; contre : tout décodage passé = ambiance).
- Étape 2a (lecture seule suffit, hors-ligne) : ~60% conditionnel. Étape 2 en vidéo
  (battre le contrôle à mémoire permutée) : ~50% conditionnel. Étape 3 (texte) : ~65%
  conditionnel.
- **"Mémoire qui marche" publiable avant le 15/08 : ~30%.**
- **Contribution publiable quoi qu'il arrive : ~90%** (le sondage est une section d'article
  dans les deux sens ; l'anatomie reste le socle).

## Ce qui existe déjà et est réutilisé tel quel
Le rejeu hors-ligne des écritures (validé au bit près) ; les 165 vidéos + 30 paires
même-prompt sur disque (rien à collecter) ; la recette d'entraînement complète ; le banc
d'évaluation vidéo avec son contrôle à mémoire permutée ; le juge de couleur validé ;
le format compte-rendu téléphone.
