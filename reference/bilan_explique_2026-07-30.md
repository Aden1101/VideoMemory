# Ce qu'on a construit, testé et trouvé — version expliquée (30/07/2026)

*(Tous les chemins sont donnés pour que tu vérifies chaque affirmation toi-même : vidéos,
logs, scores bruts. Rien n'est à croire sur parole.)*

---

## 1. Notre mémoire, pas à pas

Le générateur vidéo (LongLive, 1.3B) ne voit que les ~12 dernières images latentes : tout ce
qui sort de cette fenêtre est perdu, sauf un petit cache permanent (le "sink") qui garde les
toutes premières images de la vidéo.

Notre module ajoute une mémoire à l'intérieur du modèle, en trois morceaux :

- **L'écrivain** (le cœur TTT) : à chaque fois qu'un paquet d'images sort de la fenêtre,
  il le comprime dans un état interne de taille FIXE (~28M de valeurs par bloc, 3 blocs :
  ça ne grossit jamais, même sur une vidéo infinie). Techniquement : des "fast weights",
  des poids qui s'ajustent pendant la génération, mis à jour par une règle d'écriture
  apprise (fichier : `wan/modules/video_memory.py`).
- **Le décodeur d'ancre** : sur demande, il lit l'état et reconstruit UNE image-souvenir
  (floue, 33M de paramètres, entraîné à reconstruire les images perdues).
- **Le splice au retour** : quand le prompt ramène un sujet déjà vu, on glisse
  l'image-souvenir décodée dans la fenêtre du modèle au moment du changement de scène ;
  le générateur s'en sert comme point d'appui pour re-créer le sujet.

L'écrivain et le décodeur sont entraînés HORS-LIGNE (sans le gros modèle dans la boucle),
sur 165 vidéos collectées une fois pour toutes : c'est ce qui a permis d'itérer vite.
Étiquette de ce système complet dans les fichiers de résultats : **`memdec20`**.

## 2. Le protocole de test

**Scénarios** (12 jeux de sujets variés × 2 graines aléatoires = 24 vidéos par scénario) :

| Étiquette | Déroulé | Ce que ça teste |
|---|---|---|
| S1 | A → B → retour A (A ouvre la vidéo) | Le terrain historique, facile |
| S3 / S4 | A → B → C → A, ou A → B → A → C → A (A ouvre) | Plus d'interférence, mais A reste dans le sink |
| **S2** | **X → A → B → retour A (A n'ouvre PAS)** | **Le terrain dur : le sink retient X, pas A** |
| S2p/S3p/S4p | pareil, mais le retour est demandé en PARAPHRASE | Punit les méthodes qui matchent sur les mots |

**Les cinq systèmes comparés** (mêmes vidéos, mêmes graines, comparaison appariée) :

| Étiquette | Qui c'est |
|---|---|
| `stockoff` | LongLive officiel, tel quel |
| `memzero` | Notre modèle, mémoire coupée (la référence des deltas) |
| `memdec20` | **Notre mémoire** (écrivain + image-souvenir décodée) |
| `framebank` | Le concurrent publié (Echo-Forcing) : bibliothèque de vraies images qui grossit + ré-injection au retour, routée par similarité de prompt |
| `oracle1` | La triche : ré-injecter la vraie image parfaite (= plafond) |

**La métrique** : similarité DINO (réseau de reconnaissance d'images, standard du domaine)
entre le sujet au RETOUR et ce même sujet à sa PREMIÈRE apparition. Deux variantes : image
entière ("full") et recadrée sur le sujet ("crop60"). Statistique : delta apparié vs
`memzero` sur les 24 vidéos + test de signe (on compte les vidéos où c'est mieux/pire).

## 3. Les résultats, un par un

### 3.1 Terrain facile (A ouvre la vidéo : S1, S3, S4) : tout le monde "réussit", pour une mauvaise raison

Deltas vs sans-mémoire : LongLive +0.22 à +0.25, bibliothèque +0.20 à +0.30, nous +0.11 à
+0.13, triche +0.21 à +0.33. LongLive a l'air d'avoir une mémoire : c'est son sink : les
premières images (donc A) ne partent jamais. La bibliothèque brille aussi : en prompt
verbatim, son routage par mots est infaillible (similarité 1.0), elle ré-injecte une image
parfaite. Nous, on est plafonnés (~+0.13) par le canal écriture→lecture (mesuré par une
sonde dédiée, cf. §5).

### 3.2 Terrain dur (A n'ouvre pas : S2, S2p) : LE résultat

D'abord un piège qu'on a dû désamorcer : le sink épingle la frame 0 (le sujet intercalaire
X) pendant TOUTE la vidéo, ce qui contamine tout le rendu, y compris le segment qui sert de
référence au scoring (il ne ressemble qu'à 0.165 à un rendu propre de A). Re-mesuré contre
un rendu propre de A (même vidéo, même graine) :

| Système | Score absolu | Delta vs sans-mémoire | Vidéos mieux/pire | p |
|---|---|---|---|---|
| Sans mémoire | 0.224 | — | — | — |
| **Notre mémoire** | **0.658** | **+0.434** | **24+ / 0−** | **< 1e-4** |
| Bibliothèque (Echo) | 0.250 | +0.027 | 13+ / 11− | 0.84 |
| Triche (vraie image) | 0.251 | +0.027 | 13+ / 11− | 0.84 |
| LongLive | 0.127 | −0.096 | 7+ / 17− | 0.064 |

En paraphrase (S2p) : nous **+0.357 (23+/1−)**, bibliothèque +0.093 (n.s.), LongLive −0.06.

**Pourquoi ce renversement.** La bibliothèque stocke ce qui a été RENDU : comme le rendu de
A était déjà pollué par X, elle stocke et ré-injecte la pollution : même la "vraie image"
(triche) échoue pour la même raison. Notre mémoire réancre la génération depuis son état
interne : c'est le seul mécanisme qui survit à la dégradation du rendu. En plus, la
bibliothèque rate 17% de ses rappels en paraphrase (routage par mots), nous 0.

### 3.3 Ce que ça donne comme message de papier

Sur le seul terrain où une mémoire est réellement NÉCESSAIRE (sujet absent du cache), toutes
les méthodes existantes échouent : cache positionnel, bibliothèque de frames, et même la
frame parfaite : seule la mémoire à état appris ramène le sujet, 24 vidéos sur 24.

## 4. Vérifie toi-même (anti-artefact)

1. **Regarde les vidéos S2 côte à côte** [sans mémoire | bibliothèque | nous] :
   `reference/runs/reid_scen/eyepack_s2/S2_t*_s0_memzero-framebank-memdec20.mp4`
   (3 fichiers). Au dernier segment : qui ramène le sujet ?
2. **La preuve de la contamination de référence** : la planche
   `reference/runs/reid_scen/eyepack_s2/S2_board_reference_vs_cleanA_vs_arms.png` :
   colonne 1 = le segment que le scoring utilisait comme "A" (tu verras qu'il ressemble à
   X), colonne 2 = A propre, colonnes suivantes = ce que chaque système produit au retour.
3. **Les scores bruts, vidéo par vidéo** :
   `reference/runs/reid_scen/results/` (un JSON par vidéo) et le re-scoring propre dans
   `reference/runs/reid_scen/results/clean_rescore/`. Le résumé se régénère avec :
   `python reference/runs/reid_scen/run_reid_scen.py --mode summary --scen S2,...`
4. **Les prédictions écrites AVANT les mesures** (anti-rationalisation) :
   `reference/memory2_probes_registry.md`.
5. **Les audits complets** (y compris nos deux fausses pistes corrigées, assumées) :
   `reference/scenarios_design_report.md` sections 8-9, `reference/memory2_probes_report.md`.
6. **Le routage de la bibliothèque, décision par décision** (ses 17% d'erreurs en
   paraphrase) : logs `[framebank] routing` dans les résumés + JSONs d'événements.

## 5. Les limites, sans maquillage

1. **Catégorie, pas encore individu.** Le sujet ramené est le bon type dans le bon décor,
   mais le test d'instance calibré est NÉGATIF : le A ramené n'est pas plus proche de SON A
   (0.658) que d'un A d'une autre graine (0.665), alors que la métrique sait distinguer
   deux graines (calibration : 0.933 vs 0.799). La cause est cartographiée : le canal
   écriture→lecture plafonne les détails fins (sonde "thruwriter" : même une image PARFAITE
   écrite puis relue immédiatement ne rend que +0.13) ; le correctif visé (terme d'instance
   en espace DINO) est spécifié pour la v2.1.
2. **Terrain facile : on ne bat pas LongLive chez lui.** S1 absolu : nous 0.812 vs LongLive
   0.847. Son sink fait gratuitement ce que notre mémoire paie. Le papier assume : notre
   valeur est là où le sink ne peut rien.
3. **Petite régression en paraphrase double-revisite** (S4p : −0.01 n.s.) : le
   conditionnement du décodeur n'a jamais vu de paraphrases à l'entraînement. Corrigeable.
4. **Puissance statistique** : 24 vidéos par scénario (12 jeux de sujets × 2 graines). Le
   24+/0− de S2 est solide (p<1e-4), mais les petits deltas (±0.03) ailleurs ne sont pas
   interprétables individuellement. Extension prévue pour les chiffres du papier.
5. **Un seul substrat** : tout est mesuré sur LongLive-1.3B. La généralisation à d'autres
   backbones est une hypothèse, pas un résultat.
