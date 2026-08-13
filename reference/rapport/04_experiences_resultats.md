# 4. Expériences et résultats

Cette section retrace le parcours expérimental complet du stage, dans l'ordre chronologique. Le fil conducteur est une seule question: une mémoire à poids rapides (fast-weight, de type TTT), injectée dans un générateur vidéo autorégressif entraîné par distillation DMD sans données réelles, peut-elle améliorer la cohérence à long terme (identité du sujet) sans dégrader le mouvement? La réponse honnête, construite run après run, est nuancée: chaque tentative a produit un enseignement précis, souvent négatif, et ces résultats négatifs constituent une part essentielle des conclusions du stage.

## 4.1 Protocole d'évaluation commun

Toutes les évaluations rapportées ci-dessous suivent le même protocole, durci au fil des runs à mesure que nous découvrions les angles morts des métriques.

**Générations.** Six prompts held-out (jamais vus à l'entraînement), graine fixée (seed 42), vidéos longues de 240 frames latentes, soit 957 images décodées (environ 40 secondes), très au-delà de l'horizon d'entraînement (environ 5 secondes par fenêtre).

**Trois bras systématiques.** Une leçon des premiers mois (audit de validité des baselines) est qu'une comparaison à deux bras est trompeuse. Chaque évaluation comporte donc:

- **mem-ON** (memon): le checkpoint entraîné, mémoire active;
- **mem-ZERO** (memzero): le même checkpoint, injection mémoire mise à zéro à l'inférence (variable d'environnement `MEM_ZERO_ALL=1`). Ce bras isole la dérive due au co-entraînement LoRA seul; la différence memon moins memzero mesure l'effet spécifique de l'injection;
- **AF-stock**: le checkpoint Anchor-Forcing officiel, sans mémoire ni fine-tuning. C'est la référence absolue.

**Métriques.** VBench (Subject Consistency SC, Background Consistency BC, Dynamic Degree DD), une métrique de mouvement lag-2 insensible au scintillement (flicker-blind, seuil de gel pré-enregistré à 0.01), la dérive DINO par frame par rapport à une ancre initiale, et surtout la similarité cosinus d'identité id_cos(f0, ft). Après avoir constaté que la valeur id_cos sur une seule frame finale est bruitée et a produit un faux verdict (voir 4.4), la métrique primaire est devenue le **tail-mean id_cos**: la moyenne de la courbe id_cos sur les 300 dernières frames, et l'écart memon moins memzero de ce tail-mean, avec un décompte par prompt (n/6 positifs). Les seuils sont pré-enregistrés: plancher de bruit DINO 0.18, tolérance de parité 0.03.

**Évaluation sink-starvation.** À partir de c2, nous ajoutons une condition d'évaluation où le sink d'attention (les tokens d'ancrage que le cache KV conserve en permanence) est réduit de 3 à 1. Cette privation crée artificiellement un travail pour la mémoire: si elle retient réellement le contenu que le cache a perdu, elle doit aider dans cette condition. C'est le test « it works » le plus direct que nous ayons construit.

## 4.2 Point de départ: une injection qui nuit ou se dissout

Avant la série décrite ici, deux constats avaient été établis sur les premières versions du module (runs e1 et e2, backbone gelé, seule la mémoire entraînée).

La première ablation propre (e1_rerun2, step 250 puis confirmée à 1000) montrait mem-ON avec une dérive DINO tardive de 0.501 contre 0.204 pour mem-OFF, la mémoire nuisant sur 8 prompts sur 8. Un audit a ensuite requalifié ce résultat: environ 97 % de l'écart provenait d'un artefact d'injection quasi-DC (biais constant par token, assombrissement et moirage), et une fois ce biais retiré (`MEM_DC_REMOVE=1`), l'effet net de la mémoire tombait dans le plancher de bruit: la mémoire était **inerte**, pas toxique. Le correctif d'amplitude qui a suivi (e2_amp: normalisation RMS relative par chunk plus retrait du DC) a été pire: en épinglant l'injection à une amplitude fixe à chaque chunk, il l'accumulait hors distribution sur 240 frames et **dissolvait** le sujet en taches abstraites (DINO 0.753 contre 0.271 pour mem-OFF, pire sur 8/8), tout en restant quasiment invisible pour VBench (SC 0.952 contre 0.966). Ce dernier point, la cécité de VBench devant une vidéo qui se dissout, a motivé le durcissement du protocole de la section 4.1.

Le tableau était donc posé: soit l'injection est neutralisée (inerte), soit elle dégrade. La suite du stage a consisté à chercher un régime où elle aide.

## 4.3 La série e: les garde-fous (e3_antifreeze, e4_antiflicker)

L'hypothèse de la série e était qu'on pouvait contraindre l'optimisation par des termes de régularisation (garde-fous) pour empêcher les dégénérescences connues (gel du mouvement, dissolution, biais DC) et laisser émerger un comportement utile de la mémoire.

**e3_antifreeze** ajoutait un terme d'appariement de mouvement à la référence AF-stock (mm_ext). À l'étape 250, le résultat semblait être le premier résultat non dégénéré du projet:

| VBench, e3 step 250 | mem-ON | mem-OFF | AF-stock |
|---|---|---|---|
| SC | 0.914 | 0.935 | 0.936 |
| BC | 0.927 | 0.942 | 0.948 |
| DD | 0.996 | 1.000 | 1.000 |

L'entraînement ne dégradait plus le mouvement (mem-OFF égal au stock sur toutes les dimensions) et la mémoire ne gelait plus la vidéo (DD 0.996), au prix d'environ 2 points d'identité (SC 0.914 contre 0.935).

Ce résultat était **transitoire**. À l'étape 500, le même run avait gelé:

| e3, trajectoire mem-ON | step 250 | step 500 |
|---|---|---|
| DD (mouvement) | 0.996 | **0.046** |
| SC (consistance sujet) | 0.914 | **0.966** |

Le point crucial est que SC **monte** quand la vidéo gèle: une vidéo immobile est parfaitement « consistante ». C'est le piège de Goodhart observé en direct dans un seul run: l'optimiseur a trouvé que geler bat la métrique de consistance. Deux contrôles ont écarté les explications faciles. D'une part, l'inspection visuelle à l'étape 250 révélait déjà un scintillement de luminance par patch (+22.7 % d'écart-type temporel par patch par rapport au jumeau, netteté -28 %): le run n'était « propre » qu'aux yeux des métriques. D'autre part, une évaluation sur 6 prompts jamais vus a montré que le gel n'est **pas** du surapprentissage des 8 prompts d'entraînement: mem-ON held-out DD 0.072 (gèle aussi fort que les prompts entraînés à 0.046) alors que AF-stock reste à 0.994. Le gel est un **attracteur général du self-rollout DMD**, pas une mémorisation.

**e4_antiflicker** a été la dernière tentative de cette famille: une pile de contraintes conçue pour bloquer le scintillement (passe-bas temporel EMA sur l'injection, métrique de mouvement lag-2 aveugle au flicker, retrait de la moyenne temporelle de la dose, tripwire de détection). Le résultat a été encore plus net:

| e4, VBench | mem-ON @250 | mem-OFF @250 | mem-ON @500 | mem-OFF @500 |
|---|---|---|---|---|
| SC | 0.943 | 0.939 | 0.989 | 0.989 |
| DD | 0.450 | 0.942 | **0.000** | **0.000** |

À l'étape 500, les deux bras sont totalement gelés (DD 0.0) avec une SC record de 0.989: la pile anti-flicker a simplement renvoyé l'optimiseur vers le gel, encore plus vite.

**Conclusion de la série e: le whack-a-mole est structurel.** Chaque garde-fou ferme un mode de dégénérescence et l'optimisation en trouve un autre dans l'espace des fréquences laissé libre: le retrait du DC mène à la dissolution, l'anti-dissolution mène au gel, l'anti-gel mène au flicker, l'anti-flicker ramène au gel. Nous appelons ce constat la **triade de Goodhart**: le gel bat SC, le biais DC bat les métriques de dérive, le flicker bat DD (qui compte le scintillement comme du mouvement). L'ensemble des solutions accessibles à l'optimiseur sous ces contraintes est l'union des angles morts des métriques et des solutions inertes. La série e a donc été arrêtée, avec deux décisions: ne plus jamais conclure sur VBench seul, et changer de stratégie, en s'attaquant à la cause plutôt qu'aux symptômes.

## 4.4 c1_cotrain: co-entraînement pur DMD propre

Le diagnostic issu de la série e et de la littérature (VideoSSM notamment, qui fait fonctionner une mémoire globale sous un DMD fenêtré) désignait deux différences majeures avec les systèmes qui marchent: notre backbone était gelé (la mémoire ne pouvait pas être « absorbée » par le réseau) et nos garde-fous déformaient l'objectif. **c1_cotrain** est le premier run honnête de la nouvelle famille: tous les garde-fous à zéro (DMD pur), substrat d'entraînement sain de 248 000 prompts (au lieu des 8 prompts sur-appris), LoRA du backbone entièrement dégelée et co-entraînée avec la mémoire, injection bornée (normalisation relative, retrait du DC, gate borné à 0.1).

**Trajectoire (évaluation normale, écart tail-mean id_cos memon moins memzero, vérifié dans les courbes sauvegardées):**

| c1_cotrain | écart id_cos (tail 300) | prompts positifs | DD memon | DD memzero |
|---|---|---|---|---|
| step 250 | -0.123 | 0/6 | 0.878 | 0.872 |
| step 500 | **-0.019** | 3/6 | 0.739 | 0.678 |
| step 1250 | -0.119 | 1/6 | **0.228** | 0.683 |

Trois enseignements, dans l'ordre où nous les avons appris.

**Premier: pas de gel.** À l'étape 250, le mouvement est sain (DD environ 0.88, lag-2 très au-dessus du seuil de gel), sur les deux bras. La recette de co-entraînement propre échappe à l'attracteur qui a tué e3 et e4. C'est le gain net de c1.

**Deuxième: l'injection nuit d'abord, puis est neutralisée.** À 250, l'effet spécifique de l'injection sur l'identité est franchement négatif (-0.123, dommageable sur 6 prompts sur 6). À 500, cet écart est réduit à -0.019, à l'intérieur de la tolérance de parité de 0.03: le co-entraînement a appris au backbone à absorber la lecture mémoire, atteignant le niveau « do-no-harm » (ne pas nuire). Il faut être précis sur ce que cela signifie: do-no-harm veut dire **inerte**, pas utile. Sur une dérive simple (le sujet reste à l'écran, le cache KV et le sink suffisent), la mémoire n'a pas de travail net à fournir; c'est le « mur » diagnostiqué en amont.

Une parenthèse méthodologique importante: le script d'évaluation avait étiqueté l'étape 500 « ACTIVE-NEGATIVE » sur la base de l'id_cos à une seule frame finale et d'une comparaison au stock qui facturait à la mémoire la dérive commune de la LoRA. Deux audits indépendants du code (chemin d'inférence propre, mémoire chargée et vérifiée 36/36, memzero bit-exact) plus un recalcul en tail-mean ont montré que c'était un **artefact de métrique**, pas une régression réelle: la trajectoire honnête est bien -0.123 puis -0.019. Cette leçon (échantillonner une courbe bruitée en un point produit des verdicts instables) a fixé la métrique primaire du reste du stage.

**Troisième: le do-no-harm de l'étape 500 est une fenêtre transitoire.** À 1250, l'écart d'identité redevient nettement négatif (-0.119, 1/6) et surtout l'injection **tue le mouvement**: DD memon 0.228 contre memzero 0.683. Le bras memzero, lui, reste sain et stable (identité tail 0.784, supérieure même au stock 0.759): la recette de co-entraînement est propre, c'est bien **notre injection** qui est le problème, et le co-entraînement prolongé aggrave la dépendance du modèle à cette injection nocive. Le meilleur checkpoint de c1 est vers 500; continuer n'améliore rien.

**Probes d'inférence à coût nul (checkpoint 1250).** Avant de reconstruire quoi que ce soit, deux hypothèses de correctif ont été testées sans entraînement, directement à l'inférence:

| Probe c1 (ckpt 1250, sink-starvation) | écart id_cos (tail 300) | DD memon | DD memzero |
|---|---|---|---|
| sink réduit à 1 (la mémoire devrait aider) | -0.102 (1/6) | **0.211** (figé) | 0.650 |
| MEM_REJECT_STREAM (réjection par soustraction) | -0.127 (1/6) | 0.217 | 0.683 |

Les deux sont **négatifs**: même privée de son sink, la mémoire de c1 n'aide pas (elle nuit et fige le mouvement), et la réjection de flux n'améliore rien. Conclusion: c1 n'est pas rattrapable par un patch d'inférence; le défaut est architectural et il faut reconstruire.

## 4.5 c2_eviction: le correctif eviction-feed

L'autopsie de c1 a identifié un défaut de conception fondamental: la mémoire **écrivait le chunk courant**, c'est-à-dire exactement ce que la fenêtre d'attention voit déjà. Elle était structurellement redondante avec le cache KV, donc sans travail à faire, donc disponible pour être « goodhartée ». **c2_eviction** corrige cela: la mémoire écrit désormais le chunk **évincé**, celui qui vient de sortir de la fenêtre d'attention (FIFO par bloc, avec un décalage vérifié exactement égal à la fenêtre de 4 chunks). Elle devient le complément du cache: elle détient ce que le sink et la fenêtre ont perdu. S'y ajoute un co-entraînement avec cache-dropout (probabilité 0.3 de réduire le sink de 3 à 0 par rollout), pour que la mémoire reçoive un gradient quand le cache est privé de son ancre. Le reste est identique à c1 (DMD pur, garde-fous à zéro, substrat 248k, gate 0.1). L'implémentation a été validée avant lancement: selftest bit-exact quand le mécanisme est désactivé, revue indépendante du chemin BPTT/FSDP.

**Trajectoire sink-starvation (sink 1, la condition où la mémoire doit aider), tous chiffres recalculés depuis les courbes sauvegardées:**

| c2, sink 1 | écart id_cos (tail 300) | prompts positifs | DD memon | DD memzero | SC memon |
|---|---|---|---|---|---|
| c1 (référence) | -0.102 | 1/6 | 0.211 | 0.650 | 0.966 |
| c2 step 250 | -0.055 | 3/6 | 0.783 | 0.839 | 0.971 |
| c2 step 500 | -0.053 | 3/6 | 0.356 | 0.856 | 0.979 |
| c2 step 750 | **-0.040** | 2/6 | 0.239 | 0.311 | 0.979 |
| c2 step 1000 | **-0.253** | 0/6 | 0.911 | 0.711 | 0.958 |
| c2 step 1250 | -0.138 | 1/6 | 0.989 | 0.994 | **0.800** |

En évaluation normale (sink 3), l'étape 500 donne un écart de -0.071 (1/6) avec DD memon 0.589: sur la dérive simple, la mémoire reste légèrement nocive.

**Lecture en deux phases.**

*Phase 1 (250 à 750): un vrai progrès.* Le problème de mouvement de c1 est corrigé dès 250: memon DD 0.78, dynamique comme memzero (0.84), là où c1 figeait à 0.21. Le dommage d'identité est divisé par deux (-0.055 contre -0.102) et **converge** de façon monotone vers zéro (-0.055, -0.053, -0.040). À 750, l'analyse d'attribution est encourageante: la baisse de DD visible sur memon (0.24) touche **aussi** memzero (0.31), donc c'est l'attracteur d'auto-similarité du co-entraînement DMD qui dé-dynamise les deux bras, pas l'injection (l'écart memon moins memzero sur DD se réduit de -0.50 à -0.07). Tout indiquait une convergence vers « actif et do-no-harm », l'objectif que c1 avait manqué.

*Phase 2 (1000 et 1250): la bascule.* La convergence ne tient pas. À l'étape 1000, l'écart d'identité s'effondre à -0.253 (0/6 positifs, pire que c1) alors que le mouvement remonte à DD 0.91: le modèle est entré dans un régime de **flicker à haute énergie**, très dynamique mais incohérent (le sujet se métamorphose). À 1250 le régime persiste (id_cos memon 0.353 en tail, DD 0.99) et la SC chute à 0.800 contre 0.951 pour le stock: cette fois, même VBench voit le scintillement. La non-monotonie des deux colonnes DD (0.78, 0.36, 0.24, 0.91, 0.99) est la signature de l'oscillation gel-flicker déjà rencontrée dans la série e.

**Verdict honnête sur c2.** L'eviction-feed est une amélioration claire et mesurable par rapport à c1 aux checkpoints précoces et intermédiaires (mouvement réparé, dommage d'identité divisé par deux et décroissant), mais il n'atteint pas un do-no-harm **stable**: avec plus d'entraînement, le système retombe dans la dégénérescence par scintillement. À ce stade, l'interprétation par défaut était que le mur du data-free (aucun terme d'identité dans l'objectif DMD) finit toujours par gagner, l'eviction-feed n'ayant fait que retarder l'échéance. L'autopsie qui suit a changé cette interprétation.

## 4.6 Autopsie du flicker: une pathologie d'optimisation, pas le mur

Plutôt que de conclure sur les seules métriques d'évaluation, nous avons dépouillé la télémétrie d'entraînement complète de c2 (tfevents, étapes 0 à 1355). Le résultat contredit l'hypothèse « oscillation de Goodhart » et localise précisément la cause.

**Ce que la télémétrie montre.** Le flicker n'est pas une oscillation: c'est une **rampe monotone non amortie des paramètres lents** de la mémoire. La dérive des poids qkv fait croître la RMS de la lecture d'un facteur 8 sur le run; la norme du gate (gatew) monte linéairement de 0.30 à 1.76 sans jamais atteindre de plateau, et accélère vers l'étape 900. L'amplitude d'injection croît donc continûment jusqu'à franchir, vers l'étape 1000, un seuil de cohérence au-delà duquel la dispersion temporelle d'amplitude (notre canal ampdisp_temporal) explose: c'est le flicker. Trois suspects sont **innocentés** par la même télémétrie: l'état à poids rapides lui-même (statenorm épinglé à 0.999 tout le run, la renormalisation de l'écriture Muon fonctionne, contrairement à notre première hypothèse), le critique DMD et les normes de gradient (plats, donc ni cycle limite ni instabilité du critique). Les facteurs facilitants sont tous identifiés dans le code: learning rate constant (aucun scheduler), EMA de poids désactivée en mode LoRA, weight decay faible, et le plafond d'injection (mem_stream_scale) jamais atteint (pic à 0.37 pour un seuil de 1.0).

**Confirmation causale à coût nul.** Si la cause est une sur-amplitude de l'injection, la brider à l'inférence doit réduire le dommage sans aucun réentraînement. La probe `MEM_STREAM_SCALE=0.5` sur le checkpoint 1000 (sink 1) donne:

| ckpt 1000, sink 1 | écart id_cos (tail 300) | SC memon | DD memon |
|---|---|---|---|
| injection pleine | -0.253 (0/6) | 0.958 | 0.911 |
| injection bridée à 0.5 | **-0.120** (0/6) | 0.974 | 0.944 |

Le dommage est divisé par deux par un simple facteur scalaire à l'inférence. Il n'est pas annulé (la dérive des poids lents est déjà « cuite » dans le checkpoint), ce qui indique que le correctif doit s'appliquer **pendant l'entraînement**.

**Portée du diagnostic.** C'est le résultat le plus important de cette phase: la dégénérescence de c2 à 1000+ n'est pas la manifestation inévitable du mur data-free, c'est une **pathologie d'optimisation identifiée et bornable** (rampe non amortie de paramètres lents sur-pilotant l'injection). Elle relève d'outils standards: plafonnement d'amplitude, schedulers, EMA, clipping. La littérature fournit d'ailleurs une preuve d'existence: VideoSSM entraîne une mémoire globale sous le même DMD fenêtré data-free, de façon stable, avec précisément ce type d'hygiène (mémoire contractive, co-entraînement complet, EMA et schedules). c2 prouve donc que le data-free **non stabilisé** dégénère, pas que le data-free est impossible.

## 4.7 c3_stable: le pack de stabilité (en cours)

**c3_stable** traduit l'autopsie en un correctif d'entraînement: c2 (eviction-feed conservé) plus un « pack stabilité à parité VideoSSM » dont chaque élément cible la rampe identifiée, et non l'état à poids rapides (déjà sous contrôle):

1. `mem_stream_scale 0.5`: plafond dur de l'injection à 0.5 fois la RMS locale du flux résiduel (borne l'amplitude quelle que soit la dérive des poids lents, le levier le plus directement causal);
2. `lr_mem` réduit de moitié (5e-5): réduit le moteur de la rampe;
3. `weight_decay 0.05`: rappel des poids qkv/proj;
4. EMA de poids réactivée (0.99, trainable-only, à partir de l'étape 250; elle était désactivée en mode LoRA dans c1/c2), l'évaluation chargeant obligatoirement l'ombre EMA;
5. décroissance cosinus des learning rates à partir de l'étape 400 (planchers 0.1x pour la LoRA, 0.05x pour la mémoire): supprime la rampe tardive;
6. clipping explicite des gradients mémoire à 1.0 (le clipping FSDP les ignorait, ils étaient donc non clippés dans c1 et c2);
7. annealing du cache-dropout de 0.3 vers 0.1 sur 800 étapes;
8. `use_decay` (oubli des poids rapides) activé en couverture, secondaire d'après la télémétrie.

Le run part d'un redémarrage propre (base.pt + final.pt, mémoire réinitialisée), cible 2000 étapes. La validation pré-lancement a suivi le même standard que c2: selftest en 8 phases (bit-exact pack désactivé, aller-retour de sauvegarde EMA exact), et une revue de code indépendante qui a intercepté un bug bloquant avant lancement (l'ombre EMA n'était jamais sauvegardée en mode LoRA, ce qui aurait rendu tout l'amortissement invisible à l'évaluation).

**État au moment de la rédaction:** c3_stable a été lancé le 15 juillet 2026 et est **en cours d'entraînement** (4 GPU, sondes de démarrage saines: gradients mémoire et gradients de decay non nuls, EMA trainable-only active). Aucun checkpoint n'a encore été évalué; le dossier `reference/runs/c3_stable/` ne contient pas encore d'évaluation. La trajectoire (250, 500, ..., 2000, en conditions normale et sink-starvation) est à compléter. La règle de décision est pré-enregistrée: si c3 tient « actif, dynamique et do-no-harm » jusqu'à 2000 étapes, c'est la démonstration recherchée que la dégénérescence de c2 était bien une pathologie d'optimisation corrigeable; si c3 dégénère malgré la totalité des stabilisateurs standards, c'est au contraire la meilleure preuve accumulée en faveur du mur data-free. Dans les deux cas, une limite reste acquise: la stabilisation amortit, elle ne crée pas d'objectif; un effet **positif** sur l'identité (au-delà du do-no-harm) exigerait un changement de cible d'entraînement (reconstruction sur vidéo réelle, ou témoin non factorisé de type « Anchor Resurrection »), discuté en section suivante.

## 4.8 Bilan de la campagne

| Run | Hypothèse testée | Résultat | Enseignement |
|---|---|---|---|
| e1 / e2 | injection sur backbone gelé | artefact DC puis dissolution; VBench aveugle | l'amplitude d'injection découplée du flux est structurellement dangereuse; durcir les métriques |
| e3 / e4 | garde-fous anti-dégénérescence | gel @500 (DD 0.996 vers 0.046, SC monte); e4 gèle à DD 0.0 | whack-a-mole structurel; triade de Goodhart (gel bat SC, DC bat la dérive, flicker bat DD); attracteur DMD général, pas du surapprentissage |
| c1_cotrain | co-entraînement DMD pur, propre | pas de gel; injection nocive @250 (-0.123) neutralisée @500 (-0.019), puis re-dégradation @1250; probes de sauvetage négatives | le co-entraînement absorbe l'injection mais do-no-harm = inerte; la mémoire écrit un contenu redondant, défaut architectural |
| c2_eviction | écrire le chunk évincé (complément du cache) | mouvement réparé, dommage divisé par deux et décroissant (250-750), puis flicker (1000+) | amélioration réelle mais instable; l'autopsie montre une rampe d'optimisation corrigeable, pas le mur |
| c3_stable | pack de stabilité ciblant la rampe | en cours | trajectoire à compléter |

Aucun run n'a atteint l'objectif initial (une mémoire qui **améliore** l'identité à long terme). Mais la campagne a produit trois résultats solides et vérifiés: une caractérisation précise des modes de dégénérescence du self-rollout DMD data-free et de la manière dont ils trompent les métriques standards (la triade de Goodhart, avec le protocole d'évaluation durci qui en découle); l'identification du défaut architectural de la mémoire initiale (redondance avec le cache) et sa correction mesurable par l'eviction-feed; et le diagnostic télémétrique complet de l'instabilité résiduelle, avec sa confirmation causale à coût nul et son correctif en cours de test. La frontière exacte entre « pathologie d'optimisation » et « mur de l'objectif data-free » est précisément ce que c3_stable est en train de mesurer.
