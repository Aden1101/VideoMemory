# REGISTRE PRÉ-ENREGISTRÉ : batterie arbitre post-débat (2026-07-28)

Écrit AVANT tout run. Les deltas = re-ID apparié vs memzero, 24 cellules, harness l1.
Références : oracle1 +0.224 ; plateau memdec12/20 +0.127/+0.129 ; SE par bras ≈ 0.05.

## P1 : ablation bloc contigu (toutes les écritures du segment A neutralisées, offline)
- Avocat TTT-pur : effet NET attendu ; sa ligne rouge dure n°1 = Δloss recon < 0.02 (alors
  le +0.127 ne vient pas des fast weights).
- Avocat du diable : ||Δétat||/||état|| 0.35-0.60 ; Δnorm_mse décodée +0.10 à +0.20 ;
  P(Δloss < 0.03) = 30% (scénario tueur du projet).
- Règle : Δ petit ⇒ STOP route pure + alerte projet. Δ net ⇒ fondation confirmée.

## THRUWRITER : vraie frame → écriture réelle → relecture jeune → splice (campagne)
- Avocat TTT-pur : +0.13 à +0.15 ⇒ le DOSAGE de l'écriture est la variable (P2/P3).
- Avocat du diable : +0.10 à +0.15, P(>+0.18) = 25% ; recon relue 0.55-0.70.
- Règle : ≈ +0.13 avec contenu parfait ⇒ le canal écriture+lecture est le goulot (modèle p
  mort). ≈ +0.20 ⇒ canal sain, le CONTENU est le levier ⇒ programme H2/règle d'écriture.

## TWIN/CROSS-SEED : vraie frame NETTE du mauvais sujet (même prompt, autre seed)
- Modèle p strict : ≈ 0 (le contenu faux ne paie rien).
- Règle : > +0.05 ⇒ le modèle "contenu = tout" est faux par construction ; une part du
  gain est générique (stabilisation par frame nette compatible scène).

## BLUR σ~24 (vraie frame floutée au niveau de fidélité de nos décodeurs)
- Avocat du diable : +0.04 à +0.14 selon calibration ; P(explique tout le plateau) ≈ 55%.

## MEMON20 : lecture active au retour + écrivain A2
- Les DEUX avocats : ≈ 0 (+0.01 [−0.03, +0.04]). Tourne pour fermer l'objection
  "jamais testé avec écrivain non-aveugle", pas comme gate.

## Distribution des 24 deltas (FAIT, 0$, 2026-07-28)
memdec20 : 7 cellules ~0 | 12 continues 0.03-0.18 | 5 jackpots ≥0.35 (jusqu'à 0.57).
memdec12 : forme identique. ⇒ ni bimodal strict ni nudge uniforme : mélange à queue de
jackpots ; le levier = taux de clic (p continu). L'écrivain 2.0 n'a pas changé la forme.

---
# ADDENDUM 2026-08-03 (pré-enregistré AVANT run)

## Audit poids (demande Aden : "on n'a pas changé de poids depuis le début ?")
- l1@2000-EMA (nos bras) : mtime 2026-07-18, sha256[:16]=64d704fb0e22948f : INCHANGÉ depuis
  avant la 1re campagne re-ID. base.pt : 2026-04-22 (import d'origine). lora.pt LongLive
  stock : 2026-07-17, sha256[:16]=c4e43b87d62d4b06 (audit HF antérieur : match). Aucun .pt
  modifié dans le repo (git). CONCLUSION : aucun poids n'a bougé. La différence visible dès
  le début d'inférence entre nos bras et stockoff = le LoRA d'entraînement mémoire (attendu,
  documenté : memzero 0.686 vs stockoff 0.847) + config rope. Hashes désormais enregistrés
  pour tout futur audit.

## Contrôles reviewer S2 (le trou identifié par risks.md, attaque n°1)
Bras à 24 cellules S2, dose 1.0, même routage que memdec20 :
- **noiseanchor** (ancre = bruit pur splicé à la récence) : PRÉDICTION +0.02 à +0.08
  (le gate 3 à n=3/dose 0.288 donnait 9% du gain ; à pleine dose le prompt repeint un peu
  mieux mais S2 saut-de-monde ne donne presque rien au texte seul).
- **wronganchor** (ancre décodée d'une AUTRE cellule) : PRÉDICTION +0.00 à +0.06, avec
  contaminations visibles du mauvais contenu.
Règle : si l'un des deux ≥ la moitié de +0.434, le claim "c'est le CONTENU de la mémoire"
tombe et le papier doit être reformulé (récence générique). Sinon, attaque n°1 parée.

### RÉSULTAT (2026-08-03, 24 cellules S2, re-scoring propre) — LES DEUX PRÉDICTIONS RÉFUTÉES
- **noiseanchor : +0.2899** (23+/1−, p = 3.0e-06) = **66.8 %** du +0.434, prédit +0.02 à +0.08.
- **wronganchor : +0.4588** (24+/0−, p = 1.2e-07) = **105.7 %**, prédit +0.00 à +0.06 ; et
  contamination du sujet SOURCE **nulle** (0.0426 contre 0.0426 pour memdec20, p = 0.84).
- Règle appliquée : les deux dépassent la moitié ⇒ **le claim « c'est le CONTENU de la mémoire »
  TOMBE** sur le tier cross-scène ; reformulation « récence générique » requise.
  Détail complet : `composite_gates_report.md` §11.

---
# PILOTE "TEXTE INSUFFISANT" (pré-enregistré AVANT le run, 2026-08-03, après le §11)

**La question.** Le §11 montre que sur le tier cross-scène le contenu splicé n'est pas lu. Mais
tous les prompts de retour y DÉCRIVENT le sujet en entier : le texte suffisait. Est-ce que le
canal lit le contenu quand le texte NE PEUT PAS repeindre le sujet ? Le bras `wronganchor` est le
juge : même tuyauterie, contenu faux. **La signature d'une lecture = `memdec20` > `wronganchor`.**

**Les 2 cellules** (forme S2p, 240 f, seed 0, prompts NEUFS vérifiés held-out) :
* **t17 « générique »** : A = *a horse with an unusual jet-black coat and a long snow-white mane*
  (anti-stéréotype : la corrélation apprise est robe claire → crinière sombre) ; retour = *the
  animal from earlier returns to the meadow* — **zéro attribut**.
* **t18 « ambiguë »** : A = chien BRUN dans un parc, B = chien BLANC dans LE MÊME parc ; retour =
  *the first dog comes back*. Le texte ne peut pas choisir : 2 alternatives, hasard = 50 %.

**Métriques** (toutes crop60 ET full, contre des références PROPRES) :
1. `reid` = cos(retour, **A\***) avec A\* = segment 0 du rollout **SS1** de la même cellule (A ouvre,
   prompt complet, rien avant lui).
2. **Le contraste clé** = `reid(memdec20) − reid(wronganchor)`, par cellule.
3. t18 seulement, **désambiguïsation** : `margin` = cos(retour, A\*) − cos(retour, **B\***) avec
   B\* = le segment 2 (chien blanc) du MÊME rollout, identique pour tous les bras.
4. t17 seulement, **attributs** : CLIP ViT-B/32, cos(retour, "a jet-black horse with a white
   mane") − cos(retour, "a brown horse with a dark mane"), + l'œil (la robe est-elle noire ?).

**PRÉDICTIONS.**

| | H_lu : « le contenu est lu quand le texte manque » | H_récence (ce que dit le §11) |
|---|---|---|
| t17 contraste memdec20 − wronganchor (crop60) | **≥ +0.10** | **|Δ| ≤ 0.05**, signe aléatoire |
| t17 attributs CLIP (memdec20) | > 0, robe noire + crinière blanche visibles | ≤ 0, cheval **stéréotypé** (robe claire ou brune), attributs perdus pour les DEUX bras |
| t18 margin A\*−B\* (memdec20) | **> +0.05** (le brun revient) | ≈ 0 ou **négatif** (biais vers le chien le plus récent = le blanc, ou chien générique) |
| t18 contraste memdec20 − wronganchor | **≥ +0.10** | **|Δ| ≤ 0.05** |
| absolu des 4 bras | — | scores **bas** partout (le prompt appauvri ne sait pas quoi peindre) |

**Règle de décision du pilote** (n = 2, donc c'est un GATE de continuation, pas une preuve) :
* les DEUX cellules montrent le contraste ≥ +0.10 dans le bon sens **et** l'œil confirme
  (robe noire / bon chien) ⇒ **régime trouvé**, investir : 12 cellules × 2 seeds sur ce tier ;
* aucune des deux ⇒ **le canal ne lit pas, même quand le texte manque** : le résultat négatif du
  §11 se généralise, et c'est la conclusion du papier ;
* une sur deux ⇒ zone grise, pas d'investissement avant une 3e cellule du même type que celle qui
  a marché.

**Confondant assumé et annoncé d'avance** : avec un prompt de retour appauvri, le modèle peut ne
pas ramener le bon TYPE de sujet du tout (l'absolu s'effondre pour tous les bras). C'est pourquoi
le juge est le CONTRASTE apparié memdec20 − wronganchor et non le niveau absolu.

### RÉSULTAT DU PILOTE (2026-08-03, 2 cellules, seed 0)
- **t17 « générique » : VOID par construction** — au retour les 5 bras rendent une **vache**
  (« the animal from earlier » perd l'ESPÈCE) ; aucun cheval à retrouver, la question des
  attributs est sans objet. Contraste memdec20 − wronganchor +0.031 crop60. Le confondant
  annoncé s'est réalisé. Leçon de design : appauvrir l'ATTRIBUT, jamais la CATÉGORIE.
- **t18 « ambiguë » : dans le bon sens, faiblement** — contraste **+0.1028 crop60** (seuil H_lu
  atteint) mais **+0.0144 full** ; à l'œil memdec20 ramène un labrador doré, wronganchor un
  labrador crème. La métrique `margin` est **void** (la référence propre ne sépare les 2 chiens
  que de 0.0070).
- **Verdict : ZONE GRISE** (1 cellule exploitable sur 2, une seule vue sur deux) ⇒ **pas
  d'investissement** ; d'abord réécrire les prompts (catégorie gardée) et changer le
  discriminateur de couleur, puis 3 cellules du type t18.
- **Trouvaille séparée (documentée §12.1)** : le **routage par mots meurt en régime générique** —
  0/2 rappels corrects, 1 faux rappel, et sur t17 l'argmax désigne l'OUVREUR. Argument direct
  pour un routage par contenu.

---
# MINI-ROUND CORRIGÉ « TEXTE INSUFFISANT » (scellé AVANT le run, 2026-08-03)

Verdict œil d'Aden sur le pilote : *« notre chien est LE plus proche du brun, mais aucun des deux
n'est un bon chien »* — signal directionnel, qualité médiocre. Les 3 correctifs pré-enregistrés
sont appliqués : (1) prompts de retour = **catégorie gardée, attribut retiré** ; (2) +3 cellules
du type t18 (ambiguïté même-catégorie, couleurs franches) ; (3) **juge couleur dédié**.

**5 cellules** (seed 0, forme S2p, retour au segment 3 frame 180, splice par vérité du planning
pour TOUS les bras) : **t19** (= t17 corrigé, *the horse comes back to the meadow*, prompts A/B/C
verbatim identiques : duplication ASSUMÉE pour la comparabilité), **t18** chien brun/blanc,
**t20** chat noir/blanc, **t21** pick-up rouge/bleu, **t22** perroquet jaune/vert.

**6 bras** : `memzero` | **`memdec21un`** = config-Aden (porte α : union+bruit en scène continue,
`memdec20` bit à bit en saut de monde) | `memdec20` = notre-pur | `wronganchor` | `noiseanchor` |
`framebank_gt` (QUAND = planning, sélection = argmax de la méthode ; décision du routeur loguée).

**Juge attribut** (le DINO ne sépare pas : +0.007 sur t18) : CLIP ViT-B/32,
`attr = cos(crop, "a <couleur A> <catégorie>") − cos(crop, "a <couleur B> <catégorie>")`, plus un
juge histogramme (distance couleur du crop aux DEUX références réelles). **Gate de validité** : le
juge doit classer correctement les références propres (A\* de chaque cellule + le segment B vu) ;
toute cellule où il échoue est déclarée VOID avant de regarder les bras.

**PRÉDICTIONS.**

| | H_lu (le contenu est lu quand le texte manque) | H_récence (le §11 se généralise) |
|---|---|---|
| **contraste `memdec21un` − `wronganchor`, attr CLIP** | **≥ +0.08 sur ≥ 4/5 cellules** | \|Δ\| < 0.08 partout, signe aléatoire |
| contraste `memdec20` − `wronganchor` (reid crop60) | ≥ +0.08 sur ≥ 3/5 | \|Δ\| < 0.05 |
| `noiseanchor` | nettement sous les deux bras décodés | ≈ `memdec20` (à ±0.05) |
| `framebank_gt` (vraie frame, sélection par mots) | ≥ nos bras sur l'attribut | rendu contrôlé par le prompt, attribut au hasard |
| t19 (cheval corrigé) | un CHEVAL revient, robe sombre + crinière claire | un cheval revient (catégorie sauvée) mais attributs **stéréotypés** |

**RÈGLE DE DÉCISION (scellée)** : contraste config-Aden − wronganchor **≥ +0.08 sur ≥ 4/5
cellules valides** ⇒ **le contenu est lu, territoire papier confirmé**, on investit (12×2). 2 ou 3
cellules sur 5 ⇒ zone grise, une 3e salve avant tout investissement. ≤ 1/5 ⇒ **le négatif du §11
est général** : le canal splice→rendu n'est pas gouverné par le contenu, et c'est LA conclusion.

### RÉSULTAT DU MINI-ROUND CORRIGÉ (2026-08-03, 5 cellules x 6 bras, seed 0)
- Gate du juge : **3/3 A\* bien classés** ; t18 et t21 **VOID** parce que le segment B rendu ne
  porte PAS son attribut (le « blue pickup » est ROUGE) — pathologie S2, pas défaut du juge.
- **Règle scellée : 0/2 cellules valides** (t20 −0.0203, t22 −0.0126 ; 0/4 avec les VOID) ⇒
  branche « ≤ 1/5 : le négatif du §11 est GÉNÉRAL ».
- Correctif de prompt validé (t19 : un CHEVAL revient partout, plus de vache) mais **attribut
  perdu** : nos bras rendent un cheval brun à crinière brune là où memzero et framebank_gt
  gardent la crinière claire. Sur t21 `wronganchor` rend le bon rouge, pas nous.
- `memdec21un` ≡ `memdec20` (porte α fermée sur ce tier) : la config définitive d'Aden n'est PAS
  jugée ici. Détail : `composite_gates_report.md` §13.

---
# MINI-BATTERIE TRAINING-FREE (scellée AVANT tout run, 2026-08-04)

**D'où elle vient.** Le §11 (`wronganchor` = 105.7 % du gain, contamination nulle) et le §13
(0/2 en régime texte-insuffisant) disent la même chose : *le canal splice → rendu n'est pas
gouverné par le CONTENU de ce qu'on splice*. Deux explications restent en concurrence et aucune
n'a été départagée :

* **E1 « l'ancre est illisible »** : notre ancre décodée est FLOUE (au-delà de σ = 32 px en
  espace DINO, phase B) ; le modèle ne lit que des frames NETTES (la banque Echo, elle, splice
  une vraie frame et son contenu EST lu : elle ramène le bon sujet). Le contenu ne serait donc
  pas absent de l'ancre, il serait sous le seuil de lecture.
* **E2 « le slot fait tout »** : ce qui compte est l'effacement du contexte au slot de récence
  (gate 2 : reculer l'ancre d'UN slot détruit 98 % du gain), et le prompt repeint A. Le contenu
  n'est jamais lu, net ou flou.

Et une objection reviewer n'a jamais été mesurée : **le correctif minimal**. En S2/SS2 l'ouvreur
X est épinglé dans le sink et domine tous les segments suivants (§13.1). Un reviewer dira : *«
votre +0.43 n'est que la levée de l'emprise de l'ouvreur ; rafraîchissez le sink au switch et
vous l'obtenez pour rien, sans mémoire »*.

**Lecture de code faite AVANT le run (à noter au rapport) :** avec `global_sink: true` (nos
configs ET `stock_official.yaml`), `_recache_after_switch` ne remet pas le cache à zéro et passe
`sink_recache_after_switch=False` ; les `sink_size` premiers slots (1 chez nous, **3** chez
LongLive stock) gardent donc les k/v de la frame 0 pendant TOUT le rollout, à travers tous les
switches. **LongLive ne rafraîchit jamais son sink au switch** — confirmé dans
`wan/modules/causal_model.py` (la protection `write_start_index = max(local_start_index,
sink_tokens)` et la branche `sink_recache_after_switch`).

## Les 3 bras (additifs, env-gated, défaut OFF)

| bras | ce qui change vs la référence | hook |
|---|---|---|
| `sdedit_ours` | l'ancre décodée de NOTRE état est **affûtée** avant splice : bruitage flow-matching à σ puis re-débruitage par le modèle de base en **mode 1 frame** (cache KV frais de 1 frame, `current_start=0`), conditionné au **prompt de RETOUR tel quel** | `MEM_SDEDIT=σ` |
| `sdedit_wrong` | **identique**, sur l'ancre décodée d'une AUTRE cellule (le contrôle décisif) | `MEM_SDEDIT=σ` + `MEM_DECODE_SRC` |
| `sinkpurge` | **aucun splice, aucune mémoire** (= `memzero`) ; au recache du retour les slots du sink épinglé sont réécrits avec la **dernière frame vraie courante** (mécanique `_write_sink_anchor`, déjà utilisée par `ORACLE_SPLICE_POS=sink0`) | `MEM_SINKPURGE=1` |

**6 cellules, seed 0** : `SS2` t12/t13/t14 (tier même décor, le seul où le score ne peut venir
que du sujet) + `S2` t2/t4/t11 (le tier des contrôles du §11, donc directement comparable à
`wronganchor`). Références déjà sur disque : `memzero`, `memdec20`, `wronganchor` (S2 ;
sur SS2 `wronganchor` est produit dans la même campagne, avec les dumps d'ancre qui lui manquent).
Scoring = `score_comp.py` : sujet crop60 vs **A\* propre** (SS1 seg0 pour SS2, S3 seg0 pour S2),
fond (complément du crop, retour vs queue de B), cut2, + les tripwires DD-ellipse et CLIP-texte.

**Choix de σ : pré-enregistré comme une décision D'ŒIL, pas de métrique.** Une planche
[ancre brute | affûtée à plusieurs σ] est rendue AVANT tout rollout. Règle écrite d'avance :
on retient **le σ le plus BAS qui rend l'ancre visiblement nette en gardant le sujet
reconnaissable** ; si tous les σ testés remplacent le sujet par une re-génération du prompt
(« veste turquoise »), la batterie est **RE-CATÉGORISÉE** (elle ne teste plus E1, elle mesure un
beau perturbateur) — c'est noté comme tel et un seul σ est exécuté.

## PRÉDICTIONS

| mesure | **H_lecture** (E1 : l'ancre est illisible parce que floue) | **H_slot** (E2 : le §11 se généralise) |
|---|---|---|
| **contraste `sdedit_ours` − `sdedit_wrong`, sujet crop60** (LE chiffre) | **≥ +0.08 sur ≥ 4/6 cellules** | **\|Δ\| < 0.05**, signe aléatoire, ≤ 1/6 cellule au-dessus de +0.08 |
| `sdedit_ours` vs `memdec20` (sujet crop60) | **> +0.08** (l'affûtage débloque) | ±0.05, indiscernable |
| `sdedit_wrong` vs `wronganchor` | ±0.05 | ±0.05 |
| contamination : `sdedit_wrong` vs le A\* du sujet SOURCE | visible (> +0.05 d'excès) | nulle, comme `wronganchor` (−0.000) |

**Chiffres absolus attendus** (SS2 t12/13/14, sujet crop60 : `memzero` 0.265, `memdec20` 0.689) :
`sdedit_ours` **dans [0.60, 0.75]** sous H_slot ; **> 0.77** sous H_lecture.

**Bénéfice collatéral attendu même sous H_slot** (pré-enregistré pour ne pas être raconté après
coup) : une ancre affûtée est plus proche de la variété latente réelle, donc le raccord doit
être **moins brutal** — `cut2` de `sdedit_ours` prédit **entre 8 et 14** contre 17.5 pour
`memdec20` (SS2 5 cellules), fond ± 0.05. Si c'est le seul effet, c'est un résultat de confort,
pas un résultat de mémoire, et il sera écrit comme tel.

### `sinkpurge` — la prédiction la plus incertaine des trois
* **borne basse (H_slot pure)** : le sink ne vaut qu'un slot sur 13, le retirer ne change presque
  rien : delta vs `memzero` **+0.02 à +0.05**.
* **borne haute (H_ouvreur)** : l'emprise de l'ouvreur est LA cause du score bas de `memzero`
  (§13.1 : le B rendu ne porte même pas son attribut), la purger rend l'essentiel du gain :
  delta **≥ +0.21** = la moitié du +0.424 de `memdec20`.
* **ma prédiction ponctuelle : +0.05 à +0.20** (soit 12-47 % du gain), absolu SS2 dans
  **[0.32, 0.47]**. `cut2` ≤ 2× celui de `memzero` (aucun splice, donc aucun saut visuel).
* **tripwire de stabilité** (le sink existe pour la stabilité, StreamingLLM) : si `dd_ell`
  s'effondre, si `cut2` explose ou si `clip_text` chute sous `memzero`, la purge DÉSTABILISE et
  c'est ça le résultat.

**RÈGLES DE DÉCISION (scellées)**
1. `sdedit_ours` − `sdedit_wrong` **≥ +0.08 sur ≥ 4/6** ⇒ **E1 confirmée** : l'ancre floue était
   le goulot, le contenu EST lisible une fois net ⇒ territoire papier rouvert, on investit
   (affûtage co-entraîné, 12 cellules × 2 seeds). 2 ou 3 sur 6 ⇒ zone grise, 3e salve avant tout
   investissement. **≤ 1/6 ⇒ E1 est ÉLIMINÉE** : le contenu n'est pas lu même NET, et la
   conclusion négative du §11/§13 devient robuste à l'objection « vos ancres sont floues ».
2. `sinkpurge` **≥ +0.21** (moitié du gain) ⇒ **l'objection du correctif minimal est FONDÉE** :
   `sinkpurge` doit devenir une ligne de baseline du papier et notre +0.43 doit être annoncé
   *au-dessus de ce plancher*, pas au-dessus de `memzero`. **< +0.05** ⇒ objection parée.
   Entre les deux ⇒ à reporter honnêtement comme plancher partiel.
3. Toute cellule où la planche d'œil déclare le contenu détruit par l'affûtage est comptée
   **VOID pour la règle 1** (et le dit avant de regarder les scores).

**Confondant assumé et annoncé d'avance** : `sdedit_*` fait passer l'ancre par le PRIOR du modèle
conditionné au prompt de retour ; à σ élevé la sortie tend mécaniquement vers « une frame
générée par le prompt », c'est-à-dire vers `noiseanchor`. Le contraste apparié
`sdedit_ours` − `sdedit_wrong` est immunisé (les deux subissent le même prior), mais les niveaux
absolus, eux, ne le sont pas. C'est pourquoi le juge est le CONTRASTE.

---
# LOT OFFENSIF DE LA NUIT (scellé AVANT tout run, 2026-08-04)

**D'où il vient.** §11 (`wronganchor` = 105.7 % du gain, contamination nulle), §13 (0/2 en régime
texte-insuffisant) et la mini-batterie training-free disent la même chose : *le canal splice →
rendu n'est pas gouverné par le CONTENU*. Trois leviers du panel sont testés ici pour faire LIRE
la mémoire, chacun avec son **pré-écran** et son **jumeau de contrôle**. Règle transverse
(inchangée) : **le comparateur de CHAQUE bras est son propre wronganchor / jumeau, JAMAIS memzero**
; seuil **≥ +0.08 sur ≥ 4/5 (ou 4/6) cellules valides** ; pour la légende, le jumeau FAUX doit en
plus être **NÉGATIF**.

## T0 — corrélation bande-basse (0 GPU, sur les mp4 `wronganchor` S2 déjà sur disque)

**La question.** §11.5 mesure la contamination en espace DINO et trouve 0.0426 vs 0.0426. Mais
DINO est un descripteur sémantique : si l'ancre étrangère est copiée dans une **bande basse**
(layout 8×8, luminance, disposition des masses), « pas lu » deviendrait « lu dans une bande que le
juge ignore ». Test **apparié 2×2** par cellule t (24 cellules S2, seed 0 et 1) :

| | ancre PROPRE de t | ancre ÉTRANGÈRE (celle de t+1, réellement splicée dans `wronganchor`) |
|---|---|---|
| retour de `memdec20` | corr **attendue haute** si lu | corr de contrôle |
| retour de `wronganchor` | corr de contrôle | corr **attendue haute** si lu |

`excès = [corr(ret_wrong, anc_étrangère) − corr(ret_wrong, anc_propre)] + [corr(ret_dec,
anc_propre) − corr(ret_dec, anc_étrangère)]`, en corrélation de Pearson sur l'image 8×8 en niveaux
de gris (et 8×8×3 couleur en secondaire), sur DEUX fenêtres : **early** (les 12 premières frames
latentes après le splice, là où une copie serait visible) et **scored** (la fenêtre de score).

**PRÉDICTION** (H_slot, ce que dit le §11) : excès moyen **dans [−0.02, +0.05]**, signes
**≤ 14/24**, sur les deux fenêtres. **PRÉDICTION alternative** (H_bande-basse) : excès ≥ +0.10
avec ≥ 18/24 signes sur la fenêtre *early*.
**RÈGLE** : excès ≥ +0.10 ET ≥ 18/24 sur *early* ⇒ « pas lu » est **REQUALIFIÉ** en « lu dans une
bande invisible à DINO », et le papier doit le dire. Sinon le négatif du §11 tient tel quel.

## LEVIER 1 — LÉGENDE-VLM (L1). Pré-écran D'ABORD, à ~0 rollout

**Pré-écran (le gate).** Les 24 ancres décodées S2 sont déjà sur disque
(`reid_scen/dec_anchors/*.pt`). On les passe au VAE, on mesure la **couleur du sujet** (crop
central 60 %, juge couleur de `score_textpoor2.py` : histogramme teinte pondéré S·V + moyennes
S/V) et on compare l'accord avec le **A\* propre** de la cellule (S3 seg 0) à l'accord de l'ancre
**ÉTRANGÈRE** (celle de t+1, = ce que splice `wronganchor`) avec le même A\*.
`avantage = d(anc_étrangère, A*) − d(anc_propre, A*)` (> 0 = notre ancre est plus proche).

**PRÉDICTION** : le §11.7 et le diagnostic informationnel de l'ultraplan affirment que *« l'ancre
conserve exactement couleur + layout (bande basse) »*. Si c'est vrai, l'avantage est **positif sur
16-19 des 24 cellules**, avantage moyen **≥ +0.05**. Mon pari personnel, après le §13.4 (nos bras
rendent un cheval brun là où A\* est noir, un pick-up gris là où A\* est rouge) : **positif mais
plus faible**, 14-17/24.
**RÈGLE SCELLÉE** : avantage moyen ≤ 0 **ou** ≤ 12/24 signes ⇒ **LEVIER 1 MORT**, noté et fermé,
aucun rollout. Sinon on tire les 4 bras.

**Les 4 bras** (8 cellules : S2 t0/t2/t3/t4/t11 + SS2 t12/t13/t14, seed 0). Aucune modification du
pipeline : la mémoire est **muette** (`memzero`) dans les 4, seul le **prompt de RETOUR** change
(scénarios `S2c` / `SS2c` = la forme S2/SS2 dont le segment de retour pointe l'entrée 3 de la liste
de prompts, dont le texte est *prompt de A verbatim + suffixe de 14 mots*) :

| bras | suffixe = légende contrainte construite sur… | rôle |
|---|---|---|
| `caption_ours` | **notre ancre décodée** (catégorie ← prompt, couleurs ← mesurées sur le crop) | le bras |
| `caption_wrong` | l'ancre décodée d'une **AUTRE cellule** (même catégorie, couleurs fausses) | **LE contrôle directionnel : delta NÉGATIF exigé** |
| `padded` | adjectifs **neutres**, exactement le même nombre de mots | contrôle de longueur de prompt |
| `framebank_vlm` | une **VRAIE frame** de la 1re apparition de A | plafond |

Anti-affabulation : **pas de VLM libre en v1**. La catégorie vient du prompt (elle est connue), les
couleurs sont MESURÉES (nommage par plus proche voisin dans une palette de 12 termes). Selftest
gratuit : suffixe vide ⇒ prompts[3] ≡ prompts[0] ⇒ le rollout doit être **bit à bit** le
`memzero` S2 déjà sur disque.

**PRÉDICTIONS** | H_texte (le texte est le canal prouvé, la légende le porte) : `caption_ours −
caption_wrong` **≥ +0.08 sur ≥ 4/5 cellules valides**, ET `caption_wrong − padded` **< 0**, ET
`framebank_vlm ≥ caption_ours`. | H_null : |Δ| < 0.05, signe aléatoire, et les 4 bras à ±0.05 de
`memzero`.
**Mon pari ponctuel** : `caption_ours − caption_wrong` = **+0.02 à +0.09**, 2-3 cellules sur 8
au-dessus de +0.08 ⇒ **zone grise**, avec `framebank_vlm` nettement au-dessus des deux (le vrai
contenu se légende bien, le nôtre non). La probabilité que le jumeau faux sorte NÉGATIF comme
exigé : ~45 %.

## LEVIER 2 — I1 « ANCRE DÉCLARÉE BRUITÉE » (~10 lignes)

**Le défaut nommé.** L'ancre est splicée en **déclarant t = 0** (`context_timestep = ones *
context_noise`, et `context_noise = 0` dans toutes nos configs) alors qu'elle est floue,
hors-variété. Le modèle la lit comme *une frame PROPRE de mon passé immédiat* et la RMSNorm +
le contexte B l'absorbent. **Correctif** : `splice = add_noise(ancre, t*)` **ET** le
`context_timestep` de CE SLOT = t\*. Le champ est déjà **par frame** (`e0` de forme (B, F, 6, C),
modulation par frame) : rien d'exotique, la métadonnée cesse simplement de mentir.

`MEM_ANCHOR_T = t*` ∈ {250, 500} (σ ≈ 0.25 / 0.50 sur la grille FlowMatch du pipeline), additif,
défaut OFF, refusé par `inference()` legacy.
**Bras** : `i1_ours250 / i1_wrong250 / i1_ours500 / i1_wrong500`, **6 cellules** (SS2 t12/13/14 +
S2 t2/4/11, seed 0). `i1_wrong*` = **exactement** le même traitement sur l'ancre d'une AUTRE
cellule = `wronganchor` + la déclaration.

**PRÉDICTIONS** | H_impédance (le mensonge de métadonnée était le goulot) : `i1_ours − i1_wrong`
**≥ +0.08 sur ≥ 4/6**, et `i1_ours` > `memdec20` en absolu. | H_slot (le §11 se généralise) :
**|Δ| < 0.05**, signe aléatoire, ≤ 1/6 cellule au-dessus de +0.08.
**Mon pari ponctuel** : **H_slot**, contraste dans **[−0.03, +0.05]** ; et un effet de NIVEAU
attendu dans les deux hypothèses — déclarer t\* = 500 retire la moitié du peu de haute fréquence
de l'ancre, donc l'absolu des bras i1 devrait **baisser** de 0.02 à 0.10 vs `memdec20`, davantage à
500 qu'à 250. Tripwire mécanique exigé avant tout score : le slot d'ancre doit être le SEUL dont le
timestep déclaré diffère (assert), et l'écart k/v du recache doit être non nul.

## LEVIER 3 — I3/T5 ADDITIF À Δ = 1 (+ T1 masse par source)

**Ce qu'on sait déjà** (§3) : le canal additif place l'ancre en bande RoPE **9** (auto = une frame
avant la fenêtre, requête à 18, donc **Δ = 9**) et il est MORT : +3 % du chemin sur S2, +35 % à
dose forcée gain 8 sur SS2, en **ne détruisant rien** (le seul bras du bon quadrant). La question
restante est de POSITION, pas de dose : à Δ = 9 l'ancre est loin ; **`MEM_ADD_POS=17` la met à
Δ = 1 de la requête**, dans la bande de récence — la position dont le gate 2 a montré qu'elle vaut
98 % de l'effet quand on y splice.
**Bras** : `oracle1add1p17` / `oracle1add8p17` (contenu = **VRAIE frame** d'ancre = borne du
canal), **4 cellules** SS2 t12/13/14/15, seed 0. **Puis, seulement si la borne s'ouvre**,
`memdec20add8p17` (notre ancre décodée dans le même canal).
**T1, masse par source** : `_additive_attention` rend déjà `alpha` = masse de l'ancre ; on ajoute
la décomposition **exacte** `[sink | fenêtre | ancre]` par deux log-sum-exp supplémentaires
(LSE(concat) = logaddexp(LSE_sink, LSE_win), identité exacte), loguée **par bloc et par pas**.

**PRÉDICTIONS** | masse : à gain 1, la masse de l'ancre à Δ = 1 est **≥ 2×** les 2.04 % mesurés à
Δ = 9 (RoPE favorise les positions proches) — je prédis **4 à 9 %** ; à gain 8, **25 à 45 %**.
| score (sujet crop60 SS2, chemin `memzero` 0.2648 → `oracle1` 0.3764, moitié = **+0.0558**) :
gain 8 à Δ = 1 rend **+0.04 à +0.11**, soit 35-100 % du chemin, **2/4 cellules** au-dessus de la
moitié. | dégâts : à Δ = 1 le canal cesse d'être inoffensif — je prédis `cut2` qui monte de 0.73
vers **2 à 6** et le fond qui descend de 0.925 vers **0.86-0.91** (l'ancre commence à disputer le
présent déclaré). Si la masse monte comme prédit MAIS le score ne bouge pas, la conclusion
« illisible, pas non-lu » devient beaucoup plus forte qu'au §3.
**RÈGLE SCELLÉE** : `oracle1add8p17 − memzero` ≥ **+0.0558** (la moitié du chemin) sur **≥ 3/4**
cellules ⇒ **le canal s'OUVRE**, on y met notre ancre décodée (`memdec20add8p17`) et le contraste
contre `wronganchor` redevient mesurable **sans éviction**. < +0.0558 sur ≥ 2/4 ⇒ la position
n'était pas le goulot non plus, et le canal additif est clos définitivement.

### AMENDEMENT (2026-08-04, écrit APRÈS la planche d'œil, AVANT tout rollout scoré)

La planche a fait deux choses, dans l'ordre :

**1. Elle a attrapé un bug de ma première implémentation, par le témoin σ = 1.0.** Le σ = 1.0 est
sur la planche comme plancher dégénéré (plus une trace d'ancre) : c'est EXACTEMENT le sampler
4 pas du modèle partant du bruit pur, il DOIT donc rendre une image propre. Il rendait du bruit.
Cause : je débruitais un bloc d'**UNE** frame latente, alors que le backbone est entraîné et mis
en cache avec `num_frame_per_block = 3` (les frames d'un bloc s'attentionnent entre elles). Un
bloc de 1 frame est hors distribution. Corrigé : l'ancre est **répliquée à la taille de bloc
native** et on relit la frame du milieu (`MEM_SDEDIT_FRAMES`, défaut = `num_frame_per_block` ;
`=1` restaure la variante cassée comme contrôle). Après correction, σ = 1.0 rend bien l'image du
prompt de retour sur les 6 cellules. **Leçon de méthode : le témoin dégénéré a servi à valider le
mécanisme, pas à mesurer un effet — c'est pour ça qu'il était sur la planche.**

**2. Elle a montré que « σ ≈ 0.4 et 0.6 » n'est PAS RÉALISABLE sur ce backbone.** Le générateur
est distillé DMD en 4 pas : sa grille de timesteps (shift 5) est ~[1000, 937.5, 833.3, **625**].
Un appel à t = 400, 500 ou 600 est **sous la grille**, à un bruit où le modèle n'a jamais été
entraîné : la prédiction de flux est mal calibrée et `x0 = x_t − σ_t·flow` sort une **mosaïque
haute fréquence**, nette mais vide de sens (visible sur les 6 cellules, colonnes σ 0.4/0.5/0.6).
Ce n'est pas un réglage à trouver, c'est une **contrainte du substrat** : *la force SDEdit est
quantifiée sur la grille du sampler, et la plus faible force disponible est σ = 0.625*.
Amendement du choix de σ : la règle scellée (« le σ le plus BAS qui affûte en gardant le sujet
reconnaissable ») est appliquée **sur les points de grille réalisables** {0.625, 0.833}, jugés à
l'œil ; le reste (R1, R2, les prédictions chiffrées) est **inchangé**.

**3. Et elle a rendu visible une chose que le dossier n'avait jamais regardée en face : l'ANCRE
BRUTE NE CONTIENT AUCUN SUJET.** Colonne « ancre BRUTE », 6/6 cellules : ce n'est pas une version
floue de A, c'est une **tache de couleur** (t12 = un smear sombre à dominante rouge, pas d'homme,
pas de table, pas de tasse ; t13 = du vert, pas de chien ; t11 = un smear jaune, pas de théière).
Aucune campagne ne l'avait mesuré parce que TOUTES mesuraient la vidéo RENDUE, donc aucune ne
pouvait séparer « le canal splice→rendu est aveugle au contenu » de « il n'y a pas de contenu ».
**Sonde ajoutée (pré-enregistrée ici) : `probe_anchor_identity.py`** — l'ancre est VAE-décodée en
pixels et comparée en DINO à deux références propres, le A\* de SA cellule et le A\* de la cellule
SOURCE (celle dont `wronganchor` vole l'ancre).
* **PRÉDICTION** : marge (SON A\* − A\* SOURCE) **≤ +0.03, ≤ 4/6 cellules positives** ⇒ l'ancre ne
  porte aucune identité d'instance, et le négatif du §11 est expliqué EN AMONT du canal.
* Si au contraire la marge est **≥ +0.10 sur ≥ 5/6**, l'identité EST dans l'ancre et c'est bien
  le canal qui ne la lit pas — ce qui rouvrirait la piste « lecture ».
* La même mesure est refaite sur les ancres affûtées : l'affûtage ne peut pas créer une identité
  absente, et il faut le montrer plutôt que l'affirmer.

### RÉSULTAT DE LA SONDE D'IDENTITÉ (2026-08-04) — **PRÉDICTION CONFIRMÉE, ET AU-DELÀ**

DINO crop60 de l'ancre VAE-décodée contre deux A\* propres, 6 cellules :

| variante de l'ancre splicée | vs SON A\* | vs A\* SOURCE | marge | cellules + |
|---|---|---|---|---|
| **brute (ce que `memdec20` splice)** | **0.0023** | −0.0154 | **+0.018** | 4/6 |
| affûtée σ 0.625 (mosaïque, hors grille) | 0.0516 | 0.0126 | +0.039 | 3/6 |
| affûtée σ 0.7 | 0.3078 | 0.0347 | +0.273 | 6/6 |
| affûtée σ 0.833 (le σ retenu) | 0.5385 | 0.0308 | +0.508 | 6/6 |
| **σ 1.0 = PROMPT SEUL, zéro ancre** | **0.7770** | 0.0692 | +0.707 | 6/6 |

* **Prédiction (marge ≤ +0.03, ≤ 4/6 positives) : ATTEINTE sur l'ancre brute** (+0.018, 4/6).
  L'ancre décodée est à **cosinus 0.002 du sujet qu'elle est censée encoder** : en espace DINO
  elle lui est ORTHOGONALE. Ce n'est pas une identité dégradée, c'est **aucune identité**.
* **Le contrôle qui rend la mesure incontestable** : la ligne σ 1.0 ne contient **aucune**
  information d'ancre (génération pure par le prompt) et score **0.777**. La métrique sait donc
  parfaitement voir le sujet — c'est l'ancre qui n'en contient pas.
* **Et la lecture qui compte, l'échelle est MONOTONE DÉCROISSANTE** : 0.777 (prompt seul) >
  0.539 (σ0.833) > 0.308 (σ0.7) > 0.052 (σ0.625) > 0.002 (ancre brute). *Plus on garde d'ancre,
  MOINS la frame ressemble à A.* Entre les deux points propres et sur-grille (σ0.833 vs σ1.0),
  tous deux visuellement nets, injecter notre ancre coûte **−0.24 d'identité** par rapport à
  n'utiliser que le prompt. **L'ancre décodée n'est pas un signal faible : c'est un PERTURBATEUR
  actif.** À l'œil : sur t2 le A\* est un chat tigré ORANGE, σ1.0 rend un chat tigré orange,
  σ0.833 rend un chat NOIR sous un tissu rouge (la tache rouge de l'ancre) ; sur t12 la veste
  rouge devient magenta.
* **Conséquence pour le dossier** : le §11 (`wronganchor` = 106 %, contamination nulle) n'a plus
  besoin d'une explication sur le canal. Elle est EN AMONT : il n'y a rien à lire. Une ancre
  fausse rend 100 % du gain d'une ancre vraie parce que les deux sont, en contenu d'instance,
  exactement équivalentes — nulles.

### RÉSULTATS DU LOT OFFENSIF (2026-08-04)

**T0 — prédiction RÉFUTÉE, dans le sens qui requalifie le §11.** Excès apparié de corrélation
bande-basse **+0.827** (23+/1−, p = 1.2e-06) en fenêtre *early*, **+0.663** (22+/2−) en fenêtre
*scored* ; prédit « [−0.02, +0.05], ≤ 14/24 ». La règle scellée (≥ +0.10 et ≥ 18/24) est
franchie d'un facteur 8 ⇒ **« pas lu » est REQUALIFIÉ** : le contenu splicé EST lu et copié
pendant ≈ 3 s. Détail + contrôles : `composite_gates_report.md` §16.1.

**L1 pré-écran — VIVANT d'une cellule, au niveau du hasard.** Avantage couleur +0.128,
**13+/11−**, p = 0.84 (prédit 16-19/24, pari personnel 14-17/24 : les deux au-dessus de la
mesure). La règle scellée dit « ≤ 12/24 ⇒ mort » : à 13/24 les 4 bras sont tirés, en notant que
la charge utile est au mieux faible. Version discrète (terme de couleur du cœur vs vraie frame) :
notre ancre 3/8, ancre étrangère 2/8.

**ANALYSES POST-HOC (déclarées comme telles, engendrées par T0, aucune n'était pré-enregistrée)**
— elles ne peuvent pas servir de gate, seulement de direction :
1. contrôle nul (ancre d'une 3e cellule, reçue par personne) : +0.10 contre +0.47/+0.52 pour
   l'ancre reçue ⇒ la copie est SPÉCIFIQUE ;
2. échelle de bande n = 4 → 64 : la copie ne meurt pas (+0.88 → +0.73) ⇒ ce n'est pas « une
   bande basse », c'est toute la structure mesurable ;
3. centre vs bord : l'écart reçue/non-reçue est de +0.30 dans les DEUX zones, et au centre
   l'ancre non reçue tombe à −0.02 ⇒ le canal transporte la place et la forme du SUJET ;
4. **la copie ne prédit PAS le re-ID** : contraste de copie vs contraste de re-ID, Pearson
   −0.075, Spearman −0.163 (n = 24) ⇒ élargir le canal ne paie pas ;
5. **l'ancre décodée ne porte pas l'identité** : DINO(ancre propre, A\*) 0.0110 contre
   DINO(ancre étrangère, A\*) 0.0103, avantage **+0.0007**, 11+/13−, p = 0.84 ; ni sa structure
   grossière (avantage +0.014, 13+/11−, p = 0.84).
⇒ direction (à re-tester proprement, pas un gate) : **le goulot est le CONTENU écrit/décodé, pas
le canal** ; les trois leviers de la nuit travaillent tous sur le canal.

---
# LEVIER 4 — DÉCODER LE **SNAPSHOT DE FIN DE SEGMENT A** (scellé AVANT run, 2026-08-04, nuit)

**D'où il vient.** 16.1 quater : notre ancre est une image de l'INTERLUDE (bande basse : vs A\*
+0.042, vs B +0.298, vs X +0.344 ; B plus proche que A sur 21/24, p = 3e-4). Le décodeur
reconstruit *la frame qui vient d'être évincée* ; au retour, c'est B. **Correctif** : décoder
l'état **snapshotté à la FIN DU SEGMENT DE A**, où le contenu le plus récemment évincé EST A.

**Aucun code nouveau.** `MEM_EPISODIC=1` snapshotte déjà l'état à la fin de la première
apparition d'un sujet qui sera revisité (`inference_schedule`) et le RESTAURE au revisit
(`_switch_recache`, dans le bloc `is_revisit`, **avant** le décodage), et `_memdec_frame` lit
l'état vivant. Sous `MEM_ZERO_ALL=1` l'injection mémoire est mise à zéro dans les deux passes
(`causal_model.py`), donc restaurer l'état **ne peut rien changer au rendu** : son seul
consommateur est le décodeur. D'où le bras `memdec_snap` = `MEM_ZERO_ALL=1 + MEM_EPISODIC=1`,
strictement `memdec20` avec un autre ÉTAT en entrée du décodeur.

**PRÉ-ÉCRAN (le gate, sur l'ancre dumpée, métrique T0)** — pilote 2 cellules S2 (t2 = chat/camion
rouge, t3 = homme en veste bleue/flamants roses : la cellule la plus diagnostique de la planche) :

| mesure (corrélation bande-basse 8×8) | ancre COURANTE (mesurée) | ancre SNAPSHOT, prédiction |
|---|---|---|
| vs **A\*** (le sujet à rappeler) | **+0.042** | **+0.20 à +0.35** |
| vs **B** (l'interlude) | +0.298 | **< +0.15**, et **< la valeur vs A\*** |
| vs **X** (l'ouvreur épinglé) | +0.344 | **reste haut, +0.20 à +0.35** (l'ouvreur est dans l'état dès la frame 0 : le snapshot ne l'enlève pas) |

**RÈGLE SCELLÉE du pré-écran** : `corr(ancre_snap, A*) ≥ +0.15` **ET** `> corr(ancre_snap, B)` sur
**2/2** cellules du pilote ⇒ **le colis est réparé**, on tire les 4 cellules restantes (t0, t4,
t6, t11) et le bras complet. 1/2 ⇒ zone grise, une 3e cellule. 0/2 ⇒ le snapshot ne répare pas le
contenu et le problème est dans **l'écriture** (l'état lui-même), pas dans le choix de l'instant.

**PRÉDICTION SUR LE RE-ID** (6 cellules, comparateurs déjà sur disque : `memdec20`,
`wronganchor`) | H_colis-réparé : `memdec_snap − wronganchor` **≥ +0.08 sur ≥ 4/6** (crop60), et
`memdec_snap > memdec20`. | H_canal-aveugle-à-l'identité (ce que dit 16.1 bis : la quantité copiée
ne prédit pas le score) : **|Δ| < 0.05**, signe aléatoire, même avec un colis correct.
**Mon pari ponctuel, honnête** : le pré-écran passe (le snapshot EST le bon instant) mais le re-ID
ne suit qu'à moitié — contraste **+0.02 à +0.08**, 2/6 cellules au-dessus du seuil — parce que
l'ouvreur X reste dans l'état et que l'ancre décodée reste floue (σ > 32 px, §12 du dossier).
**Confondant assumé** : si le pré-écran passe ET le re-ID ne suit pas, la conclusion n'est PAS
« la mémoire ne sert à rien » mais « le colis est au bon instant et reste illisible » — c'est-à-
dire le programme décodeur/fidélité, distinct du programme écriture.

### GATE A (lot défensif, 2026-08-04, tier MÊME DÉCOR SS2, 5 cellules seed 0)
- **Règle scellée (≥ +0.08 sur ≥ 4/5) : 0/4 → PRÉMISSE FALSIFIÉE, finetune NON autorisé.**
  `memdec20 − wronganchor` = **−0.0020** crop60 (2+/2−) / −0.0153 full ; `wronganchor − memzero`
  = **+0.3538** contre +0.3208 pour nous (l'ancre d'une AUTRE vidéo fait mieux) ; `noiseanchor`
  à +0.011 de nous. t16 manquant (OOM de co-run) : sans effet, le max atteignable serait 1/5.
- 3e mesure indépendante convergente (§11 cross-scène, §13 texte-insuffisant, §15 même-décor) :
  le gain (+0.32 sur memzero, 5/5) est réel mais vient de l'EFFACEMENT au slot de récence.

### RÉSULTAT DE LA MINI-BATTERIE (2026-08-04) — les 3 bras tournés, 23 rollouts audités

| règle scellée | mesuré | verdict |
|---|---|---|
| **R1** contraste `sdedit_ours` − `sdedit_wrong` ≥ +0.08 sur ≥ 4/6 | **+0.0221, 3+/3−, sign-p 1.000, sd 0.142, 2/6 cellules ≥ +0.08.** MAIS le témoin `sdedit_wrong` − `wronganchor` (MÊME ancre fausse, seul l'affûtage change ⇒ différence de contenu NULLE) fait **3/6** avec sd 0.197 : le plancher de bruit par cellule vaut 2-2,5× le seuil. | **R1 VOID par sous-puissance** (il fallait 2-3 graines/cellule) — notre contraste fait MOINS que son propre témoin sans contenu ; valeur centrale (+0.022 ± 0.058) pile dans la bande H_slot. **E1 reste ÉLIMINÉE** par les 2 autres routes (ancre : cos 0.002 ; rollout : `ours` − `memdec20` = +0.017 n.s.). |
| **R2** `sinkpurge` − `memzero` ≥ +0.21 ⇒ objection fondée ; < +0.05 ⇒ parée | **+0.1189** (5+/1−, 6 cellules, sd 0.078, 5/6 au-dessus de +0.08 et 0/6 en dessous de −0.08) ; **+0.1462** sur les 3 cellules SS2 = **34 %** du gain de `memdec20` | **plancher PARTIEL, à reporter** — et gratuit sur fond/raccord/mouvement/texte, donc plus gênant que le chiffre seul |
| prédiction collatérale : `cut2` de `sdedit_ours` entre 8 et 14 (vs 17.5 pour `memdec20`) | **17.5** (et le fond se dégrade 0.459 → 0.307) | **RATÉE** |
| σ ≈ 0.4 / 0.6 demandés | **non réalisables** : la force SDEdit est quantifiée sur la grille du sampler distillé (min 0.625) | contrainte de substrat, documentée |

Détail complet : `composite_gates_report.md` §14.

### RÉSULTAT DU LEVIER 4 (2026-08-04, pilote 2 cellules) — **0/2, règle scellée appliquée**
- Pré-écran : SNAPSHOT vs A\* **+0.027** | vs B +0.431 | vs X +0.565 (courante : +0.039 / +0.350 /
  +0.488). Gain apparié sur A\* **−0.012**. t3 (la cellule diagnostique) : vs A\* **−0.268**.
- Prédiction scellée par ligne : vs A\* « +0.20 à +0.35 » → +0.323 sur t2 (**dans la bande**) mais
  −0.268 sur t3 ; vs B « < +0.15 et < vs A\* » → **+0.431, RÉFUTÉ** ; vs X « +0.20 à +0.35 » →
  +0.565, au-dessus.
- Re-ID (t2) : `memdec_snap − wronganchor` −0.026 crop60 ; `− memdec20` +0.012. Rien ne bouge.
- **Branche atteinte : « le problème est dans l'ÉCRITURE (l'état), pas dans l'instant ».**
- Correction d'une sur-interprétation faite en cours de nuit : « l'ouvreur X domine l'ancre »
  n'est PAS établi — sur 24 cellules X contre B est un match nul (14/24 bande basse, 10/24 DINO,
  p = 0.54). Les deux battent A\* (21/24, p = 3e-4) : c'est ça, le résultat.

### RÉSULTAT DU LEVIER 2 (2026-08-04, 4 paires sur 6, t* = 500) — **règle INATTEIGNABLE**
- `i1_ours500 − i1_wrong500` crop60 : **−0.0055**, signes **2+/2−**, **0/4** au-dessus de +0.08
  (t2 +0.070, t11 +0.026, t4 −0.029, t14 −0.089). Avec 2 cellules restantes, max 2/6 < 4/6.
- **H_slot confirmée sur ses 3 critères** (|Δ| < 0.05 ; signe aléatoire ; ≤ 1/6 au-dessus).
- `i1_ours500 − memdec20` = −0.004 : la déclaration honnête du bruit ne change rien.
- Hook validé in situ (1 slot, sigma 0.5, rel-L2 0.919, 11 vraies frames a t=0) : le bras a bien
  mesure l'intervention voulue. **Le defaut nomme existe ; il n'est pas le goulot.**
- Consequence d'ordonnancement : la vague `B2` (t* = 250) est le premier candidat a sacrifier si
  le GPU manque, le levier 1 (`C1`/`C2`) n'ayant aucune cellule mesuree.

### RÉSULTAT DU LEVIER 1 (2026-08-04, 8 cellules, 33 rollouts) — **ÉCHOUE 2/8**
- `caption_ours − caption_wrong` crop60 : **+0.0429**, 5+/3−, p = 0.73, **2/8** ≥ +0.08 (règle : ≥ 4/5).
- Jumeau faux **NÉGATIF** comme exigé (`caption_wrong − padded` = −0.0057) : la 2e condition passe.
- Plafond `framebank_vlm − caption_wrong` = +0.0518 (2/8) : une VRAIE frame ne se légende guère mieux.
- Cohérent avec le pré-écran (13/24, p = 0.84) : la charge utile de la légende est au hasard.
**LOT OFFENSIF CLOS — 5 leviers, 5 verdicts.**

---

# MÉMOIRE 2.1 « T0 » — RÉ-ENTRAÎNEMENT ÉCRIVAIN + DÉCODEUR SUR LA MÉTRIQUE T0
## (gates scellés le 2026-08-04, écrits AVANT toute exécution — aucun run, GPU ou CPU, n'avait tourné)

**Le raisonnement, en une ligne.** La nuit du 04/08 a fermé le canal (T0 : le splice est LU,
+0.83 d'excès apparié, ×20 la prédiction), fermé l'instant (levier 4 : 0/2), fermé l'additif
(levier 3 : 2/4) et fermé la légende (levier 1 : 13/24, p = 0.84). Ce qui reste debout est
**l'ÉCRITURE** : l'ancre décodée est une image de l'ambiance du rollout (vs A\* **+0.042**,
vs B **+0.298**, vs X **+0.344**, B > A sur 21/24, p = 3e-4), pas du sujet. Le décodeur fait ce
qu'on lui a appris (reconstruire la frame **évincée**) ; c'est **l'OBJECTIF** qui pose la mauvaise
question. 2.1 change l'objectif : on entraîne l'écrivain **et** le décodeur à maximiser
directement **la métrique T0** (corrélation bande-basse) contre le segment **A**, et à la
minimiser contre l'interlude **B**, les autres vidéos et le **jumeau**.

**Pourquoi T0 et pas DINO / le cosinus latent** (les deux métriques déjà mortes) :
- DINO (phase B) : gate `pos_gap` **−0.185**, 10/10 val négatives, et la dose-réponse de flou
  montre que **même la vérité terrain floutée à σ=8 px échoue** ⇒ le cosinus DINO d'une image
  hors-variété mesure « à quel point c'est flou », pas « qui c'est ». Inutilisable tant que le
  rendu n'est pas réparé — et B1 a prouvé que le rendu ne se répare pas par le mixer.
- Cosinus latent crop60 (J3) : `pos_gap` **−0.19 à −0.36** à tous les poolings, centré ou non.
- **T0 est la seule métrique du dossier qui VOIT le défaut** (elle mesure +0.042 vs +0.298 : elle
  classe l'ancre du bon côté du diagnostic) **et qui tolère le flou par construction** (moyenne
  n×n : elle vit dans la bande que le décodeur atteint réellement). C'est exactement le critère
  qui manquait aux deux termes précédents.

**Métrique, définition gelée** (celle de `prescreen_offensif.lowband`, réutilisée par import, pas
recopiée) : frames → niveaux de gris → `adaptive_avg_pool2d(n)` → aplaties → **centrées par
frame** → **L2-normalisées** ; `corr(ancre, ensemble)` = moyenne sur les frames de l'ensemble du
produit scalaire. `n = 8` (la valeur de tout le dossier). Variante latente (à départager, voir
D0) : mêmes opérations sur le latent 16×60×104, réduction de canaux `chan` (pooling par canal)
ou `gray` (moyenne des canaux).

**Objectif d'entraînement** : `loss = rec (régulariseur, poids RÉDUIT) + lam_t0 * T0`, où le
terme T0 est contrastif sur l'ancre décodée depuis l'état à la position de **RETOUR simulée** :
positifs = frames du segment **A** de la même vidéo, négatifs = frames du segment **B** de la même
vidéo + frames des **autres** vidéos du bon côté du split + le **jumeau** (même prompt, graine
+100000, le négatif dur validé empiriquement : 93/145 sondes). Positions de retour = l'état de
**FIN de séquence** (forcé) + 1-2 états post-switch échantillonnés (règle min_gap du collecteur).

---

## D0 — DÉCISION D'ESPACE (latent poolé vs VAE-decode + pixel), scellée AVANT le premier entraînement

Le terme T0 peut vivre (a) sur le **latent avg-poolé** (gratuit) ou (b) sur **1 frame VAE-décodée
puis poolée** (10.9 GB de backward mesurés en phase B, donc faisable). La règle décide, pas le
goût.

**RÈGLE SCELLÉE D0.** L'espace **latent** est retenu **si et seulement si les trois** sont vrais :
1. **D0.a (accord de géométrie, CPU)** : sur les 30 ancres décodées dont le **latent** (`reid_scen/
   dec_anchors/*.pt`) **et** le **pixel** (`reid_scen/anchor_px/*.pt`) sont sur disque, la matrice
   de similarité bande-basse en latent et celle en pixel doivent s'accorder à **Spearman ≥ 0.70**
   sur les paires croisées **et** sur **≥ 50 %** de plus-proche-voisin identique.
2. **D0.b (cas connu « une vraie frame de A »)** : sur ≥ 15 vidéos switch val du corpus, une
   **vraie** frame latente de A doit corréler davantage avec l'ensemble A qu'avec l'ensemble B, en
   latent, sur **≥ 90 %** des vidéos (le pixel le fait trivialement).
3. **D0.c (cas connu « l'ancre actuelle », le test qui compte)** : sur les mêmes vidéos val,
   l'ancre décodée par l'écrivain+décodeur **A2** depuis l'état de fin de séquence doit reproduire
   en latent le classement mesuré en pixel — **B > A** — sur **≥ 70 %** des vidéos, ET les deux
   espaces doivent être d'accord vidéo par vidéo sur le signe de `corr(A) − corr(B)` dans
   **≥ 75 %** des cas.
**Sinon ⇒ espace PIXEL** (banque de références pré-décodée hors ligne, VAE différentiable
seulement du côté ancre, `--t0_k` frames par vidéo). Aucun autre espace n'est autorisé.
*(D0.a et D0.b sont CPU et tournent avant tout GPU ; D0.c coûte ~5 GPU-min et passe en premier.)*

---

## G1 — GATE TOY (sur-apprentissage 8 vidéos) : **l'état PEUT-IL porter A ?**

Reprise exacte du protocole F3.1 (8 vidéos switch, `--fixed_positions`, `cond_dropout = 1.0` :
l'état est la SEULE route), mais la quantité surveillée est la **marge T0** de l'ancre décodée à
la position de retour : `m = corr(ancre, A) − corr(ancre, B)`.

- **Valeur de départ, mesurée et publiée** : `m = 0.042 − 0.298 = **−0.256**` (24 cellules,
  16.1 quater) ; la variante snapshot (levier 4) donne −0.217 (t2) et −0.590 (t3). La marge est
  donc **franchement négative** aujourd'hui, sur les deux instants.
- **RÈGLE SCELLÉE G1** : après sur-apprentissage (≤ 600 steps), sur les 8 vidéos du toy,
  **marge moyenne ≥ +0.10** ET **≥ 6/8 vidéos à marge > 0**, avec `corr(ancre, A) ≥ +0.15`.
  ⇒ l'état PEUT porter A, on paie l'entraînement complet.
- **STOP si** marge moyenne < +0.05 **ou** ≤ 4/8 vidéos positives : **l'état ne peut pas**, la
  cartouche est tirée, on écrit le négatif et on arrête la ligne mémoire.
- Zone grise (+0.05 à +0.10, ou 5/8) : **un** seul re-essai d'hyperparams autorisé, noté au
  registre, puis application de la règle.
- **Contrôle obligatoire au même run** : `h-shuffle` (la séquence h d'une AUTRE vidéo) doit laisser
  la marge **≤ 0** — sinon la marge est obtenue par le décodeur seul (cond/prior), pas par l'état,
  et G1 est **VOID**.
- **Contrôle 2 (anti-Goodhart)** : la marge doit monter **sans** que `corr(ancre, A)` soit atteinte
  par effondrement de la variance spatiale de l'ancre (une ancre quasi-constante a une corrélation
  mal définie) : `t0/anchor_std` (écart-type spatial de l'ancre avant normalisation) doit rester
  ≥ 25 % de sa valeur au step 0. Sous ce seuil ⇒ G1 **VOID**.

## G2a — TRIPWIRE HELD-OUT HORS LIGNE (26 vidéos val du corpus, ~5 GPU-min)

Même mesure que G1 mais sur les **vidéos val** (split par vidéo, F4.1, jumeaux du bon côté).
- **RÈGLE SCELLÉE G2a** : `marge moyenne ≥ +0.05` ET `≥ 60 %` des vidéos val à marge > 0.
  Sinon ⇒ **STOP avant tout rollout** (l'objectif est appris mais ne généralise pas : c'est le
  verdict train-36 %/val-5 % du J3 qui se répète, et il ne mérite pas 6 rollouts).

## G2b — GATE HELD-OUT EN SITU (le pré-écran du levier 4, sur 6 cellules)

Mesure **identique** à celle de `snap_prescreen.py` / 16.1 quater, sur l'ancre réellement dumpée
par le bras `memdec_t0` pendant les 6 rollouts de la campagne (le dump est gratuit, il sort du
même rollout que la vidéo) : `corr(ancre, A*)`, `corr(ancre, B)`, `corr(ancre, X)`, références
inchangées (A\* = segment 0 du S3 propre de la cellule ; B = l'interlude ; X = l'ouvreur).
- **RÈGLE SCELLÉE G2b** : `corr(ancre, A*) ≥ +0.15` **ET** `> corr(ancre, B)` sur **≥ 4/6**
  cellules. Sinon ⇒ **STOP**, on ne score pas les vidéos (le colis n'est toujours pas A).
- Référence à battre, mesurée : ancre courante **+0.042 / +0.298 / +0.344**.

## G3 — CAMPAGNE VIDÉO (le juge final), seulement si G1 + G2a + G2b sont franchis

6 cellules : **3 S2** (t2, t4, t11) + **3 SS2** (t12, t13, t14) — celles dont `memdec20` **et**
`wronganchor` sont déjà sur disque, donc seul `memdec_t0` est à tirer (6 rollouts).
Scoring : re-ID crop60 contre le **A\* propre** (segment 0 du rollout SS1/S3 de la même cellule,
règle anti-contamination du §11.3), code de scoring inchangé.
- **LE JUGE** : `memdec_t0 − wronganchor` (le jumeau qui splice une ancre étrangère : il contrôle
  l'effacement de contexte, le seul effet qui ait jamais été mesuré) **≥ +0.08 sur ≥ 4/6 cellules**.
- Secondaire, rapporté mais **non décisionnel** : `memdec_t0 − memdec20`.
- **PRÉDICTION HONNÊTE, écrite avant** : mon pari ponctuel est **+0.02 à +0.06**, **2/6** cellules
  au-dessus du seuil — parce que 16.1 bis a mesuré que **la quantité copiée ne prédit pas le
  re-ID** (Pearson du contraste : −0.075, Spearman −0.163, n = 24). Autrement dit : **même si G1 et
  G2b passent, G3 peut échouer, et ce ne serait pas une contradiction** mais la confirmation que le
  canal est aveugle à l'identité. Branche pré-assignée dans ce cas : *« l'ancre porte enfin A et le
  rendu ne le lit pas ⇒ le goulot est le CANAL de lecture (splice), pas l'écriture »*, et c'est le
  papier négatif complet (les deux bouts fermés), pas une itération de plus.
- **Puissance, dite d'avance** : le témoin sans contenu de la mini-batterie (`sdedit_wrong −
  wronganchor`, différence de contenu NULLE) fait 3/6 au-dessus de +0.08 avec sd 0.197 par cellule.
  **À 6 cellules et 1 graine, le plancher de bruit vaut 2-2.5× le seuil** : un G3 positif à 4/6
  devra donc être répliqué sur une 2e graine avant toute publication ; un G3 négatif, lui, est
  concluant (il va dans le sens du plancher).

## RÈGLES TRANSVERSES SCELLÉES
1. **Aucun re-scoring contre une référence contaminée** : A\* propre uniquement (leçon 2× payée).
2. **Chaque run recommencé est noté au registre** (date, raison, hyperparams changés). Budget
   annoncé : **≤ 3 allers-retours d'hyperparams** avant le 15/08, kill le 15/08 quoi qu'il arrive.
3. **wandb obligatoire** (`mem2_writer_T0`), marge T0 loggée **dès le step 0**, métriques par âge
   et par kind conservées.
4. **Contrat de warm-start** : `writer/delta_l1 == 0.0` au step 0 (F2.3) et gradient non nul sur
   `qkv.weight` après le premier backward (F3.3), asserts inchangés.
5. **`lam_t0 = 0` doit être byte-safe** : le terme n'est pas construit, le run reproduit A2.
6. **GPU** : pré-check d'**EXCLUSIVITÉ** (aucun voisin ≥ 20 GB) + second pré-check jitté, jamais
   idx6, **max 2 cartes**.


### RÉSULTAT D0 (partie CPU, 2026-08-04, 0 GPU) — **le latent est REJETÉ, le terme T0 vivra en PIXEL**

| sous-test | mesure | seuil scellé | verdict |
|---|---|---|---|
| **D0.a** accord de géométrie latent↔pixel (30 ancres décodées, paires (latent, pixel) du même objet) | `chan` Spearman **+0.506**, plus-proche-voisin **21 %** ; `gray` **+0.454** / **18 %** | ≥ +0.70 **et** ≥ 50 % | **FAIL, franc** |
| **D0.b** cas connu « une VRAIE frame de A » (18 vidéos switch val) | `chan` **94 %** (marge +0.363) ; `gray` **94 %** (+0.361) | ≥ 90 % | PASS |
| **D0.c** cas connu « l'ancre actuelle » (A2, pixel vs latent) | **TIRÉE PAR L'AUDIT (2026-08-04, 18 vidéos val)** : `chan` B>A 39 % / accord de signe 72 % ; `gray` 56 % / 56 % | ≥ 70 % / ≥ 75 % | **FAIL ⇒ PIXEL confirmé** (D0 n'est plus provisoire) |

**Lecture.** D0.b dit que la métrique latente est *saine* (une vraie frame de A se range du bon
côté) ; D0.a dit qu'elle **n'ordonne pas les ANCRES DÉCODÉES comme le pixel** — et c'est sur des
ancres décodées que le gate se joue. Le VAE de Wan n'est pas une fonction spatialement alignée de
la luminance : deux ancres proches en latent poolé ne le sont pas en bande basse rendue. Un seul
sous-test suffit à rejeter (« latent SSI les trois ») ⇒ **espace PIXEL**, banque de références
pré-décodée hors ligne (`build_bank_t0.py --with_pixel`), VAE différentiable **seulement du côté
ancre** (`--t0_k` frames/vidéo). Le défaut de `--t0_space` est passé à `pixel` dans le code.
Conséquence de coût à mesurer avant le premier run long : `--vram_probe` (la phase B mesurait
10.9 GB avec 1 décode VAE + DINO ; T0 n'a pas de DINO mais paie `t0_k` décodes).
Artefact : `mem2_runs/writer/space_decision_t0.json`.

---

### AUDIT INDÉPENDANT DES GATES T0 (2026-08-04) — **les seuils sont conformes au code, les BASELINES ne le sont pas**

Dossier complet et mesures : `memory2_t0_report.md` **§7**. Ce qui touche aux règles **scellées**
de cette section, et qui doit être tranché avant le GO :

1. **Le point de départ de G1/G2a n'est pas −0.256.** Mesuré sur la quantité que ces gates
   mesurent réellement (bande basse **pixel**, références du corpus, 18 vidéos val, écrivain +
   décodeur A2, état de fin de séquence) : **marge = +0.036, 10/18 vidéos positives,
   `corr_A` = +0.391**. Le modèle non entraîné franchit donc déjà `corr_A ≥ +0.15` (×2.6) et
   frôle G2a (≥ +0.05 et ≥ 60 %). Source : `space_decision_t0.json`, ligne
   `[D0.c] PIXEL (référence) : B > A sur 44 %`.
2. **Pourquoi −0.256 ≠ +0.036 : les références de 16.1 quater ne sont pas appariées** (positif
   A\* d'un AUTRE rollout, négatif B du MÊME rollout). L'**ORACLE** — une vraie frame du segment A
   du rollout — donne **vs A\* +0.374 | vs B +0.894, B > A sur 6/6** : la vérité terrain reproduit
   la signature publiée. « B > A sur 21/24 » ne démontre donc pas que l'ancre porte l'ambiance
   plutôt que le sujet.
3. **G2b, tel que scellé, est infranchissable** : l'ORACLE le passe sur **0/6** cellules
   (`corr(A*) ≥ +0.15` **ET** `> corr(B)`). À re-cadrer (références appariées du même rollout, ou
   design d'**excès apparié** à la `prescreen_offensif`, qui annule l'ambiance).
4. **La bande basse 8×8 est aveugle au sujet à travers les rollouts** : marge SUJET (A\* − autres
   sujets du **même** rollout propre) = **+0.038, 3/6 pour l'ORACLE** (ancre courante −0.008,
   ancre étrangère −0.063). Cohérent avec 16.1 bis (Pearson −0.075).
5. **Le jumeau n'est PAS dans les négatifs du terme T0** : les 30 jumeaux du corpus sont tous
   `single`, la banque T0 ne contient que du `switch`, et les 105 vidéos switch ont **105 prompts
   A distincts**. `build_bank_t0.py` **sort en code 1** sur ce point. La phrase « négatifs = … +
   le jumeau » de cette section est **factuellement fausse** pour T0.
6. **Calibration proposée pour G1** (plafonds mesurés en pixel, 105 vidéos) : oracle 1 frame de A
   **+0.367**, plafond moyenne-A **+0.393**, **modèle nul** (moyenne des A des autres vidéos)
   **+0.051**. Le seuil +0.10 vaut 27 % de l'oracle et 2× le modèle nul. Proposition : marge
   ≥ 40 % de l'oracle (≈ +0.15) **et** ≥ 3× la marge de l'ancre générique (contrôle VOID à
   ajouter, coût nul), et suppression de `corr_A ≥ +0.15` (déjà franchi au step 0).
7. **VRAM du terme pixel, mesurée** (K=3, `t0_k` 2) : `n_bptt` complet **35.4 GB / 9.1 s**,
   `n_bptt 12` **21.5 GB / 4.4 s**, `n_bptt 4` **16.7 GB / 3.0 s**. Le risque n° 1 annoncé
   n'existe pas. En revanche **`n_bptt ≥ 12` est un PLANCHER** : l'âge de l'ancre depuis l'état de
   fin de séquence vaut 12 writes, en dessous le gradient n'atteint plus l'écriture de A.
8. **Correctifs de code appliqués par l'auditeur** (byte-safety de `--t0_end_pos`, tripwire
   anti-Goodhart re-placé dans l'espace de la métrique, en-tête de `t0_prescreen`) : détail au
   §7.6, selftests **47/47**.

---
# ÉTAPE 1 DU PLAN 04/08 : SONDAGE DES POIDS RAPIDES (scellé AVANT implémentation et runs)
Protocole : plan_memoire_2026-08-04.md. Instrument : MLP 2-3 couches sur signature
comportementale (sondes = centroïdes k-means des activations corpus + sous-échantillon
des requêtes apprises du décodeur ; réponses blanchies). Tâche : désigner le bon individu
parmi les deux d'une paire même-prompt (hasard 50%), références = embeddings DINO des
crops-sujet (masques tâche 0, validés œil). Split strict par paire : ~18 train / 5 val /
7 test. Score par paire = marge continue moyennée (âges × images), test de Wilcoxon.

SEUILS SCELLÉS :
1. VALIDITÉ DU PROTOCOLE (borne supérieure) : le même pipeline sur les ACTIVATIONS brutes
   doit atteindre ≥90% de précision par paire sur le test. Sinon : protocole invalide,
   on répare l'instrument, AUCUNE conclusion sur la mémoire.
2. CONTRÔLE EFFONDREMENT : états permutés entre vidéos → précision ≤55%.
3. SUCCÈS de l'état : précision-sujet test ≥65% ET Wilcoxon p<0.05 sur les marges ET
   (précision-sujet − précision-fond) ≥ +10 points.
4. ZONE GRISE (55-75% sans les 3 conditions) : +30 paires générées (une nuit) AVANT
   toute conclusion. ÉCHEC : <55% ou écart sujet-fond nul → ARRÊT du chantier mémoire.
PRÉDICTION (la mienne, avant run) : précision-sujet ~60%, P(succès plein) ≈ 50-55%.

## ADDENDUM sondage (scellé avant les runs de verdict, 2026-08-04 soir)
- ANALYSE PRIMAIRE = les 30 paires, sans filtre (aucune sélection après coup possible).
  ANALYSE SECONDAIRE pré-enregistrée = le sous-ensemble "le masque porte bien le
  protagoniste" (~19 paires selon l'agent), liste FIGÉE à partir des planches et du
  verdict œil d'Aden AVANT tout score. Les deux rapportées côte à côte.
- LIMITE D'INTERPRÉTATION actée : les paires actuelles sont des vidéos à un seul sujet :
  un verdict positif = "l'identité est extractible d'un état NON PERTURBÉ" (condition
  nécessaire). La version avec interférence (paires à changement de sujet) = l'extension
  +30 paires déjà prévue, à collecter seulement si le verdict le justifie.

## RÉGLAGES FINAUX (post-audit, scellés AVANT tout run de verdict, 2026-08-04 soir)
1. La règle 1 (borne de validité ≥90%) et toutes les précisions se lisent en PAR PAIRE
   (c'était la lettre du protocole ; l'audit T9b montre que la lecture par-ligne plafonne
   à 0.911 même à signal parfait : par-paire = 1.000).
2. Procédure statistique PRIMAIRE = validation croisée par paires sur les 30 paires
   (folds étanches, hyperparamètres à l'intérieur des folds), au lieu du split fixe 18/5/7
   dont le test tirait 5/7 masques-accessoires : décision prise sur recommandation d'audit,
   AVANT tout score. Seuils inchangés : sujet ≥65% par paire, Wilcoxon p<0.05 sur les
   marges, (sujet − fond) ≥ +10 points ; zone grise 55-75% → +30 paires.
3. Liste SECONDAIRE "protagoniste" FIGÉE = les ~19 paires identifiées par l'œil de l'agent
   des masques (documentée dans probe1_report), gelée avant les scores ; l'œil d'Aden sur
   les planches servira de CONFIRMATION a posteriori, pas de sélection.

## LISTE ŒIL D'ADEN (figée AVANT lecture de tout score, 2026-08-04 tard)
Validation visuelle des 30 paires numérotées, ambiguïtés résolues par les prompts :
BONNES = {1,2,6,8,11,12,15,16,19,20,21,22,24,27,30} (15/30).
RATÉES = {3,4,5,7,9,10,13,14,17,18,23,25,26,28,29} (masque sur accessoire/décor/mauvais
personnage, ou incohérent entre les deux vidéos). Nettement plus strict que la liste
agent (~19) : l'analyse rapportera PRIMAIRE (30), SECONDAIRE-agent (19, déjà gelée),
TERTIAIRE-Aden (15). Toute divergence entre les trois est elle-même une information
(sensibilité du verdict à la qualité des masques).

## PLAFOND HUMAIN (œil d'Aden, scellé 2026-08-05, AVANT toute re-lecture de scores)
Sur les paires à masque valide, distinguabilité des deux individus PAR UN HUMAIN :
- CLAIREMENT DIFFÉRENTS : {01 pompier (veste), 02 bouteille (capuchon), 11 oiseau,
  12 athlète (tenue), 19 enfant+chat, 27 fillette} = 6 paires.
- LÉGÈREMENT différents : {06 foule, 08 verrou, 16 femme, 24 ange} = 4 paires.
- QUASI IDENTIQUES (aucun instrument ne peut les distinguer) : {15, 20, 21, 22, 30} = 5.
- Correction : la paire 24 a un masque VALIDE (le crop est bien l'ange) → 16 paires valides.
- Chez les paires à masque raté aussi, majorité de "même individu" → LES JUMEAUX DU CORPUS
  SONT BEAUCOUP MOINS DISTINCTS QUE SUPPOSÉ (même prompt → instances très proches, les
  différences sont des détails : couleur de veste, capuchon).
CONSÉQUENCE : plafond humain estimé ~74% de précision par paire sur les 15-16 valides
(6×100% + 4×~65% + 5×50%). La règle 1 (≥90% absolu) était INATTEIGNABLE MÊME POUR UN
HUMAIN sur ces matériaux : le 0.762 par-paire de l'instrument cette nuit est peut-être
proche du plafond des MATÉRIAUX, pas d'une défaillance de l'instrument. RE-ANCRAGE
scellé : la borne de validité se juge désormais sur le sous-ensemble "clairement
différents" {01,02,11,12,19,27}, seuil = plafond humain − marge (à sceller au re-run),
ET nouveaux matériaux à générer : 30 paires à prompts SANS attributs (les instances
divergent librement) pour la version définitive du test.

## SCELLEMENT FINAL v2 (exigé par l'audit R1, AVANT le premier run cv --audited, 2026-08-05)
- Règle 1 (validité) : source hdirect (activations agrégées SANS écriture), lecture par
  paire, sur le sous-ensemble décisif d'Aden {01,02,11,12,19,27} du corpus v1 : ≥ 5/6.
- Configuration des runs : --features pool (compression moyenne+max par sonde) et
  --pretrain ACTIVÉ (105 vidéos switch) = la configuration par défaut des verdicts ;
  la variante ACP = exploratoire, rapportée mais non décisionnelle.
- Liste "agent" recalculée mécaniquement = 21 paires : remplace l'ancienne mention "19"
  (jamais figée en indices) ; les sous-ensembles décisionnels restent : 30 (primaire),
  Aden-15, décisif-6 : tous lus depuis runs/subsets_v1.json (verrou de confrontation).
- Contrôle permuté : moyenne sur 25 dérangements (défaut D1 corrigé).
- Corpus v2 : AUCUN run de verdict avant que l'œil d'Aden ait labellisé la planche v2
  (M et D par paire) et que sa règle 1 propre soit scellée ici (R2 de l'audit).

## LABELS ŒIL D'ADEN, CORPUS v2 (scellés 2026-08-05, AVANT tout run v2)
- MASQUES INVALIDES : {01 camion, 03 voiture, 07 tasse, 08 voiture(+D-non), 10 tableau
  sur la 2e vidéo} → 25 paires valides.
- CLAIREMENT DIFFÉRENTS (fort) : {05,06,09,11,12,13,14,15,16,17,18,22,23*,24,25,26,27,
  28,29} = 19 paires (*23 : plusieurs guitares dans la scène, à surveiller).
- FAIBLEMENT différents : {02,04,19,20,21,30} = 6.
- Plafond humain v2 ≈ 90% sur les 25 valides ; ≈100% sur les 19 fortes.
RÈGLES v2 SCELLÉES : Règle 1 (validité) = hdirect ≥ 16/19 sur les paires fortes.
Primaire = les 25 valides ; rapport séparé sur les 19 fortes. Succès état : ≥65% par
paire sur les 25 ET Wilcoxon p<0.05 ET (sujet − fond) ≥ +10 pts. Contrôle permuté ≤55%
(25 dérangements). PRÉDICTION (la mienne) : hdirect passe ~17-18/19 ; état-sujet : je
maintiens ~55-60%, P(succès) ≈ 45-55% : le corpus est enfin capable de trancher.

## CORRECTIF DE BORNE scellé (2026-08-05 nuit, AVANT son run)
Constat : la borne "activations poolées par chunk" échoue sur LES DEUX corpus (v1 3/6 ;
v2 pair=0.444 sur 30) alors qu'un humain réussit : le pooling uniforme moyenne le sujet
(minoritaire) dans le fond : l'instrument reproduisait le défaut à mesurer. Nouvelle
source scellée : hdirect_masked = pooling des seuls tokens sous le masque-sujet (masques
existants, tokens alignés patches, masque du frame échantillonné le plus proche par
chunk). Règle 1 inchangée dans l'esprit : hdirect_masked ≥ 5/6 (v1, décisif) et ≥ 16/19
(v2, fortes). PRÉDICTION : si ça échoue ENCORE, le défaut est dans les références DINO
ou la tête, et l'étape suivante est une inspection manuelle des 6 paires décisives.
L'état, lui, reste sondé tel quel (il est global par nature : c'est l'objet mesuré).

## VERDICT FINAL DU SONDAGE (2026-08-05 matin) : LÉGAL, et il est NÉGATIF
Corpus v2 (matériaux labellisés Aden, jumeaux distincts) :
- RÈGLE 1 (borne masquée) : PASS 16/19 (0.825, p=0.011) : L'INSTRUMENT MARCHE : il
  ré-identifie depuis les activations brutes là où un humain réussit.
- RÈGLE 2 (permuté) : PASS (0.507 = hasard).
- RÈGLE 3 (la MÉMOIRE) : sujet = 0.467 par paire (p=0.82), fortes = 0.456 : HASARD.
- VERDICT SCELLÉ APPLIQUÉ : ÉCHEC → ARRÊT du chantier mémoire. Le même instrument qui
  lit l'identité dans les activations ne lit RIEN dans les poids rapides : l'identité
  n'est pas extractible de l'état tel qu'il s'écrit. (v1 reste protocole-invalide :
  borne 3/6 même masquée : matériaux quasi-clones ; v2 suffit, il était fait pour ça.)
Conclusion projet : la chaîne causale finale est complète : le canal transmet, l'ancre
ne porte que l'ambiance, ET l'ambiance est tout ce que l'écriture actuelle rend
extractible (prouvé par sondage à instrument validé, plafond humain calibré). Le
programme "écriture triée par masque-sujet / deux registres" devient LE travail futur
motivé par la preuve. L'écriture de l'article démarre (conformément au plan).

---
# AMENDEMENT PI (Aden, 2026-08-05) : l'arrêt scellé condamnait l'EXTRACTION depuis les
# états actuels ; il ne jugeait pas un RÉ-ENTRAÎNEMENT DE L'ÉCRITURE (prémisse nouvelle,
# jamais testée : aucun écrivain n'a jamais eu d'objectif d'identité). UNE tentative
# bornée autorisée, kill 15/08 inchangé. Le sondeur validé (masque-sujet → embedding,
# 16/19) devient GATE et SIGNAL D'ENTRAÎNEMENT.

## TOY D'ÉCRITURE-IDENTITÉ (seuils scellés AVANT implémentation, 2026-08-05)
Protocole : 8 paires fortes du corpus v2 (16 vidéos), co-entraînement offline
écrivain+tête-de-lecture (replay_writes, gradient à travers toutes les écritures) avec
objectif : l'embedding lu depuis l'ÉTAT doit désigner le bon jumeau (InfoNCE, cible =
embeddings sujet DINO du sondage, jumeau forcé dans le softmax). Reco gardée en
régulariseur léger.
PASS exige LES QUATRE : (1) précision par paire sur les 8 paires (overfit autorisé)
≥ 7/8 ; (2) contrôle états-permutés ≤ 5/8 ; (3) le MÊME entraînement avec écrivain GELÉ
(tête seule) fait NETTEMENT pire (sinon contradiction avec le sondage → suspicion de
fuite) ; (4) la reco sur les 8 vidéos ne se dégrade pas de plus de 2× (l'écrivain reste
une mémoire, pas un hash d'identité).
ÉCHEC ⇒ fin définitive du chantier écriture, la phrase finale du papier est acquise.
PRÉDICTION (la mienne) : P(pass) ≈ 35-40%.

## VERDICT DU TOY (2026-08-06, ~00h30, GPU 7, deux bras à terme) : FAIL → ARRÊT DÉFINITIF
- (1) précision par paire : 8/8, marge +0.4188 : PASS
- (2) permuté (25 dérangements) : 3.68/8 = hasard : PASS
- (3) écrivain GELÉ : 8/8, marge +0.4217 : ÉGAL à l'entraîné (écart 0 paire, marges
  identiques au millième ; son propre permuté 3.64/8 = hasard aussi) : **FAIL**
- (4) reco : 0.854 → 0.400 (×0.47, AMÉLIORÉE) : PASS
Lecture : sur 8 paires apprises par cœur, une simple tête de lecture entraînée sur les
états TELS QU'ÉCRITS PAR L'ÉCRIVAIN D'ORIGINE sépare déjà parfaitement les jumeaux
(et son permuté au hasard prouve qu'elle utilise bien l'appariement état↔sujet).
Ré-entraîner l'écriture avec l'objectif d'identité n'apporte STRICTEMENT RIEN de
mesurable, même dans le cas le plus favorable (in-sample, full BPTT, warm-start).
Cohérent avec le sondage : in-sample l'info idiosyncratique suffit à une tête ;
held-out rien ne généralise ; l'écriture n'est pas le levier.
RÈGLE SCELLÉE APPLIQUÉE : fin DÉFINITIVE du chantier écriture. La phrase finale du
papier est acquise : « l'identité du sujet n'est pas extractible des poids rapides,
et ré-entraîner l'écriture avec un objectif d'identité ne change pas ce constat dans
le régime le plus favorable ». Ma prédiction P(pass)≈35-40% : l'échec était le
scénario attendu. CAP UNIQUE : rédaction de l'article.
Artefacts : /shared/ssd_14T_project/adenelmi/toy_identity/ (verdict.txt, verdict.json,
result_{trained,frozen,smoke}.json, toy_{trained,frozen}.pt, logs, wandb
toy_identity/{0rakqpba (trained), hft7az85 (frozen)}) ; code
reference/runs/toy_identity/ (toy_identity.py, selftest_toy.py, run.sh).

---
# ORACLE D'ÉCRITURE MASQUÉE (GO Aden 2026-08-06, seuils scellés AVANT implémentation)
Nature : DIAGNOSTIC PAR REJEU, zéro entraînement d'écriture (hors du gel scellé, famille
des sondes). Hypothèse : la cause de l'échec d'extraction est la démocratie des tokens à
l'écriture (le fond majoritaire écrase le sujet minoritaire) ; preuve attendue par
intervention : filtrer les écritures au masque-sujet rend l'identité extractible.
PROTOCOLE : rejeu des écritures sur les 60 vidéos v2 (h déjà sur disque), 3 bras :
 (A) original = tous tokens (baseline, doit reproduire le verdict scellé ≈ hasard) ;
 (B) sujet-seul = tokens sous masque vérité-terrain, MÊME seuil d'appartenance token
     que hdirect_masked (validé 16/19) ; chunk sans token sujet → écriture SAUTÉE ;
 (C) fond-apparié = même NOMBRE de tokens que (B) par chunk, tirés du complément
     (contrôle : sélection-sujet vs simple réduction de charge d'écriture) ; mêmes
     chunks sautés que (B) pour apparier les calendriers d'écriture.
Puis signatures identiques au protocole légal (mêmes âges) + sondage CV 6 plis, 19
paires fortes, pool, SANS pré-entraînement pour les 3 bras (comparaison à armes égales ;
le bras A re-couru dans cette config exacte).
SEUILS SCELLÉS :
 - PRIMARY : bras B (sujet-seul) ≥ 14/19 paires (binomial p=0.032) = extractible.
 - DISSOCIATION : B − C ≥ 3 paires → mécanisme = SÉLECTION du sujet ;
   si B passe ET C aussi (écart <3) → effet "charge d'écriture réduite", pas sélection
   (conclusion différente, à écrire telle quelle).
 - SANITÉ : bras A dans [7..12]/19 (≈ hasard, cohérent verdict scellé).
 - ZONE GRISE B = 12-13/19 : étendre aux 25 paires valides AVANT toute conclusion.
 - B < 12 : même nourrie du sujet seul, l'écriture ne retient pas l'instance =
   négatif profond final, chiffré contre la borne 16/19.
SELFTESTS AVANT RUN COMPLET : mask=none bitwise-identique aux états existants (2 vidéos) ;
alignement masque→token identique au chemin hdirect_masked ; stats tokens/chunk loggées.
PRÉDICTION (la mienne) : P(B ≥ 14/19) ≈ 50-55% ; si B passe, P(dissociation nette) ≈ 75%.
GPU : 1 seul (7 si libre, pré-check dans la commande). Code additif dans
reference/runs/probe1/, états sous ssd_14T_project/adenelmi/probe1/oracle_maskwrite/.

NOTES DE RELECTURE PRÉ-RUN (2026-08-06, avant lancement, arbitrages hors-protocole) :
 1. Cas limite sujet > 50% du chunk (31/1500 écritures, 2.1%, 3 vidéos) : l'appariement
    exact du bras C y est impossible ; règle adoptée : C prend TOUT le complément,
    déficit loggé (keep_info["bg_deficit"]), jamais silencieux. Biais négligeable et
    plutôt favorable à C (2.1% des écritures) → conservateur pour la dissociation B−C.
 2. Fait de corpus : 0/1500 écritures sautées (sujets v2 proéminents, 21.7% des tokens
    en moyenne) ; la règle de saut reste du code vivant, inerte sur ce corpus.
 3. Le repli top-3% de l'alignement masque→token (règle de LECTURE de hdirect_masked)
    est coupé à l'ÉCRITURE (min_frac=0.0) : sinon la règle scellée "chunk vide → saut"
    serait inatteignable. Assert que le repli ne se déclenche jamais.
 4. Mode d'exécution : STREAM (rejeu à la volée, zéro état écrit sur disque ; le mode
    disque écrirait 76GB sur 101GB libres), CONDITIONNÉ à 2 selftests d'équivalence
    supplémentaires avant le run : S4a keep=tous-tokens == chemin historique bit-à-bit
    (1 vidéo) ; S4b signatures stream == signatures depuis états-disque (1 vidéo, bras
    subject). Échec de l'un → repli mode disque, bras subject d'abord, purge après.

## VERDICT DE L'ORACLE (2026-08-06 15:44, run 31 min GPU 7, S4a/S4b PASS bit-à-bit) : FAIL
| bras | strong19 | acc pair (30) | p | permuté |
|---|---|---|---|---|
| A original      | 8.00/19  | 0.421 | 0.94 | 0.515 |
| B sujet-seul    | 10.67/19 | 0.561 | 0.48 | 0.491 |
| C fond-apparié  | 7.33/19  | 0.386 | 0.94 | 0.503 |
APPLICATION MÉCANIQUE DES SEUILS : sanité A PASS (8 ∈ [7..12]) ; PRIMARY B = 10.67 < 12
(même pas la zone grise) → **B FAIL** : même nourrie EXCLUSIVEMENT des tokens du sujet
(masque vérité-terrain, seuil validé 16/19), l'écriture ne produit pas d'état dont
l'identité soit extractible. La dissociation B−C = +3.3 est sans objet (B sous le seuil) ;
noté pour honnêteté : B est numériquement le meilleur bras (+2.7 vs A), marges +0.0026 vs
−0.0012/−0.0043, TOUT sous le bruit (p=0.48).
LECTURE MÉCANISTIQUE FINALE : les 3 régimes d'écriture (tout, sujet-seul, fond-apparié)
sont au hasard ; le SEUL lecteur au-dessus du hasard reste le pooling direct des h sans
écriture (16/19). Le goulot n'est PAS la dilution du sujet par le fond : c'est la
COMPRESSION PAR ÉCRITURE-GRADIENT elle-même (≈25 pas Muon dans le MLP fast-weight) qui
ne retient pas l'information d'instance, quel que soit ce qu'on lui donne. L'ambiance
(catégorie/scène) survit à l'écriture, l'instance non : filtre passe-bas de contenu.
CONSÉQUENCES : (a) la prémisse du design deux-registres ("trier à l'écriture suffit")
est RÉFUTÉE pour cette règle d'écriture : le travail futur du papier doit viser la RÈGLE
D'ÉCRITURE (stockage non-compressif type banque/KV pour l'instance), pas le tri seul ;
(b) le papier gagne son diagnostic terminal : chaîne causale complète avec échelle
d'interventions (entrée OK 16/19 → écriture perd, ni ré-entraînement (toy) ni sélection
parfaite (oracle) ne réparent) ; (c) plus AUCUNE expérience mémoire en suspens :
cap 100% rédaction. Ma prédiction P(pass)≈50-55% : perdue (négatif plus profond que
prévu). Artefacts : runs/cv_state{,_mw_subject,_mw_bg_matched}_subject_pool_v2.json,
sig/state_mw_*/, runs/oracle_maskwrite_stream.log, selftest_maskwrite.py (S0-S4).

---
# ORACLE DE LA RÈGLE D'ÉCRITURE LINÉAIRE (GO Aden 2026-08-06 soir, seuils scellés AVANT
# implémentation). Question : l'échec est-il propre à la règle GRADIENT (TTT-MLP/Muon),
# ou toute écriture en ligne perd-elle l'instance ? Fait moteur : la moyenne masquée des
# mêmes tokens lit 16/19 ; la famille linéaire (accumulation par produits extérieurs,
# delta rule = poids rapides de la linear attention) est l'analogue matriciel d'une
# moyenne. Case jamais testée. Diagnostic par REJEU, ZÉRO entraînement (aucun paramètre
# appris : règles fixes), hors du gel scellé.
PROTOCOLE : mêmes 60 vidéos v2, mêmes h, mêmes masques (chunk_token_mask min_frac=0.0),
mêmes âges, même sondage CV (6 plis, pool, v2, sans pré-entraînement, mêmes plis).
État par bloc (5,16,27) : W (d×d), init 0. Lecture pour signatures : W @ sondes (mêmes
sondes k-means + queries décodeur), pooling mean+max inchangé.
BRAS (4, un seul PRIMARY) :
 (L1) PRIMARY : sujet-masqué, accumulation MOYENNE de produits extérieurs
      W = mean_t(v_t k_t^T), k = h/||h||, v = h (l'analogue matriciel exact de la
      moyenne prouvée 16/19) ;
 (L2) tous-tokens, même règle (la case "sans masque oracle") ;
 (L3) sujet-masqué, delta rule correctrice W ← W + β (v − W k) k^T, β=0.5, séquentiel
      par token dans l'ordre du chunk ;
 (L4) tous-tokens, delta rule β=0.5.
SEUILS SCELLÉS :
 - PRIMARY : L1 strong19 ≥ 14/19 → l'instance SURVIT à la règle linéaire : la thèse
   poids-rapides survit avec changement de règle (+ sélection à l'écriture).
 - L2-L4 : secondaires/exploratoires, rapportés tels quels (L2 ≥14 serait le résultat
   fort "sans masque" ; prédit improbable).
 - Zone grise L1 = 12-13 : étendre aux 25 paires valides AVANT conclusion.
 - L1 < 12 : les DEUX familles de règles perdent l'instance en ligne → chantier écriture
   clos pour de bon, le papier rapporte la grille 2×2 {règle}×{sélection} complète.
 - Sanité : permuté ≈ hasard sur tous les bras ; L1 comparé à la borne 16/19.
SELFTESTS : alignement masque→token = même fonction que l'oracle précédent (cité
fichier:ligne) ; apply linéaire vérifié analytiquement sur W construit à la main ;
déterminisme (2 exécutions bit-à-bit) ; stats tokens/chunk loggées. Mode STREAM.
PRÉDICTION (la mienne) : P(L1 ≥ 14/19) ≈ 55-60% ; P(L2 ≥ 14/19) ≈ 15-20% ;
L3/L4 : proches de L1/L2 respectivement.
GPU : 1 seul (7 si libre, pré-check dans la commande). Relecture de code par moi entre
selftests et run complet, comme pour l'oracle précédent.

## VERDICT DE L'ORACLE LINÉAIRE (2026-08-06 17:40, run ~50 min GPU 7, T0-T4 PASS) :
## PRIMARY FAIL, MAIS SIGNAL EXPLORATOIRE MAJEUR SUR L4
| bras | strong19 | p(strong19) | pair30 | p(30) | valid25 | permuté |
|---|---|---|---|---|---|---|
| L1 mean/sujet (PRIMARY) | 11.00/19 | 0.30 | 0.589 | 0.050 | 13.67/25 | 0.493 |
| L2 mean/tous | 9.33/19 | 0.46 | 0.444 | 0.79 | 11.33/25 | 0.496 |
| L3 delta/sujet | 10.67/19 | 0.38 | 0.533 | 0.30 | 13.33/25 | 0.481 |
| **L4 delta/tous** | **15.33/19** | **0.006** | **0.744** | **0.029** | **19.00/25** | 0.472 |
APPLICATION MÉCANIQUE : L1 = 11.0 < 12 → PRIMARY FAIL (ma prédiction 55-60% : perdue,
2e fois du jour). L4 = 15.33 ≥ 14, p=0.006, permuté sain, à 1 paire de la borne
d'instrument (16/19) : PREMIER franchissement du seuil par un état ÉCRIT, toutes
expériences confondues. Mécanisme cohérent : la delta rule n'écrit que l'ERREUR de
prédiction → le fond répétitif s'auto-supprime, le distinctif s'accumule : la règle
fait la sélection TOUTE SEULE (L4 > L3 : le masque la prive du contexte à soustraire).
Zéro masque requis = zéro problème de déploiement. STATUT LÉGAL : bras EXPLORATOIRE
(pré-enregistré secondaire, prédit 15-20%) → AUCUNE conclusion avant les 2 contrôles
scellés ci-dessous. Bonferroni ×7 bras testés en 2 jours : p_corr ≈ 0.043, limite.
Incident déclaré : TypeError au 1er lancement (glue main non couverte par T1-T4),
crash immédiat, fix 1 ligne, zéro donnée fausse possible.

# CONTRÔLES DE CONFIRMATION L4 (seuils scellés 2026-08-06 soir, AVANT exécution)
C1. CONTRÔLE SUJET-vs-DÉCOR (immédiat, ~10 min) : même CV, --target background
    (références fond seul) sur lin_delta_all. GRILLE SCELLÉE : VERT si
    strong19(sujet) − strong19(fond) ≥ 3 paires ; ROUGE sinon (mémoire de scène :
    le verdict original "enregistreur d'ambiance = ARRÊT" s'applique à L4 aussi).
    Référence d'instrument : hdirect_masked target=background pour comparaison.
C2. CONFIRMATION SUR PAIRES FRAÎCHES (pré-enregistrée, données jamais vues) :
    ~15 paires jumelles NOUVELLES (30 vidéos, prompts sans attributs INÉDITS,
    même pipeline --twin-offset), masques + planche numérotée labellisée par l'œil
    d'Aden AVANT tout score (sa loi), puis le MÊME sondage. SEUILS : sur les paires
    qu'Aden labellise fortes (attendu ~12-15) : binomial unilatéral p < 0.05 vs 0.5
    (ex. ≥10/12, ≥11/13, ≥12/15 selon le compte) ET hdirect_masked doit passer sa
    borne sur les mêmes paires (validité d'instrument sur le nouveau matériau).
    Le CV s'entraîne sur les 30 paires v2 et TESTE sur les nouvelles (zéro fuite).
    Disque : vidéos ~35GB gardées, h TRANSITOIRES (sig par vidéo puis purge du h).
PRÉDICTION (la mienne) : P(C1 vert) ≈ 60% ; P(C2 confirme | C1 vert) ≈ 65-70%.

ÉCHELLE DE LABELLISATION v3 (demandée par Aden 07/08, SCELLÉE AVANT ses réponses et
AVANT tout score) : 3 notes par paire, planche numérotée 31-51.
 M (masque) : M2 parfait / M1 partiel ou débordant / M0 rate le sujet.
 D (distinguabilité des jumeaux, AXE DÉCISIF) : D3 évidemment différents / D2 différents
   au détail / D1 très proches, un seul détail / D0 indiscernables.
 Q (qualité du sujet à l'image) : Q2 net / Q1 dégradé-petit-caché / Q0 raté.
RÈGLE DE CLASSEMENT SCELLÉE (mappe l'échelle sur les strates du protocole v2) :
 - FORTES (= le "strong19" de v2, sur lesquelles porte le seuil binomial) : D3 ou D2,
   ET M ≥ M1, ET Q ≥ Q1.
 - VALIDES (secondaire) : fortes + les D1 avec M ≥ M1 et Q ≥ Q1.
 - EXCLUES : tout D0, tout M0, tout Q0 (matériau impossible ou masque faux).
Le seuil de C2 reste : binomial unilatéral p < 0.05 vs 0.5 sur les FORTES, paramétré par
leur compte final (≥10/12, ≥11/13, ≥12/15, ≥13/17, ≥14/19, ≥15/21), ET hdirect_masked
doit passer sa borne sur les mêmes paires (validité d'instrument sur matériau neuf).
Fichier de réponses : reference/labels_v3_a_remplir.md → réponses recopiées ici.
PLANCHE ANNOTÉE (demande Aden "donne les infos des métriques pour que je réfléchisse
moins") : board/labels_board_v3_annot.jpg (1 ligne/paire, 4 vignettes, mesures écrites).
CALIBRATION DES SUGGESTIONS : TENTÉE ET ÉCHOUÉE, donc AUCUNE note D suggérée (règle :
jamais de valeur inventée). Sur les 30 paires v2 labellisées par Aden, la ressemblance
DINO automatique entre jumeaux vaut 0.798 (moy) sur ses paires FORTES vs 0.792 sur ses
FAIBLES vs 0.669 sur ses INVALIDES : elle ne sépare pas son jugement (elle l'ordonne
même à l'envers sur les invalides). 5e précédent du projet "les métriques ratent ce que
son œil voit" : MATÉRIAU PAPIER (justifie le protocole de labellisation humaine).
Seules les mesures masque/visibilité sont affichées (aire du sujet, confiance de
détection, stabilité du suivi, alerte segmentation) ; la ressemblance auto est imprimée
mais marquée explicitement NON PRÉDICTIVE.

## LABELS v3 D'ADEN (reçus 2026-08-07, SCELLÉS AVANT tout score ; verbatim + classement)
31 M2D3Q2 | 32 M0D3Q1 | 33 M2D1Q2 | 34 M0D3Q2 "rate le sujet A mais pas le sujet B" |
35 M2D3Q2 | 36 M2D2Q2 | 37 M2D1Q2 | 38 M0D3Q2 "attrape la glace sur la première" |
39 M2D3Q2 | 40 M2D3Q2 | 41 M2D3Q2 | 42 M1D3Q2 | 43 M1D1Q2 "le premier ça prend le lama
derrière" | 44 M2D3Q2 | 45 M1D3Q1 "ça attrape la planche de surf aussi avec le jumeau B
et on voit quasi rien du jumeau B" | 46 M2D3Q2 | 47 M2D2Q1 "on voit que le vêtement du
travailleur, il est un peu différent tout de même" | 48 M2D2Q2 | 49 M2D2Q1 | 50 M2D3Q2 |
51 M1D3Q2 "attrape que le bateau sur le jumeau A, attrape bateau et le mec qui conduit
sur le jumeau B"
CLASSEMENT PAR LA RÈGLE SCELLÉE (D≥2 ET M≥1 ET Q≥1 = forte ; D1 valide ; D0/M0/Q0 exclue) :
 FORTES (15) = 31,35,36,39,40,41,42,44,45,46,47,48,49,50,51
 FAIBLES (3, D1) = 33,37,43   →  VALIDES = 18
 EXCLUES (3, M0 : le masque rate le sujet) = 32,34,38
SEUIL C2 ACTIVÉ (paramétré par le compte, écrit d'avance) : **≥ 12/15 paires fortes**
(binomial unilatéral P(X≥12 | p=0.5) = 0.0176 < 0.05 ; 11/15 = 0.059 ne passe pas).
VALIDITÉ D'INSTRUMENT exigée sur le même matériau : hdirect_masked doit passer sa borne
sur ces 15 paires (sinon protocole invalide, comme v1).
ANALYSE SECONDAIRE PRÉ-DÉCLARÉE (robustesse, ne remplace pas le PRIMARY) : masques
ASYMÉTRIQUES signalés par Aden (45 planche+sujet, 51 bateau seul vs bateau+homme,
43 deux lamas) → re-score en les excluant ; rapporté quel que soit le résultat.
CORRECTION D'ARITHMÉTIQUE (07/08, AVANT le run, relevée à la relecture) : 43 est classée
FAIBLE (D1) par la règle scellée, donc déjà hors des 15 fortes ; l'exclusion ne retire que
45 et 51 → **13 paires** (31,35,36,39,40,41,42,44,46,47,48,49,50), seuil scellé
**≥ 10/13** (binomial unilatéral 0.0461 < 0.05). Le "12 paires / ≥10/12" écrit plus haut
était une erreur de comptage de ma part, corrigée avant tout score, sans changer la règle
(seuil = premier k tel que p < 0.05, appliqué au compte réel).
ÉCART GPU DÉCLARÉ (collecte h v3) : au lancement, un autre utilisateur occupait 3-7 et
0/1/2 étaient libres ; l'agent a lancé sur 0 et 1 (raison de l'interdiction disparue,
marge >25GB respectée, aucune collision). Risque associé (carte différente) réfuté par le
contrôle d'identité : 42/42 vidéos bit-à-bit identiques aux vidéos labellisées.

## VERDICT C2 v1 (2026-08-07, entraînement v2 → test v3) : **PROTOCOLE INVALIDE**
| source | strong15 | p binom (pire graine) | par graine | valid18 | permuté |
|---|---|---|---|---|---|
| lin_delta_all | 9.67/15 | 0.696 | [11,7,11] | 12.33/18 | 0.510 |
| hdirect_masked (BORNE) | 9.33/15 | 0.500 | [10,10,8] | 12.00/18 | 0.498 |
LECTURE LÉGALE : la condition de validité d'instrument scellée ("hdirect_masked doit
passer sa borne sur ces 15 paires") ÉCHOUE (9.33/15, p=0.50). Le test ne peut donc RIEN
dire du bras testé : matériau/protocole invalide, exactement le cas v1. Le 9.67/15 de
lin_delta_all n'est NI une confirmation NI une réfutation.
DIAGNOSTIC (le plus probable, à départager par C2 v2) : (a) le transfert INTER-CORPUS
échoue pour les DEUX sources (la tête apprise sur v2 ne se transporte pas), (b) et/ou le
matériau v3 est intrinsèquement plus dur : 14/21 sujets sont des HUMAINS (v2 était plus
varié) et les références DINO de crops humains se ressemblent (vêtement/pose), alors que
l'œil d'Aden, lui, les sépare (D3 sur la plupart) : nouveau cas "métrique ≠ œil".
NOTE D'HONNÊTETÉ : le verdict v2 (15.33/19) était une validation croisée INTERNE à v2
(entraînement et test sur paires disjointes du même corpus) ; C2 exigeait en plus le
transfert inter-corpus, condition PLUS DURE que celle qui avait produit le 15.33.

## C2 v2 : RÉPLICATION À PROTOCOLE IDENTIQUE (scellée 07/08 AVANT exécution)
Répéter sur le matériau v3 EXACTEMENT le protocole qui a produit le verdict v2 :
validation croisée INTERNE à v3 (mêmes plis par paire, pool, sans pré-entraînement,
3 graines), sources lin_delta_all ET hdirect_masked.
SEUILS SCELLÉS : (1) VALIDITÉ : hdirect_masked ≥ 12/15 sur les fortes (même règle
binomiale) : sinon le matériau v3 est déclaré INEXPLOITABLE (comme v1) et le verdict v2
reste le seul résultat, explicitement étiqueté "à répliquer sur matériau mieux apparié"
dans le papier ; (2) CONFIRMATION : lin_delta_all ≥ 12/15 ; (3) si l'instrument passe et
lin_delta_all échoue : la découverte v2 est RÉFUTÉE (cap rédaction, grille complète).
Aucune autre variante ne sera essayée : cette réplication est le DERNIER test du chantier
mémoire, quel que soit son résultat (garde-fou anti-chemins-multiples ; 3 tests déjà
courus aujourd'hui, tout p ultérieur serait ininterprétable).
PRÉDICTION (la mienne) : P(instrument ≥12/15) ≈ 45% ; P(lin_delta_all ≥12/15 | instrument
passe) ≈ 55%.

## VERDICT C2 v2 (2026-08-07 16h) : MATÉRIAU v3 DÉCLARÉ INEXPLOITABLE
CV interne à v3 (protocole identique à celui du verdict v2) :
 - hdirect_masked (BORNE DE VALIDITÉ) : **7.67/15** (acc 0.511) = HASARD → échec de la
   condition (1) : matériau v3 INEXPLOITABLE, comme v1. Le même instrument faisait 16/19
   sur v2. Rien ne peut être conclu sur lin_delta_all à partir de v3.
 - lin_delta_all : 5.67/15 (sous le hasard), non interprétable par la règle ci-dessus.
CAUSE la plus probable (concordante avec C2 v1) : 14/21 sujets HUMAINS ; les références
DINO de crops humains du même métier sont quasi colinéaires (vêtement/pose), alors que
l'œil d'Aden les sépare (D3 sur la plupart) : le matériau est bon pour un humain, hors de
portée de la référence visuelle utilisée. Le corpus v2 (animaux/objets/humains variés)
était, lui, à portée. LEÇON DE PROTOCOLE (matériau papier) : la borne de validité doit
être mesurée sur le matériau AVANT de l'utiliser comme test de confirmation ; ici elle a
sauvé une conclusion fausse dans les deux sens.

## C2 v3 : MATÉRIAU v4 (scellé 07/08 AVANT génération) : même test, matériau à portée
## de l'instrument. Ce n'est PAS une variante du test (protocole identique au bit près) :
## c'est le remplacement d'un matériau que la mesure a déclaré illisible.
MATÉRIAU : ~24 paires jumelles, sujets ANIMAUX / VÉHICULES / OBJETS DISTINCTIFS (zéro
humain : cause identifiée de l'échec v3), prompts sans attributs, aucun recoupement de
catégorie avec v2 NI v3, même pipeline (--twin-offset, single 90f), planche numérotée
61-84 labellisée par l'œil d'Aden (échelle M/D/Q) AVANT tout score.
ORDRE DES OPÉRATIONS, NON NÉGOCIABLE (c'est lui qui empêche le biais de sélection de
matériau) : (1) labellisation d'Aden scellée ; (2) hdirect_masked SEUL est scoré : c'est
une propriété des ACTIVATIONS, indépendante de toute mémoire, donc le mesurer d'abord ne
peut pas favoriser le bras testé ; (3) SI ET SEULEMENT SI l'instrument passe son seuil,
lin_delta_all est scoré, une seule fois. Si l'instrument échoue : matériau écarté, FIN du
chantier (deux échecs de matériau = la limite de la méthode DINO-crop est le résultat, et
il s'écrit tel quel dans le papier).
SEUILS (binomial unilatéral p<0.05 sur les paires FORTES d'Aden, paramétré par leur
compte : ≥12/15, ≥13/17, ≥14/19, ≥15/21, ≥16/23, ≥17/24) : instrument ET bras testé.
Protocole de score : CV interne 6 plis par paire, pool, sans pré-entraînement, 3 graines,
permuté 25 dérangements : IDENTIQUE aux runs v2/v3, aucun réglage nouveau.
PRÉDICTION (la mienne) : P(instrument passe sur v4) ≈ 70% (matériau choisi dans le régime
où il a déjà démontré 16/19) ; P(lin_delta_all passe | instrument passe) ≈ 50%.
C'EST LE DERNIER MATÉRIAU DU CHANTIER : quel que soit le résultat, aucune 4e fournée.

## ARM MLP-CORRECTEUR (scellé 07/08 AVANT implémentation ; question d'Aden : "faire une
## delta rule avec le MLP, on a déjà essayé ?" → JAMAIS)
CONSTAT DE CODE (video_memory.py:140 `swiglu_fast_weight_gradient`) : notre écriture
MONTE le produit scalaire <f_W(k), v> ("No minus sign: the fast weight ASCENDS this dot").
C'est une règle ASSOCIATIVE (Hebb), PAS correctrice d'erreur. Son analogue linéaire exact
est l'accumulation moyenne : 9.33/19 = hasard, comme notre MLP (8/19). La delta rule
(15.33/19) est la seule testée qui soustrait ce qu'elle prédit déjà.
RÉ-INTERPRÉTATION DE TOUTE LA GRILLE : l'axe discriminant n'est pas linéaire↔MLP mais
ASSOCIATIF ↔ CORRECTEUR D'ERREUR, et il traverse les deux architectures. Prédiction
falsifiable qui en découle : un MLP correcteur doit se comporter comme la delta rule.
INTERVENTION (rejeu, zéro entraînement, une ligne) : dans le gradient, remplacer la cible
v par le RÉSIDU (v − f_W(k)) → l'objectif devient ½||f_W(k) − v||², version non linéaire
exacte de la delta rule. Deux bras : `mlp_delta_all` (tous tokens, le miroir de L4) et
`mlp_delta_subject` (masqué, miroir de L3). Reste : Muon, normalisation, âges, sondage,
plis, corpus v2 : TOUT identique aux runs scellés.
SEUIL : ≥14/19 (même règle binomiale que l'oracle linéaire ; 15.33 pour L4).
STATUT : EXPLORATOIRE et déclaré comme tel (8e bras testé en 3 jours ; tout p doit être
lu avec une correction de multiplicité, Bonferroni ×8 ⇒ p_corr = 8·p). Ce bras ne peut
PAS remplacer la réplication v4 : il éclaire le MÉCANISME, il ne confirme pas la
découverte. Ordre de priorité GPU : v4 (réplication) d'abord, ce bras ensuite.
PRÉDICTION (la mienne) : P(mlp_delta_all ≥14/19) ≈ 55% ; si ça passe, la thèse du papier
devient "c'est la correction d'erreur qui écrit l'identité, indépendamment de
l'architecture" ; si ça échoue avec L4 qui passe, la thèse devient "la linéarité importe
aussi" et le papier le dit tel quel.

## AMENDEMENT (Aden 07/08) : « corrige TOUT d'un coup côté MLP, pas de re-test à
## l'infini » → PLAN FACTORIEL EN UNE SEULE RONDE (scellé AVANT exécution)
Les 3 différences entre notre écriture et la delta rule linéaire sont corrigées ENSEMBLE,
et les ablations tournent dans la MÊME fournée : une ronde répond à "ça marche ?" ET à
"grâce à quoi ?".
FACTEURS : (R) objectif = résidu v−f_W(k) [vs Hebb <f,v>] ; (S) mise à jour SÉQUENTIELLE
par micro-lots dans le chunk [vs un seul pas moyenné sur tout le chunk] ; (M) gradient
BRUT [vs orthogonalisation Muon, qui efface l'amplitude = la surprise].
LES 5 CELLULES (toutes sur TOUS les tokens : la delta rule linéaire a montré que le
masque nuit) :
 - A0 Hebb + moyenné + Muon = NOTRE ÉCRITURE ACTUELLE (déjà mesurée : 8.00/19)
 - A1 Résidu + moyenné + Muon (déjà implémenté, `mlp_delta_all`)
 - A2 **PRIMARY** Résidu + séquentiel + brut = l'analogue non linéaire COMPLET de la
      delta rule (`mlp_full`)
 - A3 Résidu + séquentiel + Muon (isole l'effet de l'orthogonalisation)
 - A4 Hebb + séquentiel + brut (isole l'effet de la correction d'erreur)
SEUIL : A2 ≥ 14/19 (même règle binomiale). A1/A3/A4 descriptifs, rapportés tels quels.
CHOIX DE LA TAILLE DE MICRO-LOT ET DU PAS : fait AVANT tout score d'identité, sur 2
vidéos, par un critère AVEUGLE À L'IDENTITÉ, scellé ici : maximiser la réduction relative
moyenne de ‖f_W(k)−v‖ par écriture, sous contrainte de sanité numérique (états dans la
dynamique fp16, pas d'explosion, pas d'effondrement à 0). Aucun réglage ne peut donc être
choisi "parce qu'il donne un meilleur score d'identité". Le micro-lot est une
approximation ASSUMÉE du séquentiel pur (coût : ~2000-4600 tokens/chunk), déclarée telle
quelle ; si le pur séquentiel tient dans le budget sur un sous-échantillon, il est mesuré
en contrôle sur 5 vidéos et l'écart est rapporté.
MULTIPLICITÉ : 4 nouveaux bras ⇒ 12 bras testés au total ; le p de A2 sera rapporté brut
ET corrigé (Bonferroni ×12). Une seule ronde, aucune autre variante MLP ne sera testée.
PRÉDICTION (la mienne) : P(A2 ≥14/19) ≈ 60% ; P(A1 seul suffit) ≈ 35% ; si A2 passe et A4
échoue, la correction d'erreur est LA cause ; si A4 passe aussi, c'est la séquentialité.

### RÉPARATION DU CRITÈRE AVEUGLE (07/08, AVANT tout score d'identité ; défaut de MON
### critère, relevé par l'agent qui a ajouté la mesure manquante)
DÉFAUT : "maximiser la réduction relative de ‖f_W(k)−v‖" a un optimum DÉGÉNÉRÉ : un pas
minuscule n'écrit presque rien, donc ne dégrade presque rien, donc maximise le critère,
tout en laissant l'état ≈ son INITIALISATION, laquelle est IDENTIQUE POUR TOUTES LES
VIDÉOS. Un tel réglage rendrait l'échec du sondage inévitable et ininterprétable. Mesuré :
au réglage retenu par le critère brut, A2 n'écrit que 2.7-14 % de l'état, contre 88 %
pour notre écriture actuelle (A0) et 73 % pour A1.
RÉPARATION (elle reste AVEUGLE À L'IDENTITÉ : les deux quantités sont internes à
l'écriture) : optimisation SOUS CONTRAINTE : parmi les réglages dont la PART ÉCRITE
‖état − état_sans_écriture‖/‖état‖ ≥ 0.5 (l'écriture domine l'initialisation ; ancrage
non arbitraire : A0 = 0.88, A1 = 0.73, on exige le même régime que la baseline),
maximiser la réduction d'erreur. Extension de la plage de pas vers le bas : RATIFIÉE
(même critère, aveugle à l'identité).
CONSÉQUENCE, contraire à ma prédiction : la SEULE cellule qui écrit vraiment ET descend
son erreur est **A3 = résidu + séquentiel + MUON** (micro-lot 512 : part 0.952,
R = +0.025). Sans Muon, c'est l'un ou l'autre : soit l'écriture domine et l'erreur
explose (part 1.03, R = −0.45 à −48), soit l'erreur descend et l'écriture est
négligeable (part 0.29, R = +0.075). Mécanisme : l'orthogonalisation borne la taille du
pas, ce qui permet d'écrire beaucoup SANS diverger. Ma prédiction "Muon efface la
surprise, donc le brut sera meilleur" est donc RÉFUTÉE par une mesure aveugle à
l'identité, avant tout score : je la déclare perdue ici.
NOUVEAU PRIMARY : **A3** (`mlp_full_muon`, micro-lot 512), seuil inchangé ≥14/19 sur v2.
ABLATIONS (descriptives, même corpus, rapportées telles quelles) : A1 (résidu, moyenné,
Muon) ; A2' (résidu, séquentiel mb32, BRUT, pas ×0.5 : écrit mais ne descend pas =
l'ablation "sans Muon") ; A4 (Hebb, séquentiel, mb512 : l'ablation "sans correction
d'erreur") ; A0 déjà mesurée (8.00/19). Le pur séquentiel (micro-lot 1) est hors budget
(~3.9 h/bras) : NON couru, et ce plafond est déclaré, pas silencieux.
NOTE DE PROTOCOLE : le changement de PRIMARY intervient APRÈS avoir vu des données, mais
UNIQUEMENT des données aveugles à l'identité (part écrite et réduction d'erreur, deux
quantités internes à la règle d'écriture). Aucun score de ré-identification n'a été
calculé pour aucun bras MLP au moment de cette décision : les fichiers de sondage MLP
n'existent pas. C'est la garantie qui rend l'amendement légitime, et elle est vérifiable
(aucun cv_mlp_*.json sur disque à cette date).

## LABELS v4 D'ADEN (reçus 2026-08-07, SCELLÉS AVANT tout score ; verbatim + classement)
61 M2D1Q2 | 62 M2D2Q2 | 63 M2D0Q2 | 64 M2D0Q2 | 65 M2D2Q2 | 66 M2D2Q2 | 67 M2D0Q2 |
68 M2D2Q2 | 69 M2D0Q2 | 70 M2D2Q2 | 71 M1D0Q2 "jumeau A ça prend la vache derrière" |
72 M2D2Q2 | 73 M2D3Q2 | 74 M1D3Q2 "le deuxième ça prend pas le bus" | 75 M2D3Q2 |
76 M2D1Q2 | 77 M1D1Q2 "le premier masque loupe le jumeau A" | 78 M2D2Q2 | 79 M2D2Q2 |
80 M2D3Q2 | 81 M2D1Q2 | 82 M2D3Q2 | 83 M2D0Q1 "masque prend qu'une partie et c'est le
même" | 84 M2D3Q2
CLASSEMENT (règle scellée D≥2 ∧ M≥1 ∧ Q≥1) :
 FORTES (14) = 62,65,66,68,70,72,73,74,75,78,79,80,82,84
 FAIBLES (4, D1) = 61,76,77,81  →  VALIDES = 18
 EXCLUES (6) = 63,64,67,69,71,83  (dont **5 pour D0 : les jumeaux sont IDENTIQUES**)
SEUIL C2 ACTIVÉ pour 14 paires : **≥ 11/14** (binomial unilatéral 0.0287 < 0.05 ;
10/14 = 0.090 ne passe pas). Instrument ET bras testé, même seuil, ordre inchangé
(hdirect_masked SEUL d'abord).
CONSTAT MAJEUR (matériau papier, relevé par Aden : "les animaux se ressemblent trop") :
le générateur CLONE les animaux d'espèce étroite (éléphant, renard, dauphin, papillon,
vache : D0) et VARIE les humains (v3 : D3 sur la majorité). Effet ciseaux du projet :
là où le matériau est riche (humains) l'instrument DINO-crop est aveugle ; là où
l'instrument voit (animaux/objets) le matériau est dégénéré. Leçon de conception :
préférer les catégories SUPERORDONNÉES à forte variabilité intra-classe (chien, oiseau,
voiture, fleur) aux espèces étroites (girafe, éléphant), et changer d'instrument pour
les humains (embedding facial) plutôt que de changer de sujets.

## VERDICT INSTRUMENT v4 (07/08) : 7.67/14 = HASARD... MAIS INCOHÉRENCE DE PROTOCOLE
## DÉTECTÉE AVANT TOUTE CONCLUSION (erreur de MA consigne)
Instrument sur v4 : strong14 7.67/14 (0.548), p=0.605, permuté 0.511 ; seuil ≥11/14 non
atteint. MAIS : la borne de validité scellée (15.67/19 sur v2) a été mesurée AVEC
pré-entraînement de la tête (fichier `cv_hdirect_masked_subject_pool_pre_v2.json`,
`pretrain=True`, 105 vidéos switch hors paires), alors que les instruments v3 ET v4 ont
été courus SANS pré-entraînement (ma consigne, pour apparier le protocole du bras testé).
Les deux verdicts "matériau inexploitable" (v3 puis v4) reposent donc sur une comparaison
POMMES/ORANGES : on exige d'un instrument non pré-entraîné le score d'un instrument
pré-entraîné. La cellule manquante n'a JAMAIS été mesurée : hdirect_masked sur v2 SANS
pré-entraînement.
TEST DE COHÉRENCE (lancé immédiatement, aveugle au bras testé : il ne concerne que la
lecture directe des activations, sans aucune mémoire) :
 (i) hdirect_masked, v2, SANS pré-entraînement. Si le score s'effondre aussi (≈ hasard),
     alors le pré-entraînement est une CONDITION DE VALIDITÉ de l'instrument, les
     verdicts v3/v4 sont NULS et non avenus, et il faut re-courir les instruments v3 et
     v4 AVEC pré-entraînement avant toute conclusion sur le matériau.
 (ii) Si au contraire il reste ≥14/19 sans pré-entraînement, alors l'instrument est bien
     valide sans lui, et les verdicts "matériau inexploitable" v3/v4 tiennent tels quels.
RÈGLE SCELLÉE AVANT DE VOIR (i) : quel que soit son résultat, le bras testé
(lin_delta_all) sera scoré sur v4 UNIQUEMENT si l'instrument passe ≥11/14 dans la MÊME
configuration de pré-entraînement que celle où il sera scoré. Aucune configuration ne
sera choisie après avoir vu un score d'identité.
PRÉDICTION (la mienne) : P(l'instrument s'effondre sans pré-entraînement) ≈ 65% (105
vidéos supplémentaires contre 16-20 paires d'entraînement, l'écart de données est énorme,
et le diagnostic train/test a montré que tout se joue sur la généralisation).

### RÉSULTAT DU TEST DE COHÉRENCE (07/08 soir) : ma prédiction gagnée à moitié, les
### verdicts de matériau TIENNENT quand même
| corpus | seuil | SANS pré-entraînement | AVEC pré-entraînement |
|---|---|---|---|
| v2 | ≥16/19 | 12.00/19 (p=0.32) | **15.67/19 (p=0.032)** |
| v3 | ≥12/15 | 7.67/15 | 10.67/15 (p=0.30), graines [9,11,12] |
| v4 | ≥11/14 | 7.67/14 | 8.67/14 (p=0.61), graines [7,10,9] |
Le pré-entraînement vaut +3.67 paires sur v2, +3.00 sur v3, +1.00 sur v4 : c'est bien une
CONDITION DE VALIDITÉ de l'instrument (ma consigne était fautive). MAIS, dans la
configuration CORRECTE, v3 et v4 échouent quand même leur seuil : **les deux verdicts
"matériau inexploitable" TIENNENT**, ils n'étaient pas des artefacts de ma consigne.
Structure des corpus vérifiée identique (single 90f, 25 écritures) : la différence est le
CONTENU, pas le format.
### DÉCOUVERTE MAJEURE (et correction d'une erreur de cadrage qui date du 05/08)
À armes égales sur v2 (sans pré-entraînement, mêmes plis) : **la MÉMOIRE delta-rule
(15.33/19) LIT MIEUX que la lecture directe des activations masquées (12.00/19)**.
Donc `hdirect_masked` n'est PAS une borne supérieure : c'est un autre sondeur, avec sa
propre efficacité en données, et il est MOINS bon que l'état delta-rule à données égales.
Mécanisme cohérent avec notre hypothèse : la delta rule retire la composante PRÉDICTIBLE
(le fond, le mode commun), ce qui est une suppression du fond plus forte que le masquage
spatial, lequel laisse le contexte de scène déjà mélangé dans les tokens du sujet par
l'attention. Conséquence de cadrage à corriger PARTOUT (papier compris) : parler de
"référence" et non de "borne d'instrument" ; le 16/19 n'a jamais été un plafond.
### CE QUE CELA IMPLIQUE POUR LA SUITE (aucun score d'identité n'est engagé ici)
Le maillon faible de la chaîne de mesure n'est ni le matériau ni le sondeur : c'est la
RÉFÉRENCE (embedding DINO d'un crop de sujet). Sur v4, l'œil d'Aden juge 14 paires
distinguables (D2/D3) alors que la référence échoue : 6e précédent "l'œil voit ce que la
métrique rate". La voie propre n'est donc PAS de re-générer un 4e matériau, mais de
RÉPARER LA RÉFÉRENCE, avec un critère lui aussi aveugle au bras testé : l'accord de
l'instrument avec les labels D d'Aden. Candidats (tous sans régénération, sur les crops
déjà sur disque) : embedding facial (facenet installé et testé le 07/08) pour v3 ; DINOv2
plus grand / CLIP image / ensemble pour v4 ; crop plein cadre vs crop serré.
RÈGLE SCELLÉE : l'instrument peut être amélioré autant qu'on veut TANT QU'ON NE REGARDE
QUE (a) son accord avec les labels d'Aden et (b) ses contrôles permutés ; le bras testé
n'est scoré QU'UNE FOIS, après gel de la référence. Toute amélioration de référence sera
listée au papier, y compris celles qui n'ont pas marché.

## VARIABILITÉ DE CHAÎNE (07/08 soir) : ±2 PAIRES SUR strong19 — À DÉCLARER AU PAPIER
Contrôle de device sur le bras ANCRE `lin_delta_all` (config identique, mêmes graines) :
 - GPU (chiffre scellé) : 15.33/19, p=0.0062, graines [16,16,14]
 - CPU (contrôle)       : 13.33/19, p=0.057,  graines [13,12,15]
Les hyperparamètres retenus diffèrent sur les 6 plis ; les signes de marge ne concordent
que sur 23/30 paires. CONSÉQUENCE : la variabilité de bout en bout de notre chaîne est
d'environ **2 paires sur 19**, du même ordre que les écarts qu'on cherche à interpréter.
 (a) Un écart de 1-2 paires entre deux bras N'EST PAS interprétable. Les seuils scellés
     restent la règle de décision, mais tout verdict "juste au-dessus" ou "juste en
     dessous" doit être déclaré fragile.
 (b) Le CONTRASTE qui porte le projet survit largement : 15.33 ou 13.33 contre 8.00 pour
     l'écriture actuelle, c'est 5 à 7 paires d'écart, très au-delà du bruit de chaîne.
 (c) ANNEXE DE ROBUSTESSE déclarée MAINTENANT (avant tout score MLP) : les deux bras
     ancres (A0 `state` et `lin_delta_all`) seront re-courus à 10 graines, distribution
     complète rapportée (médiane + étendue), et le papier citera cette distribution, pas
     un point unique. Rien n'est retiré : les runs 3 graines scellés restent la référence.
Artefact : runs/ctrl_device_cpu_lin_delta_all_v2.json.
ASSOUPLISSEMENT DE LA RÈGLE GPU (motivé, 07/08) : la marge ≥25GB visait les runs de
génération/entraînement. Le sondage tient dans <2GB (240 signatures × 1536 nombres) : un
co-run sur une carte à ≥8GB libres est autorisé pour les SONDAGES uniquement, jamais pour
la génération. Motif : les 8 cartes portent des jobs d'autres utilisateurs depuis 35 min
et le blocage n'a aucune justification technique.

## RÈGLE DE CONCEPTION PERMANENTE (imposée par Aden, 08/08, après une erreur de ma part)
Aden a relevé que la campagne « échelle de fidélité » avait été conçue sur le SUJET
D'OUVERTURE (A→B→A), c'est-à-dire un sujet épinglé dans le sink, que le modèle peut faire
revenir SANS aucune mémoire. Le terrain informatif (sujet non-ouvrant) est documenté au
registre depuis juillet : je l'avais et ne l'ai pas appliqué.
**RÈGLE, à vérifier AVANT toute génération, sans exception :**
 1. Écrire ce que produirait le bras NUL (sans mémoire, sans insertion). **Si le bras nul
    peut réussir, le design est invalide** : on change de terrain, on ne lance rien.
 2. Prouver mécaniquement, pas supposer, que le signal testé est le seul chemin possible
    (ici : qu'aucune frame du segment cible ne subsiste dans le contexte au retour).
 3. Relire le registre pour le régime concerné AVANT d'écrire la spec, pas après.
Cette règle prime sur la vitesse : une campagne de 3 h lancée sur un terrain faux coûte
plus cher que 10 minutes de vérification.

## ÉPREUVE DE GALERIE (07/08 soir) : CE QUE MESURE VRAIMENT NOTRE ÉCHELLE
Protocole : même têtes que les runs scellés (reproduites, 0 inversion de signe sur 720
lignes/bras), récupération du sujet dans une galerie, 3 niveaux.
| bras | 1/2 | 1/10 | rang méd. /60 | rang méd. de la JUMELLE /60 |
|---|---|---|---|---|
| lecture directe PRÉ (réf. 15.67/19) | 0.550 | **0.183** | **9.0** | 9.0 |
| lecture directe sans pré | 0.528 | 0.117 | 12.5 | 13.0 |
| état delta L4 | **0.583** | 0.078 | 24.0 | 25.0 |
| MLP correcteur A3 | 0.578 | 0.044 | 23.5 | 24.0 |
| écriture actuelle | 0.456 | 0.011 | 26.5 | 27.0 |
(hasard : 0.50 / 0.10 / rang 30.5 ; permuté : rang 30.6-31.2 partout)
DEUX CONCLUSIONS, à des étages différents, toutes deux à écrire au papier :
 1. **L'IDENTIFICATION ABSOLUE est hors de portée de TOUTE notre chaîne**, mémoire ou
    pas : la meilleure lecture (activations directes) ne place son sujet en tête qu'à
    2.2% (hasard 1.7%) et donne à la JUMELLE exactement le même rang médian (9.0) qu'au
    vrai sujet. Ce qui survit dans la meilleure lecture possible, c'est le VOISINAGE
    (catégorie/apparence dictée par le prompt), jamais l'instance. ⇒ CORRECTION DE
    FORMULATION OBLIGATOIRE PARTOUT : notre résultat par paires mesure une
    DISCRIMINATION ENTRE DEUX CANDIDATS TRÈS PROCHES, pas une identification. Toute
    phrase du type "la mémoire sait qui c'était" est à proscrire.
 2. **Le voisinage, lui, est bien perdu par l'ÉCRITURE** : activations 9.0 de rang médian
    contre 23.5-26.5 pour les états. Écart de 11-15 rangs, très au-delà du bruit : la
    compression d'écriture détruit l'essentiel de ce que les activations portent encore.
 3. Cohérent avec la découverte du 07/08 : à 1-parmi-2 l'état delta (0.583) reste AU
    DESSUS de la lecture directe pré-entraînée (0.550). Discrimination fine et placement
    en voisinage sont deux capacités distinctes, portées différemment.
CONSÉQUENCE OPÉRATIONNELLE : la campagne vidéo "insertion de l'image récupérée" est
ANNULÉE dans sa forme prévue (le rang-1 est un mauvais sujet dans ~100% des cas, pour
TOUTES les sources : les bras mémoire et permuté seraient indiscernables par
construction, on ne testerait que le canal). Si une phase vidéo doit exister, elle devra
partir d'un oracle (image VRAIE insérée) pour calibrer le canal, avec mémoire vs permuté
en bras secondaires, ou attendre une référence réparée.
Artefacts : runs/retrieval_gallery_v2.json, board/retrieval_board_v2.jpg (+3 parties),
reference/runs/probe1/retrieval_board.py.

## ANNEXE DE ROBUSTESSE 10 GRAINES (déclarée avant les scores MLP, livrée 07/08 soir)
strong19, corpus v2, config identique aux runs scellés, 10 graines au lieu de 3 :
| bras | médiane | min | max | les 10 valeurs |
|---|---|---|---|---|
| écriture actuelle (state) | 7.0 | 6 | 10 | 10,7,7,8,8,8,7,7,7,6 |
| **règle delta (lin_delta_all)** | **13.5** | 11 | 16 | 16,16,14,13,12,11,14,12,14,12 |
| MLP correcteur (A3) | 10.5 | 6 | 14 | 12,12,12,9,6,14,11,6,9,10 |
LECTURE, sans complaisance :
 (a) Le 15.33/19 des 3 graines scellées était le HAUT de la distribution : la médiane à
     10 graines est **13.5**, et le seuil de significativité (14/19) n'est atteint que par
     4 graines sur 10. **Le résultat phare doit donc être présenté comme une
     distribution (médiane 13.5, étendue 11-16), jamais comme "15.33"**, et sa
     significativité comme marginale au niveau des graines individuelles.
 (b) Le CONTRASTE, lui, est robuste : delta médiane 13.5 contre écriture actuelle 7.0,
     et les distributions NE SE RECOUVRENT QUE sur une valeur (delta min 11 > state max
     10). C'est la conclusion qui porte le papier, et elle survit.
 (c) Le MLP correcteur (médiane 10.5, étendue 6-14) est INSTABLE : il chevauche
     largement les deux autres. Confirme l'échec de son seuil et interdit toute
     affirmation sur la hiérarchie MLP-vs-linéaire au-delà de "le linéaire est
     au-dessus en médiane, avec recouvrement".
 (d) Cohérent avec la variabilité de chaîne mesurée par le contrôle de device (±2 paires).
Artefacts : runs/annex_seeds10_{state,lin_delta_all,mlp_full_muon}_subject_pool_v2.json.

## DIAGNOSTIC TRAIN vs TEST (07/08, à la demande d'Aden : "sur le set de train la mémoire
## est bonne ou pas ?") : instrumentation, aucun test d'hypothèse, aucun p nouveau
Sondeur entraîné sur les plis, évalué sur SES PROPRES paires d'entraînement puis sur le
test (6 plis × 2 graines, source lin_delta_all) :
 - corpus v2 : TRAIN 1.000  TEST 0.567
 - corpus v3 : TRAIN 1.000  TEST 0.479
(cfg fixées à la main pour le diagnostic, pas celles de la recherche d'hyperparamètres :
les chiffres de TEST ne remplacent pas ceux des runs scellés, seul le contraste compte.)
LECTURE : la tête AJUSTE PARFAITEMENT ses paires d'entraînement dans les deux corpus →
ce n'est jamais un problème de capacité, d'optimisation ni de "pas assez entraîné" ; tout
se joue sur la GÉNÉRALISATION. Corollaire : (a) le 15.33/19 de v2 était bien de la
généralisation (paires tenues à l'écart), (b) le par-cœur parfait explique le 8/8 du toy
et l'égalité écrivain-gelé, (c) plus de données d'entraînement de la TÊTE ne changerait
pas la nature du problème, seule la qualité du matériau de test le peut.

AMENDEMENT C2 (06/08 soir, AVANT toute labellisation/score) : la règle anti-recoupement
n'a laissé que 13 paires licites sur les 20 candidats → une 2e vague de candidats
inédits est générée dans la foulée, cible totale ~20 paires. Tout est labellisé par
l'œil d'Aden AVANT tout score ; le seuil reste le binomial p<0.05 paramétré par le
compte de paires fortes qu'il déclarera. Vague 1 : 13 prompts (performer, mail carrier,
rooster, fishing boat, barista, ballet dancer, motorcyclist, ice cream vendor, hot air
balloon, police officer, peacock, bus driver, llama), ids 9100000-12, --no-h (h
transitoires demain), GPUs 7+3.

## VERDICT C1 (2026-08-06 ~19h, GPU 7, lancé par moi après stall de l'agent) : VERT
lin_delta_all --target background : strong19 = 8.67/19 (0.456), pair30 0.433, p=0.93,
permuté 0.484-0.501 = hasard total sur le DÉCOR, contre 15.33/19 (p=0.006) sur le SUJET.
Écart sujet−fond = 6.67 paires ≥ 3 : DOUBLE DISSOCIATION nette : l'état delta-tous lit
spécifiquement le sujet, pas la scène. (Artefact : runs/cv_lin_delta_all_background_
pool_v2.json, log c1_bg_control.log.) → C2 (paires fraîches) ENGAGÉ : génération
nocturne 15 paires v3, planche œil Aden demain AVANT tout score.
Si C1 ROUGE : L4 = mémoire de scène, retour au cap rédaction, grille 2×2 au papier.
Si C1 vert et C2 confirme : la thèse poids-rapides est VIVANTE avec règle delta
sans masque, training-free : bascule vers lecture/décodage (post-15/08 : décision
Aden sur le calendrier papier).
