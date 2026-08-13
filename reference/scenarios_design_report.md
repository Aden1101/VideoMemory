# MATRICE DE SCÉNARIOS MULTI-SWITCH + BASELINE FRAME-BANK — RAPPORT DE DESIGN (2026-07-29)

Implémentation, selftests et smoke uniquement. **La campagne n'est pas lancée** : les chiffres
ci-dessous sont des coûts et des calibrations, pas des résultats de re-ID.

| Livrable | Fichier |
|---|---|
| driver N-switch (additif, `inference()` INTACTE) | `pipeline/switch_causal_inference.py` : `inference_schedule()`, `_switch_recache()` |
| banque de frames Echo-Forcing-like | `pipeline/switch_causal_inference.py` : `_fb_mode/_fb_reset/_fb_key/_fb_store/_fb_select` |
| scénarios, triples, paraphrases, table des bras, scoring 1ère apparition | `reference/runs/reid_scen/scenarios.py` |
| driver de campagne (run / score / summary) | `reference/runs/reid_scen/run_reid_scen.py` |
| selftests obligatoires | `reference/runs/reid_scen/selftest_scenarios.py` (**52/52 PASS**) |
| calibration du seuil de routage | `reference/runs/reid_scen/prompt_key_matrix.py` + `results/prompt_key_matrix.json` |
| lanceur shardé (pré-check GPU inclus, refus > 4 GPU et idx 6) | `reference/runs/reid_scen/launch_scen.sh` |
| smoke (vidéos + planches PNG) | `reference/runs/reid_scen/smoke/`, logs `reference/runs/reid_scen/logs/` |
| mp4 (disque projet, home à 100 %) | `/shared/ssd_14T_project/adenelmi/reid_scen/videos/` |

Rien n'est commité. Aucun fichier existant n'a été modifié en dehors d'ajouts dans
`pipeline/switch_causal_inference.py` (méthodes nouvelles uniquement).

---

## 1. ECHO-FORCING : LE MÉCANISME RÉEL (sourcé)

**Titre exact** : *Echo-Forcing: A Scene Memory Framework for Interactive Long Video Generation*,
arXiv **2605.16003** (mai 2026). Auteurs : Mingqiang Wu, Weilun Feng, Zhefeng Zhang, Haotong Qin,
Yuqi Li, Guoxin Fan, Xiaokun Liu, Zhulin An, Libo Huang, Yongjun Xu, Chuanguang Yang.

Ce que le papier fait **réellement** (vérifié sur `arxiv.org/html/2605.16003v1`, §3.1 Hierarchical
Temporal Memory, §3.2 Scene Recall Frames, §3.3 Difference-aware Memory Decay, annexes A.2 et B) :

- **Espace de stockage = KV, pas des frames.** Le nom « Scene Recall Frame » est trompeur : une
  recall frame est une **compression spatiale de KV RÉELS**. Pour chaque scène, M blocs candidats
  de sa phase stable sont fusionnés position par position par un softmax :
  `α_{s,j,u} = Softmax_j(e_{s,j,u})`, `K^rec_{s,u} = Σ_j α_{s,j,u} K^raw_{s,j,u}` (Eq. 7). Le
  résultat est une paire K/V « spatially structured », pas une image.
- **Mémoire hiérarchique** : ancres stables (N_anc = **12** ancres roulantes) + historique
  compressé (N_cmp = **3** frames, sélection par « Drift-Gated Phase Compression » : score
  d'alignement de phase entre clés historiques et requêtes calibrées, plus une porte de dérive)
  + fenêtre récente (N_rec = **3**, fenêtre locale L = 21), le tout sous RoPE **relatif**.
- **Banque de scènes qui CROÎT** : « Historical Scene Recall Frames are stored in a scene memory
  pool », sans borne de capacité annoncée → O(nombre de scènes), pas O(1).
- **Routage automatique (annexe B)** : **cosinus entre le prompt courant et chacun des prompts
  précédents**. Trois modes : *smooth* si le meilleur match est la scène immédiatement précédente
  et dépasse τ_smooth ; **recall si le meilleur match est une scène NON ADJACENTE et dépasse
  τ_rec** ; *hard cut* sinon. Les valeurs numériques de τ ne sont pas données dans le papier.
- **Injection** : les KV de rappel sont **concaténés à la mémoire d'ancres** (donc lus par
  l'attention comme des tokens de contexte ordinaires).
- **Training-free, backbone gelé**, appliqué à Self-Forcing et LongLive. Éval « scene recall »
  = segments appariés (**plans 4-6 vs plans 1-3**), subject + background consistency ;
  29.47 → 32.58 sur l'alignement texte, imaging quality 70.48 → 72.83 à 120 s.

**Conséquence pour nous.** C'est bien la baseline obligatoire (cf.
`reference/frame_recall_preemption_check.md` §2.1). Notre portage garde **l'esprit** (banque qui
croît, une entrée par scène, routage par cosinus de prompt, contrainte de non-adjacence, seuil,
injection dans le cache au switch) et change **le substrat** : nous stockons **la dernière frame
latente** du segment sortant et la splicions à la position récence — exactement la mécanique
d'`oracle1`, ce qui rend le bras directement comparable à nos bornes existantes. Nous ne
réimplémentons ni la fusion spatiale multi-blocs, ni le RoPE relatif, ni la décroissance
différentielle : ce sont des optimisations de fidélité de scène, pas le mécanisme de rappel, et
notre substrat AF a déjà son propre schéma positionnel (le splice à la récence est la seule
position qui marche chez nous, courbe de position du 2026-07-24).

---

## 2. CE QUI A ÉTÉ IMPLÉMENTÉ

### 2.1 Driver N-switch — additif, prouvé bitwise

`inference()` n'est **pas touchée** : les campagnes existantes restent reproductibles par
construction. La généralisation est une méthode **nouvelle**, `inference_schedule(noise, prompts,
schedule, subjects=None, ...)` où `schedule = [(frame_de_départ, prompt_idx), ...]`. Un segment
dont le **sujet** est déjà apparu est un REVISIT : les hooks de rappel se déclenchent et l'ancre
est prise dans le segment de sa **première apparition** (`a_end - 1`, généralisation exacte de
`self._first_switch_frame`). Les écritures mémoire continuent sur tout le rollout et le splice a
lieu à **chaque** retour (vérifié : S4 déclenche aux segments 2 et 4).

Le selftest (a) rejoue le schedule hérité `[(0,A),(sw1,B),(sw2,A)]` et compare aux latents
d'`inference()` : **max|diff| = 0.000e+00 sur memzero, oracle, oracle1, oracle3b, memdec20**.

### 2.2 Baseline `framebank`

- **preserve** : à chaque switch, la dernière frame latente du segment sortant entre dans la
  banque (une entrée par SEGMENT → la banque croît : 1, 2, 3, 4 en S4), clé = **embedding de
  prompt poolé UMT5** L2-normalisé (même pooling que le `cond` du décodeur, aucun tirage RNG).
- **recall** : à **chaque** transition (pas seulement aux revisits — Echo-Forcing n'a pas d'oracle
  qui lui dit lesquelles sont des retours, donc les **faux rappels sont mesurés, pas cachés**),
  argmax du cosinus sur les entrées **non adjacentes**, rappel ssi `s_max ≥ τ`. Départage
  `FRAMEBANK_TIE` : `recent` (défaut, l'observation la plus fraîche) ou `first`.
- **injection** : la frame retrouvée occupe le slot de fenêtre le plus récent, le reste de la
  fenêtre reste la vraie queue récente — **mécanique identique à `oracle1`**.
- **`framebank_oracle`** : même banque, sélection par identité de sujet vérité-terrain = plafond
  de la banque.
- Tous les choix (candidats, similarités, décision, correct/faux) sont loggés dans
  `results/fb_logs/*.json` et agrégés par `--mode summary` (taux de rappel, taux de sujet
  correct, faux rappels sur les switchs non-revisit).

### 2.3 Scénarios (240-300 frames latentes = 60-75 s)

| nom | schedule (frames latentes) | revisits scorés | pourquoi |
|---|---|---|---|
| `S1` | A(0-81) B(81-162) A(162-240) | seg2 | pont vers la campagne l1 (triples 0-7 = les 8 paires **verbatim** → mêmes prompts, même `cell_seed`) |
| `S2` | **C(0-60) A(60-120)** B(120-180) A(180-240) | seg3 vs **seg1** | sujet **NON OUVRANT** : le sink de LongLive tient la frame 0 = C, pas le sujet |
| `S3` | A(0-75) B(75-150) C(150-225) A(225-300) | seg3 vs seg0 | 2 interludes, 150 frames d'interférence |
| `S4` | A(0-60) B(60-120) **A(120-180)** C(180-240) **A(240-300)** | seg2 **et** seg4, vs seg0 | double revisite : la 2ᵉ résurrection part-elle de l'original ou de la 1ʳᵉ ? |
| `S2p/S3p/S4p` | idem, mais le prompt du revisit est une **PARAPHRASE** de A | idem | seule variante où le routage peut se tromper (cf. §3) |

12 TRIPLES `(A, B, C)` : les 8 paires historiques + un 3ᵉ décor chacune, plus 4 nouveaux triples
couvrant véhicule / personne / animal / objet rigide. **Selftest (f) : 0 des 48 prompts (36 +
12 paraphrases) n'apparaît dans le corpus d'entraînement vidprom ni dans les 165 vidéos v3 sur
disque**, dans les deux sens de sous-chaîne.

Scoring : re-ID de **chaque** segment de revisit contre le segment de la **première apparition**
(pas l'ouverture), fenêtres `settle=20` (segment 0) / `resettle=12` (segments post-switch), 3
frames latentes coupées en fin de segment — conventions verbatim de la campagne l1.
`reid_resA_origA` garde son nom (moyenne sur les revisits) pour rester comparable ;
`reid_by_revisit` garde le détail par retour.

---

## 3. LE FAIT QUI CHANGE LE DESIGN : LE ROUTAGE PAR PROMPT EST TRIVIAL EN VERBATIM

`prompt_key_matrix.py` mesure le cosinus entre les 48 clés de prompt.

| régime | match correct | pire distracteur **dans le même rollout** | marge |
|---|---|---|---|
| revisit **verbatim** (S1-S4) | **1.0000** (même chaîne) | **0.7218** (`herd of sheep` vs `beekeeper`, t7) | +0.28 |
| revisit **paraphrasé** (S2p/S3p/S4p) | 0.665 – 0.780 | 0.469 – 0.7218 | +0.067 à +0.277 **par triple** |

Trois conséquences, à décider par Aden :

1. **En S1/S2/S3, `framebank` est BITWISE `oracle1`.** Le routage ne peut pas se tromper
   (1.0000 vs 0.7218, τ = 0.90), et le contenu banké EST la frame que l'oracle splice. Le
   selftest (b) le prouve bit à bit sur A→B→A. Faire tourner les deux bras brûlerait un
   cinquième de la campagne sur un doublon → `DEFAULT_ARMS` n'en garde qu'un.
   *C'est déjà un résultat* : une banque de frames réelles façon Echo-Forcing atteint la borne
   `oracle1` (+0.224 sur la table l1) dès qu'il y a une seule apparition antérieure du sujet.
2. **En S4, les trois bras diffèrent vraiment** : `framebank`/`framebank_oracle` rappellent la
   **plus fraîche** occurrence de A (segment 2 = la 1ʳᵉ résurrection, donc déjà dégradée),
   `oracle1` garde la vraie première apparition. C'est le test « la résurrection en chaîne
   dérive-t-elle ? », et le seul endroit où `framebank_oracle` gagne son coût en verbatim.
3. **Globalement, aucun seuil unique ne sépare « paraphrase du même sujet » (min 0.665) de
   « autre scène du même triple » (max 0.7218)** : le routage par cosinus de prompt n'est **pas
   robuste à la reformulation**. C'est une faiblesse réelle et publiable de la baseline, et c'est
   la seule configuration où `framebank_oracle` mesure un vrai coût de sélection. Le seuil des
   variantes `p` est fixé à **`fb_tau = 0.68`**, ce qui donne les deux types d'erreur, prédits
   avant la campagne :
   - **ratés** (la paraphrase n'atteint pas τ, donc aucun rappel = le bras retombe sur `memzero`)
     sur les triples **0** (0.6792) et **10** (0.6652), soit **2/12** ;
   - **faux rappels** possibles aux switchs non-revisit sur les triples **0** (B↔C 0.7054),
     **3** (B↔C 0.7075) et **7** (B↔C 0.7218, B←A 0.7022).
   `framebank_oracle` − `framebank` mesurera exactement la somme de ces deux coûts.

---

## 4. SELFTESTS — **52/52 PASS** (`logs/selftest.log`)

| test | mesure | verdict |
|---|---|---|
| (a) schedule hérité == `inference()` | max\|diff\| latents **0.000e+00** sur memzero / oracle / oracle1 / oracle3b / memdec20 ; sanity oracle1≠memzero = 6.19 | **PASS bitwise** |
| (b) `framebank_oracle` et `framebank` en A→B→A == `oracle1` | **0.000e+00** les deux ; routage : seg2 → seg0, sim 1.0000, CORRECT ; aucun faux rappel au switch B | **PASS bitwise** |
| (b2) N-switch A→B→A→C→A | rappel aux **deux** retours, sujet correct les deux fois (seg2→seg0, seg4→seg2) ; banque 1→2→3→4 ; aucun faux rappel sur B et C (s_max 0.7054 < τ) | **PASS** |
| (b3) revisit **paraphrasé** (τ = 0.68) | la comptabilité de SUJET marque toujours seg2 et seg4 ; le routeur apparie A' à la scène A verbatim du seg0 avec un cosinus **non trivial** (0.7803), puis seg4 → seg2 (1.0000, ce segment était lui aussi conditionné sur A') | **PASS** |
| (c) pas de fuite d'env | `memzero` rejoué APRÈS tous les bras : **0.000e+00** ; overlay `MEM_WRITER` restauré (`_memwriter_applied = None`) | **PASS bitwise** |
| (d) scoring 1ère apparition | vidéo synthétique honnête (revisit = 1ʳᵉ apparition) → reid **1.0000**, contrôle 0.4603 ; vidéo **piège** (revisit repeint identique au **segment 0**, 1ʳᵉ apparition différente) → reid **0.4784** alors qu'un bug « compare au segment 0 » donnerait 1.0000 ; idem crop60 | **PASS** |
| (d0) comptabilité des schedules | rôles / première apparition / revisits / fenêtres de scoring strictement internes aux segments, sur les 7 scénarios ; les variantes `p` conservent la structure de sujets et conditionnent bien chaque revisit sur la paraphrase | **PASS** |
| (e) `py_compile` | 5 fichiers (pipeline + 3 nouveaux + `run_reid_probe.py`) | **PASS** |
| (f) grep held-out | 48 prompts vs `vidprom_filtered_extended_switch.txt` **et** vs les 165 vidéos v3 sur disque, sous-chaîne dans les deux sens : **0 hit** ; `TRIPLES[0:8][:2] == PAIRS` verbatim | **PASS** |

Note : les textures aléatoires ne marchent pas pour (d) — DINO les mappe toutes sur « bruit
coloré » (cos 0.996). Le test utilise 4 motifs géométriques dont la matrice de cosinus DINO a été
mesurée (diagonale 1.0, hors-diagonale 0.41-0.71).

---

## 5. SMOKE (1 cellule S2 + 1 cellule S3, bras memzero / framebank / memdec20, GPU 1)

6 rollouts, triple 0 (`brown dog with a white chest` / `red vintage car` / `wooden sailboat`),
seed 42, `logs/smoke.log`. **Tout s'est déroulé comme prévu.**

**Contrôle visuel** (`reference/runs/reid_scen/smoke/*.png`, lignes = segments, 4 frames réparties
par segment) : les 4 segments de S2 et S3 apparaissent aux bonnes frontières, et le retour du
sujet est visible. La planche la plus parlante est la paire **S3 memzero vs S3 framebank** :
segments 0-2 identiques (les bras ne divergent qu'au switch de retour), puis au revisit
**memzero laisse le chien DANS LE LAC** (la scène C persiste : échec de rejoin classique) tandis
que **framebank le ramène dans la prairie verte** de sa première apparition. En S2, le sujet
n'ouvre pas : segment 0 = voilier, le chien arrive au segment 1, et le revisit du segment 3 est
bien scoré contre le segment 1.

**Routage `framebank`** (log complet dans `results/fb_logs/`) :

- S2 : seg1 (A entre, scène neuve) → aucun candidat non adjacent → **NO RECALL** ; seg2 (B entre) →
  candidat seg0 (C) à **0.7054** < τ 0.90 → **NO RECALL** ; seg3 (A revient) → seg0 (C) 0.6132,
  **seg1 (A) 1.0000** → rappelle **seg1**, CORRECT.
- Agrégat des 2 cellules : rappel 100 % aux revisits, **sujet correct 100 %**, **0 faux rappel**
  sur les 4 switchs non-revisit. Conforme à la calibration §3.

**Chiffres n=1, à ne PAS lire comme un résultat** (une cellule, aucune statistique) :

| | memzero | memdec20 | framebank |
|---|---|---|---|
| S2 `reid` full / crop60 | 0.351 / 0.298 | 0.542 / 0.556 | 0.559 / 0.572 |
| S3 `reid` full / crop60 | 0.512 / 0.517 | 0.568 / 0.612 | 0.751 / 0.775 |

Le seul point à retenir : le **niveau absolu de `memzero` s'effondre** sur ces terrains
(0.351 en S2, 0.512 en S3) contre **0.686** sur le scénario S1 de la campagne l1. La matrice
attaque bien un régime plus dur que le sujet-ouvrant — ce qui était l'objectif.

---

## 6. COÛT DE LA CAMPAGNE ET SHARDS RECOMMANDÉS

Mesuré au smoke sur RTX 6000 Ada : **3.9 min par rollout de 240 frames** (3:56 et 3:53),
**4.9 min par rollout de 300 frames** (4:50 et 4:54), **~3.6 min de build de pipeline** (2 builds
par processus de shard : MAG + stockoff), **7.6 / 9.4 MB par mp4** (240 / 300 frames).
Scoring DINO : **~8 s par bras et par cellule**, soit ~1 % du coût — négligeable.

Bras de la matrice (`DEFAULT_ARMS`) : `stockoff`, `memzero`, `memdec20`, `framebank`, `oracle1`
(+ `framebank_oracle` en S4 et dans les variantes `p`).

### Coût par cellule (1 triple × 1 seed)

| scénario | frames | bras | min/cellule |
|---|---|---|---|
| S2 | 240 | 5 | **19.5** |
| S3 | 300 | 5 | **24.5** |
| S4 | 300 | 6 | **29.4** |
| S2p | 240 | 6 | 23.4 |
| S3p / S4p | 300 | 6 | 29.4 |

### Nombre de cellules recommandé

La campagne l1 a détecté `memdec20` +0.126 avec **24 cellules** (8 prompts × 3 seeds, 21+/3−,
sign-p 0.023), et son détail par paire montre que **la variance de prompt domine celle de seed**.
Je recommande donc **12 triples × 2 seeds = 24 cellules par scénario** : même puissance
statistique, 50 % de diversité de prompts en plus. Passer à 3 seeds (36 cellules) n'est utile que
pour S4, où deux revisits sont scorés séparément.

| plan | contenu | GPU-h | mur sur 4 GPU |
|---|---|---|---|
| **A — lecture minimale** | S2 seul, 24 cellules | **7.8** | **~2.0 h** |
| **B — matrice standard (recommandé)** | S2 + S3 + S4, 24 cellules | **29.5** | **~7.4 h** |
| B+ | idem avec 36 cellules en S4 | 35.4 | ~8.9 h |
| C — option routage | S2p + S3p + S4p, 24 cellules | 32.9 | ~8.3 h |
| pont l1 (optionnel) | S1, 24 cellules | 7.8 | ~2.0 h |

Stockage : plan B = 24 × 16 rollouts ≈ 384 mp4 ≈ **3.4 GB** sur le disque projet (220 GB libres).
Le home est à **100 %** (14 GB libres) : les mp4 y sont interdits, `--vid-dir` pointe par défaut
sur le disque projet. **Ne pas passer `--dump-frames` sur la campagne** : les planches PNG font
7 MB pièce (2.5 GB sur 360 cellules-bras) et tomberaient sur le home ; c'est un outil de smoke,
à pointer sur le disque projet si tu en veux quand même.

### Shards

`bash reference/runs/reid_scen/launch_scen.sh S2,S3,S4 "0,1,3,4" "0,1"` : round-robin des 12
triples sur 4 GPU (3 triples par shard), pré-check d'occupation **dans** la commande de lancement,
refus dur sur l'index 6 et sur plus de 4 GPU. Chaque shard traite ses triples pour les 3
scénarios, donc 2 builds seulement. Reprise : les rollouts dont le mp4 existe sont sautés.

---

## 7. CE QUI RESTE / DÉCISIONS POUR ADEN

1. **Lancer** le plan A ou B (je n'ai rien lancé au-delà du smoke).
2. **Trancher sur les variantes `p`** : elles sont le seul endroit où la baseline peut se tromper
   de scène. Sans elles, `framebank` = `oracle1` en S1-S3 et le papier dira « une banque de frames
   réelles atteint la borne oracle », ce qui est vrai mais rend la comparaison peu discriminante.
3. **`FRAMEBANK_TIE`** : le défaut est `recent` (l'observation la plus fraîche, choix naturel
   d'une banque qui croît). `first` transformerait `framebank` en `oracle1` même en S4. Un
   A/B sur ce seul flag en S4 coûterait 11.8 GPU-h de plus.
4. **Non implémenté volontairement** : `MEM_DECODE_THRU` (arm `oracle1thru`) refuse de tourner
   sous le driver N-switch — son arithmétique d'éviction suppose l'ancre unique héritée ;
   l'assert le dit explicitement plutôt que de produire un chiffre faux.
5. **Non implémenté** : clé de banque par DINO de la frame (la clé prompt suffit et est
   déterministe), fusion spatiale multi-blocs d'Echo-Forcing, RoPE relatif.
6. **wandb** : rien à logger tant que la campagne n'est pas lancée ; le driver écrit des JSON
   (`results/*_scores.json`, `results/summary_scen.json`, `results/fb_logs/*.json`). Si tu veux du
   wandb sur la campagne, c'est ~10 lignes dans `summarize()` — dis-le avant le lancement.

---

## 8. AUDIT DU DÉCLENCHEMENT DU SPLICE `memdec20` (2026-08-02)

Mission reçue : « le hook `MEM_DECODE` se déclenche à CHAQUE recache post-switch, donc en S2/S2p
le splice au switch X→A contamine le segment de PREMIÈRE apparition de A (référence du scoring)
et fait s'effondrer `memdec20` (0.19 absolu, −0.38 vs `memzero`) ; gate-le par le routeur
framebank, supprime les mp4, relance le bras. »

**Ce diagnostic est faux, et la relance a été refusée.** Le déclenchement était déjà correct ;
le vrai coupable est la RÉFÉRENCE de scoring en S2. Détail et preuves ci-dessous.

### 8.1 Le hook ne se déclenche QUE sur les retours (3 preuves indépendantes)

`_switch_recache` ligne 1004 (avant patch) : `_dec_ckpt = os.environ.get("MEM_DECODE", "") if
is_revisit else ""`, où `is_revisit` vient de la comptabilité SUJET de `inference_schedule`
(`first_app[s] < i`). En S2 (`subjects [2,0,1,0]`) cela vaut `[F,F,F,T]` : le switch X→A
(segment 1) n'est PAS un revisit et ne déclenche rien.

1. **Logs de campagne, 142/144 rollouts `memdec20` relus** (2 rollouts perdus dans un log
   tronqué) : le nombre de lignes `[MEM_DECODE] decoded anchor spliced` par rollout est
   **exactement 1 en S2/S2p/S3/S3p et exactement 2 en S4/S4p**, sans une seule exception
   (min = max sur chaque scénario). C'est très précisément le comptage attendu.
2. **Position dans le log** : en S2 la ligne de splice précède immédiatement
   `num_recache_frames: 12, recache_start_frame: 168, current_start_frame: 180`, c'est-à-dire le
   recache du segment 3 (le retour), et jamais celui de la frame 60 (le switch X→A).
3. **Les scores le disent aussi** : `ctrl_B_origA` et `self_origA`, qui ne portent que sur des
   segments ANTÉRIEURS au retour, sont **identiques au chiffre près entre `memzero`, `memdec20`
   et `framebank`** (0.624 / 0.889 en S2). Un splice à la première apparition les aurait
   nécessairement bougés. Mesure directe complémentaire : `cos(segment de 1re apparition de A,
   segment d'ouverture)` vaut 0.670 / 0.669 / 0.669 pour les trois bras — la partie
   pré-retour du rollout est bit-identique entre bras.

### 8.2 Le vrai problème : en S2 la référence de re-ID n'est pas une image de A

Avec `local_attn_size: 12` et `sink_size: 1` + `global_sink: true`, la fenêtre lisible ne fait
que 12 frames latentes et le sink épingle la frame 0 pour les 240 frames. **En S2 la frame 0
appartient à C (l'intercalaire d'ouverture), pas à A.** Résultat : la scène d'ouverture ne quitte
jamais la vidéo, y compris pendant le segment censé être la « première apparition de A ».

Vérification à l'œil sur 3 cellules (planches :
`/shared/ssd_14T_project/adenelmi/reid_scen/audit_2026-08-02/S2_t{2,4,11}_s0_eye.png`, lignes =
`memzero` / `memdec20` / `framebank`, colonnes = 6 instants répartis sur la vidéo ; scripts et
JSON de mesure dans `reference/runs/reid_scen/audit_2026-08-02/`, log de selftest dans
`reference/runs/reid_scen/logs/selftest_gating_2026-08-02.log`) :

| cellule | A | ouverture C | segment 1 = « 1re apparition de A » (memzero) | retour, memdec20 |
|---|---|---|---|---|
| S2_t2_s0 | chat tigré orange | camion de pompiers | camion de pompiers **+** un chat cartoon collé à droite | chat tigré photoréaliste **seul**, camion disparu |
| S2_t11_s0 | théière en cuivre | fanfare dans la rue | fanfare **+** une théière tenue par un musicien | théière en cuivre sur plan de marbre **seule** |
| S2_t4_s0 | cheval blanc crinière noire | potier au tour | potier au tour, **aucun cheval** | cheval blanc dans l'herbe haute |

Autrement dit : en S2, `memdec20` est **le seul bras qui produit effectivement A au retour**, et
il est pénalisé pour ça, parce qu'il est comparé à un segment de référence qui montre l'ouverture.

Chiffres (24 cellules, DINO full, fenêtres de scoring standard) :

| S2 (n=24) | cos(1re apparition A, ouverture) | cos(retour, ouverture) | cos(retour, 1re apparition) = **le score publié** |
|---|---|---|---|
| memzero | 0.670 | 0.358 | 0.566 |
| memdec20 | 0.669 | **0.017** | **0.188** |
| framebank | 0.669 | **0.517** | **0.775** |

Lecture :
* la référence est à **0.670** de l'ouverture, soit **plus proche d'un autre sujet que le retour
  ne l'est d'elle** (0.566 pour `memzero`) : le nombre publié en S2 ne mesure pas une identité de
  sujet, il mesure la persistance de la scène d'ouverture ;
* `framebank` marque 0.775 en **réimportant** l'ouverture (0.517 : la frame réelle qu'il splice
  contient le camion / la fanfare) ;
* `memdec20` marque 0.188 en **s'en échappant complètement** (0.017).

En S3/S4 la première apparition de A EST l'ouverture (`cos(1re apparition, ouverture)` = 0.948
par construction), la référence est donc propre et les chiffres de ces scénarios restent valides.

**La mesure qui tranche.** On note A* le rendu PROPRE de A : le segment 0 du rollout **S3** de la
MÊME cellule (même triple, même seed — `cell_seed` ne dépend pas du scénario), où A ouvre la vidéo
et où rien ne le précède. On compare chaque segment S2 à A* (24 cellules, DINO full) :

| segment S2 comparé à A* (rendu propre de A, même seed) | cos |
|---|---|
| **« 1re apparition de A » = la RÉFÉRENCE du scoring** | **0.165** |
| retour, `memzero` | 0.223 |
| retour, `framebank` | 0.250 |
| retour, `oracle1` | 0.250 |
| **retour, `memdec20`** | **0.658** |

La référence contre laquelle toute la campagne S2 est scorée ressemble au sujet A à **0.165**.
`framebank` et `oracle1`, qui splicent une frame RÉELLE de ce segment, héritent mécaniquement de
son contenu (0.250) et sont récompensés pour ça. **`memdec20` est le seul bras qui produit
réellement A au retour (0.658, ~2.6x tous les autres), et c'est exactement pour cela qu'il est
noté −0.378.** Le verdict S2 publié est inversé.

### 8.3 Conséquences sur les verdicts de la campagne

| scénario | référence de scoring | verdict `memdec20` (full) |
|---|---|---|
| S2, S2p | **INVALIDE** (sujet absent ou minoritaire dans la référence) | −0.378 / −0.382 : **artefact de métrique, à ne pas publier tel quel** |
| S3, S3p | valide (A ouvre) | **+0.125 / +0.122** |
| S4 | valide (A ouvre) | **+0.120** (par retour : seg2 +0.077, seg4 +0.163) |
| S4p | valide (A ouvre) | **−0.013** (9+/15−) : seule régression réelle, sur revisit PARAPHRASÉ |

Le terrain « sujet non-ouvrant » — celui que la matrice a été construite pour attaquer — n'est
donc **pas encore mesuré** : il l'est par une métrique qui y est aveugle. C'est le vrai chantier
ouvert par cette campagne.

### 8.4 Ce qui a été ajouté quand même : `MEM_DECODE_ROUTED` / bras `memdec20r`

Le bug annoncé n'existait pas, mais l'asymétrie méthodologique qu'il visait, elle, est réelle :
`memdec20` reçoit du planning un signal ORACLE « ce switch est un retour », que `framebank` n'a
pas (son routeur doit le déduire du cosinus de prompt). Additif, par défaut inactif :

* `pipeline/switch_causal_inference.py` : `_route_on()` (le routeur tourne aussi pour
  `MEM_DECODE_ROUTED`), `_fb_select(..., mode=)` (décision seule, frame jetée), et le gate
  `_fires` dans `_switch_recache` ;
* `scenarios.py` : clé `MEM_DECODE_ROUTED="0"` dans `_OFF_SCEN` (aucune fuite entre bras) et bras
  **`memdec20r`** = `memdec20` + `MEM_DECODE_ROUTED=1`. `memdec20` garde son gate ground-truth,
  donc **les 144 mp4 déjà produits restent valides et comparables** ;
* `inference()` (chemin legacy 2-switch, harness `reid_probe_l1`) n'est pas touché : aucune
  référence à `_fb_*` / `_route_on` dans son corps, et le selftest (g3) le vérifie bitwise avec la
  variable ABSENTE de l'environnement.

Selftests ajoutés à `selftest_scenarios.py` (bloc `(g)`), **67/67 checks verts** :

| check | résultat |
|---|---|
| (g) `memdec20` splice uniquement aux retours, S2-shaped `C→A→B→A` | splices aux segments `[3]`, attendu `[3]` |
| (g) idem S3-shaped `A→B→C→A` | `[3]` |
| (g) idem S4-shaped `A→B→A→C→A` | `[2, 4]` |
| (g) contrôle négatif `A→B→C` (aucun retour) | `[]` |
| (g) retour PARAPHRASÉ, S2p-shaped | `[3]` |
| (g2) `memdec20r` route EXACTEMENT comme `framebank` (4 formes) | décisions identiques |
| (g2) `memdec20r` splice là où son routeur a rappelé (4 formes) | oui |
| (g3) `inference()` legacy BITWISE, `MEM_DECODE_ROUTED` ABSENTE de l'env | `max|diff| = 0.000e+00` |
| (g3) `inference_schedule` BITWISE, variable absente vs `"0"` | `max|diff| = 0.000e+00` |
| (a)/(b)/(c) préexistants (bitwise legacy, framebank==oracle1, non-fuite d'env) | inchangés, verts |

Le comptage `(g)` est la reproduction en petit du comptage fait sur la campagne réelle : les deux
disent la même chose, le hook n'a jamais sur-déclenché.

### 8.5 Ce qui n'a PAS été fait, et pourquoi

* **Aucun mp4 supprimé, aucune relance.** L'inférence est déterministe à seed fixe : relancer
  `memdec20` à l'identique aurait reproduit les mêmes fichiers, et le gater par le routeur ne peut
  pas transformer −0.378 en positif (le routeur rappelle correctement 91.7 % des retours : il
  déclencherait au même endroit, à 4 événements près sur 48). ~11 GPU-h économisées.
* **Pas de commit** (règle standing).

### 8.6 Prochaine carte recommandée

Le blocage n'est plus le bras, c'est la mesure et le substrat :

1. **Réparer la référence S2** : scorer le retour contre un rendu PROPRE de A (par ex. le segment
   0 du rollout S3 de la même cellule, même seed, où A ouvre) au lieu du segment 1 pollué. C'est
   une re-mesure offline sur les vidéos déjà sur disque, ~0 GPU-h.
2. **Test d'instance** : même-seed vs autre-seed contre ce rendu propre, pour séparer « rappel qui
   préserve l'instance » de « re-rendu depuis le prompt ». Sans ça, le « A propre » que produit
   `memdec20` en S2 n'est pas encore attribuable à la mémoire.
3. **S4p** est la seule régression réelle : le décodeur est conditionné FiLM sur le prompt, et un
   revisit paraphrasé lui donne un vecteur de conditionnement qu'il n'a pas vu.

---

## 9. RE-SCORING PROPRE S2/S2p + TEST D'INSTANCE (2026-08-02, suite de la section 8)

Section 8 a montré que la référence de re-ID de S2/S2p (le segment de « première apparition de
A ») ne montre pas A mais l'ouverture épinglée par le sink. On rescore donc les deux scénarios
contre une référence valide, et on demande ensuite si le A ramené est bien CELUI de la cellule.

Outillage : `reference/runs/reid_scen/rescore_clean.py`
(`--mode measure|aggregate|power`), sorties dans `results/clean_rescore/`. Aucune génération :
tout est offline sur les 288 mp4 déjà sur disque.

### 9.1 Protocole

**A\* = le rendu PROPRE de A** = segment 0 du rollout **S3** de la MÊME cellule (même triple,
même seed — `cell_seed` ne dépend pas du scénario), où A OUVRE la vidéo et où rien ne le précède.
Les bras MAG sont scorés contre le A\* MAG ; **`stockoff` contre SON PROPRE A\*** (checkpoint
différent : une référence MAG confondrait pipeline et bras). Fenêtres de scoring, stride et
DINOv2-vitb14 : conventions verbatim de `score_video_scen`.

Repère : la référence contaminée que la campagne utilisait est à **0.165** (full) / **0.180**
(crop60) de A\*. C'est le chiffre qui invalide l'ancien scoring.

### 9.2 S2 / S2p re-scorés propre (24 cellules appariées, delta vs `memzero`, test de signe)

**S2, DINO full** (entre parenthèses : l'ANCIEN scoring contaminé, conservé exprès — les deux
colonnes racontent l'histoire de la contamination)

| bras | score propre | delta vs memzero | signes | sign-p | (ancien score / ancien delta) |
|---|---|---|---|---|---|
| `memzero` | 0.2235 | — | — | — | (0.5660 / —) |
| **`memdec20`** | **0.6577** | **+0.4343** | **24+/0−** | **<0.0001** | (0.1877 / **−0.3782**) |
| `framebank` | 0.2500 | +0.0266 | 13+/11− | 0.84 | (0.7747 / +0.2088) |
| `oracle1` | 0.2505 | +0.0270 | 13+/11− | 0.84 | (0.7744 / +0.2084) |
| `stockoff` | 0.1272 | −0.0962 | 7+/17− | 0.064 | (0.6292 / +0.0633) |

**S2p, DINO full**

| bras | score propre | delta vs memzero | signes | sign-p | (ancien score / ancien delta) |
|---|---|---|---|---|---|
| `memzero` | 0.2326 | — | — | — | (0.5443 / —) |
| **`memdec20`** | **0.5895** | **+0.3569** | **23+/1−** | **<0.0001** | (0.1620 / **−0.3823**) |
| `framebank` | 0.3251 | +0.0925 | 17+/7− | 0.064 | (0.6843 / +0.1400) |
| `framebank_oracle` | 0.3234 | +0.0908 | 16+/8− | 0.15 | (0.7000 / +0.1557) |
| `oracle1` | 0.3238 | +0.0912 | 16+/8− | 0.15 | (0.6995 / +0.1552) |
| `stockoff` | 0.1754 | −0.0572 | 8+/16− | 0.15 | (0.6350 / +0.0907) |

`crop60` dit la même chose partout (S2 `memdec20` +0.4232, 24+/0− ; S2p +0.3129, 24+/0− ;
`framebank` +0.034 / +0.091 non significatif ; `stockoff` −0.106 / −0.078, p=0.023).

**Inversion complète, dans les deux sens.** Sous une référence valide, `memdec20` passe de
−0.378 à **+0.434 avec 24 signes positifs sur 24** — le résultat le plus net que le projet ait
produit sur ce terrain. Symétriquement, `framebank` et `oracle1` passent de +0.21 à **un nul**
(+0.027, p=0.84) : leurs +0.21 étaient entièrement le fait de réimporter l'ouverture contenue
dans la frame réelle qu'ils splicent. `stockoff` passe de +0.06 à **négatif**. Sur le terrain
sujet-non-ouvrant, la banque de frames réelles (Echo-Forcing) **ne fait rien**, et l'ancre décodée
depuis la mémoire est la seule chose qui ramène le sujet.

### 9.3 Test d'instance : NÉGATIF, et informatif

Question : le A ramené est-il **LE** A de cette cellule, ou juste **un** A du bon prompt ?
Mesure appariée : `cos(retour, A* même seed)` vs `cos(retour, A* de l'AUTRE seed du même triple)`
— même prompt, instance différente.

| scénario | bras | même seed | autre seed | delta | signes | sign-p |
|---|---|---|---|---|---|---|
| S2 | **`memdec20`** | 0.6577 | 0.6654 | **−0.0076 ± 0.0235** | 11+/13− | 0.84 |
| S2 | `framebank` (contrôle) | 0.2500 | 0.2430 | +0.0070 ± 0.0088 | 14+/10− | 0.54 |
| S2 | `memzero` | 0.2235 | 0.2260 | −0.0025 ± 0.0094 | 11+/13− | 0.84 |
| S2p | **`memdec20`** | 0.5895 | 0.5724 | **+0.0171 ± 0.0162** | 13+/11− | 0.84 |
| S2p | `framebank` (contrôle) | 0.3251 | 0.3150 | +0.0102 ± 0.0108 | 14+/10− | 0.54 |

**Puissance de la mesure** (`--mode power`, 12 triples) : deux seeds du même prompt SONT
distinguables par cette métrique — A\* même instance (split-half) **0.9329**, A\* instance
différente **0.7994**, soit une **marge de +0.1335, positive sur 12/12 triples**. Un vrai rappel
d'instance devrait donc produire un delta de cet ordre.

**Verdict sec : le retour de A de `memdec20` est un RE-RENDU depuis le prompt, pas un rappel
d'instance.** Le delta observé (−0.008 ± 0.024) exclut, à 24 cellules, un effet de la taille que
la métrique sait détecter (+0.13). Pire : la valeur absolue **0.658 est INFÉRIEURE au niveau
inter-seeds 0.799** — l'ancre décodée ramène un A plus éloigné de celui de la cellule qu'un A
tiré d'un autre seed ne l'est. La mémoire porte donc la **catégorie et le prompt**, pas
l'individu, ce qui rejoint exactement le diagnostic « régression de catégorie » du 2026-07-27.

Note sur le contrôle : `framebank` est lui aussi nul (+0.007, p=0.54), mais c'est attendu et non
informatif ici — son score propre en S2 est 0.250, c'est-à-dire qu'il **ne produit pas A du
tout** ; on ne peut pas tester l'identité d'un individu sur un segment qui montre un camion de
pompiers. Le contrôle de sensibilité utile est le `--mode power` ci-dessus, qui lui est concluant.

### 9.4 Eyepack S2 pour l'œil (`reference/runs/reid_scen/eyepack_s2/`)

* `S2_board_reference_vs_cleanA_vs_arms.png` — 3 lignes (cellules t2 = chat tigré, t4 = cheval
  blanc, t11 = théière cuivre), 5 colonnes : **référence du scoring** (S2 seg1) | **CLEAN A\***
  (S3 seg0, même seed) | retour `memzero` | retour `framebank` | retour `memdec20`. On y lit d'un
  coup d'œil que les colonnes 1, 3 et 4 montrent l'ouverture (camion de pompiers / tour de potier
  / fanfare) et que seule la colonne 5 rejoint la colonne 2.
* `S2_t{2,4,11}_s0_memzero-framebank-memdec20.mp4` — hstack 3 bras, résolution native
  (2496×480), crf 23, libellés incrustés, **11.8 / 11.8 / 20.3 MB**.

### 9.5 Ce que ça change pour la suite

1. **Le résultat S2/S2p est à republier inversé** : `memdec20` +0.43 / +0.36 (24+/0−, 23+/1−),
   `framebank` nul. C'est le premier terrain où l'ancre décodée bat nettement la baseline publiée.
2. **La contribution est de niveau CATÉGORIE/SCÈNE, pas INSTANCE** — à dire explicitement, c'est
   la limite honnête du dispositif actuel et ça recentre le terme d'instance (InfoNCE DINO) comme
   le vrai chantier de la mémoire 2.1.
3. **Le scoring de la matrice doit passer à la référence A\*** partout où le sujet n'ouvre pas
   (S2/S2p aujourd'hui, tout scénario non-ouvrant demain) ; l'ancienne colonne est à garder comme
   pièce à conviction de la contamination par le sink.
