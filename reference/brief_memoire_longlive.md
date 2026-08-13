# Brief : faire fonctionner une mémoire TTT/MLP pour la vidéo longue sur LongLive

## 1. Ce qu'on veut faire

Faire fonctionner une **mémoire à poids rapides** (fast-weight / test-time training, TTT) : un petit **MLP** dont les poids sont mis à jour à la volée pendant la génération, greffé dans un modèle de **génération de vidéo longue autoregressive**, avec **LongLive comme baseline**. Objectif concret et modeste : que cette mémoire **améliore la baseline, même un peu**, sur des vidéos **longues** (dizaines de secondes à minutes), typiquement en **tenant mieux l'identité / la cohérence du sujet** sur la durée. La baseline (LongLive) tient déjà bien le long grâce à sa fenêtre d'attention locale + un "sink" (premières frames épinglées) ; on veut qu'une mémoire apprise ajoute quelque chose par-dessus.

## 2. Le substrat technique

- **Backbone** : Wan2.1-T2V-1.3B, diffusion vidéo **causale streaming** distillée en **4 pas par DMD** (Distribution Matching Distillation) en **self-rollout** (génération chunk par chunk auto-conditionnée).
- **Base** : **LongLive** (extension de Self-Forcing pour vidéo longue + interactive), atteint ~240s avec juste : **fenêtre d'attention locale** (12 frames latentes) + **attention sink** (les 3 premières frames latentes gardées en permanence dans le cache KV). **Pas de mémoire apprise** dans LongLive.
- **Notre mémoire** : cellule **MLP SwiGLU** à poids rapides, état de taille fixe, insérée dans 3 blocs du transformer ([5,16,27]). Elle **lit** (interroge son état avec les tokens courants), **injecte** sa sortie dans le flux résiduel via une **porte apprise** (gated-AdaLN), et **écrit** (met à jour ses poids) via un optimiseur **Muon** (Newton-Schulz) multi-pas avec renorm vers l'init. Variante clé : **eviction-feed** = on écrit le **chunk évincé** de la fenêtre (l'info qui sort du cache), pas le chunk courant.
- **Entraînement** : mémoire **co-entraînée** (avec un LoRA sur le backbone) par la loss DMD. **Data-free** : pas de dataset de vraie vidéo longue ; le seul signal vient du **teacher DMD** (modèle bidirectionnel à **contexte court** qui juge si un petit bout de vidéo est réaliste), et ce teacher **ne voit pas** le cache KV ni le passé lointain.

## 3. Ce qu'on a essayé, et ce qui a échoué (chronologique + mécanismes)

**a. Garde-fous explicites** (forcer un comportement non-dégénéré). Whack-a-mole de dégénérescences : le modèle **gèle** (image figée), ou se **dissout**, ou **flicke** (clignote entre 2 images). Chaque garde-fou qui bloque une dégénérescence en révèle une autre. Triade de Goodhart : le gel bat la métrique de consistance, un biais quasi-constant bat la métrique de dérive, le flicker bat la métrique de mouvement.

**b. Co-entraînement propre pur-DMD.** Plus de gel, mais l'injection se fait **neutraliser** vers un régime **inerte** (do-no-harm sans aider) sur la dérive simple.

**c. Diagnostic du "pas de job".** Sur une **dérive monotone**, le **sink tient déjà l'ancre d'identité gratuitement** (les premières frames restent visibles pour toujours). La mémoire n'a **pas de travail marginal** : ce qu'elle retiendrait est déjà dans le sink ou redondant avec la fenêtre locale. Son optimum sous DMD est de ne rien faire.

**d. Eviction-feed** (écrire le chunk évincé). Mieux au début (plus inerte), mais à un certain point d'entraînement part en **flicker** (mouvement incohérent, identité effondrée). Cause : pathologie d'optimisation (rampe non-amortie des paramètres lents qui sur-pilotent l'injection).

**e. Pack de stabilisation** (cap d'amplitude + EMA des poids + decay du lr + clipping). Flicker réglé. Meilleur résultat obtenu : au bon checkpoint (poids EMA lissés, mi-entraînement), la mémoire est **do-no-harm ET pleinement dynamique, stable jusqu'à 2 minutes, sans flicker**. MAIS "do-no-harm" n'est pas "aide" : le gain d'identité (similarité DINO du sujet, apparié avec/sans mémoire, ~6 prompts held-out) est **dans le bruit** (delta ~0).

**f. Re-base sur LongLive** (exactement le même entraînement, poids LongLive, attribution propre car pas de mémoire concurrente). La mémoire est **plus active** (elle bouge le mouvement, pas inerte) mais **net-négative sur l'identité** du début à la fin de l'entraînement, **ne converge jamais** vers do-no-harm. Lecture : sur une base sans mémoire concurrente, la mémoire a de la place pour agir, et **sans signal d'identité, plus elle agit, plus elle coûte de l'identité** ; l'entraînement ne corrige jamais ça.

**g. Probe re-ID sur revisit** (test décisif récent). Scénario **A→B→A** : sujet A distinctif (un chien), on passe à B (une voiture), puis A revient. Question : le A ressuscité ressemble-t-il **plus** à l'original **avec** mémoire qu'**sans** (similarité DINO, crop du sujet) ? C'est le scénario où la mémoire **devrait** enfin servir. **Résultat : elle n'aide pas, elle nuit légèrement** (A ressuscité **moins** fidèle avec mémoire, delta ~ −0.06/−0.07). POURQUOI, visuellement clair : **la mémoire retient le contexte RÉCENT (B) et le traîne dans le retour de A** (la voiture de B reste en fond quand le chien revient). Sans mémoire, le retour à A est plus propre.

## 4. Les faits établis (diagnostic)

- **La mémoire stocke de la RÉCENCE, pas de l'identité.** Elle retient le passé immédiat (Markovien) ; sur un revisit ça devient nuisible (elle traîne le contexte récent, qui est le plus faux).
- **L'objectif data-free ne contient aucun signal d'identité.** Le teacher DMD juge une **distribution** sur une fenêtre courte : un sujet **différent mais plausible** obtient un bon score. Il ne compare jamais à une instance de référence. Donc **rien ne dit à la mémoire quoi retenir** ni ne pénalise la dérive d'identité.
- **Comparaison de référence (VideoSSM, MemGen)** : des systèmes très proches (même backbone Wan-1.3B, même DMD self-rollout, mémoire apprise) qui **marchent**. Leur différence : ils s'entraînent avec une **loss de reconstruction sur de la VRAIE vidéo**. La cible (les vraies frames) **contient le sujet précis** : la dérive est **mesurable et pénalisée**, et le gradient **remonte dans la mémoire** pour lui apprendre à retenir l'identité. En data-free, on est structurellement privés de ce signal.
- **Le "do-no-harm" obtenu était en partie un artefact** de la base forte (le sink tenait déjà l'identité, la mémoire apprenait juste à ne pas déranger). Quand elle a vraiment de la place (LongLive), elle **nuit**.
- **Pièges métriques** : Goodhart (gel gonfle la consistance, flicker gonfle le mouvement). Évaluation en 3-way (mem-ON | mem-OFF | stock), similarité DINO du sujet en queue de rollout, vérification humaine sur vidéos côte-à-côte.

## 5. La question ouverte (à toi de réfléchir)

Comment faire pour qu'une mémoire **TTT / MLP à poids rapides**, greffée sur **LongLive**, en génération de vidéo **longue autoregressive**, **améliore la baseline même un peu** (idéalement : tenir l'identité/le sujet sur la durée, ou aider sur un revisit), **étant donné** : un objectif data-free (DMD, teacher à contexte court) qui ne récompense pas l'identité longue-portée, un sink qui tient déjà l'ancre sur la dérive simple, et une mémoire qui aujourd'hui stocke la récence et non l'identité ?

On ne te donne **pas** de piste exprès : on veut voir ce que tu proposes de toi-même, si tu retombes sur les mêmes leviers que nous ou si tu en trouves d'autres.

Ce qu'on attend de toi, concrètement :
1. **Vérifie contre le code** (chemins section 7) : confirme ou corrige les affirmations de ce brief (l'archi mémoire, l'eviction-feed, la loss, le probe re-ID). Ne prends rien pour argent comptant.
2. **Checke la littérature** (arXiv, 2024-2026) : mémoires TTT / fast-weight / gated-delta / SSM pour la génération vidéo autoregressive longue (au moins VideoSSM, MemGen, LongLive, Self-Forcing, AnchorForcing, LaCT, TTT-Video, Titans, et ce que tu trouves d'autre). Dis clairement **ce qui est déjà fait / préempté** et **où il reste un trou publiable**.
3. **Propose tes propres idées.** Si tu ne retombes pas sur une piste évidente, **donne quand même TES idées** de mémoire (mécanisme + objectif d'entraînement + protocole d'éval) que **tu juges à la fois publiables ET qui auraient une vraie chance de battre la baseline** (même un peu). Pour chaque idée : le mécanisme précis, pourquoi ça donnerait à la mémoire un signal/travail que le data-free actuel ne lui donne pas, ce que ça coûte (données ? compute ?), son degré de nouveauté vs la littérature, et comment tu la mesurerais proprement (3-way, DINO sujet, oeil humain, sans artefact Goodhart).

Sois concret et critique : classe tes pistes par (faisabilité × nouveauté × chance de marcher).

## 6. Contraintes / ce qui compte

- Substrat imposé : mémoire **fast-weight MLP TTT**, sur **LongLive**, vidéo longue.
- **Novelty** : une mémoire par reconstruction sur vraie vidéo est **déjà faite** (VideoSSM/MemGen) ; le refaire tel quel n'apporte rien de neuf même si ça marchait. On cherche un angle **publiable**.
- Barre **modeste** : battre la baseline **même un peu**, proprement mesuré (3-way, DINO sujet, oeil humain), sans artefact Goodhart (gel/flicker).
- Idéalement rester **data-free** ou supervision minimale (c'est notre niche), mais tu peux discuter le compromis si tu juges qu'il le faut.

## 7. Où checker le code (pour vérifier ce brief)

Repo : `/shared/ssd_14T/home/adenelmi/Anchor-Forcing`
- `wan/modules/video_memory.py` : la cellule mémoire (`VideoMemoryLayer`, MLP SwiGLU fast-weight, écriture Muon Newton-Schulz + renorm-to-init, param `x_write` de l'eviction-feed).
- `wan/modules/causal_model.py` : l'injection gated-AdaLN, le FIFO d'éviction, les hooks `MEM_ZERO_ALL` (ablation mem-OFF) et `mem_stream_scale`, les diagnostics.
- `trainer/distillation.py` : la loss DMD, le co-entraînement LoRA + mémoire, l'EMA, les schedulers du pack de stabilisation.
- `pipeline/streaming_training.py` : le self-rollout streaming ; `pipeline/switch_causal_inference.py` : l'inférence à switch (étendue en A→B→A pour le probe re-ID).
- `configs/ttt_memory.yaml` et `reference/runs/l1_longlive/l1_longlive.yaml` : la config du run LongLive actuel (stability pack + eviction-feed).
- `reference/runs/reid_probe/` : le probe re-ID revisit (`run_reid_probe.py`, `results/pair0_scores.json` = les chiffres, `videos/` = les A→B→A).
- Évals 3-way : `reference/runs/l1_longlive/eval*/` (memon.yaml / memzero.yaml / stock_longlive.yaml, run_eval.sh, metrics.py = la métrique DINO id_cos).
