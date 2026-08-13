# Mémoire à poids rapides (fast-weight / test-time training) pour la génération de vidéo longue

## Étude d'un module MAG greffé dans Anchor-Forcing (Wan2.1-T2V-1.3B, DMD self-rollout)

*Document technique, état au 15 juillet 2026. Généré comme base de rapport de stage: il documente le code, le pipeline d'entraînement, l'ensemble des expériences et les résultats (négatifs inclus), de façon honnête et vérifiée contre le code et les métriques réelles.*

---

## Résumé exécutif

**Objectif.** Corriger la dérive d'identité/sujet dans la génération autoregressive de vidéos longues, en greffant une **mémoire à poids rapides** (fast-weight / test-time training, notée MAG) dans **Anchor-Forcing** (backbone Wan2.1-T2V-1.3B, diffusion vidéo causale streaming distillée en 4 pas par **DMD self-rollout**, entraînement **sans données vidéo longues réelles**).

**Parcours expérimental.** (1) Des **garde-fous** explicites (série e3/e4) pour forcer un comportement non-dégénéré: ils gèlent puis "flickerent" (whack-a-mole structurel, triade de Goodhart). (2) Un **co-entraînement propre pur-DMD** (c1): plus de gel, mais l'injection de mémoire est neutralisée vers un régime **inerte** (do-no-harm sans aide) sur la dérive simple. (3) Le fix **eviction-feed** (c2): la mémoire écrit désormais le chunk **évincé** (hors fenêtre d'attention) au lieu du chunk courant redondant, façon AHN/VideoSSM. Résultat: aux checkpoints tôt/moyens, une **vraie mémoire MLP co-entraînée do-no-harm et dynamique** (ce que c1 ratait), mais une **instabilité (flicker)** apparaît vers l'étape 1000. (4) Une **autopsie** de la télémétrie montre que ce flicker n'est **pas** le "mur" fondamental mais une **pathologie d'optimisation réparable** (une rampe monotone non-amortie des paramètres lents qui sur-pilote l'injection). (5) Un **pack de stabilisation** (c3, inspiré de VideoSSM) est en cours d'évaluation.

**Bilan honnête.** Le module de mémoire n'est **pas** le coupable des dégénérescences: le fix eviction-feed lui donne enfin un rôle (do-no-harm) et le flicker est réparable. Le **mur réel** est l'**objectif data-free**: le juge DMD (un teacher à contexte court qui ne voit pas le cache) ne **supervise pas** l'identité d'instance sous-spécifiée par le texte. Stabiliser **amortit** mais ne **redirige pas** cet objectif: le mieux atteignable en data-free est le **do-no-harm**; un effet d'identité **positif** exige une cible supplémentaire (reconstruction sur **vraie vidéo**, famille VideoSSM/MemGen, ou un **témoin non-factorisé data-free**, l'idée "Anchor Resurrection" A→B→A par splice de KV). Vis-à-vis de la littérature, le mécanisme est une **reproduction** (AHN/VideoSSM); la contribution défendable est l'**analyse rigoureuse**, l'**intégrité métrique** (les métriques VBench sont trompeuses: le gel gonfle la Subject Consistency, le flicker gonfle le Dynamic Degree), et le **diagnostic** "on ne peut pas récompenser une mémoire longue-portée avec un juge court-portée data-free".

---

## Sommaire

1. **Contexte et état de l'art** — le problème de la dérive en vidéo longue, la lignée Self-Forcing → LongLive → Anchor-Forcing, la distillation DMD, les mémoires fast-weight/TTT, et les travaux qui préemptent (VideoSSM, MemGen, AHN, SSD).
2. **Architecture du module de mémoire (MAG)** — la cellule fast-weight SwiGLU, l'écriture Muon (Newton-Schulz + renorm-to-init), l'injection gated-AdaLN, les garde-fous d'injection, et le mécanisme clé eviction-feed (FIFO + `x_write`), code à l'appui.
3. **Pipeline d'entraînement et modifications** — la distillation DMD, le self-rollout streaming, le co-entraînement LoRA + mémoire, le BPTT, les garde-fous de la série e, et les runs c1 / c2 / c3 (tableau des configurations).
4. **Expériences et résultats** — le parcours chronologique complet avec les trajectoires chiffrées (e → c1 → c2 → c3), l'autopsie du flicker, honnête sur les résultats négatifs.
5. **Méthodologie d'évaluation** — le protocole 3-way, l'évaluation sink-starvation, les métriques (VBench, motion lag-2, id_cos DINO tail-mean), et les leçons d'intégrité métrique.
6. **Conclusions et travaux futurs** — ce qui marche, les murs structurels, le positionnement vis-à-vis de la littérature, et les pistes concrètes (Anchor Resurrection, reconstruction sur vraie vidéo, benchmark revisit/re-ID).

---

