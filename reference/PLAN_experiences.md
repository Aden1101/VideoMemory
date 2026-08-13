# Plan d'expériences — mémoire TTT sur Anchor-Forcing

Roadmap ordonnée du moins cher / plus sûr au plus cher. Chaque étape **conditionne** la suivante :
on ne paie le gros matériel qu'une fois la méthode prouvée sur le petit.
Voir le contexte technique dans [`RECAP_memoire.md`](RECAP_memoire.md).

---

## 🎯 Finalité

Un Anchor-Forcing + mémoire TTT qui **améliore mesurablement la cohérence / l'identité sur long
horizon** vs l'AF de base, avec une **table d'ablation propre** (la mémoire aide-t-elle, et quels
ingrédients comptent) pour le superviseur / papier.

**Modèle de référence visé** = write appris × multi-couches × horizon long × teacher 14B.

---

## Vue d'ensemble

| # | Expérience | Question | Coût GPU | Tient sur 1 GPU ? |
|---|---|---|---|---|
| **E0** | Baseline AF, mesure de la dérive | où/quand l'AF dérive ? | 1 GPU (inférence) | ✅ |
| **E1** | Phase 1, 1 couche (lecture seule) | go/no-go + plomberie | 1× 48 Go | ✅ |
| **E2** | Phase 1, ~5 couches (lecture seule) | la capacité (couches) aide-t-elle ? | 2× 48 Go ou fenêtre rognée | ~ |
| **E3** | Phase 2, 1 couche, BPTT tronqué (~2) | **le write appris aide-t-il ?** (cœur) | 1× 80 Go ou 2× 48 + coupes | ❌ |
| **E4** | Phase 2, ~5 couches, tronqué, teacher 1.3B | la méthode complète (sans gros teacher) | multi-GPU / 80 Go | ❌ |
| **E5** | Référence : Phase 2 + 14B + horizon long | modèle final / livrable | 8× 80 Go (`train_long`) | ❌ |

---

## Détail des étapes

### E0 — Mesurer la baseline AVANT de mémoriser quoi que ce soit
*1 GPU, inférence seule, quasi gratuit.*
Générer des vidéos **longues** avec l'AF vanilla et **quantifier la dérive** (identité / cohérence).
C'est le **mètre-étalon** : sans lui, impossible de dire si la mémoire aide. Donne aussi **l'horizon**
où l'AF commence à dériver = où évaluer.
→ **Étape la plus négligée et la plus importante.** Figer ce protocole d'éval (mêmes prompts, même
horizon, même teacher) et le garder **identique** pour toutes les expés.

### E1 — Phase 1, 1 couche (lecture seule)
*1 GPU, `configs/ttt_phase1.yaml` déjà prêt, sûr car `g_init=0` → ne peut pas régresser.*
Le gate s'ouvre-t-il, et AF+mémoire bouge-t-il **quoi que ce soit** sur la métrique E0 ?
**Go/no-go + test de plomberie** (le pipeline tourne sur vraies données, save/éval OK).
N'attends pas le gros gain ici — c'est le **signal de vie**.

### E2 — Phase 1, ~5 couches (lecture seule)
*2× 48 Go, ou fenêtre rognée (streaming ~40, `num_frame_per_block` 2) sur 1 GPU.*
Pré-requis : rendre `MEMORY_LAYERS` pilotable depuis le YAML (codé en dur `causal_model.py:717`).
Plus de **capacité** (couches) aide-t-il, *write encore non entraîné* ?
→ isole le levier « couches » du levier « write appris ».

### E3 — Phase 2, 1 couche, BPTT tronqué (prof. ~2)
*1× 80 Go (ou 2× 48 + coupes). Vrai dev : write différentiable + gestion des caches KV/flow in-place.*
**Apprendre le write** (lrs, k/v) aide-t-il vs lecture seule (E1) ?
→ **test central de l'hypothèse** : le gain est-il bien dans le write appris ? Première vraie expé
« méthode ».

### E4 — Phase 2, ~5 couches, BPTT tronqué, teacher 1.3B
*Multi-GPU / 80 Go.*
Combine les deux leviers gagnants (write appris + multi-couches). La méthode complète (sans le gros
teacher) donne-t-elle un gain **clair et robuste** ?

### E5 — Run de référence : Phase 2, multi-couches, teacher 14B, horizon long (streaming 240)
*8× 80 Go, classe `train_long`.*
Le modèle final de qualité. **À ne payer qu'une fois E3-E4 ont prouvé que la méthode marche.** Livrable.

---

## Logique de décision (go / kill)

- Après **E0** : si l'AF **ne dérive pas** à l'horizon visé → la mémoire n'a rien à réparer, **revois la
  prémisse**.
- Après **E1-E2** : un résultat nul est **possible** (lecture seule = intervention faible) — **pas un
  kill**, ça pousse juste à prioriser E3.
- Après **E3** : si le write appris (1 couche) **ne bat toujours pas** la baseline → signal fort que
  l'approche mémoire n'aide pas *cette* tâche. **Reconsidère avant de dépenser** sur E4-E5.

---

## Ablations (se construisent toutes seules)

Les comparaisons **sont** les ablations : E1↔E3 (lecture vs write appris), E1↔E2 (1 vs 5 couches),
E4↔E5 (teacher 1.3B vs 14B), + balayage de la profondeur de troncature BPTT en E3.

---

## Concrètement, maintenant

- Tient sur **le GPU qu'on peut avoir** : **E0 + E1** (et E2 rognée). C'est le périmètre à attaquer dès
  qu'une carte se libère (le GPU 6 est KO, le reste est pris — cf. RECAP §5).
- **E3+** attendent un accès **80 Go / multi-GPU**.

**En une phrase :** mesurer la dérive (E0) → prouver la plomberie (E1) → tester la capacité (E2) →
tester le cœur de la méthode, le write appris (E3) → combiner (E4) → scaler au modèle de référence (E5).
