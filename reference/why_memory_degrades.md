# Pourquoi la mémoire MAG dégrade subject-consistency (diagnostic multi-agents, 2026-06-28)

6 agents (5 lentilles + consensus), ancrés dans le code, run wf_a8685515-d0b. Verdict ci-dessous.

## Le constat (ce que l'eval prouve)
La mémoire dégrade subject_consistency, max de dégâts EXACTEMENT à l'horizon d'entraînement (~6s, 24 latents) :
gap -0.163 à 6s, qui rétrécit de façon monotone avec la longueur (-0.084 à 60s, -0.070 à 120s). Le score mem
MONTE avec la longueur (0.80 -> 0.88) pendant que la base reste plate (~0.96). Même forme sur background
(-0.045 -> -0.018). Config mem vs base identique sauf memory_layers : la mémoire EST la cause. Cette forme
(pire en court, mieux en long) élimine d'emblée tout problème d'horizon : un carry-over serait bon en court et
empirerait en long. Donc les dégâts sont un PRÉFIXE de taille FIXE dilué par la longueur.

## Cause racine #1 (dominante, confiance ~0.6) : le cold-start n'est jamais supervisé
Les 5 diagnostics convergent. L'entraînement masque les gradients aux frames pré-switch :
- pipeline/streaming_switch_training.py:116 : start_gradient_frame_index = switch_frame_index (=12, switch_choices [12]).
- pipeline/streaming_switch_training.py:204 : enable_grad = local_start_frame >= start_gradient_frame_index ; sinon no_grad.
Donc la passe finale qui alimente la loss DMD ne supervise le read+gate que sur un état fast-weight DÉJÀ chaud
(>=4 écritures). Le read est apply-first depuis l'état d'AVANT le chunk (video_memory.py:374), et sur le 1er
chunk cet état = init_state (params de base génériques, video_memory.py:357-360). Ce read FROID est injecté à
pleine échelle dans le résidu (causal_model.py:593 x = x + mem_out) via une gate ouverte (gatew ~1.26), mais ce
régime froid n'a JAMAIS été sur le chemin de la loss. À l'inférence l'état démarre froid à la frame 0 et
reproduit exactement ce régime non supervisé.

## Cause #2 (amplificateur, ~0.2-0.3) : pas de terme d'identité + gate ouverte
Loss DMD par-frame (distribution matching), aucun terme cross-frame d'identité -> la mémoire peut casser la
cohérence du sujet tout en gardant le réalisme par-frame (d'où generator_loss sain ~0.12 alors que subject
s'effondre). Gate = readout linéaire brut sans sigmoid (video_memory.py:355), bias 0.1 -> 1.26 : fixe le plancher
(~-0.07 à toutes longueurs) et amplifie le read froid.

## Cause #3 (REFUTÉE comme moteur, ~0.06-0.1) : frames_since_switch OOD
Vrai mismatch (causal_model.py:577-578 : à l'inférence switch_global_frame=None -> fss croît 0..959), MAIS l'OOD
croît avec la longueur -> prédirait pire en LONG, l'inverse de la donnée. À corriger, mais PAS la cause du signe inversé.

## Pourquoi c'est pire à l'horizon (le signe inversé)
Le read froid non supervisé est un préfixe de taille FIXE (~1er chunk, frames 0-11). weight_norm_update renorme
chaque ligne à sa norme d'origine (video_memory.py:185-187) -> état porté borné quelle que soit la longueur (pas
de blow-up en long). VBench subject_consistency moyenne la similarité DINO par-frame sur TOUTE la vidéo. À 6s ce
préfixe corrompu = ~50% des frames -> 0.80. À 60/120/240s il vaut ~5/2.5/1.25% -> dilué -> remonte vers 0.88.

## L'expérience la moins chère qui tranche (GRATUIT, pas de nouvelle inférence)
Re-scorer les vidéos mem 6s DÉJÀ générées en découpant la similarité par-frame : latents [0:12] (froid, non
supervisé) vs [12:24] (chaud, supervisé). Prédiction cause 1 : le gap mem-base est concentré dans [0:12] et se
referme dans [12:24]. Prédiction cause 3 : le gap GRANDIRAIT vers les frames tardives. -> sépare net cause 1 de 3.
Confirmation causale (1 inférence 6s + 1 VBench) : gater mem_out à 0 sur le 1er chunk (garder les écritures),
causal_model.py:593. Si subject 6s remonte ~0.80 -> ~0.88+, cause 1 prouvée. (Check gratuit : dbg_read_rms/
dbg_out_rms video_memory.py:393-394 élevés sur chunks 0-1 puis décroissants.)

## Le fix probable (superviser le régime froid à l'entraînement)
1. streaming_switch_training.py:116 : start_gradient_frame_index = 0 quand requires_grad (superviser tout le
   chunk, y compris 0 écriture), et/ou
2. mélanger des rollouts SANS switch (switch_frame_index=None) -> les frames 0-11 à 0-11 écritures reçoivent du
   gradient (l'inférence est sans switch : aligner le régime), et/ou
3. warm-up de gate (rampe 0->1 sur les ~12 premières frames) pour atténuer le read froid tant que l'état n'est pas informatif.
Secondaire : borner gate_extra / passer switch_global_frame à l'inférence ; ajouter un régularisateur cross-frame
d'identité à la loss. Mitigation inférence immédiate testable : la rampe de gate appliquée à causal_model.py:593.

## Ce qu'on arrête (le cross-chunk n'est PAS la solution)
Le BPTT cross-chunk prédit une dégradation qui CROÎT avec la longueur, l'opposé strict de la donnée ; et
weight_norm_update borne déjà l'état porté. Test de mise au rebut (attendu sans effet) : reset d'état tous les 24
latents à l'inférence, re-éval 120s -> scores inchangés. Le levier est la SUPERVISION du cold-start, pas l'horizon.
