# Carte du domaine KV-cache / streaming AR video + positionnement, 2026-06-29

Workflow wf_435fc461-566, 6 agents (5 clusters + carte).

## La carte du domaine (familles + ce que chacune fait)

Le champ "génération vidéo autorégressive en streaming" (2025-2026) est quasi entièrement bâti sur Wan2.1-T2V-1.3B distillé en peu d'étapes (DMD / Self-Forcing). Il se découpe en cinq familles.

**1. Distillation / schémas "X-Forcing" (combler le train-test gap).** La lignée fondatrice : Diffusion Forcing (2407.01392, bruit par-token indépendant) puis DFoT/History-Guided (2502.06764, rollout au-delà de l'horizon) puis CausVid (2412.07772, DMD 50 vers 4 étapes, teacher bidir vers student causal) puis le pivot Self-Forcing (2506.08009, self-rollout AVEC cache KV pendant l'entraînement + perte vidéo-niveau, tue l'exposure bias). Tout le reste est "Self-Forcing + X" : Causal Forcing (2602.02214, teacher AR pour restaurer l'injectivité frame-level, Dynamic 68 vs 57), Causal Forcing++ (2605.15141), Diagonal Distillation (2603.09488, schéma diagonal + flow-matching, 31 FPS, Total 84.48), AAPT/APT2 (2506.09350, voie GAN sans teacher DMD), Self-Resampling (2512.15702, teacher-free), BAgger (2512.12080, DAgger anti-drift), RL : AR-CoPO (2603.17461), KVPO (2605.14278).

**2. Ingénierie de cache / sink (training-free pour la plupart).** Le motif universel : sink persistant + fenêtre glissante + RoPE re-ancré. Le sink devient progressivement plus malin : sink frame fixe (LongLive 2509.22622, Rolling Forcing 2509.25161) vers deep sink à phase RoPE réalignée (Deep Forcing 2512.05081, Dynamic 57.56) vers sink EMA fusionnant les tokens évincés (Reward Forcing 2512.04678, Dynamic 66.95) vers mémoire structurée Sink/Tail/History (Relax Forcing 2603.21366, Dynamic 65.67) vers sink dynamique retrouvé avec gate anti-collapse (DySink 2605.21028, Dynamic 63.52) vers politiques per-head/future-aware (Pyramid Forcing 2605.13111, Future Forcing 2605.30083, Echo-Forcing 2605.16003, MemRoPE 2603.12513). Plus efficacité pure : VideoMLA (2605.30351, KV -92.7%), HiStream (2512.21338, 76x), étude quantization 33 méthodes (2603.27469), AdaState (2605.30349, anti-anchor).

**3. Mémoire de récupération explicite (revisit / re-ID).** Deux sous-types. (a) Scène/pose-keyed (monde 3D) : Context-as-Memory (2506.03141, FOV-overlap), WorldMem (2504.12369), VMem (2506.18903, surfels), Memory Forcing (2510.03198), spmem (2506.05284), I3DM (2603.23413), RELIC (2512.04040), GIM-World (2606.02436), AnchorWeave (2602.14941, revisit PSNR 21.04). (b) Contenu/entité-keyed (offline, multi-shot, script/agentique) : EntityMem/EntityBench (2605.15199), StoryMem (2512.19539), VideoMemory (2601.03655), A2RD (2605.06924), IAMFlow (2605.18733). Plus retrieval sparse sur le cache brut : MoC (2508.21058), OmniMem (2605.30519, Dynamic 82.29), MemFlow (2512.14699).

**4. Mémoire APPRISE implicite (fast-weight / SSM / TTT) -- NOTRE famille.** VideoSSM (2512.04519, gated-delta SSM + gate de décroissance, DMD self-rollout), ARL2 (2605.16579, Gated-DeltaNet linéaire), MAG (2512.18741, cache KV compressé encode/décode, définit MAG-Bench), TTT-Video (2504.05298, MLP fast-weight nonlinéaire, teacher-forced), LaCT (2505.23884, large-chunk TTT SwiGLU), Long-Context SSM World Models (2505.20171, Mamba bloc), Titans (2501.00663, parent langage : surprise-write + decay gate), Fast Spatial Memory/Elastic-TTT (2604.07350, EMA + Fisher elastic), DiT-Mem (2511.19229), REFINE (2602.16704), MemoryPack (2510.01784).

**5. Notre base : Anchor Forcing (2603.13405).** Cache-centric, non appris : sink cache (3f) + junction cache (3f post-switch) + local rolling (W=9) + re-cache guidé par anchor au switch + RoPE tri-région bornée. Headline : Dynamic 73.00 en interactif 60s vs LongLive 26.37. Ne rapporte AUCUN revisit/re-ID.

## Sur quoi tout le monde se bat (et ce qui est saturé)

- **subject_consistency : SATURÉ, 0.88-0.99 partout.** VideoSSM 92.51, LongLive 91.09, MAG-30s 99.18, ARL2 95.63, AAPT 96.22, IAMFlow 95.88. Wan2.1 base est déjà ~0.97. Ce n'est PAS un axe de compétition.
- **background_consistency ~0.93-0.98, motion_smoothness ~0.98, temporal flickering ~0.97 : saturés partout.**
- **VBench Total 5s : saturé**, tout le monde dans ~1 point : CausVid 81.20 (le plancher), puis Self-Forcing 83.00, MAG 83.52, LongLive 83.52, VideoSSM 83.95, Anchor-Forcing 83.99, Diagonal 84.48. Statistiquement mince, ingagnable de façon convaincante.
- **Débit temps réel : table stakes**, 17-31 FPS sur un seul H100/B200 (Diagonal 31, AAPT 24.8, Reward 23.1, LongLive 20.7). Être plus rapide ne fait plus un papier.
- **Longueur max : résolue.** Self-Forcing++ 4min15 (99.9% de la capacité RoPE), Rolling Sink 5-30min, LongLive 240s. La longueur brute est saturée ; la STABILITÉ à la longueur est la vraie question.
- **RoPE / décalage positionnel : résolu au moins 3 fois** (Anchor tri-région, Context borné [0,N], Rolling/Future remap dynamique).
- **dynamic_degree à l'échelle minute : ÉTAIT le champ de bataille, désormais bondé.** Passé de 30-45 (LongLive 37.5, Self-Forcing 35) à 57-67 par pure chirurgie de sink training-free (Deep 57.56, Relax 65.67, Reward 66.95, DySink 63.52), et même OmniMem 82.29, Anchor-Forcing 73.00, Pyramid+Causal 86.39 à 60s. Rendements décroissants : il faut désormais ÉGALER ces chiffres, pas les améliorer.
- **Efficacité / mémoire linéaire : bondé** (ARL2 2.26x/-54%, OmniMem 3.3x, VMem 12x, MoC 7x, VideoMLA -92.7%). Difficile à gagner seul, dominé par des groupes à ressources kernel.
- **Mémoire apprise gated-decay + write via self-rollout : essentiellement FERMÉ.** VideoSSM (gate alpha + beta, DMD self-rollout, même backbone Wan-1.3B/chunk-3/4-step) est notre thèse exacte, déjà publiée et SOTA. Le decay gate seul n'est pas novateur (Titans, ARL2 forget gate, MemRoPE EMA, Elastic-TTT). Et 2602.21204 (NVIDIA, ICML 2026) prouve que la mémoire TTT fast-weight EST de l'attention linéaire apprise, ce qui dégonfle la nouveauté architecturale. REFINE (2602.16704, GRPO) et Resampling Forcing entraînent déjà le write sous rollout.

## Les espaces blancs réels (problèmes non résolus)

1. **Re-identification / revisit SUJET, contenu-keyed, en streaming distillé interactif.** Toutes les méthodes contenu-keyed (EntityMem, StoryMem, VideoMemory) sont offline, multi-shot, script/agentique, full-step+LoRA, et pré-cuisent la banque d'entités AVANT génération. Personne ne fait du re-ID sujet en ligne, sans entités connues, à vitesse 4-step streaming.
2. **Sujet DÉCOUPLÉ de la pose caméra (A vers B vers A avec prompt switch).** Toutes les méthodes pose-keyed récupèrent par FOV caméra ; un sujet qui part et revient DANS UNE SCÈNE DIFFÉRENTE leur est invisible. Une mémoire content-addressable du sujet survivant à un changement de scène/prompt est ouverte.
3. **Aucune mémoire récurrente/fast-weight APPRISE n'a jamais été benchmarkée sur revisit/re-ID.** VideoSSM, ARL2, TTT-Video, LaCT sautent tous cet axe. Le ring du re-ID est occupé seulement par cache-compressé (MAG) et retrieval/symbolique (A2RD, IAMFlow, Context-as-Memory, DySink, AnchorWeave). Une mémoire récurrente entrant sur MAG-Bench est un territoire littéralement inoccupé.
4. **Le WRITE nonlinéaire fast-weight entraîné à travers le self-rollout causal** avec cache rolling in-place reste non fait : VideoSSM/ARL2 utilisent un état LINÉAIRE (scan analytiquement différentiable) pour l'esquiver ; LaCT/TTT-Video font du TEACHER-FORCING pour l'esquiver. C'est notre route Problème-B. Mais ne vaut que si nonlinéaire > linéaire sur l'identité long-horizon (seule preuve : TTT-Video +34 Elo).
5. **Plafond loin d'être atteint sur le re-ID.** MAG-Bench revisit PSNR 18.99-20.77 / SSIM 0.60-0.66 / LPIPS 0.17-0.23 ; EntityMem llm_face_accuracy 0.406, face_fidelity 0.740 ; LVBench-C character ~0.44 (5min). Le problème sous-jacent est GÉNUINEMENT non résolu, pas juste sous-benchmarké.
6. **L'intersection 3-way inoccupée :** mémoire récurrente apprise X prompt-switch interactif (régime Anchor-Forcing) X revisit/re-ID sujet. Anchor-Forcing possède le switch mais avec un cache HAND-DESIGNED et zéro re-ID. VideoSSM a la mémoire apprise mais pas de switch ni de re-ID. MAG/DySink font le revisit mais single-prompt et retrieval/compression. Aucun papier dans les trois cellules.
7. **Trou de benchmark :** aucun benchmark ne mesure le re-ID sujet APRÈS un prompt switch. MAG-Bench = revisit caméra géométrique sous UN prompt ; EntityBench = multi-shot scripté ; Context-as-Memory = FOV pose. Aucun ne teste "le sujet quitte la fenêtre pendant un switch A vers B et doit être rappelé au retour vers A". Définir ce benchmark est en soi de l'espace blanc.

## Où NOUS pouvons nous placer/concurrencer (honnête, avec la baseline à battre)

**Lecture brutale.** Notre cadrage headline -- "mémoire fast-weight/TTT apprise + decay gate, entraînée via DMD self-rollout sur Wan2.1-1.3B" -- est déjà publié comme VideoSSM (2512.04519) : même backbone, gate injection + décroissance, état gated-delta, DMD self-rollout, et il bat déjà LongLive/Self-Forcing (Dynamic 50.50 vs 37.5/35.0). Combiné à 2602.21204 (TTT = attention linéaire) et REFINE/Resampling (write sous rollout déjà fait), la cellule "mémoire gated apprise + write self-rollout" est FERMÉE. On ne gagne pas là.

**Le seul créneau défendable : l'intersection 3-way.** Une mémoire récurrente apprise, content-addressable, à write entraîné par self-rollout causal, dans le régime Anchor-Forcing interactif (switch), évaluée sur le re-ID/revisit SUJET (A vers B vers A). Cela esquive : le cluster scène/pose-keyed fermé (on n'est pas caméra-keyed) ET le cluster entité offline (on est streaming/distillé, sans banque pré-cuite).

**Baselines à battre, explicitement, sur le MÊME axe :**
- VideoSSM (2512.04519) : la mémoire apprise gated-delta linéaire. Il faut montrer que le write nonlinéaire par gradient bat le gated-delta à COMPUTE ÉGAL.
- ARL2 (2605.16579) : l'état GDN fast-weight linéaire, write sur passe clean.
- MAG (2512.18741) : le leader actuel du re-ID, cache compressé, MAG-Bench PSNR 18.99-20.77. Il faut un nombre revisit loin du plafond ET le battre.
- DySink (2605.21028) et Context-as-Memory (2506.03141) : les baselines retrieval. Risque structurel : le retrieval RELIT les pixels exacts, donc il est structurellement avantagé sur le rappel d'identité exacte.
- Anchor-Forcing lui-même (2603.13405) : son anchor re-cache hand-designed sur le revisit interactif.

**Axes de victoire à revendiquer :** revisit PSNR/SSIM/LPIPS (MAG-Bench-style) + CLIP-per-segment interactif + dynamic_degree sous switching. JAMAIS subject/background_consistency (saturés 0.92-0.99), JAMAIS VBench Total 5s (saturé 83-84.5).

## Ce qu'il faudrait pour gagner cet espace

1. **NE PAS pitcher le decay gate ni "self-rollout write" comme la nouveauté.** Ils sont pris (VideoSSM, ARL2, REFINE, Titans). La nouveauté doit vivre dans (a) le write NONLINÉAIRE par gradient à travers le rollout causal, et (b) la TÂCHE re-ID interactif.
2. **Construire/adopter un benchmark re-ID sujet sous prompt-switch** (sujet quitte fenêtre+sink pendant A vers B, doit être rappelé au retour A). Il n'existe pas. C'est une contribution en soi, et c'est benchmark-led, ce qui est défendable.
3. **Expérience de GATING (go/no-go) :** une sonde rapide montrant que notre mémoire nonlinéaire bat un baseline gated-delta linéaire sur l'identité long-horizon. Si non, l'espace est fermé, on pivote.
4. **Comparaison à compute APPARIÉ contre la meilleure politique training-free** (Pyramid 2605.13111 / Future 2605.30083 / Deep 2512.05081). C'est la barre la plus dure : Pyramid+Causal atteint déjà Dynamic 86 à 60s GRATUITEMENT. Si on ne bat pas le training-free à coût égal, il n'y a pas de papier.
5. **Ablations head-to-head obligatoires :** vs état GDN d'ARL2, vs anchor re-cache d'Anchor-Forcing, vs retrieval DySink, sur le MÊME benchmark revisit interactif.
6. **Attention au conflit de nom :** MAG publié (2512.18741) vs notre "MAG" interne. Renommer.
7. Contrainte de réalité interne (nos notes) : le BPTT cross-chunk est infaisable au-delà de ~5s ; le write reste cappé within-chunk (21f). Donc viser within-chunk write + eval 60/120/240s, refactor seulement si le drift est horizon-limité.

## Verdict (l'angle le plus défendable, ou pivoter)

**Angle headline générique ("mémoire apprise gated-decay via self-rollout") : MORT.** VideoSSM est notre thèse exacte, publiée, SOTA. Ne pas y aller.

**Le seul angle défendable :** "une mémoire fast-weight content-addressable dont le WRITE NONLINÉAIRE est entraîné par self-rollout causal, dans le régime interactif d'Anchor-Forcing, évaluée sur le re-ID/revisit SUJET après prompt-switch (A vers B vers A)." C'est l'unique intersection 3-way inoccupée, sur le seul axe (re-ID) vraiment loin du plafond, où aucune mémoire apprise récurrente n'est jamais entrée.

**Condition de survie (gating experiment, à faire EN PREMIER) :** prouver, sur une sonde rapide, (a) que notre write nonlinéaire bat un gated-delta linéaire (style VideoSSM/ARL2) sur l'identité long-horizon à compute égal, ET (b) produire un nombre revisit loin du plafond qui bat MAG (20.77 PSNR) et qui approche le retrieval (DySink/Context-as-Memory). Si l'un des deux échoue, l'espace est effectivement fermé.

**Pivot si échec :** le gap "joint prompt-switch + long-horizon dynamics + revisit sur UN seul modèle", que personne ne gagne aujourd'hui (Anchor/Delta font le switch ; Deep/Relax/DySink le long-horizon ; AnchorWeave le revisit caméra). Gagner les trois sur un modèle est intégratif mais réel. Risque : le cluster entité (EntityMem, StoryMem, VideoMemory, Dec-2025 à Mai-2026) fonce vers le re-ID sujet ; la fenêtre se resserre. Différenciation = la combinaison streaming + distillé + write appris, pas le re-ID dans l'abstrait.

## Table compacte des papiers clés

| Papier | arXiv | Famille | Chiffre clé | Laisse de l'espace blanc pour nous ? |
|---|---|---|---|---|
| Anchor Forcing (NOTRE BASE) | 2603.13405 | Cache hand-designed interactif | Interactif 60s Dynamic 73.00 vs LongLive 26.37 ; pas de re-ID | Oui : aucun revisit/re-ID rapporté |
| VideoSSM | 2512.04519 | Mémoire apprise gated-delta | 60s Dynamic 50.50, Subj 92.51 ; pas de re-ID | NON sur l'archi (= notre thèse) ; oui sur re-ID + write nonlinéaire |
| ARL2 | 2605.16579 | Fast-weight linéaire (GDN) | 2.26x speedup, -54% VRAM ; Subj 95.63 | Concurrent direct ; pas de re-ID |
| MAG / MAG-Bench | 2512.18741 | Cache KV compressé | revisit PSNR 18.99-20.77 / LPIPS 0.17-0.23 | Leader re-ID mais loin du plafond ; pas de mémoire apprise |
| Self-Forcing | 2506.08009 | Distillation (pivot) | Total 83.00 ; 60s Dynamic 35.0 | Base universelle |
| LongLive | 2509.22622 | Switch + sink (Étape-1 base) | 20.7 FPS, 240s ; 60s Dynamic 37.5 | Battue par tous sur Dynamic |
| Self-Forcing++ | 2510.02283 | Horizon long | 4min15 (99.9% RoPE) ; 100s Dynamic 54.12 | Longueur saturée |
| Causal Forcing(++) | 2602.02214 / 2605.15141 | Distillation teacher AR | Dynamic 68 vs 57 (+19.3%) | Distillation, pas mémoire |
| Diagonal Distillation | 2603.09488 | Distillation diagonale | 31 FPS, Total 84.48 | Vitesse+qualité saturées |
| Rolling Forcing | 2509.25161 | Sink + rolling denoise | drift 0.01 vs SF 1.66 ; 15.79 FPS | Drift forward seulement |
| Deep Forcing | 2512.05081 | Sink training-free | 60s Dynamic 57.56 vs LongLive 45.55 | Barre Dynamic training-free haute |
| Relax Forcing | 2603.21366 | Sink/Tail/History | 60s Dynamic 66.49 | Dynamic saturé |
| Reward Forcing | 2512.04678 | EMA-sink + Re-DMD | 60s Dynamic 66.95 ; 23.1 FPS | Dynamic saturé |
| DySink | 2605.21028 | Retrieval dynamic sinks | 50s Dynamic 63.52 ; gate anti-collapse | Concurrent retrieval sur notre niche |
| Future Forcing | 2605.30083 | KV future-aware (TF) | 60s Subj jusqu'à 98.15 | Subj saturé |
| Pyramid Forcing | 2605.13111 | KV per-head (TF) | Causal+Pyramid Dynamic 86.39 @60s | Barre training-free très haute |
| OmniMem | 2605.30519 | Retrieval sparse KV | 60s Dynamic 82.29 (+52.3%) | Retrieval, pas de re-ID PSNR |
| Context-as-Memory | 2506.03141 | Retrieval pose/FOV | revisit PSNR 20.22 / LPIPS 0.30 | Revisit CAMÉRA, pas sujet |
| AnchorWeave | 2602.14941 | Retrieval point-cloud | revisit PSNR 21.04 / SSIM 0.674 | Revisit caméra, loin du plafond |
| EntityMem / EntityBench | 2605.15199 | Entité offline multi-shot | face_fidelity 0.740, llm_acc 0.406 | re-ID sujet non résolu mais offline |
| TTT-Video | 2504.05298 | TTT-MLP nonlinéaire (TF) | +34 Elo vs Mamba2/GDN | Seule preuve nonlinéaire > linéaire ; teacher-forced |
| LaCT | 2505.23884 | Large-chunk TTT | match full causal @14B | Teacher-force volontaire (gap ouvert) |
| Titans | 2501.00663 | Mémoire surprise+decay (langage) | >2M tokens | Decay gate déjà établi (pas novateur) |
| TTT = attention linéaire | 2602.21204 | Preuve théorique | TTT ≡ attention linéaire apprise | MENACE : dégonfle nouveauté archi |
| REFINE | 2602.16704 | Fast-weight GRPO rollout | write sous rollout (RL) | Conteste "train write via rollout" |
| MemRoPE | 2603.12513 | Mémoire EMA training-free | minute-à-heure, cache fixe | Pré-empte "decay memory" sans training |
| Echo-Forcing | 2605.16003 | Mémoire scène TF interactive | switch + scene-recall | Chevauche turf Anchor, training-free |
| AdaState | 2605.30349 | Anti-anchor adaptatif | critique sink/anchor sur Dynamic | Critique directe de notre approche sink |
| VideoMLA | 2605.30351 | Cache KV latent low-rank | KV -92.7% ; 60s Dynamic 0.958 | Efficacité bondée |

(Confiance globale des sweeps : 0.72-0.80. Les chiffres PDF d'Anchor-Forcing MAG-Bench ~29.84/0.892/0.082 sont NON vérifiés ; à confronter au repo/HTML avant citation.)
