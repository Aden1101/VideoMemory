# Brief MÉMOIRE 2.0 : co-entraîner l'ÉCRIVAIN (GO Aden 2026-07-27)

## Le diagnostic qui déclenche ce chantier (verrouillé par élimination)
Trois décodeurs (v1.0 naïf, v1.1 conditionné, v1.2 loss-sujet + 2.5× données) → même plateau
re-ID +0.11/+0.15 (~55% de l'oracle +0.224), val 0.87→0.81 sans gain re-ID, ancres = scène
correcte mais sujet infidèle (cheval brun au lieu de blanc). Le goulot n'est PAS le lecteur :
**l'information d'INSTANCE n'est pas (assez) dans l'état.** L'écrivain de l1 (gelé, entraîné
à l'aveugle par DMD via un canal mort) stocke la catégorie et la scène, pas l'instance.
Objectif 2.0 : que l'écriture apprenne à stocker le détail d'instance PARCE QU'on le relit.

## L'INSIGHT ARCHITECTURAL CENTRAL : l'écrivain silencieux ⇒ entraînement 100% OFFLINE
Dans le design 2 final, la mémoire N'INJECTE PLUS RIEN dans le forward (MEM_ZERO_ALL=1 en
permanence ; elle ne s'exprime qu'au revisit via la frame décodée splicée). Donc les entrées
d'écriture : les hidden states h des chunks évincés aux blocs [5,16,27] (le x_write de
l'eviction-feed) : sont produites par le backbone GELÉ sans aucune rétroaction de la mémoire
⇒ **h est une DONNÉE, indépendante des paramètres de l'écrivain.** On peut donc :
1. collecter une fois les SÉQUENCES de h (par bloc, par chunk évincé) + frames latentes +
   cond, sur des rollouts courts à switch ;
2. entraîner offline (write-REPLAY) : état_0 = init → pour chaque chunk évincé :
   état = write_θ(état, h_chunk) (différentiable en θ_write : qkv/lrs/decay de la cellule)
   → aux positions échantillonnées : décoder l'état (décodeur co-entraîné ou warm-start) →
   loss de reconstruction (pondérée sujet) sur la frame cible (+ cible d'instance, cf. test
   discriminant) ;
3. AUCUN modèle vidéo dans la boucle : itérations rapides, BPTT à travers les writes
   (attention à la profondeur : ~15-25 writes par vidéo, gradient checkpointing des writes
   au besoin ; les leçons snapshot/restore ne s'appliquent PAS ici, pas de cache).
Validation finale inchangée : arm memdec2.0 dans le harness 24 cellules.

## Budgets et contraintes DURES
- h par chunk évincé par bloc : (1, 4680, 1536) → 14.4MB bf16, ×3 blocs ≈ 43MB/chunk.
  Vidéo 90 frames ≈ 25 writes ≈ 1.1GB. **Disque projet : ~350GB libres, plafond 200GB pour
  cette collecte** → arbitrages à chiffrer par le design : nb vidéos × longueur × fp16-vs-fp8
  (le write tolère-t-il h fp8 ? à tester tôt : la maison a prouvé une tolérance énorme côté
  ancres, à vérifier côté h), tokens sous-échantillonnés = NON (change la sémantique du write).
- Home disk quasi plein (15GB libres) : TOUT sur le disque projet.
- GPUs partagés/volatils : pré-check DANS la commande, marge >25GB, co-run sur nos GPUs.
- Règles maison : selftests byte-identical, wandb obligatoire, py_compile, code additif
  (l'écrivain 2.0 = nouveaux params chargés par-dessus, l'inférence memdec doit pouvoir
  charger un "writer ckpt" en plus du decoder ckpt : hook MEM_WRITER à prévoir, symétrique
  de MEM_DECODE).
- Held-out : les 8 paires du probe exclues de tout corpus (grep-selftest).
- Question ouverte au design : le décodeur est-il co-entraîné from-scratch, warm-start v1.2,
  ou gelé d'abord (curriculum écrivain-seul puis joint) ? Et l'écriture à deux clés
  (FiLM prompt sur les clés d'écriture) entre-t-elle en 2.0 ou en 2.1 ?
- Cible d'entraînement : frames évincées (comme v1) ET/OU un terme d'instance explicite
  (le test discriminant en cours dira si l'échec est régression-de-catégorie : si oui, une
  loss contrastive inter-vidéos sur les états/ancres est le candidat n°1 pour forcer le
  détail discriminant).
- Éval de sortie : memdec20 ≥ +0.16 visé (battre le plateau +0.12), ancre du cheval PÂLE,
  et régression : les writes 2.0 ne cassent rien (la mémoire est silencieuse en génération,
  donc risque quasi nul par construction : le vérifier quand même).

## Références
plan_final_sink_virtuel.md (tout l'historique de gating), design2-frame-recall-gating (mémoire),
localplan_B/C/D, wan/modules/video_memory.py (cellule : step()/write_only, AnchorFrameDecoder),
reference/runs/frame_dec/ (collecteur/trainer/harness existants), causal_model.py L1342-1385
(eviction-feed : où h est poppé : le collecteur 2.0 sauvera CES tenseurs).
