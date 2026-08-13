# ULTRAPLAN 2026-08-04 : faire lire la mémoire, et finir le papier
Synthèse du panel (mécaniste / inventeur TF / économiste) + batterie SDEdit en cours.

## Le diagnostic unifié (trois rapports convergents)
1. MÉCANIQUE : les queries sont formées à SNR ≤ −4.4dB → la sélection d'attention est
   positionnelle, pas contentuelle. L'ancre est DÉCLARÉE t=0 (propre) alors qu'elle est
   floue >32px : hors-variété, atténuée par RMSNorm, absorbée par le contexte B.
   La texture d'instance est resynthétisée par le TEXTE au dernier pas (σ 0.625→0).
2. INFORMATIONNEL : l'ancre conserve exactement couleur+layout (bande basse) : tous nos
   échecs (§13) sont des échecs de COULEUR : désadaptation d'impédance, pas mémoire vide.
3. ÉCONOMIQUE : la monnaie rare = jours-Aden (~6 restants d'exploration) ; taux mesuré de
   destruction de claims 4/6 manches ; l'écriture est le chemin critique dès le 8 août.

## LE PLAN (v. économiste, amendé par le panel)
- **J0 (aujourd'hui)** : COMMIT ADEN (bloquant). Réécriture abstract+claim1 à la vérité
  §11-13 (2h). Analyses 0-GPU : T0 (corrélation bande-basse des mp4 wronganchor existants :
  peut requalifier "pas lu" en "lu dans une bande invisible au juge") + pré-écran L1
  (accord couleur ancre/A* sur les 24 dumps : décide la légende-VLM avant tout rollout).
- **LA NUIT DES PROBES (une seule, bornée, ~65 GPU-h, 3 GPUs)** :
  (a) DÉFENSIF OBLIGATOIRE : contrôles wronganchor/noise sur SS2 24 cellules (= GATE A du
  finetune), attribution SLOT_OFFSET, seeds 1-2 S2/S2p, memdec20r.
  (b) OFFENSIF SÉLECTIONNÉ (si pré-écrans verts) : L1 légende-VLM (5 cellules, jumeaux
  wrongcaption NÉGATIF exigé + paddedprompt + framebank+VLM), I1 déclarer-le-bruit-de-
  l'ancre (10 lignes, le seul correctif qui nomme le défaut mesuré), I3/T5 additif à Δ=1
  (masse vs score : sépare lire/effacer), T1/T2 mesures de masse par source.
  (c) la batterie SDEdit (en cours) rejoint ce lot.
- **GATE A (7 août)** : contraste contenu (wronganchor) ≥ +0.08 sur ≥4/5 SS2 → l'option
  finetune vit. Sinon : prémisse falsifiée, AUCUN finetune, et le papier intègre le résultat.
- **7-15 août** : Memento (code+poids publics) tourné CHEZ NOUS comme baseline entraînée
  (paie dans les 2 branches) + MBench (défense métrique externe). ÉCRITURE dès le 8 août,
  chemin critique (sections indépendantes du claim d'abord).
- **GATE B (15 août)** : si A passé ET un levier TF a ouvert un contraste → on amplifie le
  levier TF (pas de finetune). Si A passé et TF nul → finetune 18-22 août, borné, critère
  pré-enregistré, UN redémarrage, kill dur le 22. Si A échoué → écriture + défense.
- **1er SEPT : GEL D'EXPLORATION ABSOLU.** 15 sept : draft gelé. 15-24 : polish/soumission.
- Règle d'arrêt : si (sections non écrites × 1 jour) > (jours jusqu'au 15/09 − 7), tout
  calcul s'arrête.

## Ce qui peut faire gagner le "mémoire qui marche" (par ordre de probabilité panel)
1. L1 légende-VLM (texte = le canal prouvé ; contrôle directionnel fort) : TF, 1.5 GPU-h.
2. I1 ancre-déclarée-bruitée (le correctif du mensonge de métadonnée) : TF, 10 lignes.
3. Finetune teacher-texte (la version apprise de L1, paires contrefactuelles) : 5j, ~20-25%.
4. I2 transplant k/v ; I3 additif Δ=1 ; L2 SDEdit-en-sortie ; L3 mémoire-vérificateur.
Règle transverse : le comparateur de CHAQUE bras est son propre wronganchor (jamais memzero),
seuil ≥ +0.08 sur ≥4/5 ; L1/L4 exigent en plus le delta NÉGATIF du jumeau faux.
