# PLAN FINAL sink-virtuel (synthèse des 3 plans locaux, 2026-07-23)

Sources : localplan_A_design.md (review à charge), localplan_B_impl.md (implémentation),
localplan_C_train_eval.md (training/éval). Ce fichier = les DÉCISIONS. Le détail vit dans les 3 plans.

## Convergences (adoptées sans débat)
- Insertion = CONCAT AU MOMENT DE L'ATTENTION (template prompt_cache), jamais d'écriture
  physique dans le cache (8 sites d'arithmétique cassés sinon). Dans LES DEUX branches
  (sink>0 ET sink==0, la branche cache-dropout historiquement oubliée).
- Décodeur = sous-module de VideoMemoryLayer (périmètre "memory.*") → hérite FSDP-ignore,
  all-reduce, lr_mem, clip, EMA trainable-only, save/load, exclusion LoRA. (Bug n°1 sinon.)
- Décodage 1×/forward HORS de la région checkpointée, (k_virt,v_virt) en entrées explicites
  (contrat ttt_state/mem_ema_prev/mem_x_write, qui a déjà mordu 3 fois).
- MEM_ZERO_ALL = skip STRUCTUREL (aucune concat) = ablation exacte. Test d'acceptation :
  bit-identique à memory_layers:[] (selftest standard).
- Teacher RSD = généralisation du twin e3 (snapshot→recache splice→rollout no_grad
  MEM_ZERO_ALL→restore, même RNG = paire exacte). Teacher = memzero+splice (0.906), jamais
  memon+splice (cible mouvante). Pas de second modèle en VRAM (~11GB clones CPU/step revisit).
- Loss = MSE x0 dense normalisée RMS-teacher sur R=2 chunks de résurrection, + DMD partout
  (ancre on-manifold). λ_RSD calibré à L_RSD/L_DMD ∈ [0.3,0.5]. Phase 1 : LoRA generator
  GELÉE (lora_unfreeze_from_block: 30) → attribution pure mémoire.
- Curriculum = LISTE D'ÉVÉNEMENTS multi-switch (p_revisit 0.5, gap ≥ 24 frames = éviction
  totale garantie par assert, switch2 aligné frontière BPTT). Séquences sans revisit = DMD
  pur = régression do-no-harm gratuite. cache_dropout/text_dropout OFF en v1 (C).
- Tripwires wandb dès step 0 : masse d'attention des tokens virtuels/bloc (alerte 30%),
  RMS(k/v virt), gate/bloc, rsd/gap_r1-r2, rsd/causal_gain (twin memzero 1 step sur 10).
- Éval : arm memvirt dans le harness 24 cellules, seuils PRÉ-ENREGISTRÉS (delta > +0.05 ET
  sign-p < 0.05, cible = combler ≥ 25% du gap oracle-memzero), 8 paires FRAÎCHES held-out
  à l'éval finale seulement, DD segmenté + self_resA + ctrl_B_origA + œil humain
  (anti-wallpaper), baselines : oracle = upper-bound WorldKV-parfait, AdaState = citer +
  différencier (réimplémenter seulement sur exigence reviewer), stock = 3-way toujours.

## Arbitrages (là où A et B divergeaient)
1. **RoPE des clés virtuelles : option A retenue** = bande de frames FIXE sink-adjacente,
   grille spatiale réelle (m = 1560 = une frame synthétique pleine). Argument décisif : la
   thèse entière est "parler l'interface que le backbone maîtrise" ; toutes les clés jamais
   attendues sont ropées, et le prompt_cache (précédent maison) rope dans une bande fixe.
   Le no-rope de B (registres libres) = ablation v2, pas le défaut. Corollaires : activation
   des tokens seulement après la première éviction (collision de bandes) + assert bandes
   disjointes + gérer les deux modes rope comme prompt_cache.
2. **Gate : mécanisme A (merge log-sum-exp), règle anti-starvation A, SANS le double-zéro
   de B.** Le zero-init naïf n'est pas inerte (softmax à somme fixe, clé nulle = logit 0) ;
   le logit-bias −8 (V2 de B) affame le gradient (masse e−8). Le merge LSE avec g scalaire
   LINÉAIRE init 0 + skip structurel à g==0 donne l'inertie EXACTE et ∂out/∂g|0 ≠ 0.
   EXACTEMENT UN zéro dans la chaîne : g=0, to_v init normal (PAS zero-init, PAS de tanh-gate
   redondant). LSE : FA3 la renvoie déjà (jetée dans attention.py), seconde attention sur
   m=1560 clés en sdpa fp32. Fallback si LSE indisponible : V2 logit-bias (dégradé documenté).
3. **m = 1560 dès la v1** (A et le plancher oracle1 le justifient littéralement : une frame
   latente pleine suffit) ; {780, 390, 128} = ablation de compression APRÈS un premier
   résultat positif. ~21M params (3 blocs), <0.15GB/GPU.
4. **Multi-switch** : schedule événements + registre segment_tails dès la v1 (B étape 3) ;
   écriture à deux clés FiLM zero-init (B §0.4) implémentée mais ACTIVÉE en phase 2 seulement
   (un flag de plus au selftest byte-identical ; v1 = single-revisit d'abord, la courbe
   re-ID vs nb-de-segments viendra avec).

## Le DERNIER trou de gating (A §3.3) — probe 0$ AVANT toute ligne de code
L'oracle splice les 30 blocs ; le design n'insère qu'à [5,16,27]. Personne n'a mesuré un
splice restreint aux 3 blocs mémoire. **Probe "oracle3b"** : recache normal, snapshot du
cache, recache oracle, restauration des 27 blocs ∉ {5,16,27} → seuls les 3 blocs mémoire
portent le KV de A. Même harness 24 cellules, arm `oracle3b`.
Lecture : ≥ ~70% du delta survit (≥ +0.15) → [5,16,27] confirmé, GO implémentation.
Delta s'effondre → insérer plus large (cellule partagée, décodeurs par bloc) = re-chiffrer.
Bonus du même hook : probe position-sink (la frame A splicée côté sink plutôt que dernier
slot de fenêtre) pour valider la bande RoPE retenue — à faire tourner dans la même campagne
si le temps GPU le permet, sinon juste après.

## RÉSULTAT oracle3b (2026-07-23 soir) : **NO-GO du périmètre [5,16,27]**
oracle3b − memzero = **+0.023 (full) / +0.024 (crop60)**, signes 16+/8− / 17+/7−, p 0.15/0.06.
Seuil pré-enregistré : ≥ +0.15. **~10% seulement du delta oracle (+0.220) survit quand le KV
de A n'alimente que les 3 blocs mémoire.** L'info d'identité est distribuée en profondeur :
il faut alimenter (beaucoup) plus de blocs. Le probe à 40 min vient d'invalider le design à
décodeurs k/v par bloc AVANT toute ligne de code.

## PIVOT DE READOUT (à stress-tester avant adoption) : "frame synthétique recachée"
Au lieu de décoder l'état en k/v par bloc : **décoder l'état en UNE frame LATENTE synthétique
(16×60×104) et la RECACHER via le forward existant (recache 1 frame)** → chaque bloc des 30
calcule ses propres k/v depuis cette frame, contenu in-distribution partout, par construction.
Arguments : (a) c'est LITTÉRALEMENT le mécanisme d'oracle1 (+0.224, 24+/0−), notre chiffre le
plus solide : le student remplace la frame réelle par une frame décodée, le teacher EST la
cible ; (b) réutilise la machinerie recache existante ; (c) cible auxiliaire gratuite :
la frame décodée peut être supervisée à reconstruire la queue évincée de A (self-generated,
data-free, AHN-style au niveau frame) en plus du RSD end-to-end ; (d) UN décodeur (pas 30) ;
coût ≈ +1/3 de forward chunk par écriture d'état. Différenciateur vs AdaState (état = 1 frame
dénoisée) à défendre : notre frame est SYNTHÉTISÉE À LA DEMANDE depuis un état associatif
multi-sujet (adressable par le contexte courant), pas un buffer de la dernière frame ; la
préemption se re-vérifie (localplan_D à venir).

## BASELINE CRITIQUE identifiée (question Aden 2026-07-24) : la "frame bank"
Garder 1 frame par segment sortant + resplicer au retour du prompt = ~le score oracle1 sur le
revisit EXPLICITE, training-free. À AJOUTER comme baseline au papier (implémentation = oracle1
+ trigger auto). Conséquence : notre claim ne peut pas être "mieux que la bank sur revisit
explicite" (plafond commun = oracle). La mémoire apprise gagne sa vie sur : (1) revisits
IMPLICITES/paraphrasés (bank keyée texte = aveugle ; état adressé par contenu), (2) pas de
politique de snapshot (compression continue O(1) vs banque qui grossit + heuristiques),
(3) SYNTHÈSE conditionnée au contexte courant vs replay d'une frame figée (même-sujet-autre-
action sans contamination du décor), (4) multi-sujet dans un état fixe. L'ÉVAL DOIT contenir :
arm frame-bank + paires à retours paraphrasés/implicites + même-sujet-autre-action. Le
curriculum d'entraînement doit inclure des retours paraphrasés (pas seulement prompt identique).

## VERDICT AGENT D (localplan_D_pivot.md) : pivot adopté en version (a') deux étages
- **V1 = splice événementiel de la frame DÉCODÉE via frames_override EXISTANT** (mécanisme
  exact d'oracle1, zéro modif du cache). Décodeur entraîné OFFLINE par reconstruction des
  frames évincées (data-free, sans rollout, SANS RSD en v1 : loss = DMD + L_rec aux, plus de
  snapshot 9.2GB/step). Backbone zero-shot en v1. Dev ≈ 3-4 jours au lieu de 6.
- V2 si sous-oracle : hybride "phantom forward + concat" (k/v des 30 blocs capturés sans
  écrire le cache, gradient RSD propre, persistance par chunk).
- Découvertes code : (1) la position sink persistante est GRATUITE (frame en tête de
  frames_override) ; (2) localplan_B se trompait, le cache physique PORTE du gradient
  (CopySlices, le BPTT existant en dépend) : recache 1-frame avec graphe < 0.5GB ; (3) le
  vrai trou de gating restant = la TOLÉRANCE à l'imperfection de la frame synthétique.
- B2 (insertion attention) SUPPRIMÉ en v1 ; B1 = un seul décodeur état→frame ; B3/B6
  inchangés + miroir frames_override côté training (~15 lignes).
- Préemption : éviter la variante "frame rafraîchie chaque chunk" (convergerait vers
  AdaState) ; l'événementiel reste défendable (TTT multi-segment, synthèse à la demande,
  objectif+éval résurrection).

## Séquence d'exécution (révision 2)
0. ~~oracle3b~~ NO-GO 3-blocs. ~~Design pass pivot~~ FAIT (D).   ← ON EST ICI
0ter. ~~Balayage profondeur~~ FAIT : montée régulière 3(+0.02)→6(+0.11)→10(+0.16)→15(+0.20)→
   30(+0.22), pas de coude : l'identité est DISTRIBUÉE en profondeur, le recache (30 couches
   gratuites) est la seule voie rationnelle ; la répartition compte plus que le nombre
   (6 répartis >> 3 concentrés). Figure de papier.
   ~~oracle-dose~~ FAIT : **TOLÉRANCE ÉNORME** : ancre à σ=0.3 → +0.223 (identique au parfait),
   σ=0.6 (frame majoritairement bruit) → +0.200 (89% du gain), σ=0.9 → +0.128 (57%), tous
   24+/0− ou 23+/1−. Le backbone est un débruiteur : une ancre bruitée est IN-DISTRIBUTION.
   → le décodeur n'a PAS besoin d'être bon, juste sémantiquement correct. Dernier risque de
   faisabilité du design 2 : ÉLIMINÉ. (Quadrant sur les arms dose en cours : le bruit
   adoucit-il aussi le cut ×10 ?)
   QUADRANT (probe 0, 144 vidéos) : oracle/oracle1 = cut ×10.5 au retour + rewind du décor
   (−0.35) ; memzero/stock = doux (×1.3-1.45) mais identité perdue/moyenne. LE BON QUADRANT
   (identité + douceur) EST VIDE ; stock (ancre persistante AF) en est le plus proche →
   argument pour l'ancre décodée en position sink PERSISTANTE plutôt qu'en splice événementiel.
   AUDIT STOCK : notre colonne stock avait rope_enhance (Tri-Region d'AF, absent du LongLive
   officiel) → colonne stockoff (rope absolu fidèle) EN GÉNÉRATION ; on garde les deux (l'écart
   = la part positionnelle d'AF).
   ~~probe position~~ FAIT, courbe COMPLÈTE et définitive (delta re-ID / cut) :
   slot0 (l'instant d'avant) +0.224 / ×10.9 ; slot3 +0.019 / ×1.6 ; slot6 +0.016 / ×1.4 ;
   slot11 (zone morte du budget) 0.000 ; sink swappé à chaud −0.02 n.s. / ×1.4.
   **CONCLUSION : la résurrection est un phénomène de RÉCENCE PURE.** Le modèle continue depuis
   son passé immédiat déclaré ; une ancre placée ailleurs est à peine lue (3 frames en arrière =
   −92% d'effet). Puissance et violence sont indissociables PAR LA POSITION : la réconciliation
   doit se faire par le CONTENU de l'ancre, pas par sa position.

## DESIGN v1 FINAL (données en main, 2026-07-24 nuit)
1. **Position : récence, point.** L'ancre décodée est splicée au dernier slot du recache au
   revisit (mécanisme oracle1 exact, seul qui marche).
2. **Transition, par tier** : cross-scène (prompt change) → le cut est de la grammaire de
   montage, acceptable SI l'œil d'Aden confirme que le cut oracle ressemble à un cut de scène
   et pas à un glitch (clips calib envoyés). Même-scène (caméra continue) → l'ancre COMPOSITE
   (identité du sujet × géométrie du monde COURANT, cf. brainstorm_rejoin_vs_rewind) : le cut
   disparaît parce que le "passé immédiat déclaré" n'est plus un monde étranger. C'est
   naturellement ce que produit un décodeur conditionné au contexte → v2 ; la v1 (reconstruction
   de l'évincé) couvre le tier cross-scène.
3. Le reste du plan D inchangé : décodeur état→frame offline (reconstruction de l'évincé),
   splice via frames_override, pas de RSD en v1, 3-4 jours de dev, selftests byte-identical.
GO recommandé. Demandes à Aden : (a) verdict œil sur les clips calib (cut oracle = cut de
scène ou glitch ?), (b) commit de tout le code probes (non-commité !), (c) feu vert dev v1.

## CORRECTIF v1 (question Aden 2026-07-24 soir) : le BIAIS DE RÉCENCE de l'objectif naïf
"Reconstruire la frame la plus récemment évincée" apprend à lire la trace la plus FRAÎCHE de
l'état. Au revisit, la trace fraîche = B → le décodeur naïf ressusciterait B (la pathologie
memon historique). Plan corrigé :
- **v1.0** (collecte single-prompt EN COURS, inchangée) : gate scientifique "l'info est-elle
  récupérable de l'état ?" via reconstruction held-out. N'est PAS le verdict memdec.
- **v1.1** (avant tout verdict memdec) : (i) conditionnement du décodeur par le prompt du
  retour (cross-attn légère sur l'embedding T5 poolé ou FiLM) ; (ii) 2e collecte sur rollouts
  À SWITCH : paires (état post-écritures-B, cible = frame de A, condition = prompt de A) =
  l'exact task du revisit, data-free (collecteur à étendre : switch driver déjà dans le
  pipeline train/inférence). Lancer cette collecte PENDANT le train v1.0.
1'. B0→B6 révisés version D (3-4 jours) ; selftests C adaptés (TEACHER-* reportés en v2).
2'. Décodeur offline (~2 jours de train léger, verdict reconstruction MSE/DINO).
3'. Éval zero-shot splice-décodée vs oracle (le harness 24 cellules, arm "memdec").
4'. Si sous-oracle : V2 hybride + RSD end-to-end.
1. Étapes B0→B6 (~6 jours dev, chaque étape flag-OFF byte-identical + selftest S0-S6).
2. Selftests C#0-9 (held-out grep, TEACHER_NOOP bitwise, TEACHER==oracle-probe bitwise,
   éviction réelle, MEM_ZERO_ALL total, grad probe, rank-sync, profile_vram, harness).
3. Run RSD v1 : 2000 steps, 4 GPUs, ~36-40h wall, go/no-go à 500 (rsd/gap_r1 −20% sinon stop).
4. Éval : campagne memvirt 24 cellules + DD segmenté + fraîches + côte-à-côte pour Aden.
Budget total : ~2 semaines jusqu'au verdict v1. Deadline ICLR ~fin sept : 2 itérations possibles.
