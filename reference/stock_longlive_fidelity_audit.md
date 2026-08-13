# Audit de fidélité : arm "stock LongLive" vs LongLive publié (v1.0)

Date : 2026-07-24.
Question : notre colonne stock (poids officiels Efficient-Large-Model/LongLive-1.3B, lora.pt,
exécutés dans NOTRE code) reproduit-elle le comportement du LongLive PUBLIÉ dans SON code ?

## Sources vérifiées

- Officiel : github NVlabs/LongLive, branche `v1.0` (la branche `main` est passée à LongLive 2.0,
  Wan2.2-TI2V-5B, non pertinente pour la 1.3B). Fichiers téléchargés et lus en intégralité :
  `configs/longlive_inference.yaml`, `configs/longlive_interactive_inference.yaml`,
  `configs/longlive_inference_infinity.yaml`, `pipeline/switch_causal_inference.py`,
  `pipeline/interactive_causal_inference.py`, `wan/modules/causal_model.py`,
  `wan/modules/causal_model_infinity.py` (copies dans le scratchpad de session).
- Nous : `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/reid_probe_l1/stock_probe.yaml`,
  `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/pipeline/switch_causal_inference.py`,
  `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/wan/modules/causal_model.py`,
  `/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/reid_probe_l1/run_reid_probe.py`
  (campagne : frames=240, switch1=81, switch2=162).

## Config d'inférence officielle LongLive 1.0 (longlive_inference.yaml / interactive)

denoising_step_list [1000, 750, 500, 250], warp_denoising_step true, num_frame_per_block 3,
local_attn_size 12, timestep_shift 5.0, sink_size 3, global_sink true, context_noise 0,
use_ema false, LoRA rank 256 / alpha 256 sur longlive_base.pt. Interactive : num_output_frames 240,
switches tous les 40 frames latents. Au switch (pipeline switch et interactive, mêmes mécaniques) :
KV cache CONSERVÉ (global_sink true), cross-attn cache resetté, recache des derniers
min(local_attn_size=12, position) frames sous le NOUVEAU prompt à timestep context_noise=0,
sink protégé pendant le recache (write_start_index = max(local_start, sink_tokens) quand
is_recompute), `sink_recache_after_switch = not global_sink` = False.

RoPE officiel : ABSOLU. `causal_rope_apply(…, start_frame=frame_courant)`, les clés sont stockées
DÉJÀ ropées à leur position absolue, le sink garde ses positions d'écriture (frames 0..2), la
query est ropée à sa position absolue. Aucun re-mapping positionnel : au frame N, la distance
query→sink vaut N (au revisit de notre probe : 162 à 240 frames). Le paramètre `rope_enhance`
N'EXISTE PAS dans le causal_model.py officiel. Le README v1.0 crédite un contributeur externe
pour une variante "KV-cache relative RoPE" (causal_model_infinity.py, block-relativistic RoPE,
config séparée avec use_infinite_attention: true, num_output_frames 1050) : NON-DÉFAUT, absente
des configs publiées standard et interactive. LongLive 2.0 ajoutera un multi_shot_rope_offset :
deux indices convergents que le far-rope du sink est la faiblesse reconnue du 1.0.

## Tableau de divergences

| Réglage | LongLive officiel v1.0 | Notre arm stock | Verdict |
|---|---|---|---|
| denoising_step_list | [1000,750,500,250], warp true | idem | fidèle |
| num_frame_per_block | 3 | 3 | fidèle |
| timestep_shift / scheduler | 5.0, même scheduler (même lignée de code) | 5.0 | fidèle |
| local_attn_size (fenêtre) | 12 frames latents | 12 | fidèle |
| sink_size | 3 frames épinglées | 3 (natif, choix documenté dans le yaml) | fidèle |
| global_sink (KV gardé au switch) | true | true | fidèle |
| Recache au switch | derniers min(12, pos) frames, nouveau prompt, cross-attn reset, sink protégé | code hérité identique (`_recache_after_switch`) | fidèle |
| A→B→A double switch | pas dans leur switch pipeline, mais l'interactive officiel fait du multi-switch avec exactement la même mécanique par switch | notre extension = même mécanique appliquée 2 fois | fidèle (protocole) |
| context_noise | 0 | 0 | fidèle |
| EMA / LoRA rank-alpha / base ckpt | false / 256-256 / longlive_base.pt | false / 256-256 / base.pt (sha256 = HF, audit antérieur) | fidèle |
| **RoPE** | **absolu, clés cachées ropées à position d'écriture, distance query→sink = 162-240 au revisit** | **rope_enhance: true = Tri-Region d'Anchor-Forcing : clés stockées NON ropées puis re-ropées à chaque step en 3 bandes, sink à 0-2, fenêtre locale à ~12-21, query épinglée à min(21-3, pos) = 18, distance query→sink ≤ 20 pour toujours** | **DIVERGENT** |
| use_prompt_cache (junction KV d'AF) | inexistant | true dans le yaml, mais INERTE : SwitchCausalInferencePipeline.inference() n'initialise ni ne passe jamais prompt_cache au générateur (seuls causal_inference.py et interactive_causal_inference.py le font), le wrapper reçoit donc None | divergent sur le papier, sans effet sur les 24 vidéos |
| memory / ttt | inexistant | memory_layers [] → use_memory=False sur les 30 blocs, ttt_cache et switch_global_frame ne nourrissent que la branche mémoire (absente) | no-op vérifié |
| seed | 0/1 | géré par cellule par le runner | sans objet |

Vérifié aussi : notre chemin rope_enhance=False est sémantiquement identique à l'officiel
(stockage des clés ropées absolues, attention sink+fenêtre, mêmes index de roll/insert), donc
flipper le flag suffit à retrouver l'inférence officielle.

## Impact attendu de la divergence RoPE

(a) Identité longue durée : le Tri-Region maintient l'ancre (sink) à une distance positionnelle
≤ 20 frames, DANS la fenêtre d'entraînement de Wan (21). L'officiel lit le sink à 160-240 frames
de distance : extrapolation RoPE, attention à l'ancre atténuée/dispersée. La lora officielle a
certes été entraînée (streaming long tuning) sous rope absolu, donc elle "sait" lire un sink
lointain, mais l'existence même du patch infinity communautaire et du rope_offset de la 2.0
indique que cette lecture se dégrade. Notre stock est donc vraisemblablement AVANTAGÉ sur la
lisibilité de l'ancre. Contre-argument honnête : le régime tri-region est hors-distribution pour
cette lora (jamais entraînée avec), le signe n'est pas prouvable par le code seul.

(b) Revisit A→B→A : après le recache B→A, le SEUL contenu visuel de l'ère A restant dans le
cache est le sink (3 frames, protégées), le reste de la fenêtre est du contenu B ré-encodé sous
le prompt A. La résurrection stock repose donc entièrement sur la lecture du sink. Le Tri-Region
la maximise (ancre "à côté" de la query), l'officiel la lit à ~162 frames. Attendu : la
résurrection officielle ≤ la nôtre, donc le 0.81 est plutôt une BORNE HAUTE
("poids LongLive + lisibilité d'ancre AF") qu'une mesure du LongLive publié.

Validité interne préservée : memon et memzero (probe_l1.yaml) tournent aussi en
rope_enhance: true, le delta causal mémoire (−0.02/−0.03) est symétrique et reste valide.
C'est uniquement l'ÉTIQUETTE "stock LongLive = 0.81" qui est infidèle au système publié.

## Verdict et recommandation

NON fidèle sur un point : rope_enhance: true applique le Tri-Region RoPE d'Anchor-Forcing
(arXiv 2603.13405), mécanique ABSENTE de l'inférence officielle LongLive 1.0 (rope absolu).
Tout le reste (sink 3, fenêtre 12, recache 12 frames au switch, steps, shift, block 3,
global_sink, context_noise, LoRA) est conforme. use_prompt_cache: true est infidèle sur le
papier mais prouvé inerte dans ce pipeline.

À faire dans stock_probe.yaml :
1. `rope_enhance: false`
2. `use_prompt_cache: false` (hygiène, aucun effet attendu)

Puis re-générer la colonne stock des campagnes (24 vidéos, ~1h) pour obtenir le vrai chiffre
"LongLive publié". Garder l'ancienne colonne sous l'étiquette "LongLive + tri-region rope" :
l'écart entre les deux quantifie gratuitement la part du 0.81 due au positional scheme d'AF
vs aux poids LongLive, un point utile pour le papier (le mur d'invariance de B).
