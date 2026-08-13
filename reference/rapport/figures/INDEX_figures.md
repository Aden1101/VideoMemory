# Figures du rapport de stage

Dossier : `reference/rapport/figures/`  (toutes les images du rapport, PNG)

| Fichier | Ce que c'est | Légende suggérée | Section |
|---|---|---|---|
| `memory_module_diagram.png` | Schéma du module de mémoire MAG (lecture / injection / écriture + FIFO d'éviction) | *Architecture du module de mémoire à poids rapides inséré dans un bloc transformer de Wan : la lecture interroge l'état W avec les tokens courants, l'injection réinjecte via une porte apprise (gated-AdaLN), l'écriture Muon met à jour W à partir du chunk évincé (innovation eviction-feed).* | 6.3 (architecture / méthode) |
| `drift_breakdancer_60to120s_memoff.png` | Filmstrip 2 min, mémoire OFF : dérive d'attribut (pantalon bleu → blanc) | *Dérive d'identité sur une génération de 2 minutes sans mémoire : la couleur du pantalon change (bleu → blanc) et la scène glisse. Illustration du problème que l'on attaque.* | intro / évaluation (le problème) |
| `drift_firefighter_60to120s_memoff.png` | Filmstrip 2 min, mémoire OFF : « téléportation » de scène (rue enfumée → champ) | *Le pompier est « téléporté » de scène en scène au fil du rollout : le contexte visuel dérive faute d'ancrage long-terme.* | intro / évaluation (le problème) |
| `drift_chef_60to120s_memoff.png` | Filmstrip 2 min, mémoire OFF : exemple plus stable (alternative plus calme) | *Exemple plus stable : le sujet et la scène tiennent mieux, mais les ustensiles et la composition dérivent lentement.* | intro / évaluation (optionnel) |
| `vram_oom_battle.png` | Tableau : faire tenir l'entraînement en VRAM (2 GPU, fenêtre 42 → 24 frames) | *La « bataille OOM » : sur 2 GPU, une série d'optimisations cumulées réduit le déficit mémoire jusqu'à faire tenir l'entraînement de l'écriture, à une fenêtre de 24 frames (~5.8 s), mesuré.* | 6.5 (ingénierie / VRAM) |

Notes :
- Les filmstrips sont extraits des vidéos **mémoire OFF** (dérive brute) = ils montrent le problème, pas notre correction. 6 images réparties sur les ~2 minutes (t = 0 à ~120 s).
- Le schéma est le cœur de la méthode : OK pour un rapport d'école (pas de risque publication ici).
- Les vidéos côte-à-côte (mem-ON | mem-OFF | AF-stock) restent dans les dossiers d'éval `reference/runs/*/eval*/sidebyside/` si tu veux des captures supplémentaires.
