#!/usr/bin/env python
# ============================================================================================
# Telechargement des poids, aux CHEMINS que le code attend.
#
# Ajoute le 2026-09-09, apres la perte de l'arbre de travail : le README donne bien les deux
# commandes huggingface-cli, mais les chemins exacts (et le fait que le tokenizer vive dans un
# sous-dossier `google/umt5-xxl/`) ne se devinent qu'en lisant utils/wan_wrapper.py. Ce script
# les fige.
#
#   python download_weights.py            # les deux depots, reprise automatique
#   python download_weights.py --only af  # seulement les checkpoints Anchor-Forcing
#
# Reprise : huggingface_hub reprend un fichier interrompu, donc relancer est sans risque et
# sans cout. Rien n'est ecrase si le fichier est deja complet et de bonne empreinte.
# ============================================================================================
import argparse
import os
import sys
import time

REPOS = {
    "wan": ("Wan-AI/Wan2.1-T2V-1.3B", "wan_models/Wan2.1-T2V-1.3B", 17.6),
    "af": ("young98/AnchorForcing", "ckpt", 8.8),
}


def human(n):
    return f"{n/1e9:.1f} Go"


def local_size(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def main():
    ap = argparse.ArgumentParser(description="poids Wan2.1 + Anchor-Forcing")
    ap.add_argument("--only", choices=sorted(REPOS), default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    keys = [a.only] if a.only else list(REPOS)
    t0 = time.time()
    for k in keys:
        repo, dest, gb = REPOS[k]
        os.makedirs(dest, exist_ok=True)
        print(f"\n=== {repo} -> {dest}/  (~{gb} Go) ===", flush=True)
        before = local_size(dest)
        snapshot_download(repo_id=repo, local_dir=dest, max_workers=a.workers,
                          resume_download=True)
        after = local_size(dest)
        print(f"    {human(after)} sur disque (+{human(after - before)} cette fois)", flush=True)

    print(f"\nTermine en {(time.time()-t0)/60:.1f} min.", flush=True)
    for k in keys:
        _, dest, _ = REPOS[k]
        print(f"  {dest:<34} {human(local_size(dest))}")


if __name__ == "__main__":
    sys.exit(main())
