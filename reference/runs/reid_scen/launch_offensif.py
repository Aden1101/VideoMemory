#!/usr/bin/env python
# ============================================================================================
# SUPERVISEUR du LOT OFFENSIF (2026-08-04), machine PARTAGEE + le lot DEFENSIF en cours.
#
# Regles ops (memoire projet), appliquees telles quelles :
#   * pre-check d'occupation A L'INSTANT du lancement, re-verifie avant chaque lot ;
#   * marge > 25 Go (MIN_FREE_MB), JAMAIS l'index 6 (casse), JAMAIS plus de 2 GPU a moi
#     (le lot defensif prend le reste : la coordination se fait par les pre-checks). MESURE
#     PRISE EN COURS DE NUIT : la machine s'est retrouvee a 6 rollouts sur 3 GPU (deux autres
#     superviseurs qui se doublent), tout le monde OOM au pic. On redescend a UN GPU et a une
#     barre de 30 Go = "un GPU vraiment vide", pour ne pas ajouter au thrash et pour que NOS
#     rollouts (le canal additif alloue ~1 Go de plus) ne tombent pas. Barre finale 26 Go =
#     celle du lot defensif (une barre plus haute que la sienne = je ne gagne JAMAIS un GPU),
#     MAX_GPUS = 1 (je ne detiens jamais plus d'un tiers des 3 GPU reellement a nous), un
#     SECOND pre-check jitte, et un critere d'EXCLUSIVITE : on ne prend pas un GPU qui porte
#     deja un GROS voisin (>= BIG_PROC_MB), car deux rollouts 1.3B ne tiennent pas sur 48 Go ;
#     SECOND pre-check jitte evite que les deux superviseurs sautent sur le meme GPU ;
#   * reprise par mp4 existant, mp4 tronques (< 1 Mo) supprimes avant de commencer ;
#   * un lot qui echoue est re-tente 3 fois (OOM d'un co-run hostile), puis abandonne.
#
# L'ORDRE DES VAGUES est un ordre de VALEUR DECROISSANTE PAR GPU-HEURE : chaque levier rend un
# verdict lisible meme si la nuit est coupee en deux.
#   A1  levier 3, la borne du canal additif a Delta = 1 (dose forcee)      5 rollouts
#   C0  levier 1, le temoin BIT A BIT (suffixe vide == memzero sur disque) 1 rollout
#   B1  levier 2, l'ancre declaree a t* = 500, 6 cellules x 2 bras        12 rollouts
#   C1  levier 1, les 4 bras de legende sur les 5 cellules S2             20 rollouts
#   A2  levier 3, le meme canal a gain 1 (courbe de dose a Delta = 1)      4 rollouts
#   C2  levier 1, les 4 bras de legende sur les 3 cellules SS2            12 rollouts
#   B2  levier 2, t* = 250 (le second point de la courbe)                 12 rollouts
#
#   python launch_offensif.py                      # tout, dans l'ordre
#   python launch_offensif.py --waves A1,C0        # une partie
# ============================================================================================
import argparse
import glob
import os
import random
import subprocess
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
RES = os.path.join(HERE, "results", "offensif")
LOGS = os.path.join(HERE, "logs")
BROKEN = {6}
MIN_FREE_MB = 26000
MAX_GPUS = 1        # surcharge par --max-gpus ; un seul superviseur, jamais deux
# un voisin est "gros" (= un autre rollout, ou le reserveur d'un autre utilisateur) au-dela de
# ce seuil ; en dessous c'est un job de scoring/VAE, avec lequel on peut co-habiter sans risque.
BIG_PROC_MB = 20000

I1_CELLS = [("SS2", [12, 13, 14]), ("S2", [2, 4, 11])]
ADD_CELLS = [("SS2", [12, 13, 14, 15])]
CAP_S2 = [("S2c", [0, 2, 3, 4, 11])]
CAP_SS2 = [("SS2c", [12, 13, 14])]

# vague -> [(scen, triples, arm)]
WAVES = {
    "A1": [(sc, ts, a) for sc, ts in ADD_CELLS for a in ("oracle1add8p17",)]
          + [("SS2", [12], "oracle1add8p9")],
    "C0": [("S2c", [2], "caption_none")],
    "B1": [(sc, ts, a) for sc, ts in I1_CELLS for a in ("i1_ours500", "i1_wrong500")],
    "C1": [(sc, ts, a) for sc, ts in CAP_S2
           for a in ("caption_ours", "caption_wrong", "padded", "framebank_vlm")],
    "A2": [(sc, ts, a) for sc, ts in ADD_CELLS for a in ("oracle1add1p17",)],
    "C2": [(sc, ts, a) for sc, ts in CAP_SS2
           for a in ("caption_ours", "caption_wrong", "padded", "framebank_vlm")],
    "B2": [(sc, ts, a) for sc, ts in I1_CELLS for a in ("i1_ours250", "i1_wrong250")],
    # conditionnelle : NOTRE ancre dans le canal additif, seulement si la borne s'ouvre (A1)
    "A3": [(sc, ts, a) for sc, ts in ADD_CELLS for a in ("memdec20add8p17",)],
    # LEVIER 4 (le plus haut rendement apres T0) : decoder le SNAPSHOT de fin de segment A.
    # D0 = pilote 2 cellules (t3 = homme en veste bleue / flamants roses, la plus diagnostique
    # de la planche ; t2 = chat roux / camion rouge), qui porte le PRE-ECRAN scelle.
    "D0": [("S2", [2, 3], "memdec_snap")],
    "D1": [("S2", [0, 4, 6, 11], "memdec_snap")],
}
# le levier 4 passe devant 1/2/3 : il repare la CHARGE UTILE, les trois autres le CANAL, et T0
# a montre que le canal n'est pas le goulot.
ORDER = ["A1", "D0", "D1", "C0", "B1", "C1", "A2", "C2", "B2"]


def gpu_free():
    """(index -> MiB libres) LU MAINTENANT."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         env=env).stdout
    return {int(i): int(f) for i, f in
            (tuple(x.strip() for x in line.split(",")) for line in out.strip().splitlines())}


def gpu_busy_pids():
    """index -> nombre de processus de calcul EN COURS. Lecon ops de la nuit : un rollout 1.3B
    de cette campagne monte a ~24 Go et la carte en fait 48 ; DEUX rollouts ne tiennent donc PAS
    (ils s'ecroulent au pic, ce qui a fait tomber 4 lots offensifs ET plusieurs lots defensifs).
    Un pre-check en MEMOIRE LIBRE ne suffit pas : un voisin lance il y a 30 s n'a pas encore
    alloue son pic. Le seul critere stable est l'EXCLUSIVITE : on ne prend qu'un GPU VIDE."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    uuid2idx = {}
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, env=env).stdout
    for line in out.strip().splitlines():
        i, u = [x.strip() for x in line.split(",")]
        uuid2idx[u] = int(i)
    n = {i: 0 for i in uuid2idx.values()}
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         env=env).stdout
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] in uuid2idx and int(parts[1]) >= BIG_PROC_MB:
            n[uuid2idx[parts[0]]] += 1        # on ne compte que les GROS voisins
    return n


def clean_truncated(arm):
    for p in glob.glob(os.path.join(VID, f"*_{arm}.mp4")):
        if os.path.getsize(p) < 1_000_000:
            print(f"[off] mp4 tronque supprime: {p} ({os.path.getsize(p)} o)")
            os.remove(p)


def todo(scen, triples, arm, seed=0):
    return [t for t in triples
            if not (os.path.exists(os.path.join(VID, f"{scen}_t{t}_s{seed}_{arm}.mp4"))
                    and os.path.getsize(os.path.join(VID, f"{scen}_t{t}_s{seed}_{arm}.mp4"))
                    > 1_000_000)]


def run_wave(wave, chunk, poll, seeds):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    queue = []
    for scen, triples, arm in WAVES[wave]:
        clean_truncated(arm)
        left = todo(scen, triples, arm)
        for i in range(0, len(left), chunk):
            queue.append((scen, left[i:i + chunk], arm))
    print(f"\n===== VAGUE {wave}: {sum(len(q[1]) for q in queue)} rollouts, {len(queue)} lots "
          f"=====", flush=True)
    running, tries = {}, {}
    while queue or running:
        for pid in list(running):
            proc, g, lot = running[pid]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                print(f"[off] lot {lot[0]} {lot[1]} ({lot[2]}) sur GPU {g} termine "
                      f"(rc={proc.returncode})" + ("" if ok else "  <-- ECHEC, voir le log"),
                      flush=True)
                key = (lot[0], tuple(lot[1]), lot[2])
                if not ok and tries.get(key, 0) < 3:
                    queue.append(lot)
                elif not ok:
                    print(f"[off] lot {key} ABANDONNE apres 3 essais", flush=True)
                del running[pid]
        if queue and len(running) < MAX_GPUS:
            busy = {g for _, g, _ in running.values()}
            nproc = gpu_busy_pids()
            for g, mb in sorted(gpu_free().items(), key=lambda kv: -kv[1]):
                if not queue or len(running) >= MAX_GPUS:
                    break
                if g in BROKEN or g in busy or mb < MIN_FREE_MB or nproc.get(g, 1) > 0:
                    continue
                # DEUXIEME pre-check EXCLUSIF, apres une attente aleatoire : le lot defensif
                # tourne en parallele et lit les memes GPU. Un voisin lance il y a 30 s n'a pas
                # encore alloue son pic, donc la memoire libre ment ; on exige un GPU sans AUCUN
                # processus de calcul, deux fois de suite.
                time.sleep(5 + 15 * random.random())
                if gpu_free().get(g, 0) < MIN_FREE_MB or gpu_busy_pids().get(g, 1) > 0:
                    print(f"[off] GPU {g} pris entre les deux pre-checks, on passe", flush=True)
                    continue
                lot = queue.pop(0)
                scen, tri, arm = lot
                tri = todo(scen, tri, arm)         # un orphan a pu finir le lot entre-temps
                if not tri:
                    print(f"[off] lot {scen} {lot[1]} ({arm}) deja sur disque, saute", flush=True)
                    continue
                lot = (scen, tri, arm)
                key = (scen, tuple(tri), arm)
                tries[key] = tries.get(key, 0) + 1
                tag = f"{arm}_{scen}_{'-'.join(str(t) for t in tri)}"
                log = os.path.join(LOGS, f"off_{tag}_g{g}.log")
                cmd = [PY, os.path.join(HERE, "run_reid_scen.py"), "--mode", "run",
                       "--scen", scen, "--triples", ",".join(str(t) for t in tri),
                       "--seeds", ",".join(str(s) for s in seeds), "--arms", arm,
                       "--vid-dir", VID, "--res-dir", RES]
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                fh = open(log, "w")
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
                running[p.pid] = (p, g, lot)
                busy.add(g)
                print(f"[off] GPU {g} libre={mb} MiB -> {scen} t{tri} ({arm}), pid {p.pid}, "
                      f"log {os.path.basename(log)}", flush=True)
        if queue and not running:
            print(f"[off] aucun GPU libre (>= {MIN_FREE_MB} MiB) pour {len(queue)} lots, "
                  f"nouvelle tentative dans {poll}s | libre = {gpu_free()} | "
                  f"processus = {gpu_busy_pids()}", flush=True)
        time.sleep(poll)
    left = {f"{sc}/{a}": todo(sc, ts, a) for sc, ts, a in WAVES[wave] if todo(sc, ts, a)}
    print(f"===== VAGUE {wave} TERMINEE, manquantes: {left or 'aucune'} =====", flush=True)
    return not left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", default=",".join(ORDER))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-gpus", type=int, default=1,
                    help="jamais plus de 2 (budget du lot), et UN SEUL superviseur a la fois")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    global MAX_GPUS
    assert 1 <= args.max_gpus <= 2, "budget du lot offensif : 2 GPU au maximum"
    MAX_GPUS = args.max_gpus
    for w in args.waves.split(","):
        if w:
            assert w in WAVES, f"vague inconnue {w} (connues: {list(WAVES)})"
            run_wave(w, args.chunk, args.poll, seeds)
    print("[off] FINI", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
