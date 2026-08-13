#!/usr/bin/env python
# ============================================================================================
# LOT DEFENSIF (nuit du 2026-08-03/04) -- file de taches ORDONNEE PAR PRIORITE, chainee sur les
# GPU reellement libres. Machine hostile + un autre agent au travail : pre-check a l'instant du
# lancement, jamais l'index 6, jamais plus de MAX_GPUS a moi, reprise par mp4 existant.
#
# ORDRE (le premier decide le finetune, il passe donc en premier) :
#   1. GATE A   SS2 (meme decor) t12-t16 seed 0 : `noiseanchor` (qui DUMPE l'ancre) puis
#               `wronganchor` (qui la relit) -> contraste memdec20 - wronganchor par cellule.
#               Regle scellee : >= +0.08 sur >= 4/5 => le finetune est autorise.
#   2. ATTRIB   S2 6 cellules : `memdec20s0` / `wronganchor_s0` / `noiseanchor_s0` (convention de
#               fenetre d'oracle1) -> separe "n'importe quelle frame lisse" de "la fenetre
#               decalee" (question ouverte du §11.5).
#   3. ROUTEUR  `memdec20r` sur S2p/S3p/S4p, 24 cellules (symetrie de routage avec la banque).
#   4. SEEDS    3e graine (seed_idx 2) sur S3 (les A* propres) puis S2/S2p, 4 bras (puissance).
#
#   python launch_defensive.py                 # tout, dans l'ordre
#   python launch_defensive.py --only 1        # GATE A seul
# ============================================================================================
import argparse
import os
import subprocess
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
RES = os.path.join(HERE, "results", "defensive")
LOGS = os.path.join(HERE, "logs")
BROKEN = {6}
MIN_FREE_MB = 26000
MAX_GPUS = 4
T12 = "12,13,14,15,16"
T24 = ",".join(str(i) for i in range(12))

# (priorite, nom, scen, arms, triples, seeds, extra) -- l'ordre de la liste EST l'ordre d'exec
TASKS = [
    (1, "gateA_noise", "SS2", "noiseanchor", T12, "0", []),
    (1, "gateA_wrong", "SS2", "wronganchor", T12, "0", []),
    (2, "attrib_s0", "S2", "memdec20s0,wronganchor_s0,noiseanchor_s0", "0,1,2,3,4,5", "0", []),
    (3, "router_S2p", "S2p", "memdec20r", T24, "0,1", []),
    (3, "router_S3p", "S3p", "memdec20r", T24, "0,1", []),
    (3, "router_S4p", "S4p", "memdec20r", T24, "0,1", []),
    (4, "seed2_S3ref", "S3", "memzero", T24, "2", []),
    (4, "seed2_S2", "S2", "memzero,memdec20,framebank,stockoff", T24, "2", []),
    (4, "seed2_S2p", "S2p", "memzero,memdec20,framebank,stockoff", T24, "2", []),
]
# une tache est decoupee en LOTS de `chunk` triples pour que plusieurs GPU y travaillent
CHUNK = {1: 2, 2: 3, 3: 4, 4: 3}


def gpu_free():
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         env=env).stdout
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}


def missing(scen, arms, triples, seeds):
    out = []
    for t in triples:
        for s in seeds:
            for a in arms:
                p = os.path.join(VID, f"{scen}_t{t}_s{s}_{a}.mp4")
                if not (os.path.exists(p) and os.path.getsize(p) > 1_000_000):
                    out.append((t, s, a))
    return out


def main():
    global MAX_GPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="ne lancer que cette priorite")
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--max-gpus", type=int, default=0,
                    help="plafond de cartes A MOI (pause de coordination : 1 pour laisser la "
                         "place au lot offensif)")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    MAX_GPUS = args.max_gpus or MAX_GPUS

    queue = []
    for prio, name, scen, arms, tri, seeds, extra in TASKS:
        if args.only and prio != args.only:
            continue
        T = [int(x) for x in tri.split(",")]
        A = arms.split(",")
        Sd = [int(x) for x in seeds.split(",")]
        miss = missing(scen, A, T, Sd)
        if not miss:
            print(f"[def] {name}: deja complet, saute")
            continue
        todo = sorted({t for t, _, _ in miss})
        c = CHUNK[prio]
        for i in range(0, len(todo), c):
            lot = todo[i:i + c]
            queue.append((prio, name, scen, arms, lot, seeds))
        print(f"[def] {name}: {len(miss)} rollouts manquants sur {len(todo)} triples "
              f"-> {len(range(0, len(todo), c))} lots")
    print(f"[def] {len(queue)} lots en file, dans l'ordre de priorite", flush=True)

    running, tries = {}, {}
    while queue or running:
        for pid in list(running):
            proc, g, item = running[pid]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                print(f"[def] {item[1]} {item[4]} (GPU {g}) rc={proc.returncode}"
                      f"{'' if ok else '  <-- ECHEC'}", flush=True)
                if not ok and tries.get(str(item), 0) < 3:
                    queue.insert(0, item)
                del running[pid]
        # les lots de PRIORITE 1 ne se lancent jamais en parallele d'une priorite superieure :
        # la file est deja triee, on prend simplement le premier lot disponible.
        if queue and len(running) < MAX_GPUS:
            busy = {g for _, g, _ in running.values()}
            free = gpu_free()
            # GATE A : la vague `wronganchor` ne peut pas demarrer avant que TOUS les dumps
            # `noiseanchor` de son tier existent -> on ne lance pas un lot dont la source manque.
            for g, mb in sorted(free.items(), key=lambda kv: -kv[1]):
                if not queue or len(running) >= MAX_GPUS:
                    break
                if g in BROKEN or g in busy or mb < MIN_FREE_MB:
                    continue
                item = queue[0]
                prio, name, scen, arms, lot, seeds = item
                if "wronganchor" in arms:
                    sys.path.insert(0, HERE)
                    import scenarios as S
                    need = [S.wrong_anchor_src(scen, t, int(s))
                            for t in lot for s in seeds.split(",")]
                    if not all(os.path.exists(n) for n in need):
                        print(f"[def] {name} {lot}: dumps sources pas encore la, on attend",
                              flush=True)
                        break
                queue.pop(0)
                tries[str(item)] = tries.get(str(item), 0) + 1
                tri = ",".join(str(t) for t in lot)
                log = os.path.join(LOGS, f"def_{name}_g{g}_{tri.replace(',', '-')}.log")
                cmd = [PY, os.path.join(HERE, "run_reid_scen.py"), "--mode", "run",
                       "--scen", scen, "--triples", tri, "--seeds", seeds,
                       "--arms", arms, "--res-dir", RES]
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                     env=env, cwd=REPO)
                running[p.pid] = (p, g, item)
                busy.add(g)
                print(f"[def] P{prio} {name} lot {lot} -> GPU {g} ({mb} MiB), pid {p.pid}",
                      flush=True)
        time.sleep(args.poll)
    print("[def] FILE VIDE -- lot defensif termine", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
