#!/usr/bin/env python
# ============================================================================================
# SUPERVISEUR des 2 CONTROLES REVIEWER S2 (noiseanchor puis wronganchor), machine HOSTILE.
#
# Regles ops (memoire projet) :
#   * pre-check d'occupation DANS la commande de lancement, a l'instant du lancement, et
#     RE-VERIFIE avant chaque nouvelle vague : un GPU libre il y a 30 s n'est pas un GPU libre ;
#   * jamais l'index 6 (casse), jamais plus de 4 GPU a moi en meme temps ;
#   * un autre agent tourne en parallele sur cette machine : on ne prend QUE ce qui est
#     reellement libre a l'instant t, et on ne compte jamais sur un GPU qu'on n'a pas obtenu ;
#   * reprise par mp4 existant (run_reid_scen saute une cellule deja rendue), les mp4 tronques
#     (< 1 Mo) sont supprimes avant de commencer.
#
# ORDRE DES PHASES (structurel, pas un choix) : la vague `noiseanchor` PERSISTE l'ancre qu'elle
# decode avant de la jeter (MEM_DECODE_DUMP), et c'est cette ancre que le bras `wronganchor` du
# triple precedent splice. Phase 1 complete => phase 2 possible, 48 rollouts au lieu de 72.
#
#   python launch_controls.py                 # les 2 phases, 12 triples x 2 seeds
#   python launch_controls.py --phase 2       # reprise
# ============================================================================================
import argparse
import glob
import os
import subprocess
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python"
VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
RES = os.path.join(HERE, "results", "controls")
LOGS = os.path.join(HERE, "logs")
BROKEN = {6}
MIN_FREE_MB = 27000          # marge > 25 Go demandee, +2 Go de securite (co-run hostile)
MAX_GPUS = 4


def gpu_free():
    """(index -> MiB libres) LU MAINTENANT."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         env=env).stdout
    d = {}
    for line in out.strip().splitlines():
        i, f = (x.strip() for x in line.split(","))
        d[int(i)] = int(f)
    return d


def clean_truncated(scen, arms):
    n = 0
    for a in arms:
        for p in glob.glob(os.path.join(VID, f"{scen}_t*_s*_{a}.mp4")):
            if os.path.getsize(p) < 1_000_000:
                print(f"[sup] mp4 tronque supprime: {p} ({os.path.getsize(p)} o)")
                os.remove(p)
                n += 1
    return n


def done_cells(scen, arm, triples, seeds):
    return {(t, s) for t in triples for s in seeds
            if os.path.exists(os.path.join(VID, f"{scen}_t{t}_s{s}_{arm}.mp4"))
            and os.path.getsize(os.path.join(VID, f"{scen}_t{t}_s{s}_{arm}.mp4")) > 1_000_000}


def run_phase(arm, scen, triples, seeds, chunk, poll):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    clean_truncated(scen, [arm])
    todo = [t for t in triples
            if any((t, s) not in done_cells(scen, arm, triples, seeds) for s in seeds)]
    queue = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    print(f"\n===== PHASE {arm}: {len(todo)} triples a faire, {len(queue)} lots de {chunk} "
          f"=====", flush=True)
    running = {}                       # pid -> (proc, gpu, lot)
    tries = {}                         # tuple(lot) -> essais (un bug ne doit pas boucler)
    while queue or running:
        for pid in list(running):
            proc, g, lot = running[pid]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                print(f"[sup] lot {lot} sur GPU {g} termine (rc={proc.returncode})"
                      f"{'' if ok else '  <-- ECHEC, voir le log'}", flush=True)
                if not ok and tries.get(tuple(lot), 0) < 3:
                    queue.append(lot)          # re-tente plus tard (OOM d'un co-run, etc.)
                elif not ok:
                    print(f"[sup] lot {lot} ABANDONNE apres 3 essais", flush=True)
                del running[pid]
        if queue and len(running) < MAX_GPUS:
            busy = {g for _, g, _ in running.values()}
            free = gpu_free()
            for g, mb in sorted(free.items(), key=lambda kv: -kv[1]):
                if not queue or len(running) >= MAX_GPUS:
                    break
                if g in BROKEN or g in busy or mb < MIN_FREE_MB:
                    continue
                lot = queue.pop(0)
                tries[tuple(lot)] = tries.get(tuple(lot), 0) + 1
                tri = ",".join(str(t) for t in lot)
                log = os.path.join(LOGS, f"controls_{arm}_g{g}_{tri.replace(',', '-')}.log")
                cmd = [PY, os.path.join(HERE, "run_reid_scen.py"), "--mode", "run",
                       "--scen", scen, "--triples", tri,
                       "--seeds", ",".join(str(s) for s in seeds),
                       "--arms", arm, "--res-dir", RES]
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                fh = open(log, "w")
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
                running[p.pid] = (p, g, lot)
                busy.add(g)
                print(f"[sup] GPU {g} libre={mb} MiB -> lot {lot} ({arm}), pid {p.pid}, "
                      f"log {os.path.basename(log)}", flush=True)
        if queue and not running:
            print(f"[sup] aucun GPU libre (>= {MIN_FREE_MB} MiB) pour {len(queue)} lots, "
                  f"nouvelle tentative dans {poll}s | libre = {gpu_free()}", flush=True)
        time.sleep(poll)
    left = [t for t in triples
            if any((t, s) not in done_cells(scen, arm, triples, seeds) for s in seeds)]
    print(f"===== PHASE {arm} TERMINEE, cellules manquantes: {left or 'aucune'} =====",
          flush=True)
    return not left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="S2")
    ap.add_argument("--triples", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--phase", type=int, default=0, help="0 = les deux, 1 = noiseanchor, "
                                                         "2 = wronganchor")
    ap.add_argument("--chunk", type=int, default=2, help="triples par lot (2 = 4 cellules)")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    triples = [int(x) for x in args.triples.split(",") if x != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]

    if args.phase in (0, 1):
        ok = run_phase("noiseanchor", args.scen, triples, seeds, args.chunk, args.poll)
        if args.phase == 0 and not ok:
            print("[sup] phase 1 incomplete: la phase 2 a besoin de TOUS les dumps "
                  "(triple (t+1) % 12) -- on lance quand meme ce qui est possible", flush=True)
    if args.phase in (0, 2):
        import scenarios as S                                            # noqa: E402
        sys.path.insert(0, HERE)
        miss = [t for t in triples for s in seeds
                if not os.path.exists(S.wrong_anchor_src(args.scen, t, s))]
        if miss:
            print(f"[sup] dumps manquants pour les triples {sorted(set(miss))} -- ces cellules "
                  f"echoueront (assert explicite), les autres passent", flush=True)
        run_phase("wronganchor", args.scen, triples, seeds, args.chunk, args.poll)
    print("[sup] FINI", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
