#!/usr/bin/env python
# ============================================================================================
# SUPERVISEUR de la MINI-BATTERIE TRAINING-FREE (2026-08-04), machine PARTAGEE.
#
# Regles ops (memoire projet), appliquees telles quelles :
#   * pre-check d'occupation DANS la commande de lancement, a l'instant du lancement, et
#     RE-VERIFIE avant chaque nouveau lot : un GPU libre il y a 30 s n'est pas un GPU libre ;
#   * marge > 25 Go (MIN_FREE_MB), jamais l'index 6 (casse), jamais plus de 4 GPU a la fois ;
#   * reprise par mp4 existant, mp4 tronques (< 1 Mo) supprimes avant de commencer ;
#   * un lot qui echoue est re-tente 3 fois (OOM d'un co-run hostile), puis abandonne.
#
# Une VAGUE = un bras sur une liste de (scenario, triples). Les vagues sont donnees en ligne de
# commande et executees DANS L'ORDRE (les dependances sont reelles : sdedit_wrong* et wronganchor
# ont besoin des dumps d'ancre decodee de la vague `dump`).
#
#   python launch_tfree.py --waves dump
#   python launch_tfree.py --waves sinkpurge,sdedit_ours04
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
VID_REPRO = "/shared/ssd_14T_project/adenelmi/reid_scen/videos_tfree_repro"
RES = os.path.join(HERE, "results", "tfree")
LOGS = os.path.join(HERE, "logs")
BROKEN = {6}
MIN_FREE_MB = 30000          # 2026-08-04 : un AUTRE agent du projet occupe les 8 GPU
                             # (campagne defensive memdec20r, ~23 Go/GPU). Pic mesure de notre
                             # rollout : 23.1 Go (OOM au decode VAE final). 24 Go = le minimum
                             # qui tient ; les 8 re-essais absorbent les collisions.
                             # defensive memdec20r). A 27 Go, un co-run qui arrive APRES le
                             # pre-check nous fait OOM au decode VAE final du rollout.
MAX_GPUS = 2                 # on laisse de l'air a l'autre agent

# les 6 cellules de la batterie : 3 du tier MEME DECOR (le seul ou le score ne peut venir que du
# sujet) + 3 du tier cross-scene des controles du §11 (donc directement comparables a wronganchor)
CELLS = [("SS2", [12, 13, 14]), ("S2", [2, 4, 11])]
# la vague `dump` produit les ancres decodees qui MANQUENT sur SS2 (celles de S2 sont sur disque) ;
# t15 est inclus parce que wronganchor/sdedit_wrong de t14 lit l'ancre de t15 (permutation +1).
DUMP = [("SS2", [12, 13, 14, 15])]


def gpu_free():
    """(index -> MiB libres) LU MAINTENANT."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         env=env).stdout
    return {int(i): int(f) for i, f in
            (tuple(x.strip() for x in line.split(",")) for line in out.strip().splitlines())}


def jobs_of(wave):
    """[(scen, triples, arm, vid_dir, extra_args)] pour une vague."""
    if wave == "dump":
        # memdec20 re-genere dans un repertoire SCRATCH (les mp4 officiels sont intacts) avec
        # --dump-decanchor : effet de bord pur, aucun tirage RNG. Le mp4 regenere sert ensuite de
        # CONTROLE BIT A BIT du legacy (compare a celui deja sur disque).
        return [(sc, ts, "memdec20", VID_REPRO, ["--dump-decanchor"]) for sc, ts in DUMP]
    return [(sc, ts, wave, VID, []) for sc, ts in CELLS]


def clean_truncated(vid_dir, arm):
    for p in glob.glob(os.path.join(vid_dir, f"*_{arm}.mp4")):
        if os.path.getsize(p) < 1_000_000:
            print(f"[sup] mp4 tronque supprime: {p} ({os.path.getsize(p)} o)")
            os.remove(p)


def todo_triples(scen, triples, arm, vid_dir, seed=0):
    out = []
    for t in triples:
        p = os.path.join(vid_dir, f"{scen}_t{t}_s{seed}_{arm}.mp4")
        if not (os.path.exists(p) and os.path.getsize(p) > 1_000_000):
            out.append(t)
    return out


def run_wave(wave, chunk, poll, seeds):
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    queue = []
    for scen, triples, arm, vid_dir, extra in jobs_of(wave):
        os.makedirs(vid_dir, exist_ok=True)
        clean_truncated(vid_dir, arm)
        left = todo_triples(scen, triples, arm, vid_dir)
        for i in range(0, len(left), chunk):
            queue.append((scen, left[i:i + chunk], arm, vid_dir, extra))
    print(f"\n===== VAGUE {wave}: {sum(len(q[1]) for q in queue)} cellules, {len(queue)} lots "
          f"=====", flush=True)
    running, tries = {}, {}
    while queue or running:
        for pid in list(running):
            proc, g, lot = running[pid]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                print(f"[sup] lot {lot[0]} {lot[1]} ({lot[2]}) sur GPU {g} termine "
                      f"(rc={proc.returncode})" + ("" if ok else "  <-- ECHEC, voir le log"),
                      flush=True)
                key = (lot[0], tuple(lot[1]), lot[2])
                if not ok and tries.get(key, 0) < 8:
                    queue.append(lot)
                elif not ok:
                    print(f"[sup] lot {key} ABANDONNE apres 8 essais", flush=True)
                try:
                    os.remove(os.path.join(LOGS, f".gpu{g}.lock"))
                except OSError:
                    pass
                del running[pid]
        if queue and len(running) < MAX_GPUS:
            busy = {g for _, g, _ in running.values()}
            for g, mb in sorted(gpu_free().items(), key=lambda kv: -kv[1]):
                if not queue or len(running) >= MAX_GPUS:
                    break
                if g in BROKEN or g in busy or mb < MIN_FREE_MB:
                    continue
                # ANTI-COURSE (lecon du 2026-08-04) : deux superviseurs lances en parallele ont
                # lu "48 Go libres" sur le MEME GPU a 10 s d'intervalle et s'y sont ecrases (OOM).
                # Un verrou fichier par GPU rend la prise ATOMIQUE entre superviseurs. La regle
                # "un seul superviseur, vagues chainees" reste la bonne pratique ; ceci est le
                # filet.
                lock = os.path.join(LOGS, f".gpu{g}.lock")
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                except FileExistsError:
                    try:                       # verrou orphelin (superviseur mort) -> on nettoie
                        if not os.path.exists(f"/proc/{open(lock).read().strip()}"):
                            os.remove(lock)
                    except Exception:          # noqa: BLE001
                        pass
                    continue
                lot = queue.pop(0)
                scen, tri, arm, vid_dir, extra = lot
                key = (scen, tuple(tri), arm)
                tries[key] = tries.get(key, 0) + 1
                tag = f"{arm}_{scen}_{'-'.join(str(t) for t in tri)}"
                log = os.path.join(LOGS, f"tfree_{tag}_g{g}.log")
                cmd = [PY, os.path.join(HERE, "run_reid_scen.py"), "--mode", "run",
                       "--scen", scen, "--triples", ",".join(str(t) for t in tri),
                       "--seeds", ",".join(str(s) for s in seeds), "--arms", arm,
                       "--vid-dir", vid_dir, "--res-dir", RES] + extra
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                fh = open(log, "w")
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
                running[p.pid] = (p, g, lot)
                busy.add(g)
                print(f"[sup] GPU {g} libre={mb} MiB -> {scen} t{tri} ({arm}), pid {p.pid}, "
                      f"log {os.path.basename(log)}", flush=True)
        if queue and not running:
            print(f"[sup] aucun GPU libre (>= {MIN_FREE_MB} MiB) pour {len(queue)} lots, "
                  f"nouvelle tentative dans {poll}s | libre = {gpu_free()}", flush=True)
        time.sleep(poll)
    left = {}
    for scen, triples, arm, vid_dir, _ in jobs_of(wave):
        m = todo_triples(scen, triples, arm, vid_dir)
        if m:
            left[f"{scen}/{arm}"] = m
    print(f"===== VAGUE {wave} TERMINEE, manquantes: {left or 'aucune'} =====", flush=True)
    return not left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", required=True, help="liste ordonnee, ex: dump,sinkpurge")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    for w in args.waves.split(","):
        if w:
            run_wave(w, args.chunk, args.poll, seeds)
    print("[sup] FINI", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
