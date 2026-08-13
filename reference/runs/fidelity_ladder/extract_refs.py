#!/usr/bin/env python
# ============================================================================================
# ECHELLE DE FIDELITE -- extraction des references depuis les shards v2 (CPU, aucun GPU).
#
# Pour chacun des 10 scenarios :
#   twin_<sid>.pt    frame latente 80 du JUMEAU v2 (meme prompt, graine +100000)  -> barreau
#                    `meme_categorie` (ORACLE_ANCHOR_SRC)
#   other_<sid>.pt   frame latente 80 de la video v2 PRINCIPALE du scenario donneur
#                    ((i+5) % 10, autre categorie)                                -> barreau
#                    `autre_categorie`
#   main81_<sid>.pt  latents fp16 stockes [0:81] de la video principale           -> verif
#                    bit a bit du segment A par run_fidelity
# + refs/manifest.json : graines LUES dans l'index (jamais recalculees), chemins, donneurs.
#
# Les shards sont ouverts en mmap (mem2_data.load_part) : seules les tranches touchees
# montent en RAM. Shapes des .pt d'ancre : [1, 1, 16, 60, 104] (la shape que
# _anchor_content assert contre `real`).
# ============================================================================================
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C                                              # noqa: E402

sys.path.insert(0, os.path.join(C.REPO, "reference/runs/mem2"))
import torch                                                             # noqa: E402
from mem2_data import load_part                                          # noqa: E402


def main():
    with open(C.V2_INDEX) as f:
        index = json.load(f)
    by_vid = {v["video_id"]: v for v in index}

    need = {}       # video_id -> meta, pour les 10 mains + 10 twins
    for sc in C.SCENARIOS:
        for pfx in ("main", "twin"):
            vid = f"{pfx}:{sc['cidx']}"
            assert vid in by_vid, f"{vid} absent de l'index v2"
            need[vid] = by_vid[vid]

    # verification des graines (formule de collect_pairs, verifiee et non recalculee a l'aveugle)
    for sc in C.SCENARIOS:
        m = need[f"main:{sc['cidx']}"]
        t = need[f"twin:{sc['cidx']}"]
        assert m["prompt"] == sc["prompt"], \
            f"{sc['sid']}: prompt de la table != index ('{m['prompt']}' vs '{sc['prompt']}')"
        assert m["prompt"] == t["prompt"], f"{sc['sid']}: prompts main/twin differents"
        assert int(m["seed"]) == 42 + 500000 + sc["cidx"], \
            f"{sc['sid']}: graine main {m['seed']} != formule 42+500000+cidx"
        assert int(t["seed"]) == int(m["seed"]) + 100000, \
            f"{sc['sid']}: graine twin {t['seed']} != main+100000"
        assert m["mode"] == "single" and int(m["frames"]) == C.V2_FRAMES

    # groupement par part pour ne charger chaque shard qu'une fois
    by_part = {}
    for vid, meta in need.items():
        by_part.setdefault(meta["part"], []).append((vid, meta))

    frames = {}     # video_id -> {"f80": T(16,60,104) fp16, "lat81": T(81,16,60,104) fp16}
    for part, entries in sorted(by_part.items()):
        print(f"[extract] {os.path.basename(part)} : {len(entries)} videos", flush=True)
        obj = load_part(part)
        for vid, meta in entries:
            rec = obj["videos"][meta["vi"]]
            assert int(rec["corpus_idx"]) == int(meta["corpus_idx"]), \
                f"{vid}: corpus_idx du record != index"
            lat = rec["latents"]
            assert lat.shape == (C.V2_FRAMES, 16, 60, 104) and lat.dtype == torch.float16, \
                f"{vid}: latents {tuple(lat.shape)} {lat.dtype}"
            frames[vid] = {"f80": lat[C.ANCHOR_FRAME].clone(),
                           "lat81": lat[:C.SW1].clone()}
        del obj

    os.makedirs(C.REFS_DIR, exist_ok=True)
    manifest = {}
    for i, sc in enumerate(C.SCENARIOS):
        sid = sc["sid"]
        dn = C.SCENARIOS[C.donor_of(i)]
        m = need[f"main:{sc['cidx']}"]
        t = need[f"twin:{sc['cidx']}"]
        twin = frames[f"twin:{sc['cidx']}"]["f80"][None, None]          # [1,1,16,60,104]
        other = frames[f"main:{dn['cidx']}"]["f80"][None, None]
        torch.save({"anchor": twin, "src_video": f"twin:{sc['cidx']}",
                    "frame": C.ANCHOR_FRAME}, C.twin_pt(sid))
        torch.save({"anchor": other, "src_video": f"main:{dn['cidx']}",
                    "frame": C.ANCHOR_FRAME}, C.other_pt(sid))
        torch.save({"latents": frames[f"main:{sc['cidx']}"]["lat81"],
                    "src_video": f"main:{sc['cidx']}"}, C.main81_pt(sid))
        manifest[sid] = {
            "board_no": sc["board_no"], "cidx": sc["cidx"], "category": sc["category"],
            "prompt": sc["prompt"], "interlude": sc["interlude"],
            "seed_main": int(m["seed"]), "seed_twin": int(t["seed"]),
            "donor_sid": dn["sid"], "donor_cidx": dn["cidx"], "donor_category": dn["category"],
            "twin_pt": C.twin_pt(sid), "other_pt": C.other_pt(sid),
            "main81_pt": C.main81_pt(sid), "anchor_frame": C.ANCHOR_FRAME,
        }
        print(f"[extract] {sid} board{sc['board_no']:02d} cidx {sc['cidx']} "
              f"seed {m['seed']} twin {t['seed']} donor {dn['sid']} ({dn['category']})",
              flush=True)

    tmp = C.REFS_MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, C.REFS_MANIFEST)
    print(f"[extract] manifest -> {C.REFS_MANIFEST}")


if __name__ == "__main__":
    main()
