#!/usr/bin/env python
# ============================================================================================
# PLANCHE D'OEIL DE L'AFFUTAGE (mini-batterie training-free, 2026-08-04) -- A REGARDER AVANT
# TOUT ROLLOUT. C'est le gate pre-enregistre du choix de sigma :
#   "on retient le sigma le plus BAS qui rend l'ancre visiblement nette en gardant le sujet
#    reconnaissable ; si tous les sigma remplacent le sujet par une re-generation du prompt, la
#    batterie est RE-CATEGORISEE (elle ne teste plus E1) et un seul sigma est execute."
#
# Une ligne par cellule, colonnes :
#   [A* reel] la vraie tete du sujet (frame du rollout ou A OUVRE) = la cible de reconnaissance
#   [brute]   l'ancre DECODEE de l'etat par le decodeur A2, telle qu'elle est splicee aujourd'hui
#   [sigma..] la meme ancre affutee par le prior du modele (SDEdit latent, mode 1 frame, prompt
#             de RETOUR verbatim). sigma = 1.0 est le plancher degenere : plus une trace d'ancre,
#             c'est une generation pure par le prompt -- il est sur la planche EXPRES, comme
#             temoin de "a quoi ressemble le contenu detruit".
#
#   python make_sdedit_board.py
# ============================================================================================
import argparse
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
DEC = "/shared/ssd_14T_project/adenelmi/reid_scen/dec_anchors"
OUT = os.path.join(HERE, "eyepack_tfree")
CELLS = [("SS2", 12, "cafe"), ("SS2", 13, "pelouse"), ("SS2", 14, "boulangerie"),
         ("S2", 2, "chat tabby"), ("S2", 4, "cheval blanc"), ("S2", 11, "theiere cuivre")]
CLEAN_REF = {"SS2": "SS1", "S2": "S3"}
SIGMAS = [0.4, 0.5, 0.6, 0.833, 1.0]


def label_strip(w, h, text):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (18, 18, 18))
    d = ImageDraw.Draw(im)
    d.text((6, max(0, h // 2 - 6)), text, fill=(235, 235, 235))
    return im


def main():
    import torch
    from omegaconf import OmegaConf
    from torchvision.io import read_video
    from PIL import Image
    from run_reid_probe import build_pipeline, MEM2_DEC_DEFAULT
    import scenarios as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    sigmas = [float(x) for x in args.sigmas.split(",") if x]

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(REPO, "reference/runs/reid_probe_l1/probe_l1.yaml"))
    cfg.distributed = False
    pipe = build_pipeline(cfg, device)
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipe.kv_cache1 = None
    from wan.modules.video_memory import TTTBlockCache
    pipe.ttt_cache = [TTTBlockCache() for _ in range(pipe.num_transformer_blocks)]
    os.environ["MEM_DECODE"] = MEM2_DEC_DEFAULT          # requis par l'assert du hook
    os.environ["MEM_SDEDIT_SEED"] = "5150241"
    os.makedirs(args.out, exist_ok=True)

    def to_img(px):
        """(B,1,3,H,W) in [-1,1] -> PIL"""
        a = ((px[0, 0].float().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1) * 255)
        return Image.fromarray(a.to(torch.uint8).cpu().numpy())

    def decode(lat):
        px = pipe.vae.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
        pipe.vae.model.clear_cache()
        return to_img(px)

    rows, meta = [], {}
    for scen, ti, name in CELLS:
        fr = S.revisit_start_frames(scen)[0]
        pt = os.path.join(DEC, f"{scen}_t{ti}_s0_decanchor.f{fr}.pt")
        assert os.path.exists(pt), f"ancre decodee manquante: {pt} (vague `dump` non faite ?)"
        a = torch.load(pt, map_location="cpu", weights_only=False)
        a = (a["anchor"] if isinstance(a, dict) and "anchor" in a else a).to(torch.bfloat16)

        prompts, sched, _ = S.schedule_for(scen, ti)
        ret_prompt = prompts[sched[S.revisit_segments(scen)[0]][1]]
        cond = pipe.text_encoder(text_prompts=[ret_prompt])

        # A* reel : une frame pixel du milieu du segment 0 du rollout ou A OUVRE
        ref_scen = CLEAN_REF[scen]
        rp = f"{VID}/{ref_scen}_t{ti}_s0_memzero.mp4"
        lo, hi = S.score_segments(ref_scen, S.SCENARIOS[ref_scen]["frames"])[0]
        v, _, _ = read_video(rp, pts_unit="sec", output_format="THWC")
        k = min(4 * ((lo + hi) // 2) - 3, v.shape[0] - 1)
        astar = Image.fromarray(v[k].numpy())

        raw_img = decode(a)
        raw_img.save(os.path.join(args.out, f"anchor_raw_{scen}_t{ti}.png"))
        imgs = [("A* reel", astar), ("ancre BRUTE", decode(a))]
        m = {"prompt_retour": ret_prompt, "hf_brute": pipe._lat_hf(a), "sigmas": {}}
        for s in sigmas:
            os.environ["MEM_SDEDIT"] = str(s)
            o = pipe._maybe_sdedit(a.clone(), cond, fr)
            cos = float(torch.nn.functional.cosine_similarity(
                o.float().flatten()[None], a.float().flatten()[None]).item())
            m["sigmas"][str(s)] = {"cos_vs_brute": cos, "hf": pipe._lat_hf(o)}
            im = decode(o)
            im.save(os.path.join(args.out, f"anchor_s{s}_{scen}_t{ti}.png"))
            imgs.append((f"sigma {s}", im))
        os.environ.pop("MEM_SDEDIT", None)
        meta[f"{scen}_t{ti}"] = m
        rows.append((f"{scen} t{ti} {name}", imgs))
        print(f"[board] {scen} t{ti} ({name}) fait | HF brute {m['hf_brute']:.3f} -> " +
              " ".join(f"s{s}:{m['sigmas'][str(s)]['hf']:.2f}(cos {m['sigmas'][str(s)]['cos_vs_brute']:.2f})"
                       for s in sigmas), flush=True)

    # ---- assemblage -------------------------------------------------------------------------
    W, H = rows[0][1][0][1].size
    pad, hdr, lab = 6, 22, 150
    ncol = len(rows[0][1])
    board = Image.new("RGB", (lab + ncol * (W + pad) + pad, hdr + len(rows) * (H + pad) + pad),
                      (12, 12, 12))
    for j, (cap, _) in enumerate(rows[0][1]):
        board.paste(label_strip(W, hdr, cap), (lab + j * (W + pad), 0))
    for i, (rowname, imgs) in enumerate(rows):
        y = hdr + i * (H + pad)
        board.paste(label_strip(lab - pad, H, rowname), (0, y))
        for j, (_, im) in enumerate(imgs):
            board.paste(im, (lab + j * (W + pad), y))
    p = os.path.join(args.out, "planche_sdedit_sigmas.jpg")
    board.save(p, quality=88)
    with open(os.path.join(args.out, "sdedit_board_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[board] {p}  ({os.path.getsize(p) / 1e6:.2f} Mo, {board.size[0]}x{board.size[1]})")


if __name__ == "__main__":
    main()
