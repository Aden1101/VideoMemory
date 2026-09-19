#!/usr/bin/env python
"""Tache n 1, les masques DU SUJET (plan_memoire_2026-08-04, correctif d'auto-audit n 2).

"Les masques DU SUJET n'existent pas encore : segmentation guidee par le nom du sujet du
 prompt (GroundingDINO+SAM) sur les images propres du corpus, une fois, avec planche de
 validation a l'oeil AVANT tout."

Ce que fait ce script, une fois, pour les 165 videos du corpus (2 segments-sujet pour les
switch, 1 pour les single) :
  1. echantillonne ~10 frames latentes du segment (probe1_common.sample_segment_frames) ;
  2. les VAE-decode en pixels propres (chemin identique a build_bank.dino_audit_embed :
     une frame latente decodee SEULE = une frame pixel) ;
  3. detecte le sujet par un detecteur OUVERT-VOCABULAIRE guide par le texte (OWL-ViT
     large-patch14, l'equivalent HF leger de GroundingDINO ; `transformers` est fige a
     4.33.2 dans cet env et n'a pas GroundingDino, on ne met pas a jour une dependance du
     trainer pour une tache annexe). Les requetes viennent du nom du sujet extrait du
     prompt ; le detecteur ARBITRE entre elles (score median sur les frames) ;
  4. segmente au SAM (facebook/sam-vit-huge) avec la boite comme invite ;
  5. ecrit masque + crop-sujet + crop-fond (le complement) + l'embedding DINO des deux ;
  6. rend une PLANCHE de validation a l'oeil (grille image | masque), JPEG < 0.5 MB.

Les modeles sont telecharges sur le DISQUE PROJET (HF_HOME = probe1/hf) : le home est plein.

Usage (pre-check GPU dans la commande, regle de la maison ; jamais l'index 6) :
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
  CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/build_masks.py --all
  ... --limit 4            # fumee
  ... --board_only         # re-rend seulement la planche depuis ce qui est sur disque
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

os.environ.setdefault("HF_HOME", C.HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(C.HF_HOME, "hub"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from PIL import Image  # noqa: E402

import mem2_data as D  # noqa: E402

OWL_MODEL = "google/owlvit-large-patch14"
SAM_MODEL = "facebook/sam-vit-huge"
# (le seuil de detection est 0 : on garde la MEILLEURE boite de chaque requete, pas un jeu)
SWITCH_MARGIN = 1.15          # un candidat plus tardif doit battre le precedent de 15 % relatif
CROP_PAD = 0.10               # marge autour de la boite du masque pour le crop-sujet
MIN_AREA_FRAC = 0.002         # un masque plus petit que 0.2 % de l'image = detection ratee
MAX_AREA_FRAC = 0.85          # un masque couvrant > 85 % de l'image ne separe plus sujet/fond


# ---------------------------------------------------------------------------------------------
# modeles
# ---------------------------------------------------------------------------------------------
def load_detectors(device):
    from transformers import (OwlViTForObjectDetection, OwlViTProcessor, SamModel,
                              SamProcessor)
    owl_p = OwlViTProcessor.from_pretrained(OWL_MODEL)
    owl_m = OwlViTForObjectDetection.from_pretrained(OWL_MODEL).eval().to(device)
    sam_p = SamProcessor.from_pretrained(SAM_MODEL)
    sam_m = SamModel.from_pretrained(SAM_MODEL).eval().to(device)
    for m in (owl_m, sam_m):
        for p in m.parameters():
            p.requires_grad_(False)
    return owl_p, owl_m, sam_p, sam_m


def to_pils(frames01):
    """(T, 3, H, W) dans [0,1] -> liste de PIL.Image RGB."""
    arr = (frames01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(a) for a in arr]


@torch.no_grad()
def detect(owl_p, owl_m, pils, queries, device, batch=4):
    """Score et meilleure boite de CHAQUE requete sur CHAQUE frame.
    Retour : scores (T, Q) float32, boxes (T, Q, 4) float32 en pixels (x0, y0, x1, y1)."""
    T, Q = len(pils), len(queries)
    scores = np.zeros((T, Q), dtype=np.float32)
    boxes = np.zeros((T, Q, 4), dtype=np.float32)
    sizes = torch.tensor([[p.height, p.width] for p in pils])
    for i in range(0, T, batch):
        chunk = pils[i:i + batch]
        inp = owl_p(text=[list(queries)] * len(chunk), images=chunk, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        out = owl_m(**inp)
        res = owl_p.post_process_object_detection(
            outputs=out, threshold=0.0, target_sizes=sizes[i:i + len(chunk)].to(device))
        for j, r in enumerate(res):
            s, b, lab = r["scores"].float().cpu(), r["boxes"].float().cpu(), r["labels"].cpu()
            for q in range(Q):
                sel = (lab == q).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                k = sel[s[sel].argmax()]
                scores[i + j, q] = float(s[k])
                boxes[i + j, q] = b[k].numpy()
    return scores, boxes


def pick_query(scores, queries):
    """Le detecteur arbitre : on part du candidat le plus prioritaire (celui que l'extraction
    du prompt a classe premier) et un candidat plus tardif ne le remplace que s'il le bat de
    SWITCH_MARGIN en score median sur les frames du segment. La mediane (et non la moyenne)
    evite qu'une seule frame chanceuse decide."""
    med = np.median(scores, axis=0)
    best = 0
    for q in range(1, len(queries)):
        if med[q] > med[best] * SWITCH_MARGIN:
            best = q
    return int(best), med


@torch.no_grad()
def segment(sam_p, sam_m, pils, boxes, device, batch=4):
    """SAM invite par la boite -> masque binaire (T, H, W) bool + score IoU predit (T,)."""
    T = len(pils)
    H, W = pils[0].height, pils[0].width
    masks = np.zeros((T, H, W), dtype=bool)
    iou = np.zeros((T,), dtype=np.float32)
    for i in range(0, T, batch):
        chunk = pils[i:i + batch]
        bx = [[[float(x) for x in boxes[i + j]]] for j in range(len(chunk))]
        inp = sam_p(chunk, input_boxes=bx, return_tensors="pt").to(device)
        out = sam_m(**inp, multimask_output=True)
        m = sam_p.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())
        sc = out.iou_scores.float().cpu()               # (B, 1, 3)
        for j in range(len(chunk)):
            k = int(sc[j, 0].argmax())
            masks[i + j] = m[j][0, k].numpy().astype(bool)
            iou[i + j] = float(sc[j, 0, k])
    return masks, iou


# ---------------------------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------------------------
def box_of_mask(mask, pad=CROP_PAD):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    H, W = mask.shape
    py, px = int(round((y1 - y0) * pad)), int(round((x1 - x0) * pad))
    return (max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px))


def make_crops(frame01, mask):
    """frame01 (3, H, W) dans [0,1], mask (H, W) bool ->
       (crop-sujet (3, h, w), crop-fond (3, H, W)).
    Le crop-SUJET est la boite serree du masque avec l'exterieur du masque mis A zero (c'est le
    sujet, pas sa vignette de contexte). Le crop-fond est le complement : l'image entiere avec
    le sujet mis a zero, exactement l'esprit du `zero_center` de score_comp.py, mais avec le
    vrai masque au lieu du carre central."""
    m = torch.from_numpy(mask)[None].float()
    subj_full = frame01 * m
    bg = frame01 * (1.0 - m)
    bb = box_of_mask(mask)
    if bb is None:
        return frame01 * 0.0, bg
    y0, y1, x0, x1 = bb
    return subj_full[:, y0:y1, x0:x1].contiguous(), bg


def save_jpeg(t01, path, quality=88, max_w=320):
    a = (t01.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    im = Image.fromarray(a)
    if im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), Image.BILINEAR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, quality=quality)


# ---------------------------------------------------------------------------------------------
# un segment
# ---------------------------------------------------------------------------------------------
def process_segment(meta, seg, latents, models, device, args, force_qi=None):
    owl_p, owl_m, sam_p, sam_m, vae = models
    key = C.seg_key(meta["video_id"], seg["name"])
    # corpus a sujets non HUMAINS (v4) : on coupe le repli "a person" a la source, sinon
    # `pick_query` peut le preferer et poser le masque sur un figurant du decor.
    sp = C.subject_phrase(seg["prompt"],
                          person_fallback=getattr(args, "corpus", "v1")
                          not in C.NO_PERSON_CORPORA)
    idx = C.sample_segment_frames(seg["f0"], seg["f1"], k=args.frames)
    px = C.decode_latent_frames(vae, latents[idx], device, batch=args.vae_batch)
    pils = to_pils(px)

    scores, boxes = detect(owl_p, owl_m, pils, sp["queries"], device, batch=args.det_batch)
    qi, med = pick_query(scores, sp["queries"])
    if force_qi is not None:
        qi = int(force_qi)
    masks, iou = segment(sam_p, sam_m, pils, boxes[:, qi], device, batch=args.sam_batch)

    H, W = masks.shape[-2:]
    area = masks.reshape(len(idx), -1).mean(axis=1)
    # IoU(masque, boite) > 0 est la condition du contrat de selftest : le masque doit
    # recouper la detection qui l'a produit (un masque vide, ou parti ailleurs, est invalide).
    iou_box = np.zeros(len(idx), dtype=np.float32)
    for t in range(len(idx)):
        x0, y0, x1, y1 = boxes[t, qi]
        bm = np.zeros((H, W), dtype=bool)
        bm[max(0, int(y0)):min(H, int(np.ceil(y1))),
           max(0, int(x0)):min(W, int(np.ceil(x1)))] = True
        inter = np.logical_and(masks[t], bm).sum()
        union = np.logical_or(masks[t], bm).sum()
        iou_box[t] = inter / max(1, union)
    valid = (iou_box > 0) & (area >= MIN_AREA_FRAC) & (area <= MAX_AREA_FRAC)

    subj_emb, bg_emb, crops = [], [], []
    for t in range(len(idx)):
        cs, cb = make_crops(px[t], masks[t])
        if cs.numel() == 0 or min(cs.shape[-2:]) < 4:
            cs = px[t] * 0.0
        subj_emb.append(cs)
        bg_emb.append(cb)
        crops.append((cs, cb))
    subj_e = C.dino_embed(torch.stack([Fn.interpolate(c[None], size=(224, 224),
                                                      mode="bilinear", align_corners=False)[0]
                                       for c in subj_emb]), device)
    bg_e = C.dino_embed(torch.stack(bg_emb), device)

    if not args.no_crops:
        for t in range(len(idx)):
            base = os.path.join(C.MASK_DIR, "crops", key, f"f{idx[t]:03d}")
            save_jpeg(crops[t][0], base + "_subj.jpg")
            save_jpeg(crops[t][1], base + "_bg.jpg")
            save_jpeg(px[t], base + "_orig.jpg")     # l'image PROPRE, pour la planche d'oeil

    rec = {
        "video_id": meta["video_id"], "segment": seg["name"], "key": key,
        "prompt": seg["prompt"], "queries": sp["queries"], "query_found": sp["found"],
        "query_chosen": sp["queries"][qi], "query_index": int(qi),
        "query_medians": [float(x) for x in med],
        "query_joint": bool(force_qi is not None),
        "frames": np.asarray(idx, dtype=np.int32),
        "boxes": boxes[:, qi].astype(np.float32),
        "det_scores": scores[:, qi].astype(np.float32),
        "sam_iou": iou.astype(np.float32),
        "iou_box": iou_box.astype(np.float32),
        "area_frac": area.astype(np.float32),
        "valid": valid,
        "flagged": bool((~valid).any() or not sp["found"]),
        "masks_packed": np.packbits(masks.reshape(len(idx), -1), axis=1),
        "mask_shape": (int(H), int(W)),
        "dino_subj": subj_e.numpy().astype(np.float32),
        "dino_bg": bg_e.numpy().astype(np.float32),
    }
    np.savez_compressed(os.path.join(C.MASK_DIR, key + ".npz"), **rec)
    torch.save({"subj": subj_e, "bg": bg_e, "frames": torch.tensor(idx),
                "valid": torch.from_numpy(valid)}, os.path.join(C.REFS_DIR, key + ".pt"))
    return rec


def unpack_masks(rec):
    T = rec["frames"].shape[0]
    H, W = [int(x) for x in rec["mask_shape"]]
    return np.unpackbits(rec["masks_packed"], axis=1, count=H * W).reshape(T, H, W).astype(bool)


# ---------------------------------------------------------------------------------------------
# planche de validation a l'oeil
# ---------------------------------------------------------------------------------------------
def cell_measures(rec, masks):
    """Les MESURES AUTOMATIQUES ecrites dans la cellule (materiau v4) : ce que la machine sait
    dire du masque, pour qu'Aden n'ait a juger que ce que lui seul peut juger (l'identite).
      aire  = fraction de l'image sous le masque, mediane sur les frames valides ;
      conf  = score de detection median (confiance du detecteur ouvert-vocabulaire) ;
      stab  = STABILITE DU SUIVI = IoU moyenne entre masques de frames consecutives : 1.0 =
              le masque designe la meme chose d'un bout a l'autre, ~0 = il saute d'un objet
              a l'autre (le defaut qui a fait exclure des paires v3) ;
      alerte = le drapeau du .npz (frame invalide ou tete non reconnue) OU une mesure hors
              des clous (aire < 1 % ou > 60 %, conf < 0.15, stab < 0.5)."""
    v = rec["valid"].astype(bool) if "valid" in rec.files else np.ones(len(masks), bool)
    area = float(np.median(rec["area_frac"][v])) if v.any() else 0.0
    conf = float(np.median(rec["det_scores"][v])) if v.any() else 0.0
    ious, idx = [], np.flatnonzero(v)
    for a, b in zip(idx[:-1], idx[1:]):
        ma, mb = masks[a], masks[b]
        u = np.logical_or(ma, mb).sum()
        ious.append(float(np.logical_and(ma, mb).sum()) / float(u) if u else 0.0)
    stab = float(np.mean(ious)) if ious else 1.0
    alert = bool(rec["flagged"]) or not (0.01 <= area <= 0.60) or conf < 0.15 or stab < 0.5
    return {"area": area, "conf": conf, "stab": stab, "alert": alert}


def build_board(keys, out_path, cols=3, thumb_w=208, quality=78, max_bytes=500_000,
                numbers=None, measures=False):
    """Grille [image | masque] pour ~24 segments varies, JPEG leger (< 0.5 MB) : l'objet que
    l'oeil d'Aden doit valider AVANT tout usage des masques.
    `numbers` : {cle de segment -> numero de PAIRE}. C'est la planche NUMEROTEE (celle a
    laquelle les listes d'oeil scellees au registre font reference) : le numero est ecrit en
    grand, en tete de cellule, et les deux jumeaux d'une paire portent le MEME numero."""
    from PIL import ImageDraw
    cells = []
    for k in keys:
        p = os.path.join(C.MASK_DIR, k + ".npz")
        if not os.path.exists(p):
            continue
        rec = np.load(p, allow_pickle=True)
        masks = unpack_masks(rec)
        t = int(np.argmax(rec["valid"])) if rec["valid"].any() else 0
        cdir = os.path.join(C.MASK_DIR, "crops", k)
        fi = int(rec["frames"][t])
        origp = os.path.join(cdir, f"f{fi:03d}_orig.jpg")
        if not os.path.exists(origp):
            continue
        orig = Image.open(origp).convert("RGB")
        m = Image.fromarray((masks[t] * 255).astype(np.uint8)).resize(orig.size, Image.NEAREST)
        over = Image.new("RGB", orig.size, (255, 40, 120))
        mid = Image.blend(orig, Image.composite(over, orig, m), 0.55)   # le sujet, surligne
        sw = int(rec["query_index"]) != 0                # le detecteur a change de candidat
        mes = cell_measures(rec, masks) if measures else None
        cells.append((k, ("~" if sw else "") + str(rec["query_chosen"]),
                      float(rec["det_scores"][t]),
                      bool(mes["alert"]) if mes else bool(rec["flagged"]),
                      orig, mid, m.convert("RGB"), mes))
    if not cells:
        return None
    tw = thumb_w
    th = max(1, round(tw * cells[0][4].height / cells[0][4].width))
    lab = 15 if numbers is None else 26
    if measures:
        lab += 15                       # 2e ligne de texte : les mesures automatiques
    cw, ch = tw * 3, th + lab
    rows = (len(cells) + cols - 1) // cols
    board = Image.new("RGB", (cols * cw, rows * ch), (12, 12, 14))
    dr = ImageDraw.Draw(board)
    big = None
    if numbers is not None:
        try:
            from PIL import ImageFont
            big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            big = None
    for i, (k, q, s, fl, orig, mid, m, mes) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = c * cw, r * ch
        # colonne 1 : image sans le sujet (= le crop-fond) ; colonne 2 : sujet surligne ;
        # colonne 3 : le masque binaire
        for j, im in enumerate((orig, mid, m)):
            board.paste(im.resize((tw, th), Image.BILINEAR), (x + j * tw, y + lab))
        txt = f"{'!' if fl else ' '}{k[:26]} | {q[:22]} | {s:.2f}"
        col = (255, 120, 160) if fl else (200, 210, 220)
        if numbers is None:
            dr.text((x + 3, y + 2), txt, fill=col)
        else:
            no = numbers.get(k, 0)
            dr.text((x + 4, y + 2), f"{no:02d}", fill=(255, 220, 90), font=big)
            dr.text((x + 42, y + 8), txt, fill=col)
        if mes is not None:
            dr.text((x + 42, y + 26),
                    f"aire {mes['area'] * 100:.1f}%  conf {mes['conf']:.2f}  "
                    f"stab {mes['stab']:.2f}  {'ALERTE' if mes['alert'] else 'ok'}",
                    fill=(255, 120, 160) if mes["alert"] else (150, 200, 160))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # contrainte du plan : la planche doit rester LEGERE (consultable sur telephone). On baisse
    # la qualite par paliers jusqu'a tenir le budget, plutot que de rendre un fichier lourd.
    for q in (quality, 65, 52, 42):
        board.save(out_path, quality=q, optimize=True)
        if os.path.getsize(out_path) <= max_bytes:
            break
    return out_path


def board_keys(vids, pairs, n=24):
    """~24 segments varies : d'abord les segments signales (c'est eux qu'il faut voir), puis un
    balayage des tetes de sujet distinctes, en privilegiant les videos des paires (celles dont
    depend le sondage). Deterministe."""
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    segs = []
    for v in vids:
        for s in C.segments(v):
            k = C.seg_key(v["video_id"], s["name"])
            p = os.path.join(C.MASK_DIR, k + ".npz")
            if not os.path.exists(p):
                continue
            rec = np.load(p, allow_pickle=True)
            segs.append({"key": k, "head": str(rec["query_chosen"]),
                         "flag": bool(rec["flagged"]), "pair": v["video_id"] in pair_ids})
    segs.sort(key=lambda d: d["key"])
    out, heads = [], set()
    for d in [x for x in segs if x["flag"]][:8]:
        out.append(d["key"])
        heads.add(d["head"])
    for want_pair in (True, False):
        for d in segs:
            if len(out) >= n:
                break
            if d["pair"] != want_pair or d["key"] in out or d["head"] in heads:
                continue
            out.append(d["key"])
            heads.add(d["head"])
    for d in segs:                                  # complement si trop peu de tetes distinctes
        if len(out) >= n:
            break
        if d["key"] not in out:
            out.append(d["key"])
    return out[:n]


def fix_pairs(pairs, models, device, args, corpus="v1"):
    """LE CORRECTIF de coherence des JUMEAUX.

    Une paire, c'est DEUX rollouts du MEME prompt : la liste de requetes candidates y est donc
    identique par construction. Mais le detecteur les arbitre video par video, et il lui arrive
    de trancher differemment des deux cotes (mesure : `main:204394` -> "medium-sized brown dog",
    `twin:204394` -> "small bird" ; `main:142397` -> "vase", `twin:142397` -> "small table").
    Le sondeur comparerait alors "un chien contre un oiseau" au lieu de "ce chien contre CET
    AUTRE chien" : la tache changerait de nature sur ces paires.

    Correctif : la requete est choisie UNE fois par PAIRE, sur les scores medians des DEUX
    videos (moyenne des deux medianes, meme regle de marge de priorite), puis les segments qui
    n'avaient pas ce choix sont RECALCULES avec la requete imposee. Aucun autre changement."""
    import mem2_data as D
    cache = D.PartCache(size=1)
    vids, _ = C.pair_dataset(verbose=False, corpus=corpus)
    byid = C.vids_by_id(vids)
    todo, checked = [], 0
    for p in pairs:
        recs, keys = {}, {}
        for role in ("main", "twin"):
            keys[role] = C.seg_key(p[role], "A")
            f = os.path.join(C.MASK_DIR, keys[role] + ".npz")
            if not os.path.exists(f):
                break
            recs[role] = np.load(f, allow_pickle=True)
        if len(recs) != 2:
            continue
        checked += 1
        qa = [str(x) for x in recs["main"]["queries"]]
        qb = [str(x) for x in recs["twin"]["queries"]]
        assert qa == qb, f"paire {p['corpus_idx']}: listes de requetes differentes {qa} vs {qb}"
        med = (np.asarray(recs["main"]["query_medians"], dtype=np.float64)
               + np.asarray(recs["twin"]["query_medians"], dtype=np.float64)) / 2.0
        joint = 0
        for q in range(1, len(qa)):
            if med[q] > med[joint] * SWITCH_MARGIN:
                joint = q
        for role in ("main", "twin"):
            if int(recs[role]["query_index"]) != joint or not bool(
                    recs[role].get("query_joint", np.array(False))):
                todo.append((p[role], joint, qa[joint], int(recs[role]["query_index"])))
    print(f"[masks] coherence des jumeaux : {checked} paires verifiees, "
          f"{len(todo)} segments a recalculer", flush=True)
    n_changed = sum(1 for _, j, _, old in todo if j != old)
    print(f"[masks]   dont {n_changed} avec un CHANGEMENT de requete", flush=True)
    todo.sort(key=lambda t: (byid[t[0]]["part"], byid[t[0]]["vi"]))
    for vid, joint, qtext, old in todo:
        meta = byid[vid]
        seg = C.segments(meta)[0]
        rec0 = D.get_record(meta, cache)
        r = process_segment(meta, seg, rec0["latents"], models, device, args, force_qi=joint)
        del rec0
        if joint != old:
            print(f"[masks]   {vid}: requete {old} -> {joint} ({qtext!r}), "
                  f"signale={r['flagged']}", flush=True)
    return len(todo), n_changed


def pair_board_keys(pairs, numbered=False):
    """Une cellule pour CHACUNE des 60 videos de paire (30 paires cote a cote) : c'est ce
    corpus-la, et lui seul, dont depend le sondage, l'oeil doit le voir en entier.
    `numbered` : ordre de la planche NUMEROTEE = celui de C.board_numbers (identifiant de la
    video principale, lexicographique), le seul auquel les listes d'oeil scellees renvoient.
    Rend (cles, {cle: numero de paire})."""
    order = sorted(pairs, key=lambda q: q["board_no"]) if numbered else \
        sorted(pairs, key=lambda q: (q["split"], q["corpus_idx"]))
    out, nums = [], {}
    for p in order:
        for role in ("main", "twin"):
            k = C.seg_key(p[role], "A")
            if os.path.exists(os.path.join(C.MASK_DIR, k + ".npz")):
                out.append(k)
                nums[k] = int(p["board_no"])
    return out, nums


def boards_to_make(vids, pairs, args):
    """(cles, chemin, numeros) des planches a rendre : la planche VARIEE des ~24 segments du
    corpus, la planche des 30 paires, et la planche NUMEROTEE (l'objet du verdict d'oeil)."""
    suf = "" if args.corpus == "v1" else f"_{args.corpus}"
    # les planches du corpus v1 ne doivent JAMAIS etre ecrasees par celles du v2 (l'oeil d'Aden
    # a rendu son verdict sur les v1) : le suffixe porte sur les trois.
    yield (board_keys(vids, pairs, args.board_n),
           args.board.replace(".jpg", f"{suf}.jpg"), None, {})
    keys, _ = pair_board_keys(pairs)
    yield keys, os.path.join(C.BOARD_DIR, f"masks_board_pairs{suf}.jpg"), None, {}
    keys, nums = pair_board_keys(pairs, numbered=True)
    yield (keys, os.path.join(C.BOARD_DIR, f"masks_board_pairs_numerote{suf}.jpg"), nums, {})
    # PLANCHE de LABELLISATION (materiau v4) : UNE ligne par PAIRE (cols=2 : la principale et
    # sa jumelle cote a cote, jamais coupees par un saut de ligne) + les MESURES AUTOMATIQUES
    # ecrites dans chaque cellule. C'est celle que l'oeil d'Aden annote.
    yield (keys, os.path.join(C.BOARD_DIR, f"labels_board{suf}.jpg"), nums,
           {"cols": 2, "measures": True, "thumb_w": 232, "max_bytes": 2_000_000})


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="les 165 videos du corpus")
    ap.add_argument("--only_pairs", action="store_true", help="seulement les 60 videos des paires")
    ap.add_argument("--limit", type=int, default=0, help="fumee : n premieres videos")
    ap.add_argument("--frames", type=int, default=C.FRAMES_PER_SEGMENT)
    ap.add_argument("--vae_batch", type=int, default=4)
    ap.add_argument("--det_batch", type=int, default=4)
    ap.add_argument("--sam_batch", type=int, default=4)
    ap.add_argument("--no_crops", action="store_true")
    ap.add_argument("--redo", action="store_true", help="recalcule meme si le .npz existe")
    ap.add_argument("--board_only", action="store_true")
    ap.add_argument("--fix_pairs", action="store_true",
                    help="impose a chaque PAIRE une requete unique (les jumeaux partagent le "
                         "prompt) et recalcule les segments concernes")
    ap.add_argument("--board_n", type=int, default=24)
    ap.add_argument("--corpus", default="v1", choices=("v1", "v2", "v3", "v4"))
    ap.add_argument("--board", default=os.path.join(C.BOARD_DIR, "masks_board.jpg"))
    args = ap.parse_args()

    C.ensure_dirs()
    C.set_determinism(0)
    vids, pairs = C.pair_dataset(verbose=True, corpus=args.corpus)

    if args.board_only:
        for keys, out, nums, opts in boards_to_make(vids, pairs, args):
            opts = {"max_bytes": 1_200_000 if nums else 500_000, **opts}
            p = build_board(keys, out, numbers=nums, **opts)
            if p:
                print(f"[masks] planche -> {p} ({os.path.getsize(p) / 1e6:.2f} MB, "
                      f"{len(keys)} cellules)")
        return

    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sel = [v for v in vids if (not args.only_pairs or v["video_id"] in pair_ids)]
    sel.sort(key=lambda v: (v["part"], v["vi"]))     # une part mmap ouverte a la fois
    if args.limit:
        sel = sel[:args.limit]
    assert args.all or args.only_pairs or args.limit or args.fix_pairs, \
        "precise --all, --only_pairs, --limit ou --fix_pairs"

    device = "cuda"
    print(f"[masks] {len(sel)} videos, modeles sur {device} (HF_HOME={C.HF_HOME})", flush=True)
    models = load_detectors(device)
    vae = C.get_vae(device)
    models = models + (vae,)
    cache = D.PartCache(size=1)

    if args.fix_pairs:
        n_todo, n_ch = fix_pairs(pairs, models, device, args, corpus=args.corpus)
        for keys, out, nums, opts in boards_to_make(vids, pairs, args):
            opts = {"max_bytes": 1_200_000 if nums else 500_000, **opts}
            p = build_board(keys, out, numbers=nums, **opts)
            if p:
                print(f"[masks] planche -> {p} ({os.path.getsize(p) / 1e6:.2f} MB, "
                      f"{len(keys)} cellules)")
        print(f"[masks] TERMINE (fix_pairs) : {n_todo} recalcules, {n_ch} changements")
        return

    t0, done, flagged = time.time(), 0, []
    for n, meta in enumerate(sel):
        segs = C.segments(meta)
        todo = [s for s in segs if args.redo or not os.path.exists(
            os.path.join(C.MASK_DIR, C.seg_key(meta["video_id"], s["name"]) + ".npz"))]
        if not todo:
            continue
        rec0 = D.get_record(meta, cache)
        latents = rec0["latents"]
        for s in todo:
            r = process_segment(meta, s, latents, models, device, args)
            done += 1
            if r["flagged"]:
                flagged.append(r["key"])
        del rec0, latents
        if (n + 1) % 5 == 0 or n == len(sel) - 1:
            el = time.time() - t0
            print(f"[masks] {n + 1}/{len(sel)} videos, {done} segments, {len(flagged)} signales, "
                  f"{el / 60:.1f} min ({el / max(1, n + 1):.1f} s/video)", flush=True)

    # un fichier par corpus : la campagne v2 ecrasait silencieusement le resume de la v1
    # (audit 2 du 05/08 ; artefact de documentation, aucun effet sur le protocole).
    C.save_json({"corpus": args.corpus, "n_videos": len(sel), "n_segments_done": done,
                 "flagged": flagged},
                os.path.join(C.MASK_DIR, f"_summary_{args.corpus}.json"))
    for keys, out, nums, opts in boards_to_make(vids, pairs, args):
        opts = {"max_bytes": 1_200_000 if nums else 500_000, **opts}
        p = build_board(keys, out, numbers=nums, **opts)
        if p:
            print(f"[masks] planche -> {p} ({os.path.getsize(p) / 1e6:.2f} MB, "
                  f"{len(keys)} cellules)")
    print(f"[masks] TERMINE : {done} segments, {len(flagged)} signales -> {flagged[:12]}")


if __name__ == "__main__":
    main()
