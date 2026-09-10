#!/usr/bin/env python
# ============================================================================================
# BLIND JUDGING BOARDS -- generator. Rebuilt 2026-09-09 from the sixteen boards published
# between 13 and 29 August (their source is gone with the working tree; only the HTML pages
# survived, and their exact wording, answer options and "copy my answers" mechanism were read
# back from them).
#
# NOTE ON LANGUAGE: the strings SHOWN TO THE JUDGE stay in French, deliberately. They are the
# experimental protocol -- the August campaigns were judged with that exact wording, and
# changing it would break comparability with those judgments. Only the code is in English.
#
# THREE FORMATS, one per question the campaign asked:
#
#   pair2   "Reconnait-on la personne ?" (22/08, 30 questions)
#           one image on top (decoder output OR the real image blurred), two real people below,
#           which one is it?  ->  measures the INSTANCE FIDELITY of the output against the blur
#           ladder.  Options: left | right | cannot tell.
#
#   rank3   "La mosaique a-t-elle disparu ?" (23/08, 10 rows x 6 answers)
#           the real person on the left, three shuffled reconstructions (the old one + two new):
#           rank all three, then judge each.  ->  compares DECODER VARIANTS against each other.
#           Judgment options: yes | roughly | no.
#
#   duel2   "Lequel est bien la personne du debut ?" (23 to 29 August, up to 20 scenes)
#           two PAIRED versions of the same video return, left and right, which one brings the
#           person back.  ->  the final test, on video, between two arms.  Options: L | R |
#           cannot tell.
#
# HOUSE RULES, all applied here and all verified by selftest_boards.py:
#  1. The answer key lives in key_<id>.json, NEVER in the page. The selftest reads the produced
#     HTML back and fails if it finds an arm name, a file path or the real column order.
#  2. The permutation comes from a FROZEN salted hash (like fidelity_common.letter_perm): the
#     same board regenerated gives the same shuffle, so a lost board can be rebuilt.
#  3. "Cannot tell" is a first-class answer, never a blank: the instructions tell the judge to
#     use it, and the scorer counts it separately instead of diluting it into chance.
#  4. The side / letter of the true subject is drawn PER QUESTION, never per board.
#  5. The page is self-contained: no external script, no external font, media as data URIs. It
#     goes into an Artifact as is (16 MB ceiling, checked at build time).
#
# Usage:
#   python boards.py demo --out /tmp/demo          # builds one board of each type
#   from boards import build_pair2, build_rank3, build_duel2
# ============================================================================================
import base64
import hashlib
import html
import json
import mimetypes
import os
import time

# FROZEN salt. Changing it invalidates every board already judged: never touch it.
SALT = "eyeboards-2026-09-09"
CODE_VERSION = "1"
ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
LETTERS = "XYZWVU"

_JUDGE = {"o": "oui", "a": "a peu pres", "n": "non"}


# ----------------------------- permutation deterministe ---------------------------------------
def perm(board_id, item_id, n):
    """Deterministic permutation of n slots for (board, question). Returns the list of source
    indices in display order.

    Fisher-Yates driven by an sha256 byte stream: reproducible on any machine and without numpy,
    unlike random.shuffle whose implementation is not a contract."""
    assert 1 <= n <= len(LETTERS), f"n={n} out of range"
    idx = list(range(n))
    stream = hashlib.sha256(f"{SALT}|{board_id}|{item_id}".encode()).digest()
    k = 0
    for i in range(n - 1, 0, -1):
        if k >= len(stream):                       # extend the stream for a long board
            stream += hashlib.sha256(stream).digest()
        j = stream[k] % (i + 1)
        k += 1
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def code_checksum(board_id, payload):
    return hashlib.sha256(f"{SALT}|{board_id}|{payload}".encode()).hexdigest()[:6]


# ----------------------------- medias --------------------------------------------------------
def data_uri(path):
    """Local file -> data URI. The MIME type comes from the extension; an unknown type is an
    error (a mute board because an mp4 was served as octet-stream costs several hours of
    generation)."""
    mime, _ = mimetypes.guess_type(path)
    assert mime, f"unknown MIME type for {path}"
    with open(path, "rb") as f:
        raw = f.read()
    assert raw, f"empty file: {path}"
    return f"data:{mime};base64," + base64.b64encode(raw).decode(), len(raw)


def _media_tag(path, kind, cls):
    uri, n = data_uri(path)
    if kind == "video":
        tag = (f'<video class="{cls}" src="{uri}" muted loop playsinline preload="metadata" '
               f'controls></video>')
    else:
        tag = f'<img class="{cls}" src="{uri}" alt="">'
    return tag, n


# ----------------------------- la page --------------------------------------------------------
_CSS = """
:root{--ground:#F4F3F1;--panel:#FFF;--sunk:#EBE8E3;--ink:#1B1917;--soft:#6D6862;
 --line:#DEDAD3;--amber:#9C5A1E;--amber-soft:#F6EADC;--ok:#3F6B4A}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#151412;
 --panel:#1E1D1A;--sunk:#111110;--ink:#F0EDE7;--soft:#A49E96;--line:#302E2A;
 --amber:#D9A05B;--amber-soft:#3A2A18;--ok:#7FB08C}}
:root[data-theme="dark"]{--ground:#151412;--panel:#1E1D1A;--sunk:#111110;--ink:#F0EDE7;
 --soft:#A49E96;--line:#302E2A;--amber:#D9A05B;--amber-soft:#3A2A18;--ok:#7FB08C}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);padding:0 0 64px;
 font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:0 20px}
h1{font-size:26px;margin:32px 0 8px;font-weight:600;letter-spacing:-.01em}
.lede{color:var(--soft);max-width:64ch;margin:0 0 8px}
.lede b{color:var(--ink);font-weight:600}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:16px;margin:18px 0}
.num{font:600 12px/1 ui-monospace,monospace;color:var(--soft);letter-spacing:.08em}
.q{margin:10px 0 12px;font-weight:500}
.row{display:flex;gap:12px;flex-wrap:wrap}
.col{flex:1 1 200px;min-width:180px}
.cap{font:600 12px/1.4 ui-monospace,monospace;color:var(--soft);margin:6px 0 4px;
 text-transform:uppercase;letter-spacing:.06em}
img.m,video.m{width:100%;border-radius:6px;background:var(--sunk);display:block}
.probe{max-width:340px;margin:0 auto 14px}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
button.opt{font:inherit;padding:7px 13px;border-radius:999px;cursor:pointer;
 border:1px solid var(--line);background:var(--panel);color:var(--ink)}
button.opt:hover{border-color:var(--amber)}
button.opt[aria-pressed="true"]{background:var(--amber-soft);border-color:var(--amber);
 color:var(--amber);font-weight:600}
.bar{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);
 padding:14px 20px;margin-top:28px}
.bar .wrap{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.count{font:600 13px ui-monospace,monospace;color:var(--soft)}
code#code{display:block;flex:1 1 320px;background:var(--sunk);border:1px solid var(--line);
 border-radius:6px;padding:9px 11px;font:12px/1.5 ui-monospace,monospace;word-break:break-all;
 color:var(--soft);min-height:20px}
.done{color:var(--ok);font-weight:600}
"""

_JS = """
(function(){
 var ANS={},N=%(nslots)d,BID=%(bid)s,SUM=%(sum)s,VER=%(ver)s;
 function crc(s){/* the same truncated sha256 as code_checksum, page side */
  return SUM;}
 document.addEventListener('click',function(e){
  var b=e.target.closest('button.opt'); if(!b) return;
  var slot=b.dataset.slot, val=b.dataset.val;
  document.querySelectorAll('button.opt[data-slot="'+slot+'"]').forEach(function(o){
   o.setAttribute('aria-pressed', o===b?'true':'false');});
  ANS[slot]=val; render();});
 function render(){
  var out='',done=0;
  for(var i=0;i<N;i++){var v=ANS[String(i)];out+= v?v:'.'; if(v)done++;}
  document.getElementById('count').textContent=done+' / '+N+' repondues';
  var el=document.getElementById('code');
  if(done<N){el.textContent='reponds aux '+N+' cases : le code apparait ici';
             el.classList.remove('done');}
  else{el.textContent=BID+'~'+VER+'~'+out+'~'+SUM; el.classList.add('done');}
 }
 document.getElementById('copy').addEventListener('click',function(){
  var t=document.getElementById('code').textContent;
  navigator.clipboard&&navigator.clipboard.writeText(t);});
 render();
})();
"""


def _page(title, lede, cells, board_id, nslots, checksum):
    body = "\n".join(cells)
    js = _JS % {"nslots": nslots, "bid": json.dumps(board_id),
                "sum": json.dumps(checksum), "ver": json.dumps(CODE_VERSION)}
    return f"""<title>{html.escape(title)}</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>{html.escape(title)}</h1>
{lede}
{body}
</div>
<div class="bar"><div class="wrap">
<span class="count" id="count"></span>
<code id="code"></code>
<button class="opt" id="copy">copier</button>
</div></div>
<script>{js}</script>
"""


def _finish(out_dir, board_id, title, lede, cells, slots, key, kind, nslots):
    os.makedirs(out_dir, exist_ok=True)
    # The checksum depends ONLY on the board: it authenticates the returned code (right board,
    # right version), it encrypts nothing -- the key lives elsewhere.
    checksum = code_checksum(board_id, kind)
    page = _page(title, lede, cells, board_id, nslots, checksum)
    html_path = os.path.join(out_dir, f"{board_id}.html")
    with open(html_path, "w", encoding="utf8") as f:
        f.write(page)
    size = os.path.getsize(html_path)
    assert size <= ARTIFACT_MAX_BYTES, (
        f"board is {size/1e6:.1f} MB > the 16 MB Artifact ceiling: shrink the media "
        f"(the August campaign kept ~0.29 MB per clip, cf. anonymize.py)")
    key_path = os.path.join(out_dir, f"key_{board_id}.json")
    with open(key_path, "w", encoding="utf8") as f:
        json.dump({"board_id": board_id, "kind": kind, "version": CODE_VERSION,
                   "salt": SALT, "checksum": checksum, "n_slots": nslots,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "slots": slots, "items": key}, f, indent=1, ensure_ascii=False)
    return {"html": html_path, "key": key_path, "bytes": size, "n_slots": nslots}


# ----------------------------- pair2 ----------------------------------------------------------
def build_pair2(items, out_dir, board_id="pair2", title="Reconnait-on la personne ?"):
    """items: [{id, probe, true, twin, meta?}] -- probe = the image to judge (decoder output or
    the real image blurred), true/twin = the two real people.

    The side of the true subject is drawn PER QUESTION (rule 4): knowing where the probe comes
    from therefore does not help, which the instructions state explicitly to the judge."""
    lede = ('<p class="lede">En haut, <b>une image</b>. En dessous, <b>deux personnes '
            'reelles</b> : l\'une est celle de l\'image, l\'autre est son sosie, genere avec '
            'le meme texte et une autre graine. Tu dis laquelle.</p>'
            '<p class="lede">L\'image du haut vient parfois de la memoire du modele, parfois '
            'de la vraie image rendue floue. Savoir d\'ou elle vient ne t\'aide pas : le cote '
            'du bon sujet est tire au sort a chaque question. Regarde la personne, rien '
            'd\'autre. <b>Si tu ne peux pas trancher, dis-le</b> plutot que de deviner : une '
            'reponse au hasard abime la mesure, un doute est une information.</p>')
    cells, key, slots, total = [], [], [], 0
    for n, it in enumerate(items):
        order = perm(board_id, it["id"], 2)            # 0 -> true, 1 -> twin
        cand = [it["true"], it["twin"]]
        sides = ["gauche", "droite"]
        cols = []
        for pos, src in enumerate(order):
            tag, _ = _media_tag(cand[src], "image", "m")
            cols.append(f'<div class="col"><div class="cap">{sides[pos]}</div>{tag}</div>')
        probe, _ = _media_tag(it["probe"], "image", "m")
        opts = "".join(
            f'<button class="opt" data-slot="{total}" data-val="{v}">{lbl}</button>'
            for v, lbl in (("g", "celle de gauche"), ("d", "celle de droite"),
                           ("?", "impossible a dire")))
        cells.append(
            f'<div class="cell"><div class="num">{n+1:02d}</div>'
            f'<div class="probe">{probe}</div>'
            f'<div class="q">cette image, c\'est laquelle des deux personnes ?</div>'
            f'<div class="row">{"".join(cols)}</div>'
            f'<div class="opts">{opts}</div></div>')
        # the key: which side holds the TRUE subject
        true_side = "g" if order[0] == 0 else "d"
        key.append({"id": it["id"], "slot": total, "true_side": true_side,
                    "meta": it.get("meta", {})})
        slots.append({"slot": total, "item": it["id"], "answers": ["g", "d", "?"]})
        total += 1
    return _finish(out_dir, board_id, title, lede, cells, slots, key, "pair2", total)


# ----------------------------- rank3 -----------------------------------------------------------
def build_rank3(items, out_dir, board_id="rank3", title="La mosaique a-t-elle disparu ?"):
    """items: [{id, truth, variants: {name: path} (exactly 3), meta?}].

    Two answers per variant -- a RANK (1..3) and a judgment -- because the 23/08 board asked for
    both: the rank compares variants against each other (which survives even if none is good),
    the judgment says whether one clears the absolute bar."""
    lede = ('<p class="lede">A gauche <b>la vraie personne</b>. A droite, trois '
            'reconstructions faites depuis la memoire, melangees a chaque ligne. Pour chaque '
            'ligne, <b>classe les trois</b> de la meilleure a la moins bonne, puis dis pour '
            'chacune si c\'est bien cette personne.</p>'
            '<p class="lede"><b>oui</b> : je le reconnaitrais, je tiens le pari &middot; '
            '<b>a peu pres</b> : ca pourrait etre lui, je ne parierais pas &middot; '
            '<b>non</b> : personne en particulier, ou quelqu\'un d\'autre.</p>')
    cells, key, slots, total = [], [], [], 0
    for n, it in enumerate(items):
        names = list(it["variants"])
        assert len(names) == 3, f"rank3 expects 3 variants, {it['id']} has {len(names)}"
        order = perm(board_id, it["id"], 3)
        truth, _ = _media_tag(it["truth"], "image", "m")
        cols = [f'<div class="col"><div class="cap">la vraie personne</div>{truth}</div>']
        item_key = {"id": it["id"], "letters": {}, "meta": it.get("meta", {})}
        for pos, src in enumerate(order):
            letter = LETTERS[pos]
            tag, _ = _media_tag(it["variants"][names[src]], "image", "m")
            ranks = "".join(
                f'<button class="opt" data-slot="{total}" data-val="{r}">{r}</button>'
                for r in (1, 2, 3))
            judge = "".join(
                f'<button class="opt" data-slot="{total+1}" data-val="{v}">{lbl}</button>'
                for v, lbl in _JUDGE.items())
            cols.append(
                f'<div class="col"><div class="cap">{letter}</div>{tag}'
                f'<div class="cap">rang</div><div class="opts">{ranks}</div>'
                f'<div class="cap">c\'est lui ?</div><div class="opts">{judge}</div></div>')
            item_key["letters"][letter] = {"variant": names[src],
                                           "rank_slot": total, "judge_slot": total + 1}
            slots.append({"slot": total, "item": it["id"], "answers": ["1", "2", "3"]})
            slots.append({"slot": total + 1, "item": it["id"], "answers": list(_JUDGE)})
            total += 2
        cells.append(f'<div class="cell"><div class="num">{n+1:02d}</div>'
                     f'<div class="row">{"".join(cols)}</div></div>')
        key.append(item_key)
    return _finish(out_dir, board_id, title, lede, cells, slots, key, "rank3", total)


# ----------------------------- duel2 ------------------------------------------------------------
def build_duel2(items, out_dir, board_id="duel2",
                title="Lequel est bien la personne du debut ?"):
    """items: [{id, arms: {arm_name: video_path} (exactly 2), meta?}].

    This is the final board of 29 August. There is no right answer in the answer-key sense: we
    count preferences per arm, and the scorer tests them against a coin flip."""
    lede = ('<p class="lede">Dans chaque scene, une personne apparait, une autre scene se '
            'deroule, puis on la fait revenir. Tu vois <b>deux versions du meme retour</b>, '
            'cote a cote : a gauche (G) et a droite (D). Une seule question : <b>laquelle des '
            'deux ramene la personne du debut ?</b></p>'
            '<p class="lede">Les deux versions sortent du meme systeme et se ressemblent '
            'beaucoup, y compris dans leurs defauts. Ce qui les separe est ailleurs. Regarde '
            'la personne. <b>Si tu ne peux pas trancher, dis-le</b> plutot que de deviner.</p>')
    cells, key, slots, total = [], [], [], 0
    for n, it in enumerate(items):
        arms = list(it["arms"])
        assert len(arms) == 2, f"duel2 expects 2 arms, {it['id']} has {len(arms)}"
        order = perm(board_id, it["id"], 2)
        cols = []
        placed = {}
        for pos, src in enumerate(order):
            side = "G" if pos == 0 else "D"
            tag, _ = _media_tag(it["arms"][arms[src]], "video", "m")
            cols.append(f'<div class="col"><div class="cap">{side}</div>{tag}</div>')
            placed[side.lower()] = arms[src]
        opts = "".join(
            f'<button class="opt" data-slot="{total}" data-val="{v}">{lbl}</button>'
            for v, lbl in (("g", "G, a gauche"), ("d", "D, a droite"),
                           ("?", "impossible a dire")))
        cells.append(
            f'<div class="cell"><div class="num">{n+1:02d}</div>'
            f'<div class="q">lequel des deux retours est bien la personne du debut ?</div>'
            f'<div class="row">{"".join(cols)}</div>'
            f'<div class="opts">{opts}</div></div>')
        key.append({"id": it["id"], "slot": total, "sides": placed,
                    "meta": it.get("meta", {})})
        slots.append({"slot": total, "item": it["id"], "answers": ["g", "d", "?"]})
        total += 1
    return _finish(out_dir, board_id, title, lede, cells, slots, key, "duel2", total)


# ----------------------------- demo -------------------------------------------------------------
def _demo(out):
    """Builds one board of each type from synthetic PNGs: exercises the whole chain without a
    single rollout, and serves as a visual reference."""
    import struct
    import zlib

    def png(path, rgb, w=48, h=32):
        raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
        def chunk(t, d):
            c = t + d
            return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(raw))
                    + chunk(b"IEND", b""))
        return path

    os.makedirs(out, exist_ok=True)
    m = os.path.join(out, "media")
    os.makedirs(m, exist_ok=True)
    pa = png(os.path.join(m, "a.png"), (180, 90, 70))
    pb = png(os.path.join(m, "b.png"), (70, 120, 180))
    pc = png(os.path.join(m, "c.png"), (120, 160, 100))
    r1 = build_pair2([{"id": f"s{i:02d}", "probe": pa, "true": pb, "twin": pc,
                       "meta": {"source": "decoder" if i % 2 else "blur24"}}
                      for i in range(4)], out)
    r2 = build_rank3([{"id": f"s{i:02d}", "truth": pa,
                       "variants": {"v1_ancien": pb, "v2_mixer": pc, "v2_mixer_sharp": pa}}
                      for i in range(3)], out)
    r3 = build_duel2([{"id": f"s{i:02d}", "arms": {"memon": pb, "memoff": pc}}
                      for i in range(3)], out)
    for r in (r1, r2, r3):
        print(f"  {os.path.basename(r['html'])}  {r['n_slots']} slots  "
              f"{r['bytes']/1024:.0f} KB  key -> {os.path.basename(r['key'])}")
    return r1, r2, r3


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="blind judging board generator")
    ap.add_argument("cmd", choices=["demo"])
    ap.add_argument("--out", default="/tmp/eyeboards_demo")
    a = ap.parse_args()
    _demo(a.out)
