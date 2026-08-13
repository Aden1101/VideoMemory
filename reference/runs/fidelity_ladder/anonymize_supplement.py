#!/usr/bin/env python
# ============================================================================================
# SUPPLEMENT (4 nouvelles scenes, regime non-ouvrant) -- anonymisation du paquet.
# 4 scenarios x 6 barreaux, lettres A-F permutees (sel independant), pleine resolution a
# DEBIT FIXE 2 passes (la taille ne peut pas trahir le barreau), micro-clips 2 parties
# (5 s du passage du sujet + le retour), mtimes uniformises, cle dans key_supplement.json
# UNIQUEMENT, planche + grille de reponse.
# ============================================================================================
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity3_common as C                                             # noqa: E402

FFMPEG = "/usr/bin/ffmpeg"
MICRO_MAX = 0.29 * 1024 * 1024
FULL_KBPS = 1400
MTIME = 1755100000


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert r.returncode == 0, f"echec: {' '.join(cmd)}\n{r.stderr.decode()[-2000:]}"


def probe_duration(path):
    r = subprocess.run(["/usr/bin/ffprobe", "-v", "quiet", "-of", "json",
                        "-show_format", path], stdout=subprocess.PIPE)
    return float(json.loads(r.stdout)["format"]["duration"])


def encode_full_cbr(src, dst):
    with tempfile.TemporaryDirectory() as td:
        plog = os.path.join(td, "pass")
        base = [FFMPEG, "-y", "-i", src, "-c:v", "libx264", "-b:v", f"{FULL_KBPS}k",
                "-maxrate", f"{int(FULL_KBPS * 1.3)}k", "-bufsize", f"{FULL_KBPS * 2}k",
                "-preset", "slow", "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1",
                "-passlogfile", plog]
        run(base + ["-pass", "1", "-f", "null", "/dev/null"])
        run(base + ["-pass", "2", "-movflags", "+faststart", dst])
    sz = os.path.getsize(dst) / 1e6
    assert sz < 30.0, f"{dst}: {sz:.1f} MB >= 30 MB"
    return sz


def encode_micro(src, dst):
    dur = probe_duration(src)
    g0, g1 = C.MICRO_GLIMPSE
    d_total = (g1 - g0) + (dur - C.MICRO_RETURN_T)
    fc = (f"[0:v]trim=start={g0}:end={g1},setpts=PTS-STARTPTS,scale=416:240[a];"
          f"[0:v]trim=start={C.MICRO_RETURN_T},setpts=PTS-STARTPTS,scale=416:240[b];"
          f"[a][b]concat=n=2:v=1:a=0[v]")
    kbps = int((MICRO_MAX * 8 * 0.94) / d_total / 1000)
    for _ in range(4):
        with tempfile.TemporaryDirectory() as td:
            plog = os.path.join(td, "pass")
            base = [FFMPEG, "-y", "-i", src, "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.4)}k",
                    "-bufsize", f"{kbps * 2}k", "-preset", "veryslow", "-pix_fmt", "yuv420p",
                    "-an", "-map_metadata", "-1", "-passlogfile", plog]
            run(base + ["-pass", "1", "-f", "null", "/dev/null"])
            run(base + ["-pass", "2", "-movflags", "+faststart", dst])
        if os.path.getsize(dst) <= MICRO_MAX:
            return os.path.getsize(dst) / 1e6, kbps
        kbps = int(kbps * 0.85)
    raise AssertionError(f"{dst}: > {MICRO_MAX / 1e6:.2f} MB apres 4 essais")


def main():
    manifest = C.load_manifest()
    os.makedirs(C.ANON_DIR, exist_ok=True)
    os.makedirs(C.MICRO_DIR, exist_ok=True)
    missing = [(sc["sid"], a) for sc in C.SCENARIOS for a in C.ARMS
               if not os.path.exists(C.raw_mp4(sc["sid"], a))]
    assert not missing, f"videos brutes manquantes: {missing}"

    key = {}
    for sc in C.SCENARIOS:
        sid = sc["sid"]
        perm = C.letter_perm(sid)
        key[sid] = {"board_no": sc["board_no"], "cidx": sc["cidx"],
                    "prompt": sc["prompt"], "interlude": sc["interlude"],
                    "opener": sc["opener"], "seed_main": manifest[sid]["seed_main"],
                    "donor_sid": manifest[sid]["donor_sid"], "letters": perm}
        for L in C.LETTERS:
            arm = perm[L]
            src = C.raw_mp4(sid, arm)
            full = os.path.join(C.ANON_DIR, f"{sid}_{L}.mp4")
            micro = os.path.join(C.MICRO_DIR, f"{sid}_{L}.mp4")
            if not os.path.exists(full):
                sz = encode_full_cbr(src, full)
                print(f"[anon_sup] {sid}_{L} full {sz:.1f} MB (debit fixe)", flush=True)
            if not os.path.exists(micro):
                szm, kbps = encode_micro(src, micro)
                print(f"[anon_sup] {sid}_{L} micro {szm:.2f} MB ({kbps} kbps)", flush=True)

    with open(C.KEY_JSON, "w") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    print(f"[anon_sup] cle -> {C.KEY_JSON} (NE PAS copier pres des videos)")

    rows = []
    for sc in C.SCENARIOS:
        sid = sc["sid"]
        n = sid.replace("scenario", "")
        vids = "\n".join(
            f'      <figure><figcaption>{n}-{L}</figcaption>'
            f'<video controls preload="none" src="micro/{sid}_{L}.mp4"></video>'
            f'<div class="dl"><a href="{sid}_{L}.mp4">pleine résolution</a></div></figure>'
            for L in C.LETTERS)
        rows.append(f"""
  <section>
    <h2>Scénario {n} — sujet : «&nbsp;{sc['prompt']}&nbsp;»</h2>
    <p class="q">{C.PLANCHE_NOTE}
       Pour CHACUNE des 6 vidéos : <b>le sujet est-il bien revenu à la fin ?</b>
       (oui / à peu près / non)</p>
    <div class="grid">
{vids}
    </div>
  </section>""")
    html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{C.PLANCHE_TITLE}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 1rem; background:#111; color:#eee; }}
 h1 {{ font-size: 1.2rem; }} h2 {{ font-size: 1rem; margin: 1.2rem 0 .3rem; }}
 .q {{ color:#bbb; font-size:.9rem; margin:.2rem 0 .6rem; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:.6rem; }}
 figure {{ margin:0; }} figcaption {{ font-weight:bold; margin-bottom:.2rem; }}
 video {{ width:100%; background:#000; }}
 .dl {{ font-size:.8rem; }} .dl a {{ color:#8ab4f8; }}
 section {{ border-top:1px solid #333; padding-top:.5rem; }}
</style></head><body>
<h1>Scènes supplémentaires — 4 scénarios × 6 vidéos (A–F)</h1>
<p class="q">{C.PLANCHE_NOTE} Une seule question par vidéo : oui / à peu près / non.</p>
{''.join(rows)}
</body></html>"""
    with open(os.path.join(C.ANON_DIR, "planche.html"), "w") as f:
        f.write(html)
    lines = [f"{C.PLANCHE_TITLE} -- grille de reponse",
             "Question unique par video : le sujet est-il bien revenu a la fin ?",
             "Reponses possibles : oui / a peu pres / non", ""]
    for sc in C.SCENARIOS:
        n = sc["sid"].replace("scenario", "")
        lines.append(f"Scenario {n} ({sc['prompt']}):")
        lines += [f"  {n}-{L} : ______" for L in C.LETTERS]
        lines.append("")
    with open(os.path.join(C.ANON_DIR, "answer_sheet.txt"), "w") as f:
        f.write("\n".join(lines))

    for root, _, files in os.walk(C.ANON_DIR):
        for fn in files:
            os.utime(os.path.join(root, fn), (MTIME, MTIME))
    import glob
    sizes = [os.path.getsize(p) for p in glob.glob(os.path.join(C.ANON_DIR, "*.mp4"))]
    print(f"[anon_sup] paquet pret : {C.ANON_DIR} ; tailles pleine-res "
          f"{min(sizes) / 1e6:.1f}-{max(sizes) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
