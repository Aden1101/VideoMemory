#!/usr/bin/env python
# ============================================================================================
# ECHELLE DE FLOU FINE -- anonymisation du paquet (CPU + ffmpeg).
#
# 3 scenarios (08, 09, 10 du regime non-ouvrant) x 7 niveaux de nettete de la frame inseree
# (sigma 0 / 1 / 2 / 3 / 4 / 6 / 24 px), lettres A-G PERMUTEES par scenario avec un sel
# independant : l'ordre croissant du flou n'est pas devinable, ni depuis les noms, ni depuis
# les tailles (pleine resolution encodee a DEBIT FIXE 2 passes -- un crf ferait des fichiers
# plus petits pour les flous forts, un canal de fuite d'ordre), ni depuis les mtimes
# (uniformises). Cle dans reference/runs/fidelity_ladder/key_blur.json UNIQUEMENT.
# Sorties : anon_blur/scenarioNN_L.mp4 (+ micro/, meme decoupe que le paquet non-ouvrant),
# planche.html, answer_sheet.txt.
# ============================================================================================
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity2_common as C                                             # noqa: E402

FFMPEG = "/usr/bin/ffmpeg"
MICRO_MAX = 0.29 * 1024 * 1024
FULL_KBPS = 1400                          # debit fixe -> taille identique pour les 7 niveaux
MTIME = 1755000000


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


def build_planche(manifest, out_html):
    rows = []
    for sid in C.BLUR_SIDS:
        n = sid.replace("scenario", "")
        vids = "\n".join(
            f'      <figure><figcaption>{n}-{L}</figcaption>'
            f'<video controls preload="none" src="micro/{sid}_{L}.mp4"></video>'
            f'<div class="dl"><a href="{sid}_{L}.mp4">pleine résolution</a></div></figure>'
            for L in C.BLUR_LETTERS)
        rows.append(f"""
  <section>
    <h2>Scénario {n} — sujet : «&nbsp;{manifest[sid]['prompt']}&nbsp;»</h2>
    <p class="q">{C.PLANCHE_NOTE}
       Pour CHACUNE des 7 vidéos : <b>le sujet est-il bien revenu à la fin ?</b>
       (oui / à peu près / non)</p>
    <div class="grid">
{vids}
    </div>
  </section>""")
    html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Échelle de netteté — planche aveugle</title>
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
<h1>Échelle de netteté — 3 scénarios × 7 vidéos (A–G)</h1>
<p class="q">{C.PLANCHE_NOTE} Une seule question par vidéo : oui / à peu près / non.</p>
{''.join(rows)}
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)


def main():
    manifest = C.load_manifest()
    os.makedirs(C.ANON_BLUR_DIR, exist_ok=True)
    os.makedirs(C.MICRO_BLUR_DIR, exist_ok=True)

    missing = [(sid, a) for sid in C.BLUR_SIDS for a in C.BLUR_LADDER
               if not os.path.exists(C.raw_mp4(sid, a))]
    assert not missing, f"videos brutes manquantes: {missing}"

    key = {}
    for sid in C.BLUR_SIDS:
        perm = C.blur_letter_perm(sid)
        assert sorted(perm.values()) == sorted(C.BLUR_LADDER)
        key[sid] = {"prompt": manifest[sid]["prompt"],
                    "seed_main": manifest[sid]["seed_main"],
                    "letters": perm,
                    "sigmas": {L: C.BLUR_LADDER[a] for L, a in perm.items()}}
        for L in C.BLUR_LETTERS:
            arm = perm[L]
            src = C.raw_mp4(sid, arm)
            full = os.path.join(C.ANON_BLUR_DIR, f"{sid}_{L}.mp4")
            micro = os.path.join(C.MICRO_BLUR_DIR, f"{sid}_{L}.mp4")
            if not os.path.exists(full):
                sz = encode_full_cbr(src, full)
                print(f"[anon_blur] {sid}_{L} full {sz:.1f} MB (debit fixe)", flush=True)
            if not os.path.exists(micro):
                szm, kbps = encode_micro(src, micro)
                print(f"[anon_blur] {sid}_{L} micro {szm:.2f} MB ({kbps} kbps)", flush=True)

    with open(C.KEY_BLUR_JSON, "w") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    print(f"[anon_blur] cle -> {C.KEY_BLUR_JSON} (NE PAS copier pres des videos)")

    build_planche(manifest, os.path.join(C.ANON_BLUR_DIR, "planche.html"))
    lines = ["ECHELLE DE NETTETE -- grille de reponse",
             "Question unique par video : le sujet est-il bien revenu a la fin ?",
             "Reponses possibles : oui / a peu pres / non", ""]
    for sid in C.BLUR_SIDS:
        n = sid.replace("scenario", "")
        lines.append(f"Scenario {n} ({manifest[sid]['prompt']}):")
        lines += [f"  {n}-{L} : ______" for L in C.BLUR_LETTERS]
        lines.append("")
    with open(os.path.join(C.ANON_BLUR_DIR, "answer_sheet.txt"), "w") as f:
        f.write("\n".join(lines))

    for root, _, files in os.walk(C.ANON_BLUR_DIR):
        for fn in files:
            os.utime(os.path.join(root, fn), (MTIME, MTIME))
    import glob
    sizes = [os.path.getsize(p) for p in glob.glob(os.path.join(C.ANON_BLUR_DIR, "*.mp4"))]
    print(f"[anon_blur] paquet pret : {C.ANON_BLUR_DIR} ; tailles pleine-res "
          f"{min(sizes) / 1e6:.1f}-{max(sizes) / 1e6:.1f} MB (uniformes par construction)")


if __name__ == "__main__":
    main()
