#!/usr/bin/env python
# ============================================================================================
# Le cache d'embeddings de prompt, verifie -- CPU, sans encodeur reel (11.4 Go), ~1 s.
#
# LE test est P3 : un prompt relu du cache doit rendre le tenseur AU BIT PRES. Ce depot compare
# des effets de la taille du bruit de chaine (±2 paires sur 19, mesure du 07/08) ; un arrondi
# introduit par le cache serait invisible et fausserait tout. Aucune quantification n'est donc
# permise ici, meme "inoffensive".
#
#   python tests/test_prompt_cache.py
# ============================================================================================
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import prompt_cache as pc  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


TMP = tempfile.mkdtemp(prefix="promptcache_")
try:
    A = "a woman in a red coat walks along a canal at dusk"
    B = "a baker kneading dough in a warm bakery"

    # --- P1 : la cle est stable et distingue ce qui doit l'etre -------------------------------
    check("P1 cle stable", pc.key_of(A) == pc.key_of(A))
    # LE piege que la cle doit couvrir : la passe 1 et la passe 2 doivent travailler dans la
    # meme precision, parce que inference.py convertit tout le pipeline en bfloat16. Un cache
    # float32 servi a un modele bfloat16 donne un conditionnement faux, sans erreur.
    check("P1d le dtype fait partie de la cle",
          pc.key_of(A, torch.float32) != pc.key_of(A, torch.bfloat16))
    check("P1b prompts differents -> cles differentes", pc.key_of(A) != pc.key_of(B))
    check("P1c une espace change la cle (pas de normalisation silencieuse)",
          pc.key_of(A) != pc.key_of(A + " "))

    # --- P2 : absent avant, present apres ------------------------------------------------------
    check("P2 absent au depart", not pc.has(A, TMP, torch.float32)
          and pc.get(A, TMP, torch.float32) is None)
    emb = torch.randn(512, 4096, dtype=torch.float32)
    pc.put(A, emb, TMP)
    check("P2b present apres ecriture", pc.has(A, TMP, torch.float32))

    # --- P3 : LE test -- relecture bit a bit ---------------------------------------------------
    back = pc.get(A, TMP, torch.float32)
    check("P3 relecture identique AU BIT PRES", torch.equal(back, emb),
          f"max|diff| = {float((back-emb).abs().max())}")
    check("P3b dtype preserve", back.dtype == emb.dtype)
    check("P3c forme preservee", back.shape == emb.shape)
    for dt in (torch.float16, torch.bfloat16):
        e2 = torch.randn(8, 16).to(dt)
        pc.put("x" + str(dt), e2, TMP)
        check(f"P3d {str(dt).split('.')[-1]} aussi bit a bit",
              torch.equal(pc.get("x" + str(dt), TMP, dt), e2))

    # --- P4 : get_many rend l'ordre d'entree et les manquants ---------------------------------
    # Catches l'erreur qui ne leverait aucune exception : associer un prompt au conditionnement
    # d'un autre parce que les lignes ont ete melangees.
    # un embedding bf16 ne doit PAS repondre a une demande float32
    pc.put(A, emb.to(torch.bfloat16), TMP)
    check("P1e bf16 et f32 cohabitent sans se confondre",
          pc.get(A, TMP, torch.float32).dtype is torch.float32
          and pc.get(A, TMP, torch.bfloat16).dtype is torch.bfloat16)

    got, missing = pc.get_many([B, A, B], TMP, torch.float32)
    check("P4 les manquants sont reperes", missing == [0, 2], str(missing))
    check("P4b le present est a sa place", torch.equal(got[1], emb))
    check("P4c les absents sont None", got[0] is None and got[2] is None)

    # --- P5 : un cache incoherent est une erreur bruyante, pas un mauvais conditionnement ------
    p = pc.path_of(B, TMP, torch.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"prompt": "un autre texte", "embed": torch.zeros(4)}, p)
    try:
        pc.get(B, TMP, torch.float32)
        raise SystemExit("FAIL P5 : un cache contenant un autre prompt a ete accepte")
    except RuntimeError:
        check("P5 cache incoherent refuse bruyamment", True)

    # --- P6 : un fichier abime se re-encode au lieu de planter ---------------------------------
    with open(p, "wb") as f:
        f.write(b"pas un tenseur")
    check("P6 fichier abime -> None (on re-encode)", pc.get(B, TMP, torch.float32) is None)

    # --- P7 : l'ecriture est atomique (aucun .tmp ne survit) -----------------------------------
    leftovers = [f for _, _, fs in os.walk(TMP) for f in fs if ".tmp" in f]
    check("P7 aucun fichier temporaire laisse", not leftovers, str(leftovers))

    st = pc.stats(TMP)
    check("P8 les statistiques comptent les entrees", st["n"] >= 3 and st["bytes"] > 0, str(st))

    print(f"\n{len(OK)}/{len(OK)} tests cache de prompts PASS")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
