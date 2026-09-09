# ============================================================================================
# CACHE DISQUE DES EMBEDDINGS DE PROMPT -- la passe 1 de la separation en deux passes.
#
# POURQUOI (2026-09-09). L'encodeur de texte umT5-XXL pese 11.4 Go, le generateur avec son
# LoRA 3.6 Go, le cache KV 3.2 Go. Sur une machine a memoire unifiee de 24 Go partagee avec le
# systeme, les tenir ENSEMBLE ne rentre pas : mesure faite, le systeme a tue le processus
# (SIGKILL) apres avoir sature 15.5 Go de fichier d'echange. Le poser sur l'accelerateur
# affame le generateur, le poser en RAM fait deborder le systeme. Il n'y a pas de bon cote.
#
# La sortie est de ne jamais les tenir en meme temps. Un processus encode les prompts et se
# TERMINE -- c'est la seule facon de rendre vraiment la memoire, un dechargement dans le meme
# processus laisse toujours les poids quelque part dans le pool partage. Un second processus
# genere, sans jamais construire l'encodeur.
#
# CE QUE CA NE COUTE PAS. Le programme de prompts est toujours connu d'avance dans ce depot :
# meme le chemin dit "interactif" lit ses scenarios depuis un fichier, il n'existe aucune
# saisie clavier. Le pre-calcul ne retire donc aucune capacite.
#
# GARANTIE D'IDENTITE. On stocke le tenseur EXACTEMENT tel que l'encodeur l'a produit, meme
# dtype, sans quantification. Une lecture du cache rend donc le meme tenseur au bit pres, et
# un run avec cache est indiscernable d'un run sans. C'est non negociable : ce depot compare
# des effets de la taille du bruit de chaine, un arrondi introduit ici serait invisible et
# faux.
#
# CLE = sha256 du texte du prompt. Donc le cache se reutilise d'un run a l'autre et d'une
# campagne a l'autre, et deux prompts identiques ne sont encodes qu'une fois.
# ============================================================================================
import hashlib
import os

import torch

DEFAULT_DIR = os.environ.get("AF_PROMPT_CACHE", "cache/prompt_embeds")


def dtype_tag(dtype) -> str:
    """Nom court et stable d'un dtype torch, pour la cle."""
    return str(dtype).replace("torch.", "")


def key_of(prompt: str, dtype=None) -> str:
    """La cle d'un prompt, DTYPE COMPRIS.

    Le texte est pris VERBATIM (pas de strip, pas de normalisation) : deux prompts qui different
    d'une espace sont deux conditionnements differents.

    Le dtype fait partie de la cle parce que le script d'inference convertit TOUT le pipeline en
    bfloat16, encodeur compris (inference.py). Un embedding encode en float32 puis converti n'est
    PAS celui qu'un encodeur bfloat16 aurait produit : poids et activations different. Les
    melanger donnerait un conditionnement subtilement faux, sans aucune erreur -- exactement le
    genre de bug que ce dossier a deja paye deux campagnes.
    """
    tag = dtype_tag(dtype) if dtype is not None else "float32"
    return hashlib.sha256(f"{tag}|{prompt}".encode("utf-8")).hexdigest()


def path_of(prompt: str, cache_dir: str = None, dtype=None) -> str:
    k = key_of(prompt, dtype)
    d = cache_dir or DEFAULT_DIR
    return os.path.join(d, k[:2], f"{k}.pt")            # 2 niveaux : 256 dossiers max


def has(prompt: str, cache_dir: str = None, dtype=None) -> bool:
    return os.path.exists(path_of(prompt, cache_dir, dtype))


def put(prompt: str, embed: torch.Tensor, cache_dir: str = None) -> str:
    """Ecrit l'embedding d'UN prompt. Ecriture atomique (fichier temporaire + rename) : un
    processus tue en plein ecriture ne doit pas laisser un cache a moitie ecrit qu'une lecture
    ulterieure prendrait pour valide -- c'est exactement le correctif que le dossier avait deja
    du appliquer au hook de dump en aout."""
    p = path_of(prompt, cache_dir, embed.dtype)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"prompt": prompt, "embed": embed.detach().cpu(),
                "dtype": str(embed.dtype), "shape": list(embed.shape)}, tmp)
    os.replace(tmp, p)
    return p


def get(prompt: str, cache_dir: str = None, dtype=None):
    """Rend l'embedding stocke, ou None. Verifie que le prompt enregistre est bien celui
    demande : une collision de sha256 est impossible en pratique, mais un cache copie a la main
    ou un fichier corrompu, non."""
    p = path_of(prompt, cache_dir, dtype)
    if not os.path.exists(p):
        return None
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None                                      # cache abime -> on re-encode
    if obj.get("prompt") != prompt:
        raise RuntimeError(f"cache incoherent : {p} contient un autre prompt que celui demande")
    return obj["embed"]


def get_many(prompts, cache_dir: str = None, dtype=None):
    """Rend (liste d'embeddings ou None, indices manquants)."""
    out = [get(p, cache_dir, dtype) for p in prompts]
    missing = [i for i, e in enumerate(out) if e is None]
    return out, missing


def stats(cache_dir: str = None):
    d = cache_dir or DEFAULT_DIR
    n = tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".pt"):
                n += 1
                tot += os.path.getsize(os.path.join(root, f))
    return {"dir": d, "n": n, "bytes": tot}
