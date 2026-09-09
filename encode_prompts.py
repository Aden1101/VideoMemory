#!/usr/bin/env python
# ============================================================================================
# PASSE 1 : encoder les prompts, ecrire le cache, et SE TERMINER.
#
# C'est le point : le processus meurt. Un dechargement dans le meme processus laisse toujours
# les 11.4 Go quelque part dans le pool partage d'une memoire unifiee ; seule la fin du
# processus les rend vraiment au systeme. Ensuite la passe 2 (inference/inference.py avec
# AF_LAZY_TEXT_ENCODER=1) ne construit jamais l'encodeur et garde la memoire pour le
# generateur.
#
#   python encode_prompts.py --file ckpt/prompts/bench/vbench_all_dimension_extended.txt --limit 1
#   python encode_prompts.py --file inference/interactive_case.jsonl        # scenarios a switch
#   python encode_prompts.py --stats                                        # etat du cache
#
# Puis :
#   AF_LAZY_TEXT_ENCODER=1 PYTHONPATH=. python inference/inference.py --config_path ...
#
# Formats acceptes : .txt (un prompt par ligne) et .jsonl (cle "prompts", une liste par ligne,
# le format des scenarios de revisite). Les doublons ne sont encodes qu'une fois.
# ============================================================================================
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_prompts(path):
    if path.endswith(".jsonl"):
        out = []
        with open(path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.extend(obj["prompts"] if isinstance(obj.get("prompts"), list) else [obj])
        return out
    # `.rstrip()` COMPLET, pas seulement le saut de ligne : c'est exactement ce que fait
    # utils.dataset.TextDataset. La cle du cache est le texte verbatim, donc une espace de fin
    # non retiree ici produirait un defaut de cache silencieux a chaque run.
    with open(path, encoding="utf-8") as f:
        return [l.rstrip() for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="passe 1 : cache des embeddings de prompt")
    ap.add_argument("--file", action="append", default=[],
                    help="fichier de prompts (.txt ou .jsonl) ; repetable")
    ap.add_argument("--prompt", action="append", default=[], help="un prompt litteral ; repetable")
    ap.add_argument("--limit", type=int, default=0, help="ne prendre que les N premiers")
    ap.add_argument("--batch", type=int, default=4, help="prompts encodes par lot")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
                    help="precision d'encodage. bfloat16 par DEFAUT parce que inference.py fait "
                         "`pipeline.to(dtype=torch.bfloat16)` : la passe 1 doit encoder dans la "
                         "meme precision que la passe 2, sinon le conditionnement differe.")
    ap.add_argument("--stats", action="store_true", help="afficher l'etat du cache et sortir")
    a = ap.parse_args()

    from utils import prompt_cache

    if a.stats:
        st = prompt_cache.stats(a.cache_dir)
        print(f"cache {st['dir']} : {st['n']} prompts, {st['bytes']/1e6:.0f} Mo")
        return 0

    prompts = list(a.prompt)
    for f in a.file:
        prompts += read_prompts(f)
    if a.limit:
        prompts = prompts[:a.limit]
    if not prompts:
        ap.error("rien a encoder : donne --file ou --prompt")

    # dedoublonnage en CONSERVANT l'ordre (pour que le journal soit lisible)
    seen, uniq = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    import torch                                 # importe ici, apres l'argparse
    want = getattr(torch, a.dtype)
    todo = [p for p in uniq if not prompt_cache.has(p, a.cache_dir, want)]
    print(f"{len(prompts)} prompts lus, {len(uniq)} uniques, {len(uniq)-len(todo)} deja en cache, "
          f"{len(todo)} a encoder", flush=True)
    if not todo:
        st = prompt_cache.stats(a.cache_dir)
        print(f"rien a faire. cache : {st['n']} prompts, {st['bytes']/1e6:.0f} Mo")
        return 0

    from utils.device import describe
    from utils.wan_wrapper import WanTextEncoder

    print(describe(), flush=True)
    t0 = time.time()
    te = WanTextEncoder()                      # chargement AVIDE ici : c'est le but du script
    te = te.to(dtype=want)                     # MEME conversion que inference.py sur le pipeline
    print(f"encodeur charge en {time.time()-t0:.0f} s, precision {a.dtype}", flush=True)

    done, t0 = 0, time.time()
    with torch.no_grad():
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            emb = te.encode(chunk).to(want)    # encode() court-circuite le cache, exprès
            for p, e in zip(chunk, emb):
                prompt_cache.put(p, e, a.cache_dir)
            done += len(chunk)
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  ({el/done:.1f} s/prompt, reste "
                  f"~{el/done*(len(todo)-done)/60:.1f} min)", flush=True)

    st = prompt_cache.stats(a.cache_dir)
    print(f"\ncache {st['dir']} : {st['n']} prompts, {st['bytes']/1e6:.0f} Mo")
    print("passe 1 terminee. La passe 2 se lance avec AF_LAZY_TEXT_ENCODER=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
