#!/usr/bin/env python
"""Nettoyage wandb (projet anchor-forcing-memory): supprime les runs inutiles + renumérote le reste.
Lance-le TOI (permission wandb) :  ! /shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python reference/wandb_cleanup.py
Vérifie les noms ci-dessous AVANT de lancer. Suppression = irréversible."""
import wandb
api = wandb.Api()
ent = api.default_entity

DELETE = {"out", "out_vis", "e1_rerun_crosschunk", "e1_rerun_clean"}      # les 4 inutiles

RENAME = {                                                               # renumérotage 08 -> 11
    "e1_rerun2":                 "08_clean-run → DC artifact, inert",
    "fable_darkprobe_eval1000":  "09_dc-probe → DC bias measured",
    "e2_amp":                    "10_amplitude → dissolves",
    "e2_amp2_streamgate":        "11_envelope → freezes",
}

for r in list(api.runs(f"{ent}/anchor-forcing-memory")):
    if r.name in DELETE:
        print("DELETE ", r.name); r.delete()
    elif r.name in RENAME:
        new = RENAME[r.name]; print("RENAME ", r.name, "->", new)
        r.name = new; r.update()
print("--- done. remaining (chronological): ---")
for r in api.runs(f"{ent}/anchor-forcing-memory", order="+created_at"):
    print("  ", r.name)
