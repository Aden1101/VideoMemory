#!/usr/bin/env python
# ============================================================================================
# TOY D'ECRITURE-IDENTITE (amendement PI, 2026-08-05) -- seuils SCELLES avant implementation
# dans memory2_probes_registry.md, section "TOY D'ECRITURE-IDENTITE".
#
# QUESTION. Le sondage (probe1_report.md) a prouve DEUX choses : (a) l'identite EST lisible
# depuis les activations masquees-sujet (16/19 paires fortes, p = 0.011 : l'instrument marche),
# (b) elle N'EST PAS lisible depuis l'etat des poids rapides tel qu'il s'ecrit aujourd'hui
# (0.467 par paire, p = 0.82 : hasard). L'arret scelle condamnait l'EXTRACTION. Il ne jugeait
# pas un RE-ENTRAINEMENT DE L'ECRITURE : aucun ecrivain n'a jamais eu d'objectif d'identite.
# Ce fichier pose la question minimale : en SUR-APPRENTISSAGE sur 8 paires, l'ecriture
# peut-elle apprendre a rendre l'identite lisible ?
#
# CE QUI EST CO-ENTRAINE.
#   (a) l'ECRIVAIN theta_w = qkv/lrs/k_norm/decay_logit des 3 cellules (blocs 5/16/27),
#       warm-start l1@2000-EMA par `train_writer.build_cells` (contrat F2.3 : delta_l1 == 0 au
#       step 0), rejeu differentiable `replay_writes` en BPTT COMPLET (les 25 ecritures) ;
#   (b) la TETE DE LECTURE = LA signature du sondage, importee telle quelle :
#       sondes fixes (`probe1/sig/probes.pt`) -> `build_signatures.read_state` (swiglu, la
#       lecture du decodeur) -> `build_signatures.compress_signature` (moyenne+max par sonde,
#       3 x 256 x 2 = 1536 nombres) -> `train_probe.ProbeMLP` (MLP court -> espace DINO).
#   Rien n'est recopie : les quatre briques sont IMPORTEES de reference/runs/probe1/ et de
#   reference/runs/mem2/, et un selftest verifie que le chemin differentiable rend BIT A BIT la
#   meme signature que `build_signatures.signature` (le chemin du verdict du sondage).
#
# L'OBJECTIF. InfoNCE : l'embedding lu depuis l'etat (fin de sequence + 2 ages intermediaires)
# doit etre plus proche des embeddings DINO du crop-SUJET de SA video que de ceux du JUMEAU
# (meme prompt, autre graine -- force dans chaque softmax par construction : les 16 videos des
# 8 paires sont les 16 classes) et des 7 autres paires. La reconstruction (norm_mse de la frame
# decodee par le decodeur v1.2 GELE) reste en REGULARISEUR (poids 0.1) : l'ecrivain doit rester
# une memoire, pas devenir un hash d'identite.
#
# LES 4 CRITERES SCELLES (mesures mecaniquement en fin de run, verdict PASS/FAIL sec) :
#   (1) precision PAR PAIRE sur les 8 paires (sur-apprentissage autorise) >= 7/8 ;
#   (2) controle etats-permutes <= 5/8 (25 derangements, regle du sondage) ;
#   (3) le MEME entrainement avec ecrivain GELE (tete seule) fait NETTEMENT pire ;
#   (4) la reco sur les 8 videos ne se degrade pas de plus de 2x.
# ECHEC => fin definitive du chantier ecriture.
#
# Usage (pre-check GPU DANS la commande de lancement, regle de la maison) :
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/toy_identity/toy_identity.py --tag trained
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/toy_identity/toy_identity.py --tag frozen \
#       --freeze_writer
#   python reference/runs/toy_identity/toy_identity.py --stage verdict     # les 4 criteres
# ============================================================================================
import argparse
import json
import os
import random
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE1_DIR = os.path.join(REPO, "reference/runs/probe1")
MEM2_DIR = os.path.join(REPO, "reference/runs/mem2")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, PROBE1_DIR, MEM2_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
import probe1_common as C  # noqa: E402
from build_probes import PROBES_PT  # noqa: E402
from build_signatures import compress_signature, read_state  # noqa: E402
from train_probe import ProbeMLP, load_refs  # noqa: E402
from train_writer import (WRITE_PARAMS, build_cells, delta_l1,  # noqa: E402
                          get_write_params, norm_mse_per, ref_theta_l1)
from wan.modules.video_memory import AnchorFrameDecoder, replay_writes  # noqa: E402

HEAVY = "/shared/ssd_14T_project/adenelmi/toy_identity"
N_PAIRS = 8                     # les 8 PREMIERES paires FORTES du corpus v2 (registre)
N_AGES_READ = 3                 # "fin de sequence + 2 ages intermediaires"
RECO_AGES = (1, 10)             # les deux ages de la reco temoin (regulariseur + critere 4)


# ---------------------------------------------------------------------------------------------
# donnees : les 8 paires FORTES du corpus v2, leurs h, leurs references DINO
# ---------------------------------------------------------------------------------------------
def select_pairs(n_pairs=N_PAIRS):
    """Les `n_pairs` premieres paires de la liste 'CLAIREMENT DIFFERENTS' du corpus v2, scellee
    au registre (LABELS OEIL D'ADEN, CORPUS v2) et importee depuis probe1_common
    (ADEN_V2_STRONG_19 = {5,6,9,11,12,13,14,15,16,...} en numeros de planche). Toutes sont
    valides par construction (fortes ⊂ valides), ce que l'on asserte."""
    vids, pairs = C.pair_dataset(corpus="v2", verbose=False)
    sub = C.sealed_subsets(pairs)                     # verrou : confronte au gel horodate
    strong = {p["corpus_idx"] for p in pairs if p["board_no"] in C.ADEN_V2_STRONG_19}
    assert strong == sub["strong19"], "liste FORTE recalculee != sous-ensemble gele"
    assert strong <= sub["valid25"], "une paire forte n'est pas dans les 25 valides"
    keep = sorted([p for p in pairs if p["corpus_idx"] in strong],
                  key=lambda p: p["board_no"])[:n_pairs]
    assert len(keep) == n_pairs, f"{len(keep)} paires fortes, il en faut {n_pairs}"
    return C.vids_by_id(vids), keep


class ToyData:
    """Les 16 videos (8 paires x {main, twin}) : leurs h en RAM CPU (fp16), les frames-cibles de
    la reco, la condition de prompt, et les references DINO du crop-sujet.

    Les 16 videos sont a la fois le jeu d'entrainement ET le jeu d'evaluation : le protocole
    scelle demande un SUR-APPRENTISSAGE (aucun held-out ici, c'est le point du toy). Le selftest
    'split trivial' l'asserte plutot que de le sous-entendre."""

    def __init__(self, blocks, device, n_pairs=N_PAIRS, n_ages=N_AGES_READ, verbose=True):
        self.blocks, self.device = blocks, device
        byid, pairs = select_pairs(n_pairs)
        self.pairs = pairs
        self.metas, self.refs, self.of_pair, self.twin_index = [], [], [], []
        for p in pairs:
            base = len(self.metas)
            for off, role in enumerate(("main", "twin")):
                m = byid[p[role]]
                r = load_refs(m["video_id"], "subject", seg="A")      # (n_valid, 768) L2, probe1
                assert r is not None and r.shape[0] > 0, f"pas de reference pour {p[role]}"
                self.metas.append(m)
                self.refs.append(r.to(device))
                self.of_pair.append(p["board_no"])
                self.twin_index.append(base + (1 - off))
        self.ages = {}          # video_id -> [indices d'ecriture lus par la tete]
        self.reco_pos = {}      # video_id -> [(state_i, target_frame, age)]
        for m in self.metas:
            ags = C.state_ages(m, C.N_AGES)
            assert ags[-1]["name"] == "end_seq"
            sel = ags[-n_ages:]
            self.ages[m["video_id"]] = [int(a["i"]) for a in sel]
            pos = []
            for i in self.ages[m["video_id"]]:
                for a in RECO_AGES:
                    j = i - (a - 1)
                    if j < 0:
                        continue
                    pos.append({"i": int(i), "j": int(j), "age": int(a),
                                "frame": int(D.frame_of_write(m, j))})
            self.reco_pos[m["video_id"]] = pos
        self.i_max = max(max(v) for v in self.ages.values())
        # ---- h, frames-cibles et conditions en RAM CPU (les shards ne sont plus touches) ----
        cache = D.PartCache(size=4)
        self.h, self.frames, self.cond = {}, {}, {}
        t0 = time.time()
        for m in self.metas:
            vid = m["video_id"]
            hs = D.load_h(m, cache, self.blocks, upto=self.i_max, device=None,
                          dtype=torch.float16)
            self.h[vid] = {b: [t.clone() for t in hs[b]] for b in self.blocks}
            self.frames[vid] = D.load_frames(m, cache, [p["frame"] for p in self.reco_pos[vid]])
            ca, _ = D.load_conds(m, cache)
            self.cond[vid] = ca
        cache.clear()
        nb = sum(t.numel() * t.element_size() for v in self.h.values() for s in v.values()
                 for t in s)
        if verbose:
            print(f"[toy] {len(self.pairs)} paires (planche v2 {[p['board_no'] for p in pairs]}), "
                  f"{len(self.metas)} videos, ages lus {self.ages[self.metas[0]['video_id']]}, "
                  f"{nb / 1e9:.1f} GB de h en RAM ({time.time() - t0:.0f} s)", flush=True)

    def n_videos(self):
        return len(self.metas)

    def h_gpu(self, vid, dtype=torch.bfloat16):
        """Les h d'une video sur le GPU, dans le dtype de DEPLOIEMENT (bf16) : le cast fp16 ->
        bf16 se fait AVANT le rejeu, jamais dedans (contrat de `replay_writes`)."""
        return {b: [t.to(self.device, dtype) for t in self.h[vid][b][:self.i_max + 1]]
                for b in self.blocks}

    def ref_bank(self, rng):
        """(V, 768) : UNE image de reference tiree par video et par pas (le sujet bouge d'une
        frame a l'autre : c'est l'augmentation naturelle de la tache, idiome `_ref_bank` du
        sondeur). Les 16 lignes = les 16 classes du softmax : le jumeau y est TOUJOURS."""
        return torch.stack([r[rng.randrange(r.shape[0])] for r in self.refs], 0)


# ---------------------------------------------------------------------------------------------
# le trainer
# ---------------------------------------------------------------------------------------------
class ToyTrainer:
    def __init__(self, args, data=None, verbose=True):
        self.args = args
        self.device = torch.device(args.device)
        self.rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        C.set_determinism(args.seed)

        self.config = OmegaConf.load(args.config)
        self.config.distributed = False
        # 'direct' = la cellule reconstruite depuis le ckpt (leger, pas de 1.3B ni de T5) ;
        # build_states.py l'utilise deja pour les etats du sondage, et le contrat F2.3 ci-dessous
        # interdit toute divergence silencieuse.
        self.cells, self.blocks, _ = build_cells(self.config, self.device, mode="direct")
        self.ref, ref_src = ref_theta_l1(self.config, self.blocks, self.device)
        d0 = delta_l1(self.cells, self.ref)
        assert d0 == 0.0, f"F2.3 : theta_w != theta_l1 au step 0 (delta_l1 = {d0:.3e})"
        if verbose:
            print(f"[toy] cellules {self.blocks} depuis {ref_src} de {self.config.lora_ckpt}, "
                  f"delta_l1 = {d0}", flush=True)

        # theta_w entrainable (feuilles bf16) + maitres fp32 : recette train_writer (design 3.5,
        # le forward reste dans le dtype de deploiement, donc la fonction entrainee EST la
        # fonction deployee).
        self.wparams, self.masters = {}, {}
        for bi, cell in self.cells.items():
            for p in cell.parameters():
                p.requires_grad_(False)
            for name, p in get_write_params(cell).items():
                p.requires_grad_(not args.freeze_writer)
                self.wparams[(bi, name)] = p
                m = p.detach().float().clone().requires_grad_(not args.freeze_writer)
                self.masters[(bi, name)] = m
        n_w = sum(p.numel() for p in self.wparams.values())

        # ---- la tete de lecture : LA signature du sondage ------------------------------------
        self.probes, self.head = None, None
        if args.lam_id > 0:
            pk = torch.load(args.probes, map_location="cpu", weights_only=False)
            assert list(pk["blocks"]) == self.blocks, \
                f"batterie de sondes sur {pk['blocks']} != blocs memoire {self.blocks}"
            self.probes = pk["probes"].to(self.device)
            self.d_feat = self.probes.shape[0] * self.probes.shape[1] * 2   # 3 x 256 x 2
            self.head = ProbeMLP(self.d_feat, args.hidden, args.depth, C.DINO_DIM,
                                 args.dropout).to(self.device)
            if verbose:
                print(f"[toy] sondes {tuple(self.probes.shape)} ({pk['n_centroids']} centroides "
                      f"+ {pk['n_queries']} requetes), tete {self.d_feat} -> {C.DINO_DIM} : "
                      f"{sum(p.numel() for p in self.head.parameters()) / 1e6:.2f}M", flush=True)

        # ---- le decodeur v1.2, GELE : il est la MESURE de reco, pas un parametre -------------
        # detach_states=False (F3.3) : sans cela le regulariseur de reco n'aurait AUCUN gradient
        # vers l'ecrivain et ne regulariserait rien.
        self.decoder = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512,
                                          detach_states=False).to(self.device)
        ck = torch.load(args.dec_ckpt, map_location="cpu", weights_only=False)
        sd = ck["decoder"] if isinstance(ck, dict) and "decoder" in ck else ck
        missing, unexpected = self.decoder.load_state_dict(sd, strict=True)
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        if verbose:
            print(f"[toy] decodeur GELE depuis {args.dec_ckpt} "
                  f"(step {ck.get('step', '?') if isinstance(ck, dict) else '?'})", flush=True)

        self.data = data if data is not None else ToyData(self.blocks, self.device,
                                                          n_pairs=args.n_pairs,
                                                          verbose=verbose)
        groups = []
        if not args.freeze_writer:
            groups.append({"params": list(self.masters.values()), "lr": args.lr_w,
                           "weight_decay": 0.0})
        if self.head is not None:
            groups.append({"params": list(self.head.parameters()), "lr": args.lr_h,
                           "weight_decay": args.wd_h})
        assert groups, "rien a entrainer (--freeze_writer avec --lam_id 0)"
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: (min(1.0, (s + 1) / max(1, args.warmup))
                                 * (args.min_lr_ratio + (1 - args.min_lr_ratio)
                                    * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, args.steps)))))))
        self.mu = self.sd = None
        self.peak_vram = 0.0
        self.step = 0
        if verbose:
            print(f"[toy] theta_w : {n_w / 1e6:.2f}M parametres sur {self.blocks} "
                  f"({'GELES' if args.freeze_writer else 'ENTRAINES'})", flush=True)

    # ---------- etats -----------------------------------------------------------------------
    def states_of(self, meta, grad=True):
        """Rejoue TOUTES les ecritures jusqu'au dernier age lu et rend la liste des etats par
        bloc. BPTT complet quand l'ecrivain est entraine (n_bptt=None) ; quand il est gele, le
        graphe est coupe (n_bptt=0) -- le FORWARD de replay_writes est inchange, donc les etats
        sont bit a bit ceux du rejeu complet."""
        h = self.data.h_gpu(meta["video_id"])
        n_bptt = 0 if (self.args.freeze_writer or not grad) else self.args.n_bptt
        states = {}
        for bi in self.blocks:
            _, all_st = replay_writes(self.cells[bi], h[bi], detach_state=False,
                                      n_bptt_writes=n_bptt, use_checkpoint=True,
                                      return_all=True)
            states[bi] = all_st
        del h
        return states

    # ---------- la signature (chemin DIFFERENTIABLE, bit a bit celui du sondage) --------------
    def signature_of(self, states, i):
        """L'etat a l'ecriture i -> (3 x 256 x 2 = 1536,). `read_state` et `compress_signature`
        sont IMPORTES de build_signatures : c'est litteralement le chemin du verdict du sondage,
        sans le @torch.no_grad() du script de collecte (selftest T2 : egalite bit a bit)."""
        sig = torch.stack([read_state(self.probes[k], states[bi][i])
                           for k, bi in enumerate(self.blocks)], 0)
        return compress_signature(sig).reshape(-1)

    def features_of(self, meta, states):
        return torch.stack([self.signature_of(states, i)
                            for i in self.data.ages[meta["video_id"]]], 0)

    @torch.no_grad()
    def fit_norm(self):
        """Standardisation d'entree de la tete, calculee UNE FOIS sur les 48 lignes de depart
        (16 videos x 3 ages, ecrivain l1) puis GELEE. Ce n'est pas un blanchiment ajuste sur un
        train (il n'y a pas de held-out ici) : c'est un preconditionneur fixe, pour que la tete
        ne parte pas dans une echelle absurde. Il ne peut pas fuir : il ne voit aucune etiquette."""
        rows = []
        for meta in self.data.metas:
            st = self.states_of(meta, grad=False)
            rows.append(self.features_of(meta, st).float())
            del st
        X = torch.cat(rows, 0)
        self.mu = X.mean(0)
        self.sd = X.std(0).clamp_min(1e-5)
        return X

    def embed(self, feats):
        return self.head((feats.float() - self.mu) / self.sd)

    # ---------- la reco (regulariseur + critere 4) --------------------------------------------
    def reco(self, meta, states):
        pos = self.data.reco_pos[meta["video_id"]]
        dec = [tuple(torch.cat([states[bi][p["i"]][k] for p in pos], 0) for k in range(3))
               for bi in self.blocks]
        cond = self.data.cond[meta["video_id"]][None].expand(len(pos), -1).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not self.args.no_amp):
            pred = self.decoder(dec, cond=cond)
        tgt = self.data.frames[meta["video_id"]].to(self.device).unsqueeze(1)
        return norm_mse_per(pred, tgt).mean()

    # ---------- un pas ------------------------------------------------------------------------
    def train_step(self, videos):
        a = self.args
        self.sync_masters_to_cells()
        self.opt.zero_grad(set_to_none=True)
        E = self.data.ref_bank(self.rng) if self.head is not None else None
        agg = {"loss": 0.0, "id": 0.0, "rec": 0.0, "acc": 0.0, "n": 0}
        for vi in videos:
            meta = self.data.metas[vi]
            states = self.states_of(meta)
            loss = torch.zeros((), device=self.device)
            if self.head is not None:
                z = self.embed(self.features_of(meta, states))            # (n_ages, 768)
                logits = (z @ E.transpose(0, 1)) / a.temp
                y = torch.full((z.shape[0],), vi, dtype=torch.long, device=self.device)
                li = Fn.cross_entropy(logits, y)
                loss = loss + a.lam_id * li
                agg["id"] += float(li)
                agg["acc"] += float((logits.argmax(-1) == y).float().mean())
            if a.rec_w > 0:
                lr_ = self.reco(meta, states)
                loss = loss + a.rec_w * lr_
                agg["rec"] += float(lr_)
            (loss / len(videos)).backward()
            agg["loss"] += float(loss)
            agg["n"] += 1
            del states, loss
        wgn = 0.0
        if not a.freeze_writer:
            wgn = self.gather_grads()
            torch.nn.utils.clip_grad_norm_(list(self.masters.values()), a.clip)
        hgn = 0.0
        if self.head is not None:
            hgn = float(torch.nn.utils.clip_grad_norm_(list(self.head.parameters()), a.clip))
        self.opt.step()
        self.sched.step()
        self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated() / 1e9)
        n = max(1, agg["n"])
        return {"train/loss": agg["loss"] / n, "train/id": agg["id"] / n,
                "train/rec": agg["rec"] / n, "train/acc_row": agg["acc"] / n,
                "writer/grad_norm": wgn, "head/grad_norm": hgn,
                "writer/delta_l1": delta_l1(self.cells, self.ref),
                "train/lr": self.sched.get_last_lr()[-1], "vram/peak_gb": self.peak_vram}

    def sync_masters_to_cells(self):
        with torch.no_grad():
            for k, m in self.masters.items():
                self.wparams[k].copy_(m.to(self.wparams[k].dtype))

    def gather_grads(self):
        gn = 0.0
        for k, m in self.masters.items():
            g = self.wparams[k].grad
            m.grad = None if g is None else g.detach().float().clone()
            if m.grad is not None:
                gn += float(m.grad.pow(2).sum())
            self.wparams[k].grad = None
        return gn ** 0.5

    # ---------- evaluation : precision PAR PAIRE + controle permute + reco --------------------
    @torch.no_grad()
    def evaluate(self, n_perm=25, seed=1234, with_reco=True):
        """Chaque ligne (video v, age a) doit etre plus proche des references de v que de celles
        de son JUMEAU : c'est la lecture PAR PAIRE du sondage (`train_probe.evaluate`), reprise
        a l'identique -- marge = cos moyen aux references propres moins cos moyen a celles du
        jumeau, moyennee sur les ages puis sur les deux videos de la paire."""
        self.sync_masters_to_cells()
        if self.head is not None:
            self.head.eval()
        P, rows_vid, recos = [], [], []
        for vi, meta in enumerate(self.data.metas):
            states = self.states_of(meta, grad=False)
            if self.head is not None:
                z = self.embed(self.features_of(meta, states))
                P.append(z)
                rows_vid += [vi] * z.shape[0]
            if with_reco:
                recos.append(float(self.reco(meta, states)))
            del states
        out = {"reco_mean": float(np.mean(recos)) if recos else float("nan"),
               "reco_by_video": recos}
        if self.head is not None:
            P = torch.cat(P, 0)
            out.update(self.score(P, rows_vid))
            perm = [self.score(P, self.derangement(rows_vid, seed + k))
                    for k in range(int(n_perm))]
            out["permuted"] = {
                "acc_pair": float(np.mean([p["acc_pair"] for p in perm])),
                "n_pairs_correct": float(np.mean([p["n_pairs_correct"] for p in perm])),
                "acc_row": float(np.mean([p["acc_row"] for p in perm])),
                "n_perm": int(n_perm)}
            self.head.train()
        return out

    def derangement(self, rows_vid, seed):
        """Permutation SANS point fixe des 16 videos : la ligne de la video v est confrontee aux
        references de la video u != v (et a celles du jumeau de u). Si la tete s'etait effondree
        sur une constante, ou si elle lisait autre chose que l'etat, la precision ne bougerait
        pas -- c'est le controle scelle n 2."""
        uniq = sorted(set(rows_vid))
        rng = np.random.RandomState(seed)
        perm = None
        for _ in range(200):
            cand = rng.permutation(len(uniq))
            if all(cand[i] != i for i in range(len(uniq))):
                perm = cand
                break
        assert perm is not None, "aucun derangement trouve"
        remap = {u: uniq[int(perm[i])] for i, u in enumerate(uniq)}
        return [remap[v] for v in rows_vid]

    def score(self, P, rows_vid):
        marg_v = {}
        for r in range(P.shape[0]):
            v = rows_vid[r]
            own = self.data.refs[v]
            oth = self.data.refs[self.data.twin_index[v]]
            m = float((P[r][None] @ own.transpose(0, 1)).mean()
                      - (P[r][None] @ oth.transpose(0, 1)).mean())
            marg_v.setdefault(v, []).append(m)
        vmarg = {v: float(np.mean(m)) for v, m in marg_v.items()}
        pmarg = {}
        for v, m in vmarg.items():
            pmarg.setdefault(self.data.of_pair[v], []).append(m)
        pmarg = {p: float(np.mean(m)) for p, m in pmarg.items()}
        rows = [m for ms in marg_v.values() for m in ms]
        return {"acc_row": float(np.mean([m > 0 for m in rows])),
                "acc_video": float(np.mean([m > 0 for m in vmarg.values()])),
                "acc_pair": float(np.mean([m > 0 for m in pmarg.values()])),
                "n_pairs_correct": int(sum(1 for m in pmarg.values() if m > 0)),
                "n_pairs": len(pmarg), "margin_mean": float(np.mean(list(pmarg.values()))),
                "pair_margins": {str(k): v for k, v in sorted(pmarg.items())},
                "video_margins": {str(k): v for k, v in sorted(vmarg.items())}}

    def save(self, path):
        writer = {str(bi): {name: self.wparams[(bi, name)].detach().to(torch.bfloat16).cpu()
                            for (b, name) in self.wparams if b == bi}
                  for bi in self.blocks}
        torch.save({"writer": writer,
                    "head": (self.head.state_dict() if self.head is not None else None),
                    "norm": (None if self.mu is None else
                             {"mu": self.mu.cpu(), "sd": self.sd.cpu()}),
                    "step": self.step, "args": vars(self.args),
                    "theta_l1_ckpt": str(self.config.lora_ckpt)}, path)
        print(f"[toy] ckpt -> {path}", flush=True)


# ---------------------------------------------------------------------------------------------
# le run
# ---------------------------------------------------------------------------------------------
def run(args):
    import wandb
    assert os.environ.get("WANDB_MODE", "") != "disabled", \
        "regle de la maison : WANDB_MODE=disabled est interdit"
    os.makedirs(HEAVY, exist_ok=True)
    wandb.init(project=args.wandb_project, name=f"{args.run_name}_{args.tag}", dir=HEAVY,
               config=vars(args))
    tr = ToyTrainer(args)
    if tr.head is not None:
        tr.fit_norm()
    base = tr.evaluate(n_perm=args.n_perm)
    print(f"[toy:{args.tag}] DEPART : reco {base['reco_mean']:.4f}"
          + (f" | paires {base['n_pairs_correct']}/{base['n_pairs']} "
             f"(marge {base['margin_mean']:+.4f}) | permute "
             f"{base['permuted']['n_pairs_correct']:.2f}/{base['n_pairs']}"
             if tr.head is not None else ""), flush=True)
    wandb.log({f"start/{k}": v for k, v in base.items() if isinstance(v, float)}, step=0)

    order = list(range(tr.data.n_videos()))
    hist = [{"step": 0, "n_pairs_correct": base.get("n_pairs_correct", -1),
             "margin_mean": base.get("margin_mean", float("nan")),
             "reco_mean": base["reco_mean"],
             "permuted": (base["permuted"]["n_pairs_correct"] if tr.head is not None else -1)}]
    t0 = time.time()
    for s in range(args.steps):
        tr.step = s + 1
        if (s * args.bs) % len(order) == 0:
            tr.rng.shuffle(order)
        batch = [order[(s * args.bs + i) % len(order)] for i in range(args.bs)]
        logs = tr.train_step(batch)
        if s == 0 and not args.freeze_writer:
            g = tr.masters[(tr.blocks[0], "qkv.weight")].grad
            assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, \
                "VIOLATION : qkv.weight n'a recu AUCUN gradient au premier backward"
            print(f"[toy] gradient jusqu'a theta_w : |sum| qkv.weight = "
                  f"{float(g.abs().sum()):.3e} (non nul, fini) ; VRAM {tr.peak_vram:.1f} GB",
                  flush=True)
        if (s + 1) % args.log_every == 0 or s == 0:
            wandb.log(logs, step=s + 1)
            print(f"[toy:{args.tag}] step {s + 1}/{args.steps} loss {logs['train/loss']:.4f} "
                  f"id {logs['train/id']:.4f} rec {logs['train/rec']:.4f} "
                  f"acc {logs['train/acc_row']:.3f} d_l1 {logs['writer/delta_l1']:.2e} "
                  f"({(time.time() - t0) / (s + 1):.1f} s/step)", flush=True)
        if (s + 1) % args.eval_every == 0 and s + 1 < args.steps:
            ev = tr.evaluate(n_perm=args.n_perm)
            hist.append({"step": s + 1, "n_pairs_correct": ev.get("n_pairs_correct", -1),
                         "margin_mean": ev.get("margin_mean", float("nan")),
                         "reco_mean": ev["reco_mean"],
                         "permuted": (ev["permuted"]["n_pairs_correct"]
                                      if tr.head is not None else -1)})
            wandb.log({f"eval/{k}": v for k, v in ev.items() if isinstance(v, float)}
                      | {"eval/permuted_pairs": ev["permuted"]["n_pairs_correct"]
                         if tr.head is not None else 0.0}, step=s + 1)
            print(f"[toy:{args.tag}] step {s + 1} EVAL paires "
                  f"{ev.get('n_pairs_correct', -1)}/{args.n_pairs} reco {ev['reco_mean']:.4f} "
                  f"permute {ev['permuted']['n_pairs_correct'] if tr.head is not None else -1}",
                  flush=True)

    final = tr.evaluate(n_perm=args.n_perm)
    hist.append({"step": args.steps, "n_pairs_correct": final.get("n_pairs_correct", -1),
                 "margin_mean": final.get("margin_mean", float("nan")),
                 "reco_mean": final["reco_mean"],
                 "permuted": (final["permuted"]["n_pairs_correct"]
                              if tr.head is not None else -1)})
    tr.save(os.path.join(HEAVY, f"toy_{args.tag}.pt"))
    res = {"tag": args.tag, "args": vars(args), "start": base, "final": final, "hist": hist,
           "minutes": (time.time() - t0) / 60, "peak_vram_gb": tr.peak_vram,
           "pairs_board_no": [p["board_no"] for p in tr.data.pairs],
           "pairs_corpus_idx": [int(p["corpus_idx"]) for p in tr.data.pairs],
           "video_ids": [m["video_id"] for m in tr.data.metas]}
    out = os.path.join(HEAVY, f"result_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[toy:{args.tag}] FIN : paires {final.get('n_pairs_correct', -1)}/{args.n_pairs} "
          f"| permute {final['permuted']['n_pairs_correct'] if tr.head is not None else -1:.2f}"
          f"/{args.n_pairs} | reco {base['reco_mean']:.4f} -> {final['reco_mean']:.4f} "
          f"(x{final['reco_mean'] / base['reco_mean']:.2f}) -> {out}", flush=True)
    wandb.log({f"final/{k}": v for k, v in final.items() if isinstance(v, float)}, step=args.steps)
    wandb.finish()
    return res


# ---------------------------------------------------------------------------------------------
# LE VERDICT : les 4 criteres scelles, appliques mecaniquement
# ---------------------------------------------------------------------------------------------
def verdict(args):
    paths = {t: os.path.join(HEAVY, f"result_{t}.json") for t in ("trained", "frozen")}
    for t, p in paths.items():
        assert os.path.exists(p), f"il manque le run '{t}' ({p})"
    R = {t: json.load(open(p)) for t, p in paths.items()}
    tr, fz = R["trained"]["final"], R["frozen"]["final"]
    n = R["trained"]["args"]["n_pairs"]
    c1 = tr["n_pairs_correct"]
    c2 = fz["n_pairs_correct"]
    perm = tr["permuted"]["n_pairs_correct"]
    r0, r1 = R["trained"]["start"]["reco_mean"], tr["reco_mean"]
    ok1 = c1 >= args.gate_pairs
    ok2 = perm <= args.gate_perm
    ok3 = (c1 - c2) >= args.gate_gap
    ok4 = r1 <= args.gate_reco * r0
    ok = bool(ok1 and ok2 and ok3 and ok4)
    lines = [
        "=" * 92,
        "TOY D'ECRITURE-IDENTITE -- LES 4 CRITERES SCELLES (registre, 2026-08-05)",
        "=" * 92,
        f"(1) precision par paire >= {args.gate_pairs}/{n}        : {c1}/{n} "
        f"(marge moyenne {tr['margin_mean']:+.4f}) : {'PASS' if ok1 else 'FAIL'}",
        f"(2) etats permutes <= {args.gate_perm}/{n}              : {perm:.2f}/{n} "
        f"({tr['permuted']['n_perm']} derangements) : {'PASS' if ok2 else 'FAIL'}",
        f"(3) ecrivain GELE nettement pire (>= {args.gate_gap} paires) : gele {c2}/{n} vs "
        f"entraine {c1}/{n} (ecart {c1 - c2:+d}) : {'PASS' if ok3 else 'FAIL'}",
        f"(4) reco <= {args.gate_reco:.0f}x le depart           : {r0:.4f} -> {r1:.4f} "
        f"(x{r1 / r0:.2f}) : {'PASS' if ok4 else 'FAIL'}",
        "-" * 92,
        f"VERDICT : {'PASS' if ok else 'FAIL'}"
        + ("" if ok else "  =>  fin definitive du chantier ecriture (regle scellee)"),
        "=" * 92]
    print("\n".join(lines), flush=True)
    obj = {"verdict": ok, "criteres": {
        "1_pairs": {"value": c1, "threshold": args.gate_pairs, "pass": bool(ok1)},
        "2_permuted": {"value": perm, "threshold": args.gate_perm, "pass": bool(ok2)},
        "3_frozen_gap": {"trained": c1, "frozen": c2, "gap": c1 - c2,
                         "threshold": args.gate_gap, "pass": bool(ok3)},
        "4_reco": {"start": r0, "final": r1, "ratio": r1 / r0,
                   "threshold": args.gate_reco, "pass": bool(ok4)}},
        "n_pairs": n, "trained": {k: v for k, v in tr.items() if not isinstance(v, list)},
        "frozen": {k: v for k, v in fz.items() if not isinstance(v, list)},
        "text": "\n".join(lines)}
    with open(os.path.join(HEAVY, "verdict.json"), "w") as f:
        json.dump(obj, f, indent=1)
    return obj


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="toy ecriture-identite (amendement PI)")
    ap.add_argument("--stage", default="run", choices=("run", "verdict"))
    ap.add_argument("--tag", default="trained", help="nom du bras (trained | frozen)")
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--probes", default=PROBES_PT)
    ap.add_argument("--dec_ckpt", default=C.DEC_V12)
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4, help="videos par pas (gradient accumule)")
    ap.add_argument("--lam_id", type=float, default=1.0,
                    help="poids du terme d'identite (0 = terme JAMAIS construit = reco seule)")
    ap.add_argument("--rec_w", type=float, default=0.1, help="poids du regulariseur de reco")
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr_w", type=float, default=1e-4)
    ap.add_argument("--lr_h", type=float, default=1e-3)
    ap.add_argument("--wd_h", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n_bptt", type=int, default=None, help="None = BPTT complet")
    ap.add_argument("--freeze_writer", action="store_true", help="critere 3 : la tete seule")
    ap.add_argument("--n_perm", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--run_name", default="toy_identity")
    ap.add_argument("--wandb_project", default="toy_identity")
    # seuils SCELLES au registre : des flags pour etre lisibles dans wandb.config, PAS pour
    # etre bouges.
    ap.add_argument("--gate_pairs", type=int, default=7)
    ap.add_argument("--gate_perm", type=float, default=5.0)
    ap.add_argument("--gate_gap", type=int, default=2,
                    help="'nettement pire' = au moins 2 paires d'ecart (8 paires : 1 paire est "
                         "le pas de quantification, 2 = un ecart net)")
    ap.add_argument("--gate_reco", type=float, default=2.0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.stage == "verdict":
        v = verdict(a)
        sys.exit(0 if v["verdict"] else 1)
    run(a)
