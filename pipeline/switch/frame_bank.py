"""The Echo-Forcing baseline: a pool of real past frames, routed back by similarity."""
import os

import torch

from pipeline.switch.anchor_decode import _memdec_pooled_cond



class FrameBankMixin:

    # ---- the competing baseline: keep real past frames and route them back --------------------
    # Echo-Forcing (2605.16003) keeps a pool of real frames from past scenes and, at a scene
    # change, brings one back when the entering prompt is close enough to a past one. Ported here
    # in the same spirit: the pool holds the last frame of every finished segment, keyed by that
    # segment's prompt, and the winner is spliced at the most recent window slot. Same mechanism
    # as our own recall; only where the content comes from differs.
    #   FRAMEBANK=""       off (default)
    #   FRAMEBANK="key"    real routing: argmax cosine over non-adjacent entries, recall iff >= tau
    #   FRAMEBANK="truth"  ceiling of the same bank: ground-truth prompt identity, no cosine
    #   FRAMEBANK_TAU      routing threshold (default 0.90)
    #   FRAMEBANK_TIE      "recent" (default, freshest true observation) | "first" (1st appearance)
    def _fb_mode(self):
        return (os.environ.get("FRAMEBANK", "") or "").strip().lower()

    def _route_on(self):
        """Is the Echo-Forcing ROUTER (bank + cosine decision) needed at all this rollout?
        True for the framebank arms (which also splice the banked frame) and for
        MEM_DECODE_ROUTED, which borrows the router's decision only. Default
        off on both counts => `_fb_store`/`_fb_select` stay no-ops and every existing arm is
        byte-identical."""
        return bool(self._fb_mode()) or bool(int(os.environ.get("MEM_DECODE_ROUTED", "0") or 0))

    def _fb_reset(self):
        self._fb_bank = []       # list of {seg, prompt_idx, end_frame, key, frame}
        self._fb_events = []     # per-switch routing log (dumped by the driver for error analysis)

    @staticmethod
    def _fb_key(cond):
        """L2-normalized pooled prompt embedding = the scene key. Same pooling as the decoder's
        cond (non-padding rows only), fp32, CPU: deterministic, draws no RNG (the paired-arm
        noise stream must stay identical)."""
        k = _memdec_pooled_cond(cond).float()
        return k / k.norm().clamp_min(1e-12)

    def _fb_store(self, output, current_start_frame, seg_idx, prompt_idx, cond, subject=None):
        """Scene lifecycle 'preserve': the segment that just ended enters the pool.
        `subject` = the ground-truth identity of the segment (which may be described by several
        different prompt strings, as in the paraphrase scenarios); it is used only by the
        ground-truth selection and the correctness bookkeeping, the real router never sees it."""
        if not self._route_on():
            return
        assert current_start_frame >= 1, "FRAMEBANK store before any frame was produced"
        self._fb_bank.append({
            "seg": int(seg_idx), "prompt_idx": int(prompt_idx),
            "subject": int(prompt_idx if subject is None else subject),
            "end_frame": int(current_start_frame),
            "key": self._fb_key(cond),
            "frame": output[:, current_start_frame - 1:current_start_frame].detach().clone(),
        })
        print(f"[FRAMEBANK] preserve: segment {seg_idx} (prompt {prompt_idx}, subject "
              f"{self._fb_bank[-1]['subject']}), last latent frame {current_start_frame - 1} "
              f"-> pool size {len(self._fb_bank)}", flush=True)

    def _fb_select(self, seg_idx, prompt_idx, cond, subject=None, mode=None, force=None):
        """Scene lifecycle 'recall'. Returns (frame or None, event dict). The event dict logs
        every candidate similarity and whether the pick was the right subject -> the error
        analysis of the selection (easy at bank size 1-2 in S1/S2, 2-4 in S3/S4).

        `mode` overrides FRAMEBANK: MEM_DECODE_ROUTED calls this with mode='key' to obtain the
        routing decision ('is the entering prompt a revisit of a non-adjacent banked scene?')
        while FRAMEBANK stays off and the recalled frame is thrown away, same QUAND as the
        framebank arm, a different QUOI.

        `force` (FRAMEBANK_GT, pilote 'texte insuffisant' 2026-08-03) replaces the router's
        decision by the schedule's ground-truth revisit flag, without touching the selection:
        force=True recalls the argmax candidate even below tau, force=False recalls nothing.
        Default None = the published behaviour, bitwise. Why: with an impoverished return prompt
        ('the animal from earlier returns') the prompt cosine collapses, the bank never fires and
        the arm silently degenerates into a no-recall run, and the comparison would then be
        'a splice vs no splice', not 'which content'. What the router would have done is kept in
        the event (router_recalled / router_chosen_seg / router_s_max): it is a result of its own,
        not a detail (word routing dies in the generic regime)."""
        mode = mode or self._fb_mode()
        tau = float(os.environ.get("FRAMEBANK_TAU", "0.90") or 0.90)
        tie = (os.environ.get("FRAMEBANK_TIE", "recent") or "recent").strip().lower()
        assert tie in ("recent", "first"), f"FRAMEBANK_TIE must be recent|first, got {tie!r}"
        subj = int(prompt_idx if subject is None else subject)
        q = self._fb_key(cond)
        # Echo-Forcing routing rule: the immediately preceding scene is the recent window, never
        # a "recall" -> it is excluded from the candidate pool.
        cands = [e for e in self._fb_bank if e["seg"] != seg_idx - 1]
        sims = [float(torch.dot(q, e["key"])) for e in cands]
        ev = {"seg": int(seg_idx), "prompt_idx": int(prompt_idx), "subject": subj, "mode": mode,
              "tau": tau, "tie": tie, "bank_size": len(self._fb_bank),
              "candidates": [{"seg": e["seg"], "prompt_idx": e["prompt_idx"],
                              "subject": e["subject"], "end_frame": e["end_frame"], "sim": s}
                             for e, s in zip(cands, sims)]}
        pick = None
        if mode == "truth":
            ok = [e for e in cands if e["subject"] == subj]
            if ok:
                pick = ok[-1] if tie == "recent" else ok[0]
            ev["rule"] = "ground-truth subject identity (bank ceiling)"
        elif mode == "key":
            if cands:
                smax = max(sims)
                best = [e for e, s in zip(cands, sims) if s >= smax - 1e-6]
                ev["s_max"] = smax
                if smax >= tau:
                    pick = best[-1] if tie == "recent" else best[0]
            ev["rule"] = f"argmax cosine over non-adjacent scenes, recall iff s_max >= {tau}"
        else:
            raise AssertionError(f"FRAMEBANK={mode!r} (expected ''|'key'|'truth')")
        if force is not None:
            # what the router would have done is kept as is, it is a result of its own
            ev["router_recalled"] = pick is not None
            ev["router_chosen_seg"] = None if pick is None else int(pick["seg"])
            ev["router_chosen_subject"] = None if pick is None else int(pick["subject"])
            ev["router_correct"] = bool(pick is not None and pick["subject"] == subj)
            ev["forced"] = bool(force)
            ev["rule"] += " | FRAMEBANK_GT: the moment comes from the schedule, the selection "
            ev["rule"] += ("stays the router's (argmax even below tau)" if mode == "key"
                           else "stays the ground-truth one")
            pick = None
            if force and cands:
                if mode == "truth":
                    ok = [e for e in cands if e["subject"] == subj]
                    pick = (ok[-1] if tie == "recent" else ok[0]) if ok else None
                else:
                    smax = max(sims)
                    best = [e for e, s in zip(cands, sims) if s >= smax - 1e-6]
                    pick = best[-1] if tie == "recent" else best[0]
        ev["recalled"] = pick is not None
        ev["chosen_seg"] = None if pick is None else int(pick["seg"])
        ev["chosen_prompt_idx"] = None if pick is None else int(pick["prompt_idx"])
        ev["chosen_subject"] = None if pick is None else int(pick["subject"])
        ev["correct"] = bool(pick is not None and pick["subject"] == subj)
        self._fb_events.append(ev)
        print(f"[FRAMEBANK] recall @segment {seg_idx} (prompt {prompt_idx}, subject {subj}): " +
              " ".join(f"seg{c['seg']}(p{c['prompt_idx']}):{c['sim']:.4f}" for c in ev["candidates"])
              + f" -> {'segment ' + str(ev['chosen_seg']) if ev['recalled'] else 'NO RECALL'}"
              + (" [CORRECT]" if ev["correct"] else (" [WRONG SUBJECT]" if ev["recalled"] else "")),
              flush=True)
        return (None if pick is None else pick["frame"]), ev
