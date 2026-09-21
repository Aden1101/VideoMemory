"""The Echo-Forcing baseline: a pool of real past frames, routed back by similarity."""
import os

import torch

from pipeline.switch.decoded_anchor import _memdec_pooled_cond
from utils.env import env_float, env_int


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
        MEM_DECODE_ROUTED, which borrows the router's decision only. With both
        off, `_fb_store` and `_fb_select` do nothing."""
        return bool(self._fb_mode()) or bool(env_int("MEM_DECODE_ROUTED"))

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
        analysis of the selection (the bank holds one to four entries in the schedules used).

        `mode` overrides FRAMEBANK: MEM_DECODE_ROUTED calls this with mode='key' to obtain the
        routing decision ('is the entering prompt a revisit of a non-adjacent banked scene?')
        while FRAMEBANK stays off and the recalled frame is thrown away: the same moment as the
        framebank arm, a different content.

        `force` (FRAMEBANK_GT) replaces the router's decision with the schedule's ground-truth
        revisit flag and leaves the selection alone: True recalls the argmax candidate even
        below tau, False recalls nothing, None lets the router decide. It exists because a
        vague return prompt ("the animal from earlier returns") collapses the prompt cosine, the
        bank never fires, and the comparison becomes a splice against no splice instead of one
        content against another. What the router would have done stays in the event
        (router_recalled, router_chosen_seg), since how often it fails is a result too."""
        mode = mode or self._fb_mode()
        tau = env_float("FRAMEBANK_TAU", 0.90)
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
        if mode not in ("truth", "key"):
            raise ValueError(f"FRAMEBANK={mode!r} (expected ''|'key'|'truth')")

        def choose(threshold):
            """The banked entry this mode selects, or None. `threshold` only binds the cosine
            router; the forced recall passes -inf to take the argmax whatever its score."""
            if mode == "truth":
                pool = [e for e in cands if e["subject"] == subj]
            else:
                smax = max(sims) if cands else float("-inf")
                pool = [e for e, s in zip(cands, sims) if s >= smax - 1e-6] if smax >= threshold else []
            return (pool[-1] if tie == "recent" else pool[0]) if pool else None

        def chosen(prefix, pick, fields):
            """What the event log keeps of a selection: whether one was made, which entry, and
            whether it shows the right subject."""
            out = {f"{prefix}recalled": pick is not None}
            out.update({f"{prefix}chosen_{f}": None if pick is None else int(pick[f]) for f in fields})
            out[f"{prefix}correct"] = bool(pick is not None and pick["subject"] == subj)
            return out

        pick = choose(tau)
        if mode == "truth":
            ev["rule"] = "ground-truth subject identity (bank ceiling)"
        else:
            if cands:
                ev["s_max"] = max(sims)
            ev["rule"] = f"argmax cosine over non-adjacent scenes, recall iff s_max >= {tau}"
        if force is not None:
            # what the router would have done is kept as is, it is a result of its own
            ev.update(chosen("router_", pick, ("seg", "subject")))
            ev["forced"] = bool(force)
            ev["rule"] += " | FRAMEBANK_GT: the moment comes from the schedule, the selection "
            ev["rule"] += ("stays the router's (argmax even below tau)" if mode == "key"
                           else "stays the ground-truth one")
            pick = choose(float("-inf")) if force else None
        ev.update(chosen("", pick, ("seg", "prompt_idx", "subject")))
        self._fb_events.append(ev)
        print(f"[FRAMEBANK] recall @segment {seg_idx} (prompt {prompt_idx}, subject {subj}): " +
              " ".join(f"seg{c['seg']}(p{c['prompt_idx']}):{c['sim']:.4f}" for c in ev["candidates"])
              + f" -> {'segment ' + str(ev['chosen_seg']) if ev['recalled'] else 'NO RECALL'}"
              + (" [CORRECT]" if ev["correct"] else (" [WRONG SUBJECT]" if ev["recalled"] else "")),
              flush=True)
        return (None if pick is None else pick["frame"]), ev
