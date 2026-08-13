#!/usr/bin/env python
"""Snapshot of e1_ro_5L_metrics training trends (loss stability / divergence check)."""
import statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

D = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_metrics"
ea = EventAccumulator(D, size_guidance={'scalars': 0}); ea.Reload()
tags = ea.Tags().get('scalars', [])


def series(t):
    return ea.Scalars(t) if t in tags else []


def fmt(v):
    return "  ".join(f"s{x.step}:{x.value:.3g}" for x in v)


print("=== TREND CHECK (e1_ro_5L_metrics) ===")
for t in ["generator/generator_loss", "critic/critic_loss", "generator/generator_grad_norm"]:
    v = series(t)
    if not v:
        continue
    print(f"\n{t}  ({len(v)} pts, last step {v[-1].step})")
    print("  last12:", fmt(v[-12:]))
    if len(v) >= 12:
        a = st.mean(x.value for x in v[-12:-6])
        b = st.mean(x.value for x in v[-6:])
        arrow = "UP (worsening)" if b > a * 1.15 else ("DOWN (improving)" if b < a * 0.85 else "~flat")
        print(f"  mean prev6={a:.3g} -> last6={b:.3g}   [{arrow}]")

print("\n-- memory (first -> last) --")
for t in ["memory/gatew_b5", "memory/memout_b5", "memory/gate_b5", "memory/read_b5", "memory/drift_b5"]:
    v = series(t)
    if v:
        print(f"  {t}: {v[0].value:.3g} -> {v[-1].value:.3g}")
