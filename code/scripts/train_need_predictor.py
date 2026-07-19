#!/usr/bin/env python
"""Train a learned need-predictor for slide-evidence gating.

Labels are mined automatically: a segment is "hard" if an offline baseline
translation misses most of the reference's rare terms (term recall <= 0.4).
Features are online-available source-side statistics only (no reference, no
translation): computable from the raw source text at gate-decision time.
Two feature scopes are evaluated: full-segment (feasibility upper bound) and
prefix-30% (what a runtime gate actually sees early in the segment).

Evaluation is talk-held-out: train on all talks except --holdout-talk, report
AUC/precision/recall on the held-out talk, and emit per-segment gate decisions
for end-task evaluation with oracle_killtest_runner.py (gated_oracle condition
reads the emitted gate_on field).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def term_recall(hyp: str, terms: list[str]) -> float:
    h = " " + hyp.lower() + " "
    return sum(1 for t in terms if t.lower().rstrip("s") in h) / len(terms)


def features(source: str, prefix_frac: float = 1.0) -> list[float]:
    words = source.split()
    n = max(1, int(round(len(words) * prefix_frac)))
    words = words[:n]
    text = " ".join(words)
    lens = [len(w.strip(".,;·!?")) for w in words] or [0]
    long_words = sum(1 for L in lens if L >= 8)
    vlong_words = sum(1 for L in lens if L >= 11)
    return [
        len(words),
        float(np.mean(lens)),
        float(max(lens)),
        long_words,
        long_words / len(words) if words else 0.0,
        vlong_words,
        sum(ch.isdigit() for ch in text),
        len(set(text)) / max(len(text), 1),          # char diversity
        sum(1 for w in words if len(w) >= 6 and w[0].isupper()),
    ]


FEATURE_NAMES = ["n_words", "mean_wlen", "max_wlen", "n_long", "frac_long",
                 "n_vlong", "n_digits", "char_div", "n_cap_long",
                 "lp_mean", "lp_min", "lp_frac15", "lp_frac25", "lp_n15"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="np_labels.jsonl from offline_label")
    parser.add_argument("--logprobs", default=None,
                        help="np_logprobs.jsonl: adds model-confidence features "
                             "(mean/min token logprob, low-confidence fractions)")
    parser.add_argument("--holdout-talk", required=True)
    parser.add_argument("--hard-threshold", type=float, default=0.4)
    parser.add_argument("--out-gates", default=None,
                        help="write per-segment gate decisions for the holdout talk")
    parser.add_argument("--target-fire-rate", type=float, default=0.55,
                        help="calibrate decision threshold to this fire rate on train")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.labels)]
    rows = [r for r in rows if r.get("hypothesis")]
    lp_map = {}
    if args.logprobs:
        for l in open(args.logprobs):
            r = json.loads(l)
            if r.get("logprobs"):
                lp_map[r["id"]] = r["logprobs"]
        rows = [r for r in rows if r["id"] in lp_map]
        print(f"with logprobs: {len(rows)}")
    y = np.array([1 if term_recall(r["hypothesis"], r["oracle_terms"]) <= args.hard_threshold
                  else 0 for r in rows])
    talks = np.array([r["talk_id"] for r in rows])
    print(f"segments {len(rows)}, hard rate {y.mean():.2f}")

    def lp_features(rid):
        lps = lp_map[rid]
        a = np.array(lps) if lps else np.array([0.0])
        return [float(a.mean()), float(a.min()),
                float((a < -1.5).mean()), float((a < -2.5).mean()),
                float((a < -1.5).sum())]

    for scope, frac in [("full-segment", 1.0), ("prefix-30%", 0.30)]:
        X = np.array([features(r["source"], frac) for r in rows])
        if lp_map:
            X = np.hstack([X, np.array([lp_features(r["id"]) for r in rows])])
        tr = talks != args.holdout_talk
        te = ~tr
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], prob)
        # calibrate threshold on train to target fire rate
        tr_prob = clf.predict_proba(X[tr])[:, 1]
        thresh = float(np.quantile(tr_prob, 1 - args.target_fire_rate))
        pred = (prob >= thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average="binary",
                                                     zero_division=0)
        print(f"[{scope}] holdout {args.holdout_talk}: AUC {auc:.3f} | "
              f"P {p:.2f} R {r:.2f} F1 {f:.2f} | fire rate {pred.mean():.2f} "
              f"(hard rate {y[te].mean():.2f})")
        top = sorted(zip(FEATURE_NAMES[:X.shape[1]], clf.coef_[0]), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  top features: {[(n, round(c, 2)) for n, c in top]}")
        if args.out_gates and scope == "prefix-30%":
            gates = {rows[i]["id"]: bool(pred[j])
                     for j, i in enumerate(np.where(te)[0])}
            json.dump(gates, open(args.out_gates, "w"))
            print(f"  wrote {sum(gates.values())}/{len(gates)} gate-on decisions "
                  f"-> {args.out_gates}")


if __name__ == "__main__":
    main()
