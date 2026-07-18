#!/usr/bin/env python
"""Analyze oracle kill-test runs: AL, chrF, term recall per condition.

AL follows Ma et al. (2019): with source length |x| in read units and target
length |y| in words, d_i = number of source units read when target word i was
committed; AL = (1/tau) * sum_{i<=tau} (d_i - (i-1)/r), tau = first i with
d_i = |x|, r = |y|/|x|. Reported in *source units* (2 words for el, 4 chars zh).
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import sacrebleu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--per-item", default=None, help="optional per-item TSV out")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.runs)]
    by_group: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        lang = "el" if r["id"].startswith("el") else "zh"
        by_group[(lang, r["condition"])].append(r)

    print(f"{'group':10} {'cond':7} {'n':>3} {'chrF':>6} {'termR':>6} {'AL':>6} {'len_ratio':>9}")
    for (lang, cond) in sorted(by_group):
        rs = by_group[(lang, cond)]
        chrf = sacrebleu.corpus_chrf([r["hypothesis"] for r in rs],
                                     [[r["reference"] for r in rs]]).score
        term_r = sum(term_recall(r) for r in rs) / max(len(rs), 1)
        al = sum(avg_lagging(r) for r in rs) / max(len(rs), 1)
        lr = sum(len(r["hypothesis"].split()) / max(len(r["reference"].split()), 1)
                 for r in rs) / max(len(rs), 1)
        print(f"{lang:10} {cond:7} {len(rs):3d} {chrf:6.1f} {term_r:6.2f} {al:6.2f} {lr:9.2f}")

    if args.per_item:
        with open(args.per_item, "w") as f:
            f.write("id\tcond\tchrf\tterm_recall\tal\n")
            for r in rows:
                chrf = sacrebleu.sentence_chrf(r["hypothesis"], [r["reference"]]).score
                f.write(f"{r['id']}\t{r['condition']}\t{chrf:.1f}\t"
                        f"{term_recall(r):.2f}\t{avg_lagging(r):.2f}\n")


def term_recall(r: dict) -> float:
    terms = r.get("oracle_terms") or []
    if not terms:
        return 0.0
    hyp = " " + re.sub(r"[^a-z\- ]", " ", r["hypothesis"].lower()) + " "
    hit = sum(1 for t in terms if f" {t.lower().rstrip('s')}" in hyp
              or f" {t.lower()}" in hyp)
    return hit / len(terms)


def avg_lagging(r: dict) -> float:
    x = r["n_src_units"]
    events = r["events"]
    if not events or x == 0:
        return float(x)
    # expand events into per-target-word d_i
    d = []
    prev = 0
    for n_read, n_total in events:
        d.extend([n_read] * (n_total - prev))
        prev = n_total
    y = len(d)
    rate = y / x
    tau = next((i + 1 for i, di in enumerate(d) if di >= x), y)
    s = sum(d[i] - i / rate for i in range(tau))
    return s / max(tau, 1)


if __name__ == "__main__":
    main()
