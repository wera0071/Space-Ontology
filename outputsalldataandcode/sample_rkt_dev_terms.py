#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random

from rkt_thesaurus_utils import concept_name, direct_parents, load_concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="rkt_dev_terms.txt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relation", default="ВЫШЕ")
    args = ap.parse_args()

    concepts = load_concepts(args.jsonl)
    candidates = [
        concept_name(c)
        for c in concepts
        if concept_name(c) and direct_parents(c, relation=args.relation)
    ]

    rng = random.Random(args.seed)
    sample = candidates[:]
    rng.shuffle(sample)
    sample = sample[: min(args.n, len(sample))]

    with open(args.out, "w", encoding="utf-8") as f:
        for term in sample:
            f.write(term.upper() + "\n")

    print(f"Wrote {args.out}")
    print(f"Concepts with {args.relation}: {len(candidates)}")
    print(f"Sample size: {len(sample)}")


if __name__ == "__main__":
    main()
