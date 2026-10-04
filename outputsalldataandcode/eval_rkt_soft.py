#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
from collections import defaultdict

from rkt_thesaurus_utils import (
    ancestors,
    build_parent_graph,
    build_term_index,
    load_concepts,
    resolve_term,
)


def read_dataset(data_path, read_fn=lambda x: x, sep="\t"):
    vocab = defaultdict(list)
    with codecs.open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line_split = line.rstrip("\n").split(sep)
            if len(line_split) < 2:
                continue
            word = line_split[0].upper()
            if word == "WORD":
                continue
            vocab[word].append(read_fn(line_split[1]))
    return vocab


def norm_preds(preds):
    out = []
    for p in preds:
        if isinstance(p, (list, tuple)) and p:
            out.append(str(p[0]))
        else:
            out.append(str(p))
    return out


def norm_gold(g):
    out = []

    def walk(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for t in v:
                walk(t)
            return
        out.append(str(v))

    walk(g)
    return set(out)


def map_mrr_at_k(reference, submitted, k=15):
    ap_list = []
    rr_list = []
    for word, gold_raw in reference.items():
        gold = norm_gold(gold_raw)
        preds = norm_preds(submitted.get(word, []))[:k]
        if not gold:
            continue

        hits = 0
        ap_sum = 0.0
        first_rank = None
        for i, cand in enumerate(preds, start=1):
            if cand in gold:
                hits += 1
                ap_sum += hits / i
                if first_rank is None:
                    first_rank = i

        ap_list.append(ap_sum / len(gold))
        rr_list.append(0.0 if first_rank is None else 1.0 / first_rank)

    if not ap_list:
        return 0.0, 0.0
    return sum(ap_list) / len(ap_list), sum(rr_list) / len(rr_list)


def build_soft_reference(reference, concepts, depth):
    graph = build_parent_graph(concepts)
    term_index, _, _ = build_term_index(concepts, include_english=False, use_morph=True)

    soft = {}
    unresolved = []
    for word, gold_raw in reference.items():
        ids, _ = resolve_term(word, term_index, use_morph=True)
        if ids:
            soft[word] = ancestors(ids[0], graph, max_depth=depth)
        else:
            soft[word] = sorted(norm_gold(gold_raw))
            unresolved.append(word)
    return soft, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_path", required=True)
    ap.add_argument("--reference_path", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    submitted = read_dataset(args.predict_path)
    reference = read_dataset(args.reference_path, json.loads)
    concepts = load_concepts(args.jsonl)

    direct_map, direct_mrr = map_mrr_at_k(reference, submitted, k=args.k)
    soft_reference, unresolved = build_soft_reference(reference, concepts, depth=args.depth)
    soft_map, soft_mrr = map_mrr_at_k(soft_reference, submitted, k=args.k)

    print(f"Results for {args.predict_path}:")
    print(f"\tdirect MAP@{args.k} = {direct_map}")
    print(f"\tdirect MRR@{args.k} = {direct_mrr}")
    print(f"\tsoft(depth<={args.depth}) MAP@{args.k} = {soft_map}")
    print(f"\tsoft(depth<={args.depth}) MRR@{args.k} = {soft_mrr}")
    if unresolved:
        print(f"\tunresolved reference terms: {len(unresolved)}")


if __name__ == "__main__":
    main()
