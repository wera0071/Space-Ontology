#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

from rkt_thesaurus_utils import (
    build_parent_graph,
    concept_id,
    concept_name,
    direct_parents,
    load_concepts,
)


def read_words(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to concepts.lii_thes.kosmos.json")
    ap.add_argument("--out", default="rkt_reference.tsv")
    ap.add_argument("--debug_out", default="rkt_reference_debug.tsv")
    ap.add_argument("--words_path", default=None, help="Optional list of conceptstr values to keep")
    ap.add_argument("--relation", default="ВЫШЕ")
    ap.add_argument("--include_empty", action="store_true")
    ap.add_argument(
        "--ancestor_depth",
        type=int,
        default=1,
        help="1 = only direct ВЫШЕ parents; 2 = parents plus grandparents",
    )
    args = ap.parse_args()

    concepts = load_concepts(args.jsonl)
    graph = build_parent_graph(concepts, relation=args.relation)
    keep_words = read_words(args.words_path)

    rows = []
    debug_rows = []
    for concept in concepts:
        cid = concept_id(concept)
        name = concept_name(concept)
        if not cid or not name:
            continue
        if keep_words is not None and name.upper() not in keep_words:
            continue

        if args.ancestor_depth <= 1:
            parents = direct_parents(concept, relation=args.relation)
        else:
            from rkt_thesaurus_utils import ancestors

            parents = ancestors(cid, graph, max_depth=args.ancestor_depth)

        if not parents and not args.include_empty:
            continue

        rows.append((name.upper(), parents))
        debug_rows.append(
            {
                "conceptid": cid,
                "conceptstr": name,
                "gold_size": len(parents),
                "gold_ids": parents,
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for word, parents in rows:
            f.write(f"{word}\t{json.dumps(parents, ensure_ascii=False)}\n")

    with open(args.debug_out, "w", encoding="utf-8") as f:
        f.write("conceptid\tconceptstr\tgold_size\tgold_ids\n")
        for row in debug_rows:
            f.write(
                f"{row['conceptid']}\t{row['conceptstr']}\t{row['gold_size']}\t"
                f"{json.dumps(row['gold_ids'], ensure_ascii=False)}\n"
            )

    print(f"Wrote {args.out}")
    print(f"Wrote {args.debug_out}")
    print(f"Concepts in dump: {len(concepts)}")
    print(f"Reference rows: {len(rows)}")


if __name__ == "__main__":
    main()
