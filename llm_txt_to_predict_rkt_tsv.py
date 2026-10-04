#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import OrderedDict
from typing import List

from rkt_thesaurus_utils import (
    build_term_index,
    clean_text,
    concept_name,
    load_concepts,
    resolve_term,
)


def split_hypernyms(rest: str) -> List[str]:
    rest = clean_text(rest)
    if not rest:
        return []
    parts = re.split(r"\s*;\s*", rest)
    if len(parts) == 1:
        parts = re.split(r"\s*,\s*", rest)
    out = []
    for part in parts:
        part = clean_text(part).strip(" .,:;")
        if part:
            out.append(part)
    return out


def read_llm_txt(path: str) -> "OrderedDict[str, List[str]]":
    out = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" in line:
                term, rest = line.split("\t", 1)
            else:
                m = re.match(r"^(.+?)\s*[:\-]\s*(.+)$", line.strip())
                if m:
                    term, rest = m.group(1), m.group(2)
                else:
                    term, rest = line.strip(), ""
            out[clean_text(term)] = split_hypernyms(rest)
    return out


def prob_from_rank(rank_1based: int, mode: str) -> float:
    r = max(1, int(rank_1based))
    if mode == "linear":
        return max(0.0, 1.0 - (r - 1) * 0.05)
    if mode == "exp":
        return 0.85 ** (r - 1)
    return 1.0 / r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--jsonl", required=True, help="Path to concepts.lii_thes.kosmos.json")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--debug_tsv", default=None)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--ids_per_hypernym", type=int, default=3)
    ap.add_argument("--include_english", action="store_true")
    ap.add_argument("--no_morph", action="store_true")
    ap.add_argument("--keep_input_word", action="store_true")
    ap.add_argument("--score_mode", default="harmonic", choices=["harmonic", "linear", "exp"])
    args = ap.parse_args()

    concepts = load_concepts(args.jsonl)
    use_morph = not args.no_morph
    term_index, concept_by_id, _ = build_term_index(
        concepts,
        include_english=args.include_english,
        use_morph=use_morph,
    )
    llm = read_llm_txt(args.in_txt)

    rows = []
    debug_rows = []
    counts = []

    for term, hypernyms in llm.items():
        term_ids, term_key = resolve_term(term, term_index, use_morph=use_morph)
        if term_ids and not args.keep_input_word:
            word_out = concept_name(concept_by_id[term_ids[0]]).upper()
        else:
            word_out = clean_text(term).upper()

        picked = []
        seen = set()
        for hypernym in hypernyms:
            ids, matched_key = resolve_term(hypernym, term_index, use_morph=use_morph)
            selected = []
            for cid in ids[: max(0, args.ids_per_hypernym)]:
                if cid in seen:
                    continue
                seen.add(cid)
                cname = concept_name(concept_by_id.get(cid, {}))
                prob = prob_from_rank(len(picked) + 1, args.score_mode)
                picked.append((word_out, cid, cname, prob))
                selected.append(cid)
                if len(picked) >= args.topk:
                    break

            if args.debug_tsv:
                debug_rows.append(
                    (
                        word_out,
                        term,
                        json.dumps(term_ids, ensure_ascii=False),
                        term_key,
                        hypernym,
                        matched_key,
                        json.dumps(selected, ensure_ascii=False),
                        str(bool(selected)),
                    )
                )

            if len(picked) >= args.topk:
                break

        rows.extend(picked)
        counts.append(len(picked))

    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "cand", "cand_name", "prob"])
        for word, cid, cname, prob in rows:
            w.writerow([word, cid, cname, f"{prob:.6f}"])

    if args.debug_tsv:
        with open(args.debug_tsv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "word",
                    "input_term",
                    "input_term_ids",
                    "input_term_key",
                    "hypernym_in",
                    "matched_key",
                    "mapped_ids",
                    "matched",
                ]
            )
            w.writerows(debug_rows)

    print(f"Wrote {args.out_tsv}")
    print(f"Concepts in dump: {len(concepts)}")
    print(f"Terms in txt: {len(llm)}")
    if counts:
        print(f"min/avg/max preds: {min(counts)} / {sum(counts) / len(counts):.2f} / {max(counts)}")
    if args.debug_tsv:
        print(f"Wrote debug: {args.debug_tsv}")


if __name__ == "__main__":
    main()
