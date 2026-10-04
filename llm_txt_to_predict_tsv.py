#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional


RUS_STOPWORDS = {
    "с", "со", "для", "по", "из", "от", "до", "к", "ко", "у", "о", "об", "обо",
    "под", "над", "при", "через", "без", "в", "во", "на", "за", "между",
    "и", "или", "либо", "а", "но", "что", "как", "это", "тот", "та", "те",
    "его", "ее", "её", "их", "мой", "моя", "мои", "твой", "твоя", "наш", "ваш",
    "человек", "люди"
}


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("«", '"').replace("»", '"')
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_key(s: str) -> str:
    s = _norm_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clean_phrase_for_match(s: str) -> str:
    s = _norm_key(s)
    s = re.sub(r"[.,:!?()\[\]{}\"'`]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_hypernyms(rest: str) -> List[str]:
    if not rest:
        return []
    parts = [p.strip() for p in rest.split(";")]
    out = []
    for p in parts:
        p = _norm_text(p)
        p = p.strip(" ,.;")
        if p:
            out.append(p)
    return out


def _extract_head_candidates(phrase: str) -> List[str]:
    p = _clean_phrase_for_match(phrase)
    if not p:
        return []
    toks = [t for t in p.split() if t]
    if not toks:
        return []
    cands = []
    if len(toks) == 1:
        return [toks[0]]
    for t in reversed(toks):
        if t not in RUS_STOPWORDS and len(t) > 1:
            cands.append(t)
            break
    for t in toks:
        if t not in RUS_STOPWORDS and len(t) > 1 and t not in cands:
            cands.append(t)
    if toks[-1] not in cands:
        cands.append(toks[-1])
    return cands[:3]


def read_llm_txt(path: str) -> "OrderedDict[str, List[str]]":
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            if "\t" in line:
                term, rest = line.split("\t", 1)
            else:
                m = re.match(r"^(\S+)\s+(.*)$", line.strip())
                if not m:
                    term, rest = line.strip(), ""
                else:
                    term, rest = m.group(1), m.group(2)
            term = _norm_text(term)
            rest = _norm_text(rest)
            out[term] = _split_hypernyms(rest)
    return out


def load_senses_index(ruwordnet_dir: str, pos: str) -> Dict[str, List[str]]:
    p = os.path.join(ruwordnet_dir, f"senses.{pos}.xml")
    idx: Dict[str, List[str]] = defaultdict(list)

    for _, el in ET.iterparse(p, events=("end",)):
        if el.tag == "sense":
            lemma = el.attrib.get("lemma") or el.attrib.get("name") or ""
            synset_id = el.attrib.get("synset_id") or ""
            k = _clean_phrase_for_match(lemma)
            if k and synset_id:
                idx[k].append(synset_id)
        el.clear()

    for k, vals in list(idx.items()):
        seen = set()
        dedup = []
        for sid in vals:
            if sid not in seen:
                seen.add(sid)
                dedup.append(sid)
        idx[k] = dedup

    return dict(idx)


def load_synset_names(ruwordnet_dir: str, pos: str) -> Dict[str, str]:
    p = os.path.join(ruwordnet_dir, f"synsets.{pos}.xml")
    names: Dict[str, str] = {}

    for _, el in ET.iterparse(p, events=("end",)):
        if el.tag == "synset":
            sid = (el.attrib.get("id") or "").strip()
            title = (el.attrib.get("ruthes_name") or el.attrib.get("name") or "").strip()
            if sid:
                names[sid] = title
        el.clear()

    return names


def build_synset_name_index(synset_names: Dict[str, str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for sid, name in synset_names.items():
        k = _clean_phrase_for_match(name)
        if k:
            idx[k].append(sid)
    for k, vals in list(idx.items()):
        seen = set()
        out = []
        for sid in vals:
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
        idx[k] = out
    return dict(idx)


def _dedup_ids(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def map_hypernym_to_ids(
    hyper: str,
    senses_idx: Dict[str, List[str]],
    synset_name_idx: Optional[Dict[str, List[str]]] = None,
    ids_per_hypernym: int = 5,
    enable_head_fallback: bool = True,
    enable_partial_name_match: bool = False,
) -> Tuple[List[str], str]:
    limit = max(0, int(ids_per_hypernym))
    if limit == 0:
        return [], "limit0"

    raw = _norm_text(hyper)
    k = _clean_phrase_for_match(raw)
    if not k:
        return [], "empty"

    ids = senses_idx.get(k, [])
    if ids:
        return ids[:limit], "sense_exact"

    if synset_name_idx:
        ids = synset_name_idx.get(k, [])
        if ids:
            return ids[:limit], "synset_name_exact"

    if enable_head_fallback:
        for head in _extract_head_candidates(k):
            ids = senses_idx.get(head, [])
            if ids:
                return ids[:limit], f"sense_head:{head}"
            if synset_name_idx:
                ids = synset_name_idx.get(head, [])
                if ids:
                    return ids[:limit], f"synset_name_head:{head}"

    if enable_partial_name_match and synset_name_idx:
        toks = [t for t in k.split() if len(t) > 2 and t not in RUS_STOPWORDS]
        if toks:
            scores: Dict[str, int] = defaultdict(int)
            for nk, sids in synset_name_idx.items():
                score = sum(1 for t in toks if t in nk)
                if score > 0:
                    for sid in sids:
                        scores[sid] += score
            if scores:
                ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                return [sid for sid, _ in ranked[:limit]], "synset_name_partial"

    return [], "no_match"


def write_predictions_tsv(out_path: str, rows: List[Tuple[str, str, str, float]]) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "cand", "cand_name", "prob"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}"])


def run_predict_py(predict_py: str, model_dir: str, words: List[str]) -> List[Tuple[str, str, str, float]]:
    if not words:
        return []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp_in:
        for w in words:
            tmp_in.write(w + "\n")
        tmp_in_path = tmp_in.name

    tmp_out = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".tsv")
    tmp_out_path = tmp_out.name
    tmp_out.close()

    cmd = [
        os.environ.get("PYTHON", "python"),
        predict_py,
        "--model_dir",
        model_dir,
        "--input_path",
        tmp_in_path,
        "--output_path",
        tmp_out_path,
    ]
    subprocess.check_call(cmd)

    pred_rows: List[Tuple[str, str, str, float]] = []
    with open(tmp_out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            w = (row.get("word") or "").strip()
            cand = (row.get("cand") or "").strip()
            cand_name = (row.get("cand_name") or "").strip()
            prob_s = (row.get("prob") or "0").strip()
            try:
                prob = float(prob_s)
            except Exception:
                prob = 0.0
            if w and cand:
                pred_rows.append((_norm_text(w).upper(), cand, _norm_text(cand_name), prob))

    for p in (tmp_in_path, tmp_out_path):
        try:
            os.unlink(p)
        except Exception:
            pass

    return pred_rows


def _prob_from_rank(rank_1based: int, mode: str) -> float:
    r = max(1, int(rank_1based))
    if mode == "harmonic":
        return 1.0 / r
    if mode == "linear":
        return max(0.0, 1.0 - (r - 1) * 0.05)
    if mode == "exp":
        return 0.85 ** (r - 1)
    return 1.0 / r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--ruwordnet_dir", required=True)
    ap.add_argument("--pos", default="N", choices=["N", "V", "A"])
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--ids_per_hypernym", type=int, default=5)
    ap.add_argument("--debug_tsv", default=None)

    ap.add_argument("--predict_py", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--fallback_model", action="store_true")

    ap.add_argument("--no_head_fallback", action="store_true")
    ap.add_argument("--use_synset_name_match", action="store_true")
    ap.add_argument("--partial_synset_name_match", action="store_true")
    ap.add_argument("--score_mode", default="harmonic", choices=["harmonic", "linear", "exp"])
    ap.add_argument("--upper_words", action="store_true")

    args = ap.parse_args()

    llm = read_llm_txt(args.in_txt)
    senses_idx = load_senses_index(args.ruwordnet_dir, args.pos)
    syn_names = load_synset_names(args.ruwordnet_dir, args.pos)
    synset_name_idx = build_synset_name_index(syn_names) if (args.use_synset_name_match or args.partial_synset_name_match) else None

    all_rows: List[Tuple[str, str, str, float]] = []
    debug_rows: List[Tuple[str, str, str, str, str, str]] = []

    words_out = []
    for w in llm.keys():
        ww = _norm_text(w)
        words_out.append(ww.upper() if args.upper_words else ww)

    fallback_rows_by_word: Dict[str, List[Tuple[str, str, str, float]]] = {}
    if args.fallback_model and args.predict_py and args.model_dir:
        fb = run_predict_py(args.predict_py, args.model_dir, words_out)
        tmp: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)
        for w, cand, cand_name, prob in fb:
            tmp[_norm_text(w).upper()].append((w, cand, cand_name, prob))
        for w in tmp:
            tmp[w].sort(key=lambda x: x[3], reverse=True)
        fallback_rows_by_word = dict(tmp)

    stats_pred_counts = []

    for term, hypers in llm.items():
        word_base = _norm_text(term)
        word_out = word_base.upper() if args.upper_words else word_base
        word_key_upper = _norm_text(word_out).upper()

        picked: List[Tuple[str, str, str, float]] = []
        seen_ids = set()

        for h in hypers:
            ids, reason = map_hypernym_to_ids(
                hyper=h,
                senses_idx=senses_idx,
                synset_name_idx=synset_name_idx,
                ids_per_hypernym=args.ids_per_hypernym,
                enable_head_fallback=not args.no_head_fallback,
                enable_partial_name_match=args.partial_synset_name_match,
            )

            mapped_pairs = []
            for sid in ids:
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                cname = _norm_text(syn_names.get(sid, ""))
                prob = _prob_from_rank(len(picked) + 1, args.score_mode)
                picked.append((word_out, sid, cname, prob))
                mapped_pairs.append(sid)
                if len(picked) >= args.topk:
                    break

            if args.debug_tsv is not None:
                debug_rows.append((
                    word_out,
                    _norm_text(h),
                    json.dumps(mapped_pairs, ensure_ascii=False),
                    str(bool(mapped_pairs)),
                    reason,
                    json.dumps(_extract_head_candidates(h), ensure_ascii=False),
                ))

            if len(picked) >= args.topk:
                break

        if args.fallback_model and len(picked) < args.topk:
            fb_rows = fallback_rows_by_word.get(word_key_upper, [])
            for _, sid, cname, _ in fb_rows:
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                prob = _prob_from_rank(len(picked) + 1, args.score_mode)
                picked.append((word_out, sid, _norm_text(cname), prob))
                if len(picked) >= args.topk:
                    break

        all_rows.extend(picked)
        stats_pred_counts.append(len(picked))

    write_predictions_tsv(args.out_tsv, all_rows)

    if args.debug_tsv is not None:
        with open(args.debug_tsv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "hypernym_in", "mapped_ids", "matched", "reason", "head_candidates"])
            for r in debug_rows:
                w.writerow(list(r))

    print(f"Wrote {args.out_tsv}")
    print(f"Words in txt: {len(llm)}")
    if stats_pred_counts:
        mn = min(stats_pred_counts)
        av = sum(stats_pred_counts) / len(stats_pred_counts)
        mx = max(stats_pred_counts)
        print(f"min/avg/max preds: {mn} / {av:.2f} / {mx}")
    if args.debug_tsv is not None:
        print(f"Wrote debug: {args.debug_tsv}")


if __name__ == "__main__":
    main()