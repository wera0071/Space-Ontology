#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple


# Ключевые слова домена РКТ для выбора смысла (sense)
DEFAULT_RKT_KEYWORDS = [
    "косм", "космич", "рак", "орбит", "спутник", "космодром", "марс", "луна",
    "аппарат", "полет", "посадк", "носител", "двигател", "ступен", "мисс",
    "марсоход", "луноход", "орбитер", "экзомарс", "межпланетн", "выведен",
]

# Алиасы: если слово не находится как лемма, пробуем эти варианты
ALIASES = {
    "марсоход": ["планетоход", "ровер", "марсианский ровер"],
    "орбитер": ["орбитальный аппарат", "космический аппарат"],
    "прилунение": ["посадка", "посадка на луну", "лунная посадка"],
    "ракетоплан": ["космоплан", "ракетный самолет", "ракетный самолёт"],
    "микроспутник": ["малый спутник", "искусственный спутник", "спутник"],
    "наноспутник": ["малый спутник", "спутник", "кубсат"],
    "экзомарс": ["космическая миссия", "миссия", "экспедиция"],
    "байконур": ["космодром", "космодром байконур"],
    "луноход": ["планетоход", "самоходный аппарат", "луноход"],
}


def norm_ru(s: str) -> str:
    return (s or "").strip().lower().replace("ё", "е")


def build_lemma_index_from_synsets(synsets_xml: str) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, str]]]:
    """
    Парсит synsets.N.xml:
      <synset id="147272-N" ruthes_name="..." definition="..." part_of_speech="N">
          <sense id="147272-N-712535">КРЕСТНЫЙ РОДИТЕЛЬ</sense>
      </synset>

    Возвращает:
      lemma2syn: lemma -> {synset_id, ...}
      syn_meta: synset_id -> {"ruthes_name":..., "definition":..., "pos":...}
    """
    lemma2syn: Dict[str, Set[str]] = defaultdict(set)
    syn_meta: Dict[str, Dict[str, str]] = {}

    for ev, el in ET.iterparse(synsets_xml, events=("end",)):
        if el.tag == "synset":
            sid = (el.attrib.get("id") or "").strip()
            if sid:
                syn_meta[sid] = {
                    "ruthes_name": norm_ru(el.attrib.get("ruthes_name", "")),
                    "definition": norm_ru(el.attrib.get("definition", "")),
                    "pos": (el.attrib.get("part_of_speech", "") or el.attrib.get("partOfSpeech", "") or "").strip(),
                }
                for s in el.findall("sense"):
                    lemma = norm_ru(s.text)
                    if lemma:
                        lemma2syn[lemma].add(sid)
            el.clear()

    return lemma2syn, syn_meta


def build_hypernym_graph(rel_xml: str) -> Dict[str, List[str]]:
    """
    child_id -> [parent_id,...] для гиперонимии.
    Берём:
      - name="hypernym"
      - name="instance hypernym"
    НЕ берём:
      - *hyponym* (включая instance hyponym)
    """
    hyp: Dict[str, List[str]] = defaultdict(list)

    for ev, el in ET.iterparse(rel_xml, events=("end",)):
        if el.tag == "relation":
            name = (el.attrib.get("name") or "").lower().strip()
            is_hyper = ("hypernym" in name) and ("hyponym" not in name)
            if is_hyper:
                child = (el.attrib.get("child_id") or "").strip()
                parent = (el.attrib.get("parent_id") or "").strip()
                if child and parent:
                    hyp[child].append(parent)
        el.clear()

    # удалим дубли, сохраняя порядок
    for c, ps in list(hyp.items()):
        seen = set()
        out = []
        for p in ps:
            if p not in seen:
                seen.add(p)
                out.append(p)
        hyp[c] = out

    return hyp


def score_synset(sid: str, syn_meta: Dict[str, Dict[str, str]], keywords: List[str]) -> int:
    m = syn_meta.get(sid, {})
    txt = (m.get("ruthes_name", "") + " " + m.get("definition", "")).strip()
    if not txt:
        return 0
    return sum(1 for kw in keywords if kw in txt)


def expand_hypernyms(start_sid: str, hyp: Dict[str, List[str]], depth: int) -> List[str]:
    """
    Возвращает множество hypernym синсетов до depth (1/2/3), исключая start_sid.
    """
    out: Set[str] = set()
    q = deque([(start_sid, 0)])
    while q:
        sid, d = q.popleft()
        if d >= depth:
            continue
        for p in hyp.get(sid, []):
            if p not in out:
                out.add(p)
                q.append((p, d + 1))
    return sorted(out)


def get_candidates(lemma2syn: Dict[str, Set[str]], word_norm: str) -> Tuple[List[str], str, str]:
    """
    Возвращает (candidates, used_form, how):
      how: exact | alias | none
    """
    c = lemma2syn.get(word_norm, set())
    if c:
        return sorted(list(c)), word_norm, "exact"

    for a in ALIASES.get(word_norm, []):
        a_norm = norm_ru(a)
        c = lemma2syn.get(a_norm, set())
        if c:
            return sorted(list(c)), a_norm, "alias"

    return [], word_norm, "none"


def choose_synset_with_nonempty_gold(
    candidates: List[str],
    syn_meta: Dict[str, Dict[str, str]],
    hyp: Dict[str, List[str]],
    keywords: List[str],
    depth: int,
) -> Tuple[str, List[str], List[Tuple[str, int]]]:
    """
    1) сортирует кандидатов по domain-score
    2) перебирает в этом порядке и выбирает первый sid, у которого expand_hypernyms(depth) непустой
    3) если у всех пусто — возвращает лучший по score (gold пустой)
    """
    scored = [(sid, score_synset(sid, syn_meta, keywords)) for sid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_sid = scored[0][0]

    for sid, sc in scored:
        gold = expand_hypernyms(sid, hyp, depth=depth)
        if gold:
            return sid, gold, scored

    return best_sid, [], scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruwordnet_dir", required=True)
    ap.add_argument("--words_path", required=True)
    ap.add_argument("--pos", default="N")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", default="reference_ruwordnet.tsv")
    ap.add_argument("--debug_out", default="reference_ruwordnet_debug.tsv")
    ap.add_argument("--keywords", default=",".join(DEFAULT_RKT_KEYWORDS))
    args = ap.parse_args()

    synsets_xml = os.path.join(args.ruwordnet_dir, f"synsets.{args.pos}.xml")
    rel_xml = os.path.join(args.ruwordnet_dir, f"synset_relations.{args.pos}.xml")

    if not os.path.exists(synsets_xml):
        raise SystemExit(f"Не найден файл: {synsets_xml}")
    if not os.path.exists(rel_xml):
        raise SystemExit(f"Не найден файл: {rel_xml}")

    keywords = [norm_ru(x) for x in args.keywords.split(",") if norm_ru(x)]

    lemma2syn, syn_meta = build_lemma_index_from_synsets(synsets_xml)
    hyp = build_hypernym_graph(rel_xml)

    with open(args.words_path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    ref_rows: List[Tuple[str, List[str]]] = []
    dbg_rows: List[Dict[str, object]] = []

    for w in words:
        word_u = w.upper()
        w_norm = norm_ru(w)

        candidates, used_form, how = get_candidates(lemma2syn, w_norm)

        if not candidates:
            ref_rows.append((word_u, []))
            dbg_rows.append({
                "word": word_u,
                "how_found": how,
                "used_form": used_form,
                "chosen_sense": "NO_SENSES",
                "gold_size": 0,
                "scored_candidates": [],
            })
            continue

        chosen_sid, gold, scored = choose_synset_with_nonempty_gold(
            candidates=candidates,
            syn_meta=syn_meta,
            hyp=hyp,
            keywords=keywords,
            depth=args.depth,
        )

        ref_rows.append((word_u, gold))
        dbg_rows.append({
            "word": word_u,
            "how_found": how,
            "used_form": used_form,
            "chosen_sense": chosen_sid,
            "gold_size": len(gold),
            "scored_candidates": scored,
        })

    # reference для eval.py (json.loads)
    with open(args.out, "w", encoding="utf-8") as g:
        for word_u, ids in ref_rows:
            g.write(f"{word_u}\t{json.dumps(ids, ensure_ascii=False)}\n")

    # debug TSV
    with open(args.debug_out, "w", encoding="utf-8") as g:
        g.write("word\thow_found\tused_form\tchosen_sense\tgold_size\tscored_candidates\n")
        for r in dbg_rows:
            g.write(
                f"{r['word']}\t{r['how_found']}\t{r['used_form']}\t{r['chosen_sense']}\t{r['gold_size']}\t"
                f"{json.dumps(r['scored_candidates'], ensure_ascii=False)}\n"
            )

    print("Wrote", args.out)
    print("Wrote", args.debug_out)


if __name__ == "__main__":
    main()