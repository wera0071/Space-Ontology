#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт расчёта метрик MAP@K и MRR@K для K=1,5,10,15 по всем 16 (+9) конфигурациям
(модель + промпт), использованным в курсовой.

Считаются три варианта эталона:
  direct -- эталон строится только по отношению ВЫШЕ из JSONL-дампа
            (это "канонический" эталон работы);
  soft   -- эталон расширен предками по ВЫШЕ глубины <=2 (мягкая оценка);
  ext    -- эталон расширен понятиями, связанными с термином отношениями
            АССОЦ, ЦЕЛОЕ, ЧАСТЬ (то, что просил Б.В. Добров: "не только
            по отношению ВЫШЕ, но и по другим").

Скрипт использует таксеньи утилиты тезауруса РКТ из rkt_thesaurus_utils.py.

Запуск (из корня проекта taxoenrich-master):
    python3 eval_all_k.py

Выходы:
  metrics_all.csv      -- длинная таблица: model,prompt,k, метрики во всех режимах;
  metrics_summary.txt  -- сводка лучших конфигураций по K и таблицы для копи-пасты.

Файл прогнозов (predictions) ожидается в outputsalldataandcode/<имя>.tsv
со столбцами word, cand, cand_name, prob (header первой строки).
Файл эталона рассчитан как outputsalldataandcode/rkt_dev_reference_50.tsv
со столбцами  word \t JSON-список conceptid.
JSONL-дамп тезауруса -- outputsalldataandcode/concepts.lii_thes.kosmos.json.
"""
import csv
import json
import os
import sys
from collections import defaultdict

# базовый путь -- текущая рабочая директория (корень проекта)
ROOT = os.getcwd()
OUT_DIR = os.path.join(ROOT, "outputsalldataandcode")
JSONL = os.path.join(OUT_DIR, "concepts.lii_thes.kosmos.json")
REF = os.path.join(OUT_DIR, "rkt_dev_reference_50.tsv")

# импорт утилит из проекта
sys.path.insert(0, ROOT)
from rkt_thesaurus_utils import (
    ancestors,
    build_parent_graph,
    build_term_index,
    clean_text,
    load_concepts,
    resolve_term,
)

KS = [1, 5, 10, 15]

# 16 конфигураций основной серии + 9 дополнительной = 25 строк
CONFIGS = [
    ("gpt-5.4",          "zero",          "gpt54_zero.tsv"),
    ("gpt-5.4",          "cot",           "gpt54_cot.tsv"),
    ("gpt-5.4",          "ontology",      "gpt54_ontology.tsv"),
    ("gpt-5.4",          "rkt_basic",     "gpt54_rkt_basic.tsv"),
    ("gpt-5.4",          "rkt_few_shot",  "gpt54_rkt_few_shot.tsv"),
    ("gpt-5.4",          "rkt_role",      "gpt54_rkt_role.tsv"),
    ("claude-sonnet-4.6","zero",          "claude46_zero.tsv"),
    ("claude-sonnet-4.6","cot",           "claude46_cot.tsv"),
    ("claude-sonnet-4.6","ontology",      "claude46_ontology.tsv"),
    ("claude-opus-4.7",  "zero",          "opus47_zero.tsv"),
    ("claude-opus-4.7",  "cot",           "opus47_cot.tsv"),
    ("claude-opus-4.7",  "ontology",      "opus47_ontology.tsv"),
    ("claude-opus-4.7",  "rkt_basic",     "opus47_rkt_basic.tsv"),
    ("claude-opus-4.7",  "rkt_few_shot",  "opus47_rkt_few_shot.tsv"),
    ("claude-opus-4.7",  "rkt_role",      "opus47_rkt_role.tsv"),
    ("deepseek-v4-pro",  "zero",          "dsv4_zero.tsv"),
    ("deepseek-v4-pro",  "cot",           "dsv4_cot.tsv"),
    ("deepseek-v4-pro",  "ontology",      "dsv4_ontology.tsv"),
    ("deepseek-v4-pro",  "rkt_basic",     "dsv4_rkt_basic.tsv"),
    ("deepseek-v4-pro",  "rkt_few_shot",  "dsv4_rkt_few_shot.tsv"),
    ("deepseek-v4-pro",  "rkt_role",      "dsv4_rkt_role.tsv"),
    ("glm-5.1",          "zero",          "glm51_zero.tsv"),
    ("glm-5.1",          "cot",           "glm51_cot.tsv"),
    ("kimi-k2.6",        "zero",          "kimi26_zero.tsv"),
    ("kimi-k2.6",        "cot",           "kimi26_cot.tsv"),
]


def read_predictions(path):
    """Читает TSV предсказаний (word, cand, cand_name, prob) и
    возвращает {word: [conceptid, ...]} в порядке их появления."""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].upper()
            if w == "WORD":
                continue
            out[w].append(str(parts[1]))
    return out


def read_reference(path):
    """Читает эталон вида  ТЕРМИН \t ["id1","id2",...]
    возвращает {ТЕРМИН: set(ids)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].upper()
            try:
                ids = [str(x) for x in json.loads(parts[1])]
            except Exception:
                continue
            out[w] = set(ids)
    return out


def map_mrr_at_k(ref, sub, k):
    """Стандартный подсчёт MAP@K и MRR@K на наборе терминов ref
    (см. eval_true.py в проекте). Полностью воспроизводит формулу
    из основного скрипта оценки."""
    aps, rrs = [], []
    for word, gold in ref.items():
        if not gold:
            continue
        preds = sub.get(word, [])[:k]
        hits = 0
        ap_sum = 0.0
        first_rank = None
        for i, cand in enumerate(preds, start=1):
            if cand in gold:
                hits += 1
                ap_sum += hits / i
                if first_rank is None:
                    first_rank = i
        aps.append(ap_sum / len(gold))
        rrs.append(0.0 if first_rank is None else 1.0 / first_rank)
    if not aps:
        return 0.0, 0.0
    return sum(aps) / len(aps), sum(rrs) / len(rrs)


def build_other_rel_gold(concepts, ref_terms,
                         relations=("АССОЦ", "ЦЕЛОЕ", "ЧАСТЬ")):
    """Для каждого термина выборки находит conceptid связанных понятий
    по отношениям, отличным от ВЫШЕ. Используется для расширенного эталона."""
    term_index, _, _ = build_term_index(concepts,
                                        include_english=False, use_morph=True)
    by_id = {str(c.get("conceptid")): c
             for c in concepts if c.get("conceptid") is not None}
    rels_upper = {r.upper() for r in relations}

    out = {}
    for term in ref_terms:
        ids, _ = resolve_term(term, term_index, use_morph=True)
        rel_ids = set()
        if ids:
            for cid in ids:
                concept = by_id.get(str(cid))
                if not concept:
                    continue
                for rel in concept.get("relats") or []:
                    rstr = clean_text(str(rel.get("relationstr", ""))).upper()
                    if rstr in rels_upper:
                        rid = str(rel.get("conceptid", "")).strip()
                        if rid:
                            rel_ids.add(rid)
        out[term] = rel_ids
    return out


def main():
    concepts = load_concepts(JSONL)
    direct_ref = read_reference(REF)

    # soft reference (depth<=2 по ВЫШЕ)
    graph = build_parent_graph(concepts, relation="ВЫШЕ")
    term_index, _, _ = build_term_index(concepts,
                                        include_english=False, use_morph=True)

    soft_ref = {}
    for w, gold in direct_ref.items():
        ids, _ = resolve_term(w, term_index, use_morph=True)
        soft = set(gold)
        if ids:
            soft |= set(ancestors(ids[0], graph, max_depth=2))
        soft_ref[w] = soft

    other_rel_gold = build_other_rel_gold(concepts, list(direct_ref.keys()))
    ext_ref = {w: (direct_ref[w] | other_rel_gold.get(w, set()))
               for w in direct_ref}

    rows = []
    for model, prompt, fname in CONFIGS:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            continue
        sub = read_predictions(path)
        for k in KS:
            d_map, d_mrr = map_mrr_at_k(direct_ref, sub, k)
            s_map, s_mrr = map_mrr_at_k(soft_ref, sub, k)
            e_map, e_mrr = map_mrr_at_k(ext_ref, sub, k)
            rows.append({
                "model": model, "prompt": prompt, "k": k,
                "direct_map": d_map, "direct_mrr": d_mrr,
                "soft_map":   s_map, "soft_mrr":   s_mrr,
                "ext_map":    e_map, "ext_mrr":    e_mrr,
            })

    csv_path = "metrics_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {csv_path}: {len(rows)} строк")

    # Сводки
    out_lines = []
    out_lines.append("=== Лучшая (модель, промпт) для каждого K, direct MAP ===")
    for k in KS:
        sub = [r for r in rows if r["k"] == k]
        sub.sort(key=lambda r: r["direct_map"], reverse=True)
        top = sub[0]
        out_lines.append(
            f"K={k}: {top['model']:18s} + {top['prompt']:14s}  "
            f"MAP={top['direct_map']:.3f} MRR={top['direct_mrr']:.3f}"
        )

    out_lines.append("\n=== Лучшая для каждого K, soft MAP (depth<=2 по ВЫШЕ) ===")
    for k in KS:
        sub = [r for r in rows if r["k"] == k]
        sub.sort(key=lambda r: r["soft_map"], reverse=True)
        top = sub[0]
        out_lines.append(
            f"K={k}: {top['model']:18s} + {top['prompt']:14s}  "
            f"MAP={top['soft_map']:.3f} MRR={top['soft_mrr']:.3f}"
        )

    out_lines.append("\n=== Лучшая для каждого K, ext MAP (ВЫШЕ+АССОЦ+ЦЕЛОЕ+ЧАСТЬ) ===")
    for k in KS:
        sub = [r for r in rows if r["k"] == k]
        sub.sort(key=lambda r: r["ext_map"], reverse=True)
        top = sub[0]
        out_lines.append(
            f"K={k}: {top['model']:18s} + {top['prompt']:14s}  "
            f"MAP={top['ext_map']:.3f} MRR={top['ext_mrr']:.3f}"
        )

    by_conf = defaultdict(dict)
    for r in rows:
        by_conf[(r["model"], r["prompt"])][r["k"]] = r

    for label, mfield, rfield in [
        ("direct", "direct_map", "direct_mrr"),
        ("soft",   "soft_map",   "soft_mrr"),
        ("ext",    "ext_map",    "ext_mrr"),
    ]:
        out_lines.append(f"\n=== Полная таблица K=1/5/10/15, {label} ===")
        out_lines.append(
            f"{'model':18s} {'prompt':14s} "
            + " ".join([f"MAP@{k} MRR@{k}" for k in KS])
        )
        for (m, p), per_k in by_conf.items():
            line = f"{m:18s} {p:14s}"
            for k in KS:
                r = per_k[k]
                line += f"  {r[mfield]:.3f} {r[rfield]:.3f}"
            out_lines.append(line)

    avg_extra = sum(len(other_rel_gold[w]) for w in direct_ref) / len(direct_ref)
    avg_soft = sum(len(soft_ref[w]) for w in direct_ref) / len(direct_ref)
    avg_direct = sum(len(direct_ref[w]) for w in direct_ref) / len(direct_ref)
    out_lines.append(
        f"\nСреднее число эталонных id на термин: "
        f"direct={avg_direct:.2f}, soft={avg_soft:.2f}, "
        f"ext={avg_direct+avg_extra:.2f} "
        f"(добавлено за счёт других отношений: {avg_extra:.2f})"
    )

    summary_path = "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
