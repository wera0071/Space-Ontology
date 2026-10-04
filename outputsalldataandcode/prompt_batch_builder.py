"""
Собирает готовые промпты под батчевый прогон LLM.

Берёт список терминов (по одному в строке) и шаблон промпта,
разбивает термины на батчи по N штук и пишет .txt файлы,
которые можно копипастить в чат с моделью.

Шаблоны:
  zero       — zero-shot
  cot        — chain-of-thought
  ontology   — ontology-aligned

Формат ответа модели (то, что нужно скопировать обратно):
  ТЕРМИН => гипероним1; гипероним2; ...

Парсер llm_txt_to_predict_rkt_tsv.py принимает обе формы:
  ТЕРМИН<TAB>гипероним1; ...
  ТЕРМИН => гипероним1; ...
(если у тебя версия только под TAB — добавь обработку '=>' там, см. README в конце файла)

Пример:
  python prompt_batch_builder.py \\
      --terms rkt_dev_terms_50.txt \\
      --template ontology \\
      --batch_size 5 \\
      --out_dir prompts/ontology_b5
"""

import argparse
import os
from pathlib import Path

TEMPLATES = {
    "zero": (
        "Перечисли гиперонимы для каждого русского существительного из списка ниже.\n"
        "Формат ответа — строго по одной строке на термин:\n"
        "ТЕРМИН => гипероним1; гипероним2; гипероним3; ...\n"
        "Никаких пояснений, заголовков, нумерации. Максимум 15 гиперонимов на термин.\n"
        "\n"
        "Термины:\n"
        "{LIST}\n"
    ),
    "cot": (
        "Для каждого русского существительного из списка ниже сначала подумай про себя,\n"
        "какие гиперонимы наиболее вероятны, но не показывай ход рассуждений.\n"
        "Затем выведи только итоговый список (максимум 15) через \"; \".\n"
        "\n"
        "Формат ответа — строго по одной строке на термин:\n"
        "ТЕРМИН => гипероним1; гипероним2; гипероним3; ...\n"
        "Никаких пояснений, заголовков, нумерации.\n"
        "\n"
        "Термины:\n"
        "{LIST}\n"
    ),
    "ontology": (
        "Перечисли гиперонимы для каждого русского существительного из списка ниже\n"
        "так, как их обычно задают в тезаурусе (краткие словарные наименования).\n"
        "\n"
        "Требования:\n"
        "- только существительные или именные группы;\n"
        "- без пояснений, примеров и комментариев;\n"
        "- от более общего к менее общему;\n"
        "- избегай слишком частных и случайных ассоциаций;\n"
        "- максимум 15 гиперонимов на термин;\n"
        "- ответ строго по одной строке на термин в формате:\n"
        "  ТЕРМИН => гипероним1; гипероним2; гипероним3; ...\n"
        "\n"
        "Термины:\n"
        "{LIST}\n"
    ),
}


def read_terms(path: Path) -> list[str]:
    terms = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                terms.append(t)
    return terms


def build_batches(terms: list[str], size: int) -> list[list[str]]:
    return [terms[i:i + size] for i in range(0, len(terms), size)]


def render(template_key: str, batch: list[str]) -> str:
    tpl = TEMPLATES[template_key]
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    return tpl.replace("{LIST}", listing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", required=True, help="path to file with terms (one per line)")
    ap.add_argument("--template", required=True, choices=sorted(TEMPLATES.keys()))
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--single_file", action="store_true",
                    help="write one combined file with all batches separated by a marker")
    args = ap.parse_args()

    terms = read_terms(Path(args.terms))
    batches = build_batches(terms, args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_file:
        combined = []
        for i, batch in enumerate(batches, 1):
            combined.append(f"===== BATCH {i}/{len(batches)} =====\n")
            combined.append(render(args.template, batch))
            combined.append("\n")
        path = out_dir / f"{args.template}_b{args.batch_size}_all.txt"
        path.write_text("".join(combined), encoding="utf-8")
        print(f"wrote {path} ({len(batches)} batches, {len(terms)} terms)")
    else:
        for i, batch in enumerate(batches, 1):
            text = render(args.template, batch)
            path = out_dir / f"{args.template}_b{args.batch_size}_{i:02d}.txt"
            path.write_text(text, encoding="utf-8")
        print(f"wrote {len(batches)} prompt files to {out_dir}/ "
              f"({len(terms)} terms, batch size {args.batch_size})")

    answers_path = out_dir / "ANSWERS_TEMPLATE.txt"
    answers_path.write_text(
        "# Сюда складывай ответы модели одной кучей.\n"
        "# Формат строки (как в промпте): ТЕРМИН => гипероним1; гипероним2; ...\n"
        "# Между батчами можно ничего не вставлять — парсер сам разберёт.\n"
        "# После заполнения прогони:\n"
        "#   python llm_txt_to_predict_rkt_tsv.py \\\n"
        "#       --in_txt <этот файл> \\\n"
        "#       --jsonl concepts.lii_thes.kosmos.json \\\n"
        "#       --out_tsv <model>_<template>.tsv \\\n"
        "#       --debug_tsv <model>_<template>_debug.tsv \\\n"
        "#       --topk 15 --include_english\n",
        encoding="utf-8",
    )
    print(f"wrote answers template: {answers_path}")


if __name__ == "__main__":
    main()
