#!/usr/bin/env bash
# Полный прогон: 3 модели × 3 промпта × 50 терминов
# Перед запуском:
#   export OHMYLAMA_API_KEY=sk-lama-...
#   export GIGACHAT_AUTH_KEY=<base64>     # если используешь gigachat
#
# Все выходы попадут в outputs/<model>_<template>.txt и .tsv
set -e

TERMS=rkt_dev_terms_50.txt
JSONL=concepts.lii_thes.kosmos.json
REF=rkt_dev_reference_50.tsv
SOFT_REF=rkt_dev_reference_50_soft_depth2.tsv
OUT=outputs

mkdir -p "$OUT"

# ----- описание прогонов -----
# формат: provider|model|tag (tag используется в имени файла)
RUNS=(
  "openai|gpt-5.4|gpt54"
  "openai|claude-sonnet-4.6|claude46"
  "openai|deepseek-v4-pro|dsv4"
  "gigachat|GigaChat-Max|gigaMax"
)

TEMPLATES=(zero cot ontology)

for run in "${RUNS[@]}"; do
  IFS='|' read -r provider model tag <<< "$run"
  for tpl in "${TEMPLATES[@]}"; do
    out_txt="$OUT/${tag}_${tpl}.txt"
    out_tsv="$OUT/${tag}_${tpl}.tsv"
    debug_tsv="$OUT/${tag}_${tpl}_debug.tsv"

    if [ -f "$out_txt" ]; then
      echo ">>> SKIP $out_txt (уже есть)"
    else
      echo ">>> RUN  $tag / $tpl"
      python run_llm_batch.py \
        --provider "$provider" \
        --model "$model" \
        --terms "$TERMS" \
        --template "$tpl" \
        --out "$out_txt" \
        --concurrency 5
    fi

    echo ">>> MAP  $tag / $tpl"
    python llm_txt_to_predict_rkt_tsv.py \
      --in_txt "$out_txt" \
      --jsonl "$JSONL" \
      --out_tsv "$out_tsv" \
      --debug_tsv "$debug_tsv" \
      --topk 15 --include_english

    echo ">>> EVAL direct $tag / $tpl"
    python eval_true.py \
      --predict_path "$out_tsv" \
      --reference_path "$REF" \
      --k 15 | tee "$OUT/${tag}_${tpl}.metrics.txt"

    echo ">>> EVAL soft   $tag / $tpl"
    python eval_rkt_soft.py \
      --predict_path "$out_tsv" \
      --reference_path "$REF" \
      --jsonl "$JSONL" \
      --k 15 --depth 2 | tee "$OUT/${tag}_${tpl}.metrics.soft.txt"
  done
done

echo
echo "=== СВОДКА ==="
for f in "$OUT"/*.metrics.txt; do
  echo "--- $f ---"
  cat "$f"
done
