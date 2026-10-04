import argparse
import json
from taxoenrich.utils import read_dataset


def norm_preds(preds):
    """
    Предсказания могут быть:
      - ["123-N","456-N",...]
      - [["123-N", prob], ["456-N", prob], ...]
    -> приводим к списку строковых id.
    """
    out = []
    for p in preds:
        if isinstance(p, (list, tuple)) and p:
            out.append(str(p[0]))
        else:
            out.append(str(p))
    return out


def norm_gold(g):
    """
    Gold (reference) может быть:
      - ["123-N","456-N",...]
      - [["123-N", ...], ["456-N", ...], ...]  (на всякий случай)
      - вложенные списки
    -> возвращаем set всех id-строк.
    """
    out = []

    def walk(v):
        if v is None:
            return

        if isinstance(v, (list, tuple)):
            # СЛУЧАЙ A: список строк -> берём ВСЕ
            if v and all(isinstance(x, str) for x in v):
                out.extend(v)
                return

            # СЛУЧАЙ B: вложенные списки/кортежи
            for t in v:
                walk(t)
            return

        # одиночное значение
        out.append(str(v))

    walk(g)
    return set(out)


def map_mrr_at_k(reference, submitted, k=15):
    """
    MAP@K и MRR@K как в твоём скрине:
      MAP@K = (1/N) * sum_n AP@K_n
      AP@K  = (1/R) * sum_{i=1..K} Precision@i * I[y_i=1]
      MRR@K = (1/N) * sum_n 1/rank_n   (rank первого релевантного в topK)
    где R = число релевантных ответов (gold) для слова.
    """
    ap_list = []
    rr_list = []

    # берём только те слова, что есть в reference (как обычно делают)
    for word, gold_raw in reference.items():
        gold = norm_gold(gold_raw)
        preds = norm_preds(submitted.get(word, []))[:k]

        if not gold:
            # если вдруг пустой gold — пропускаем, иначе деление на 0
            continue

        # --- AP@K ---
        hits = 0
        ap_sum = 0.0
        first_rank = None

        for i, cand in enumerate(preds, start=1):  # i = 1..K
            if cand in gold:
                hits += 1
                ap_sum += hits / i  # Precision@i
                if first_rank is None:
                    first_rank = i

        ap = ap_sum / len(gold)  # делим на R (как в статье)
        ap_list.append(ap)

        # --- RR@K ---
        rr = 0.0 if first_rank is None else 1.0 / first_rank
        rr_list.append(rr)

    if not ap_list:
        return 0.0, 0.0

    return sum(ap_list) / len(ap_list), sum(rr_list) / len(rr_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", required=True)
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    submitted = read_dataset(args.predict_path)
    reference = read_dataset(args.reference_path, json.loads)

    mapk, mrrk = map_mrr_at_k(reference, submitted, k=args.k)

    print(f"Results for {args.predict_path}:")
    print(f"\tMAP@{args.k} = {mapk}")
    print(f"\tMRR@{args.k} = {mrrk}")