import argparse
import pandas as pd
from collections import defaultdict

def load_pred(path: str):
    df = pd.read_csv(path, sep="\t")
    # ожидаем колонки: word, cand, cand_name, prob
    df = df.sort_values(["word", "prob"], ascending=[True, False])
    preds = defaultdict(list)
    for w, sub in df.groupby("word"):
        preds[w] = sub["cand"].astype(str).tolist()
    return preds

def load_ref(path: str):
    df = pd.read_csv(path, sep="\t")
    # ожидаем минимум: word, cand
    refs = defaultdict(set)
    for w, sub in df.groupby("word"):
        refs[w] = set(sub["cand"].astype(str).tolist())
    return refs

def ap_at_k(pred_list, rel_set, k):
    hit = 0
    s = 0.0
    for i, c in enumerate(pred_list[:k], start=1):
        if c in rel_set:
            hit += 1
            s += hit / i
    # нормируем на число релевантных (или на min(|rel|, k))
    denom = min(len(rel_set), k)
    return (s / denom) if denom > 0 else 0.0

def rr_at_k(pred_list, rel_set, k):
    for i, c in enumerate(pred_list[:k], start=1):
        if c in rel_set:
            return 1.0 / i
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_path", required=True)
    ap.add_argument("--reference_path", required=True)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    preds = load_pred(args.predict_path)
    refs = load_ref(args.reference_path)

    # считаем только по словам, которые есть и в предсказаниях, и в reference
    words = sorted(set(preds.keys()) & set(refs.keys()))
    if not words:
        raise SystemExit("Нет пересечения слов между predict и reference (проверь формат/колонки).")

    ap_scores = []
    rr_scores = []

    for w in words:
        ap_scores.append(ap_at_k(preds[w], refs[w], args.k))
        rr_scores.append(rr_at_k(preds[w], refs[w], args.k))

    mapk = sum(ap_scores) / len(ap_scores)
    mrrk = sum(rr_scores) / len(rr_scores)

    print(f"Words evaluated: {len(words)}")
    print(f"MAP@{args.k}: {mapk:.6f}")
    print(f"MRR@{args.k}: {mrrk:.6f}")

if __name__ == "__main__":
    main()
