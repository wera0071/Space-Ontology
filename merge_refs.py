import json
import argparse

def read_ref(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            w, js = line.split("\t", 1)
            d[w] = set(json.loads(js))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Finish_reference_ruwordnet.tsv")
    ap.add_argument("--b", required=True, help="rkt_reference_clean.tsv")
    ap.add_argument("--out", default="reference_merged.tsv")
    ap.add_argument("--drop_empty", action="store_true", help="выкинуть строки где итоговый gold пустой")
    args = ap.parse_args()

    A = read_ref(args.a)
    B = read_ref(args.b)

    words = sorted(set(A) | set(B))
    kept = 0
    total = 0

    with open(args.out, "w", encoding="utf-8") as g:
        for w in words:
            ids = sorted(list(A.get(w, set()) | B.get(w, set())))
            if args.drop_empty and not ids:
                continue
            g.write(f"{w}\t{json.dumps(ids, ensure_ascii=False)}\n")
            total += 1
            if ids:
                kept += 1

    print("Wrote", args.out)
    print("Rows:", total, "Nonempty:", kept)

if __name__ == "__main__":
    main()