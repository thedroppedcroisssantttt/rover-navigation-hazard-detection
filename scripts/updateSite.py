# rebuild docs/results.json for the project page from experiments.csv

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

rootDir = Path(__file__).resolve().parents[1]
docsDir = rootDir / "docs"
metricKeys = ["val_map50", "real_map50", "real_map5095", "ms"]


def loadRuns(csvPath, order):
    # average each config over its seeds, return runs in display order + max seed count
    grouped = defaultdict(list)
    names = {}
    with open(csvPath, newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["key"]].append(row)
            names[row["key"]] = row["name"]
    runs = []
    for key, rows in grouped.items():
        run = {"key": key, "name": names[key]}
        for m in metricKeys:
            vals = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
            run[m] = round(sum(vals) / len(vals), 4) if vals else None
        runs.append(run)
    rank = {k: i for i, k in enumerate(order)}
    runs.sort(key=lambda r: rank.get(r["key"], len(rank)))
    return runs, max((len(v) for v in grouped.values()), default=1)


def findComparisons(imgDir, captions):
    # pair <name>_left.* with <name>_right.* for the slider
    pairs = []
    for left in sorted(imgDir.glob("*_left.*")):
        stem = left.name.rsplit("_left", 1)[0]
        right = next(imgDir.glob(f"{stem}_right.*"), None)
        if right is None:
            continue
        pairs.append({"title": stem, "before": f"img/{left.name}",
                      "after": f"img/{right.name}", "caption": captions.get(stem, "")})
    return pairs


def main():
    parser = argparse.ArgumentParser(description="refresh the project page data")
    parser.add_argument("--csv", default=str(rootDir / "results" / "experiments.csv"))
    parser.add_argument("--order", nargs="*", default=["baseline", "contrast", "blur", "noise", "lighting", "all"])
    parser.add_argument("--nSynthetic", type=int, default=9766)
    parser.add_argument("--captions", default=str(docsDir / "img" / "captions.json"),
                        help="optional json: image name -> caption")
    args = parser.parse_args()

    runs, seeds = loadRuns(Path(args.csv), args.order)
    capPath = Path(args.captions)
    captions = json.loads(capPath.read_text()) if capPath.exists() else {}
    imgDir = docsDir / "img"
    comparisons = findComparisons(imgDir, captions) if imgDir.exists() else []

    sidesPath = imgDir / "sides.json"
    sides = json.loads(sidesPath.read_text()) if sidesPath.exists() else {"left": "Baseline", "right": "Augmented"}
    out = {"placeholder": False, "n_synthetic": args.nSynthetic, "seeds": seeds,
           "runs": runs, "comparisons": comparisons, "sides": sides}
    (docsDir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote docs/results.json: {len(runs)} runs, {len(comparisons)} comparison pairs")


if __name__ == "__main__":
    main()
