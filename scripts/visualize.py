# turn the experiment log into the report figures and tables

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no window, just files
import matplotlib.pyplot as plt

rootDir = Path(__file__).resolve().parents[1]
runOrder = ["baseline", "contrast", "blur", "noise", "lighting", "all"]
metricKeys = ["val_map50", "real_map50", "real_map5095", "ms"]

# report style: times new roman to match the paper, quiet grays, one dark highlight
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
dustGray, nightBlack, ruleGray = "#8f8c84", "#111214", "#d9d5cc"


def loadRuns(csvPath):
    # average every metric over seeds; returns rows in runOrder plus seed counts
    byKey = defaultdict(list)
    names = {}
    with open(csvPath, newline="") as f:
        for row in csv.DictReader(f):
            byKey[row["key"]].append(row)
            names[row["key"]] = row["name"]
    runs = []
    for key in sorted(byKey, key=lambda k: runOrder.index(k) if k in runOrder else 99):
        rows = byKey[key]
        run = {"key": key, "name": names[key], "nSeeds": len(rows)}
        for m in metricKeys:
            vals = [float(r[m]) for r in rows]
            run[m] = sum(vals) / len(vals)
            # sample std, only meaningful with 2+ seeds
            run[m + "_std"] = (sum((v - run[m]) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        run["gap"] = run["real_map50"] - run["val_map50"]
        runs.append(run)
    return runs


def plotRealMap(runs, outPath):
    # horizontal bars of real-photo mAP50, best in black, baseline as a dashed reference
    best = max(runs, key=lambda r: r["real_map50"])
    base = next((r for r in runs if r["key"] == "baseline"), None)
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(runs) + 1.2))
    names = [r["name"] for r in runs][::-1]
    vals = [r["real_map50"] for r in runs][::-1]
    errs = [r["real_map50_std"] for r in runs][::-1]
    colors = [nightBlack if r is best else dustGray for r in runs][::-1]
    ax.barh(names, vals, color=colors, height=0.6)
    for y, (v, e) in enumerate(zip(vals, errs)):
        if e > 0:   # error bar only for configs run with 2+ seeds
            ax.errorbar(v, y, xerr=e, color="#555", capsize=3, lw=1)
        ax.text(v + e + 0.012, y, f"{v:.3f}", va="center", fontsize=10)
    if base:
        ax.axvline(base["real_map50"], color="#555", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mAP@0.5 on real lunar photographs")
    ax.xaxis.grid(True, color=ruleGray, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(outPath)
    plt.close(fig)


def plotValVsReal(runs, outPath):
    # paired bars: renders (val) vs real photos, the gap between them is the story
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = range(len(runs))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [r["val_map50"] for r in runs], w, color=ruleGray, edgecolor="#999", label="Rendered (validation)")
    ax.bar([x + w / 2 for x in xs], [r["real_map50"] for r in runs], w, color=nightBlack, label="Real photographs (test)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["name"].replace(" (no augmentation)", "") for r in runs], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mAP@0.5")
    ax.yaxis.grid(True, color=ruleGray, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.savefig(outPath)
    plt.close(fig)


def plotCurves(runsDir, outPath):
    # val mAP50 per epoch from each ultralytics results.csv; skips quietly if none
    logs = sorted(Path(runsDir).glob("*_s*/results.csv"))
    if not logs:
        print("no run logs found in", runsDir, "-> skipping training curves")
        return False
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    shades = ["#111214", "#5e5b55", "#8f8c84", "#b9b3a6", "#3a3a37", "#a39e93"]
    for i, log in enumerate(logs):
        with open(log, newline="") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        epochs = [int(float(r["epoch"])) for r in rows]
        mapVals = [float(r["metrics/mAP50(B)"]) for r in rows]
        ax.plot(epochs, mapVals, color=shades[i % len(shades)], lw=1.4, label=log.parent.name.rsplit("_s", 1)[0])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP@0.5 (renders)")
    ax.yaxis.grid(True, color=ruleGray, lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outPath)
    plt.close(fig)
    return True


def fmtVal(run, m, d=3):
    # value, plus ± std when there are repeat seeds
    s = f"{run[m]:.{d}f}"
    return s + (f" ± {run[m + '_std']:.{d}f}" if run["nSeeds"] > 1 else "")


def writeTables(runs, outDir):
    # same table twice: markdown for the readme, booktabs latex for the report
    best = max(runs, key=lambda r: r["real_map50"])
    head = ["Configuration", "Val AP50", "Real AP50", "Real AP50:95", "Gap", "ms/img"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in runs:
        cells = [r["name"], fmtVal(r, "val_map50"), fmtVal(r, "real_map50"), fmtVal(r, "real_map5095"),
                 f"{r['gap']:+.3f}", f"{r['ms']:.1f}"]
        md.append("| " + " | ".join(f"**{c}**" if r is best else c for c in cells) + " |")
        texCells = [c.replace("±", r"$\pm$") for c in cells]
        if r is best:
            texCells = [r"\textbf{" + c + "}" for c in texCells]
        tex.append(" & ".join(texCells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (outDir / "table_results.md").write_text("\n".join(md) + "\n")
    (outDir / "table_results.tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="figures and tables from the experiment log")
    parser.add_argument("--csv", default=str(rootDir / "results" / "experiments.csv"))
    parser.add_argument("--runsDir", default=str(rootDir / "runs"))
    parser.add_argument("--outDir", default=str(rootDir / "results" / "figures"))
    args = parser.parse_args()

    outDir = Path(args.outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    runs = loadRuns(args.csv)
    plotRealMap(runs, outDir / "fig_realMap50.png")
    plotValVsReal(runs, outDir / "fig_valVsReal.png")
    plotCurves(args.runsDir, outDir / "fig_trainingCurves.png")
    writeTables(runs, outDir)

    for r in runs:
        print(f"{r['key']:9s} real AP50 {r['real_map50']:.3f}  gap {r['gap']:+.3f}  seeds {r['nSeeds']}")
    print("wrote figures and tables to", outDir)


if __name__ == "__main__":
    main()
