# split renders into train/val, keep the real photos as test, write data.yaml

import argparse
import random
from pathlib import Path

rootDir = Path(__file__).resolve().parents[1]
dataDir = rootDir / "data"
rawDir = dataDir / "raw" / "lunar"


def readIds(name):
    # frame ids from one of the dataset's *_IDs.txt files
    path = rawDir / f"{name}_IDs.txt"
    return {int(x) for x in path.read_text().split() if x.strip()}


def relPath(p):
    # path relative to data/, with ./ so ultralytics resolves it next to the txt file
    return "./" + p.relative_to(dataDir).as_posix()


def main():
    parser = argparse.ArgumentParser(description="train/val/test split for the lunar dataset")
    parser.add_argument("--valFrac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=["cam_anomaly", "mismatch"],
                        help="which *_IDs.txt lists to drop from training")
    args = parser.parse_args()

    badIds = set()
    for name in args.exclude:
        ids = readIds(name)
        print(f"excluding {len(ids):4d} frames from {name}_IDs.txt")
        badIds |= ids

    renders = sorted((rawDir / "images" / "render").glob("render*.png"))
    keep = [p for p in renders if int(p.stem.replace("render", "")) not in badIds]

    # fixed seed so every run trains on the exact same split
    rng = random.Random(args.seed)
    rng.shuffle(keep)
    nVal = round(len(keep) * args.valFrac)
    valSet, trainSet = sorted(keep[:nVal]), sorted(keep[nVal:])

    testSet = sorted((dataDir / "real" / "images").glob("*.png"))
    if not testSet:
        raise SystemExit("no real test images, run convertLabels.py first")

    for name, items in (("train", trainSet), ("val", valSet), ("test", testSet)):
        (dataDir / f"{name}.txt").write_text("\n".join(relPath(p) for p in items) + "\n")

    (rootDir / "data.yaml").write_text(
        "# lunar hazard rocks, made by scripts/splitData.py\n"
        "# no path: key, so paths resolve next to this file (mac and colab)\n"
        "train: data/train.txt\n"
        "val: data/val.txt\n"
        "test: data/test.txt\n"
        "names:\n"
        "  0: largeRock\n")

    print(f"renders kept {len(keep)} of {len(renders)} -> train {len(trainSet)}, val {len(valSet)}")
    print(f"test (real photos): {len(testSet)}")
    print("wrote data/train.txt, data/val.txt, data/test.txt, data.yaml")


if __name__ == "__main__":
    main()
