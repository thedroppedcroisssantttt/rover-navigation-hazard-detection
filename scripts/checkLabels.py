# draw converted boxes on sample images to check they line up

import argparse
import random
from pathlib import Path

import cv2

rootDir = Path(__file__).resolve().parents[1]


def drawYolo(imgPath, lblPath, outPath):
    # read a yolo label file and draw its boxes on the image
    img = cv2.imread(str(imgPath))
    h, w = img.shape[:2]
    lines = lblPath.read_text().split("\n") if lblPath.exists() else []
    n = 0
    for line in filter(None, lines):
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        n += 1
    cv2.putText(img, f"{imgPath.name}  boxes: {n}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(outPath), img)


def main():
    parser = argparse.ArgumentParser(description="visual check of converted labels")
    parser.add_argument("--n", type=int, default=6, help="how many renders to draw")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outDir = rootDir / "results" / "labelCheck"
    outDir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rawDir = rootDir / "data" / "raw" / "lunar"
    renders = sorted((rawDir / "images" / "render").glob("*.png"))
    for p in rng.sample(renders, args.n):
        drawYolo(p, rawDir / "labels" / "render" / f"{p.stem}.txt", outDir / f"check_{p.name}")

    realDir = rootDir / "data" / "real"
    reals = sorted((realDir / "images").glob("*.png"))
    for p in rng.sample(reals, min(4, len(reals))):
        drawYolo(p, realDir / "labels" / f"{p.stem}.txt", outDir / f"check_{p.name}")

    print(f"saved {args.n + min(4, len(reals))} images to {outDir}")


if __name__ == "__main__":
    main()
