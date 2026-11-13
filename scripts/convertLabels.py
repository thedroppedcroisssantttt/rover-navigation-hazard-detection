# convert the dataset's pixel boxes to yolo format for renders and real photos

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

rootDir = Path(__file__).resolve().parents[1]
rawDir = rootDir / "data" / "raw" / "lunar"
renderW, renderH = 720, 480   # every render is 720x480
rockClass = 0                 # single class: largeRock


def toYolo(x, y, w, h, imgW, imgH):
    # pixel box (top-left x, y, width, height) -> yolo line, clipped to the image
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(imgW, x + w), min(imgH, y + h)
    if x2 <= x1 or y2 <= y1:
        return None  # box fell fully outside, skip it
    xc = (x1 + x2) / 2 / imgW
    yc = (y1 + y2) / 2 / imgH
    bw = (x2 - x1) / imgW
    bh = (y2 - y1) / imgH
    return f"{rockClass} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def readBoxes(csvPath, keepColors=None):
    # group csv rows by frame number; optionally keep only some colors
    boxesByFrame = defaultdict(list)
    with open(csvPath, newline="") as f:
        for row in csv.DictReader(f):
            if keepColors and row.get("Color") not in keepColors:
                continue
            boxesByFrame[int(row["Frame"])].append((
                float(row["TopLeftCornerX"]), float(row["TopLeftCornerY"]),
                float(row["Length"]), float(row["Height"])))
    return boxesByFrame


def convertRenders():
    # one label file per render, empty when the frame has no large rocks
    boxesByFrame = readBoxes(rawDir / "bounding_boxes.csv")
    labelDir = rawDir / "labels" / "render"   # yolo finds labels by swapping images/ -> labels/
    labelDir.mkdir(parents=True, exist_ok=True)

    renderPaths = sorted((rawDir / "images" / "render").glob("render*.png"))
    nBoxes, nEmpty = 0, 0
    for imgPath in renderPaths:
        frameId = int(imgPath.stem.replace("render", ""))
        lines = [l for l in (toYolo(*b, renderW, renderH) for b in boxesByFrame.get(frameId, [])) if l]
        (labelDir / f"{imgPath.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        nBoxes += len(lines)
        nEmpty += not lines
    print(f"renders: {len(renderPaths)} label files, {nBoxes} boxes, {nEmpty} empty frames")


def convertReal(keepColors):
    # resize each real photo to its mask frame, then write image + label
    realDir = rawDir / "real_moon_images"
    outImg = rootDir / "data" / "real" / "images"
    outLbl = rootDir / "data" / "real" / "labels"
    outImg.mkdir(parents=True, exist_ok=True)
    outLbl.mkdir(parents=True, exist_ok=True)

    nPhotos, nBoxes = 0, 0
    for cam in ("PCAM", "TCAM"):
        boxesByFrame = readBoxes(realDir / f"{cam}_bounding_boxes.csv", keepColors)
        for imgPath in sorted(realDir.glob(f"{cam}*.png")):
            frameId = int(imgPath.stem.replace(cam, ""))
            img = cv2.imread(str(imgPath))
            mask = cv2.imread(str(realDir / f"g_{imgPath.name}"), cv2.IMREAD_UNCHANGED)
            maskH, maskW = mask.shape[:2]
            # boxes live in the mask's frame, so match the photo to it
            if img.shape[:2] != (maskH, maskW):
                img = cv2.resize(img, (maskW, maskH), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(outImg / imgPath.name), img)

            lines = [l for l in (toYolo(*b, maskW, maskH) for b in boxesByFrame.get(frameId, [])) if l]
            (outLbl / f"{imgPath.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            nPhotos += 1
            nBoxes += len(lines)
    print(f"real: {nPhotos} photos, {nBoxes} boxes (colors: {', '.join(keepColors)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert lunar boxes to yolo format")
    parser.add_argument("--realColors", nargs="+", default=["Blue"],
                        help="which real-photo box colors to keep (Blue = large rocks)")
    args = parser.parse_args()
    convertRenders()
    convertReal(args.realColors)
