# save side-by-side detections on real photos for the report and site slider

import argparse
import json
from pathlib import Path

import cv2

rootDir = Path(__file__).resolve().parents[1]
realDir = rootDir / "data" / "real"
defaultPhotos = ["PCAM3", "PCAM9", "TCAM2", "TCAM14"]   # mix of both cameras, near and far rocks
gtColor = (80, 200, 255)                                 # amber-ish in bgr for ground-truth boxes


def drawGroundTruth(imagePath):
    # draw the yolo label boxes for a real photo (only large rocks, as in the test set)
    img = cv2.imread(str(imagePath))
    h, w = img.shape[:2]
    lblPath = realDir / "labels" / f"{imagePath.stem}.txt"
    lines = lblPath.read_text().split("\n") if lblPath.exists() else []
    for line in filter(None, lines):
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), gtColor, 2)
    return img


def drawModel(model, imagePath, conf):
    # one model's detections on one photo
    return model.predict(str(imagePath), conf=conf, verbose=False)[0].plot(line_width=2)


def shrink(img, maxWidth):
    # keep the site light
    h, w = img.shape[:2]
    if w <= maxWidth:
        return img
    return cv2.resize(img, (maxWidth, int(h * maxWidth / w)), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="side-by-side detections on real lunar photos")
    parser.add_argument("--left", default=None, help="weights for the left side; omit for ground truth")
    parser.add_argument("--right", required=True, help="weights for the right side")
    parser.add_argument("--leftName", default=None)
    parser.add_argument("--rightName", default="Model")
    parser.add_argument("--images", nargs="*", default=None, help="real photo paths; default is 4 picks")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default=str(rootDir / "docs" / "img"))
    parser.add_argument("--maxWidth", type=int, default=1280)
    args = parser.parse_args()

    from ultralytics import YOLO   # imported here so --help works without it

    leftModel = YOLO(args.left) if args.left else None
    rightModel = YOLO(args.right)
    leftName = args.leftName or ("Ground truth" if leftModel is None else "Left model")

    images = [Path(p) for p in args.images] if args.images else [realDir / "images" / f"{n}.png" for n in defaultPhotos]
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    for old in list(outDir.glob("*_left.jpg")) + list(outDir.glob("*_right.jpg")):
        old.unlink()   # clear the previous pair set so the slider never mixes two comparisons

    for img in images:
        left = drawGroundTruth(img) if leftModel is None else drawModel(leftModel, img, args.conf)
        right = drawModel(rightModel, img, args.conf)
        for tag, arr in (("left", left), ("right", right)):
            dest = outDir / f"{img.stem}_{tag}.jpg"
            cv2.imwrite(str(dest), shrink(arr, args.maxWidth), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("saved pair", img.stem)

    (outDir / "sides.json").write_text(json.dumps({"left": leftName, "right": args.rightName}))
    print(f"labels: {leftName} | {args.rightName}")


if __name__ == "__main__":
    main()
