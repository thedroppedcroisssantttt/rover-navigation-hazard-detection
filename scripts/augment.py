# build an augmented copy of the training set, photometric only so labels copy over

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

rootDir = Path(__file__).resolve().parents[1]
dataDir = rootDir / "data"


# ---- the four transforms (img is uint8 bgr, rng is a numpy generator) ----

def contrastExposure(img, rng):
    # harsh lunar lighting: random exposure gain, contrast stretch, then gamma
    x = img.astype(np.float32) / 255.0
    gain = rng.uniform(0.6, 1.5)                 # over/under exposure
    contrast = rng.uniform(0.6, 1.7)             # stretch around the image mean
    mean = x.mean()
    x = (x - mean) * contrast + mean
    x = np.clip(x * gain, 0, 1)
    gamma = rng.uniform(0.7, 1.5)                # pushes midtones darker/lighter
    x = x ** gamma
    return (x * 255).astype(np.uint8)


def gaussianBlur(img, rng):
    # camera softness / slight motion: odd kernel 3-7, sigma 0.5-2
    k = int(rng.choice([3, 5, 7]))
    sigma = rng.uniform(0.5, 2.0)
    return cv2.GaussianBlur(img, (k, k), sigma)


def sensorNoise(img, rng):
    # sensor grain renders don't have: gaussian read noise + a bit of shot noise
    x = img.astype(np.float32)
    readStd = rng.uniform(3, 12)                       # in 0-255 units
    shot = rng.normal(0, 1, x.shape) * np.sqrt(np.maximum(x, 1)) * rng.uniform(0.2, 0.6)
    x = x + rng.normal(0, readStd, x.shape) + shot
    return np.clip(x, 0, 255).astype(np.uint8)


def directionalLighting(img, rng):
    # low sun angle: multiply by a linear brightness ramp at a random direction
    h, w = img.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(0.3, 0.8)                   # how dark the far side gets
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # project each pixel onto the light direction, scale to 0-1
    proj = (xx - w / 2) * np.cos(angle) + (yy - h / 2) * np.sin(angle)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    ramp = 1.0 - strength + strength * proj           # dark side -> bright side
    x = img.astype(np.float32) * ramp[..., None]
    return np.clip(x, 0, 255).astype(np.uint8)


augSets = {
    "contrast": [contrastExposure],
    "blur": [gaussianBlur],
    "noise": [sensorNoise],
    "lighting": [directionalLighting],
    # order matters a little: lighting/exposure first, then blur, noise last (like a real camera)
    "all": [directionalLighting, contrastExposure, gaussianBlur, sensorNoise],
}


def applyAug(img, transforms, rng, p):
    # run each transform with probability p
    for fn in transforms:
        if rng.random() < p:
            img = fn(img, rng)
    return img


def readList(path):
    # image paths from a ./-style list file, made absolute
    return [dataDir / line[2:] for line in path.read_text().split() if line.startswith("./")]


def makePreview(augName, transforms, srcPaths, rng, outPath, n=4):
    # top row originals, bottom row augmented (forced on, p=1) for the report
    picks = [srcPaths[i] for i in rng.choice(len(srcPaths), n, replace=False)]
    tops, bottoms = [], []
    for p in picks:
        img = cv2.imread(str(p))
        aug = applyAug(img.copy(), transforms, rng, 1.0)
        tops.append(cv2.resize(img, (360, 240)))
        bottoms.append(cv2.resize(aug, (360, 240)))
    grid = np.vstack([np.hstack(tops), np.hstack(bottoms)])
    cv2.putText(grid, "original", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(grid, f"augmented: {augName}", (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outPath), grid)
    print("preview ->", outPath)


def main():
    parser = argparse.ArgumentParser(description="photometric augmentation for the lunar renders")
    parser.add_argument("--aug", required=True, choices=sorted(augSets))
    parser.add_argument("--p", type=float, default=0.5, help="chance each transform fires")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--previewOnly", action="store_true")
    args = parser.parse_args()

    transforms = augSets[args.aug]
    rng = np.random.default_rng(args.seed)
    trainPaths = readList(dataDir / "train.txt")

    makePreview(args.aug, transforms, trainPaths, np.random.default_rng(args.seed + 100),
                rootDir / "results" / f"augPreview_{args.aug}.png")
    if args.previewOnly:
        return

    outImg = dataDir / "aug" / args.aug / "images"
    outLbl = dataDir / "aug" / args.aug / "labels"
    outImg.mkdir(parents=True, exist_ok=True)
    outLbl.mkdir(parents=True, exist_ok=True)

    newList = []
    for i, src in enumerate(trainPaths):
        img = applyAug(cv2.imread(str(src)), transforms, rng, args.p)
        dst = outImg / src.name
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])   # low compression = faster
        # labels don't move since nothing geometric changed
        srcLbl = src.parent.parent.parent / "labels" / src.parent.name / f"{src.stem}.txt"
        shutil.copyfile(srcLbl, outLbl / f"{src.stem}.txt")
        newList.append("./" + dst.relative_to(dataDir).as_posix())
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(trainPaths)}")

    (dataDir / f"train_{args.aug}.txt").write_text("\n".join(newList) + "\n")
    baseYaml = (rootDir / "data.yaml").read_text()
    (rootDir / f"data_{args.aug}.yaml").write_text(
        baseYaml.replace("train: data/train.txt", f"train: data/train_{args.aug}.txt"))
    print(f"{args.aug}: {len(newList)} augmented images, wrote data_{args.aug}.yaml")


if __name__ == "__main__":
    main()
