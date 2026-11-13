# hugging face space demo: upload a lunar or mars terrain photo, get hazard rocks boxed

from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
weightsPath = HERE / "best.pt"
examplesDir = HERE / "examples"

# load once at startup, not per request
model = YOLO(str(weightsPath))


def sizeLabel(boxAreaFrac: float) -> str:
    # give a rough size category from the share of the image a box covers
    if boxAreaFrac > 0.05:
        return "large"
    if boxAreaFrac > 0.01:
        return "medium"
    return "small"


@spaces.GPU(duration=20)
def detect(image: np.ndarray, conf: float, iou: float):
    # run the detector on one uploaded rgb image
    if image is None:
        return None, "Upload an image to start.", []

    # zerogpu attaches a gpu only inside this function, cpu fallback when run locally
    device = 0 if torch.cuda.is_available() else "cpu"
    result = model.predict(image, conf=conf, iou=iou, device=device, verbose=False)[0]

    # plot() returns bgr, flip back to rgb for gradio
    annotated = result.plot(line_width=2)[:, :, ::-1]

    h, w = image.shape[:2]
    rows = []
    for i, (xyxy, score, cls) in enumerate(zip(result.boxes.xyxy.tolist(),
                                               result.boxes.conf.tolist(),
                                               result.boxes.cls.tolist()), start=1):
        x1, y1, x2, y2 = xyxy
        areaFrac = ((x2 - x1) * (y2 - y1)) / (w * h)
        rows.append([i, model.names[int(cls)], round(score, 3),
                     sizeLabel(areaFrac), f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"])

    speed = result.speed.get("inference", 0.0)
    if rows:
        summary = (f"### {len(rows)} hazard rock{'s' if len(rows) != 1 else ''} detected\n"
                   f"Inference took {speed:.0f} ms on {'GPU' if device == 0 else 'CPU'}. "
                   f"Highest confidence: {max(r[2] for r in rows):.2f}.")
    else:
        summary = (f"### No hazard rocks found above {conf:.2f} confidence\n"
                   "Try lowering the confidence threshold.")
    return annotated, summary, rows


examples = [[str(p), 0.25, 0.5] for p in sorted(examplesDir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

with gr.Blocks(title="Lunar Hazard Rock Detector", theme=gr.themes.Default()) as demo:
    gr.Markdown(
        "# Lunar Hazard Rock Detector\n"
        "YOLOv8 trained **only on rendered lunar scenes**, with augmentations that make renders look more like real "
        "photos. Upload a real lunar or Mars terrain image to see what it detects."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Terrain image")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
            iou = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="NMS IoU threshold")
            btn = gr.Button("Detect hazards", variant="primary")
        with gr.Column():
            outImg = gr.Image(type="numpy", label="Detections")
            outMd = gr.Markdown()
    outTbl = gr.Dataframe(headers=["#", "class", "confidence", "size (rough)", "box x1, y1, x2, y2"],
                           label="Detections", interactive=False)

    btn.click(detect, [inp, conf, iou], [outImg, outMd, outTbl])
    inp.upload(detect, [inp, conf, iou], [outImg, outMd, outTbl])
    if examples:
        gr.Examples(examples, [inp, conf, iou], [outImg, outMd, outTbl], fn=detect, cache_examples=False)

if __name__ == "__main__":
    demo.launch()
