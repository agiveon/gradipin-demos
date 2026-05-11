"""Image Classification Demo — MobileNet V2 via ONNX Runtime."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

import gradio as gr
import gradipin

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

MODEL_REPO = "onnx-community/mobilenet_v2_1.0_224"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")
config_path = hf_hub_download(repo_id=MODEL_REPO, filename="config.json")

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

with open(config_path) as f:
    id2label = json.load(f)["id2label"]


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW


def classify(image) -> dict[str, float]:
    if image is None:
        return {}
    tensor = preprocess(image)
    logits = session.run(None, {input_name: tensor})[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    top5 = probs.argsort()[-5:][::-1]
    return {id2label[str(i)]: float(probs[i]) for i in top5}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 Predictions"),
    title="Image Classification",
    description="Classifies images into 1000 ImageNet categories using MobileNet V2 (ONNX Runtime).",
    examples=[
        [str(EXAMPLES_DIR / "cat.jpg")],
        [str(EXAMPLES_DIR / "dog.jpg")],
        [str(EXAMPLES_DIR / "coffee.jpg")],
        [str(EXAMPLES_DIR / "flower.jpg")],
    ],
)


def main(port: int = 7861):
    gradipin.share(demo, app="image-classify")
    demo.launch(server_port=port)


if __name__ == "__main__":
    main()
