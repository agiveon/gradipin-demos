"""Image Classification Demo — MobileNet V2."""

from pathlib import Path

import gradio as gr
import gradipin
from transformers import pipeline

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

classifier = pipeline(
    "image-classification",
    model="google/mobilenet_v2_1.0_224",
)


def classify(image) -> dict[str, float]:
    if image is None:
        return {}
    results = classifier(image, top_k=5)
    return {r["label"]: r["score"] for r in results}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 Predictions"),
    title="Image Classification",
    description="Classifies images into 1000 ImageNet categories using MobileNet V2.",
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
