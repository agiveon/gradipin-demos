"""Sentiment Analysis Demo — distilbert-base-uncased-finetuned-sst-2-english."""

import gradio as gr
import gradipin
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


def analyze(text: str) -> dict[str, float]:
    if not text.strip():
        return {}
    results = classifier(text, top_k=2)
    return {r["label"]: r["score"] for r in results}


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter text", placeholder="Type a sentence to analyze..."),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis",
    description="Classifies text as Positive or Negative using DistilBERT.",
    examples=[
        ["I absolutely love this product, it's amazing!"],
        ["The movie was terrible and a complete waste of time."],
        ["The weather is okay today, nothing special."],
    ],
)


def main(port: int = 7860):
    gradipin.share(demo, app="sentiment-analysis")
    demo.launch(server_port=port)


if __name__ == "__main__":
    main()
