"""Sentiment Analysis Demo — VADER (nltk)."""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

import gradio as gr
import gradipin

analyzer = SentimentIntensityAnalyzer()


def analyze(text: str) -> dict[str, float]:
    if not text.strip():
        return {}
    scores = analyzer.polarity_scores(text)
    return {
        "Positive": scores["pos"],
        "Negative": scores["neg"],
        "Neutral": scores["neu"],
    }


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter text", placeholder="Type a sentence to analyze..."),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis",
    description="Classifies text sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).",
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
