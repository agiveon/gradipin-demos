"""Speech-to-Text Demo — Whisper Tiny."""

from pathlib import Path

import gradio as gr
import gradipin
from transformers import pipeline

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
)


def transcribe(audio) -> str:
    if audio is None:
        return ""
    result = transcriber(audio)
    return result["text"]


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or upload audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech to Text",
    description="Transcribes audio to text using OpenAI Whisper (tiny).",
    examples=[
        [str(EXAMPLES_DIR / "sample_speech.flac")],
    ],
)


def main(port: int = 7862):
    gradipin.share(demo, app="speech-to-text")
    demo.launch(server_port=port)


if __name__ == "__main__":
    main()
