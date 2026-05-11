"""Speech-to-Text Demo — Whisper Tiny via faster-whisper (CTranslate2)."""

from pathlib import Path

import gradio as gr
import gradipin

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

_model = None


def _load_model():
    global _model
    if _model is not None:
        return
    from faster_whisper import WhisperModel

    _model = WhisperModel("tiny", compute_type="int8")


def transcribe(audio) -> str:
    if audio is None:
        return ""
    _load_model()
    segments, _ = _model.transcribe(audio)
    return " ".join(seg.text.strip() for seg in segments)


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or upload audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech to Text",
    description="Transcribes audio to text using OpenAI Whisper (tiny) via faster-whisper.",
    examples=[
        [str(EXAMPLES_DIR / "sample_speech.flac")],
    ],
)


def main(port: int = 7862):
    gradipin.share(demo, app="speech-to-text")
    demo.launch(server_port=port)


if __name__ == "__main__":
    main()
