# Gradipin Demos

Three Gradio ML demos showcasing [Gradipin](https://pypi.org/project/gradipin/) — stable URLs for your Gradio demos with one line of code.

## Demos

| App | Model | Port |
|-----|-------|------|
| Sentiment Analysis | distilbert-base-uncased-finetuned-sst-2-english | 7860 |
| Image Classification | google/mobilenet_v2_1.0_224 | 7861 |
| Speech to Text | openai/whisper-tiny | 7862 |

## Setup

```bash
pip install -r requirements.txt
```

Then configure your Gradipin API key (get one from [gradipin.lovable.app/dashboard](https://gradipin.lovable.app/dashboard)):

```bash
# Option 1: environment variable
export GRADIPIN_KEY=gp_live_...

# Option 2: .env file (already included)
echo "GRADIPIN_KEY=gp_live_..." > .env
```

## Run All Demos

```bash
python run_all.py
```

This launches all 3 demos concurrently. Each gets a stable Gradipin URL that persists across restarts.

## Run Individually

```bash
python -m demos.sentiment
python -m demos.image_classify
python -m demos.speech_to_text
```
