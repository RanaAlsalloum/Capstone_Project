# Deploying the Bilingual Sentiment Streamlit App

## What you need
- `app.py` (already provided)
- Trained models under this structure (copy the whole folder into your repo/project):
```
bilingual_sentiment_model/
  ├── ar/
  │   ├── ar_best.keras   (or ar_final.keras)
  │   ├── ar_final.h5     (optional legacy)
  │   ├── tokenizer.json
  │   └── label_map.json
  └── en/
      ├── en_best.keras   (or en_final.keras)
      ├── en_final.h5     (optional legacy)
      ├── tokenizer.json
      └── label_map.json
```
- `requirements.txt` (included here)

> Tip: Train locally with the notebook, then copy the produced `bilingual_sentiment_model/` folder into the same directory as `app.py` before deploying.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## Deploy on Streamlit Community Cloud
1. Push these files to a public GitHub repo:
   - `app.py`
   - `requirements.txt`
   - `bilingual_sentiment_model/` (with the `ar/` and `en/` subfolders and files inside)
2. Go to https://share.streamlit.io, connect your GitHub, and select the repo.
3. Set the entry point to `app.py` and deploy.

## Testing
Use examples like:
- English: `I love this course!` → *positive*
- English: `This is terrible` → *negative*
- Arabic: `اليوم جميل` → *neutral/positive (depends on training data)`
- Arabic: `التجربة سيئة جدًا` → *negative*

## Troubleshooting
- **Missing files**: Ensure `tokenizer.json` and `label_map.json` exist under each language folder.
- **Wrong labels**: Labels must match your dataset (e.g., `positive`, `negative`, `neutral`).
  If your datasets use different casing/names, retrain or edit `label_map.json` to match.
- **Large models**: If the repo is too big for Streamlit Cloud, consider removing `.h5` files and keep only `.keras`.
- **Apple Silicon**: For local dev on M‑series macs, you may need `tensorflow-macos==2.15.0` instead of `tensorflow-cpu`.
