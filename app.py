# app.py — Bilingual (Arabic + English) Sentiment App (Streamlit)
# ---------------------------------------------------------------
# - Auto-detects Arabic vs English
# - Routes to the correct model (ar/en)
# - Loads tokenizer & label_map per language
# - Arabic: normalization + lightweight keyword override for borderline neutral
#
# How to run:
#   pip install streamlit tensorflow==2.15.0 keras==2.15.0 pandas numpy
#   # (or tensorflow-macos on Apple Silicon)
#   streamlit run app.py --server.port 8501
#
# Expected folder structure (default):
#   bilingual_sentiment_model/
#     ├── ar/ (ar_best.keras|ar_final.keras, tokenizer.json, label_map.json)
#     └── en/ (en_best.keras|en_final.keras, tokenizer.json, label_map.json)

import json, re, os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Bilingual Sentiment (AR/EN)", page_icon="💬", layout="centered")

# =========================
# Global Config
# =========================
DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96  # must match training

# =========================
# Language Detection
# =========================
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
def detect_language_simple(text: str) -> str:
    return "ar" if ARABIC_RE.search(str(text)) else "en"

# =========================
# Arabic Normalization + Keyword Rules
# =========================
AR_DIACRITICS = r"[\u0617-\u061A\u064B-\u0652\u0670]"

def ar_normalize(s: str) -> str:
    """Lightweight Arabic normalization to reduce model confusion."""
    s = str(s)
    s = re.sub(AR_DIACRITICS, "", s)       # remove diacritics
    s = re.sub(r"[ـ]+", "", s)             # remove tatweel
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ؤ","و").replace("ئ","ي").replace("ة","ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

AR_NEG = {
    "حزين","زعلان","تعيس","سيئ","سيء","مكتئب","محبط","تعبان","كاره","اكره","كرهت",
    "مزعج","كارثي","سئ","اسوء","أسوأ","سلبية","سلبيا"
}
AR_POS = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احبه","احب","عجبني",
    "ممتازه","مبهج","ايجابي","مسرور"
}

OVERRIDE_MARGIN = 0.20  # إذا كان الفرق بسيط، نسمح بقلب المحايد

# =========================
# Model / Tokenizer Loading
# =========================
@st.cache_resource(show_spinner=False)
def load_lang_assets(model_dir: Path, lang: str):
    """Load tokenizer, label map, and model for a given language."""
    lang_dir = model_dir / lang
    with open(lang_dir / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())
    with open(lang_dir / "label_map.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    model_path = lang_dir / f"{lang}_best.keras"
    if not model_path.exists():
        model_path = lang_dir / f"{lang}_final.keras"
    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

# =========================
# Inference Helpers
# =========================
def _apply_ar_override_if_needed(label, probs, classes, text_norm):
    """
    If model predicted 'neutral' for Arabic and the text contains strong positive/negative keywords,
    and neutral is only slightly higher than the alternative, flip the label.
    """
    if label != "neutral":
        return label, float(probs[classes.index(label)])

    pmap   = {classes[i]: float(probs[i]) for i in range(len(classes))}
    has_neg = any(w in text_norm for w in AR_NEG)
    has_pos = any(w in text_norm for w in AR_POS)

    if has_neg and (pmap.get("neutral",0) - pmap.get("negative",0) <= OVERRIDE_MARGIN):
        return "negative", pmap.get("negative", 0.0)
    if has_pos and (pmap.get("neutral",0) - pmap.get("positive",0) <= OVERRIDE_MARGIN):
        return "positive", pmap.get("positive", 0.0)

    return label, pmap.get(label, 0.0)

def predict_one(text: str, model_dir: Path):
    lang = detect_language_simple(text)
    # normalize Arabic before tokenization
    txt = ar_normalize(text) if lang == "ar" else text

    tok, classes, model = load_lang_assets(model_dir, lang)
    seq = tok.texts_to_sequences([txt])
    X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(X, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = classes[idx]
    conf  = float(probs[idx])

    # Arabic override for borderline neutral
    if lang == "ar":
        label, conf = _apply_ar_override_if_needed(label, probs, classes, txt)

    return {
        "lang": lang,
        "label": label,
        "confidence": conf,
        "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }

def predict_batch(texts, model_dir: Path):
    outputs, assets = [], {}
    for t in texts:
        lang = detect_language_simple(t)
        txt  = ar_normalize(t) if lang == "ar" else t

        if lang not in assets:
            assets[lang] = load_lang_assets(model_dir, lang)
        tok, classes, model = assets[lang]

        seq = tok.texts_to_sequences([txt])
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(X, verbose=0)[0]
        idx   = int(np.argmax(probs))
        label = classes[idx]
        conf  = float(probs[idx])

        if lang == "ar":
            label, conf = _apply_ar_override_if_needed(label, probs, classes, txt)

        outputs.append({
            "text": t,
            "lang": lang,
            "label": label,
            "confidence": conf,
            **{f"p_{c}": float(probs[j]) for j, c in enumerate(classes)}
        })
    return outputs

# =========================
# UI
# =========================
st.title("💬 Bilingual Sentiment (Arabic + English)")
st.caption("Auto-detects AR/EN, normalizes Arabic, and gently fixes borderline neutral Arabic cases.")

with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
    model_dir = Path(model_dir)
    st.info("Make sure your models (ar/en) + tokenizer + label_map are under this folder.")
    st.write(f"Override margin: `{OVERRIDE_MARGIN}` (edit in code to adjust)")

st.subheader("Single Text")
text = st.text_area("Enter text (Arabic or English):", height=120,
                    placeholder="مثال: انا حزين  /  Example: I am sad")
if st.button("Predict sentiment", type="primary"):
    if text.strip():
        try:
            res = predict_one(text, model_dir)
            lang_badge = "🇸🇦 AR" if res["lang"] == "ar" else "🇬🇧 EN"
            st.markdown(f"**Language:** {lang_badge}")
            st.markdown(f"**Prediction:** `{res['label']}`  \n**Confidence:** `{res['confidence']:.3f}`")
            with st.expander("Show probabilities"):
                st.json(res["probs"])
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text.")

st.divider()

st.subheader("Batch CSV")
st.write("Upload a CSV with a column named **`text`**. The app will auto-route each row to AR/EN model.")
csv_file = st.file_uploader("Upload CSV", type=["csv"])
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        csv_file.seek(0)
        df = pd.read_csv(csv_file, encoding="latin-1")
    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'.")
    else:
        if st.button("Run batch prediction"):
            try:
                rows = df["text"].astype(str).tolist()
                out_df = pd.DataFrame(predict_batch(rows, model_dir))
                st.success(f"Predicted {len(out_df)} rows.")
                st.dataframe(out_df, use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes,
                                   file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.caption("Tip: You can also paste multiple lines in the single-text box and run them one by one.")

st.divider()
st.markdown("Made with ❤️ — For best quality, consider fine-tuning the Arabic model with a few dozen curated positive/negative samples.")
