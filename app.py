# app.py — Bilingual (Arabic + English) Sentiment Analysis
# --------------------------------------------------------
# يدعم: نص واحد، CSV، PDF، DOCX + فحص البيئة
# مضاف: قواعد عربية لفك الحياد + تنزيل النماذج من Google Drive تلقائياً

import os, sys, re, json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ---- PDF & DOCX readers (اختياري) ----
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

# ===== Lazy TensorFlow import =====
TF_IMPORT_ERROR = None
tf = None
tokenizer_from_json = None
pad_sequences = None

def ensure_tf():
    """Import TensorFlow only when needed."""
    global tf, tokenizer_from_json, pad_sequences, TF_IMPORT_ERROR
    if tf is not None:
        return True, None
    try:
        import tensorflow as _tf
        from tensorflow.keras.preprocessing.text import tokenizer_from_json as _tok_json
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
        tf = _tf
        tokenizer_from_json = _tok_json
        pad_sequences = _pad
        return True, None
    except Exception as e:
        TF_IMPORT_ERROR = e
        return False, str(e)

# ------------------------
# Config
# ------------------------
st.set_page_config(page_title="💬 Sentiment | تحليل المشاعر", page_icon="💬", layout="wide")

# 👇 عدّل هذا الرابط لو تغيّر
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Aw-A95Ep-6ByshJoveXgCLi1yOCk2o45?usp=share_link"

DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96
CLASSES_FALLBACK = ["negative", "neutral", "positive"]

# ------------------------
# Helpers: Google Drive fetch
# ------------------------
def _lang_ready(lang_dir: Path) -> bool:
    """يتأكد أن ملفات اللغة مكتملة."""
    if not lang_dir.exists():
        return False
    tok_ok = (lang_dir / "tokenizer.json").exists()
    lbl_ok = (lang_dir / "label_map.json").exists()
    keras_ok = any((lang_dir / f).exists() for f in [f"{lang_dir.name}_best.keras", f"{lang_dir.name}_final.keras",
                                                     f"{lang_dir.name}_best.h5", f"{lang_dir.name}_final.h5"])
    saved_ok = (lang_dir / "saved_model").exists()
    return tok_ok and lbl_ok and (keras_ok or saved_ok)

def ensure_models_on_disk(model_root: Path) -> Tuple[bool, str]:
    """
    يتأكد أن المجلد موجود وجاهز. لو ناقص، يحاول تنزيله من Google Drive باستخدام gdown.
    يرجّع (ok, msg).
    """
    try:
        ar_ok = _lang_ready(model_root / "ar")
        en_ok = _lang_ready(model_root / "en")
        if ar_ok and en_ok:
            return True, "Models already present."

        # حاول تنزيل من Google Drive
        import gdown
        model_root.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading models from Google Drive..."):
            # gdown يدعم تنزيل مجلد كامل من رابط drive/folders/...
            gdown.download_folder(
                url=GOOGLE_DRIVE_FOLDER_URL,
                output=str(model_root),
                quiet=False,
                use_cookies=False
            )

        ar_ok = _lang_ready(model_root / "ar")
        en_ok = _lang_ready(model_root / "en")
        if ar_ok and en_ok:
            return True, "Models downloaded from Google Drive."
        else:
            return False, "Downloaded, but expected structure not found (need ar/ and en/ inside)."
    except ModuleNotFoundError:
        return False, "gdown not installed. Add `gdown` to requirements.txt."
    except Exception as e:
        return False, f"Download failed: {e}"

# ------------------------
# Lang utils
# ------------------------
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
AR_DIACRITICS = r"[\u0617-\u061A\u064B-\u0652\u0670]"

def detect_language_simple(text: str) -> str:
    return "ar" if ARABIC_RE.search(str(text)) else "en"

def ar_normalize(s: str) -> str:
    s = str(s)
    s = re.sub(AR_DIACRITICS, "", s)
    s = re.sub(r"[ـ]+", "", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ؤ","و").replace("ئ","ي").replace("ة","ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_text(txt: str, lang: str) -> str:
    return ar_normalize(txt) if lang == "ar" else txt

# ------------------------
# Arabic rules/keywords to break neutrality
# ------------------------
AR_NEG = {
    "حزين","زعلان","تعيس","سيئ","سيء","سئ","مكتئب","محبط","تعبان","كاره",
    "مزعج","رديء","سيئة","كارثي","مقرف","فظيع","زفت","مخيّب","أسوأ","ممل",
    "كارثة","رداءة","غبن","قرف","ندمت","تافه","سيئين"
}
AR_POS = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب",
    "عجبني","مذهل","مسعد","هايل","كويس","ممتازه","تحفه","خيالي",
    "يفوز","حبيت","أفضل","مرضي","مبهر","روعة","يجنن","رهيب","مره حلو","فخم"
}
AR_NEGATIONS = {"مو","مش","ليس","ما","مو مره","مهو","مهوب","ولا"}
AR_INTENSIFIERS = {"جداً","جدًا","مره","مرة","بشكل كبير","مرة كثير","قوي"}
EMOJI_POS = {"😊","😍","🤩","😁","👍","💖","✨","👏","🥰"}
EMOJI_NEG = {"😞","😡","🤬","😢","👎","💔","😠","😭"}

EXCLAMATION_BOOST = 0.06
INTENSIFIER_BOOST = 0.07
RULE_CONF = 0.55
LOW_CONF = 0.60
NEU_MARGIN = 0.18

def _rule_score_ar(text: str) -> str | None:
    t = ar_normalize(text)
    has_pos = any(w in t for w in AR_POS) or any(e in text for e in EMOJI_POS)
    has_neg = any(w in t for w in AR_NEG) or any(e in text for e in EMOJI_NEG)
    negation = any(n in t for n in AR_NEGATIONS)
    if negation:
        if has_pos and not has_neg:
            has_pos, has_neg = False, True
        elif has_neg and not has_pos:
            has_pos, has_neg = True, False
    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    return None

def override_ar_prediction(
    text: str,
    label: str,
    probs: np.ndarray,
    classes: List[str],
    margin: float = NEU_MARGIN
) -> tuple[str, float]:
    try:
        i_neg = classes.index("negative")
        i_neu = classes.index("neutral")
        i_pos = classes.index("positive")
    except ValueError:
        return label, float(np.max(probs))
    p_neg, p_neu, p_pos = float(probs[i_neg]), float(probs[i_neu]), float(probs[i_pos])
    if label == "neutral":
        if p_neu - p_neg <= margin:
            label = "negative"
        if p_neu - p_pos <= margin:
            label = "positive"
    top_p = max(p_neg, p_neu, p_pos)
    rule = _rule_score_ar(text)
    if rule and (label == "neutral" or top_p < LOW_CONF):
        label = rule
        top_p = max(top_p, RULE_CONF)
    boost = 0.0
    excl = text.count("!")
    if excl >= 2: boost += EXCLAMATION_BOOST
    if any(w in text for w in AR_INTENSIFIERS): boost += INTENSIFIER_BOOST
    if label == "positive" and boost > 0:
        top_p = min(0.99, top_p + boost)
    if label == "negative" and boost > 0 and excl >= 3:
        top_p = min(0.99, top_p + boost/2)
    return label, top_p

# ------------------------
# Loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_lang_assets(model_root: Path, lang: str):
    ok, err = ensure_tf()
    if not ok:
        raise RuntimeError(f"TensorFlow import failed: {err}")

    lang_dir = Path(model_root) / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Language folder not found: {lang_dir}")

    # tokenizer
    tok_path = lang_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {lang_dir}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # label_map
    label_map_path = lang_dir / "label_map.json"
    if label_map_path.exists():
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]
        except Exception:
            classes = CLASSES_FALLBACK
    else:
        classes = CLASSES_FALLBACK

    # model file
    candidates = [
        lang_dir / f"{lang}_best.keras",
        lang_dir / f"{lang}_final.keras",
        lang_dir / f"{lang}_best.h5",
        lang_dir / f"{lang}_final.h5",
        lang_dir / "saved_model",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {lang_dir}")

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

def _predict_batch(texts: List[str], model_root: Path) -> pd.DataFrame:
    ok, err = ensure_tf()
    if not ok:
        raise RuntimeError(f"TensorFlow import failed: {err}")

    langs = ["ar" if ARABIC_RE.search(t or "") else "en" for t in texts]
    rows = []
    cache: Dict[str, Tuple[Any, Any, Any]] = {}

    for lang in ("ar", "en"):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        if not idxs:
            continue
        if lang not in cache:
            tok, classes, model = load_lang_assets(model_root, lang)
            cache[lang] = (tok, classes, model)
        else:
            tok, classes, model = cache[lang]

        subset = [preprocess_text(texts[i], lang) for i in idxs]
        seq = tok.texts_to_sequences(subset)
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model(X, training=False).numpy()  # يعمل مع SavedModel وKeras
        pred_idx = np.argmax(probs, axis=1)

        for j, i_global in enumerate(idxs):
            base_label = classes[int(pred_idx[j])] if int(pred_idx[j]) < len(classes) else str(int(pred_idx[j]))
            conf = float(probs[j, pred_idx[j]])
            label = base_label
            if lang == "ar":
                label, conf = override_ar_prediction(texts[i_global], base_label, probs[j], classes)
            row = {
                "text": texts[i_global],
                "lang": lang,
                "label": label,
                "confidence": float(conf),
            }
            for ci, cname in enumerate(classes):
                row[f"p_{cname}"] = float(probs[j, ci]) if ci < probs.shape[1] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

# ------------------------
# Readers
# ------------------------
def read_pdf(file) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    texts = []
    reader = PdfReader(file)
    for pg in reader.pages:
        t = (pg.extract_text() or "").strip()
        if t: texts.append(t)
    return texts

def read_docx(file) -> List[str]:
    if Document is None:
        raise RuntimeError("python-docx not installed")
    texts = []
    doc = Document(file)
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t: texts.append(t)
    return texts

def read_csv(file) -> pd.DataFrame:
    df = None
    for enc in ("utf-8","utf-8-sig","latin-1","cp1256"):
        try:
            file.seek(0); df = pd.read_csv(file, encoding=enc); break
        except UnicodeDecodeError:
            continue
    return df if df is not None else pd.read_csv(file)

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("⚙️ Settings | الإعدادات")
    model_root_str = st.text_input("Model directory | مسار الموديلات", value=str(DEFAULT_MODEL_DIR))
    model_root = Path(model_root_str)
    st.caption("سيتم تنزيل المجلد من Google Drive تلقائياً إذا لم يكن موجودًا.")
    if st.button("⬇️ Download/Refresh from Google Drive"):
        ok_dl, msg = ensure_models_on_disk(model_root)
        if ok_dl: st.success(msg)
        else: st.error(msg)

# حاول التحميل تلقائيًا لو ناقص
if not (_lang_ready(model_root / "ar") and _lang_ready(model_root / "en")):
    ok_dl, msg = ensure_models_on_disk(model_root)
    if ok_dl:
        st.info(msg)
    else:
        st.warning(msg)

# ------------------------
# Title
# ------------------------
st.title("💬 Sentiment Analysis | تحليل المشاعر (AR/EN)")
st.caption("By: Rana Alsalloum, Yaqeen Adnan, Reem Al-Rshedi")

tabs = st.tabs([
    "📝 Single Text | نص واحد",
    "📎 File (CSV / PDF / DOCX) | ملف",
    "🩺 Environment | البيئة"
])

# ------------------------
# Tab 1 - Single text
# ------------------------
with tabs[0]:
    ok_tf, err_tf = ensure_tf()
    if not ok_tf:
        st.error("TensorFlow غير متاح. افتحي تبويب Environment لمشاهدة السبب.")
    else:
        t = st.text_area("Enter text (Arabic or English):", height=140,
                         placeholder="مثال: انا سعيد اليوم / I love this product")
        if st.button("Predict"):
            if t.strip():
                try:
                    df = _predict_batch([t], model_root)
                    row = df.iloc[0]
                    lang_badge = "🇸🇦 عربي" if row["lang"] == "ar" else "🇬🇧 English"
                    st.success(f"**Language:** {lang_badge}\n\n**Prediction:** `{row['label']}`  |  **Confidence:** `{row['confidence']:.3f}`")
                    prob_cols = [c for c in df.columns if c.startswith("p_")]
                    if prob_cols:
                        st.markdown("**Probabilities:**")
                        st.dataframe(df[prob_cols].T.rename(columns={0: "probability"}))
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("اكتب نصًا أولاً.")

# ------------------------
# Tab 2 - File upload
# ------------------------
with tabs[1]:
    ok_tf, err_tf = ensure_tf()
    if not ok_tf:
        st.error("TensorFlow غير متاح.")
    else:
        up = st.file_uploader("Upload CSV / PDF / DOCX", type=["csv","pdf","docx"])
        if st.button("Run") and up:
            try:
                if up.name.lower().endswith(".csv"):
                    df_in = read_csv(up)
                    if "text" not in df_in.columns:
                        df_in = df_in.rename(columns={df_in.columns[0]: "text"})
                    texts = df_in["text"].astype(str).tolist()
                elif up.name.lower().endswith(".pdf"):
                    texts = read_pdf(up)
                else:
                    texts = read_docx(up)

                out_df = _predict_batch(texts, model_root)
                st.dataframe(out_df, use_container_width=True)
                st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(str(e))

# ------------------------
# Tab 3 - Environment
# ------------------------
with tabs[2]:
    st.write("**Python:**", sys.version)
    ok_tf, err_tf = ensure_tf()
    st.write("**TensorFlow imported?**", ok_tf)
    if ok_tf:
        st.write("TF version:", tf.__version__)
        try:
            st.write("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
        except Exception:
            pass
    else:
        st.error(err_tf)

    st.write("**Model root:**", str(model_root.resolve()))
    for lang in ("ar","en"):
        d = model_root / lang
        st.write(f"**{lang} folder exists?**", d.exists())
        if d.exists():
            try:
                st.code("\n".join([p.name for p in sorted(d.iterdir())][:50]), language="bash")
            except:
                pass
