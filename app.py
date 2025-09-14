# app.py — Bilingual (Arabic + English) Sentiment Analysis
# --------------------------------------------------------
# يدعم: نص واحد، CSV، PDF، DOCX + فحص البيئة
# مضاف: تطبيع عربي + قواعد لفك الحياد (نفي/تعجب/مكثّفات/إيموجي)
# يعمل بهيكل مجلدات: ./ar و ./en (أو ./bilingual_sentiment_model/ar & en)

import os, sys, re, io, json
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
    """Import TensorFlow + keras utils only when needed."""
    global tf, tokenizer_from_json, pad_sequences, TF_IMPORT_ERROR
    if tf is not None:
        return True, None
    try:
        import tensorflow as _tf
        # Keras 3: tokenizer_from_json من keras.preprocessing.text
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
st.caption("By: Rana Alsalloum, Yaqeen Adnan, Reem Al-Rshedi")

MAX_LEN = 96
CLASSES_FALLBACK = ["negative", "neutral", "positive"]

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
    return ar_normalize(txt) if lang == "ar" else str(txt)

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

EXCLAMATION_BOOST = 0.06   # تعزيز بسيط لو في ! كثيرة
INTENSIFIER_BOOST = 0.07   # تعزيز لو في جداً/مره
RULE_CONF = 0.55           # ثقة افتراضية لو قلبنا بالقاعدة
LOW_CONF = 0.60            # لو أعلى احتمال أقل من هذا نسمح للقاعدة تقلب
NEU_MARGIN = 0.18          # سماحية لفك الحياد

def _rule_score_ar(text: str) -> str | None:
    """قواعد سريعة: تُرجِع 'positive' أو 'negative' أو None."""
    t = ar_normalize(text)
    has_pos = any(w in t for w in AR_POS) or any(e in text for e in EMOJI_POS)
    has_neg = any(w in t for w in AR_NEG) or any(e in text for e in EMOJI_NEG)

    # نفي بسيط: "مو حلو" = سلبي ، "مو سيء" = إيجابي تقريباً
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
    """يُرجع (label, confidence) بعد القواعد والتحسينات العربية."""
    try:
        i_neg = classes.index("negative")
        i_neu = classes.index("neutral")
        i_pos = classes.index("positive")
    except ValueError:
        return label, float(np.max(probs))

    p_neg, p_neu, p_pos = float(probs[i_neg]), float(probs[i_neu]), float(probs[i_pos])

    # لو محايد وبالقرب من أحد الطرفين، نفك الحياد
    if label == "neutral":
        if p_neu - p_neg <= margin:
            label = "negative"
        if p_neu - p_pos <= margin:
            label = "positive"

    # قواعد لغوية/إيموجي إذا الثقة ضعيفة أو ما زال محايد
    top_p = max(p_neg, p_neu, p_pos)
    rule = _rule_score_ar(text)
    if rule and (label == "neutral" or top_p < LOW_CONF):
        label = rule
        top_p = max(top_p, RULE_CONF)

    # تعزيز حسب علامات التعجب والمكثِّفات
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
# Model discovery / loading
# ------------------------
def resolve_model_root(user_text: str | None = None) -> Path:
    """
    يحدد أين المجلدين ar و en.
    الترتيب:
      1) قيمة المستخدم إن وجدت
      2) مجلد العمل الحالي (./ar, ./en)
      3) ./bilingual_sentiment_model
    """
    candidates = []
    if user_text:
        candidates.append(Path(user_text))
    candidates.append(Path("."))
    candidates.append(Path("bilingual_sentiment_model"))
    for base in candidates:
        if (base / "ar").exists() and (base / "en").exists():
            return base
    # لو ما وجد، نرجع الحالي — وسيظهر خطأ أوضح لاحقًا
    return Path(user_text or ".")

@st.cache_resource(show_spinner=False)
def load_lang_assets(base_dir: Path, lang: str):
    ok, err = ensure_tf()
    if not ok:
        raise RuntimeError(f"TensorFlow import failed: {err}")

    lang_dir = Path(base_dir) / lang
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

    # model file (نفضل SavedModel إن وجد)
    candidates = [
        lang_dir / "saved_model",
        lang_dir / f"{lang}_best.keras",
        lang_dir / f"{lang}_final.keras",
        lang_dir / f"{lang}_best.h5",
        lang_dir / f"{lang}_final.h5",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {lang_dir}")

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

def _predict_batch(texts: List[str], base_dir: Path) -> pd.DataFrame:
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
            tok, classes, model = load_lang_assets(base_dir, lang)
            cache[lang] = (tok, classes, model)
        else:
            tok, classes, model = cache[lang]

        subset = [preprocess_text(texts[i], lang) for i in idxs]
        seq = tok.texts_to_sequences(subset)
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(X, verbose=0)
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
            # احتمالات مسماة (إن توفرت)
            for ci, cname in enumerate(classes):
                row[f"p_{cname}"] = float(probs[j, ci])
            rows.append(row)

    # أعيدي ترتيب النتائج بنفس ترتيب الإدخال
    out = pd.DataFrame(rows)
    if not out.empty:
        out["__ord"] = range(len(out))
        out = out.sort_values("__ord").drop(columns="__ord")
    return out

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
    user_path = st.text_input(
        "Model base folder | مسار مجلد الموديلات",
        value=".",  # نتوقع ./ar و ./en بجانب app.py
        help="ضع المسار الذي يحتوي مجلدَي ar و en. اتركيه '.' إذا كانا بجانب app.py مباشرة."
    )
    base_dir = resolve_model_root(user_path)
    st.caption(f"Using base: `{base_dir.resolve()}` → يتوقع: `{base_dir}/ar` و `{base_dir}/en`")

# ------------------------
# Tabs
# ------------------------
st.title("💬 Sentiment Analysis | تحليل المشاعر (AR/EN)")
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
                    df = _predict_batch([t], base_dir)
                    if df.empty:
                        st.error("لم أستطع توليد تنبؤ. تأكدي من وجود المجلدين ar و en بشكل صحيح.")
                    else:
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
                st.warning("اكتبي نصًا أولاً.")

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

                out_df = _predict_batch(texts, base_dir)
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

    st.write("**Looking for model folders:**")
    for name in ("ar", "en"):
        d = base_dir / name
        st.write(f"- `{d}` exists? →", d.exists())
        if d.exists():
            try:
                st.code("\n".join([p.name for p in sorted(d.iterdir())]), language="bash")
            except Exception:
                pass

    st.info("التوقع يحتاج: لكل لغة `saved_model/` + `tokenizer.json` + `label_map.json`.")
