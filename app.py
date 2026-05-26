import streamlit as st
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RoadScan — AI Damage Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS — Luxury Minimalist
# -----------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --gold: #C9A84C;
    --gold-light: #E8C96A;
    --gold-dim: rgba(201,168,76,0.15);
    --cream: #F5F0E8;
    --charcoal: #1A1A18;
    --charcoal-mid: #2C2C28;
    --charcoal-soft: #3A3A34;
    --muted: rgba(245,240,232,0.45);
    --border: rgba(201,168,76,0.25);
  }

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--charcoal) !important;
    color: var(--cream) !important;
  }

  .main .block-container {
    background-color: var(--charcoal) !important;
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 720px;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Hero wordmark ── */
  .hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 11px;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
  }

  .hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    font-size: 3.4rem;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--cream);
    margin: 0 0 0.6rem 0;
  }

  .hero-title em {
    font-style: italic;
    color: var(--gold-light);
  }

  .hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 14px;
    color: var(--muted);
    letter-spacing: 0.02em;
    line-height: 1.7;
    max-width: 440px;
    margin-bottom: 2.5rem;
  }

  .gold-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 2rem 0;
    border: none;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: var(--charcoal-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease;
  }

  [data-testid="stFileUploader"]:hover {
    border-color: rgba(201,168,76,0.55) !important;
  }

  [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
  }

  [data-testid="stFileUploaderDropzone"] > div {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 300 !important;
  }

  [data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--gold) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.25s ease !important;
  }

  [data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--gold-dim) !important;
    border-color: var(--gold) !important;
  }

  /* ── Upload label ── */
  .upload-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.6rem;
    display: block;
  }

  /* ── Displayed image ── */
  [data-testid="stImage"] img {
    border-radius: 3px !important;
    border: 1px solid var(--border) !important;
    filter: brightness(0.96) contrast(1.04);
  }

  /* ── Result card ── */
  .result-card {
    background: var(--charcoal-mid);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2.6rem 2.8rem 2.4rem;
    margin-top: 2rem;
    position: relative;
    overflow: hidden;
  }

  .result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold), var(--gold-light), transparent);
  }

  .result-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.34em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 1.4rem 0;
  }

  /* ── Two-column stat row ── */
  .result-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 2rem;
  }

  .stat-cell {
    padding: 1.4rem 1.6rem;
  }

  .stat-cell:first-child {
    border-right: 1px solid var(--border);
  }

  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.38);
    margin: 0 0 0.55rem 0;
    display: block;
  }

  .stat-value {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 400;
    font-size: 2rem;
    letter-spacing: 0.01em;
    color: var(--cream);
    line-height: 1.1;
    margin: 0;
  }

  .stat-value-gold {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 2.2rem;
    color: var(--gold-light);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
  }

  /* ── Confidence bar (custom, inside card) ── */
  .conf-bar-wrap {
    margin-bottom: 1.6rem;
  }

  .conf-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
  }

  .conf-bar-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.38);
  }

  .conf-bar-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--gold);
    letter-spacing: 0.04em;
  }

  .conf-bar-track {
    height: 5px;
    background: rgba(245,240,232,0.08);
    border-radius: 2px;
    overflow: hidden;
  }

  .conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--gold), var(--gold-light));
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }

  /* ── Severity badge ── */
  .severity-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-top: 1.4rem;
    border-top: 1px solid rgba(201,168,76,0.12);
  }

  .severity-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .severity-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: rgba(245,240,232,0.55);
    letter-spacing: 0.02em;
  }

  .severity-badge {
    margin-left: auto;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] > div {
    border-top-color: var(--gold) !important;
  }

  /* ── Fade-in animation ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .fade-in {
    animation: fadeUp 0.55s ease forwards;
  }

  /* ── Misc Streamlit overrides ── */
  [data-testid="stMarkdownContainer"] p {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 300;
    color: var(--muted);
  }

  .stAlert {
    background: var(--charcoal-mid) !important;
    border-color: var(--border) !important;
    color: var(--cream) !important;
  }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model & Labels
# -----------------------------
@st.cache_resource
def load_assets():
    mdl = load_model("road_damage_cnn.h5")
    with open("labels.json", "r") as f:
        lbl = json.load(f)
    lbl = {int(k): v for k, v in lbl.items()}
    return mdl, lbl

model, labels = load_assets()
IMG_SIZE = 128


# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="fade-in">
  <p class="hero-eyebrow">AI Infrastructure Intelligence</p>
  <h1 class="hero-title">Road<em>Scan</em></h1>
  <p class="hero-subtitle">
    Upload a road surface photograph. Our model will identify damage
    classification and confidence with precision.
  </p>
</div>
<hr class="gold-rule">
""", unsafe_allow_html=True)


# -----------------------------
# Upload Section
# -----------------------------
st.markdown('<span class="upload-label">Surface Image</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    # ── Display image ──────────────────────────────
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # ── Preprocessing ─────────────────────────────
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # ── Inference ─────────────────────────────────
    with st.spinner("Analysing surface…"):
        prediction = model.predict(img_input)

    class_index = int(np.argmax(prediction))
    confidence  = float(np.max(prediction))
    class_name  = labels[class_index]

    # ── Result card ───────────────────────────────
    if confidence >= 0.85:
        severity_label = "High Confidence"
        dot_color = "#C9A84C"
        badge_color = "rgba(201,168,76,0.18)"
        badge_border = "rgba(201,168,76,0.45)"
        badge_text = "#E8C96A"
    elif confidence >= 0.60:
        severity_label = "Moderate Confidence"
        dot_color = "#D97B2A"
        badge_color = "rgba(217,123,42,0.15)"
        badge_border = "rgba(217,123,42,0.4)"
        badge_text = "#F0A050"
    else:
        severity_label = "Low Confidence"
        dot_color = "#C0443A"
        badge_color = "rgba(192,68,58,0.15)"
        badge_border = "rgba(192,68,58,0.4)"
        badge_text = "#E06058"

    bar_width = int(confidence * 100)

    st.markdown(f"""
    <div class="result-card fade-in">
      <p class="result-eyebrow">Detection Result</p>

      <div class="result-stats">
        <div class="stat-cell">
          <span class="stat-label">Damage Classification</span>
          <p class="stat-value">{class_name}</p>
        </div>
        <div class="stat-cell">
          <span class="stat-label">Confidence Score</span>
          <p class="stat-value-gold">{confidence * 100:.1f}%</p>
        </div>
      </div>

      <div class="conf-bar-wrap">
        <div class="conf-bar-header">
          <span class="conf-bar-title">Confidence Level</span>
          <span class="conf-bar-pct">{bar_width}%</span>
        </div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:{bar_width}%;"></div>
        </div>
      </div>

      <div class="severity-row">
        <div class="severity-dot" style="background:{dot_color};
             box-shadow:0 0 6px {dot_color};"></div>
        <span class="severity-text">Model assessment complete</span>
        <span class="severity-badge" style="background:{badge_color};
              border-color:{badge_border}; color:{badge_text};">
          {severity_label}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif; font-size:13px;
              color:rgba(245,240,232,0.3); letter-spacing:0.02em; margin-top:0.5rem;">
      Accepted formats: JPG · JPEG · PNG
    </p>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────
st.markdown("""
<hr class="gold-rule" style="margin-top:3.5rem;">
<p style="font-family:'DM Sans',sans-serif; font-size:11px;
          color:rgba(245,240,232,0.22); letter-spacing:0.12em; text-align:center; margin:0;">
  ROADSCAN &nbsp;·&nbsp; AI INFRASTRUCTURE INTELLIGENCE
</p>
""", unsafe_allow_html=True)