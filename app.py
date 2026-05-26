import streamlit as st
import numpy as np
import cv2
import json
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Road Damage Detection",
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

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--charcoal) !important;
    color: var(--cream) !important;
  }

  .main .block-container {
    background-color: var(--charcoal) !important;
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 760px;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Hero ── */
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
    font-size: 3rem;
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: var(--cream);
    margin: 0 0 0.5rem 0;
  }
  .hero-title em { font-style: italic; color: var(--gold-light); }
  .hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 15px;
    color: var(--muted);
    letter-spacing: 0.02em;
    line-height: 1.7;
    margin-bottom: 2.2rem;
  }

  /* ── Section headings ── */
  .section-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.36em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 0.5rem 0;
  }
  .section-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    font-size: 1.75rem;
    color: var(--cream);
    margin: 0 0 1.2rem 0;
    letter-spacing: -0.01em;
  }

  /* ── Gold rules ── */
  .gold-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 2.2rem 0;
    border: none;
  }

  /* ── About cards ── */
  .about-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 5px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }
  .about-card {
    background: var(--charcoal-mid);
    padding: 1.5rem 1.4rem;
  }
  .about-icon {
    font-size: 1.4rem;
    margin-bottom: 0.7rem;
    display: block;
  }
  .about-card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 0.5rem 0;
  }
  .about-card-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 300;
    color: var(--muted);
    line-height: 1.65;
    margin: 0;
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
  [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
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

  /* ── Image preview ── */
  [data-testid="stImage"] img {
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
    filter: brightness(0.96) contrast(1.04);
  }
  .preview-caption {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 300;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.28);
    text-align: center;
    margin-top: 0.5rem;
  }

  /* ── Result card ── */
  .result-card {
    background: var(--charcoal-mid);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2.4rem 2.6rem 2.2rem;
    margin-top: 1.6rem;
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
  .result-stats {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 2rem;
  }
  .stat-cell { padding: 1.3rem 1.4rem; }
  .stat-cell:not(:last-child) { border-right: 1px solid var(--border); }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.35);
    margin: 0 0 0.5rem 0;
    display: block;
  }
  .stat-value {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 400;
    font-size: 1.75rem;
    letter-spacing: 0.01em;
    color: var(--cream);
    line-height: 1.1;
    margin: 0;
  }
  .stat-value-gold {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 2rem;
    color: var(--gold-light);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
  }
  .stat-value-sev {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1.5rem;
    letter-spacing: 0.02em;
    line-height: 1;
    margin: 0;
  }
  .conf-bar-wrap { margin-bottom: 1.6rem; }
  .conf-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
  }
  .conf-bar-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.35);
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
  }
  .severity-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-top: 1.3rem;
    border-top: 1px solid rgba(201,168,76,0.12);
  }
  .severity-dot {
    width: 8px; height: 8px;
    border-radius: 50%; flex-shrink: 0;
  }
  .severity-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: rgba(245,240,232,0.5);
    letter-spacing: 0.02em;
  }
  .severity-badge {
    margin-left: auto;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    border: 1px solid;
  }

  /* ── Recommendation card ── */
  .rec-card {
    background: var(--charcoal-mid);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2rem 2.4rem;
    margin-top: 0.6rem;
    position: relative;
    overflow: hidden;
  }
  .rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .rec-row {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 1.1rem;
  }
  .rec-row:last-child { margin-bottom: 0; }
  .rec-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .rec-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 0.3rem 0;
  }
  .rec-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 300;
    color: var(--cream);
    line-height: 1.6;
    margin: 0;
  }

  /* ── Plotly chart container ── */
  [data-testid="stPlotlyChart"] {
    background: transparent !important;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] > div {
    border-top-color: var(--gold) !important;
  }

  /* ── Fade-in ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-in { animation: fadeUp 0.55s ease forwards; }

  [data-testid="stMarkdownContainer"] p {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 300;
    color: var(--muted);
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load Model & Labels
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    mdl = load_model("road_damage_cnn.h5")
    with open("labels.json", "r") as f:
        lbl = json.load(f)
    lbl = {int(k): v for k, v in lbl.items()}
    return mdl, lbl

model, labels = load_assets()
IMG_SIZE = 128


# ─────────────────────────────────────────
# SECTION 1 — Header
# ─────────────────────────────────────────
st.markdown("""
<div class="fade-in">
  <p class="hero-eyebrow">Smart City Infrastructure Monitoring</p>
  <h1 class="hero-title">AI-Based Road<br><em>Damage Detection</em> System</h1>
  <p class="hero-subtitle">Smart City Infrastructure Monitoring using CNN</p>
</div>
<hr class="gold-rule">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 2 — About the Project
# ─────────────────────────────────────────
st.markdown("""
<p class="section-eyebrow">02 &nbsp;·&nbsp; About the Project</p>
<h2 class="section-title">Why Road Monitoring Matters</h2>
<div class="about-grid fade-in">
  <div class="about-card">
    <span class="about-icon">🛣️</span>
    <p class="about-card-title">Road Safety</p>
    <p class="about-card-body">
      Deteriorating road surfaces cause thousands of accidents annually.
      Early damage detection prevents vehicle damage, reduces injury risk,
      and lowers municipal liability.
    </p>
  </div>
  <div class="about-card">
    <span class="about-icon">🧠</span>
    <p class="about-card-title">CNN in Computer Vision</p>
    <p class="about-card-body">
      Convolutional Neural Networks extract spatial features from images,
      enabling automated classification of damage types — potholes, cracks,
      and surface deformations — with high precision.
    </p>
  </div>
  <div class="about-card">
    <span class="about-icon">🏙️</span>
    <p class="about-card-title">Industry Applications</p>
    <p class="about-card-body">
      Deployed in smart city platforms, drone-based road surveys,
      municipal maintenance pipelines, and autonomous vehicle
      hazard-avoidance systems worldwide.
    </p>
  </div>
</div>
<hr class="gold-rule">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 3 — Upload Area
# ─────────────────────────────────────────
st.markdown("""
<p class="section-eyebrow">03 &nbsp;·&nbsp; Upload</p>
<h2 class="section-title">Surface Image</h2>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif; font-size:13px;
              color:rgba(245,240,232,0.28); letter-spacing:0.02em; margin-top:0.4rem;">
      Drag &amp; drop or click to browse &nbsp;·&nbsp; JPG · JPEG · PNG
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# Results (Sections 4–7) — only when image uploaded
# ─────────────────────────────────────────
if uploaded_file is not None:

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── SECTION 4 — Image Preview ──────────────────
    st.markdown("""
    <p class="section-eyebrow">04 &nbsp;·&nbsp; Preview</p>
    <h2 class="section-title">Uploaded Image</h2>
    """, unsafe_allow_html=True)

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    st.markdown('<p class="preview-caption">Road surface image — awaiting analysis</p>',
                unsafe_allow_html=True)

    # ── Preprocessing & Inference ──────────────────
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    with st.spinner("Analysing surface…"):
        prediction = model.predict(img_input)

    class_index = int(np.argmax(prediction))
    confidence  = float(np.max(prediction))
    class_name  = labels[class_index]
    bar_width   = int(confidence * 100)

    # Severity mapping
    if confidence >= 0.85:
        sev_label   = "High"
        sev_color   = "#C9A84C"
        dot_color   = "#C9A84C"
        badge_bg    = "rgba(201,168,76,0.18)"
        badge_bd    = "rgba(201,168,76,0.45)"
        badge_txt   = "#E8C96A"
        rec_accent  = "#C9A84C"
        bar_accent  = "linear-gradient(90deg,#C9A84C,#E8C96A)"
    elif confidence >= 0.60:
        sev_label   = "Medium"
        sev_color   = "#D97B2A"
        dot_color   = "#D97B2A"
        badge_bg    = "rgba(217,123,42,0.15)"
        badge_bd    = "rgba(217,123,42,0.4)"
        badge_txt   = "#F0A050"
        rec_accent  = "#D97B2A"
        bar_accent  = "linear-gradient(90deg,#D97B2A,#F0A050)"
    else:
        sev_label   = "Low"
        sev_color   = "#C0443A"
        dot_color   = "#C0443A"
        badge_bg    = "rgba(192,68,58,0.15)"
        badge_bd    = "rgba(192,68,58,0.4)"
        badge_txt   = "#E06058"
        rec_accent  = "#C0443A"
        bar_accent  = "linear-gradient(90deg,#C0443A,#E06058)"

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── SECTION 5 — Prediction ─────────────────────
    st.markdown("""
    <p class="section-eyebrow">05 &nbsp;·&nbsp; Prediction</p>
    <h2 class="section-title">Detection Result</h2>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card fade-in">
      <p class="result-eyebrow">Analysis Complete</p>

      <div class="result-stats">
        <div class="stat-cell">
          <span class="stat-label">Prediction</span>
          <p class="stat-value">{class_name}</p>
        </div>
        <div class="stat-cell">
          <span class="stat-label">Confidence</span>
          <p class="stat-value-gold">{confidence * 100:.1f}%</p>
        </div>
        <div class="stat-cell">
          <span class="stat-label">Severity</span>
          <p class="stat-value-sev" style="color:{sev_color};">{sev_label}</p>
        </div>
      </div>

      <div class="conf-bar-wrap">
        <div class="conf-bar-header">
          <span class="conf-bar-title">Confidence Level</span>
          <span class="conf-bar-pct">{bar_width}%</span>
        </div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:{bar_width}%; background:{bar_accent};"></div>
        </div>
      </div>

      <div class="severity-row">
        <div class="severity-dot" style="background:{dot_color}; box-shadow:0 0 7px {dot_color};"></div>
        <span class="severity-text">Model assessment complete &nbsp;·&nbsp; CNN classification</span>
        <span class="severity-badge" style="background:{badge_bg}; border-color:{badge_bd}; color:{badge_txt};">
          {sev_label} Severity
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── SECTION 6 — Visualization ─────────────────
    st.markdown("""
    <p class="section-eyebrow">06 &nbsp;·&nbsp; Visualization</p>
    <h2 class="section-title">Probability Distribution</h2>
    """, unsafe_allow_html=True)

    probs       = prediction[0].tolist()
    class_names = [labels[i] for i in range(len(probs))]
    colors      = ["#C9A84C" if i == class_index else "rgba(201,168,76,0.22)"
                   for i in range(len(probs))]

    # Bar chart
    fig_bar = go.Figure(go.Bar(
        x=class_names,
        y=[p * 100 for p in probs],
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(family="DM Sans", size=12, color="#F5F0E8"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(44,44,40,0.6)",
        font=dict(family="DM Sans", color="#F5F0E8"),
        xaxis=dict(
            tickfont=dict(size=12, color="rgba(245,240,232,0.55)"),
            gridcolor="rgba(201,168,76,0.08)",
            linecolor="rgba(201,168,76,0.2)",
        ),
        yaxis=dict(
            ticksuffix="%",
            tickfont=dict(size=11, color="rgba(245,240,232,0.4)"),
            gridcolor="rgba(201,168,76,0.08)",
            linecolor="rgba(201,168,76,0.2)",
            range=[0, 115],
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Horizontal confidence chart
    sorted_idx   = np.argsort(probs)
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_probs = [probs[i] * 100 for i in sorted_idx]
    bar_colors   = ["#C9A84C" if sorted_idx[i] == class_index
                    else "rgba(201,168,76,0.2)" for i in range(len(sorted_idx))]

    fig_h = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_names,
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{p:.1f}%" for p in sorted_probs],
        textposition="outside",
        textfont=dict(family="DM Sans", size=12, color="#F5F0E8"),
    ))
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(44,44,40,0.6)",
        font=dict(family="DM Sans", color="#F5F0E8"),
        xaxis=dict(
            ticksuffix="%",
            tickfont=dict(size=11, color="rgba(245,240,232,0.4)"),
            gridcolor="rgba(201,168,76,0.08)",
            range=[0, 120],
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="rgba(245,240,232,0.6)"),
            gridcolor="rgba(201,168,76,0.08)",
        ),
        margin=dict(l=10, r=60, t=10, b=10),
        height=280,
        bargap=0.4,
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # ── SECTION 7 — Recommendations ───────────────
    st.markdown("""
    <p class="section-eyebrow">07 &nbsp;·&nbsp; Recommendations</p>
    <h2 class="section-title">Action & Safety Guidance</h2>
    """, unsafe_allow_html=True)

    # Dynamic recommendations based on severity
    if sev_label == "High":
        priority_title = "Immediate Maintenance Required"
        priority_body  = (f"The detected <strong>{class_name}</strong> presents a high-risk condition. "
                          "Dispatch a repair crew within 24–48 hours. Temporary road closure or warning "
                          "signage should be deployed immediately.")
        safety_title   = "High-Risk Road Condition"
        safety_body    = ("Vehicles traversing this section face elevated risk of tyre damage, "
                          "loss of control, and collision. Reduce speed limits in the affected zone "
                          "and notify traffic management authorities.")
        ops_title      = "Operational Priority — P1"
        ops_body       = ("Flag this segment in the municipal maintenance dashboard as Priority 1. "
                          "Allocate budget for full resurfacing. Document with geo-tagged photography "
                          "for insurance and compliance records.")
    elif sev_label == "Medium":
        priority_title = "Scheduled Maintenance Advised"
        priority_body  = (f"The detected <strong>{class_name}</strong> requires attention within "
                          "7–14 days. Queue for the next routine maintenance cycle and monitor "
                          "for further deterioration.")
        safety_title   = "Moderate Road Hazard"
        safety_body    = ("Road users should exercise caution in this zone. Consider advisory "
                          "speed reductions. The condition may worsen rapidly under heavy traffic "
                          "or adverse weather.")
        ops_title      = "Operational Priority — P2"
        ops_body       = ("Log as Priority 2 in the maintenance system. Assign to the next "
                          "available crew. Re-inspect if weather conditions deteriorate before "
                          "repair is completed.")
    else:
        priority_title = "Routine Inspection Recommended"
        priority_body  = (f"The detected <strong>{class_name}</strong> shows early-stage damage. "
                          "Schedule a ground inspection within 30 days to confirm classification "
                          "and track progression.")
        safety_title   = "Low Immediate Risk"
        safety_body    = ("No urgent safety intervention required at this time. Continue standard "
                          "monitoring cadence and reassess after the next precipitation event.")
        ops_title      = "Operational Priority — P3"
        ops_body       = ("Log as Priority 3. Include in the next quarterly maintenance survey. "
                          "Consider preventive sealing to avoid escalation.")

    st.markdown(f"""
    <div class="rec-card fade-in" style="border-color:{badge_bd};">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{bar_accent};"></div>

      <div class="rec-row">
        <span class="rec-icon">🔧</span>
        <div>
          <p class="rec-title" style="color:{rec_accent};">{priority_title}</p>
          <p class="rec-body">{priority_body}</p>
        </div>
      </div>

      <div style="height:1px;background:rgba(201,168,76,0.1);margin:1rem 0;"></div>

      <div class="rec-row">
        <span class="rec-icon">⚠️</span>
        <div>
          <p class="rec-title" style="color:{rec_accent};">{safety_title}</p>
          <p class="rec-body">{safety_body}</p>
        </div>
      </div>

      <div style="height:1px;background:rgba(201,168,76,0.1);margin:1rem 0;"></div>

      <div class="rec-row">
        <span class="rec-icon">📋</span>
        <div>
          <p class="rec-title" style="color:{rec_accent};">{ops_title}</p>
          <p class="rec-body">{ops_body}</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────
st.markdown("""
<hr class="gold-rule" style="margin-top:3.5rem;">
<p style="font-family:'DM Sans',sans-serif; font-size:11px;
          color:rgba(245,240,232,0.2); letter-spacing:0.14em; text-align:center; margin:0;">
  AI-BASED ROAD DAMAGE DETECTION SYSTEM &nbsp;·&nbsp; SMART CITY INFRASTRUCTURE MONITORING
</p>
""", unsafe_allow_html=True)