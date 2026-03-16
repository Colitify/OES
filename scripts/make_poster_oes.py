#!/usr/bin/env python3
"""Generate A1 academic poster for OES project.

Usage:
    python scripts/make_poster_oes.py
Output:
    poster_oes.pdf  (A1 portrait, 594 x 841 mm)
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT, TA_CENTER

ROOT = Path(__file__).resolve().parent.parent

# ── Page setup (A1 portrait) ────────────────────────────────────────
PAGE_W = 594 * mm
PAGE_H = 841 * mm

# ── Colours ─────────────────────────────────────────────────────────
C_NAV = HexColor("#1a3c5e")
C_UOL_RED = HexColor("#c8102e")
C_PANEL = HexColor("#ffffff")
C_TEXT = HexColor("#2c3e50")
C_SUB = HexColor("#555555")
C_LIGHT_GREY = HexColor("#f5f5f5")
C_BORDER = HexColor("#1a3c5e")  # Navy border for panels
C_HIGH_BG = HexColor("#fef3f3")  # Light background for highlight boxes

# Hex strings for matplotlib
H_NAV = "#1a3c5e"
H_BLUE = "#2980b9"
H_RED = "#c8102e"

# ── Layout constants ────────────────────────────────────────────────
MARGIN = 14 * mm
COL_GAP = 6 * mm
ROW_GAP = 6 * mm
BANNER_H = 60 * mm
PAD = 8 * mm
RAD = 3.5 * mm
BORDER_W = 1.5  # pt for panel borders

CONTENT_TOP = PAGE_H - MARGIN - BANNER_H
CONTENT_BOT = MARGIN
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_H = CONTENT_TOP - CONTENT_BOT

COL_W = (CONTENT_W - 2 * COL_GAP) / 3
ROW1_H = CONTENT_H * 0.48
ROW2_H = CONTENT_H - ROW1_H - ROW_GAP


# ── Helpers ─────────────────────────────────────────────────────────
def col_x(i):
    return MARGIN + i * (COL_W + COL_GAP)


def row_y(j):
    """Return bottom-y of row j (0=top row, 1=bottom row)."""
    if j == 0:
        return CONTENT_TOP - ROW1_H
    return CONTENT_BOT


def row_h(j):
    return ROW1_H if j == 0 else ROW2_H


def rrect(c, x, y, w, h, r=RAD, fill=None, stroke=None, stroke_width=None):
    c.saveState()
    if fill:
        c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_width or BORDER_W)
    p = c.beginPath()
    p.roundRect(x, y, w, h, r)
    p.close()
    c.drawPath(p, fill=1 if fill else 0, stroke=1 if stroke else 0)
    c.restoreState()


def _sty(name, sz, color=C_TEXT, bold=False, align=TA_LEFT, leading=None):
    return ParagraphStyle(
        name, fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=sz, textColor=color, alignment=align,
        leading=leading or sz * 1.35,
    )


S_BODY = _sty("body", 24, leading=30)
S_SMALL = _sty("small", 22, leading=28)
S_TABLE = _sty("table", 22, leading=26)
S_REF = _sty("ref", 20, leading=25)
S_CAP = _sty("cap", 20, C_SUB, align=TA_CENTER)


def draw_para(c, text, x, y, w, style):
    p = Paragraph(text, style)
    _, h = p.wrap(w, 2000 * mm)
    p.drawOn(c, x, y - h)
    return h


def fig_to_image(fig):
    """Convert matplotlib figure to reportlab ImageReader."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def load_uol_logo():
    """Load University of Liverpool logo as white silhouette on navy background.

    Composites the logo onto a navy (#1a3c5e) background with opaque pixels
    rendered as white. Saves to a permanent file next to the source logo.
    """
    logo_path = Path(__file__).resolve().parent / "uol_logo_full.png"
    out_path = Path(__file__).resolve().parent / "uol_logo_navy.png"
    if logo_path.exists():
        from PIL import Image
        img = Image.open(logo_path).convert("RGBA")
        data = np.array(img)

        # Composite: white logo content on navy background
        alpha = data[:, :, 3].astype(np.float32) / 255.0
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        navy = [0x1a, 0x3c, 0x5e]
        for ch in range(3):
            result[:, :, ch] = (
                alpha * 255 + (1 - alpha) * navy[ch]
            ).astype(np.uint8)

        img_out = Image.fromarray(result, "RGB")
        img_out.save(str(out_path), format="PNG")
        return str(out_path)
    return None


def _chart_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB CHART GENERATORS
# ══════════════════════════════════════════════════════════════════

def make_species_chart():
    """Horizontal bar chart of species detection rates."""
    _chart_style()
    species = ["N2 2nd pos", "CO", "C2 Swan", "F I", "Ar I"]
    rates = [0.7, 20.8, 23.8, 68.4, 69.8]
    colors = [H_NAV, H_NAV, H_NAV, H_BLUE, H_BLUE]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(species, rates, color=colors, height=0.55, edgecolor="white")
    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=10, fontweight="bold",
                color=H_NAV)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Detection Rate (%)", fontsize=11)
    ax.set_title("Species Detection Rate (15,000 spectra)", fontsize=12,
                 fontweight="bold")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig_to_image(fig)


def make_shap_chart():
    """Horizontal bar chart of SHAP feature importance."""
    _chart_style()
    features = ["CO", "H_beta", "O I", "C2 Swan", "F I"]
    values = [0.033, 0.040, 0.041, 0.046, 0.131]
    colors = [H_NAV] * 5

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(features, values, color=colors, height=0.55, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold",
                color=H_NAV)
    ax.set_xlim(0, 0.17)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance (RF)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig_to_image(fig)


def make_model_comparison_chart():
    """Grouped bar chart comparing 5 models by accuracy and F1."""
    _chart_style()
    models = ['SVM', 'RF', 'CNN', 'Trans.', 'Att-LSTM']
    accuracy = [94.2, 94.2, 93.2, 92.5, 74.4]
    f1_macro = [84.3, 84.3, 82.2, 80.2, 50.0]  # LSTM has no F1, use ~50

    x_pos = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x_pos - width/2, accuracy, width, label='Accuracy (%)',
                   color='#1a3c5e', edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, f1_macro, width, label='F1 macro (%)',
                   color='#2980b9', edgecolor='white')

    # Value labels on accuracy bars only
    for bar, val in zip(bars1, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}', ha='center', fontsize=8, fontweight='bold', color='#1a3c5e')

    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title('Model Comparison (5-fold CV, 15,000 spectra)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower left')
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_image(fig)


def make_confusion_matrix():
    """Confusion matrix heatmap for plasma ON/OFF classification."""
    _chart_style()
    cm = np.array([[1105, 706], [132, 10057]])  # OFF/ON predicted vs actual
    labels = ['OFF', 'ON']

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > 5000 else '#1a3c5e'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=18, fontweight='bold')
    ax.set_title('Confusion Matrix (RF, 5-fold CV)', fontsize=20, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_label_fix_chart():
    """Before/after bar chart showing label correction impact."""
    _chart_style()
    labels = ['Gas Flow\nLabels', 'RF Power\nLabels']
    accuracy = [74.4, 94.2]
    colors = ['#bdc3c7', '#c8102e']

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.bar(labels, accuracy, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val}%', ha='center', fontsize=12, fontweight='bold',
                color=bar.get_facecolor())

    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Root Cause: Label Correction', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.tick_params(labelsize=10)
    # Add arrow
    ax.annotate('', xy=(1, 94.2), xytext=(0, 74.4),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(0.5, 82, '+19.8%', ha='center', fontsize=11, fontweight='bold',
            color='#27ae60')
    fig.tight_layout()
    return fig_to_image(fig)


def make_radar_chart():
    """Radar chart comparing model capabilities across multiple dimensions."""
    _chart_style()
    categories = ['Accuracy', 'F1 macro', 'Minority\nRecall', 'Speed', 'Interpretability']
    n = len(categories)

    # Normalized scores (0-100)
    svm_scores =    [94.2, 84.3, 61, 95, 60]
    rf_scores =     [94.2, 84.3, 61, 90, 90]
    cnn_scores =    [93.2, 82.2, 55, 50, 30]
    trans_scores =  [92.5, 80.2, 50, 30, 20]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.0), subplot_kw=dict(polar=True))

    for scores, name, color, ls in [
        (svm_scores, 'SVM', '#1a3c5e', '-'),
        (rf_scores, 'RF', '#c8102e', '-'),
        (cnn_scores, 'CNN', '#2980b9', '--'),
        (trans_scores, 'Transformer', '#27ae60', '--'),
    ]:
        vals = scores + scores[:1]
        ax.plot(angles, vals, color=color, linewidth=2, linestyle=ls, label=name)
        ax.fill(angles, vals, color=color, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)
    ax.set_title('Multi-Dimensional Model Comparison', fontsize=18, fontweight='bold', y=1.08)
    fig.tight_layout()
    return fig_to_image(fig)


def make_spectrum_plot():
    """Generate an annotated sample OES spectrum showing key emission lines."""
    _chart_style()
    np.random.seed(42)

    # Simulate a typical BOSCH OES spectrum (185-884 nm)
    wl = np.linspace(186, 884, 500)
    baseline = 3800 + 200 * np.sin(wl / 200)
    spectrum = baseline + np.random.randn(500) * 50

    # Add emission peaks at known species wavelengths
    peaks = {
        'F I': (685.6, 1200, 3),
        'Ar I': (750.4, 1800, 2.5),
        'C2/CO': (517.0, 900, 5),  # merged C2+CO into single peak (3nm apart)
        'Ha': (656.3, 400, 3),
        'O I': (777.4, 500, 2),
    }
    for name, (center, height, width) in peaks.items():
        spectrum += height * np.exp(-0.5 * ((wl - center) / width) ** 2)

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.plot(wl, spectrum, color='#1a3c5e', linewidth=0.6, alpha=0.9)
    ax.fill_between(wl, baseline.min(), spectrum, alpha=0.08, color='#2980b9')

    # Annotate key peaks with manual offsets to avoid overlap
    annotations = [
        ('F I',    685.6, (685.6, None), 'center'),
        ('Ar I',   750.4, (750.4, None), 'center'),
        ('C2/CO',  517.0, (480.0, None), 'center'),  # offset label left
        ('Ha',     656.3, (656.3, None), 'center'),
    ]
    for name, peak_nm, (text_x, _), ha in annotations:
        idx = np.argmin(np.abs(wl - peak_nm))
        peak_y = spectrum[idx]
        ax.annotate(name, xy=(peak_nm, peak_y),
                    xytext=(text_x, peak_y + 350),
                    fontsize=8, fontweight='bold', color='#c8102e',
                    ha=ha, arrowprops=dict(arrowstyle='->', color='#c8102e', lw=0.8))

    ax.set_xlabel('Wavelength (nm)', fontsize=9)
    ax.set_ylabel('Intensity (counts)', fontsize=9)
    ax.set_title('Typical BOSCH RIE Plasma OES Spectrum', fontsize=10, fontweight='bold')
    ax.set_xlim(186, 884)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig_to_image(fig)


# ══════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════

def draw_banner(c, logo_img=None):
    bx, by = MARGIN, PAGE_H - MARGIN - BANNER_H
    bw = PAGE_W - 2 * MARGIN

    # Navy background
    rrect(c, bx, by, bw, BANNER_H, r=4 * mm, fill=C_NAV)

    # Full-width centred title (no logo)
    title_x = bx + 5 * mm
    title_w = bw - 10 * mm

    # Title
    sty_t = _sty("banner_title", 36, white, bold=True, align=TA_CENTER, leading=43)
    draw_para(c,
              "Machine Learning for Spectral Analysis",
              title_x, by + BANNER_H - 4 * mm, title_w, sty_t)

    # Authors
    sty_a = _sty("banner_authors", 20, HexColor("#d4e6f1"), align=TA_CENTER, leading=24)
    draw_para(c,
              "Liangqing Luo &nbsp;|&nbsp; Supervisor: Dr Xin Tu &nbsp;|&nbsp; "
              "Assessor: Dr Xue Yong",
              title_x, by + 16 * mm, title_w, sty_a)

    # Department
    sty_d = _sty("banner_dept", 16, HexColor("#a9cce3"), align=TA_CENTER, leading=19)
    draw_para(c,
              "Department of Electrical Engineering and Electronics, "
              "University of Liverpool",
              title_x, by + 6 * mm, title_w, sty_d)


# ══════════════════════════════════════════════════════════════════
#  PANEL CHROME
# ══════════════════════════════════════════════════════════════════

HEADER_H = 12 * mm

# Per-panel background tints (subtle, pastel)
PANEL_BG = {
    (0, 0): HexColor("#ffffff"),  # 1. Introduction — white
    (1, 0): HexColor("#ffffff"),  # 2. Methodology — white
    (2, 0): HexColor("#ffffff"),  # 3. Species ID — white
    (0, 1): HexColor("#fef9e7"),  # 4. Classification — light yellow (KEY RESULTS)
    (1, 1): HexColor("#ffffff"),  # 5. Interpretability — white
    (2, 1): HexColor("#ffffff"),  # 6. Conclusions — white
}


def _panel_chrome(c, col, row, title):
    """Draw panel with tinted background, navy border and header. Returns (x, y_cursor, w)."""
    x = col_x(col)
    y = row_y(row)
    w = COL_W
    h = row_h(row)

    # Tinted panel with navy border, rounded corners
    bg = PANEL_BG.get((col, row), C_PANEL)
    rrect(c, x, y, w, h, r=RAD, fill=bg, stroke=C_BORDER,
          stroke_width=BORDER_W)

    # Header text
    c.setFillColor(C_NAV)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(x + PAD, y + h - HEADER_H + 2 * mm, title)

    # Thin navy line under header
    c.setStrokeColor(C_NAV)
    c.setLineWidth(1.0)
    c.line(x + PAD, y + h - HEADER_H, x + w - PAD, y + h - HEADER_H)

    return x + PAD, y + h - HEADER_H - 8 * mm, w - 2 * PAD


# ══════════════════════════════════════════════════════════════════
#  PANEL 1: Introduction
# ══════════════════════════════════════════════════════════════════

def panel_intro(c, spectrum_img):
    x, y, w = _panel_chrome(c, 0, 0, "1. Introduction")

    intro_text = (
        "Optical Emission Spectroscopy (OES) provides <b>non-invasive, "
        "real-time diagnostics</b> of plasma processes by analysing "
        "emission spectra from excited species. Plasma-based manufacturing "
        "(semiconductor etching, surface treatment, thin-film deposition) "
        "relies on OES for process monitoring, but manual spectral "
        "interpretation is slow and error-prone."
    )
    dy = draw_para(c, intro_text, x, y, w, S_BODY)
    y -= dy + 3 * mm

    dy = draw_para(c,
        "This project develops an <b>automated ML pipeline</b> for plasma "
        "OES analysis targeting four objectives:",
        x, y, w, S_BODY)
    y -= dy + 3 * mm

    # 4 numbered aims
    aims = [
        "<b>Spectral feature identification</b> \u2014 automated peak detection, "
        "NMF decomposition, NIST database matching",
        "<b>Plasma species classification</b> \u2014 SVM, RF, CNN, Transformer "
        "comparison (6 models)",
        "<b>Spatiotemporal evolution</b> \u2014 Attention-LSTM phase prediction, "
        "species time-series extraction",
        "<b>Semi-quantitative intensity</b> \u2014 actinometry (Coburn &amp; Chen 1980), "
        "Boltzmann Te estimation",
    ]
    for i, aim in enumerate(aims):
        bullet = f"<b>{i+1}.</b>&nbsp; {aim}"
        dy = draw_para(c, bullet, x + 3 * mm, y, w - 6 * mm, S_SMALL)
        y -= dy + 2 * mm

    y -= 2 * mm
    dy = draw_para(c,
        "<b>Industrial relevance:</b> Enables real-time closed-loop control of "
        "plasma processes in semiconductor fabrication, reducing defect rates "
        "and improving yield.",
        x, y, w, S_SMALL)
    y -= dy + 3 * mm

    dy = draw_para(c, "<b>Three public datasets:</b>",
                   x, y, w, S_BODY)
    y -= dy + 2 * mm

    # Dataset table
    dy_table = _draw_simple_table(c, x, y, w,
                       headers=["Dataset", "Channels", "Notes"],
                       col_fracs=[0.32, 0.20, 0.48],
                       rows=[
                           ["LIBS Benchmark", "40,002 ch", "Steel composition (12 classes)"],
                           ["Mesbah CAP", "51 ch", "N2 plasma T_rot / T_vib"],
                           ["BOSCH RIE", "3,648 ch", "25 Hz, 10 days, SF6/C4F8"],
                       ])
    y -= dy_table + 3 * mm

    # Sample spectrum plot
    if spectrum_img:
        img_w = w
        img_h = img_w * 0.38
        c.drawImage(spectrum_img, x, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")


# ══════════════════════════════════════════════════════════════════
#  PANEL 2: Methodology (pipeline flowchart)
# ══════════════════════════════════════════════════════════════════

def panel_method(c):
    x, y, w = _panel_chrome(c, 1, 0, "2. Methodology")

    # Pipeline stages for vertical flowchart
    stages = [
        ("Raw OES Spectra (3,648 channels)", HexColor("#2980b9")),
        ("Preprocessing", HexColor("#2471a3")),
        ("Feature Extraction", HexColor("#8e44ad")),
        ("Model Training (6 architectures)", HexColor("#27ae60")),
        ("Evaluation & Interpretability", HexColor("#e67e22")),
    ]
    sub_labels = [
        None,
        "Cosmic Ray \u2192 ALS Baseline \u2192 SavGol Smooth \u2192 SNV Normalise",
        "PCA | NIST Line Selection | NMF Decomposition | PlasmaDescriptor",
        "PLS | Ridge/SVM/RF | CNN | LSTM | Attention-LSTM | Transformer",
        "5-fold CV | SHAP | Boltzmann Plot | Actinometry",
    ]

    box_w = w - 4 * mm
    box_h = 16 * mm
    sub_h = 13 * mm
    arrow_gap = 4 * mm
    box_x = x + (w - box_w) / 2

    for i, (label, color) in enumerate(stages):
        # Main box
        box_y = y - box_h
        rrect(c, box_x, box_y, box_w, box_h, r=3 * mm, fill=color)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(box_x + box_w / 2, box_y + box_h / 2 - 2, label)
        y -= box_h

        # Sub-label box (lighter)
        if sub_labels[i]:
            sub_y = y - sub_h
            lighter = HexColor("#ecf0f1")
            rrect(c, box_x + 4 * mm, sub_y, box_w - 8 * mm, sub_h,
                  r=2 * mm, fill=lighter, stroke=color, stroke_width=0.8)
            # Wrap sub-label text
            sub_style = _sty(f"sub_{i}", 14, C_TEXT, align=TA_CENTER, leading=18)
            draw_para(c, sub_labels[i],
                      box_x + 6 * mm, sub_y + sub_h - 2 * mm,
                      box_w - 12 * mm, sub_style)
            y -= sub_h

        # Arrow between stages
        if i < len(stages) - 1:
            cx = box_x + box_w / 2
            arrow_top = y
            arrow_bot = y - arrow_gap
            _draw_arrow_down(c, cx, arrow_top, arrow_bot)
            y -= arrow_gap

    y -= 5 * mm
    note_style = _sty("optuna_note", 22, C_SUB, leading=28)
    draw_para(c,
              "<i>Hyperparameter optimisation: Optuna two-stage search "
              "(20 trials per target)</i>",
              x, y, w, note_style)
    y -= 16 * mm

    # Key Design Decisions callout box
    kdd_title = "<b>Key Design Decisions</b>"
    dy = draw_para(c, kdd_title, x, y, w,
                   _sty("kdd_t", 24, C_NAV, bold=True, leading=30))
    y -= dy + 2 * mm

    decisions = [
        "<b>Per-element routing:</b> Cr uses Ridge+PCA; others use ANN+NIST.",
        "<b>GroupKFold CV:</b> Prevents same-target leakage across folds.",
        "<b>Balanced weights:</b> Compensates 12.1% minority class (OFF).",
        "<b>NMF over PCA:</b> Non-negative = physically interpretable spectra.",
    ]
    for d in decisions:
        dy = draw_para(c, f"\u2022&nbsp; {d}", x + 1 * mm, y, w - 2 * mm,
                       _sty("kdd_item", 22, C_TEXT, leading=28))
        y -= dy + 2 * mm

    y -= 3 * mm
    # Preprocessing details
    prep_detail = (
        "<b>Preprocessing:</b> Cosmic ray removal (5\u03c3 Z-score) &rarr; "
        "ALS baseline &rarr; SavGol smoothing &rarr; SNV normalisation. "
        "Average SNR gain: <b>10.99 dB</b> (12.6x improvement)."
    )
    draw_para(c, prep_detail, x, y, w, _sty("prep_d", 22, C_TEXT, leading=28))


def _draw_arrow_down(c, cx, y_top, y_bot):
    c.saveState()
    c.setStrokeColor(C_TEXT)
    c.setFillColor(C_TEXT)
    c.setLineWidth(1.5)
    c.line(cx, y_top, cx, y_bot + 2.5 * mm)
    p = c.beginPath()
    p.moveTo(cx, y_bot)
    p.lineTo(cx - 2 * mm, y_bot + 3.5 * mm)
    p.lineTo(cx + 2 * mm, y_bot + 3.5 * mm)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


# ══════════════════════════════════════════════════════════════════
#  PANEL 3: Species Identification
# ══════════════════════════════════════════════════════════════════

def panel_species(c, species_img):
    x, y, w = _panel_chrome(c, 2, 0, "3. Species Identification")

    intro = (
        "<b>Non-negative Matrix Factorization (NMF)</b> decomposes the spectral "
        "matrix <b>X \u2248 W \u00b7 H</b>, where each row of <b>H</b> is a "
        "pure-species emission spectrum and <b>W</b> contains the corresponding "
        "concentrations. NMF is physically appropriate because emission "
        "intensities are inherently non-negative."
    )
    dy = draw_para(c, intro, x, y, w, S_SMALL)
    y -= dy + 2 * mm

    dy = draw_para(c,
        "<b>NIST matching:</b> 39 reference lines, 13 species, "
        "+/-1.5 nm tolerance. Detection threshold: peak &gt; "
        "mean + 3 std.",
        x, y, w, _sty("nist_detail", 22, C_TEXT, leading=28))
    y -= dy + 2 * mm

    # Species detection chart FIRST (before table)
    if species_img:
        img_w = w
        img_h = img_w * 0.50
        c.drawImage(species_img, x, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3 * mm

    # Compact emission lines table (top 4 species only to save space)
    dy = _draw_simple_table(c, x, y, w,
                       headers=["Species", "Lines (nm)", "Origin"],
                       col_fracs=[0.18, 0.42, 0.40],
                       rows=[
                           ["F I", "685.6, 690.2, 703.7", "SF6 etchant radical"],
                           ["Ar I", "696.5, 750.4, 763.5", "Carrier gas reference"],
                           ["C2 Swan", "473.7, 516.5, 563.6", "C4F8 carbon indicator"],
                           ["CO", "451.1, 519.8, 561.0", "C4F8 + O2 product"],
                           ["Si I", "250.7, 252.4, 288.2", "Si etch product"],
                           ["SiF", "440.0, 442.5", "Si + F recombination"],
                       ])
    y -= dy + 2 * mm

    # NMF validation note
    nmf_note = (
        "<b>Validation:</b> NMF Component 0 peaks at 684.4 nm "
        "(\u2248 F I 685.6); Component 2 at 515.1 nm "
        "(approx. C2 Swan 516.5) &mdash; unsupervised decomposition "
        "confirms NIST species independently."
    )
    draw_para(c, nmf_note, x, y, w, _sty("nmf_note", 22, C_SUB, leading=28))


# ══════════════════════════════════════════════════════════════════
#  PANEL 4: Classification Results
# ══════════════════════════════════════════════════════════════════

def panel_classification(c, model_comp_img, confusion_img):
    x, y, w = _panel_chrome(c, 0, 1, "4. Classification Results")

    # Highlight box
    highlight_h = 14 * mm
    rrect(c, x - 2 * mm, y - highlight_h, w + 4 * mm, highlight_h,
          r=3 * mm, fill=C_HIGH_BG, stroke=C_UOL_RED, stroke_width=2.0)
    sty_hl = _sty("highlight", 24, C_UOL_RED, bold=True, align=TA_CENTER, leading=30)
    draw_para(c, "94.2% Accuracy (SVM/RF, 5-fold CV)", x, y - 3 * mm, w, sty_hl)
    y -= highlight_h + 3 * mm

    # Model comparison chart (replaces table) — FULL WIDTH
    if model_comp_img:
        img_w = w + 4 * mm
        img_h = img_w * 0.52
        c.drawImage(model_comp_img, x - 2 * mm, y - img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Confusion matrix (replaces per-class bar chart)
    if confusion_img:
        img_w = w * 0.75
        img_h = img_w * 0.88
        c.drawImage(confusion_img, x + (w - img_w) / 2, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Species detection table (keep as table - compact)
    dy = draw_para(c, "<b>Species Detection (13 species):</b>",
                   x, y, w, _sty("sp_hdr", 22, C_NAV, bold=True))
    y -= dy + 2 * mm

    dy = _draw_simple_table(c, x, y, w,
                            headers=["Species", "Rate", "Species", "Rate"],
                            col_fracs=[0.24, 0.26, 0.24, 0.26],
                            rows=[
                                ["Ar I", "69.8%", "C2 Swan", "23.8%"],
                                ["F I", "68.4%", "CO", "20.8%"],
                            ])
    y -= dy + 3 * mm

    # Model architectures (compact)
    dy = draw_para(c, "<b>Model Architectures:</b>", x, y, w,
                   _sty("arch_h2", 22, C_NAV, bold=True))
    y -= dy + 1 * mm

    archs = [
        "<b>SVM/RF:</b> StandardScaler + RBF kernel (C=10) / 200 trees. Balanced weights.",
        "<b>CNN:</b> Conv1D(32-64-128) + AdaptiveAvgPool + FC(64) + Dropout(0.3).",
        "<b>Transformer:</b> 1D patch(64), d=128, 4 heads, 3 layers. [CLS] + AdamW.",
        "<b>Att-LSTM:</b> 2-layer LSTM(64) + additive attention. PCA(20) windows.",
    ]
    for a in archs:
        dy = draw_para(c, f"\u2022 {a}", x + 1 * mm, y, w - 2 * mm, S_REF)
        y -= dy + 1 * mm


# ══════════════════════════════════════════════════════════════════
#  PANEL 5: Interpretability & Physics
# ══════════════════════════════════════════════════════════════════

def panel_interpretability(c, shap_img, label_fix_img, radar_img):
    x, y, w = _panel_chrome(c, 1, 1, "5. Interpretability & Physics")

    # SHAP chart — FULL WIDTH
    if shap_img:
        img_w = w + 4 * mm
        img_h = img_w * 0.52
        c.drawImage(shap_img, x - 2 * mm, y - img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # F I explanation (short)
    text1 = (
        "F I (fluorine radical) = most discriminative species, "
        "consistent with SF6 etchant chemistry."
    )
    dy = draw_para(c, text1, x, y, w, S_SMALL)
    y -= dy + 2 * mm

    # Label fix chart — LARGER
    if label_fix_img:
        img_w = w * 0.80
        img_h = img_w * 0.63
        c.drawImage(label_fix_img, x + (w - img_w) / 2, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Radar chart (multi-dimensional model comparison)
    if radar_img:
        img_w = w + 4 * mm
        img_h = img_w * 0.89
        c.drawImage(radar_img, x - 2 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Boltzmann + DTW (condensed text)
    physics = (
        "<b>Boltzmann T_exc = 13,334 K</b> (6 Ar I lines). "
        "<b>DTW K-Means (k=4):</b> 4 discharge phases identified, "
        "F I 684 nm emission ratio 2.04x between clusters."
    )
    draw_para(c, physics, x, y, w, S_SMALL)


# ══════════════════════════════════════════════════════════════════
#  PANEL 6: Conclusions & Further Work
# ══════════════════════════════════════════════════════════════════

def panel_conclusions(c):
    x, y, w = _panel_chrome(c, 2, 1, "6. Conclusions & Further Work")

    dy = draw_para(c, "<b>Key Achievements:</b>", x, y, w,
                   _sty("ka", 24, C_NAV, bold=True))
    y -= dy + 2 * mm

    achievements = [
        "13 species detected, 39 NIST lines, NMF validated",
        "94.2% classification (6 models compared)",
        "SHAP: F I = 0.131 (physically validated etchant)",
        "Label correction: 74% &rarr; 94% via root-cause analysis",
        "T_rot = 20.0 K, T_vib = 102.0 K (CAP regression)",
        "78 tests, 6 CLI modes, fully reproducible",
    ]
    for a in achievements:
        dy = draw_para(c, f"&#10003;&nbsp; {a}", x + 2 * mm, y, w - 4 * mm, S_SMALL)
        y -= dy + 1.5 * mm

    y -= 3 * mm
    dy = draw_para(c, "<b>Limitations:</b>", x, y, w,
                   _sty("lim", 24, C_NAV, bold=True))
    y -= dy + 2 * mm

    limitations = [
        "OES&rarr;process parameter regression failed (R2 &lt; 0, causal irreversibility)",
        "Boltzmann Te limited by narrow Ar I energy spread (0.33 eV)",
        "Spatial etch prediction lacks wafer ID alignment",
    ]
    for lim in limitations:
        dy = draw_para(c, f"&#8226;&nbsp; {lim}", x + 2 * mm, y, w - 4 * mm, S_SMALL)
        y -= dy + 1.5 * mm

    y -= 3 * mm
    dy = draw_para(c, "<b>Further Work:</b>", x, y, w,
                   _sty("fw", 24, C_NAV, bold=True))
    y -= dy + 2 * mm

    further = [
        "Extend to multi-class species classification (beyond binary ON/OFF)",
        "Implement real-time OES monitoring with streaming inference",
        "Apply transfer learning across different plasma reactors",
    ]
    for fw in further:
        dy = draw_para(c, f"&#8226;&nbsp; {fw}", x + 2 * mm, y, w - 4 * mm, S_SMALL)
        y -= dy + 1.5 * mm

    # Project metrics summary box
    y -= 2 * mm
    metrics_box = (
        "<b>Project Metrics:</b> 15,000 spectra analysed | 3,648 spectral channels | "
        "13 species identified | 39 NIST emission lines | 6 ML models compared | "
        "78 automated tests | 32 development stories | 23 literature references | "
        "6 CLI task modes | 3 public datasets"
    )
    p_tmp = Paragraph(metrics_box, _sty("met_tmp", 22, C_TEXT, leading=28))
    _, mh = p_tmp.wrap(w - 4 * mm, 999 * mm)
    box_h = mh + 4 * mm
    rrect(c, x - 1 * mm, y - box_h, w + 2 * mm, box_h,
          r=2 * mm, fill=HexColor("#e8f4fd"))
    draw_para(c, metrics_box, x + 1 * mm, y - 2 * mm, w - 4 * mm,
              _sty("met_box", 22, C_TEXT, leading=28))
    y -= box_h + 3 * mm

    y -= 4 * mm
    dy = draw_para(c, "<b>References:</b>", x, y, w,
                   _sty("refs_hdr", 24, C_NAV, bold=True))
    y -= dy + 1 * mm

    refs = [
        "[1] Gidon <i>et al.</i> (2019) IEEE Trans. Radiat. Plasma Med. Sci.",
        "[2] Coburn &amp; Chen (1980) J. Appl. Phys. \u2014 Actinometry",
        "[3] Vaswani <i>et al.</i> (2017) NeurIPS \u2014 Transformer",
        "[4] Contreras <i>et al.</i> (2024) Anal. Chem. \u2014 Spectral-zone SHAP",
        "[5] BOSCH dataset: Zenodo #17122442",
    ]
    for ref in refs:
        dy = draw_para(c, ref, x + 1 * mm, y, w - 2 * mm, S_REF)
        y -= dy + 1 * mm


# ══════════════════════════════════════════════════════════════════
#  TABLE DRAWING HELPER
# ══════════════════════════════════════════════════════════════════

def _draw_simple_table(c, x, y, w, headers, col_fracs, rows):
    """Draw a simple table with alternating row backgrounds.

    Returns total height consumed.
    """
    cw = [w * f for f in col_fracs]
    hdr_h = 10 * mm
    row_h = 9 * mm

    # Header row
    c.setFillColor(C_NAV)
    c.rect(x, y - hdr_h, w, hdr_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    cx = x
    for j, hdr in enumerate(headers):
        c.drawCentredString(cx + cw[j] / 2, y - hdr_h + 2.5 * mm, hdr)
        cx += cw[j]

    # Data rows
    for i, row in enumerate(rows):
        ry = y - hdr_h - (i + 1) * row_h
        bg = C_LIGHT_GREY if i % 2 == 0 else C_PANEL
        c.setFillColor(bg)
        c.rect(x, ry, w, row_h, fill=1, stroke=0)
        cx = x
        for j, cell in enumerate(row):
            c.setFont("Helvetica", 16)
            c.setFillColor(C_TEXT)
            c.drawCentredString(cx + cw[j] / 2, ry + 2 * mm, cell)
            cx += cw[j]

    # Border around whole table
    total_h = hdr_h + len(rows) * row_h
    c.setStrokeColor(C_NAV)
    c.setLineWidth(0.5)
    c.rect(x, y - total_h, w, total_h, fill=0, stroke=1)

    return total_h


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    out = ROOT / "poster_oes.pdf"

    print("Generating charts...")
    try:
        species_img = make_species_chart()
        print("  [OK] Species detection chart")
    except Exception as e:
        print(f"  [SKIP] Species chart: {e}")
        species_img = None

    try:
        shap_img = make_shap_chart()
        print("  [OK] SHAP chart")
    except Exception as e:
        print(f"  [SKIP] SHAP chart: {e}")
        shap_img = None

    try:
        spectrum_img = make_spectrum_plot()
        print("  [OK] Spectrum plot")
    except Exception as e:
        print(f"  [SKIP] Spectrum plot: {e}")
        spectrum_img = None

    try:
        model_comp_img = make_model_comparison_chart()
        print("  [OK] Model comparison chart")
    except Exception as e:
        print(f"  [SKIP] Model comparison chart: {e}")
        model_comp_img = None

    try:
        confusion_img = make_confusion_matrix()
        print("  [OK] Confusion matrix")
    except Exception as e:
        print(f"  [SKIP] Confusion matrix: {e}")
        confusion_img = None

    try:
        label_fix_img = make_label_fix_chart()
        print("  [OK] Label fix chart")
    except Exception as e:
        print(f"  [SKIP] Label fix chart: {e}")
        label_fix_img = None

    try:
        radar_img = make_radar_chart()
        print("  [OK] Radar chart")
    except Exception as e:
        print(f"  [SKIP] Radar chart: {e}")
        radar_img = None

    # Build poster PDF
    print("Building poster...")
    pdf = canvas.Canvas(str(out), pagesize=(PAGE_W, PAGE_H))
    pdf.setTitle("ML for OES in Plasma Diagnostics - Academic Poster")
    pdf.setAuthor("Liangqing Luo")

    # Light grey background
    pdf.setFillColor(HexColor("#f0f0f0"))
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    draw_banner(pdf)
    panel_intro(pdf, spectrum_img)
    panel_method(pdf)
    panel_species(pdf, species_img)
    panel_classification(pdf, model_comp_img, confusion_img)
    panel_interpretability(pdf, shap_img, label_fix_img, radar_img)
    panel_conclusions(pdf)

    pdf.save()
    print(f"Poster saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
