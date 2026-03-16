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


S_BODY = _sty("body", 21, leading=26)
S_SMALL = _sty("small", 20, leading=25)
S_TABLE = _sty("table", 20, leading=24)
S_REF = _sty("ref", 18, leading=22)
S_CAP = _sty("cap", 18, C_SUB, align=TA_CENTER)

# Aliases (panels 1,3,4,5 previously used _L variants — now unified)
S_BODY_L = S_BODY
S_SMALL_L = S_SMALL


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
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
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

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
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

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
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
    """Grouped bar chart comparing 5 models on Accuracy and F1 macro."""
    _chart_style()
    models = ["SVM", "RF", "CNN", "Transformer", "Att-LSTM"]
    accuracy = [94.2, 94.2, 93.2, 92.5, 74.4]
    f1_macro = [84.3, 84.3, 82.2, 80.2, 0]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy (%)",
                   color="#2980b9", edgecolor="white")
    bars2 = ax.bar(x + width/2, f1_macro, width, label="F1 macro (x100)",
                   color="#1a3c5e", edgecolor="white")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", fontsize=8,
                fontweight="bold", color="#2980b9")
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8,
                    fontweight="bold", color="#1a3c5e")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison (5-fold CV, 15,000 spectra)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_image(fig)


def make_per_class_chart():
    """Grouped bar chart for per-class Precision/Recall/F1."""
    _chart_style()
    classes = ["Plasma OFF\n(12.1%)", "Plasma ON\n(87.9%)"]
    precision = [0.87, 0.95]
    recall = [0.61, 0.99]
    f1 = [0.717, 0.968]

    x = np.arange(len(classes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.bar(x - width, precision, width, label="Precision", color="#2980b9")
    ax.bar(x, recall, width, label="Recall", color="#e67e22")
    ax.bar(x + width, f1, width, label="F1", color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Per-Class Performance (RF)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig_to_image(fig)


def make_actinometry_chart():
    """Bar chart of actinometry ratios (I_species / I_Ar)."""
    _chart_style()
    species = ["F I", "SiF", "CO", "C2\nSwan", "N2\n2pos", "O I", "Si I", "CF2"]
    ratios = [1.070, 1.002, 1.058, 1.126, 0.963, 0.967, 0.958, 0.957]
    errors = [0.044, 0.043, 0.100, 0.204, 0.031, 0.043, 0.036, 0.039]

    colors = ["#c8102e" if r > 1.05 else "#1a3c5e" for r in ratios]

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    bars = ax.bar(species, ratios, color=colors, edgecolor="white", width=0.6)
    ax.errorbar(range(len(species)), ratios, yerr=errors, fmt="none",
                ecolor="#555555", capsize=3, capthick=1)
    ax.axhline(y=1.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("I_species / I_Ar", fontsize=10)
    ax.set_title("Actinometry Ratios (ref: Ar I)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 1.4)
    ax.tick_params(labelsize=9)
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

    fig, ax = plt.subplots(figsize=(5.2, 2.0))
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

    # Title (large, fills banner)
    sty_t = _sty("banner_title", 40, white, bold=True, align=TA_CENTER, leading=44)
    draw_para(c,
              "Machine Learning for Spectral Analysis",
              title_x, by + BANNER_H - 3 * mm, title_w, sty_t)

    # Authors
    sty_a = _sty("banner_authors", 17, HexColor("#d4e6f1"), align=TA_CENTER, leading=20)
    draw_para(c,
              "Liangqing Luo &nbsp;|&nbsp; Supervisor: Dr Xin Tu &nbsp;|&nbsp; "
              "Assessor: Dr Xue Yong",
              title_x, by + 14 * mm, title_w, sty_a)

    # Department
    sty_d = _sty("banner_dept", 14, HexColor("#a9cce3"), align=TA_CENTER, leading=17)
    draw_para(c,
              "Department of Electrical Engineering and Electronics, "
              "University of Liverpool",
              title_x, by + 4 * mm, title_w, sty_d)


# ══════════════════════════════════════════════════════════════════
#  PANEL CHROME
# ══════════════════════════════════════════════════════════════════

HEADER_H = 14 * mm

# Per-panel background tints (subtle, pastel)
PANEL_BG = {
    (0, 0): HexColor("#eef5fb"),  # 1. Introduction — light blue
    (1, 0): HexColor("#ffffff"),  # 2. Methodology — white (flowchart has colours)
    (2, 0): HexColor("#eafaf1"),  # 3. Species ID — light green
    (0, 1): HexColor("#fef9e7"),  # 4. Classification — light yellow (highlight results)
    (1, 1): HexColor("#f4ecf7"),  # 5. Interpretability — light purple
    (2, 1): HexColor("#eef5fb"),  # 6. Conclusions — light blue
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

    # Header text (large, bold)
    c.setFillColor(C_NAV)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x + PAD, y + h - HEADER_H + 1 * mm, title)

    # Thick navy line under header
    c.setStrokeColor(C_NAV)
    c.setLineWidth(2.0)
    c.line(x + PAD, y + h - HEADER_H - 1 * mm, x + w - PAD, y + h - HEADER_H - 1 * mm)

    return x + PAD, y + h - HEADER_H - 5 * mm, w - 2 * PAD


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
    dy = draw_para(c, intro_text, x, y, w, S_BODY_L)
    y -= dy + 3 * mm

    dy = draw_para(c,
        "This project develops an <b>automated ML pipeline</b> for plasma "
        "OES analysis targeting four objectives:",
        x, y, w, S_BODY_L)
    y -= dy + 3 * mm

    # 4 objectives as 2x2 grid flowchart
    objectives = [
        ("1. Feature ID", "Peak detection, NMF,\nNIST matching", HexColor("#2980b9")),
        ("2. Classification", "SVM, RF, CNN,\nTransformer (6 models)", HexColor("#8e44ad")),
        ("3. Temporal", "Attention-LSTM,\nSpecies time-series", HexColor("#27ae60")),
        ("4. Intensity", "Actinometry,\nBoltzmann Te", HexColor("#e67e22")),
    ]
    grid_gap = 4 * mm
    obj_w = (w - grid_gap) / 2
    obj_h = 22 * mm
    for i, (title, desc, color) in enumerate(objectives):
        col_i = i % 2
        row_i = i // 2
        bx = x + col_i * (obj_w + grid_gap)
        by_box = y - obj_h - row_i * (obj_h + grid_gap)
        rrect(c, bx, by_box, obj_w, obj_h, r=3 * mm, fill=color)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(bx + obj_w / 2, by_box + obj_h - 8 * mm, title)
        c.setFont("Helvetica", 12)
        lines = desc.split("\n")
        for j, line in enumerate(lines):
            c.drawCentredString(bx + obj_w / 2, by_box + obj_h - 14 * mm - j * 12, line)
    y -= 2 * (obj_h + grid_gap) + 2 * mm

    dy = draw_para(c, "<b>Three public datasets:</b>",
                   x, y, w, S_BODY_L)
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
    sub_h = 14 * mm
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
            sub_style = _sty(f"sub_{i}", 16, C_TEXT, align=TA_CENTER, leading=15)
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
    note_style = _sty("optuna_note", 20, C_SUB, leading=24)
    draw_para(c,
              "<i>Hyperparameter optimisation: Optuna two-stage search "
              "(20 trials per target)</i>",
              x, y, w, note_style)
    y -= 16 * mm

    # Key Design Decisions table
    dy = draw_para(c, "<b>Key Design Decisions</b>", x, y, w,
                   _sty("kdd_t", 22, C_NAV, bold=True, leading=26))
    y -= dy + 2 * mm

    dy = _draw_simple_table(c, x, y, w,
                            headers=["Decision", "Rationale"],
                            col_fracs=[0.35, 0.65],
                            rows=[
                                ["Per-element routing", "Cr: Ridge+PCA; others: ANN+NIST"],
                                ["GroupKFold CV", "Prevents same-target leakage"],
                                ["Balanced weights", "OFF=12.1%, weighted CE loss"],
                                ["NMF over PCA", "Non-negative = physical spectra"],
                            ])
    y -= dy + 2 * mm

    y -= 3 * mm
    # Preprocessing details
    prep_detail = (
        "<b>Preprocessing rationale:</b> ALS baseline (\u03bb=10<super>5</super>) "
        "removes fluorescence continuum. Savitzky-Golay (window=11, order=3) "
        "preserves peak shapes while reducing shot noise. SNV normalisation "
        "corrects for optical path length variation between measurements. "
        "Cosmic ray removal uses Z-score median filter (threshold=5\u03c3, "
        "11-channel local window)."
    )
    draw_para(c, prep_detail, x, y, w, _sty("prep_d", 21, C_TEXT, leading=24))


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
    dy = draw_para(c, intro, x, y, w, S_SMALL_L)
    y -= dy + 2 * mm

    dy = draw_para(c,
        "<b>Automated NIST matching:</b> Each detected peak is compared against "
        "39 reference emission lines from 13 species. The algorithm selects the "
        "closest database match within +/-1.5 nm tolerance. Species with "
        "peak intensity &gt; \u03bc + 3\u03c3 (global spectrum statistics) "
        "are classified as <i>present</i>.",
        x, y, w, _sty("nist_detail", 21, C_TEXT, leading=24))
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
    draw_para(c, nmf_note, x, y, w, _sty("nmf_note", 21, C_SUB, leading=24))


# ══════════════════════════════════════════════════════════════════
#  PANEL 4: Classification Results
# ══════════════════════════════════════════════════════════════════

def panel_classification(c, model_img, perclass_img):
    x, y, w = _panel_chrome(c, 0, 1, "4. Classification Results")

    # Highlight box with UoL red border
    highlight_h = 14 * mm
    rrect(c, x - 2 * mm, y - highlight_h, w + 4 * mm, highlight_h,
          r=3 * mm, fill=C_HIGH_BG, stroke=C_UOL_RED, stroke_width=2.0)
    sty_hl = _sty("highlight", 22, C_UOL_RED, bold=True, align=TA_CENTER,
                   leading=18)
    draw_para(c, "94.2% Accuracy (SVM/RF, 5-fold CV)",
              x, y - 3 * mm, w, sty_hl)
    y -= highlight_h + 3 * mm

    # Model comparison CHART
    if model_img:
        img_w = w + 2 * mm
        img_h = img_w * 0.54
        c.drawImage(model_img, x - 1 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Per-class CHART
    if perclass_img:
        img_w = w * 0.92
        img_h = img_w * 0.60
        offset_x = x + (w - img_w) / 2
        c.drawImage(perclass_img, offset_x, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 2 * mm

    # Analysis (concise, 3 key points)
    analyses = [
        "<b>ML &gt; DL:</b> Near-linear decision boundary favours SVM/RF. "
        "DL needs more data to match.",
        "<b>Class imbalance:</b> OFF only 12.1%, recall=0.61. "
        "Balanced weights help but don't fully eliminate bias.",
        "<b>Physics validated:</b> Ar/F dominant (SF6/Ar gas); "
        "C2 only during C4F8 passivation steps.",
    ]
    for a in analyses:
        dy = draw_para(c, f"\u2022 {a}", x + 1 * mm, y, w - 2 * mm, S_SMALL_L)
        y -= dy + 1.5 * mm


# ══════════════════════════════════════════════════════════════════
#  PANEL 5: Interpretability & Physics
# ══════════════════════════════════════════════════════════════════

def panel_interpretability(c, shap_img, actin_img):
    x, y, w = _panel_chrome(c, 1, 1, "5. Interpretability & Physics")

    # SHAP chart (enlarged)
    if shap_img:
        img_w = w + 2 * mm
        img_h = img_w * 0.56
        c.drawImage(shap_img, x - 1 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3 * mm

    # Key finding box
    finding = (
        "<b>Root Cause Discovery:</b> Etch/passivation spectral difference "
        "&lt; 0.35\u03c3 (indistinguishable). RF power ON/OFF provides "
        "0.56\u03c3 separation. Classification: 74% &rarr; <b>94%</b>."
    )
    p_tmp = Paragraph(finding, S_SMALL_L)
    _, fh = p_tmp.wrap(w - 4 * mm, 999 * mm)
    box_h = fh + 4 * mm
    rrect(c, x - 1 * mm, y - box_h, w + 2 * mm, box_h,
          r=2 * mm, fill=HexColor("#e8f4fd"))
    draw_para(c, finding, x + 1 * mm, y - 2 * mm, w - 4 * mm, S_SMALL_L)
    y -= box_h + 3 * mm

    # Actinometry CHART (enlarged)
    if actin_img:
        img_w = w + 2 * mm
        img_h = img_w * 0.50
        c.drawImage(actin_img, x - 1 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3 * mm

    # Boltzmann + Temporal (concise)
    boltz_temporal = (
        "<b>Boltzmann T_exc = 13,334 K</b> (6 Ar I lines). "
        "<b>Temporal:</b> Attention-LSTM 74.4% phase classification; "
        "DTW K-Means (k=4) discovers 4 discharge phases with "
        "F I 684 nm emission ratio <b>2.04x</b> between clusters."
    )
    draw_para(c, boltz_temporal, x, y, w, S_SMALL_L)


# ══════════════════════════════════════════════════════════════════
#  PANEL 6: Conclusions & Further Work
# ══════════════════════════════════════════════════════════════════

def panel_conclusions(c):
    x, y, w = _panel_chrome(c, 2, 1, "6. Conclusions & Further Work")

    dy = draw_para(c, "<b>Key Achievements:</b>", x, y, w,
                   _sty("ka", 22, C_NAV, bold=True))
    y -= dy + 2 * mm

    achievements = [
        "<b>94.2%</b> plasma state classification (6 models compared)",
        "<b>13 species</b> auto-detected via NMF + NIST (39 lines)",
        "<b>SHAP:</b> F I = 0.131 importance (primary etchant, validated)",
        "<b>Label correction:</b> 74% &rarr; 94% via root-cause analysis",
        "<b>T_rot = 20.0 K, T_vib = 102.0 K</b> (CAP regression)",
        "<b>78 tests,</b> 6 CLI modes, fully reproducible",
    ]
    for a in achievements:
        dy = draw_para(c, f"&#10003;&nbsp; {a}", x + 2 * mm, y, w - 4 * mm, S_SMALL)
        y -= dy + 1.5 * mm

    y -= 3 * mm
    dy = draw_para(c, "<b>Limitations:</b>", x, y, w,
                   _sty("lim", 22, C_NAV, bold=True))
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
                   _sty("fw", 22, C_NAV, bold=True))
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
    p_tmp = Paragraph(metrics_box, _sty("met_tmp", 17, C_TEXT, leading=20))
    _, mh = p_tmp.wrap(w - 4 * mm, 999 * mm)
    box_h = mh + 4 * mm
    rrect(c, x - 1 * mm, y - box_h, w + 2 * mm, box_h,
          r=2 * mm, fill=HexColor("#e8f4fd"))
    draw_para(c, metrics_box, x + 1 * mm, y - 2 * mm, w - 4 * mm,
              _sty("met_box", 17, C_TEXT, leading=20))
    y -= box_h + 3 * mm

    y -= 4 * mm
    dy = draw_para(c, "<b>References:</b>", x, y, w,
                   _sty("refs_hdr", 19, C_NAV, bold=True))
    y -= dy + 1 * mm

    refs = [
        "[1] Gidon <i>et al.</i> (2019) IEEE Trans. Radiat. Plasma Med. Sci.",
        "[2] Coburn &amp; Chen (1980) J. Appl. Phys. \u2014 Actinometry",
        "[3] Vaswani <i>et al.</i> (2017) NeurIPS \u2014 Transformer",
        "[4] Contreras <i>et al.</i> (2024) Anal. Chem. \u2014 Spectral-zone SHAP",
        "[5] BOSCH dataset: Zenodo #17122442",
    ]
    for ref in refs:
        dy = draw_para(c, ref, x + 1 * mm, y, w - 2 * mm, _sty("ref_item", 15, C_TEXT, leading=18))
        y -= dy + 1 * mm


# ══════════════════════════════════════════════════════════════════
#  TABLE DRAWING HELPER
# ══════════════════════════════════════════════════════════════════

def _draw_simple_table(c, x, y, w, headers, col_fracs, rows):
    """Draw a simple table with alternating row backgrounds.

    Returns total height consumed.
    """
    cw = [w * f for f in col_fracs]
    hdr_h = 8 * mm
    row_h = 7 * mm

    # Header row
    c.setFillColor(C_NAV)
    c.rect(x, y - hdr_h, w, hdr_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 16)
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
        model_img = make_model_comparison_chart()
        print("  [OK] Model comparison chart")
    except Exception as e:
        print(f"  [SKIP] Model comparison chart: {e}")
        model_img = None

    try:
        perclass_img = make_per_class_chart()
        print("  [OK] Per-class chart")
    except Exception as e:
        print(f"  [SKIP] Per-class chart: {e}")
        perclass_img = None

    try:
        actin_img = make_actinometry_chart()
        print("  [OK] Actinometry chart")
    except Exception as e:
        print(f"  [SKIP] Actinometry chart: {e}")
        actin_img = None

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
    panel_classification(pdf, model_img, perclass_img)
    panel_interpretability(pdf, shap_img, actin_img)
    panel_conclusions(pdf)

    pdf.save()
    print(f"Poster saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
