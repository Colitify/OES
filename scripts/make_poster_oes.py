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


S_BODY = _sty("body", 18, leading=22)
S_SMALL = _sty("small", 17, leading=21)
S_TABLE = _sty("table", 17, leading=20)
S_REF = _sty("ref", 16, leading=19)
S_CAP = _sty("cap", 16, C_SUB, align=TA_CENTER)


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


def make_kdd_chart():
    """Key Design Decisions as a visual comparison table/chart."""
    _chart_style()
    decisions = ['Per-element\nRouting', 'GroupKFold\nCV', 'Balanced\nWeights', 'NMF over\nPCA']
    impacts = ['Cr: Ridge+PCA\nOthers: ANN+NIST', 'Prevents\nsample leakage', 'Fixes 12.1%\nOFF imbalance', 'Non-negative\n= physical']
    colors = [H_NAV, '#2471a3', '#8e44ad', '#27ae60']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(decisions))
    bars = ax.barh(y_pos, [85, 75, 70, 80], color=colors, height=0.6, alpha=0.85)

    for i, (dec, imp) in enumerate(zip(decisions, impacts)):
        ax.text(3, i, dec, va='center', fontsize=13, fontweight='bold', color='white')
        ax.text(88, i, imp, va='center', fontsize=11, color=colors[i], fontweight='bold')

    ax.set_xlim(0, 160)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Key Design Decisions', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_conclusions_chart():
    """Summary infographic for conclusions panel."""
    _chart_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # 4 achievement boxes
    items = [
        (50, 85, '13 Species Detected', '39 NIST lines, NMF validated', H_NAV),
        (50, 65, '94.2% Accuracy | F1=0.843', '6 models compared (SVM best)', '#c8102e'),
        (50, 45, 'Label Correction: 74% -> 94%', 'Data quality > model complexity', '#27ae60'),
        (50, 25, 'SHAP Validates Physics', 'F I = primary SF6 etchant', '#8e44ad'),
    ]

    for cx, cy, title, subtitle, color in items:
        ax.add_patch(plt.Rectangle((5, cy-8), 90, 16, facecolor=color, alpha=0.12,
                                    edgecolor=color, linewidth=2, zorder=1))
        ax.text(10, cy+2, title, fontsize=14, fontweight='bold', color=color, va='center')
        ax.text(10, cy-4, subtitle, fontsize=11, color='#555555', va='center')

    # Future work at bottom
    ax.text(50, 6, 'Future: Multi-class tracking | Real-time streaming | Transfer learning',
            fontsize=10, ha='center', color='#888888', style='italic')

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

    # Title
    sty_t = _sty("banner_title", 30, white, bold=True, align=TA_CENTER, leading=36)
    draw_para(c,
              "Machine Learning for Spectral Analysis",
              title_x, by + BANNER_H - 4 * mm, title_w, sty_t)

    # Authors
    sty_a = _sty("banner_authors", 15, HexColor("#d4e6f1"), align=TA_CENTER, leading=18)
    draw_para(c,
              "Liangqing Luo &nbsp;|&nbsp; Supervisor: Dr Xin Tu &nbsp;|&nbsp; "
              "Assessor: Dr Xue Yong",
              title_x, by + 16 * mm, title_w, sty_a)

    # Department
    sty_d = _sty("banner_dept", 13, HexColor("#a9cce3"), align=TA_CENTER, leading=16)
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

    # Header text
    c.setFillColor(C_NAV)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x + PAD, y + h - HEADER_H + 2 * mm, title)

    # Thin navy line under header
    c.setStrokeColor(C_NAV)
    c.setLineWidth(1.0)
    c.line(x + PAD, y + h - HEADER_H, x + w - PAD, y + h - HEADER_H)

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

    y -= 3 * mm
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
    sub_h = 10 * mm
    arrow_gap = 5 * mm
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
            sub_style = _sty(f"sub_{i}", 14, C_TEXT, align=TA_CENTER, leading=15)
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
    note_style = _sty("optuna_note", 17, C_SUB, leading=18)
    draw_para(c,
              "<i>Hyperparameter optimisation: Optuna two-stage search "
              "(20 trials per target)</i>",
              x, y, w, note_style)
    y -= 16 * mm

    # Key Design Decisions — as chart
    if hasattr(panel_method, '_kdd_img') and panel_method._kdd_img:
        img_w = w + 4 * mm
        img_h = img_w * 0.55
        c.drawImage(panel_method._kdd_img, x - 2 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")


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
        "<b>Automated NIST matching:</b> Each detected peak is compared against "
        "39 reference emission lines from 13 species. The algorithm selects the "
        "closest database match within +/-1.5 nm tolerance. Species with "
        "peak intensity &gt; \u03bc + 3\u03c3 (global spectrum statistics) "
        "are classified as <i>present</i>.",
        x, y, w, _sty("nist_detail", 17, C_TEXT, leading=18))
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
    draw_para(c, nmf_note, x, y, w, _sty("nmf_note", 17, C_SUB, leading=18))


# ══════════════════════════════════════════════════════════════════
#  PANEL 4: Classification Results
# ══════════════════════════════════════════════════════════════════

def panel_classification(c):
    x, y, w = _panel_chrome(c, 0, 1, "4. Classification Results")

    # Highlight box with UoL red border
    highlight_h = 14 * mm
    rrect(c, x - 2 * mm, y - highlight_h, w + 4 * mm, highlight_h,
          r=3 * mm, fill=C_HIGH_BG, stroke=C_UOL_RED, stroke_width=2.0)
    sty_hl = _sty("highlight", 18, C_UOL_RED, bold=True, align=TA_CENTER,
                   leading=18)
    draw_para(c, "94.2% Accuracy (SVM/RF, 5-fold CV)",
              x, y - 3 * mm, w, sty_hl)
    y -= highlight_h + 4 * mm

    # Model comparison table
    dy = _draw_simple_table(c, x, y, w,
                            headers=["Model", "Accuracy", "F1 macro"],
                            col_fracs=[0.46, 0.27, 0.27],
                            rows=[
                                ["SVM (RBF)", "94.2%", "0.843"],
                                ["Random Forest", "94.2%", "0.843"],
                                ["CNN (1D-Conv)", "93.2%", "0.822"],
                                ["Transformer", "92.5%", "0.802"],
                                ["Attention-LSTM", "74.4%", "\u2014"],
                            ])
    y -= dy + 4 * mm

    # Per-class table
    dy = _draw_simple_table(c, x, y, w,
                            headers=["Class", "Precision", "Recall", "F1"],
                            col_fracs=[0.34, 0.22, 0.22, 0.22],
                            rows=[
                                ["Plasma OFF", "0.87", "0.61", "0.717"],
                                ["Plasma ON", "0.95", "0.99", "0.968"],
                            ])
    y -= dy + 4 * mm

    # Per-species detection rates table
    dy = draw_para(c, "<b>Species Detection Rates (13 species, 15,000 spectra):</b>",
                   x, y, w, _sty("sp_hdr", 18, C_NAV, bold=True))
    y -= dy + 2 * mm

    dy = _draw_simple_table(c, x, y, w,
                            headers=["Species", "Detection", "Species", "Detection"],
                            col_fracs=[0.24, 0.26, 0.24, 0.26],
                            rows=[
                                ["Ar I", "69.8%", "C2 Swan", "23.8%"],
                                ["F I", "68.4%", "CO", "20.8%"],
                                ["N2 2pos", "0.7%", "Si I", "0.4%"],
                            ])
    y -= dy + 3 * mm

    # Model architecture details
    dy = draw_para(c, "<b>Model Architectures:</b>", x, y, w,
                   _sty("arch_h", 18, C_NAV, bold=True))
    y -= dy + 1.5 * mm

    archs = [
        "<b>SVM/RF:</b> StandardScaler \u2192 Classifier pipeline. RF uses 200 trees; "
        "SVM uses RBF kernel (C=10, \u03b3=scale). Both with balanced class weights.",
        "<b>CNN:</b> 3-layer Conv1D (32\u219264\u2192128 channels, kernel 7/5/3) "
        "\u2192 AdaptiveAvgPool \u2192 FC(64) \u2192 Dropout(0.3) \u2192 output. "
        "Weighted CrossEntropyLoss, mini-batch training.",
        "<b>Transformer:</b> ViT-style 1D patch embedding (patch=64, d=128, 4 heads, "
        "3 layers). [CLS] token classification. AdamW + cosine LR schedule.",
        "<b>Attention-LSTM:</b> 2-layer LSTM (hidden=64) \u2192 additive attention "
        "\u2192 FC. Trained on PCA(20) sliding windows (seq_len=10).",
    ]
    for a in archs:
        dy = draw_para(c, f"\u2022 {a}", x + 1 * mm, y, w - 2 * mm,
                       _sty("arch_item", 16, C_TEXT, leading=17))
        y -= dy + 1.5 * mm

    # Results analysis
    dy = draw_para(c, "<b>Analysis:</b>", x, y, w,
                   _sty("analysis_h", 18, C_NAV, bold=True))
    y -= dy + 1.5 * mm

    analyses = [
        "<b>Traditional ML &gt; Deep Learning:</b> SVM/RF achieve 94.2% vs "
        "CNN 93.2% and Transformer 92.5%. The plasma ON/OFF decision boundary "
        "is near-linear in feature space, giving kernel methods the advantage. "
        "Deep learning requires larger datasets to learn comparable representations.",
        "<b>Class imbalance impact:</b> Plasma OFF is only 12.1% of data. "
        "Despite balanced class weights, OFF recall = 0.61 (39% missed). "
        "The model is conservative in predicting OFF states, preferring "
        "high-confidence ON predictions (recall = 0.99).",
        "<b>Attention-LSTM underperforms (74.4%):</b> Temporal sequence "
        "classification on PCA(20) sliding windows loses spectral detail. "
        "Direct spectral classification (SVM/RF on 3,648 channels) retains "
        "full wavelength information, explaining the 20% accuracy gap.",
        "<b>Species detection validates physics:</b> Ar I (69.8%) and F I "
        "(68.4%) are dominant &mdash; consistent with SF6/Ar "
        "process gas. C2 Swan (23.8%) only appears during "
        "C4F8 passivation steps.",
    ]
    for a in analyses:
        dy = draw_para(c, f"\u2022 {a}", x + 1 * mm, y, w - 2 * mm,
                       _sty("analysis_item", 16, C_TEXT, leading=19))
        y -= dy + 2 * mm


# ══════════════════════════════════════════════════════════════════
#  PANEL 5: Interpretability & Physics
# ══════════════════════════════════════════════════════════════════

def panel_interpretability(c, shap_img):
    x, y, w = _panel_chrome(c, 1, 1, "5. Interpretability & Physics")

    # SHAP chart
    if shap_img:
        img_w = w
        img_h = img_w * 0.56
        c.drawImage(shap_img, x, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 4 * mm

    text1 = (
        "F I (fluorine radical) identified as the most discriminative "
        "species \u2014 consistent with its role as the primary etchant "
        "in SF6 plasma."
    )
    dy = draw_para(c, text1, x, y, w, S_BODY)
    y -= dy + 4 * mm

    # Key finding box
    finding = (
        "<b>Root Cause Discovery:</b> Etch/passivation spectral difference "
        "&lt; 0.35\u03c3 (indistinguishable). RF power ON/OFF provides "
        "0.56\u03c3 separation. Classification improved 74% \u2192 94%."
    )
    p_tmp = Paragraph(finding, S_BODY)
    _, fh = p_tmp.wrap(w - 6 * mm, 999 * mm)
    box_h = fh + 6 * mm
    rrect(c, x - 2 * mm, y - box_h, w + 4 * mm, box_h,
          r=3 * mm, fill=HexColor("#e8f4fd"))
    draw_para(c, finding, x + 1 * mm, y - 3 * mm, w - 4 * mm, S_BODY)
    y -= box_h + 5 * mm

    # Boltzmann result
    boltz = (
        "<b>Boltzmann T_exc:</b> Excitation temperature "
        "= <b>13,334 K</b> (Boltzmann plot, 6 Ar I lines, 696.5\u2013772.4 nm). "
        "Estimated via linear regression of ln(I\u00b7\u03bb/gA) vs E_upper."
    )
    dy = draw_para(c, boltz, x, y, w, S_SMALL)
    y -= dy + 3 * mm

    # Physics of F_I dominance
    fi_physics = (
        "<b>Why F I dominates:</b> In SF6 plasma, electron-impact "
        "dissociation produces F radicals "
        "(SF6 + e- &rarr; SF5 + F + e-). "
        "The F I 703.7 nm line (2p4 3p &rarr; 2p4 3s transition, "
        "upper state 14.5 eV) has high transition probability "
        "(A = 6.4 x 10^7 /s) "
        "and is well-separated from neighbouring lines. Its intensity directly tracks "
        "F radical density, making it the most sensitive probe of etch chemistry."
    )
    dy = draw_para(c, fi_physics, x, y, w, _sty("fi_phys", 17, C_TEXT, leading=19))
    y -= dy + 3 * mm

    # Actinometry explanation
    actin = (
        "<b>Actinometry:</b> Species concentration proportional to "
        "I_target / I_Ar (Coburn &amp; Chen 1980). "
        "Ar carrier gas at known constant flow serves as reference. "
        "F/Ar ratio = 1.07 +/- 0.04; "
        "C2/Ar ratio = 1.13 +/- 0.20 (largest variability "
        "&mdash; reflects etch/passivation cycling)."
    )
    dy = draw_para(c, actin, x, y, w, S_SMALL)
    y -= dy + 3 * mm

    # Temporal analysis
    temporal = (
        "<b>Temporal analysis:</b> Attention-LSTM achieves <b>74.4%</b> "
        "phase classification accuracy on PCA(20) embedding sequences. "
        "Attention weights reveal that transition timesteps between plasma "
        "states carry highest diagnostic information. "
        "DTW K-Means identifies 4 discharge phases (ignition, steady-state, "
        "transition, extinction) with 684 nm emission ratio &gt; 2x "
        "between clusters."
    )
    draw_para(c, temporal, x, y, w, S_SMALL)


# ══════════════════════════════════════════════════════════════════
#  PANEL 6: Conclusions & Further Work
# ══════════════════════════════════════════════════════════════════

def panel_conclusions(c, conclusions_img):
    x, y, w = _panel_chrome(c, 2, 1, "6. Conclusions & Further Work")

    # Conclusions infographic chart
    if conclusions_img:
        img_w = w + 4 * mm
        img_h = img_w * 0.64
        c.drawImage(conclusions_img, x - 2 * mm, y - img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 4 * mm

    # Limitations (brief)
    dy = draw_para(c, "<b>Limitations:</b>", x, y, w,
                   _sty("lim", 18, C_NAV, bold=True))
    y -= dy + 2 * mm
    lims = [
        "OES&rarr;process regression R2 &lt; 0 (causal irreversibility)",
        "Boltzmann Te: narrow Ar I energy spread (0.33 eV)",
    ]
    for lim in lims:
        dy = draw_para(c, f"&#8226; {lim}", x + 2 * mm, y, w - 4 * mm, S_REF)
        y -= dy + 1.5 * mm

    # References
    y -= 3 * mm
    dy = draw_para(c, "<b>References:</b>", x, y, w,
                   _sty("refs_hdr", 18, C_NAV, bold=True))
    y -= dy + 1 * mm
    refs = [
        "[1] Gidon <i>et al.</i> (2019) IEEE Trans. Radiat. Plasma Med. Sci.",
        "[2] Coburn &amp; Chen (1980) J. Appl. Phys.",
        "[3] Lee &amp; Seung (1999) Nature \u2014 NMF",
        "[4] Lundberg &amp; Lee (2017) NeurIPS \u2014 SHAP",
        "[5] Vaswani <i>et al.</i> (2017) NeurIPS \u2014 Transformer",
        "[6] Contreras <i>et al.</i> (2024) Anal. Chem.",
        "[7] BOSCH dataset: Zenodo #17122442",
    ]
    for ref in refs:
        dy = draw_para(c, ref, x + 1 * mm, y, w - 2 * mm, _sty("r", 14, C_SUB, leading=17))
        y -= dy + 0.5 * mm


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
    c.setFont("Helvetica-Bold", 14)
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
            c.setFont("Helvetica", 14)
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
        kdd_img = make_kdd_chart()
        print("  [OK] KDD chart")
    except Exception as e:
        print(f"  [SKIP] KDD chart: {e}")
        kdd_img = None

    try:
        conclusions_img = make_conclusions_chart()
        print("  [OK] Conclusions chart")
    except Exception as e:
        print(f"  [SKIP] Conclusions chart: {e}")
        conclusions_img = None

    # Store KDD image for panel_method to access
    panel_method._kdd_img = kdd_img

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
    panel_classification(pdf)
    panel_interpretability(pdf, shap_img)
    panel_conclusions(pdf, conclusions_img)

    pdf.save()
    print(f"Poster saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
