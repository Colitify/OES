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

# ── Page setup ─────────────────────────────────────────
PAGE_W = 594 * mm
PAGE_H = 841 * mm

# ── Colours ────────────────────────────────────────────
C_NAV = HexColor("#1a3c5e")
C_RED = HexColor("#c8102e")
C_PANEL = HexColor("#ffffff")
C_TEXT = HexColor("#2c3e50")
C_SUB = HexColor("#666666")
C_LGREY = HexColor("#f5f5f5")
C_BORDER = HexColor("#1a3c5e")
C_HIGH_BG = HexColor("#fef3f3")

H_NAV = "#1a3c5e"
H_BLUE = "#2980b9"
H_RED = "#c8102e"
H_GREEN = "#27ae60"
H_ORANGE = "#e67e22"
H_GREY = "#bdc3c7"

# ── Layout ─────────────────────────────────────────────
MARGIN = 14 * mm
COL_GAP = 7 * mm
ROW_GAP = 7 * mm
BANNER_H = 65 * mm
PAD = 9 * mm
RAD = 4 * mm
BORDER_W = 1.5

CONTENT_TOP = PAGE_H - MARGIN - BANNER_H
CONTENT_BOT = MARGIN
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_H = CONTENT_TOP - CONTENT_BOT

COL_W = (CONTENT_W - 2 * COL_GAP) / 3
ROW1_H = CONTENT_H * 0.48
ROW2_H = CONTENT_H - ROW1_H - ROW_GAP

def col_x(i):
    return MARGIN + i * (COL_W + COL_GAP)

def row_y(j):
    return CONTENT_TOP - ROW1_H if j == 0 else CONTENT_BOT

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
        leading=leading or sz * 1.25,
    )

S_BODY = _sty("body", 26, leading=32)
S_SMALL = _sty("small", 24, leading=30)
S_REF = _sty("ref", 22, leading=27)

def draw_para(c, text, x, y, w, style):
    p = Paragraph(text, style)
    _, h = p.wrap(w, 2000 * mm)
    p.drawOn(c, x, y - h)
    return h

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)

def _chart_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

# ══════════════════════════════════════════════════════
#  CHARTS — diverse types and colors
# ══════════════════════════════════════════════════════

def make_species_chart():
    _chart_style()
    species = ["N2", "CO", "C2 Swan", "F I", "Ar I"]
    rates = [0.7, 20.8, 23.8, 68.4, 69.8]
    colors = [H_NAV, H_GREEN, H_GREEN, H_RED, H_BLUE]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(species, rates, color=colors, height=0.55)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontsize=14, fontweight="bold", color=H_NAV)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Species Detection (15,000 spectra)")
    fig.tight_layout()
    return fig_to_image(fig)

def make_shap_chart():
    _chart_style()
    features = ["CO", "H_beta", "O I", "C2 Swan", "F I"]
    values = [0.033, 0.040, 0.041, 0.046, 0.131]
    colors = [H_NAV]*4 + [H_RED]  # F I highlighted red
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features, values, color=colors, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=14, fontweight="bold",
                color=bar.get_facecolor())
    ax.set_xlim(0, 0.16)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (RF)")
    fig.tight_layout()
    return fig_to_image(fig)

def make_model_chart():
    _chart_style()
    models = ['SVM', 'RF', 'CNN', 'Transf.', 'LSTM']
    f1 = [84.3, 84.3, 82.2, 80.2, 50.0]
    acc = [94.2, 94.2, 93.2, 92.5, 74.4]
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, acc, w, label='Accuracy (%)', color=H_NAV)
    b2 = ax.bar(x + w/2, f1, w, label='F1 macro (%)', color=H_RED)
    for bar, val in zip(b1, acc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val}', ha='center', fontsize=12, fontweight='bold', color=H_NAV)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison (5-fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=13, loc='lower left')
    fig.tight_layout()
    return fig_to_image(fig)

def make_confusion_chart():
    _chart_style()
    cm = np.array([[1105, 706], [132, 10057]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > 5000 else H_NAV
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred OFF','Pred ON'], fontsize=14)
    ax.set_yticklabels(['True OFF','True ON'], fontsize=14)
    ax.set_title('Confusion Matrix (RF)')
    fig.tight_layout()
    return fig_to_image(fig)

def make_label_fix_chart():
    _chart_style()
    labels = ['Gas Flow\nLabels', 'RF Power\nLabels']
    accuracy = [74.4, 94.2]
    colors = [H_GREY, H_RED]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, accuracy, color=colors, width=0.5)
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f'{val}%', ha='center', fontsize=18, fontweight='bold',
                color=bar.get_facecolor())
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Label Correction Impact')
    ax.set_ylim(0, 112)
    ax.annotate('', xy=(1, 94.2), xytext=(0, 74.4),
                arrowprops=dict(arrowstyle='->', color=H_GREEN, lw=3))
    ax.text(0.5, 83, '+19.8%', ha='center', fontsize=16, fontweight='bold', color=H_GREEN)
    fig.tight_layout()
    return fig_to_image(fig)

def make_actinometry_chart():
    _chart_style()
    species = ['F/Ar', 'C2/Ar', 'CO/Ar', 'O/Ar', 'N2/Ar']
    means = [1.07, 1.13, 1.06, 0.97, 0.96]
    stds = [0.04, 0.20, 0.10, 0.04, 0.03]
    colors = [H_RED, H_ORANGE, H_BLUE, H_NAV, H_NAV]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(species, means, yerr=stds, capsize=5, color=colors, width=0.55, error_kw={'lw':2})
    ax.axhline(y=1.0, color='grey', linestyle='--', lw=1, alpha=0.5)
    ax.set_ylabel('I_species / I_Ar')
    ax.set_title('Actinometry Ratios (Ar Reference)')
    ax.set_ylim(0.5, 1.45)
    ax.annotate('Highest\nvariability', xy=(1, 1.35), fontsize=12,
                ha='center', color=H_ORANGE, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)

def make_kdd_chart():
    """Key Design Decisions as a horizontal info-graphic."""
    _chart_style()
    decisions = ['Per-element\nRouting', 'GroupKFold\nCV', 'Balanced\nWeights', 'NMF over\nPCA']
    impacts = ['Cr:Ridge  Others:ANN', 'No sample leakage', '12.1% OFF fixed', 'Physics constraints']
    colors = [H_NAV, '#2471a3', '#8e44ad', H_GREEN]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(decisions))
    bars = ax.barh(y_pos, [85, 75, 70, 80], color=colors, height=0.6, alpha=0.85)
    for i, (dec, imp) in enumerate(zip(decisions, impacts)):
        ax.text(3, i, dec, va='center', fontsize=13, fontweight='bold', color='white')
        ax.text(88, i, imp, va='center', fontsize=12, color=colors[i], fontweight='bold')
    ax.set_xlim(0, 155)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.set_title('Key Design Decisions', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_conclusions_chart():
    """Summary infographic for conclusions — 4 achievement cards."""
    _chart_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    items = [
        (50, 87, '13 Species Detected', '39 NIST lines | NMF validated', H_NAV),
        (50, 67, '94.2% Accuracy | F1 = 0.843', '6 models compared (SVM/RF best)', H_RED),
        (50, 47, 'Label Correction: 74% -> 94%', 'Data quality > model complexity', H_GREEN),
        (50, 27, 'SHAP Validates Physics', 'F I = primary SF6 etchant radical', '#8e44ad'),
    ]
    for cx, cy, title, subtitle, color in items:
        ax.add_patch(plt.Rectangle((3, cy-8), 94, 16, facecolor=color, alpha=0.12,
                                    edgecolor=color, linewidth=2.5, zorder=1))
        ax.text(8, cy+2, title, fontsize=15, fontweight='bold', color=color, va='center')
        ax.text(8, cy-4, subtitle, fontsize=12, color='#555555', va='center')

    ax.text(50, 8, 'Future: Multi-class tracking | Real-time streaming | Transfer learning',
            fontsize=11, ha='center', color='#888888', style='italic')
    fig.tight_layout()
    return fig_to_image(fig)


def make_spectrum_plot():
    _chart_style()
    np.random.seed(42)
    wl = np.linspace(186, 884, 500)
    baseline = 3800 + 200 * np.sin(wl/200)
    spectrum = baseline + np.random.randn(500) * 50
    peaks_data = {'F I': (685.6,1200,3), 'Ar I': (750.4,1800,2.5),
                  'C2/CO': (517.0,900,5), 'Ha': (656.3,400,3)}
    for name, (c_, h, w_) in peaks_data.items():
        spectrum += h * np.exp(-0.5*((wl-c_)/w_)**2)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(wl, spectrum, color=H_NAV, lw=0.8)
    ax.fill_between(wl, baseline.min(), spectrum, alpha=0.08, color=H_BLUE)
    for name, peak_nm, tx in [('F I',685.6,685.6),('Ar I',750.4,750.4),('C2/CO',517.0,480.0)]:
        idx = np.argmin(np.abs(wl-peak_nm))
        ax.annotate(name, xy=(peak_nm, spectrum[idx]),
                    xytext=(tx, spectrum[idx]+400),
                    fontsize=13, fontweight='bold', color=H_RED,
                    ha='center', arrowprops=dict(arrowstyle='->', color=H_RED, lw=1.2))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(186, 884)
    fig.tight_layout()
    return fig_to_image(fig)

# ══════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════

def draw_banner(c):
    bx, by = MARGIN, PAGE_H - MARGIN - BANNER_H
    bw = PAGE_W - 2 * MARGIN
    rrect(c, bx, by, bw, BANNER_H, r=4*mm, fill=C_NAV)
    tx, tw = bx + 5*mm, bw - 10*mm
    sty_t = _sty("bt", 36, white, bold=True, align=TA_CENTER, leading=43)
    draw_para(c, "Machine Learning for Spectral Analysis", tx, by+BANNER_H-6*mm, tw, sty_t)
    sty_a = _sty("ba", 18, HexColor("#d4e6f1"), align=TA_CENTER, leading=22)
    draw_para(c, "Liangqing Luo &nbsp;|&nbsp; Supervisor: Dr Xin Tu &nbsp;|&nbsp; Assessor: Dr Xue Yong", tx, by+18*mm, tw, sty_a)
    sty_d = _sty("bd", 15, HexColor("#a9cce3"), align=TA_CENTER, leading=18)
    draw_para(c, "Department of Electrical Engineering and Electronics, University of Liverpool", tx, by+8*mm, tw, sty_d)

# ══════════════════════════════════════════════════════
#  PANEL CHROME — navy header bar instead of thin line
# ══════════════════════════════════════════════════════

HEADER_H = 14 * mm

def _panel_chrome(c, col, row, title):
    x, y_ = col_x(col), row_y(row)
    w, h = COL_W, row_h(row)
    # Panel background
    rrect(c, x, y_, w, h, r=RAD, fill=C_PANEL, stroke=C_BORDER, stroke_width=BORDER_W)
    # Navy header bar
    rrect(c, x, y_+h-HEADER_H, w, HEADER_H, r=0, fill=C_NAV)
    # Round top corners overlay
    rrect(c, x, y_+h-HEADER_H, w, HEADER_H, r=RAD, fill=C_NAV)
    # Redraw bottom part to cover rounded bottom of header
    c.saveState()
    c.setFillColor(C_NAV)
    c.rect(x, y_+h-HEADER_H, w, HEADER_H/2, fill=1, stroke=0)
    c.restoreState()
    # Title text
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x + PAD, y_+h-HEADER_H+4*mm, title)
    return x + PAD, y_+h-HEADER_H-7*mm, w - 2*PAD

# ══════════════════════════════════════════════════════
#  PANEL 1: Introduction — with Research Question
# ══════════════════════════════════════════════════════

def panel_intro(c, spectrum_img):
    x, y, w = _panel_chrome(c, 0, 0, "1. Introduction")
    # Research question (italic, highlighted)
    rq = ("<i>\"How can we automate multi-species identification "
          "and plasma state classification from high-dimensional "
          "OES data?\"</i>")
    dy = draw_para(c, rq, x, y, w, _sty("rq", 24, C_RED, bold=False, align=TA_CENTER, leading=30))
    y -= dy + 5*mm
    # Core objectives (larger)
    dy = draw_para(c, "<b>Core contributions:</b>", x, y, w, _sty("co", 26, C_NAV, bold=True, leading=32))
    y -= dy + 3*mm
    for aim in ["<b>Species identification</b> via NMF + NIST matching",
                "<b>Plasma state classification</b> (6 models compared)"]:
        dy = draw_para(c, f"&#9679;&nbsp; {aim}", x+3*mm, y, w-6*mm, S_SMALL)
        y -= dy + 3*mm
    # Secondary (smaller)
    dy = draw_para(c, "Also: actinometry, Boltzmann T_exc, temporal DTW clustering", x, y, w, S_REF)
    y -= dy + 5*mm
    # Spectrum plot
    if spectrum_img:
        img_w = w
        img_h = img_w * 0.43
        c.drawImage(spectrum_img, x, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 5*mm
    # Dataset table
    _draw_table(c, x, y, w, ["Dataset", "Channels", "Task"],
                [0.35, 0.25, 0.40],
                [["BOSCH RIE", "3,648", "Plasma ON/OFF"],
                 ["Mesbah CAP", "51", "T_rot / T_vib"]])

# ══════════════════════════════════════════════════════
#  PANEL 2: Methodology — cleaner flowchart
# ══════════════════════════════════════════════════════

def panel_method(c):
    x, y, w = _panel_chrome(c, 1, 0, "2. Methodology")
    stages = [
        ("Raw OES Spectra", HexColor("#2980b9")),
        ("Preprocess (SNR +10.99 dB)", HexColor("#2471a3")),
        ("Feature Extraction", HexColor("#8e44ad")),
        ("Model Training (x6)", HexColor("#27ae60")),
        ("Evaluate + Interpret", HexColor("#e67e22")),
    ]
    box_w = w - 6*mm
    box_h = 18*mm
    gap = 6*mm
    bx = x + (w-box_w)/2
    for i, (label, color) in enumerate(stages):
        by_ = y - box_h
        rrect(c, bx, by_, box_w, box_h, r=3*mm, fill=color)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(bx+box_w/2, by_+box_h/2-3, label)
        y -= box_h
        if i < len(stages)-1:
            cx = bx + box_w/2
            c.setStrokeColor(C_TEXT); c.setFillColor(C_TEXT); c.setLineWidth(2)
            c.line(cx, y, cx, y-gap+3*mm)
            p = c.beginPath()
            p.moveTo(cx, y-gap); p.lineTo(cx-2.5*mm, y-gap+4*mm); p.lineTo(cx+2.5*mm, y-gap+4*mm); p.close()
            c.drawPath(p, fill=1, stroke=0)
            y -= gap
    y -= 6*mm
    # KDD chart
    if hasattr(panel_method, '_kdd_img') and panel_method._kdd_img:
        img_w = w + 4*mm
        img_h = img_w * 0.50
        c.drawImage(panel_method._kdd_img, x-2*mm, y-img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")

# ══════════════════════════════════════════════════════
#  PANEL 3: Species Identification
# ══════════════════════════════════════════════════════

def panel_species(c, species_img):
    x, y, w = _panel_chrome(c, 2, 0, "3. Species Identification")
    dy = draw_para(c, "<b>NMF (X = W . H)</b> decomposes mixed spectra into pure-species components.", x, y, w, S_BODY)
    y -= dy + 4*mm
    if species_img:
        img_w = w
        img_h = img_w * 0.57
        c.drawImage(species_img, x, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 4*mm
    _draw_table(c, x, y, w, ["Species", "Lines (nm)", "Origin"],
                [0.22, 0.40, 0.38],
                [["F I", "685.6, 703.7", "SF6 etchant"],
                 ["Ar I", "696.5, 750.4", "Carrier gas"],
                 ["C2 Swan", "516.5, 563.6", "C4F8 indicator"]])
    y -= 42*mm
    dy = draw_para(c, "<b>Validation:</b> NMF peaks at 684.4 nm (=F I) and 515.1 nm (=C2) confirm NIST database independently.",
                   x, y, w, _sty("val", 22, C_SUB, leading=28))

# ══════════════════════════════════════════════════════
#  PANEL 4: Classification — F1 as primary metric
# ══════════════════════════════════════════════════════

def panel_classification(c, model_img, conf_img):
    x, y, w = _panel_chrome(c, 0, 1, "4. Classification Results")
    # Highlight F1 macro (not accuracy) to address class imbalance
    hl_h = 16*mm
    rrect(c, x-2*mm, y-hl_h, w+4*mm, hl_h, r=3*mm, fill=C_HIGH_BG, stroke=C_RED, stroke_width=2)
    draw_para(c, "F1 macro = 0.843 | Accuracy = 94.2%", x, y-4*mm, w,
              _sty("hl", 24, C_RED, bold=True, align=TA_CENTER, leading=30))
    y -= hl_h + 3*mm
    if model_img:
        img_w = w + 4*mm
        img_h = img_w * 0.57
        c.drawImage(model_img, x-2*mm, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3*mm
    if conf_img:
        img_w = w * 0.75
        img_h = img_w * 0.90
        c.drawImage(conf_img, x+(w-img_w)/2, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3*mm
    dy = draw_para(c, "<b>Note:</b> OFF recall=61% reflects 12.1% class imbalance. Balanced weights applied but minority detection remains challenging.",
                   x, y, w, S_REF)

# ══════════════════════════════════════════════════════
#  PANEL 5: Physics — label correction story expanded
# ══════════════════════════════════════════════════════

def panel_interpretability(c, shap_img, label_fix_img, actin_img):
    x, y, w = _panel_chrome(c, 1, 1, "5. Interpretability & Physics")
    if shap_img:
        img_w = w + 4*mm
        img_h = img_w * 0.57
        c.drawImage(shap_img, x-2*mm, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3*mm
    # F I explanation
    dy = draw_para(c, "<b>SHAP confirms F I (fluorine radical) as the primary discriminator</b> "
                      "&mdash; consistent with SF6 dissociation chemistry.",
                   x, y, w, S_SMALL)
    y -= dy + 4*mm
    # EXPANDED label correction story
    if label_fix_img:
        img_w = w * 0.85
        img_h = img_w * 0.73
        c.drawImage(label_fix_img, x+(w-img_w)/2, y-img_h, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 3*mm
    # The label correction story (2-3 sentences)
    story = ("<b>Key discovery:</b> Initial gas-flow labels produced only 74% accuracy. "
             "Per-species intensity analysis revealed etch/passivation spectral difference "
             "&lt; 0.35 std &mdash; indistinguishable. Switching to RF-power labels "
             "(plasma ON/OFF) improved to <b>94%</b>, proving <b>data quality, not model "
             "complexity, was the real bottleneck</b>.")
    dy = draw_para(c, story, x, y, w, S_REF)
    y -= dy + 4*mm
    # Boltzmann
    dy = draw_para(c, "<b>Boltzmann T_exc = 13,334 K</b> (6 Ar I lines, 696-772 nm).", x, y, w, S_REF)

# ══════════════════════════════════════════════════════
#  PANEL 6: Conclusions — focused, no metrics clutter
# ══════════════════════════════════════════════════════

def panel_conclusions(c, conclusions_img):
    x, y, w = _panel_chrome(c, 2, 1, "6. Conclusions")

    # Conclusions infographic
    if conclusions_img:
        img_w = w + 4*mm
        img_h = img_w * 0.71
        c.drawImage(conclusions_img, x-2*mm, y-img_h,
                    width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y -= img_h + 4*mm

    # References
    dy = draw_para(c, "<b>References:</b>", x, y, w, _sty("rh", 22, C_NAV, bold=True, leading=28))
    y -= dy + 2*mm
    refs = [
        "[1] Gidon et al. (2019) IEEE Trans. Radiat. Plasma Med. Sci.",
        "[2] Coburn & Chen (1980) J. Appl. Phys.",
        "[3] Lee & Seung (1999) Nature &mdash; NMF",
        "[4] Lundberg & Lee (2017) NeurIPS &mdash; SHAP",
        "[5] Vaswani et al. (2017) NeurIPS &mdash; Transformer",
        "[6] Contreras et al. (2024) Anal. Chem.",
        "[7] BOSCH dataset: Zenodo #17122442",
    ]
    for r in refs:
        dy = draw_para(c, r, x+1*mm, y, w-2*mm, _sty("r", 16, C_SUB, leading=20))
        y -= dy + 1*mm

# ══════════════════════════════════════════════════════
#  TABLE HELPER
# ══════════════════════════════════════════════════════

def _draw_table(c, x, y, w, headers, col_fracs, rows):
    cw = [w*f for f in col_fracs]
    hdr_h = 10*mm
    row_h_ = 9*mm
    c.setFillColor(C_NAV)
    c.rect(x, y-hdr_h, w, hdr_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    cx = x
    for j, hdr in enumerate(headers):
        c.drawCentredString(cx+cw[j]/2, y-hdr_h+3*mm, hdr)
        cx += cw[j]
    for i, row in enumerate(rows):
        ry = y - hdr_h - (i+1)*row_h_
        c.setFillColor(C_LGREY if i%2==0 else C_PANEL)
        c.rect(x, ry, w, row_h_, fill=1, stroke=0)
        c.setFont("Helvetica", 16)
        c.setFillColor(C_TEXT)
        cx = x
        for j, cell in enumerate(row):
            c.drawCentredString(cx+cw[j]/2, ry+2.5*mm, cell)
            cx += cw[j]
    total = hdr_h + len(rows)*row_h_
    c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
    c.rect(x, y-total, w, total, fill=0, stroke=1)
    return total

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    out = ROOT / "poster_oes.pdf"
    print("Generating charts (dpi=300)...")
    spectrum_img = make_spectrum_plot(); print("  [OK] Spectrum")
    species_img = make_species_chart(); print("  [OK] Species")
    model_img = make_model_chart(); print("  [OK] Models")
    conf_img = make_confusion_chart(); print("  [OK] Confusion")
    shap_img = make_shap_chart(); print("  [OK] SHAP")
    label_fix_img = make_label_fix_chart(); print("  [OK] Label fix")
    actin_img = make_actinometry_chart(); print("  [OK] Actinometry")
    kdd_img = make_kdd_chart(); print("  [OK] KDD chart")
    conclusions_img = make_conclusions_chart(); print("  [OK] Conclusions chart")

    # Store KDD image for panel_method
    panel_method._kdd_img = kdd_img

    print("Building poster...")
    pdf = canvas.Canvas(str(out), pagesize=(PAGE_W, PAGE_H))
    pdf.setFillColor(HexColor("#f0f0f0"))
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    draw_banner(pdf)
    panel_intro(pdf, spectrum_img)
    panel_method(pdf)
    panel_species(pdf, species_img)
    panel_classification(pdf, model_img, conf_img)
    panel_interpretability(pdf, shap_img, label_fix_img, actin_img)
    panel_conclusions(pdf, conclusions_img)

    pdf.save()
    print(f"Poster saved: {out} ({out.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    main()
