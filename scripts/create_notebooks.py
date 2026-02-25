"""Generate OES-019 tutorial notebooks for Plasma OES Analysis.

Creates three notebooks:
  notebooks/01_preprocessing.ipynb
  notebooks/02_classification.ipynb
  notebooks/03_temporal_analysis.ipynb

Each notebook is designed to execute top-to-bottom without errors via:
  jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb
"""

import nbformat
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src.strip())


def md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src.strip())


def save(nb: nbformat.NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Notebook 01 — Preprocessing Tutorial
# ---------------------------------------------------------------------------

NB01_CELLS = [
    md("""# 01 — Plasma OES Preprocessing Pipeline

**LIBS Benchmark dataset** — 12-class, 40002-channel OES spectra (200–1000 nm).

This notebook demonstrates:
1. Loading a small LIBS Benchmark subset
2. Applying the `Preprocessor` pipeline: cosmic-ray removal → ALS baseline → SavGol smoothing → SNV normalisation
3. Visualising raw vs preprocessed spectra
4. Detecting emission peaks with `detect_peaks`
"""),

    code("""\
import os
import sys
import warnings
from pathlib import Path

# ------------------------------------------------------------------
# Ensure we run from the project root (works whether notebook is
# launched from project root or from notebooks/ directory).
# ------------------------------------------------------------------
nb_dir = Path.cwd()
if nb_dir.name == "notebooks":
    os.chdir(nb_dir.parent)
sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # headless backend — safe for nbconvert
import matplotlib.pyplot as plt
import numpy as np

print("Working directory:", Path.cwd())
"""),

    md("## 1 · Load LIBS Benchmark (small subset)"),

    code("""\
from src.data_loader import load_libs_benchmark

# max_spectra_per_class=2 → ~192 samples total (fast load)
X, y, wl = load_libs_benchmark("data/libs_benchmark/", max_spectra_per_class=2)
print(f"Loaded : {X.shape[0]} spectra × {X.shape[1]} channels")
print(f"Wavelength range : {wl[0]:.1f} – {wl[-1]:.1f} nm")
print(f"Classes : {sorted(set(y.tolist()))}")
"""),

    md("## 2 · Select one representative spectrum per class"),

    code("""\
# Pick the first spectrum from each of the 12 classes
n_classes = len(set(y.tolist()))
class_indices = [int(np.where(y == c)[0][0]) for c in range(n_classes)]
X_vis = X[class_indices]
print(f"Reference spectra : {X_vis.shape} (one per class)")
"""),

    md("## 3 · Apply the Preprocessor pipeline"),

    code("""\
from src.preprocessing import Preprocessor

# Full pipeline: cosmic-ray removal → ALS baseline → SavGol → SNV
pp = Preprocessor(
    baseline="als",
    normalize="snv",
    denoise="savgol",
    cosmic_ray=True,
)
pp.fit(X_vis)
X_pp = pp.transform(X_vis)
print(f"Preprocessed: shape={X_pp.shape}, dtype={X_pp.dtype}")
print(f"Intensity range after SNV: [{X_pp.min():.2f}, {X_pp.max():.2f}]")
"""),

    md("## 4 · Detect emission peaks (class 0)"),

    code("""\
from src.features import detect_peaks

# detect_peaks returns a DataFrame with columns: wavelength_nm, intensity, prominence, fwhm_nm
peaks_df = detect_peaks(X_pp[0], wl, min_prominence=0.1)
peaks_wl = peaks_df["wavelength_nm"].values
peaks_int = peaks_df["intensity"].values
print(f"Detected {len(peaks_wl)} peaks in class-0 spectrum")
if len(peaks_wl) > 0:
    top5 = peaks_df.nlargest(5, "intensity")
    print("Top 5 peaks (nm):", top5["wavelength_nm"].round(1).tolist())
"""),

    md("## 5 · Visualise raw vs preprocessed spectra + peak annotations"),

    code("""\
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
class_labels = [f"class_{c:02d}" for c in range(n_classes)]
WL_STEP = 20   # subsample for display speed

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ---- top-left: raw spectra (all classes) ----
ax = axes[0, 0]
for i in range(n_classes):
    ax.plot(wl[::WL_STEP], X_vis[i, ::WL_STEP],
            color=colors[i], alpha=0.75, lw=0.8, label=class_labels[i])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Raw Spectra — 12 classes")
ax.legend(fontsize=6, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)

# ---- top-right: preprocessed spectra (all classes) ----
ax = axes[0, 1]
for i in range(n_classes):
    ax.plot(wl[::WL_STEP], X_pp[i, ::WL_STEP],
            color=colors[i], alpha=0.75, lw=0.8, label=class_labels[i])
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("SNV Intensity")
ax.set_title("Preprocessed Spectra (ALS + SavGol + SNV)")
ax.legend(fontsize=6, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)

# ---- bottom-left: raw class-0 ----
ax = axes[1, 0]
ax.plot(wl[::10], X_vis[0, ::10], color="steelblue", lw=0.8, alpha=0.9)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Class 0 — Raw Spectrum")
ax.grid(True, alpha=0.3)

# ---- bottom-right: preprocessed class-0 + peak annotations ----
ax = axes[1, 1]
ax.plot(wl, X_pp[0], color="tomato", lw=0.8, alpha=0.9, label="Preprocessed")
if len(peaks_wl) > 0:
    # Draw vertical lines for detected peaks
    for pw in peaks_wl[:30]:
        ax.axvline(pw, color="green", alpha=0.3, lw=0.7)
    # Annotate top-5 peaks
    top5 = np.argsort(peaks_int)[-5:][::-1]
    for idx in top5:
        ax.annotate(
            f"{peaks_wl[idx]:.1f}",
            xy=(peaks_wl[idx], peaks_int[idx]),
            xytext=(peaks_wl[idx] + 8, peaks_int[idx] + 0.06),
            fontsize=7,
            color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.7),
        )
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("SNV Intensity")
ax.set_title("Class 0 — Preprocessed + Peak Annotations")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(
    "Plasma OES Preprocessing Pipeline — LIBS Benchmark",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

out_path = Path("outputs/notebook_01_preprocessing.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure → {out_path}")
"""),

    md("""## Summary

| Step | Method | Key parameter |
|------|--------|---------------|
| Cosmic-ray removal | Z-score median filter | threshold = 5 σ |
| Baseline correction | Asymmetric Least Squares | λ = 1×10⁶, p = 0.01 |
| Smoothing | Savitzky–Golay | window = 11, poly = 3 |
| Normalisation | Standard Normal Variate | — |
| Peak detection | `scipy.signal.find_peaks` | prominence ≥ 0.10 |
"""),
]

# ---------------------------------------------------------------------------
# Notebook 02 — Classification Tutorial
# ---------------------------------------------------------------------------

NB02_CELLS = [
    md("""# 02 — Plasma OES Classification

**LIBS Benchmark** — SVM + CNN species classification + SHAP wavelength importance.

This notebook demonstrates:
1. Loading and preprocessing a small LIBS Benchmark subset
2. Training an SVM classifier with 5-fold cross-validation
3. Plotting the confusion matrix
4. Training a lightweight CNN on PCA features
5. Computing SHAP wavelength importance via LinearSVC + PCA back-projection
"""),

    code("""\
import os
import sys
import warnings
from pathlib import Path

nb_dir = Path.cwd()
if nb_dir.name == "notebooks":
    os.chdir(nb_dir.parent)
sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

print("Working directory:", Path.cwd())
"""),

    md("## 1 · Load and preprocess LIBS Benchmark"),

    code("""\
from src.data_loader import load_libs_benchmark
from src.preprocessing import Preprocessor
from sklearn.preprocessing import StandardScaler

# Load small subset (max 2 per sample group → ~192 spectra total)
X, y, wl = load_libs_benchmark("data/libs_benchmark/", max_spectra_per_class=2)
print(f"Dataset : {X.shape[0]} spectra × {X.shape[1]} channels")

# Lightweight preprocessing (SNV + SavGol, skip ALS for speed in CV)
pp = Preprocessor(baseline="none", normalize="snv", denoise="savgol", cosmic_ray=False)
X_pp = pp.fit_transform(X)
X_sc = StandardScaler().fit_transform(X_pp)
print(f"Preprocessed : {X_sc.shape}")
"""),

    md("## 2 · PCA dimensionality reduction"),

    code("""\
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_sc)
ev_ratio = pca.explained_variance_ratio_.sum()
print(f"PCA(50) : {X_pca.shape}, explained variance = {ev_ratio:.1%}")
"""),

    md("## 3 · SVM classification (5-fold cross-validation)"),

    code("""\
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=42)
y_pred_svm = cross_val_predict(svm, X_pca, y, cv=cv)

micro_f1_svm = f1_score(y, y_pred_svm, average="micro")
macro_f1_svm = f1_score(y, y_pred_svm, average="macro")
print(f"SVM  micro-F1 : {micro_f1_svm:.4f}")
print(f"SVM  macro-F1 : {macro_f1_svm:.4f}")
"""),

    md("## 4 · Confusion matrix"),

    code("""\
from sklearn.metrics import confusion_matrix

cm_mat = confusion_matrix(y, y_pred_svm)
n_classes = cm_mat.shape[0]
class_names = [f"C{c:02d}" for c in range(n_classes)]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm_mat, cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"SVM Confusion Matrix  (micro-F1 = {micro_f1_svm:.3f})")
# Annotate cells
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(
            j, i, str(cm_mat[i, j]),
            ha="center", va="center", fontsize=7,
            color="white" if cm_mat[i, j] > cm_mat.max() * 0.5 else "black",
        )
plt.tight_layout()

out_path = Path("outputs/notebook_02_confusion_matrix.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
"""),

    md("## 5 · CNN classification (15 epochs demonstration)"),

    code("""\
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as f1
from src.models.deep_learning import Conv1DClassifier, train_classifier, _get_safe_device

device = _get_safe_device()
X_tr, X_val, y_tr, y_val = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_tr.shape}  Val: {X_val.shape}")

cnn = Conv1DClassifier(n_classes=n_classes, n_filters=32, kernel_size=3, dropout=0.2, lr=1e-3)
trained_cnn = train_classifier(
    cnn, X_tr, y_tr, X_val, y_val, epochs=15, batch_size=32, device=device
)

# Evaluate on validation set
trained_cnn.eval()
with torch.no_grad():
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    logits = trained_cnn(X_val_t).cpu().numpy()
y_pred_cnn = logits.argmax(axis=1)

cnn_f1 = f1(y_val, y_pred_cnn, average="micro")
print(f"CNN val micro-F1 (15 epochs) : {cnn_f1:.4f}")
"""),

    md("""## 6 · SHAP wavelength importance

We use **LinearSVC** (fast, exact SHAP) trained on PCA(50) features,
then project the 50-dimensional SHAP vector back to the 40 002-channel wavelength axis
via the PCA loading matrix.
"""),

    code("""\
import shap
from scipy.ndimage import uniform_filter1d
from sklearn.svm import LinearSVC

# Train LinearSVC on full dataset (for SHAP analysis only)
lsvc = LinearSVC(C=1.0, max_iter=3000, random_state=42)
lsvc.fit(X_pca, y)

# SHAP values in PCA space — explains 30 samples
explainer = shap.LinearExplainer(lsvc, X_pca, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_pca[:30])   # (30, 50, n_classes)

# Aggregate: mean |SHAP| over samples and classes → (50,) feature importance
shap_arr = np.array(shap_values)               # (30, 50, n_classes)
shap_pca = np.abs(shap_arr).mean(axis=0).mean(axis=1)   # (50,)

# Project PCA SHAP back to wavelength space: (40002, 50) @ (50,) = (40002,)
shap_wl = np.abs(pca.components_.T @ shap_pca)   # (40002,)

# Smooth over ~200 channels (~4 nm) for readability
shap_smooth = uniform_filter1d(shap_wl, size=200)
shap_norm   = shap_smooth / shap_smooth.max()

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(wl, shap_norm, alpha=0.35, color="darkorange", label="SHAP importance")
ax.plot(wl, shap_norm, color="darkorange", lw=0.9)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Relative SHAP importance")
ax.set_title("SHAP Feature Importance — Projected to Wavelength Space (LinearSVC on PCA features)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

out_path = Path("outputs/notebook_02_shap_overlay.png")
fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
"""),

    md("""## Summary

| Model | micro-F1 (5-fold CV) |
|-------|----------------------|
| SVM (RBF, C=10) | See above |
| CNN (15 epochs, demonstration) | See above (val) |

- SHAP analysis identifies which spectral regions drive class separation.
- Both SVM and CNN operate on PCA(50) features of SNV-normalised spectra.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 03 — Temporal Analysis Tutorial
# ---------------------------------------------------------------------------

NB03_CELLS = [
    md("""# 03 — Plasma OES Temporal Analysis

**BOSCH Plasma Etching dataset** — OES time-series at ~22.8 Hz, 3648 channels (185–884 nm).

This notebook demonstrates:
1. Loading a BOSCH NetCDF day file (first 3000 time steps)
2. Computing a PCA temporal embedding (dimensionality reduction over time)
3. Plotting the 2-D trajectory in PCA space
4. Clustering discharge phases with DTW K-means (k = 4)
5. Visualising cluster membership over time
"""),

    code("""\
import os
import sys
import warnings
from pathlib import Path

nb_dir = Path.cwd()
if nb_dir.name == "notebooks":
    os.chdir(nb_dir.parent)
sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

print("Working directory:", Path.cwd())
"""),

    md("## 1 · Load BOSCH OES data (first 3 000 time steps)"),

    code("""\
from src.data_loader import load_bosch_oes

# Load Wafer_01 from Day_2024_07_02.nc
data = load_bosch_oes("data/bosch_oes/", day_file="Day_2024_07_02.nc")

# Subsample to first 3000 steps for demonstration speed
T_MAX = 3000
spectra    = data["spectra"][:T_MAX]      # (T_MAX, 3648) float32
timestamps = data["timestamps"][:T_MAX]    # (T_MAX,) seconds
wl         = data["wavelengths"]           # (3648,) nm

t_rel = timestamps - timestamps[0]   # relative seconds from start

print(f"Spectra : {spectra.shape}  (float32)")
print(f"Channels: {len(wl)}  ({wl[0]:.1f} – {wl[-1]:.1f} nm)")
print(f"Duration: {t_rel[-1]:.1f} s  @ ~{1/np.diff(timestamps).mean():.1f} Hz")
"""),

    md("## 2 · PCA temporal embedding"),

    code("""\
from src.temporal import compute_temporal_embedding

# StandardScaler → PCA(10): reduces (T, 3648) to (T, 10)
embedding, pca = compute_temporal_embedding(spectra, n_components=10)

print(f"Embedding : {embedding.shape}")
print(f"PC1+PC2 explained variance : {pca.explained_variance_ratio_[:2].sum():.1%}")
print(f"Top-3 PCs : {pca.explained_variance_ratio_[:3].round(3)}")
"""),

    md("## 3 · PCA trajectory plots"),

    code("""\
STEP = 10    # plot every 10th point for visual clarity

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- PC1 vs PC2 scatter coloured by time ----
ax = axes[0]
sc = ax.scatter(
    embedding[::STEP, 0], embedding[::STEP, 1],
    c=t_rel[::STEP], cmap="viridis", s=5, alpha=0.7,
)
plt.colorbar(sc, ax=ax, label="Time (s)")
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA Temporal Trajectory (PC1 vs PC2)")
ax.grid(True, alpha=0.3)

# ---- PC1/PC2/PC3 over time ----
ax2 = axes[1]
for pc_idx, color, label in zip(
    [0, 1, 2],
    ["steelblue", "tomato", "forestgreen"],
    ["PC1", "PC2", "PC3"],
):
    if pc_idx < embedding.shape[1]:
        ax2.plot(t_rel[::STEP], embedding[::STEP, pc_idx],
                 color=color, lw=0.7, label=label, alpha=0.85)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("PCA Score")
ax2.set_title("PCA Scores over Time")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "BOSCH Plasma Etching OES — PCA Temporal Embedding  (3 000 time steps)",
    fontsize=11,
)
plt.tight_layout()

out_path = Path("outputs/notebook_03_pca_trajectory.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
"""),

    md("## 4 · Discharge phase clustering (DTW K-means, k = 4)"),

    code("""\
from src.temporal import cluster_discharge_phases

# Euclidean K-means on PCA embedding (fast; DTW with seq_len=1 is equivalent)
labels, centroids, inertia = cluster_discharge_phases(
    embedding, k=4, metric="euclidean", seed=42
)

print(f"K-means (k=4) inertia : {inertia:.2f}")
cluster_sizes = {i: int((labels == i).sum()) for i in range(4)}
print("Cluster sizes :", cluster_sizes)
"""),

    md("## 5 · Visualise cluster membership over time"),

    code("""\
COLORS = ["steelblue", "tomato", "forestgreen", "darkorange"]
PHASE_NAMES = [f"Phase {i+1}" for i in range(4)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ---- PCA scatter coloured by cluster ----
ax = axes[0]
for k in range(4):
    mask = labels[::STEP] == k
    ax.scatter(
        embedding[::STEP][mask, 0],
        embedding[::STEP][mask, 1],
        s=5, alpha=0.7, color=COLORS[k], label=PHASE_NAMES[k],
    )
# Mark centroids
for k in range(4):
    ax.scatter(
        centroids[k, 0], centroids[k, 1],
        s=180, marker="*", color=COLORS[k],
        edgecolors="black", lw=0.8, zorder=6,
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("DTW Cluster Membership in PCA Space")
ax.legend(markerscale=1.5, fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Cluster membership over time (stacked bars) ----
ax2 = axes[1]
for k in range(4):
    member = (labels == k).astype(float)
    ax2.fill_between(t_rel, k, k + member,
                     color=COLORS[k], alpha=0.75, label=PHASE_NAMES[k], step="mid")

ax2.set_xlabel("Time (s)")
ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
ax2.set_yticklabels(PHASE_NAMES, fontsize=9)
ax2.set_title("Discharge Phase Membership over Time")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.2)

plt.suptitle(
    "BOSCH Plasma Etching — Discharge Phase Identification (k=4 clusters)",
    fontsize=11,
)
plt.tight_layout()

out_path = Path("outputs/notebook_03_clusters.png")
fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
"""),

    md("""## Summary

| Step | Method | Output |
|------|--------|--------|
| Data loading | `load_bosch_oes` (NetCDF) | (3 000, 3648) float32 |
| Temporal embedding | StandardScaler → PCA(10) | (3 000, 10) |
| Phase clustering | Euclidean K-means (k=4) | phase labels (3 000,) |

The PCA trajectory captures major process transitions (ignition, steady-state, extinction).
K-means clustering with k = 4 identifies four recurring discharge phases automatically.
"""),
]


# ---------------------------------------------------------------------------
# Build and write notebooks
# ---------------------------------------------------------------------------

def build_nb(cells):
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (pytorch_env)",
            "language": "python",
            "name": "pytorch_env",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    }
    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb_dir = Path("notebooks")

    save(build_nb(NB01_CELLS), nb_dir / "01_preprocessing.ipynb")
    save(build_nb(NB02_CELLS), nb_dir / "02_classification.ipynb")
    save(build_nb(NB03_CELLS), nb_dir / "03_temporal_analysis.ipynb")

    print("Done — 3 notebooks written.")
