# Ralph Agent Instructions (Claude Code)

## Project Context

This project implements **Machine Learning for Plasma OES Analysis** — species classification
and temporal evolution of optical emission spectra from high-voltage electrical discharge systems.

**Primary metric**: `micro_f1` (higher is better, 0–1 scale)
**Primary dataset**: LIBS Benchmark (Figshare, 12 classes, 40002 channels, 200–1000 nm)
**Secondary datasets**: Mesbah Lab CAP (N₂ OES + Trot/Tvib), BOSCH Plasma Etching (25 Hz time-series)

---

## Your Task (ONE story per iteration)

1. Read `scripts/ralph/prd.json`
2. Read `scripts/ralph/progress.txt` (check Codebase Patterns first)
3. Ensure you're on branch `ralph/plasma-oes` (create from HEAD if missing)
4. Pick the **lowest `priority` number** story where `passes: false`
5. Implement that ONE story (keep scope tight; do not touch unrelated code)
6. Run the **Quality Check** matching the story phase (see below)
7. If guardrail PASSES: commit `feat: [ID] - [Title]`, set story `passes: true` in prd.json
8. Append learnings to `scripts/ralph/progress.txt`
9. Output `<promise>COMPLETE</promise>` when ALL stories pass

---

## Quality Checks by Phase

### Phase 1 — Infrastructure stories (OES-001 to OES-009)

These stories do **NOT** write `results/metrics.json` (guardrail check is skipped by ralph.sh
when the file is absent). Verify acceptance criteria manually:

```bash
# Verify imports and basic functionality — adapt command per story
python -c "<story-specific validation command from acceptanceCriteria>"
```

Story passes if the verification command exits 0 and all acceptance criteria are met.
**Do not write results/metrics.json in these stories.**

---

### Phase 2 — Classification stories (OES-010, OES-011, OES-012, OES-013, OES-018, OES-021, OES-022)

Run the classification pipeline and emit micro_f1 to metrics.json:

```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe main.py \
  --task classify \
  --train data/libs_benchmark/train.h5 \
  --model svm \
  --cv 5 \
  --seed 42 \
  --metrics_out results/metrics.json
```

Replace `--model svm` with the model introduced in that story (e.g., `--model cnn` for OES-011).

**Guardrail behaviour (updated in OES-003)**:
- `primary_metric.name == "micro_f1"` → PASS if `current >= best - tol` (maximize)
- `primary_metric.name == "RMSE_mean"` → PASS if `current <= best + tol` (minimize, legacy)
- `primary_metric.name == "setup_ok"` → always PASS

Run guardrail:
```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe -m src.guardrail results/metrics.json --tol 0.02
```

---

### Phase 3 — Regression story (OES-014)

Run CAP temperature regression (writes to `results/metrics_cap.json`):
```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe main.py \
  --task regress \
  --train data/mesbah_cap/dat_train.csv \
  --target T_rot \
  --model ann --per_target --cv 5 --seed 42 \
  --metrics_out results/metrics_cap.json
```

Then also re-run classification to keep `results/metrics.json` current for ralph's guardrail:
```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe main.py \
  --task classify \
  --train data/libs_benchmark/train.h5 \
  --model svm --cv 3 --seed 42 \
  --metrics_out results/metrics.json
```

---

### Phase 4 — Temporal stories (OES-015, OES-016, OES-017)

These stories do **NOT** write `results/metrics.json` (guardrail skipped). Verify:
```bash
python scripts/plot_temporal_pca.py --data data/bosch_oes/     # OES-015
python scripts/plot_clusters.py --data data/bosch_oes/ --k 4   # OES-016
python scripts/train_temporal.py --data data/bosch_oes/ --epochs 50  # OES-017
```

---

### Phase 5 — Documentation stories (OES-019, OES-020, OES-023)

OES-019: Execute all notebooks:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_classification.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_temporal_analysis.ipynb
```

OES-020: Run pytest:
```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe -m pytest tests/ -v
```

OES-023: Generate report:
```bash
/d/Develop/Anaconda/envs/pytorch_env/python.exe scripts/generate_report.py
```

These do **NOT** write `results/metrics.json`.

---

## Key Implementation Notes

### Data Paths
- LIBS Benchmark: `data/libs_benchmark/train.h5`, `test.h5`, `test_labels.csv`
- Mesbah Lab: `data/mesbah_cap/dat_train.csv`, `dat_test.csv`
- BOSCH: `data/bosch_oes/*.nc` (NetCDF files)

### Download Instructions (OES-001)
LIBS Benchmark from Figshare (DOI: 10.6084/m9.figshare.c.4768790):
- Direct HDF5 download links are in the Figshare collection page
- Alternatively use: `pip install requests` and write a download script

Mesbah Lab (OES-002):
```bash
git clone https://github.com/Mesbah-Lab-UCB/Machine-Learning-for-Plasma-Diagnostics data/mesbah_cap_repo
cp data/mesbah_cap_repo/dat_*.csv data/mesbah_cap/
```

BOSCH (OES-015):
- Download from https://zenodo.org/records/17122442 — use the Zenodo API or browser download

### Existing Code to Reuse
- `src/preprocessing.py`: Preprocessor (ALS, SNV, SavGol) — extend, don't rewrite
- `src/features.py`: NIST window selection, PCA — extend with plasma lines
- `src/models/deep_learning.py`: Conv1DRegressor → adapt to Conv1DClassifier
- `src/evaluation.py`: evaluate_model → extend with evaluate_classifier
- `src/data_loader.py`: SpectralDataset → extend with load_libs_benchmark, load_mesbah_cap
- Optuna HPO framework in `src/optimization.py` — reuse for classifier HPO

### Python Environment
```
/d/Develop/Anaconda/envs/pytorch_env/python.exe
```

### Commit Format
```
feat: OES-XXX - Short title
```

### metrics.json Format (classification mode)
```json
{
  "primary_metric": {"name": "micro_f1", "value": 0.92},
  "model": "cnn",
  "task": "classify",
  "cv_folds": 5,
  "seed": 42,
  "timestamp": "2026-...",
  "git_sha": "...",
  "metrics": {
    "micro_f1": 0.92,
    "macro_f1": 0.89,
    "accuracy": 0.93,
    "ece": 0.04,
    "per_class_f1": {"class_0": 0.95, "class_1": 0.88, ...}
  }
}
```
