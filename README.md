# Dementia Detection from Speech

University College Groningen — Year 3 Project  
Speech-based dementia detection on the DementiaBank Pitt Corpus.

**Best result:** SVM + LIWC · RSKF-AUC = 0.789 · 95% CI [0.721, 0.849]  
**Baseline (Hoang et al. 2024):** 0.646

---

## Quickstart (Docker)

> No Python or conda needed.

```bash
git clone https://github.com/eliasskau/dementia-detection
cd dementia-detection
make docker-build        # one-time build (~5 min)
make docker-test         # 4 passed ✓
make docker-train        # train + export best model
make docker-evaluate     # SHAP, calibration curve, permutation test
```

Results land in `results/` on your machine (mounted as a volume).

---

## Local setup

```bash
conda create -n dementia-detection python=3.10 -y
conda activate dementia-detection
pip install -r requirements.txt
make train
make evaluate
```

---

## Running the pipeline

Everything goes through a single entrypoint:

```bash
python pipeline.py <command> [options]
```

| Command | What it does |
|---------|-------------|
| `preprocess` | `.cha` → cleaned transcripts + participant-only WAV |
| `features` | Extract eGeMAPS, LCA, SCA, LIWC, response-length + combine |
| `train` | RSKF 5×5 cross-validation for all models + feature groups |
| `search` | Hyperparameter grid search (LIWC group, all 4 classifiers) |
| `export` | Train SVM+LIWC on full dataset → `results/models/best_model.pkl` |
| `evaluate` | SHAP bar chart, calibration curve, permutation test |
| `all` | Run everything end-to-end |

```bash
python pipeline.py train --task cookie --model svm
python pipeline.py features --lca-only
python pipeline.py train --help
```

`make train` / `make evaluate` / `make pipeline` are thin wrappers around these.

---

## Data

The Pitt Corpus is access-restricted — request access at [dementia.talkbank.org](https://dementia.talkbank.org) and place the raw data at `Pitt/raw/`.

LIWC-22 is a commercial tool. Run it on `Pitt/intermediate/cleaned_transcripts/` and place the output CSVs at:
```
Pitt/processed/LIWC/{Control,Dementia}/LIWC-22 Results - {task} - LIWC Analysis.csv
```

---

## Results

| Model | Features | RSKF-AUC | 95% CI |
|-------|----------|----------|--------|
| **SVM** | **LIWC (57 Gini-selected)** | **0.789** | **[0.721, 0.849]** |
| LR | LIWC | 0.782 | — |
| RF | LIWC | 0.768 | — |
| XGBoost | LIWC | 0.753 | — |
| Hoang et al. 2024 | linguistic | 0.646 | — |

Evaluation: Repeated Stratified K-Fold (5 splits × 5 repeats = 25 folds). Splits are subject-aware — no subject's recordings appear in both train and test within a fold.

---

## Project structure

```
dementia-detection/
├── dementia_detection/       ← importable Python package
│   ├── data/                 ← transcript/audio processing + feature assembly
│   │   ├── cha_to_txt.py
│   │   ├── audio_extractor.py
│   │   ├── text_cleaner.py
│   │   ├── combine.py
│   │   ├── liwc.py
│   │   └── response_length.py
│   ├── features/             ← feature extractors
│   │   ├── acoustic.py       # eGeMAPS (88 features) via openSMILE
│   │   └── linguistic.py     # LCA + SCA via NeoSCA
│   └── models/               ← training, inference, explanation
│       ├── train.py          # RSKF, GiniSelector, bootstrap CI
│       ├── predict.py        # inference + DementiaRiskPredictor
│       ├── tune.py           # hyperparameter search
│       └── explain.py        # SHAP, calibration, permutation test
│
├── pipeline.py               ← single CLI entrypoint
├── config/config.py          ← all constants (paths, CV params, thresholds)
├── tests/
├── notebooks/                ← EDA only
├── results/
│   ├── models/               ← metrics JSONs (tracked) · .pkl (gitignored)
│   └── figures/
├── Pitt/                     ← corpus data (gitignored, access-restricted)
├── Dockerfile · Makefile
└── requirements.txt
```

The `dementia_detection/` package contains all reusable logic with no side effects. `pipeline.py` is the only entrypoint — it parses arguments and calls into the package. Nothing else runs at the top level.

---

## Tests

```bash
conda run -n dementia-detection python -m pytest tests/ -v
# or
make docker-test
```
