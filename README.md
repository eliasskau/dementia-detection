# Dementia Detection from Speech

University College Groningen — Year 3 Project  
Speech-based dementia detection on the DementiaBank Pitt Corpus.

**Best result:** SVM + LIWC · RSKF-AUC = 0.789 · 95% CI [0.721, 0.849]  
**Baseline (Hoang et al. 2024):** 0.646

---

## Quickstart with Docker (recommended)

> No Python or conda needed — Docker handles everything.

**1. Install Docker**  
Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop), open it, and wait for the whale icon to appear in your menu bar.

**2. Clone and build**
```bash
git clone https://github.com/eliasskau/dementia-detection
cd dementia-detection
make docker-build        # one-time build, ~5 min
```

**3. Verify it works**
```bash
make docker-test         # runs tests inside container — should show 4 passed
```

**4. Train and evaluate**
```bash
make docker-train        # trains all models, exports best model
make docker-evaluate     # SHAP plots, calibration curve, permutation test
```

Results land in `results/` on your machine (mounted as a volume).

> **Data note:** The Pitt Corpus is access-restricted.  
> Request access at [dementia.talkbank.org](https://dementia.talkbank.org), then place it at `Pitt/`.  
> LIWC-22 is a commercial tool — run it separately and place CSVs at `Pitt/processed/LIWC/`.

---

## Local setup (for development)

```bash
conda create -n dementia-detection python=3.10 -y
conda activate dementia-detection
pip install -r requirements.txt
make train
make evaluate
```

---

## Pipeline

Scripts run in numbered order. Each is self-contained.

| # | Script | What it does |
|---|--------|-------------|
| 01 | `preprocess_transcripts` | `.cha` transcripts → cleaned `.txt` |
| 02 | `extract_participant_audio` | Extract participant-only WAV segments |
| 03 | `extract_linguistic_features` | LCA (lexical) + SCA (syntactic) features |
| 04 | `extract_acoustic_features` | eGeMAPS 88 features via openSMILE |
| 05 | `integrate_liwc` | Merge LIWC-22 CSVs into feature table |
| 06 | `add_response_length` | Add word count + audio duration |
| 07 | `combine_features` | Merge all modalities → `combined/cookie_features.csv` |
| 08 | `train_models` | RSKF (5×5) + bootstrap CI for all models |
| 09 | `hyperparameter_search` | Grid search on LIWC group for all 4 models |
| 10 | `export_best_model` | Train SVM+LIWC on full dataset → `best_model.pkl` |
| 11 | `evaluate_best_model` | SHAP, calibration curve, permutation test |

---

## Results

| Model | Features | RSKF-AUC | 95% CI |
|-------|----------|----------|--------|
| **SVM** | **LIWC — 57 Gini-selected** | **0.789** | **[0.721, 0.849]** |
| LR | LIWC | 0.782 | — |
| RF | LIWC | 0.768 | — |
| XGBoost | LIWC | 0.753 | — |
| Hoang et al. 2024 | linguistic | 0.646 | — |

Evaluation: Repeated Stratified K-Fold (5×5 = 25 folds), subject-aware — no subject's recordings appear in both train and test.

---

## Project Structure

```
dementia-detection/
│
├── dementia_detection/          # source code (Python package)
│   ├── data/                    # transcript + audio processing
│   │   ├── cha_to_txt.py
│   │   ├── audio_extractor.py
│   │   └── text_cleaner.py
│   ├── features/                # feature extraction
│   │   ├── acoustic.py          # eGeMAPS via openSMILE
│   │   └── linguistic.py        # LCA + SCA via NeoSCA
│   └── models/                  # model training + inference
│       ├── train.py             # RSKF evaluation, GiniSelector
│       ├── explain.py           # SHAP explanations
│       └── predict.py           # inference utility
│
├── scripts/                     # numbered pipeline (01–11)
├── config/
│   └── config.py                # all constants (thresholds, CV params, paths)
├── tests/
│   └── test_splits.py
├── notebooks/                   # EDA only
├── results/
│   ├── models/                  # metrics JSONs (tracked) + .pkl (gitignored)
│   └── figures/                 # SHAP, calibration, permutation plots
├── Pitt/                        # corpus data (gitignored, access-restricted)
│
├── Dockerfile
├── Makefile
├── requirements.txt
└── environment.yml
```

---

## Tests

```bash
# Local
conda run -n dementia-detection python -m pytest tests/ -v

# Docker
make docker-test
```


---

## Setup

```bash
conda env create -f environment.yml
conda activate dementia-detection
```

Or with pip:
```bash
pip install -r requirements.txt
```

---

## Set up with Docker
'''bash

'''

## Data

The Pitt Corpus is access-restricted. Request access at https://dementia.talkbank.org/  
Place raw transcripts in `Pitt/raw/` following the existing folder structure.

LIWC-22 is a commercial tool — run it manually on the transcripts and place output CSVs in `Pitt/processed/LIWC/{Control,Dementia}/`.

---

## Pipeline

| Script | Description |
|--------|-------------|
| `01_preprocess_transcripts.py` | Clean `.cha` transcripts → `.txt` |
| `02_extract_participant_audio.py` | Extract participant-only WAV segments |
| `03_extract_linguistic_features.py` | LCA (lexical) + SCA (syntactic) features |
| `04_extract_acoustic_features.py` | eGeMAPS (88 features) via openSMILE |
| `05_integrate_liwc.py` | Merge LIWC-22 CSVs into feature table |
| `06_add_response_length.py` | Add word count + audio duration features |
| `07_combine_features.py` | Merge all modalities → `combined/cookie_features.csv` |
| `08_train_models.py` | RSKF (5×5) + bootstrap CI for all models/feature groups |
| `09_hyperparameter_search.py` | Grid search on LIWC group for all 4 models |
| `10_export_best_model.py` | Train final SVM+LIWC on full dataset → `best_model.pkl` |
| `11_evaluate_best_model.py` | SHAP, calibration curve, permutation test |

Run the full pipeline:
```bash
make pipeline
```

Or individual stages:
```bash
make features   # steps 03–07
make train      # steps 08–10
make evaluate   # step 11
```

---

## Evaluation Protocol

- **Repeated Stratified K-Fold** (5 splits × 5 repeats = 25 folds), subject-aware
- Splits at subject level — no subject's sessions appear in both train and test
- **Bootstrap 95% CI** (1000 resamples) on pooled fold predictions
- Final model fit on full dataset after evaluation

---

## Results (cookie task)

| Model | Feature group | RSKF-AUC | 95% CI |
|-------|--------------|----------|--------|
| **SVM** | **LIWC (57 Gini-selected)** | **0.789** | **[0.721, 0.849]** |
| LR | LIWC | 0.782 | — |
| RF | LIWC | 0.768 | — |
| XGBoost | LIWC | 0.753 | — |
| Hoang et al. 2024 (baseline) | linguistic | 0.646 | — |

---

## Tests

```bash
conda run -n dementia-detection python -m pytest tests/ -v
```

---

## Project Structure

```
scripts/        numbered pipeline scripts (01–11)
src/
  preprocessing/    transcript + audio processing
  feature_extraction/  acoustic + linguistic feature extractors
  models/
    train.py      RSKF evaluation + model fitting
    explain.py    SHAP explanations
    evaluate.py   metrics helpers
configs/
  config.py       all constants (thresholds, CV params, paths)
results/
  models/         trained pipelines (.pkl) + metrics (.json)
  figures/        SHAP, calibration, permutation test plots
notebooks/        EDA only
tests/            unit tests

├───Pitt 
│   ├───raw # raw data downloaded from dementia bank
│   │   ├───dementia
│   │   ├───Pitt-transcript
│   │   └───control
│   ├───intermediate # after cleaning
│   │   ├───cleaned_transcripts
│   │   └───participant_only_audio
│   └───processed # .csv of features extracted
│   │   ├───LIWC
│   │   ├───acoustic
│   │   ├───lexical
│   │   ├───syntactic
│   │   └───combined
├───models  # Stores .pkl
├───notebooks  # Contains experimental .ipynbs
├───project_name
│   ├───data  # For data processing, not storing .csv
│   ├───features
│   └───models  # For model creation, not storing .pkl
├───config
│   └───config.py
├───tests #unit tests
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
```
