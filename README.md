# Dementia Detection from Speech

Speech-based dementia detection on the DementiaBank Pitt Corpus using LIWC-22, acoustic (eGeMAPS), lexical (LCA), and syntactic (SCA) features.

**Best result:** SVM + LIWC · RSKF-AUC = 0.789 · 95% CI [0.721, 0.849] · vs. Hoang et al. 2024 baseline 0.646

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
```
