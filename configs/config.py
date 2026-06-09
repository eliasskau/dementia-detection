from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
COMBINED_DIR = ROOT / "Pitt" / "processed" / "combined"
MODELS_DIR   = ROOT / "results" / "models"
FIGURES_DIR  = ROOT / "results" / "figures"

TASK = "cookie"

# Repeated stratified k-fold evaluation
RSKF_SPLITS   = 5
RSKF_REPEATS  = 5

# Bootstrap confidence interval
BOOTSTRAP_N   = 1000
BOOTSTRAP_CI  = 0.95

# Gini feature selection thresholds (per modality)
GINI_THRESHOLD = {
    "acoustic": 0.012,
    "liwc":     0.008,
}

RANDOM_STATE = 42
