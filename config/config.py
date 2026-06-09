from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DIR        = ROOT / "Pitt" / "raw"
INTERMEDIATE_DIR = ROOT / "Pitt" / "intermediate"
TRANSCRIPT_DIR = ROOT / "Pitt" / "intermediate" / "cleaned_transcripts"
AUDIO_DIR      = ROOT / "Pitt" / "intermediate" / "participant_only_audio"
PROCESSED_DIR  = ROOT / "Pitt" / "processed"
LIWC_DIR       = ROOT / "Pitt" / "processed" / "LIWC"
COMBINED_DIR   = ROOT / "Pitt" / "processed" / "combined"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODELS_DIR   = ROOT / "results" / "models"
FIGURES_DIR  = ROOT / "results" / "figures"

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
TASKS      = ("cookie", "fluency", "recall", "sentence")
TASK       = "cookie"          # default task for model training / evaluation
CATEGORIES = ("acoustic", "syntactic", "lexical")  # feature merge order
LIWC_META  = frozenset({"Filename", "Segment"})     # non-feature LIWC columns

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

# Repeated stratified k-fold
RSKF_SPLITS   = 5
RSKF_REPEATS  = 5

# Bootstrap confidence interval
BOOTSTRAP_N   = 1000
BOOTSTRAP_CI  = 0.95

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Gini feature selection thresholds (per modality)
GINI_THRESHOLD = {
    "acoustic": 0.012,
    "liwc":     0.008,
}

RANDOM_STATE = 42
