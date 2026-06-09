import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config import RSKF_SPLITS, RSKF_REPEATS, RANDOM_STATE
from sklearn.model_selection import RepeatedStratifiedKFold


def _make_df(n_control: int, n_dementia: int, sessions_per_subject: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(n_control):
        for _ in range(sessions_per_subject):
            rows.append({"subject_id": f"C{i:03d}", "label": "Control",  "feat": float(i)})
    for i in range(n_dementia):
        for _ in range(sessions_per_subject):
            rows.append({"subject_id": f"D{i:03d}", "label": "Dementia", "feat": float(i)})
    return pd.DataFrame(rows)


def _subject_splits(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(df["label"].values)
    subject_ids = df["subject_id"].values
    unique_subj = np.unique(subject_ids)
    subj_label = np.array([
        int(y_enc[subject_ids == s].mean() >= 0.5) for s in unique_subj
    ])
    rskf = RepeatedStratifiedKFold(
        n_splits=RSKF_SPLITS, n_repeats=RSKF_REPEATS, random_state=RANDOM_STATE
    )
    return list(rskf.split(unique_subj, subj_label)), unique_subj


def test_no_subject_overlap():
    df = _make_df(60, 60, sessions_per_subject=2)
    splits, unique_subj = _subject_splits(df)
    for train_idx, test_idx in splits:
        train_set = set(unique_subj[train_idx])
        test_set  = set(unique_subj[test_idx])
        assert train_set.isdisjoint(test_set), "Subject appears in both train and test"


def test_all_subjects_covered():
    df = _make_df(60, 60)
    splits, unique_subj = _subject_splits(df)
    seen = set()
    for _, test_idx in splits:
        seen.update(unique_subj[test_idx])
    assert seen == set(unique_subj), "Not all subjects appear in a test fold"


def test_stratification():
    df = _make_df(60, 60)
    splits, unique_subj = _subject_splits(df)
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(df.drop_duplicates("subject_id")["label"].values)
    for _, test_idx in splits:
        labels_in_fold = y_enc[test_idx]
        assert len(np.unique(labels_in_fold)) == 2, "Test fold has only one class"


def test_expected_fold_count():
    df = _make_df(60, 60)
    splits, _ = _subject_splits(df)
    assert len(splits) == RSKF_SPLITS * RSKF_REPEATS
