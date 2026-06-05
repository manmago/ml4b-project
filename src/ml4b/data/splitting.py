"""Subject-based train/validation/test splitting and class-imbalance correction for ML4B.

This module covers two responsibilities:

1. **Subject-based splitting** (:func:`subject_based_split`) — partitions the
   feature DataFrame into disjoint train / val / test sets at the *subject*
   level to prevent data leakage.  See DECISIONS.md.

2. **Majority-class undersampling** (:func:`undersample_majority_class`) —
   reduces the dominance of the ``rest`` class, which accounts for ~89% of
   windows after windowing.  Without correction a model predicting "rest"
   for every window would achieve ~89% accuracy — useless in practice.
   See DECISIONS.md.

Only the *training* set is undersampled.  Val and test sets intentionally
keep the original distribution so evaluation metrics reflect real-world
conditions.
"""

import numpy as np
import pandas as pd


def subject_based_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation and test sets by subject.

    Uses subject-based splitting instead of random row splitting to prevent
    data leakage — the same subject's data must not appear in both train and test.
    This gives a more realistic estimate of how the model performs on new people.

    Args:
        df: Feature DataFrame from extract_features() with subject_id column.
        test_size: Fraction of subjects for test set. Default 0.2 = 20%.
        val_size: Fraction of subjects for validation set. Default 0.1 = 10%.
        random_state: Random seed for reproducibility. Default 42.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.
    """
    # Validate inputs early — a silent over-budget split would corrupt every
    # downstream evaluation metric.
    if "subject_id" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'subject_id' column.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size + val_size must be < 1.0, got {test_size + val_size}"
        )

    # Get the unique subject ids and shuffle deterministically with the given
    # seed. The split is on subjects, NOT on rows — this is the entire point
    # of subject-based splitting: a subject is assigned in full to exactly
    # one of train / val / test, never spread across them.
    subjects = np.array(sorted(df["subject_id"].unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_test = max(1, int(round(n_subjects * test_size)))
    # Take val from the remainder after carving out test, so val_size is
    # interpreted as a fraction of the original total (consistent with how
    # users typically think about a 70/10/20 split).
    n_val = max(1, int(round(n_subjects * val_size))) if val_size > 0 else 0

    # Guard against pathological tiny datasets where n_test + n_val could
    # leave no subjects for training.
    if n_test + n_val >= n_subjects:
        raise ValueError(
            f"Not enough subjects ({n_subjects}) to honour the requested split "
            f"(test={n_test}, val={n_val})."
        )

    # Assign subjects to splits by slicing the shuffled subject array. Slicing
    # is disjoint by construction, so the no-overlap invariant is guaranteed.
    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    # Select rows by subject membership. Using set membership keeps this O(n)
    # in the number of rows.
    train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df


def undersample_majority_class(
    df: pd.DataFrame,
    label_column: str = "exercise_name",
    majority_class: str = "rest",
    multiplier: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Undersample the majority class to reduce class imbalance.

    The rest/non-exercise class dominates the dataset (88.8% of windows)
    because participants spend more time resting than exercising.
    Without correction, a model could achieve ~89% accuracy by always
    predicting 'rest' — making it useless in practice.

    Strategy: cap the majority class at multiplier × size of the
    largest minority class. This preserves enough rest examples for
    the model to learn the rest pattern while reducing dominance.
    See DECISIONS.md for full rationale.

    Args:
        df: Feature DataFrame with label_column.
        label_column: Name of the class label column. Default 'exercise_name'.
        majority_class: Class to undersample. Default 'rest'.
        multiplier: Cap majority class at multiplier × largest minority class size.
                    Default 2.0 means rest gets at most 2× the largest exercise class.
        random_state: Seed for reproducibility. Default 42.

    Returns:
        Balanced DataFrame with undersampled majority class.
    """
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    if majority_class not in df[label_column].values:
        raise ValueError(
            f"majority_class '{majority_class}' not found in '{label_column}'."
        )

    counts = df[label_column].value_counts()
    print("Class distribution BEFORE undersampling:")
    print(counts.to_string())

    # Find the largest minority class (any class that is NOT the majority).
    # multiplier=2.0 means rest will have at most 2× that count — concretely,
    # if the largest exercise class has ~5,700 windows, rest is capped at
    # ~11,400, down from ~138,000.  This is the cap, not an exact target.
    minority_counts = counts.drop(labels=majority_class, errors="ignore")
    largest_minority_count = int(minority_counts.max())
    cap = int(round(largest_minority_count * multiplier))

    majority_rows = df[df[label_column] == majority_class]
    minority_rows = df[df[label_column] != majority_class]

    if len(majority_rows) <= cap:
        # Already below the cap — no sampling needed; return as-is.
        print(
            f"\nNo undersampling needed: majority class has {len(majority_rows)} rows "
            f"(<= cap of {cap})."
        )
        return df.reset_index(drop=True)

    # Sample without replacement so every kept row is a real observation.
    # random_state=42 ensures the same rows are selected across reruns,
    # which is critical for reproducibility of all downstream metrics.
    majority_sampled = majority_rows.sample(n=cap, random_state=random_state)

    balanced_df = pd.concat([minority_rows, majority_sampled], ignore_index=True)

    print(
        f"\nClass distribution AFTER undersampling (cap = {cap} = {multiplier}× largest minority):"
    )
    print(balanced_df[label_column].value_counts().to_string())

    return balanced_df
