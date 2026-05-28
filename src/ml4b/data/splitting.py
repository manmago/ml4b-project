"""Subject-based train/validation/test splitting for ML4B.

This module is the fourth and final stage of the data preparation pipeline.
It takes the feature DataFrame produced by
:func:`ml4b.data.features.extract_features` and partitions it into three
disjoint sets — train, validation, test — at the **subject** level.

Why subject-based? Wrist-motion data from the same person contains a
person-specific signature (handedness, joint geometry, motion style). If
windows from subject S appear in both train and test, the model can learn
to recognize S rather than the underlying exercise, which inflates test
metrics and gives no honest signal about how the model will perform on a
new individual (the project's actual deployment scenario — Apple Watch on
an unseen user). See ADR-007.
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
