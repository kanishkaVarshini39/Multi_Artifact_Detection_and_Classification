"""
Training-set subsets for experiment 1.

The brief asks: does training on combined corruptions help or hurt
single-artifact detection?  As stated it compares

    A  train on single-artifact data only      (630 samples)
    C  train on the full combined dataset      (1680 samples)

which CONFOUNDS two things at once: composition and quantity.  C has 2.7x the
data, so a win for C could simply be a win for "more data".  A third arm fixes it:

    B  train on combined data, subsampled to 630  (size-matched to A)

A vs B isolates composition.  B vs C isolates quantity.  That separation is
worth making explicitly in the report -- the brief's version cannot distinguish
the two effects.
"""
import numpy as np
from torch.utils.data import Subset


def singles_only_indices(df):
    """Rows with at most one artifact present (clean + the four singles)."""
    return np.where(df["n_artifacts"].to_numpy() <= 1)[0]


def stratified_subsample(df, n_target, seed):
    """
    Sample n_target rows while keeping every label combination represented in
    proportion to its share of the full set.
    """
    rng = np.random.default_rng(seed)
    combos = df["combo"].astype(str).to_numpy()
    uniq, counts = np.unique(combos, return_counts=True)
    share = counts / counts.sum()
    take = np.floor(share * n_target).astype(int)
    take = np.maximum(take, 1)                       # never drop a combination
    # trim or pad to hit n_target exactly, largest groups first
    while take.sum() > n_target:
        j = int(np.argmax(take - 1))
        take[j] -= 1
    while take.sum() < n_target:
        j = int(np.argmax(counts - take))
        take[j] += 1

    picked = []
    for c, k in zip(uniq, take):
        idx = np.where(combos == c)[0]
        picked.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    return np.array(sorted(picked))


def make_arms(train_ds, seed):
    """Returns {arm_name: (Subset, description)}."""
    df = train_ds.df
    a = singles_only_indices(df)
    b = stratified_subsample(df, len(a), seed)
    return {
        "A_singles_only": (Subset(train_ds, a.tolist()),
                           f"single-artifact data only, n={len(a)}"),
        "B_combined_matched": (Subset(train_ds, b.tolist()),
                               f"combined data, size-matched, n={len(b)}"),
        "C_combined_full": (train_ds,
                            f"full combined dataset, n={len(df)}"),
    }


def test_masks(df):
    """Evaluation slices of the test set, for reporting experiment 1."""
    n = df["n_artifacts"].to_numpy()
    return {
        "single_or_clean": n <= 1,
        "single_only": n == 1,
        "multi_artifact": n >= 2,
        "all": np.ones(len(n), bool),
    }
