"""
Dataset plan: which phantom gets which label combinations, and in which split.
[D1]  Resolves the spec's arithmetic conflict (item 6 implies 12 configs per
      phantom = 4800 images, but asks for 2000-2500).

Decision: keep all 400 phantoms (item 21 says phantom diversity is what stops
the classifier learning phantom-identity shortcuts) and cut to 6 configs each
= 2400, inside the stated band.

The 6 are NOT drawn uniformly at random.  Uniform sampling over the 16
combinations would leave the rare ones thinly and unevenly covered.  Instead a
QUOTA per combination is fixed up front, split proportionally across
train/val/test, and then dealt to phantoms so that:
  * every one of the 16 combinations appears in EVERY split (spec item 15)
  * no phantom ever receives the same combination twice
  * each combination is spread across many DIFFERENT phantoms, so phantom
    identity carries no information about the label
"""
import numpy as np

import config as C
from labels import COMBINATIONS, combo_name


def _largest_remainder(total, weights):
    """Split `total` into integers proportional to `weights`, summing exactly."""
    w = np.asarray(weights, float)
    exact = total * w / w.sum()
    base = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - base))[: total - base.sum()]:
        base[i] += 1
    return base


def combo_quota():
    """{combo: count} summing to N_PHANTOMS * CONFIGS_PER_PHANTOM."""
    by_card = {}
    for combo in COMBINATIONS:
        by_card.setdefault(sum(combo), []).append(combo)
    quota = {}
    for card, combos in by_card.items():
        share = _largest_remainder(C.QUOTA_BY_N_ARTIFACTS[card], [1] * len(combos))
        for combo, q in zip(combos, share):
            quota[combo] = int(q)
    return quota


def split_phantoms(rng):
    ids = rng.permutation(C.N_PHANTOMS)
    names = list(C.SPLIT_FRACTIONS)
    counts = _largest_remainder(C.N_PHANTOMS, [C.SPLIT_FRACTIONS[n] for n in names])
    out, start = {}, 0
    for name, k in zip(names, counts):
        out[name] = sorted(int(i) for i in ids[start:start + k])
        start += k
    return out


def _deal(phantom_ids, quota, rng):
    """Greedy round-based dealing: no phantom gets a combination twice."""
    need = {c: int(q) for c, q in quota.items() if q > 0}
    assigned = {p: [] for p in phantom_ids}
    for _round in range(C.CONFIGS_PER_PHANTOM):
        for p in rng.permutation(phantom_ids):
            p = int(p)
            cands = [c for c, n in need.items() if n > 0 and c not in assigned[p]]
            if not cands:
                continue
            # take from the most over-supplied combination first -> stays feasible
            best = max(cands, key=lambda c: (need[c], rng.random()))
            assigned[p].append(best)
            need[best] -= 1
    # --- repair pass: the greedy rounds can strand a few slots when the only
    # combinations still owed are ones that phantom already holds.  Place them
    # on any phantom in this split that lacks them (that phantom then carries
    # one extra config), so the quota lands exactly and no combination is lost.
    for combo, n in list(need.items()):
        while need[combo] > 0:
            cands = [p for p in phantom_ids if combo not in assigned[p]]
            if not cands:
                break
            p = min(cands, key=lambda q: (len(assigned[q]), rng.random()))
            assigned[p].append(combo)
            need[combo] -= 1

    leftover = sum(need.values())
    return assigned, leftover


def build_plan(seed=C.MASTER_SEED):
    """Returns (plan, summary). plan = list of dicts, one per sample to generate."""
    rng = np.random.default_rng(seed)
    quota = combo_quota()
    splits = split_phantoms(rng)

    # distribute each combination's quota across splits in proportion to phantom count
    names = list(splits)
    sizes = [len(splits[n]) for n in names]
    per_split_quota = {n: {} for n in names}
    for combo, q in quota.items():
        shares = _largest_remainder(q, sizes)
        for n, s in zip(names, shares):
            per_split_quota[n][combo] = int(s)

    plan, leftovers = [], {}
    for name in names:
        assigned, leftover = _deal(splits[name], per_split_quota[name], rng)
        leftovers[name] = leftover
        for pid, combos in assigned.items():
            for k, combo in enumerate(combos):
                plan.append(dict(phantom_id=pid, split=name, config_index=k,
                                 combo=combo, severity_band="train"))

    # --- supplementary out-of-range-severity test set (spec item 15 / exp. 5)
    ood_quota = {c: 0 for c in COMBINATIONS}
    n_test = len(splits["test"])
    total_ood = n_test * C.N_OOD_CONFIGS_PER_TEST_PHANTOM
    shares = _largest_remainder(total_ood, [max(quota[c], 1) for c in COMBINATIONS])
    for c, s in zip(COMBINATIONS, shares):
        ood_quota[c] = int(s)
    assigned, leftover = _deal(splits["test"], ood_quota, rng)
    leftovers["test_ood_severity"] = leftover
    for pid, combos in assigned.items():
        for k, combo in enumerate(combos):
            plan.append(dict(phantom_id=pid, split="test_ood_severity",
                             config_index=k, combo=combo, severity_band="ood"))

    summary = dict(quota={combo_name(c): q for c, q in quota.items()},
                   split_sizes={n: len(v) for n, v in splits.items()},
                   leftovers=leftovers, n_samples=len(plan))
    return plan, summary, splits
