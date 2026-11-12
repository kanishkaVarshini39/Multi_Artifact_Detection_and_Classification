"""
Dataset validation.  Run after generate_dataset.py.

Checks the properties the five resolved defects were supposed to buy:
  D1  all 16 label combinations present in every split; quota met
  D2  no sample carries a label for an artifact below its detectability floor
  D3  severity labels and handcrafted features are correlated but DISTINCT
  --  no phantom leaks across train/val/test (spec item 6)
  --  the dataset is actually learnable (quick Random Forest sanity check)
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as C
from features import FEATURE_NAMES, DETECTION_INDEX

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    rows = list(csv.DictReader(open(C.DATA / "labels.csv")))
    feats = {r["sample_id"]: r for r in csv.DictReader(open(C.DATA / "features.csv"))}
    manifest = json.loads((C.DATA / "manifest.json").read_text())
    floors = manifest["severity_floors"]
    main_rows = [r for r in rows if r["split"] != "test_ood_severity"]

    print(f"\n{len(rows)} samples ({len(main_rows)} main + "
          f"{len(rows)-len(main_rows)} OOD-severity) over {C.N_PHANTOMS} phantoms\n")

    print("--- identifiers")
    ids = [r["sample_id"] for r in rows]
    check("sample_id is globally unique", len(set(ids)) == len(ids),
          f"{len(ids) - len(set(ids))} collisions")
    fids = [r["sample_id"] for r in csv.DictReader(open(C.DATA / "features.csv"))]
    check("features.csv covers exactly the same ids", set(fids) == set(ids)
          and len(fids) == len(ids))

    print("\n--- splits and leakage")
    ph = {s: set() for s in ["train", "val", "test", "test_ood_severity"]}
    for r in rows:
        ph[r["split"]].add(int(r["phantom_id"]))
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        check(f"{a} / {b} phantom-disjoint", not (ph[a] & ph[b]),
              f"{len(ph[a])} vs {len(ph[b])} phantoms")
    check("OOD-severity set reuses exactly the test phantoms",
          ph["test_ood_severity"] == ph["test"])
    check("every phantom used", set().union(*ph.values()) == set(range(C.N_PHANTOMS)))

    print("\n--- D1: combination coverage and quota")
    for sp in ["train", "val", "test", "test_ood_severity"]:
        c = Counter(r["combo"] for r in rows if r["split"] == sp)
        check(f"{sp}: all 16 combinations present", len(c) == 16,
              f"{len(c)}/16, min {min(c.values())}, max {max(c.values())}")
    check("main set size within the spec's 2000-2500", 2000 <= len(main_rows) <= 2500,
          str(len(main_rows)))
    rates = {a: sum(int(r[f"y_{a}"]) for r in main_rows) / len(main_rows) for a in C.ARTIFACTS}
    check("per-label positive rate in [0.4, 0.6]",
          all(0.4 <= v <= 0.6 for v in rates.values()),
          ", ".join(f"{a}={v:.3f}" for a, v in rates.items()))
    nart = Counter(int(r["n_artifacts"]) for r in main_rows)
    print(f"         samples by artifact count: {dict(sorted(nart.items()))}")

    print("\n--- D2: severity floors respected")
    viol_zero = viol_floor = 0
    for r in rows:
        for a in C.ARTIFACTS:
            y = int(r[f"y_{a}"]); sp = float(r[f"sphys_{a}"])
            if y == 0 and sp != 0.0:
                viol_zero += 1
            if y == 1 and sp < floors[a] - 1e-9:
                viol_floor += 1
    check("absent artifacts carry zero severity", viol_zero == 0, f"{viol_zero} violations")
    check("present artifacts sit at or above the calibrated floor", viol_floor == 0,
          f"{viol_floor} violations")
    tr = [float(r[f"sphys_{a}"]) for r in rows if r["severity_band"] == "train"
          for a in C.ARTIFACTS if int(r[f"y_{a}"])]
    oo = [float(r[f"sphys_{a}"]) for r in rows if r["severity_band"] == "ood"
          for a in C.ARTIFACTS if int(r[f"y_{a}"])]
    print(f"         train band s_phys [{min(tr):.3f},{max(tr):.3f}]  "
          f"OOD band s_phys [{min(oo):.3f},{max(oo):.3f}]")

    print("\n--- D3: severity labels are distinct from the handcrafted features")
    for a in C.ARTIFACTS:
        idx = DETECTION_INDEX[a]
        pres = [r for r in main_rows if int(r[f"y_{a}"])]
        x = np.array([float(r[f"sev_{a}"]) for r in pres])
        y = np.array([float(feats[r["sample_id"]][f"f_{idx}"]) for r in pres])
        rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
        check(f"{a}: |corr(severity label, {idx})| < 0.99 (not the same quantity)",
              abs(rho) < 0.99, f"r={rho:+.3f}")

    print("\n--- artifact signal is present in the images (PAIRED within phantom)")
    # Unpaired comparison is the wrong test here: the cupping index also tracks
    # inclusion layout, which varies far more between phantoms than the artifact
    # moves it.  Comparing each phantom against ITSELF cancels that structure
    # term -- the same reason the floor calibration is paired.
    # Compare each phantom's CLEAN sample against its SINGLE-ARTIFACT sample, so
    # neither structure (same phantom) nor co-occurring artifacts (single-artifact
    # only) can confound the comparison.
    clean_by_ph, single_by_ph = {}, defaultdict(dict)
    for r in main_rows:
        pid = int(r["phantom_id"])
        if r["combo"] == "0000":
            clean_by_ph[pid] = r
        elif int(r["n_artifacts"]) == 1:
            which = C.ARTIFACTS[r["combo"].index("1")]
            single_by_ph[which][pid] = r
    for a in C.ARTIFACTS:
        idx = DETECTION_INDEX[a]
        deltas = []
        for pid, r in single_by_ph[a].items():
            if pid in clean_by_ph:
                deltas.append(float(feats[r["sample_id"]][f"f_{idx}"])
                              - float(feats[clean_by_ph[pid]["sample_id"]][f"f_{idx}"]))
        deltas = np.array(deltas)
        if deltas.size < 10:
            check(f"{a}: enough clean/single pairs to test", False, f"n={deltas.size}")
            continue
        eff = abs(deltas.mean()) / (deltas.std(ddof=1) + 1e-18)
        agree = float((np.sign(deltas) == np.sign(deltas.mean())).mean())
        check(f"{a}: {idx} shifts vs the same phantom's clean image (|mean/sd| > 0.3)",
              eff > 0.3, f"effect={eff:.2f}, sign-agreement={agree:.2f}, n={deltas.size}")

    print("\n--- learnability: Random Forest on the 14 handcrafted features")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, roc_auc_score
    except ImportError:
        print("  (scikit-learn unavailable, skipped)")
    else:
        def mat(split):
            rr = [r for r in rows if r["split"] == split]
            X = np.array([[float(feats[r["sample_id"]][f"f_{k}"]) for k in FEATURE_NAMES]
                          for r in rr], dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = np.array([[int(r[f"y_{a}"]) for a in C.ARTIFACTS] for r in rr])
            return X, Y
        Xtr, Ytr = mat("train"); Xte, Yte = mat("test")
        f1s, aucs, preds = [], [], []
        for i, a in enumerate(C.ARTIFACTS):
            clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
            clf.fit(Xtr, Ytr[:, i])
            p = clf.predict(Xte); pr = clf.predict_proba(Xte)[:, 1]
            preds.append(p)
            f1s.append(f1_score(Yte[:, i], p)); aucs.append(roc_auc_score(Yte[:, i], pr))
            print(f"         {a:15s} F1={f1s[-1]:.3f}  AUROC={aucs[-1]:.3f}")
        P = np.stack(preds, 1)
        exact = float((P == Yte).all(1).mean())
        print(f"         macro-F1={np.mean(f1s):.3f}  exact-match={exact:.3f}")
        check("classical baseline macro-F1 > 0.5 (dataset is learnable, not degenerate)",
              np.mean(f1s) > 0.5, f"{np.mean(f1s):.3f}")
        check("classical baseline leaves headroom for the CNN (macro-F1 < 0.98)",
              np.mean(f1s) < 0.98, f"{np.mean(f1s):.3f}")

        # difficulty vs severity -- informs whether the severity band is too generous
        print("\n--- difficulty vs severity (per-label recall on the test split)")
        Xo, Yo = mat("test_ood_severity")
        rr_o = [r for r in rows if r["split"] == "test_ood_severity"]
        for i, a in enumerate(C.ARTIFACTS):
            clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
            clf.fit(Xtr, Ytr[:, i])
            sev = np.array([float(r[f"sphys_{a}"]) for r in rr_o])
            pr = clf.predict(Xo)
            pos = Yo[:, i] == 1
            if pos.sum() < 8:
                continue
            q = np.quantile(sev[pos], [0.0, 0.25, 0.5, 0.75, 1.0])
            parts = []
            for k in range(4):
                sel = pos & (sev >= q[k]) & (sev <= q[k + 1])
                if sel.sum():
                    parts.append(f"[{q[k]:.2f}-{q[k+1]:.2f}] recall={pr[sel].mean():.2f} (n={sel.sum()})")
            print(f"         {a:15s} " + "  ".join(parts))

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
