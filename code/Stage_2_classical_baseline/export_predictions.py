"""
Export the best classical baseline's predictions.

Downstream stages need these, but they must NOT import Stage A to get them:
Stage A has its own config.py, and putting it on sys.path hands every later
stage the wrong configuration.  Writing predictions to disk keeps Stage A
self-contained and lets anything else read plain arrays.

    python3 export_predictions.py      ->  results/predictions.npz
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import config as C
from data import load


def main():
    meta = json.loads((C.RESULTS / "metrics.json").read_text())
    name = meta["best_baseline"]["name"]
    kind = {"Random Forest": "rf", "XGBoost": "xgb"}[name]
    model = joblib.load(C.RESULTS / "models" / f"{kind}_multilabel.joblib")

    _, splits, root, layout = load()
    arrays, payload = {"model_name": np.array(name)}, {}
    for split in ("test", "test_ood_severity"):
        if split not in splits:
            continue
        X, Y = splits[split]["X"], splits[split]["Y"]
        arrays[f"{split}__Y"] = Y
        arrays[f"{split}__pred"] = model.predict(X)
        arrays[f"{split}__prob"] = model.predict_proba(X)
        # as fixed-width unicode, not object dtype, so it loads with allow_pickle=False
        arrays[f"{split}__sample_id"] = splits[split]["meta"]["sample_id"].to_numpy().astype("U")
        payload[split] = int(len(Y))

    out = C.RESULTS / "predictions.npz"
    np.savez_compressed(out, **arrays)
    print(f"{name}: wrote {out}")
    for k, v in payload.items():
        print(f"  {k}: {v} samples")


if __name__ == "__main__":
    main()
