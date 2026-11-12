# Stage A - multi-seed robustness

Seeds: [0, 1, 2]. Hyperparameters fixed at the validation-selected configuration; only fitting randomness and threshold tuning vary.

| model | split | macro-F1 | exact-match |
|---|---|---|---|
| Random Forest | test | 0.9759 ± 0.0008 | 0.9050 ± 0.0028 |
| Random Forest | test_ood_severity | 0.9479 ± 0.0032 | 0.8037 ± 0.0112 |
| XGBoost | test | 0.9811 ± 0.0000 | 0.9302 ± 0.0000 |
| XGBoost | test_ood_severity | 0.9546 ± 0.0000 | 0.8333 ± 0.0000 |
