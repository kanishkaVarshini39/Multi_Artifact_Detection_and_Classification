# Stage A - Classical baseline results

Dataset: `/Users/kanishka/Documents/AI:ML Course Project /ct_multiartifact/data` (flat layout)  
Train 1680 / val 362 / test 358 samples, phantom-disjoint.

Protocol: hyperparameters selected on validation by AUROC, decision
thresholds tuned on validation by F1, test evaluated once at the end.

### rule-based

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.722 | 0.800 | 0.759 | 0.849 | 175 |
| ring | 0.905 | 0.983 | 0.942 | 0.970 | 175 |
| metal | 0.966 | 0.971 | 0.969 | 0.998 | 175 |
| scatter | 1.000 | 0.971 | 0.986 | 0.985 | 175 |

- macro-F1 **0.9139**
- micro-F1 0.9119
- exact-match accuracy **0.6676**
- Hamming loss 0.0880
- macro-AUROC 0.9504
- n = 358

### Random Forest

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.918 | 0.954 | 0.936 | 0.981 | 175 |
| ring | 0.988 | 0.954 | 0.971 | 0.997 | 175 |
| metal | 0.994 | 1.000 | 0.997 | 1.000 | 175 |
| scatter | 0.994 | 1.000 | 0.997 | 1.000 | 175 |

- macro-F1 **0.9752**
- micro-F1 0.9751
- exact-match accuracy **0.9022**
- Hamming loss 0.0244
- macro-AUROC 0.9947
- n = 358

### XGBoost

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.971 | 0.966 | 0.968 | 0.991 | 175 |
| ring | 0.988 | 0.931 | 0.959 | 0.997 | 175 |
| metal | 1.000 | 1.000 | 1.000 | 1.000 | 175 |
| scatter | 0.994 | 1.000 | 0.997 | 1.000 | 175 |

- macro-F1 **0.9811**
- micro-F1 0.9813
- exact-match accuracy **0.9302**
- Hamming loss 0.0182
- macro-AUROC 0.9968
- n = 358

### XGBoost (OOD severity)

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.886 | 0.886 | 0.886 | 0.957 | 175 |
| ring | 1.000 | 0.931 | 0.964 | 0.998 | 175 |
| metal | 0.994 | 0.989 | 0.991 | 0.999 | 176 |
| scatter | 0.994 | 0.960 | 0.977 | 0.988 | 176 |

- macro-F1 **0.9546**
- micro-F1 0.9545
- exact-match accuracy **0.8333**
- Hamming loss 0.0437
- macro-AUROC 0.9857
- n = 360

### Shortcut ablation

| feature set | macro-F1 | exact-match | beam_hardening | ring | metal | scatter |
|---|---|---|---|---|---|---|
| all 14 features | 0.9811 | 0.9302 | 0.968 | 0.959 | 1.000 | 0.997 |
| shape only (no obj_mean/bg_level) | 0.9298 | 0.7542 | 0.884 | 0.974 | 0.994 | 0.867 |
| level only (obj_mean, bg_level) | 0.7972 | 0.3184 | 0.777 | 0.657 | 0.772 | 0.983 |

### Rule-based thresholds chosen (on validation)

| artifact | rule | val F1 |
|---|---|---|
| beam_hardening | `obj_mean < 0.0091807` | 0.8102 |
| ring | `ring_rms > 0.054207` | 0.9268 |
| metal | `streak_index > 1.9653` | 0.9944 |
| scatter | `bg_level < 0.00063158` | 0.9708 |

### Shared-evidence similarity between labels

Cosine similarity of permutation-importance profiles. High = the two
labels are read from the same features.

| artifact       |   beam_hardening |   ring |   metal |   scatter |
|:---------------|-----------------:|-------:|--------:|----------:|
| beam_hardening |            1     |  0.007 |   0.033 |     0.117 |
| ring           |            0.007 |  1     |   0     |     0.012 |
| metal          |            0.033 |  0     |   1     |     0.01  |
| scatter        |            0.117 |  0.012 |   0.01  |     1     |

### Exact-match accuracy vs number of simultaneous artifacts

|   n_artifacts |   n_samples |   exact_match |   hamming |
|--------------:|------------:|--------------:|----------:|
|             0 |          45 |        1      |    0      |
|             1 |          88 |        0.9205 |    0.0199 |
|             2 |         108 |        0.8981 |    0.0255 |
|             3 |          72 |        0.9306 |    0.0174 |
|             4 |          45 |        0.9556 |    0.0167 |

### Cross-label false-positive rates

| present        | falsely_predicted   |   n |   false_rate |   recall_of_present |
|:---------------|:--------------------|----:|-------------:|--------------------:|
| scatter        | beam_hardening      |  76 |       0.0658 |              1      |
| ring           | beam_hardening      |  76 |       0.0395 |              0.8816 |
| metal          | ring                |  76 |       0.0263 |              1      |
| metal          | scatter             |  76 |       0.0132 |              1      |
| beam_hardening | ring                |  76 |       0      |              0.9737 |
| beam_hardening | metal               |  76 |       0      |              0.9342 |
| beam_hardening | scatter             |  76 |       0      |              0.9737 |
| ring           | metal               |  76 |       0      |              0.9868 |
| ring           | scatter             |  76 |       0      |              0.8947 |
| metal          | beam_hardening      |  76 |       0      |              1      |
| scatter        | ring                |  76 |       0      |              1      |
| scatter        | metal               |  76 |       0      |              1      |
