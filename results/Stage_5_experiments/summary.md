# Stage D - required experiments (spec item 18)

Classical baseline: **XGBoost**  
CNN: **resnet18_per_image_pretrained.pt** (per_image / pretrained)

## Experiment 1 - does combined training help or hurt single-artifact detection?

Three arms. The brief's version compares only A and C, which confounds training COMPOSITION with training SIZE (C has 2.7x the data). Arm B is size-matched to A, so A-vs-B isolates composition and B-vs-C isolates quantity.

| arm | n train | single_only | single_or_clean | multi_artifact | all |
|---|---|---|---|---|---|
| A_singles_only | 630 | 1.000 / 1.000 | 1.000 / 1.000 | 0.656 / 0.173 | 0.724 / 0.480 |
| B_combined_matched | 630 | 0.984 / 0.966 | 0.984 / 0.977 | 0.991 / 0.951 | 0.990 / 0.961 |
| C_combined_full | 1680 | 1.000 / 1.000 | 1.000 / 1.000 | 0.994 / 0.969 | 0.995 / 0.980 |

Cells are macro-F1 / exact-match.

On single-artifact test samples: composition effect **-0.0164** macro-F1 (A→B, size held equal), data-quantity effect **+0.0164** (B→C).

## Experiment 2 - CNN vs classical baseline

| model | beam_hardening | ring | metal | scatter | macro-F1 | exact-match | Hamming |
|---|---|---|---|---|---|---|---|
| rule-based | 0.759 | 0.942 | 0.969 | 0.986 | **0.9139** | **0.6676** | 0.0880 |
| Random Forest | 0.936 | 0.971 | 0.997 | 0.997 | **0.9752** | **0.9022** | 0.0244 |
| XGBoost | 0.968 | 0.959 | 1.000 | 0.997 | **0.9811** | **0.9302** | 0.0182 |
| CNN (per_image/pretrained) | 0.989 | 1.000 | 0.997 | 1.000 | **0.9965** | **0.9860** | 0.0035 |

CNN over the best classical baseline: **+0.0153** macro-F1, **+0.0559** exact-match.

## Experiment 3 - beam-hardening / scatter confusion

| model | scatter present → false BH | BH present → false scatter |
|---|---|---|
| XGBoost | 6.6% | 0.0% |
| CNN (per_image/pretrained) | 5.3% | 0.0% |
| XGBoost [all 14 features] | 6.6% | 0.0% |
| XGBoost [shape only (no obj_mean/bg_level)] | 42.1% | 27.6% |
| XGBoost [level only (obj_mean, bg_level)] | 55.3% | 0.0% |

## Experiment 4 - accuracy vs number of simultaneous artifacts

| model | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| XGBoost | 1.000 | 0.920 | 0.898 | 0.931 | 0.956 |
| CNN (per_image/pretrained) | 1.000 | 1.000 | 0.963 | 0.986 | 1.000 |

### Per-combination accuracy, 16 classes (spec item 16)

| combo | n | XGBoost | CNN (per_image/pretrained) |
|---|---|---|---|
| 0000 | 45 | 1.000 | 1.000 |
| 0001 | 22 | 0.909 | 1.000 |
| 0010 | 22 | 0.864 | 1.000 |
| 0100 | 22 | 0.955 | 1.000 |
| 1000 | 22 | 0.955 | 1.000 |
| 0011 | 18 | 1.000 | 1.000 |
| 0101 | 18 | 0.833 | 0.833 |
| 0110 | 18 | 0.667 | 1.000 |
| 1001 | 18 | 0.944 | 1.000 |
| 1010 | 18 | 1.000 | 1.000 |
| 1100 | 18 | 0.944 | 0.944 |
| 0111 | 18 | 0.889 | 0.944 |
| 1011 | 18 | 1.000 | 1.000 |
| 1101 | 18 | 0.889 | 1.000 |
| 1110 | 18 | 0.944 | 1.000 |
| 1111 | 45 | 0.956 | 1.000 |

## Experiment 5 - generalisation to held-out severity bands

| model | macro-F1 in-band | macro-F1 held-out | exact in-band | exact held-out |
|---|---|---|---|---|
| XGBoost | 0.9811 | 0.9546 | 0.9302 | 0.8333 |
| CNN (per_image/pretrained) | 0.9965 | 0.9836 | 0.9860 | 0.9361 |

### XGBoost - test split

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

### XGBoost - held-out severity

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

### CNN (per_image/pretrained) - test split

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.978 | 1.000 | 0.989 | 0.999 | 175 |
| ring | 1.000 | 1.000 | 1.000 | 1.000 | 175 |
| metal | 0.994 | 1.000 | 0.997 | 1.000 | 175 |
| scatter | 1.000 | 1.000 | 1.000 | 1.000 | 175 |

- macro-F1 **0.9965**
- micro-F1 0.9964
- exact-match accuracy **0.9860**
- Hamming loss 0.0035
- macro-AUROC 0.9998
- n = 358

### CNN (per_image/pretrained) - held-out severity

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.960 | 0.960 | 0.960 | 0.988 | 175 |
| ring | 1.000 | 1.000 | 1.000 | 1.000 | 175 |
| metal | 0.977 | 0.977 | 0.977 | 0.999 | 176 |
| scatter | 1.000 | 0.994 | 0.997 | 1.000 | 176 |

- macro-F1 **0.9836**
- micro-F1 0.9836
- exact-match accuracy **0.9361**
- Hamming loss 0.0160
- macro-AUROC 0.9968
- n = 360
