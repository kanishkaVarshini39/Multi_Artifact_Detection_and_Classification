# Stage B - CNN results

Device: mps. Epochs: 100. Severity heads: False.

Stage A classical target: **macro-F1 0.9811, exact-match 0.9302** (XGBoost).

| arm | macro-F1 | exact-match | Hamming | OOD macro-F1 | scatter->BH | BH->scatter |
|---|---|---|---|---|---|---|
| fixed / pretrained | 0.9965 | 0.9860 | 0.0035 | 0.9986 | 0.053 | 0.000 |
| fixed / scratch | 0.9846 | 0.9385 | 0.0154 | 0.9762 | 0.118 | 0.039 |
| per_image / pretrained | 0.9965 | 0.9860 | 0.0035 | 0.9836 | 0.053 | 0.000 |
| per_image / scratch | 0.9653 | 0.8659 | 0.0342 | 0.9315 | 0.118 | 0.039 |

### fixed / pretrained - test split

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.978 | 1.000 | 0.989 | 1.000 | 175 |
| ring | 1.000 | 1.000 | 1.000 | 1.000 | 175 |
| metal | 0.994 | 1.000 | 0.997 | 1.000 | 175 |
| scatter | 1.000 | 1.000 | 1.000 | 1.000 | 175 |

- macro-F1 **0.9965**
- micro-F1 0.9964
- exact-match accuracy **0.9860**
- Hamming loss 0.0035
- macro-AUROC 1.0000
- n = 358

### fixed / scratch - test split

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.951 | 1.000 | 0.975 | 1.000 | 175 |
| ring | 0.956 | 0.994 | 0.975 | 0.998 | 175 |
| metal | 1.000 | 1.000 | 1.000 | 1.000 | 175 |
| scatter | 0.983 | 0.994 | 0.989 | 1.000 | 175 |

- macro-F1 **0.9846**
- micro-F1 0.9845
- exact-match accuracy **0.9385**
- Hamming loss 0.0154
- macro-AUROC 0.9993
- n = 358

### per_image / pretrained - test split

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

### per_image / scratch - test split

| artifact | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| beam_hardening | 0.944 | 0.960 | 0.952 | 0.984 | 175 |
| ring | 0.966 | 0.977 | 0.972 | 0.998 | 175 |
| metal | 0.938 | 0.954 | 0.946 | 0.985 | 175 |
| scatter | 0.983 | 1.000 | 0.992 | 1.000 | 175 |

- macro-F1 **0.9653**
- micro-F1 0.9653
- exact-match accuracy **0.8659**
- Hamming loss 0.0342
- macro-AUROC 0.9917
- n = 358
