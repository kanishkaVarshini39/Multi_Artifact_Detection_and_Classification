# Stage B - multi-seed robustness

Seeds: [20261017, 20261018, 20261019]. Each arm run 3 time(s).

## Per-arm results (mean +/- sd)

| arm | n | macro-F1 | exact-match | OOD macro-F1 | OOD exact |
|---|---|---|---|---|---|
| fixed/pretrained | 3 | 0.9967 ± 0.0011 | 0.9870 ± 0.0043 | 0.9955 ± 0.0027 | 0.9824 ± 0.0105 |
| per_image/pretrained | 3 | 0.9965 ± 0.0021 | 0.9860 ± 0.0084 | 0.9843 ± 0.0007 | 0.9398 ± 0.0042 |

## Paired differences (within seed)

| comparison | metric | difference |
|---|---|---|
| fixed - per_image  [pretrained] | macro_f1 | +0.0002 ± 0.0032 (n=3) |
| fixed - per_image  [pretrained] | exact_match | +0.0009 ± 0.0126 (n=3) |
| fixed - per_image  [pretrained] | ood_macro_f1 | +0.0112 ± 0.0033 (n=3) |
