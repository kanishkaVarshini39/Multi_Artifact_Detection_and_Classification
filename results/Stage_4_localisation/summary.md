# Stage C - Grad-CAM localisation

Model: `resnet18_per_image_pretrained.pt` (per_image / pretrained). Split: test. 
Ring masks dilated by 4 px before overlap scoring.

## layer4

| artifact | n | mask area | pointing game | chance | lift | IoU | energy ratio |
|---|---|---|---|---|---|---|---|
| ring | 175 | 29.7% | 0.960 | 0.297 | x3.2 | 0.625 | 1.72 |
| metal | 175 | 2.2% | 0.280 | 0.022 | x12.8 | 0.043 | 1.68 |

| artifact | kind | concentration (present) | empty maps, present | empty maps, absent |
|---|---|---|---|---|
| beam_hardening | global | 0.176 | 0% | 62% |
| ring | localisable | 0.207 | 0% | 64% |
| metal | localisable | 0.174 | 0% | 61% |
| scatter | global | 0.154 | 0% | 84% |

## layer3

| artifact | n | mask area | pointing game | chance | lift | IoU | energy ratio |
|---|---|---|---|---|---|---|---|
| ring | 175 | 29.7% | 0.989 | 0.297 | x3.3 | 0.244 | 2.84 |
| metal | 175 | 2.2% | 0.029 | 0.022 | x1.3 | 0.030 | 1.24 |

| artifact | kind | concentration (present) | empty maps, present | empty maps, absent |
|---|---|---|---|---|
| beam_hardening | global | 0.323 | 0% | 0% |
| ring | localisable | 0.552 | 0% | 7% |
| metal | localisable | 0.268 | 0% | 0% |
| scatter | global | 0.357 | 0% | 0% |

## Causal occlusion test

| artifact | n | original | region erased | only region kept |
|---|---|---|---|---|
| ring | 175 | 0.999 (100%) | 0.167 (54%) | 0.994 (100%) |
| metal | 175 | 1.000 (100%) | 0.999 (100%) | 0.359 (57%) |
