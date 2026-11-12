# Project 29 — Multi-Artifact Detection and Classification

Objectives
1.Build a multi-artifact corpus in which the presence
vector is exact by construction, severity is recorded, and
no label is attached to an artifact that cannot be seen.

2.Establish a fair classical baseline on hand-designed fea-
tures, and a convolutional baseline, scored by identical
code on identical splits.

3.Determine what evidence each of them actually uses, by
removing the overall-brightness cue and measuring what
is lost.

4.Measure the confusion between beam hardening and
scatter under both conditions, since the physics predicts
the two should be hard to separate.

5.Test whether the regions we mark as ground truth are
the regions the model relies on, by erasing them and
re-running the model.

## What is in here

```
code/                             50 Python files, five stages + a live demo
  Stage_1_dataset_generation/     physics simulation: phantoms, spectra, projection, reconstruction
  Stage_2_classical_baseline/     14 handcrafted features -> rule-based / random forest / XGBoost
  Stage_3_cnn/                    multi-label ResNet-18, two normalisation regimes
  Stage_4_localisation/           Grad-CAM, pointing game, IoU, causal occlusion
  Stage_5_experiments/            the five required experiments
  demo.py                         end-to-end run on one fresh image, ~5 seconds

libraries/                        dependency manifests + offline wheel fetcher
training data/                    train + validation
testing data/                     test + held-out severity
results/                          metric tables and figures from the runs
metadata/                         manifest, severity floors, QC figures, all-split tables
```


## The data

Four independent artifacts — beam hardening, ring, metal streak, scatter — each
present or absent, giving 16 combinations. Every image is a 256x256 float32
reconstruction in per-pixel attenuation units.

| folder | images | phantoms | masks | combinations |
|---|---|---|---|---|
| `training data/train` | 1680 | 280 | 1176 | 16/16 |
| `training data/validation` | 362 | 60 | 253 | 16/16 |
| `testing data/test` | 358 | 60 | 251 | 16/16 |
| `testing data/held_out_severity` | 360 | 60 | 252 | 16/16 |

Splits are **separated by phantom**, so no object appears in more than one
split. `held_out_severity` is a distribution-shift test set: its artifact
strengths fall outside the band the models were trained on (below 0.20 or above
0.80 normalised, against a training band of 0.20–0.80).

Each split folder is self-contained:

```
images/            <sample_id>.npy      256x256 float32 reconstruction
masks/             <sample_id>.npz      ring and metal regions, where they exist
labels.csv         the four binary labels, severities, parameters, relative paths
features.csv       the 14 handcrafted measurements
split_summary.json counts, phantom ids, per-label positives
```

`image_path` and `mask_path` in `labels.csv` are relative to the split folder.
Masks exist only for ring and metal — beam hardening and scatter are global
artifacts with no localisable region, and that asymmetry is a finding, not a gap.


## Running it

```bash
pip install -r libraries/requirements.txt            # dataset + classical stages
pip install -r libraries/requirements-training.txt   # CNN, Grad-CAM, experiments
```

For an offline or archival rebuild, `libraries/download_wheels.sh` fetches every
wheel into `libraries/wheels/`. `libraries/pip-freeze.txt` is the exact
environment the reported numbers came from.

The fastest way to see the project work end to end:

```bash
cd code && python3 demo.py
```

It builds a fresh phantom, corrupts it, reconstructs it, measures the 14
features, classifies it with the trained network, draws the Grad-CAM map, then
deletes the metal insert and asks again. About five seconds. The last step is
the point: the prediction does not move.

Then, in order:

```bash
cd code/Stage_2_classical_baseline && python3 run_baseline.py     # seconds, tabular only
cd code/Stage_3_cnn               && python3 run_stage_b.py       # GPU/MPS recommended
cd code/Stage_4_localisation      && python3 run_stage_c.py
cd code/Stage_5_experiments       && python3 run_stage_d.py
```

Stage 1 regenerates the corpus from scratch into
`code/Stage_1_dataset_generation/data/` — it does not read the shipped copy:

```bash
cd code/Stage_1_dataset_generation
python3 scripts/calibrate_floors.py && python3 scripts/generate_dataset.py
```


## Notes for a reader checking this

**Dataset discovery.** Each stage resolves the data through
`DATASET_CANDIDATES` in its `config.py` / `settings.py`. The package layout here
is tried first. To point a stage at a different copy, edit that list — nothing
else hardcodes a data path.

**Normalisation window.** The `fixed` regime divides by a fixed physical window,
`window_mu_max = 0.03866064` per-pixel units (60 keV effective).
It is a property of the dataset, so it is recorded in
`metadata/preprocess_window.json` and read from there.

**Model weights.** Four checkpoints ship, enough to run the demo and reproduce
the headline model: the three Stage 2 models and
`resnet18_per_image_pretrained.pt`. The other six checkpoints from the
normalisation x initialisation grid and the Stage 5 experiment arms are omitted
for size; `run_stage_b.py` and `run_stage_d.py` retrain them.

**Verified before packaging.** From inside this package, the shipped checkpoint
scores **macro-F1 0.9965** on `testing data/test` at its stored thresholds
(0.9887 / 1.0000 / 0.9972 / 1.0000 for beam hardening / ring / metal / scatter),
exact-match 0.9860 — the figure reported in the write-up. All 2760 image paths
resolve and all 2760 sample ids are unique.
