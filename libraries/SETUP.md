# Environment setup

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r libraries/requirements.txt              # generation + validation
pip install -r libraries/requirements-training.txt     # modelling stage
```

`pip-freeze.txt` is a full snapshot of the environment the dataset was built in,
for exact reproduction if anything ever disagrees.

## Offline / archival copy of the libraries

`download_wheels.sh` fetches every wheel into `libraries/wheels/` so the exact
environment can be rebuilt with no network:

```bash
bash libraries/download_wheels.sh
pip install --no-index --find-links libraries/wheels -r libraries/requirements.txt
```

This is not run by default because it downloads a few hundred MB.

## A note on ASTRA Toolbox and TIGRE

The project brief (item 22) and the course guidelines (section 3) recommend
ASTRA or TIGRE for forward projection and reconstruction. **Neither installs on
Apple Silicon:**

* ASTRA 2.5.0 publishes `manylinux_x86_64` and `win_amd64` wheels only; its
  conda channel has `linux-64`, `osx-64` and `win-64` but no `osx-arm64`, and
  the `osx-64` line stopped at 2.0.0 (Intel, would need Rosetta).
* TIGRE requires CUDA and is a source build. `pip install tigre` installs an
  unrelated bioinformatics package.
* FDK is GPU-only in ASTRA even on Linux.

This pipeline therefore uses `scikit-image`'s `radon` / `iradon` (parallel-beam,
CPU). That is sufficient because the target is 2D slices, and all four artifact
mechanisms are projection-domain operators that behave identically under
parallel beam. No GPU is needed to generate the dataset.
