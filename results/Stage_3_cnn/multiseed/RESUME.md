# Multi-seed sweep — paused

The sweep writes `runs.json` after **every** run and skips any configuration
already present, so it is safe to stop and restart at any point. At most the
single run in flight is lost.

## To resume

```bash
cd "stage_b_cnn"
nohup bash -c 'python3 run_multiseed.py --inits pretrained && python3 run_multiseed.py' \
  > results/multiseed/log.txt 2>&1 &
```

Completed runs are skipped automatically — no flags or bookkeeping needed. The
two-command form just finishes the fast pretrained arms before the slow scratch
ones.

## To inspect without training

```bash
python3 run_multiseed.py --summarise-only
```

Rewrites `runs.csv`, `summary_by_arm.csv`, `paired_differences.json` and
`summary.md` from whatever is saved.

## To stop again

```bash
pkill -f "bash -c python3 run_multiseed.py"   # the wrapper first, so the
pkill -f "run_multiseed.py"                   # queued second command cannot start
```

Killing only the Python process lets the shell wrapper launch the next queued
command, so stop the wrapper first.

## State at pause

5 of 12 runs saved. Remaining:

- `fixed / scratch` — all three seeds
- `per_image / pretrained` — seed 20261019
- `per_image / scratch` — all three seeds

Estimated remaining time: roughly 2 hours (the scratch arms are the slow ones;
`per_image/scratch` alone took 31 minutes in the Stage B run).

Seeds: 20261017, 20261018, 20261019.
