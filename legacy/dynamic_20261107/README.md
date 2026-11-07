# Historical Dynamic/router branch

This directory preserves the earlier `dynamic`, `dynamic_stats` and
`dynamic_balance` implementation. Train/validation/test protocol and model
assumptions are described in `../../docs/METHODS.md`. Its label smoothing is 0.0.
Do not mix its results with the matched four-group smoothing-0.1 experiment.

The included experiment list now defaults to seed42 only, as requested for
further work. Existing earlier server multi-seed jobs/results are unaffected.
No dataset payloads, private paths or run outcomes are included.

Set `config.json`'s `audit_dir` to the prepared full-data cache, then, for example:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python legacy/dynamic_20261107/train.py --config legacy/dynamic_20261107/config.json --variant dynamic_stats --seed 42 --out runs/legacy/dynamic_stats_seed42 --resume
```

For final evaluation train all variants declared in this directory's
`experiments.json` first, then run its `evaluate.py` with a separate run root.
Use `src/prepare_data.py` for portable preparation. Historical `audit_data.py`
expects already-extracted folders and is kept for implementation traceability.
