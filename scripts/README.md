# Slurm examples

Edit partition/GPU resource directives to match your cluster. Export `OCT_REPO`,
`OCT_CONFIG`, `OCT_RUNS`, `OCT_ARCHIVE`, and optionally `OCT_PYTHON`. Use absolute
paths. Prepare and audit the data before submitting training. For example:

```bash
job=$(sbatch --parsable scripts/train.sbatch)
sbatch --dependency=afterok:$job scripts/evaluate.sbatch
```

The array contains four tasks with concurrency one and seed42 only. These
scripts are portable examples, not modifications to previously submitted jobs.
The notebook uses ordinary Python processes and requires no Slurm connection.
