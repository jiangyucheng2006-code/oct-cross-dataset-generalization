# Data access and frozen inputs

No image files or patient-level records are redistributed in this repository.
Obtain the three **current team-cleaned ZIPs** from the group shared Drive with
your own account. A renamed ZIP is fine; its bytes must match the recorded hash
for full training. Do not use the historical ZhangLabData_Cleaned package with a
different train/validation allocation.

| ZIP | SHA256 |
|---|---|
| OCT2017-cleaned.zip | `b9877f24caaa38db25853e468d14394c024b24663e27965a99edbb6bb14a8a79` |
| NEH-cleaned.zip | `cd743f7ef25e0282ff0630369ebbdfcbe6b16c9fc535047deee455db1e48cf25` |
| OCT-C8-cleaned.zip | `2bd1be754e672fd33b31decdcdf0777af9dcd0b0fd3fee3175d60b16f8e19bfe` |

The machine-readable contract is `configs/datasets.json`. These values identify
the team's processed archives, not official publisher checksums. If Drive
downloads an outer ZIP containing `zip/OCT-C8-cleaned.zip`, use the **inner ZIP**.
The notebook can unwrap that single nested ZIP into its cloud runtime.

Expected logical structure, allowing wrapper directories:

```text
OCT2017 ZIP:  train/CNV/*.jpeg, train/DME/..., train/DRUSEN/..., train/NORMAL/...
             val/<class>/..., test/<class>/...
NEH ZIP:     [optional wrapper/external_test/]<class>/*
OCT-C8 ZIP:  [optional wrapper/external_test/]<class>/*
```

The loader reads existing source split names and filename-derived patient IDs.
It refuses cross-split patient-ID or exact-content overlaps and source label
conflicts. Full mode compares original bytes, decoded pixels and grayscale
content before preprocessing. External exact matches to source are excluded;
external duplicates prefer NEH then the lexicographic first copy. Conflicting
external labels are excluded. These decisions never use model predictions.
The current server audit found no extra external exclusions; the portable
implementation additionally resolves transitive hash components.

Smoke mode checks the complete source filename index/patient split, but decodes
only up to 64 images per class in source train and validation. Its content audit
therefore does not cover the full dataset. It does not read test image bytes.
Full mode performs the whole audit and writes the selected cohort counts.
Original archives remain unchanged; disposable 256-square uint8 memory-map
caches are created under the cloud runtime data directory.

The three full selected image caches take about 7.8 GB before small metadata;
allow at least 12 GB free space for preparation. The full Colab notebook also
stages the three archives in `/content`, so allow about 25 GB total free space.
Keep caches on Colab's `/content` disk, rather than doing per-image Drive reads.
Checkpoints and results are mirrored to Drive after completed epochs.

Public acquisition sources (raw data are **not** interchangeable with team ZIPs):

- OCT2017/Kermany: https://data.mendeley.com/datasets/rscbjbr9sj/3
- NEH: see the dataset links and cohort definition in the primary JBO article,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13464582/ . Obtain the project's cleaned
  three-class archive from the group; do not substitute OCTID because of its name.
- OCT-C8: https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

Counts describe the present cleaned project cohort, not all available originals.
OCT-C8 has only 10 CNV and 7 DME images; its scores have substantial uncertainty.
Patient IDs derived from filenames are not clinical identity verification;
near duplicates and all augmentation ancestry are not resolved by exact hashing.
Dataset names alone do not establish independent acquisition.

The current project starts from already-cleaned archives. This repository
audits and prepares them; it does not claim to reconstruct the upstream cleaning
decisions from raw public downloads. That requires the original cleaning records.
