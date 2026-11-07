# Verification record

Verified on 2026-11-07 before initial publication:

- All Python modules compile; notebook code cells parse, outputs are cleared.
- 10 synthetic tests passed on Linux/Python3.11/PyTorch2.13.
- Tests cover grouped dynamic convolution, every-parameter gradient equivalence
  for the streamed noisy objective, no extra inference branch, parameter counts,
  NEH output mapping, ZIP smoke preparation, no smoke test-image reads,
  patient-overlap rejection, transitive duplicate conflicts, and rejecting smoke
  checkpoints for formal evaluation.
- Four real source-data smoke runs (Basic, Dynamic, Basic+noise, Dynamic+noise)
  each completed two epochs using 256 train and 256 validation images on an RTX
  5070 Ti. Each best checkpoint was reloaded and its validation confusion matrix
  reproduced. Per-epoch checkpoint mirrors were produced successfully.
- All 512 selected cached images matched the existing server preprocessing cache
  pixel-for-pixel and matched the original decoded image hashes.

These checks establish implementation/runnability evidence, not classification
accuracy or cross-dataset gains. The uploaded notebook has not been interactively
executed with a teammate's Google Drive account. Drive authorization, available
Colab GPU, library versions and session duration remain user-runtime conditions.
The managed Colab mount cell is not used in the Linux GPU verification.

Run local synthetic checks with `python -m pytest -q tests`. GitHub Actions runs
these on CPU and additionally validates the notebook schema. Full-dataset
training/evaluation results are not included in this initial code release.
