# Environment and portability

Python 3.10+ and PyTorch 2.2+ are required. Use the CUDA-enabled PyTorch already
provided by Colab; do not replace its CUDA stack with the cluster environment.
Install `requirements.txt` before importing NumPy/Pillow. Restart the runtime if
Colab asks after an installation. The code records the actual Python, PyTorch,
NumPy, GPU, driver, configuration and source hashes with each run.

The submitted cluster batch used Python 3.11, PyTorch 2.13.0+cu130, NumPy 2.4.6,
Pillow 11.3.0 and FP32 on an RTX 5080. These are recorded environment facts, not
required Colab package pins. Different GPU/library versions can yield different
numerical results and speed, even with the same fixed seed.

Colab GPU availability/session duration varies. Checkpoints are mirrored to Drive
at each completed epoch. The notebook restores the latest complete checkpoint
and the same configuration on restart. It cannot recover unfinished epoch work.
Only load checkpoints produced by this project in a trusted location: the resume
file includes optimizer/RNG Python objects and is loaded with `weights_only=False`.

Reference: https://research.google.com/colaboratory/faq.html
