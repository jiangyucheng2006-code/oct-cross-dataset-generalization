# Reconstruction record

The current architecture, training assumptions, original sources and differences
from the publication are documented in [docs/METHODS.md](docs/METHODS.md).
The main four-group protocol uses label smoothing 0.1. The preserved earlier
Dynamic/router implementation in `legacy/dynamic_20261107` used 0.0.

The code is an independent implementation of the described mechanisms. Matching
rounded parameter counts does not establish an exact reproduction of the
authors' architecture, data split or external performance.
