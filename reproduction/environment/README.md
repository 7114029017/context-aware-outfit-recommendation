# Reproduction environment

## Official run and reference run

The manuscript's official run `full_20261002T161428Z` and the earlier
reference run `reference_20260921T175217Z` used the same runtime. Their
captured environment evidence is under:

- `reproduction/environment/full_20261002T161428Z/`
- `reproduction/environment/reference_20260921T175217Z/`

The runtime recorded for both runs:

- Python: 3.12.3
- OS/kernel: Linux 6.17.0-1021-nvidia, aarch64
- PyTorch: 2.9.1+cu130
- PyTorch CUDA build: 13.0
- cuDNN: 9.13.0
- GPU: NVIDIA GB10
- GPU capability reported by PyTorch: 12.1
- torchvision: 0.24.1
- transformers: 4.57.1
- NumPy: 2.0.2
- scikit-learn: 1.6.1
- einops: 0.8.1
- Pillow: 11.3.0
- tqdm: 4.67.1
- ipdb: 0.13.13

The archived handoff does not establish that this 2026 runtime is
byte-for-byte or version-for-version identical to the original 2025
training environment.

## Runtime requirements

The reproduction package list is:

`reproduction/environment/requirements-reproduction-runtime.txt`

Checkpoint-evaluation requirements are documented in:

`reproduction/environment/requirements-checkpoint-eval.txt`

Python 3.12 is required: the pinned `scipy==1.18.1` needs Python 3.12 or
later, and `numpy==2.0.2` has no packages for Python 3.13. A CUDA-enabled
PyTorch / torchvision installation compatible with the target machine should
be installed before the remaining Python requirements.

Example environment setup (the complete steps are in `reproduction/README.md`
section 0.3):

    python3.12 -m venv .venv-repro
    source .venv-repro/bin/activate
    python -m pip install --upgrade pip
    python -m pip install torch==2.9.1 torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cu130
    python -m pip install -r reproduction/environment/requirements-reproduction-runtime.txt

The requirements file intentionally does not silently replace the
machine-specific CUDA PyTorch build.

## Current workstation snapshot

The post-run repository-maintenance environment is recorded in:

- `requirements-lock-current_20260924.txt`
- `environment-current_20260924.txt`

These files improve handoff traceability.

They are not presented as proof of the exact historical 2025 environment.

## Version-drift note

The reproduction preflight reports the installed PyTorch package as
`2.9.1+cu130`, while the archived requirement string records `2.9.1`.

This difference is recorded as version drift rather than silently changed.

Required imports and CUDA availability passed the completed reproduction
preflight.

## Hardware caveat

PyTorch emitted a warning that the NVIDIA GB10 reports compute capability
12.1 while the installed build reports support through capability 12.0.

The warning is preserved as environment evidence.

The completed 35-unit runs (the reference run and the official run)
nevertheless finished with status `PASSED`; this fact does not establish
equivalence to the original 2025 hardware/runtime.
