# ns-splatfacto

`ns-splatfacto` is a narrowed fork of Nerfstudio for a specific workflow:

- train `splatfacto`
- train `splatfacto-big`
- evaluate trained runs
- export gaussian splat PLYs
- export camera poses from trained runs
- create `transforms.json` from a COLMAP binary model

This repository is no longer intended to be a general Nerfstudio distribution. Viewer, web UI, generic dataset processing, mesh export, and the broader multi-method CLI surface have been cut out of the supported scope.

The Python package namespace remains `nerfstudio` to keep upstream merges manageable. The installable package name is `ns-splatfacto`.

## Scope

Supported executables:

- `ns-splatfacto`
- `ns-create-transforms`

Supported `ns-splatfacto` subcommands:

- `train splatfacto`
- `train splatfacto-big`
- `eval`
- `export-gaussian-splat`
- `export-camera-poses`
- `export-colmap-transforms`

Supported `ns-create-transforms` workflow:

- read `cameras.bin` and `images.bin` from a COLMAP model directory
- ignore `.txt` model files even if present
- write `transforms.json`

Out of scope for this fork:

- viewer and websocket services
- `ns-install-cli`
- generic `ns-train`, `ns-export`, `ns-render`, `ns-process-data`
- mesh, TSDF, point cloud, and texture export
- notebook and documentation tooling as package features

## Install

This package is intended to be installed like a standard Python console-script package.

```bash
pip install -e .
```

That installs:

- `ns-splatfacto`
- `ns-create-transforms`

No extra CLI bootstrap step is required.

## Runtime Assumptions

This fork assumes:

- a CUDA-capable PyTorch environment
- `gsplat` is available and compatible with your Torch/CUDA stack
- your own app or container handles data import and orchestration

This repo does not try to manage the full old Nerfstudio environment anymore.

## Commands

### Create `transforms.json` from COLMAP

Create `transforms.json` from a COLMAP binary model directory:

```bash
ns-create-transforms --model_dir=colmap/sparse/0 --output_file=transforms.json
```

Defaults:

- `--model_dir=.` by default
- `--output_file=.` by default
- if `--output_file=.` or a directory path is given, output becomes `./transforms.json`

Examples:

```bash
ns-create-transforms
ns-create-transforms --model_dir=colmap/sparse/0
ns-create-transforms --model_dir=colmap/sparse/0 --output_file=transforms.json
```

Notes:

- only `cameras.bin` and `images.bin` are used
- `.txt` model files are ignored
- file paths in frames default to `./images/...`

### Train

Train default splatfacto:

```bash
ns-splatfacto train splatfacto --config.data /path/to/dataset
```

Train the larger variant:

```bash
ns-splatfacto train splatfacto-big --config.data /path/to/dataset
```

This slim CLI disables viewer usage and runs with `vis="none"`.

### Evaluate

Evaluate a trained run from a saved config:

```bash
ns-splatfacto eval \
  --load-config outputs/<scene>/<method>/<timestamp>/config.yml \
  --output-path eval.json
```

### Export Gaussian Splat PLY

```bash
ns-splatfacto export-gaussian-splat \
  --load-config outputs/<scene>/<method>/<timestamp>/config.yml \
  --output-dir exports/splat
```

### Export Camera Poses From a Trained Run

```bash
ns-splatfacto export-camera-poses \
  --load-config outputs/<scene>/<method>/<timestamp>/config.yml \
  --output-dir exports/cameras
```

### Export COLMAP Transforms Through `ns-splatfacto`

If you prefer to keep everything under one executable:

```bash
ns-splatfacto export-colmap-transforms \
  --model-dir colmap/sparse/0 \
  --output-path transforms.json
```

## Package Layout

Important naming distinction:

- installable distribution: `ns-splatfacto`
- Python import namespace: `nerfstudio`

This is intentional. Keeping the `nerfstudio` module tree reduces merge pain if changes need to be pulled from upstream later.

## Development

Install development extras:

```bash
pip install -e .[dev]
```

Basic checks:

```bash
python3 -m py_compile nerfstudio/scripts/splatfacto.py nerfstudio/scripts/create_transforms.py
```

## Notes

- The repository still contains legacy code under the `nerfstudio` tree that is no longer part of the supported runtime surface.
- The package manifest has been reduced to the dependencies required for the splatfacto-only workflow.
- If you move data import fully into your own app, keep this repo focused on training/eval/export rather than rebuilding the old generic CLI surface.
