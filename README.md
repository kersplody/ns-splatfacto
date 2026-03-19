# ns-splatfacto

`ns-splatfacto` is a narrowed fork of Nerfstudio for a specific workflow:

- train `splatfacto`
- train `splatfacto-big`
- evaluate trained runs
- export gaussian splat PLYs
- export camera poses from trained runs

This repository is no longer intended to be a general Nerfstudio distribution. Viewer, web UI, generic dataset processing, mesh export, and the broader multi-method CLI surface have been cut out of the supported scope.

The Python package namespace remains `nerfstudio` to keep upstream merges manageable. The installable package name is `ns-splatfacto`.

## Scope

Supported executables:

- `ns-splatfacto`

Supported `ns-splatfacto` subcommands:

- `train splatfacto`
- `train splatfacto-big`
- `eval`
- `export-gaussian-splat`
- `export-camera-poses`

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

No extra CLI bootstrap step is required.
Converter CLIs are now maintained separately under the root-level `colmap2transforms` package while it still lives in this repo.

## Runtime Assumptions

This fork assumes:

- a CUDA-capable PyTorch environment
- `gsplat` is available and compatible with your Torch/CUDA stack
- your own app or container handles data import and orchestration

This repo does not try to manage the full old Nerfstudio environment anymore.

## Commands

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
python3 -m py_compile nerfstudio/scripts/splatfacto.py colmap2transforms/colmap2transforms.py colmap2transforms/transforms2colmap.py
```

## Notes

- COLMAP conversion now lives under the separate `colmap2transforms` package in this repo and is no longer part of the `ns-splatfacto` runtime surface.
- The repository still contains legacy code under the `nerfstudio` tree that is no longer part of the supported runtime surface.
- The package manifest has been reduced to the dependencies required for the splatfacto-only workflow.
- If you move data import fully into your own app, keep this repo focused on training/eval/export rather than rebuilding the old generic CLI surface.
