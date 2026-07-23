# Model package sample (OpenVINO EPContext + CPU fallback)

A tiny, self-contained example of an **ONNX Runtime model package** that ships two
*shape specializations* of the same model, each with a **CPU** variant and an
**OpenVINO NPU (EPContext)** variant, and lets ORT pick the right variant for the
available execution providers at load time.

The two shape specializations mirror a typical decoder split:

| Component | Input shape   | Typical use            |
| --------- | ------------- | ---------------------- |
| `prefill` | `[1, 4, 64]`  | prompt / prefill pass  |
| `iter`    | `[1, 1, 64]`  | single-token decode    |

Both specializations share **one** weights file (`weights.data`, ~65 KB) and the two
OpenVINO variants share **one** compiled context blob
(`prefill.ctx_OpenVINOExecutionProvider.bin`, ~448 KB). The model itself is a tiny
4-layer MLP (`MatMul → Add → Relu`), small enough to commit to a repo.

> The OpenVINO variants are compiled for a specific NPU + OpenVINO SDK version and
> only run on matching hardware. The **CPU variants run anywhere** and produce the
> same result, so the sample is useful even without an NPU.

## Two packaging layouts

This sample ships the **same** models in two package formats:

- [`portable_package/`](portable_package/README.md) — **portable** layout. Everything
  lives *inside* the package as content-addressed **shared assets**. Just point ORT
  at the directory and load. This is the format you ship.
- [`nonportable_package/`](nonportable_package/README.md) — **installed** layout. The
  model files live *outside* the package in [`external_assets/`](external_assets); the
  package references them by absolute path. Because a repo can't commit
  machine-specific absolute paths (and `..` relative segments are fragile), the
  variant configs ship as `*.template.json` and a small
  [`resolve.py`](nonportable_package/resolve.py) fills in the real paths after checkout.

Both packages are consumed identically by the C++ sample.

## Layout rules used here

- The base **CPU** model files (`prefill.onnx`, `iter.onnx`) live in the **same
  directory** as their shared `weights.data`.
- The **compiled** OpenVINO files (`prefill.ctx.onnx`, `iter.ctx.onnx`) live in the
  **same directory** as their shared `.bin`.
- The OpenVINO EPContext models are **weightless** — they read weights from the same
  external `weights.data` as the CPU models (nothing is duplicated).

## Run it

Build the C++ sample (see [`cpp/README.md`](cpp/README.md)), then:

```powershell
# Portable package — loads directly.
model_package_sample.exe <onnxruntime.dll> portable_package <openvino_plugin.dll>

# Non-portable package — resolve external paths first, then load.
python nonportable_package\resolve.py
model_package_sample.exe <onnxruntime.dll> nonportable_package <openvino_plugin.dll>
```

Omit the OpenVINO plugin argument to run **CPU variants only** (works on any machine):

```powershell
model_package_sample.exe <onnxruntime.dll> portable_package
```

Expected output (abridged): each component reports the selected variant and a
non-zero result, with CPU and OpenVINO producing near-identical `absmean` values.

## Regenerating the assets

The committed models/blobs were produced on a machine with the OpenVINO NPU EP.
To regenerate them (requires the OpenVINO EP and an Intel NPU):

```powershell
python tools\generate_assets.py _staging      # tiny models + OpenVINO compile
python tools\assemble_packages.py             # build both packages from _staging
```

See [`tools/`](tools) for details. `tools/` and `_staging/` are build-time only and
are not needed to consume the packages.

## Requirements

- onnxruntime **≥ 1.28** (the model-package `OrtModelPackageApi` is experimental,
  exposed via `GetExperimentalFunction` and marked `_SinceV28`).
- For the OpenVINO variants: the OpenVINO execution-provider plugin and a compatible
  Intel NPU. Use **OpenVINO EP ≥ 1.8.82.0 (OpenVINO SDK 2026.2)** — that is the first
  version that emits and validates the EPContext `compatibility_string` the package
  relies on for variant selection. Older EP builds omit the compatibility check.
