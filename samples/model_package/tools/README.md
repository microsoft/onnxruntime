# Asset generation tools (build-time only)

These scripts regenerate the sample's models and rebuild both packages. They are
**not** needed to *consume* the packages — a checkout already contains everything.

Requirements: Python with `onnxruntime` (≥ 1.28), `onnx`, `numpy`, and — for the
OpenVINO variants — the `windowsml` package plus an Intel NPU with the OpenVINO EP
(≥ 1.8.82.0 / OpenVINO SDK 2026.2, the first build with EPContext `compatibility_string`).

## `generate_assets.py`

Creates the tiny base models and compiles the OpenVINO NPU EPContext variants into a
staging directory.

```powershell
python generate_assets.py ..\_staging
```

Produces:

```
_staging/
├── base/       prefill.onnx  iter.onnx  weights.data
└── compiled/   prefill.ctx.onnx  iter.ctx.onnx  prefill.ctx_OpenVINOExecutionProvider.bin
```

- Two shape specializations (`prefill` `[1,4,64]`, `iter` `[1,1,64]`) share one
  `weights.data`.
- Compiled with `ep.context_enable=1`, `ep.context_embed_mode=0`,
  `ep.share_ep_contexts=1` so the two OpenVINO variants share **one** weightless
  `.bin` and read weights from the external `weights.data`.

## `assemble_packages.py`

Builds `portable_package/` and `nonportable_package/` (+ `external_assets/`) from
`_staging/`.

```powershell
python assemble_packages.py
```

- **Portable**: copies base and compiled files into content-addressed
  `shared_assets/sha256-<hex>/` directories and writes `ort_info.json` files with
  `sha256:` references.
- **Non-portable**: copies model files into `../external_assets/`, writes
  `ort_info.template.json` files with the `__ASSETS_DIR__` placeholder, and an
  `installed`-layout manifest.

The `sha256:<hex>` asset digests follow the model-package content-hash algorithm
(`model_package/src/asset_hasher.cc`): per-file SHA-256 over `"<hex>  <relpath>\n"`
lines sorted by path, then SHA-256 of that manifest text.

## Notes

- The committed OpenVINO `.bin` is specific to the NPU + OpenVINO SDK version it was
  compiled on. Regenerate on your own hardware if the compatibility check rejects it.
- `_staging/` is a scratch directory and can be deleted after assembly.
