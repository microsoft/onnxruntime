# Portable package

A **portable**-layout model package: fully self-contained and movable. Every file it
needs lives *inside* this directory, so a consumer only has to point ONNX Runtime at
the package root — no setup step.

## Layout

```
portable_package/
├── manifest.json                 # components + variants, "layout": "portable"
├── prefill/                      # component: prefill (input [1,4,64])
│   ├── cpu/ort_info.json         #   CPU variant  -> base model + weights
│   └── ov/ort_info.json          #   OpenVINO NPU variant -> compiled ctx + bin
├── iter/                         # component: iter (input [1,1,64])
│   ├── cpu/ort_info.json
│   └── ov/ort_info.json
└── shared_assets/
    ├── sha256-<data-hex>/        # base CPU models + shared weights, together
    │   ├── prefill.onnx
    │   ├── iter.onnx
    │   └── weights.data
    └── sha256-<bin-hex>/         # compiled OpenVINO models + shared bin, together
        ├── prefill.ctx.onnx
        ├── iter.ctx.onnx
        └── prefill.ctx_OpenVINOExecutionProvider.bin
```

Shared assets are **content-addressed**: the directory name is `sha256-<hex>` where the
hash covers both file contents and names. Variants reference them by the URI
`sha256:<hex>[/sub/path]`, which ORT resolves against the package. Two packages that
ship the same weights can therefore de-duplicate storage when installed.

## How the variants are wired

Each variant's `ort_info.json` is ORT's slot (`executor_info["ort"]`). Path-valued
`session_options` are resolved against the package at load time.

**CPU variant** (`prefill/cpu/ort_info.json`):

```json
{
  "model_file": "sha256:<data-hex>/prefill.onnx",
  "session_options": {
    "session.model_external_initializers_file_folder_path": "sha256:<data-hex>"
  }
}
```

The base model's external initializers (`weights.data`) are read from the same shared
asset directory as the `.onnx`.

**OpenVINO NPU variant** (`prefill/ov/ort_info.json`):

```json
{
  "model_file": "sha256:<bin-hex>/prefill.ctx.onnx",
  "session_options": {
    "ep.share_ep_contexts": "1",
    "session.model_external_initializers_file_folder_path": "sha256:<data-hex>",
    "ep.context_file_path": "sha256:<bin-hex>/prefill.ctx.onnx"
  }
}
```

- `ep.share_ep_contexts=1` — required so the OpenVINO EP loads the **weightless**
  weights from the external file (without it the model loads but produces all zeros).
- `model_external_initializers_file_folder_path` → the **data** asset: where
  `weights.data` lives.
- `ep.context_file_path` → the **compiled** asset: only its folder is used, so the
  EPContext `.bin` is found there. This is what lets the `.bin` and `weights.data`
  live in **different** shared assets.

Both OpenVINO variants reference the **same** `.bin`, so the compiled context is
loaded once and shared across the two shape specializations.

## Variant selection

`manifest.json` tags each variant with its EP (`CPUExecutionProvider` /
`OpenVINOExecutionProvider`), the OpenVINO variant with `"device": "npu"`, and a
`compatibility_string` that the OpenVINO EP validates against the current hardware.
At load time ORT:

- picks the **OpenVINO** variant when an OpenVINO NPU device is registered *and* the
  compatibility check passes;
- picks the **CPU** variant when no EP (or only CPU) is registered;
- fails selection if you request a device that has no matching variant (e.g. an
  OpenVINO GPU device — this package only ships CPU and OpenVINO-NPU variants).

## Load it

```powershell
..\cpp\model_package_sample.exe <onnxruntime.dll> . <openvino_plugin.dll>
# CPU-only (any machine):
..\cpp\model_package_sample.exe <onnxruntime.dll> .
```
