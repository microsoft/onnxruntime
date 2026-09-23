# OpenVINO NPU: mask QDQ changes attention Softmax with an all-zero mask

This directory is a **standalone, weight-free** issue reproducer. It is separate
from the Gemma vision investigation: it imports no model code or experiment
scripts, and generates its own ONNX models and synthetic inputs. Nothing here
has been uploaded to an issue or to a vendor.

## Expected and observed

`repro.py` builds a two-node float control, `Add(scores, mask) -> Softmax`,
and three four-node models:

```text
QuantizeLinear(mask) -> DequantizeLinear(mask) -> Add(scores, mask) -> Softmax
```

Both inputs are float32 with shape `[1, 1, 8]`. `mask` is **all zeros**,
`scores` is a deterministic `linspace(-1, 1, 8)`, and Softmax uses `axis=-1`.
The three QDQ scales/uint16 zero points are `(0.5, 200)`, `(1, 100)`,
and `(2, 50)`. All four models have **bitwise-identical ORT CPU outputs**:
QDQ of zero is zero for every scale. There are no QK/V MatMuls, causal masks,
learned weights, model metadata, or output QDQ to confound the comparison.

On a WinML OpenVINO NPU installation with ORT 1.30.0, Intel OpenVINO EP 1.8,
and EP SDK 2026.3, all four models compiled to **one `EPContext` each** with
CPU fallback disabled. Measured against their common CPU output:

| Model | CPU/NPU mean absolute error | NPU result |
|---|---:|---|
| Float-mask control | `0.0000276` | Correct within expected device precision |
| QDQ mask, scale 0.5 | `0.0000276` | Bitwise identical to NPU float control |
| QDQ mask, scale 1 | **`0.032262`** | Wrong despite all-zero mask |
| QDQ mask, scale 2 | **`0.049413`** | Error increases |

The QDQ mask's quantization **scale** changes the NPU Softmax output even
though the mask and CPU result do not change. This localizes the observable
failure to the QDQ-mask/Add/Softmax combination, but does not establish which
internal compiler pass or kernel is responsible.

## Run

Use a Python environment with NumPy, ONNX, and a **Windows ML build of ONNX
Runtime** exposing `ModelCompiler`, `get_ep_devices`, and
`register_execution_provider_library`. Windows ML's Intel OpenVINO EP and an
Intel NPU must be available. If not already installed in that environment,
install `onnx` and `numpy`; do not replace your WinML ORT wheel with an
unrelated generic wheel.

From this directory:

```powershell
& 'C:\path\to\venv\Scripts\python.exe' .\repro.py --output-dir .\output
```

By default the script discovers the provider DLL through the Windows ML
catalog and its Python projections. To use a known registered plugin DLL
instead (no WinML catalog lookup), specify `--provider-library` with
`C:\path\to\onnxruntime_providers_openvino_plugin.dll`. The runtime must
still expose the OpenVINO NPU device.

If no NPU is available, `--cpu-only` generates the four models and validates
the bitwise CPU control; its report explicitly records `ep_tested: false`.

`output\` contains four source models (`float_mask.onnx` and
`mask_scale_{0p5,1,2}.onnx`), `inputs.npz`, compiled `_ctx.onnx` models and
their platform-specific `.bin` sidecars, and `results.json` with outputs,
device identification, SDK version, and errors. The script refuses to
overwrite an existing output directory; choose another path for a second run.
The folder is gitignored. Review artifacts before attaching them to an
external issue; the **source models, inputs, and results** are synthetic and
have no trained weights. Compiled contexts are hardware/plugin-specific and
are not needed to reproduce the issue.

Run CPU-only regression checks with:

```powershell
& 'C:\path\to\venv\Scripts\python.exe' -m unittest discover -s . -p 'test_*.py'
```

See [ISSUE.md](ISSUE.md) for an issue-ready description and the exact run
results. The related, broader Gemma investigation lives in
`..\drift_investigation\SKILL.md`; this reproducer does **not** depend on it.
