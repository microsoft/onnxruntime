# QDQ MatMul NPU Compatibility

Generate a configurable opset-21 QDQ MatMul model and run it on CPU or a Windows ML NPU execution provider.

## Set up

Create the model-generation environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Create the Windows ML runner environment:

```powershell
python -m venv .venv-winml
.\.venv-winml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-winml\Scripts\python.exe -m pip install --no-deps -r .\requirements-winml.txt
```

## Generate the unit model

`generate_qdq_matmul_model.py` creates a float32-input/output model with asymmetric uint16 activation QDQ, constant-weight DQ, and MatMul. Weights can be signed or unsigned, 4-bit or 8-bit, symmetric or asymmetric, and per-tensor, per-channel, or blockwise. The default is asymmetric per-tensor uint8.

| QDQ profile | Standard opset | Q/DQ domain | Q/DQ opset |
|---|---:|---|---:|
| `onnx` | 21 | Default ONNX | 21 |
| `microsoft` | 17 | `com.microsoft` | 1 |

Select the profile with `--qdq-profile onnx` or `--qdq-profile microsoft`. Per-tensor and per-channel weights support both profiles; blockwise weights require the `onnx` profile because `com.microsoft::DequantizeLinear` has no `block_size` attribute.

Add `--add-vitisai-metadata` to write the CLIP model identity and Vitis AI quantization properties expected for uint16 activations, uint8 weights, and `OnnxStaticQuantization`.

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py `
    --weight-quantization blockwise `
    --weight-bit-width 4 `
    --weight-signedness signed `
    --weight-symmetry symmetric `
    --block-size 32
```

The default input shape is `[1, 2520, 768]`, the default weight shape is `[768, 768]`, and the default output is `unit-models\clip_visual_dq_matmul_q.onnx`. Run `.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py -h` for all options.

For symmetric signed weights, use `--omit-weight-zero-point` to omit the
optional zero-point input instead of supplying an all-zero initializer.

## Generate and run the compatibility test suite

`generate_qdq_matmul_test_suite.py` generates 100 models covering per-tensor
and per-channel weights with Microsoft and ONNX Q/DQ, plus blockwise ONNX Q/DQ.
Each category covers signed and unsigned 4-bit and 8-bit weight variants,
including both omitted and explicit zero points for symmetric signed weights.
It runs every model through `run_acc.py --log-severity-level 0`, saves the
combined logs, and writes NPU/CPU node placements and accuracy metrics to an
Excel workbook.

Run the suite from the Windows ML environment:

```powershell
.\.venv-winml\Scripts\python.exe .\generate_qdq_matmul_test_suite.py
```

Blockwise models use block size 32 on axis 0 by default. Use `--block-size`,
`--block-axis`, `--provider`, and repeatable `--provider-option KEY=VALUE`
arguments to override suite settings.


### Gemma-4-E2B-IT vision model's MatMul shapes for reference
| Gemma 4 E2B-IT vision MatMul use | Left input shape | Right input shape | Output shape | Count |
|---|---|---|---|---:|
| 768-wide projection | `[batch, num_patches, 768]` | `[768, 768]` | `[batch, num_patches, 768]` | 65 |
| Attention QK transpose | `[batch, 12, num_patches, 64]` | `[batch, 12, 64, num_patches]` | `[batch, 12, num_patches, num_patches]` | 16 |
| Attention probabilities by V | `[batch, 12, num_patches, num_patches]` | `[batch, 12, num_patches, 64]` | `[batch, 12, num_patches, 64]` | 16 |
| MLP gate/up projection | `[batch, num_patches, 768]` | `[768, 3072]` | `[batch, num_patches, 3072]` | 32 |
| MLP down projection | `[batch, num_patches, 3072]` | `[3072, 768]` | `[batch, num_patches, 768]` | 16 |
| Pooler | `[batch, _d0, num_patches]` | `[batch, num_patches, 768]` | `[batch, _d0, 768]` | 1 |
| Projector | `[batch, _d0, 768]` | `[768, 1536]` | `[batch, _d0, 1536]` | 1 |

## Run a model

`run_winml_ep.py` measures one provider:

```powershell
.\.venv-winml\Scripts\python.exe .\run_winml_ep.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider cpu `
    --iterations 100
```

Use `--provider vitisai`, `qnn`, or `openvino` for an NPU.

`run_acc.py` compares CPU and NPU outputs:

```powershell
.\.venv-winml\Scripts\python.exe .\run_acc.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider vitisai `
    --seed 1009
```

Add repeatable `--provider-option KEY=VALUE` arguments for provider-specific settings.
Both runners allow CPU fallback by default. Add `--no-cpu-fallback` to require full NPU execution.

## Compile and run a precompiled model

Compile a model for QNN:

```powershell
.\.venv-winml\Scripts\python.exe .\compile_winml_ep_model.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider qnn `
    --output .\unit-models\clip_visual_qnn_ctx.onnx
```

Run the precompiled model in a separate process:

```powershell
.\.venv-winml\Scripts\python.exe .\run_winml_ep.py `
    .\unit-models\clip_visual_qnn_ctx.onnx `
    --provider qnn `
    --iterations 100
```

## Split the Gemma 4 vision pooler

`split_gemma_vision_pooler.py` separates a Mobius Gemma 4 vision model before
the position-based pooler. The encoder component retains the fixed-shape vision
transformer and produces `vision_features`. The CPU component accepts
`vision_features` and `pixel_position_ids`, then runs the dynamic-shape spatial
pooler, projector norm, and projector.

```powershell
.\.venv-winml\Scripts\python.exe .\split_gemma_vision_pooler.py `
    C:\path\to\model.onnx
```

The default outputs are `<model>_encoder.onnx` and
`<model>_pooler_projector.onnx`, each with its own `.onnx.data` file. Use
`--encoder-output`, `--pooler-output`, and `--overwrite` to control the output
paths. The script uses `onnx-ir` to preserve and prune the QDQ graph and
external initializers.

### Correct the QDQ padding mask and split

For the original Mobius QDQ export, run `fix_and_split_gemma_vision.py`
instead of the plain splitter:

```powershell
.\.venv-winml\Scripts\python.exe .\fix_and_split_gemma_vision.py `
    C:\path\to\model.onnx
```

This leaves the source model untouched and writes
`<model>_mask_fixed_encoder.onnx` and
`<model>_mask_fixed_pooler_projector.onnx`, each with its own `.onnx.data`
file. Supply `--encoder-output`, `--pooler-output`, or `--overwrite` as
needed. The default padding bias is `-100`; `--mask-value` accepts another
negative integer down to `-65535`. The script checks the expected Mobius
mask graph and fails if its initializers or consumers differ.

Changing the original `-1e9` constant alone is **not** sufficient:
the downstream uint16 QDQ with scale 1 and zero point 0 clips every
negative bias to zero on ORT CPU. The script changes both the quantized
constant's scale and the shared zero point of the `Where` and `Unsqueeze`
QDQ pairs (to 100 for the default bias). This makes padded keys
dequantize to `-100` and valid keys to `0` at both QDQ sites. This repairs
the mask representation; it does **not** resolve the separate activation-QDQ
provider drift described below.

On a 1024x768 generated RGB gradient passed through the Gemma 4 image
processor (2,394 valid patches, 126 padded), ORT CPU produced identical
pooler/projector outputs from the corrected whole model and the corrected
split pipeline (maximum absolute difference `0`). Tapping the encoder's
post-`Unsqueeze` QDQ confirmed `-100` on padded positions and `0` on valid
positions. OpenVINO NPU compiled the corrected encoder as one `EPContext`
node. Its valid-patch features versus ORT CPU had mean absolute error
`24.5219` and cosine similarity `0.743133`: an improvement over the
original model's `26.2163` and `0.726607`, but substantial drift remains.

### Gemma 4 encoder accuracy investigation

The `google/gemma-4-E2B-it` `Gemma4ImageProcessorPil` from transformers 5.17.0
produces `[1, 2520, 768]` float32 patches in `[0, 1]` and
`[1, 2520, 2]` position IDs. A generated 1024x768 RGB gradient image
produced 2394 valid patches, 126 padded slots, and 266 pooled tokens.
The image was synthetic but went through the actual image processor; it was
not a real photograph. Earlier random normal input was *not* representative
of the processor's expected pixel range.

For the split encoder's valid patch output, using identical processor inputs:

| Comparison | Mean absolute error | Cosine similarity |
|---|---:|---:|
| OpenVINO NPU vs ORT CPU (original QDQ graph) | 26.2163 | 0.726607 |
| OpenVINO NPU vs ORT CPU (diagnostic bypass of activation QDQ pairs) | 20.6798 | 0.820065 |
| OpenVINO CPU vs ORT CPU (diagnostic bypass of activation QDQ pairs) | 0.0161 | 0.99999989 |

The early patch projection and patch embedding closely match on NPU
(cosine greater than 0.9999999); deviations accumulate in transformer
layers. In this graph, the padding attention bias is `-1e9` before QDQ,
but its next QDQ uses **uint16, scale 1, zero point 0**. ORT CPU
clips that negative value to zero, unintentionally unmasking padded
keys. OpenVINO NPU instead yields approximately `-65504` after this
QDQ at the same location. Removing only that mask QDQ in a diagnostic
CPU model improves NPU-vs-CPU cosine from 0.726607 to 0.737647:
the mask is a real model correctness issue, but does not explain the
whole provider mismatch. In addition, an unpadded 42x60 synthetic grid
still shows substantial NPU-vs-CPU differences. The near match to
the activation-QDQ-bypassed CPU graph indicates that OpenVINO does not
preserve the original activation QDQ semantics; the remaining NPU
difference needs further isolation before attributing it to a specific
kernel or precision mode.
