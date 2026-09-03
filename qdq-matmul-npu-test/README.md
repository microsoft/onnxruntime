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

`generate_qdq_matmul_model.py` creates a float32-input/output model with asymmetric uint16 activation QDQ, asymmetric uint8 constant-weight DQ, and MatMul. Weight quantization can be per-tensor, per-channel, or blockwise.

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py `
    --input-shape 1 768 `
    --weight-shape 768 512 `
    --weight-quantization blockwise `
    --block-size 32
```

The default output is `unit-models\clip_visual_dq_matmul_q.onnx`. Run `.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py -h` for all options.

### Gemma 4 E2B IT vision model's MatMul shapes for reference
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
