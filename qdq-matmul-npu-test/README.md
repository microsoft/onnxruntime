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
    --weight-quantization blockwise `
    --block-size 32
```

The default input shape is `[1, 2520, 768]`, the default weight shape is `[768, 768]`, and the default output is `unit-models\clip_visual_dq_matmul_q.onnx`. Run `.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py -h` for all options.

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
