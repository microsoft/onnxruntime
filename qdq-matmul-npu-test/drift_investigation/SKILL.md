---
name: ep-qdq-drift
description: Investigate quantized ONNX CPU-versus-EP output drift, QDQ lowering, attention mismatches, and model latency with reusable ablations, graph cuts, taps, and synthetic repros.
---

# Investigating ONNX CPU/EP drift and QDQ behavior

Use this workflow when a quantized ONNX model compiles on an execution provider (EP) but differs from ORT CPU, or when an EP seems to ignore activation quantization. The scripts in this directory are reusable investigation tools; the `vision` subdirectory contains Gemma-specific graph and pipeline helpers. Run all commands from the parent `qdq-matmul-npu-test` directory using `python -m drift_investigation.<module>`. Keep model weights, compiled contexts, input archives, and reports **outside the source directory**. Never upload learned weights or representative inputs without approval.

## Setup and evidence

Use an isolated environment with `requirements.txt` and `requirements-winml.txt` for the Windows ML runner; `requirements-gemma-vision.txt` additionally installs the image processor for Gemma examples. Graph rewriting and CPU-only model tools require `onnx`, `onnx-ir`, and NumPy; the paired runner, compilation, EP tap capture, matrices, and benchmarks require **Windows ML and a registered NPU device**. `--provider openvino` and `--provider qnn` select different WinML backends, not interchangeable compiler settings. Other ORT EP installation paths are not wired into these execution scripts.

1. Record model and external-data hashes, ORT version, provider-library hash and selected device, provider options, CPU optimization level, and input hashes. `run_acc.py --report-json` captures this provenance for new reports; old reports without it should not be treated as cross-SDK evidence. Prefer realistic, **saved** inputs (include both padded and unpadded cases where relevant). `make_gemma_vision_inputs` generates reproducible RGB gradients through the actual image processor; these are not natural-image accuracy tests.
2. Establish CPU accuracy and a fully assigned EP baseline using `run_acc.py --inputs ... --no-cpu-fallback --compile-output ... --report-json ... --output-arrays ...`. Verify the compiled model has the intended EPContext nodes rather than assuming successful inference implies EP execution. CPU BASIC avoids a known blocked-INT4/uint8 `QLinearMatMul` fusion; compare runs with the **same** CPU optimization setting.
3. Validate any graph rewrite on CPU before attributing an EP change to a compiler bug. Verify whole-model versus split-pipeline equivalence and tapped versus untapped original outputs. Adding graph outputs can change fusion.
4. Inventory direct activation Q/DQ pairs with `qdq_ablation --list`; bypass one pair, one graph-connected pattern, then layers/groups. Compare baseline versus variant **on each provider separately** using `compare_qdq_reports.py`. A bitwise-invariant EP with a measurable CPU change is evidence of lost QDQ semantics for that case, *not* proof of a particular optimization pass.
5. Bisect the graph with `extract_onnx_prefix`, `extract_attention_island`, and `vision.cut_attention_inputs` or `vision.cut_attention_prefix`. Capture exact CPU/EP intermediate tensors with `add_onnx_taps` and `capture_ep_taps`. Check the probed model's original final output remains unchanged; compare an isolated island fed the **same captured tensors** as the full model.
6. Minimize a failure to a weight-free, shape-small synthetic ONNX graph if possible. Vary one input, quantization scale, zero point, or provider property at a time, and require unchanged CPU semantics for controls. Keep a float-control model; check outputs numerically, including clipping and representable QDQ ranges. Do not infer internal EP kernels from opaque EPContext graphs.
7. Stress-test operator families with `analyze_matmul_activation_qdq` or `analyze_operator_activation_qdq`. Their matrices save inputs, original/ablated models, contexts, arrays, and resumable JSON. Require clipping and CPU-output movement before calling a pair invariant; capture explicit failure records. The MatMul study varies **weight** per-tensor/per-channel/blockwise DQ with per-tensor activation QDQ; it does not test per-channel activation quantization. Individual operator results do not settle multi-op fusion behavior.
8. Benchmark only after the accuracy question is bounded: use warmups, repeated synchronous runs, and separate CPU and EP session creation from timed inference. For split graphs, report encoder-only and end-to-end encoder-plus-CPU-pooler numbers separately and account for thread-pool contention and session loading.

## Reusable commands (PowerShell from parent directory)

```powershell
$py = 'C:\path\to\venv\Scripts\python.exe'
$work = 'C:\path\to\experiment'
$model = "$work\encoder.onnx"
& $py -m drift_investigation.qdq_ablation $model --list
& $py -m drift_investigation.qdq_ablation $model --attention-core --output "$work\attention_ablation.onnx"
& $py .\run_acc.py $model --provider openvino --inputs "$work\inputs.npz" `
    --no-cpu-fallback --compile-output "$work\baseline_ctx.onnx" `
    --report-json "$work\baseline.json" --output-arrays "$work\baseline_outputs.npz"
& $py .\run_acc.py "$work\attention_ablation.onnx" --provider openvino --inputs "$work\inputs.npz" `
    --no-cpu-fallback --compile-output "$work\ablation_ctx.onnx" `
    --report-json "$work\ablation.json" --output-arrays "$work\ablation_outputs.npz"
& $py .\compare_qdq_reports.py "$work\baseline.json" "$work\ablation.json"
```

For models with external weights, keep ablations **beside the source ONNX** so they resolve the same sidecar file; copy both source ONNX and its external data into `$work` before the example above. The example assumes this was done. The comparator requires matched input hashes, provider/options, optimization level, and no CPU fallback; model hashes can differ across variants, but shared external weights must agree. Use `--provider qnn` only when a QNN device is actually registered; no QNN result was established on the Intel host.

To test the mask/Add/Softmax failure independently of Gemma weights, generate a four-op QDQ model, a two-op float control, and identical saved inputs. The generator refuses to write unless their ORT CPU outputs match bitwise; repeat in separate directories with `--scale 0.5`, `1`, and `2` to distinguish a scale-dependent lowering error from the model's intended mask values.

```powershell
& $py -m drift_investigation.make_mask_softmax_repro --output-dir "$work\mask-scale1"
& $py .\run_acc.py "$work\mask-scale1\qdq_mask.onnx" --provider openvino `
    --inputs "$work\mask-scale1\inputs.npz" --no-cpu-fallback `
    --compile-output "$work\mask-scale1\qdq_ctx.onnx"
& $py .\run_acc.py "$work\mask-scale1\float_mask.onnx" --provider openvino `
    --inputs "$work\mask-scale1\inputs.npz" --no-cpu-fallback `
    --compile-output "$work\mask-scale1\float_ctx.onnx"
```

```powershell
& $py -m drift_investigation.analyze_matmul_activation_qdq --provider openvino `
    --output-dir "$work\matmul" --limit 2
& $py -m drift_investigation.analyze_operator_activation_qdq --provider openvino `
    --output-dir "$work\operators" --op Clip --limit 2
```

Rerun with `--resume` to finish the same selected matrix. Change the output directory when you change the case selection/configuration.
Successful cases are skipped; failed or interrupted cases are retried in numbered `.retry-N` directories, preserving previous logs and artifacts. Do not edit generated inputs or the saved study configuration between runs.

## Case study: Gemma 4 vision on OpenVINO NPU

- The dynamic OneHot pooler prevented compiling the unsplit model, so the fixed-shape encoder was compiled on NPU and the dynamic pooler/projector remained on CPU. `fix_and_split_gemma_vision.py` repairs the original mask QDQ (`-100`, scale `1`, zero point `100`) while splitting. CPU whole-versus-split equivalence was checked. The downstream attention score QDQ still clips `-100`, so this is not a substitute for recalibration from a finite-mask float model.
- Manual vision attention implements **noncausal**, scale-1 QK/Softmax/V. The apparent half-strength logits in the full NPU graph are not explained by causal masking or the standard `1/sqrt(head_dim)` scale. The decisive weight-free repro was `QuantizeLinear(mask) -> DequantizeLinear(mask) -> Add(scores, mask) -> Softmax`: even an all-zero mask gave a wrong NPU answer. Float-mask `Add -> Softmax` was correct. Changing mask-QDQ scale while retaining CPU values changed the EP's effective score scaling; no particular internal compiler pass was established.
- Isolated blocked-INT4 output projection returned NPU outputs outside the declared post-QDQ range; bypassing activation QDQ moved CPU but left NPU bitwise unchanged. A no-clipping input control reduced projection error dramatically. Full-encoder layer-wise effects were nonlinear: post-O-projection QDQ in layers 1, 2, and 15 had the largest leave-one-out final impacts.
- 200 small MatMul-QDQ cases and 60 weight-free operator cases compiled fully on the tested OpenVINO EP. uint16 MatMul input and both uint16/uint8 output QDQ were often EP-invariant; uint8 MatMul input varied with weight granularity. A Clip plus output QDQ test returned raw input on NPU (lost Clip *and* QDQ), so do not assume only quantization was dropped. These observations are tied to the tested SDK 2026.3 device/plugin, inputs, and shapes.
- A counter-scaling rewrite (`renormalize_attention_qdq`) and broad QDQ bypass were useful **diagnostics**, not model fixes. Only use the rewrite with `--verify-inputs` on representative inputs; CPU equivalence did not hold universally under all ORT optimizations. Removing all activation QDQ is a different model, not proof of better end-to-end accuracy.
- Warmed encoder-only medians on synthetic padded/unpadded processor inputs: original CPU/NPU about 6.2–7.5 s / 0.92 s, mask-and-O-projection-ablation about 7.0 s / 0.93 s, all-activation-QDQ-removed about 4.5–5.0 s / 0.92 s. Compilation excluded. Whole split-pipeline results depend on CPU pooler thread allocation and loaded ORT sessions; never report these ratios as hardware peak.

The parent `README.md` contains the detailed commands, provenance, and experiment tables. `vision/README.md` covers the model-specific repro and benchmarking helpers. Preserve raw reports and failure cases locally; do not publish generated model weights or data by default.
