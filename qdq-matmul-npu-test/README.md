# QDQ MatMul NPU Compatibility

Generate a configurable opset-21 QDQ MatMul model and run it on CPU or a Windows ML NPU execution provider.

Reusable CPU/EP drift investigation scripts, tests, and the agent/developer workflow are in
[`drift_investigation\SKILL.md`](drift_investigation/SKILL.md). Run those tools as
`python -m drift_investigation.<module>` from this directory. The original MatMul generator,
WinML runner, model splitter, and report comparator remain here as shared entry points.
The independent, weight-free OpenVINO mask-QDQ/attention bug package and issue draft
are in [`openvino-attention-repro\README.md`](openvino-attention-repro/README.md).

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

`generate_qdq_matmul_model.py` creates a float32-input/output model with
asymmetric uint16 activation QDQ, constant-weight DQ, and MatMul.
`--activation-type uint8` selects a uint8 input/output activation QDQ
with default scale and zero points adjusted to cover approximately
the same float range. `--activation-scale`, `--output-scale`,
`--activation-zero-point`, and `--output-zero-point` override them.
Weights can be signed or unsigned, 4-bit or 8-bit, symmetric or
asymmetric, and per-tensor, per-channel, or blockwise. The default
weight profile is asymmetric per-tensor uint8.

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

### Check whether an EP preserves MatMul activation QDQ

`drift_investigation\analyze_matmul_activation_qdq.py` reuses the suite's per-tensor,
per-channel, and blockwise weight categories. Each case compiles and
runs **both** the original model and the same model with **one** input
or output activation QDQ pair bypassed. It compares CPU and NPU output
shifts on the **same saved input**, verifies an all-`EPContext` compiled
model with CPU fallback disabled, and requires measurable clipping
and CPU output change before classifying NPU invariance.

```powershell
.\.venv-winml\Scripts\python.exe -m drift_investigation.analyze_matmul_activation_qdq `
    --provider openvino
```

The default 40-case matrix covers five existing weight categories
(ONNX and `com.microsoft` per-tensor/per-channel, plus ONNX blockwise),
signed symmetric INT4 and asymmetric UINT8 weights, uint16 and uint8
activation QDQ, and separate input- and output-clipping conditions.
All inputs use deterministic bounded-normal data of shape
`[1,128,128]`; the respective input QDQ clips about **51%** of
values, or the output QDQ clips about **14%**. To use the Gemma-sized
MatMul and block size, for example:

```powershell
.\.venv-winml\Scripts\python.exe -m drift_investigation.analyze_matmul_activation_qdq `
    --shape vision --block-size 128 --block-axis 0 `
    --category blockwise_onnx --weight-variant int4_symmetric_no_zp `
    --output-dir .\unit-models\activation-qdq-study\vision-int4
```

Use repeatable `--category`, `--weight-variant`, `--activation-type`,
and `--regime` selectors to expand or narrow the matrix.
`--all-weight-variants` runs the **200-case** small-shape matrix with
all ten variants in the original suite; the two representative weight
variants above are the default.
`--limit N` records partial progress; rerun with `--resume` to finish.
Failed or interrupted cases are retried into numbered directories without
deleting the previous attempt's artifacts.
Generated ONNX models, contexts, paired output arrays, per-case logs,
and `summary.json` stay under the ignored
`unit-models\activation-qdq-study\` directory for cleanup.
The study uses **CPU BASIC** optimizations: CPU ALL incorrectly fuses
uint8 activation + blocked INT4 weight into a `QLinearMatMul` that
rejects blocked weight scales on this ORT build.

With the installed Windows ML OpenVINO NPU plugin, all **200 small**
(50 weight configurations, both activation types and stress regimes)
and **26 additional Gemma-sized** cases compiled into a single
`EPContext` without CPU fallback. In the measured clipping cases:

| Weight quantization | uint16 input QDQ | uint8 input QDQ | uint16/uint8 output QDQ |
|---|---|---|---|
| Per-tensor, all ten weight variants | NPU invariant | NPU responds | NPU invariant |
| Per-channel, symmetric weights | NPU invariant | NPU responds | NPU invariant |
| Per-channel, asymmetric weights | NPU invariant | NPU invariant | NPU invariant |
| Blockwise, all ten weight variants (axis 0; axis 1 sampled) | NPU invariant | NPU invariant | NPU invariant |

For uint8 **input** QDQ, 32 of 50 small-shape weight
configurations changed NPU output; 18 did not. All 50 uint16
input cases and all 100 output-QDQ cases were NPU-invariant.
The 200-case local run was saved in three resumable batches:
`matrix\summary.json` (40), `matrix-extra-input\summary.json` (80),
and `matrix-extra-output\summary.json` (80). The Gemma-sized checks
live under `vision-block128`, `vision-uint8weights`, and
`vision-block128-axis1`. A combined 226-row table is in
`unit-models\activation-qdq-study\activation_qdq_results.csv`.

Here **NPU invariant** means bitwise-identical outputs before and after
removing the activation QDQ, despite a measurable CPU shift; it does
not reveal the compiler pass or native kernel. All **100 small-shape
output-clip cases** also returned NPU values outside the original
output QDQ's representable range. Domain (`com.microsoft` versus ONNX)
did not change the classification for matched categories.
For two diagnostic INT4 cases, removing output QDQ from **both**
models isolated input QDQ: per-tensor uint8 activation then matched
CPU/NPU to about `5.8e-5` MAE, while blockwise uint8 activation
still produced about `0.101` MAE with QDQ and near-match without it.
Removing the five Vitis AI metadata properties from these two
diagnostic pairs left **both NPU outputs bitwise unchanged**, so that
metadata is not what selects their different behavior.
QNN results remain untested because this machine does not offer a
QNN NPU device.

### Check QDQ around weight-free Gemma operators

`drift_investigation\analyze_operator_activation_qdq.py` generates small models for the
15 non-MatMul operator families found with activation QDQ in the
corrected Gemma encoder: Add, Clip, Concat, Gather, Gelu,
LpNormalization, Mul, Neg, Reshape, Slice, Softmax, Sub, Transpose,
Unsqueeze, and Where. Each model uses float data inputs, an optional
constant/shape input as required by the operator, and **no weights**.
For uint16 and uint8 activation QDQ it separately stresses the input
and output QDQ, then runs the original and one-pair-bypassed variants
on both CPU and NPU with identical inputs. The reported clipped
fraction and CPU output shift guard against a false-green test.

```powershell
.\.venv-winml\Scripts\python.exe -m drift_investigation.analyze_operator_activation_qdq `
    --provider openvino `
    --source-model C:\path\to\model_mask_fixed_encoder.onnx
```

Use repeatable `--op`, `--activation-type`, and `--regime` arguments to
narrow the default **60-case** matrix; `--limit N` and `--resume`
preserve partial progress. As in the MatMul study, CPU uses BASIC
optimizations, compilation requires only EPContext nodes, and CPU
fallback is disabled. Generated artifacts are ignored under
`unit-models\operator-qdq-study\`, with per-case JSON under
`matrix\summary.json` and a combined `operator_qdq_results.csv`.
This tests **individual operators**, not composite fusions such as
the mask-QDQ/Add/Softmax case below.

All 60 weight-free cases compiled to one OpenVINO NPU EPContext.
Removing **input** activation QDQ changed CPU substantially but
left NPU **bitwise unchanged in all 30 cases** (both activation
types). For **output** QDQ, NPU was bitwise unchanged in **27/30**:

| Exceptional output-QDQ case | CPU shift MAE | NPU shift MAE | What happened |
|---|---:|---:|---|
| Clip, uint16 | 0.14358 | 0.01207 | Original NPU output equals raw input: both Clip and QDQ lost |
| Clip, uint8 | 0.14377 | 0.01207 | Same failure |
| Neg, uint8 | 0.15580 | 0.16278 | NPU responds; delta alignment cosine 0.99944, residual numeric error |

The Clip case has `Clip(x,-2,2)` followed by output QDQ limited
to roughly `[-1,1]`. In the original NPU model the output equals
`x` **bit-for-bit**, even outside both limits; without output QDQ,
NPU computes Clip with only about `0.00012` MAE versus CPU.
It is therefore unsafe to classify every non-invariant case as
correctly implemented QDQ. The 60 per-case model/array/report sets
identify the exact exceptions, and the uint8 Neg NPU values exceed
the output QDQ bound by at most `0.00021`, consistent with a
different numerical/rounding issue rather than losing clipping.


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

Mobius exports vision attention as explicit QK MatMul, mask Add, Softmax,
and probability-times-V MatMul because its implementation documents
an ONNX `Attention` failure (crash/NaN) above roughly 2,430 tokens.
The manual path is **not** required just to express scale 1.0 or
noncausal attention: standard ONNX Attention opsets 23/24 have
`scale` and `is_causal` attributes. Replacing the export with that
op would need separate validation at the model's 2,520-token length.

On a 1024x768 generated RGB gradient passed through the Gemma 4 image
processor (2,394 valid patches, 126 padded), ORT CPU produced identical
pooler/projector outputs from the corrected whole model and the corrected
split pipeline (maximum absolute difference `0`). Tapping the encoder's
post-`Unsqueeze` QDQ confirmed `-100` on padded positions and `0` on valid
positions. OpenVINO NPU compiled the corrected encoder as one `EPContext`
node. Its valid-patch features versus ORT CPU had mean absolute error
`24.5219` and cosine similarity `0.743133`: an improvement over the
original model's `26.2163` and `0.726607`, but substantial drift remains.

### Reusable CPU/EP QDQ ablation experiments

`drift_investigation\qdq_ablation.py` inventories **direct activation** `QuantizeLinear` ->
`DequantizeLinear` pairs in any ONNX model, then bypasses selected pairs
without changing weight dequantization. The rewrite checks shared Q/DQ
parameters, output uses, and data types before modifying the graph. It
rejects an unsupported pair rather than silently changing its semantics.
Use `--bypass-name NAME` repeatedly for exact individual pairs, or
`--attention-core` to find QDQ triples through the ONNX graph:
QK MatMul -> QDQ -> mask Add -> QDQ -> Softmax -> QDQ. The latter
uses graph connectivity rather than Mobius-specific numeric node IDs.
Variants must be written **beside the source model**: they reuse its
external weight files instead of copying them. Place one copy of the
corrected encoder and its `.onnx.data` in a dedicated experiment folder,
then keep all generated artifacts there. The script reports the number
of pairs changed and saves an `.ablation.json` selection manifest beside
each new variant. Removing that folder later will not affect the
original or the main corrected components.

`run_acc.py` can compare any variant on identical saved `.npz` inputs
with OpenVINO or QNN (`--provider openvino` or `--provider qnn`).
`--compile-output` checks that compilation produces only `EPContext` nodes;
`--no-cpu-fallback` prevents a successful CPU run from looking like an
NPU result. Use `--compiled-model` to reuse an already compiled context.
The bypassed operators themselves remain floating-point ONNX nodes in
the selected NPU `EPContext`; only the Q/DQ wrappers disappear. An
`EPContext` is opaque about the provider's internal fusion, host
pre/postprocessing, or native kernel selection.
Save reports and both providers' output arrays to measure the effect of
an ablation **on each provider**, not just their final discrepancy.
Use `--provider-option KEY=VALUE` for provider-specific settings.

For this Gemma 4 model, the optional processor dependencies are in
`requirements-gemma-vision.txt`. The following creates the same
1024x768 synthetic RGB gradient through the pretrained image processor
(no model weights downloaded), then inventories and bypasses layer 0:

```powershell
$source = 'C:\path\to\vision_encoder'
$vision = Join-Path $source 'qdq-experiments'
$py = '.\.venv-winml\Scripts\python.exe'
New-Item -ItemType Directory -Path $vision -Force | Out-Null
Copy-Item -LiteralPath "$source\model_mask_fixed_encoder.onnx" -Destination $vision
Copy-Item -LiteralPath "$source\model_mask_fixed_encoder.onnx.data" -Destination $vision
& $py -m pip install -r .\requirements-gemma-vision.txt
& $py -m drift_investigation.make_gemma_vision_inputs --output "$vision\qdq_gradient_inputs.npz" `
    --model-only-output "$vision\qdq_encoder_inputs_only.npz"
& $py -m drift_investigation.qdq_ablation "$vision\model_mask_fixed_encoder.onnx" `
    --list --group-regex 'layers[.](\d+)[./]'
& $py -m drift_investigation.qdq_ablation "$vision\model_mask_fixed_encoder.onnx" `
    --bypass 'layers[.]0[.]' `
    --output "$vision\model_mask_fixed_encoder_ablate_layer0.onnx"
```

The corrected Gemma encoder contains **1,374** direct activation QDQ
pairs: 79 per layer (80 in layer 15) and 109 outside the layers.
Mobius uses both `layers.N.` and `layers.N/` in names: matching only
`layers.N.` selects 69 pairs per layer and **misses 10 normalization
QDQ pairs per layer**. To bypass all but a chosen group, use `--keep`
instead of `--bypass`. For broad selections, protect the repaired
padding-mask QDQ with `--protect 'Where_31|Unsqueeze_32'`. Selection is
by regex over each QuantizeLinear node name and is independent of the EP.

```powershell
& $py .\run_acc.py "$vision\model_mask_fixed_encoder.onnx" `
    --provider openvino --inputs "$vision\qdq_gradient_inputs.npz" `
    --valid-mask valid_mask --no-cpu-fallback `
    --compile-output "$vision\baseline_openvino_ctx.onnx" `
    --report-json "$vision\baseline.json" --output-arrays "$vision\baseline_outputs.npz"
& $py .\run_acc.py "$vision\model_mask_fixed_encoder_ablate_layer0.onnx" `
    --provider openvino --inputs "$vision\qdq_gradient_inputs.npz" `
    --valid-mask valid_mask --no-cpu-fallback `
    --compile-output "$vision\layer0_openvino_ctx.onnx" `
    --report-json "$vision\layer0.json" --output-arrays "$vision\layer0_outputs.npz"
& $py .\compare_qdq_reports.py "$vision\baseline.json" "$vision\layer0.json"
```

Run the same commands with `--provider qnn` and QNN-specific compiled
model/report paths when investigating QNN. The report comparison requires
identical input and external-weight hashes, provider and provider options,
disabled CPU fallback, matching outputs, and actual EPContext nodes.
Compilation may change fusion or partitioning, so an ablation result is
not proof that an individual QDQ was ignored without local tensor taps.
Use `--warmup-iterations 2 --iterations 5` (or more) for warmed mean,
median, and p90 inference latency instead of interpreting a single run.
Compiled `.bin` files can be hundreds of MB; retain only the contexts
needed for follow-up.

With the saved gradient input and corrected mask, each tested variant
compiled to **one `EPContext` and no other ONNX nodes** and ran with ORT
CPU fallback disabled. The output comparisons below use valid patches
only. Shifts are measured against the *same provider's* corrected-model
baseline, rather than comparing different model variants directly.

| Activation QDQ bypassed | Pairs | CPU/NPU cosine | CPU shift MAE | NPU shift MAE |
|---|---:|---:|---:|---:|
| None (corrected mask baseline) | 0 | 0.743133 | 0 | 0 |
| Dot-named layer 0 | 69 | 0.741410 | 5.4148 | 3.77067 |
| Layer-0 QK/mask/Softmax core | 3 | 0.747486 | 0.448239 | 3.77067 |
| Slash-named layer normalization | 160 | 0.743167 | 0.850485 | **0** |
| All dot-named layers | 1,105 | 0.993162 | 17.7721 | 16.9308 |
| Dot-named attention | 992 | 0.988638 | 17.8847 | 16.9308 |
| Attention `MatMul_`, `Add_`, `Softmax_` | 128 | 0.844929 | **0.459342** | **16.9308** |
| All 16 QK/mask/Softmax cores | **48** | 0.845028 | **0.455415** | **16.9308** |
| All except corrected mask | 1,372 | 0.978389 | 18.0494 | 16.9318 |

The initial "outside layers" ablation used a **dot-only** layer regex:
its 267 selected pairs included the 160 slash-named normalization
pairs. It gave cosine `0.740985`, not evidence that 267 pairs actually
lived outside the transformer. Correcting the grouping is why both
name separators must be included when measuring a whole layer.

The NPU output for the 48-pair graph-selected core, 128-pair score,
and 992-pair attention ablations is **bit-for-bit identical**, while
their CPU outputs differ substantially. Layer 0's three core pairs
likewise reproduce the NPU output from removing all 69 dot-named
layer-0 pairs. Separately bypassing `MatMul_` (32),
`Add_` (80), or `Softmax_` (16) QDQ leaves NPU output unchanged; the
CPU shifts are respectively `0`, `0.359258`, and `0.462673` MAE.
Bypassing any pair of these categories (`Add_` + `Softmax_` (96),
`MatMul_` + `Softmax_` (48), or `Add_` + `MatMul_` (112)) also leaves
NPU output unchanged. In layer 0, bypassing only its eight score QDQ
pairs yields **bit-identical NPU output** to bypassing all 69 dot-named
layer-0 pairs; the CPU outputs differ. Thus the large NPU shift requires
a **QDQ pattern change**, not simply the removal of one category. The
compiled OpenVINO subgraph and internal kernel choice remain opaque:
the final-output ablation alone cannot identify which kernel or
quantization assumption caused the discrepancy. Local tensor taps or
compiler diagnostics are needed to distinguish those explanations.

As controls, removing 96 Q/K/V projection QDQ pairs leaves the NPU
output unchanged (CPU shift MAE `0.370271`). Removing 112 MLP QDQ
pairs shifts NPU output by MAE `0.0225473` versus CPU shift `2.63712`.
These are QDQ-ablation effects, **not** direct tests that the underlying
Q/K/V MatMul, normalization, or MLP kernels are correct. Replaying
the same Q/K/V and mask values through a smaller attention graph
can distinguish local computation from full-graph compilation effects.
Merely tapping an intermediate output can change fusion, so the
probed full-model output must first be compared against the unprobed
baseline.

### Layer-0 attention replay

`drift_investigation\extract_attention_island.py` exposes the full encoder's Q, K, V,
attention mask, and V-weighted sum, then extracts the QK MatMul ->
QDQ -> mask Add -> QDQ -> Softmax -> QDQ -> value MatMul island.
It captures the full encoder's CPU intermediate inputs, checks that
the new graph outputs do **not** change its original CPU result, and
checks that the standalone island reproduces its attention output.
The same standalone model and saved inputs can then be passed to
`run_acc.py --provider openvino` or `--provider qnn`.

```powershell
& $py -m drift_investigation.extract_attention_island "$vision\model_mask_fixed_encoder.onnx" `
    --softmax 'vision_encoder/encoder/layers.0/self_attn/Softmax_node_130' `
    --tap-output "$vision\encoder_l0_tapped.onnx" `
    --island-output "$vision\attention_core_l0.onnx" `
    --inputs "$vision\qdq_gradient_inputs.npz" --valid-mask valid_mask `
    --baseline-arrays "$vision\baseline_outputs.npz" `
    --island-inputs "$vision\attention_core_l0_inputs.npz"
& $py .\run_acc.py "$vision\attention_core_l0.onnx" `
    --provider openvino --inputs "$vision\attention_core_l0_inputs.npz" `
    --no-cpu-fallback --compile-output "$vision\attention_core_l0_openvino_ctx.onnx"
```

With CPU-captured Q/K/V/mask inputs, the isolated CPU/NPU attention
output has MAE `0.002334` and cosine `0.999991`. Removing the island's
three score QDQ pairs changes **CPU** output (MAE `0.000218`) but
leaves **NPU** output bitwise identical. Conversely, exposing those
values in the **full NPU encoder** leaves its final output bitwise
identical to the unprobed NPU baseline, yet its layer-0 V-weighted
attention output differs from isolated attention: MAE `0.0750` when
the island is fed the **same NPU-captured Q/K/V and mask inputs**.
Thus the isolated QK/mask/Softmax/value math is not sufficient to
reproduce the full-encoder mismatch. The issue depends on surrounding
compilation context, not on a changed mask value: the tapped NPU
padding mask matches ORT CPU exactly.

`drift_investigation\add_onnx_taps.py` can expose the same graph values on other variants;
`drift_investigation\capture_ep_taps.py` saves them from a compiled OpenVINO or QNN model
**only if** its original EP output is bitwise unchanged by the taps.
On the full encoder with only the layer-0 QK MatMul/mask Add/Softmax
QDQ trio bypassed, tapping again preserves its final CPU and NPU
outputs. The NPU Q and V tensors and mask remain bitwise identical to
baseline (K changes only `6.2e-14` MAE), while the layer-0 V-weighted
attention output changes by `0.075047` MAE. **After** that three-pair
ablation, full-model NPU attention output is bitwise identical to the
standalone NPU attention island running the *original QDQ* on the
same NPU-captured Q/K/V/mask. This directly separates an upstream
projection change from the context-dependent attention computation.

For a smaller reproducer, `drift_investigation\extract_onnx_prefix.py` retains just the
ancestors of a chosen value:

```powershell
& $py -m drift_investigation.extract_onnx_prefix "$vision\model_mask_fixed_encoder.onnx" `
    --value 'v_vision_encoder.encoder.layers.0.self_attn.MatMul_131' `
    --output-name attention_context `
    --output "$vision\attention_core_l0_prefix.onnx"
& $py -m drift_investigation.qdq_ablation "$vision\attention_core_l0_prefix.onnx" `
    --attention-core `
    --output "$vision\attention_core_l0_prefix_ablate_qdq.onnx"
```

This **265-node** prefix has the same model inputs and Q/K/V upstream
operators. Its CPU attention output is bitwise identical to the full
model's tapped CPU output, and its NPU attention output is bitwise
identical to the full model's tapped NPU output. The prefix CPU/NPU
comparison has MAE `0.085006`, cosine `0.992209`. Removing just its
three score QDQ pairs changes CPU output by `0.000218` MAE but NPU
output by `0.075047` MAE; CPU/NPU MAE improves to `0.015992`,
cosine to `0.998889`. The ablated prefix also reproduces the full
ablated NPU attention output bit-for-bit. This is a compact, processor-input
reproducer of the mismatch. The compiled OpenVINO subgraph still
does not expose its kernel selection, so a specific fused-kernel
implementation has **not** been proven at fault.
The prefix CPU outputs with ORT's BASIC and default ALL graph
optimizations are bitwise identical; that difference in ORT
optimization settings does not explain the measured drift.

For a **self-contained repro** that can later be tested with QNN,
pass `--self-contained` and write to a subfolder. This saves only
the prefix's used external weights (about 34 MB), not a second copy
of the full encoder's 113 MB sidecar:

```powershell
$repro = Join-Path $vision 'repro-l0-attention'
& $py -m drift_investigation.extract_onnx_prefix "$vision\model_mask_fixed_encoder.onnx" `
    --value 'v_vision_encoder.encoder.layers.0.self_attn.MatMul_131' `
    --output-name attention_context --self-contained `
    --output "$repro\attention_core_l0_prefix.onnx"
Copy-Item -LiteralPath "$vision\qdq_encoder_inputs_only.npz" -Destination "$repro\inputs.npz"
& $py -m drift_investigation.qdq_ablation "$repro\attention_core_l0_prefix.onnx" `
    --attention-core --output "$repro\attention_core_l0_prefix_ablate_qdq.onnx"
```

The packaged baseline and ablated models, single sidecar, and saved
processor inputs can be used to rerun CPU or attempt the same comparison
on another registered NPU EP; compilation remains provider-dependent.
We confirmed that repackaging preserves **both CPU and OpenVINO NPU
outputs bit-for-bit** relative to the parent experiment. The input
comes from the *synthetic RGB gradient processed by Gemma 4*, not a
natural photograph.

To rule out padding as the cause, generate an **unpadded** 1200x840
gradient through the same processor:

```powershell
& $py -m drift_investigation.make_gemma_vision_inputs `
    --output "$vision\unpadded_with_mask.npz" `
    --model-only-output "$vision\qdq_unpadded_processor_inputs.npz" `
    --gradient-width 1200 --gradient-height 840
```

It produces 2,520 valid patches and **zero padded positions**.
`repro-l0-attention\inputs_unpadded.npz` is a copy of the model-only
archive. On that input, the prefix still has CPU/NPU attention MAE
`0.083324`, cosine `0.992934`; bypassing the three QDQ pairs changes
CPU output by just `0.000228` MAE but NPU output by `0.072816`
MAE, improving CPU/NPU MAE to `0.016741`, cosine to `0.998830`.
The mismatch does **not** require padded keys or a nonzero mask bias.

On this prefix, passing `load_config={"NPU":{"NPU_QDQ_OPTIMIZATION":"YES"}}`
to the WinML OpenVINO plugin did not change NPU output. `NO` changed
it by only `1.3e-8` MAE. Neither setting repaired this mismatch; the
exact internal interpretation of that provider property remains
unverified.

### Test causal masking versus attention-score scaling

The Mobius `Gemma4VisionAttention` export explicitly computes
`Softmax(Q @ K^T + padding_bias) @ V` with **score scale 1.0**.
Transformers 5.17.0's `Gemma4VisionAttention` likewise sets
`scaling=1.0` and `is_causal=False`; the E2B vision config has
head dimension 64. The standard default `1/sqrt(64) = 0.125`
would also be wrong for this model.

`drift_investigation\probe_attention_score_scale.py` replays the standalone island's
**actual** scalar uint16 QDQ arithmetic on captured NPU Q/K/V/mask
inputs, samples query positions across all heads, and compares
causal and candidate QK-scale references to the tapped full-encoder
NPU attention output:

```powershell
& $py -m drift_investigation.probe_attention_score_scale "$vision\attention_core_l0.onnx" `
    --inputs "$vision\attention_core_l0_npu_inputs.npz" `
    --ep-output "$vision\attention_core_l0_full_npu_context.npz" `
    --cpu-outputs "$vision\attention_core_l0_npu_inputs_outputs.npz" `
    --samples 64
```

The NumPy scale-1 reference agrees with standalone ORT CPU to
`1.3e-6` MAE. Across 64 evenly spaced queries, the original full
NPU output is **much closer to roughly half-strength logits**
(scale `0.5`-`0.505`) than the model's scale `1.0`:

| Processor-generated input | Scale 1.0 MAE | Scale 0.5 MAE | Scale 0.505 MAE | Causal scale-1 MAE |
|---|---:|---:|---:|---:|
| 1024x768, 126 padded patches | 0.07739 | 0.00486 | 0.00449 | 0.13312 |
| 1200x840, no padding | 0.07511 | 0.00520 | 0.00469 | 0.13239 |

Applying the usual head-dimension default scale `0.125` instead
produces MAE `0.31067` and `0.31802`, respectively. The fitted
half-strength result is not explained by simply falling back to
standard `1/sqrt(head_dim)` scaling.

On the unpadded input's **last** query, causal and noncausal CPU
references are identical (there are no future keys), yet the NPU
still differs by `0.11203` MAE. An inadvertent triangular causal
mask therefore does **not** explain this error. Effective score
attenuation (or equivalently an incorrect Softmax temperature)
is the leading hypothesis for OpenVINO's **full-graph QDQ-sensitive
attention lowering**. The exact internal operation and why the
fitted scale is slightly above 0.5 are still unknown; the isolated
attention island itself respects the model's scale.

### Experimental workaround, not a substitute for an EP fix

An exact `Transpose` pair around Softmax changes neither CPU nor NPU
output: ORT BASIC eliminates it. Replacing Softmax with stable
`ReduceMax`/`Sub`/`Exp`/`ReduceSum`/`Div` retains those ops through
ORT BASIC and changes CPU output by only `3.2e-8` MAE, but still leaves
NPU output **bitwise identical** to the broken path. Merely doubling
logits before Softmax changes the model's CPU semantics (MAE `0.0346`
versus the original), so it is not a portable correction. Doubling
QK scores **before** their existing QDQ would also saturate many
values; about 17% of sampled QK scores exceed half of its range.

`drift_investigation\renormalize_attention_qdq.py` instead rewrites each selected
attention mask-Add QDQ as `Mul(2) -> QDQ(scale*2) -> Mul(0.5)`.
It gives the Q/DQ pair a private doubled scale if its previous
scale was shared. The added Mul nodes **survive ORT BASIC** while
retaining the original quantization grid. Use representative
`--verify-inputs` (repeatable) to require **bitwise CPU agreement**
before treating a generated variant as valid:

```powershell
& $py -m drift_investigation.renormalize_attention_qdq "$vision\model_mask_fixed_encoder.onnx" `
    --output "$vision\model_mask_fixed_encoder_qdq_renormalized.onnx" `
    --verify-inputs "$vision\qdq_gradient_inputs.npz" `
    --verify-inputs "$vision\qdq_unpadded_processor_inputs.npz" `
    --valid-mask valid_mask
```

On both tested processor inputs, the full 16-layer rewrite has
**bitwise-identical ORT CPU output**, compiles to one OpenVINO NPU
`EPContext` with CPU fallback disabled, and improves valid-patch
CPU/NPU cosine from `0.743133` to `0.845213` (MAE `24.5219` to
`17.6839`). On the 265-node prefix it improves padded CPU/NPU
attention MAE from `0.085006` to `0.015965`, and unpadded MAE
from `0.083324` to `0.016719`, again with bitwise-identical CPU
outputs. **Do not ship this as a universal semantic fix**: an
unrelated synthetic attention fixture changed its output under
ORT's ALL optimizations despite identical intermediate QDQ values.
The CLI rejects such an input when provided for verification.

The `0.845` full-encoder cosine remains inadequate. Against ORT
CPU with dot-named layer activation QDQ bypassed, the *renormalized
NPU* result instead reaches cosine `0.993165` and MAE `2.9523`.
This points to a second source of drift: other activation QDQ semantics
are not being preserved by the provider. For example, bypassing
112 MLP QDQ pairs changes CPU output by `2.637` MAE but NPU by only
`0.0225`. The residual `0.993` mismatch still needs local operator
comparisons; do not attribute all of it to attention fusion.

To check whether ORT CPU QDQ fusion was the culprit, optimized CPU
graphs are saved under `qdq-experiments` as
`encoder_ort_cpu_{disabled,basic,all}_optimized.onnx`.
All three optimization levels produce **bitwise-identical CPU output**
on the padded processor input. They retain all three layer-0
attention-score QDQ pairs, and all retain 145 float `MatMul` nodes.
The respective `QuantizeLinear` counts are 1,374 / 1,358 / 899;
there are **no** `MatMulInteger`, `QLinearMatMul`, or `MatMulNBits`
kernels in any of these CPU graphs. ALL does prune redundant QDQ,
but neither an integer MatMul fusion nor the CPU optimizer explains
the NPU's wrong effective attention scale.

The in-tree OpenVINO EP reads `ORT_OPENVINO_ENABLE_DEBUG` to dump
its handed-off ONNX subgraph in **non-Release** builds, not the
post-compiler NPU graph. Its `ORT_OPENVINO_PERF_COUNT=<directory>`
hook requires a compiled model with profiling enabled. With
`load_config={"NPU":{"PERF_COUNT":true}}`, the installed Windows ML
plugin generated **no CSV** in our test. The officially documented
NPU `optimization-level=0` yielded bitwise-identical incorrect
output; `NPU_QDQ_OPTIMIZATION=YES` also left output unchanged and
`NO` changed it by only `1.3e-8` MAE. Detailed lowering/fusion logs
may therefore require an Intel debug build or vendor instrumentation.

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
