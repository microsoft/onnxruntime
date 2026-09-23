# Vision drift investigation tools

## Synthetic images used in the Gemma investigation

The image-processor inputs came from deterministic RGB gradients, not photographs.
Red rises from left to right, green rises from top to bottom, and blue stays at
127 throughout. The center is gray; the corners transition from dark blue
(upper left) to magenta (upper right), teal (lower left), and pale yellow
(lower right). Both source images are saved here for visual inspection:

| Processor case | Image | Patch coverage |
|---|---|---|
| Padded, 1024 x 768 | [View RGB gradient](examples/gradient_1024x768.png) | 2,394 valid, 126 padded |
| Unpadded, 1200 x 840 | [View RGB gradient](examples/gradient_1200x840.png) | 2,520 valid, none padded |

`drift_investigation.make_gemma_vision_inputs.gradient_image(width, height)`
generates these exact images. The saved PNGs are source images only; the model
inputs are processor-produced patch tensors and position IDs, not PNG pixels.

Run these tools from `qdq-matmul-npu-test` so both the `drift_investigation`
package and the root `split_gemma_vision_pooler.py` helper are importable:

```powershell
python -m drift_investigation.vision.cut_attention_prefix model.onnx `
  --boundary "/model/.../DequantizeLinear_output_0" --input-name hidden_states `
  --output artifacts\attention_prefix.onnx

python -m drift_investigation.vision.cut_attention_inputs attention.onnx `
  --cut query_float=query --cut key_float=key --cut value_float=value `
  --output artifacts\attention_inputs.onnx

python -m drift_investigation.vision.extract_projection_island encoder.onnx `
  --layer 12 --projection o --output artifacts\o_projection.onnx

python -m drift_investigation.vision.extract_projection_island encoder.onnx `
  --matmul-name-contains "/custom/output_projection/MatMul" `
  --output artifacts\projection.onnx

python -m drift_investigation.vision.make_synthetic_repro reduced.onnx `
  --expected-count .qweight=4 --expected-count .scales=4 `
  --output artifacts\synthetic.onnx

python -m drift_investigation.vision.vary_mask_qdq attention_mask_qdq.onnx `
  --scale 0.1 --zero-point 1000 --inputs inputs\unpadded.npz `
  --output artifacts\attention_mask_qdq_variant.onnx

python -m drift_investigation.vision.make_short_sequence_repro 128 `
  --model models\attention_mask_qdq.onnx `
  --model models\attention_mask_qdq_removed.onnx `
  --inputs inputs\unpadded.npz --output-dir artifacts\short_128
```

The cutters require `onnx-ir` and use graph utilities from the root
`split_gemma_vision_pooler.py`. They never overwrite an output. The source
graph must have one output. Boundaries must be named, non-input values whose
type/shape can be read directly or inferred from adjacent Q/DQ nodes.
`extract_projection_island` requires the exact chain
`Q -> DQ -> MatMul <- DQ`, followed by `Q -> DQ`, with a unique selected
MatMul and unique output-Q/output-DQ consumers.

`make_synthetic_repro` requires `onnx` and NumPy. It recognizes Gemma/Olive
initializer suffixes (`.qweight`, `.qzeros`, `.scales`,
`position_embedding_table_quantized`, and `.weight_quantized`), updates
associated scale/zero-point tensors, preserves activation calibration, and
writes one external-data file named `<output>.data`. Unrecognized external
tensors are rejected by default because they may contain learned parameters.
Use `--expected-count` to make a particular reduced model's contract exact.

## Minimal mask-QDQ/Add/Softmax repro

`vary_mask_qdq` changes only a named uint16 mask-QDQ scale and zero point.
Its defaults match the initializer names in the original Intel repro, but
`--scale-initializer` and `--zero-point-initializer` support renamed models.
By default, both `-100` and `0` must remain exactly representable after
float32 QDQ, and at least one NPZ archive is required to prove every CPU
output remains bitwise identical. NPZ keys are discovered from model inputs.
`--skip-cpu-check` is available when ONNX Runtime is unavailable, but weakens
the reproducibility contract.

`make_short_sequence_repro` changes static input/output/value-info dimensions
equal to `--source-length` (default `2520`) and slices QKV/mask arrays along
configurable axes. It supports one or more weight-free attention models and
writes resized models plus `inputs.npz` to a new directory. The default axes
match the original shapes: query/value sequence axis 2 and key/mask sequence
axis 3. It requires an all-zero unpadded mask unless
`--allow-nonzero-mask` is given.

Limitations: this is a metadata resize rather than arbitrary graph rewriting.
It is valid only when sequence-dependent computation accepts the shorter
shape without sequence-sized initializers or embedded constants. The script
does not compile an EPContext model and does not claim NPU equivalence; use
the separate compile/run tools with the generated external artifacts.

## Latency benchmarks

Both benchmarks require NumPy, an ONNX Runtime Python build exposing the
chosen EP, a discoverable NPU, and the parent directory's `run_acc.py` and
`run_winml_ep.py`. `--provider openvino` is the default; select
`--provider qnn` or `--provider vitisai` only on a host with the corresponding
NPU device and compiled contexts. Repeat `--provider-option KEY=VALUE` for
runtime options and use contexts compiled with matching EP settings.
Compiled models must contain exactly one `EPContext` node. CPU and compiled
encoders must expose identical inputs/outputs and exactly one output.

```powershell
python -m drift_investigation.vision.benchmark_encoder_latency `
  --variant baseline models\encoder.onnx models\encoder_ctx.onnx `
  --variant ablated models\encoder_no_qdq.onnx models\encoder_no_qdq_ctx.onnx `
  --input-set padded inputs\padded.npz --valid-mask padded=valid_mask `
  --warmups 2 --iterations 7 --report results\encoder_latency.json

python -m drift_investigation.vision.benchmark_vision_pipeline_latency `
  --variant baseline models\encoder.onnx models\encoder_ctx.onnx `
  --input-set padded inputs\padded.npz --valid-mask padded=valid_mask `
  --pooler models\pooler_projector.onnx --feature-input-name vision_features `
  --forward-input pixel_position_ids --report results\pipeline_latency.json
```

`--variant` and `--input-set` are repeatable. Input NPZ keys must match model
inputs; a `--valid-mask SET=KEY` key is removed from session inputs and used
only by the encoder accuracy comparison. The pipeline benchmark sends the
encoder's sole output to `--feature-input-name` and copies each
`--forward-input` from the encoder input dictionary to the CPU pooler.
Reports are created incrementally and are never overwritten; `--resume`
requires an existing report with an identical hashed configuration.

No models, input archives, compiled contexts, or benchmark results are stored
in this directory. Only the two Intel-repro scripts with a reusable minimal
mask-QDQ/Add/Softmax contract were generalized; generated contexts, inputs,
outputs, and comparison JSON files remain external.
