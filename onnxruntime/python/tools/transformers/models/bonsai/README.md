# Bonsai 2 ONNX conversion

These tools convert the language component of a PrismML Bonsai 2 MLX pack to ONNX and run greedy text generation with
ONNX Runtime. Both the 2B (tied output head) and 27B (untied output head, swish attention gate) packs are supported.

The converter preserves the published 2-bit, group-128 affine weights:

- `MatMulNBits` runs packed linear layers with zero point 1.
- `GatherBlockQuantized` runs the packed token embedding.
- ONNX tensor operations apply the signed, normalized Walsh-Hadamard transforms (block 512 for 2B, 1024 for 27B).
- `LinearAttention` runs the Qwen3.5 gated-delta recurrence.

## Dependencies

Install `numpy`, `onnx`, `onnxruntime`, `safetensors`, `torch`, and `transformers`.

Download a complete Bonsai 2 MLX pack. For example:

```powershell
hf download prism-ml/Ternary-Bonsai-2-2B-mlx-2bit --local-dir C:\models\bonsai-2-2b
```

## Convert

```powershell
python convert_to_onnx.py `
  --model-dir C:\models\bonsai-2-2b `
  --output C:\models\bonsai-2-2b.onnx `
  --sequence-length 24
```

The exported model has a fixed sequence length and returns logits only for the position selected by `logits_index`.
Generation recomputes the fixed window instead of using a KV or recurrent-state cache.
Export a fresh Q2 graph with this converter before comparing formats. Older graphs can have a different
recurrent-attention implementation; matching T8 and Q2 outputs alone does not establish source-model correctness.

## Run

```powershell
python run_onnx.py `
  --model C:\models\bonsai-2-2b.onnx `
  --model-dir C:\models\bonsai-2-2b `
  --prompt "Capital of France? Answer with one word." `
  --max-new-tokens 1
```

The prompt plus generated tokens must fit within the sequence length selected during conversion.

## T8 ternary bitplanes on ARM64

Export directly from the original Bonsai MLX checkpoint with `--weight-format t8`:

```powershell
python convert_to_onnx.py `
  --model-dir C:\models\bonsai-2-2b `
  --output C:\models\bonsai-2-2b-t8\model.onnx `
  --sequence-length 24 --weight-format t8

python convert_to_onnx.py `
  --model-dir C:\models\bonsai-2-27b `
  --output C:\models\bonsai-2-27b-t8\model.onnx `
  --sequence-length 24 --weight-format t8
```

This path reads the original `model.safetensors`, encodes each linear matrix directly into T8 buffers,
and exports `MatMulTernary` nodes. It does not create or read an intermediate Q2 ONNX graph, dequantize
the full weights, or requantize the model. The checkpoint's affine 2-bit codes are losslessly repacked and
round-trip checked before export. Embeddings remain Q2; a tied output head is encoded independently as T8.
Omitting `--weight-format` preserves the existing Q2 export behavior.

Use a separate output directory for each export and retain the graph together with **all** external-data
files written beside it. Small graphs can embed their weights; large exports use external data.
Direct exports record `t8.export=direct_checkpoint` and the format/source/round-trip metadata in the graph.
The direct 2B and 27B exports were checked against the earlier T8 graphs: all 552 matrices' bitplanes
and scales are bit-identical, and both direct exports pass the eager PyTorch full-logit comparison described below.

For an existing Q2 ONNX graph, `convert_to_t8.py` remains available to losslessly re-encode its
linear weights as T8 v1, `unit5x128-planes-v1`.
This is the same tensor layout as the AVX512 reference and MLX implementation:

- Five ternary rows share each base-3 byte: trit 0 is zero, 1 is positive, 2 is negative.
- Eight little-endian uint32 bitplanes have shape `[ceil(N/5), 8, K/32]`.
- Float16 scales have shape `[ceil(N/5), K/128, 5]`; padded rows have scale zero.
- Every matrix is expanded back to the original Q2 codes and checked bit-for-bit during conversion.
- Embedding gathers retain their original Q2 layout. A tied output head gets a separate T8 copy.

T8 stores 1.725 bits per weight including scales, before row padding. The existing ONNX Q2 format uses
2.266 bits per weight with float32 scales and packed zero points (the source MLX format uses 2.25).
Total model savings depend on the embedding and other unchanged tensors; the 2B tied head requires an additional
copy in T8.

```powershell
python convert_to_t8.py `
  --input C:\models\bonsai-2-2b.onnx `
  --output C:\models\bonsai-2-2b-t8.onnx
```

The Q2-graph re-encoder's output consists of the graph, an `.onnx.data` weight file and a `.validation.json` conversion report.
Existing output files are never overwritten.

Run the result with `run_onnx.py` as above, using an ARM64 build of this branch. The new
`com.microsoft.MatMulTernary` CPU operator uses the T8 gate decoder and NEON masked add/subtract with float32
accumulation; it does not expand a dense weight matrix. Other CPU architectures have a scalar reference path.
Unmodified ONNX Runtime releases do not recognize this operator. This is not an `onnxruntime-genai` model,
and does not add KV/recurrent caching or vision support.

## Comparing T8 and Q2

Use the same newly built runtime for both formats:

```powershell
python benchmark_t8.py --mode model `
  --model-dir C:\models\bonsai-2-2b `
  --q2 C:\models\bonsai-2-2b.onnx `
  --t8 C:\models\bonsai-2-2b-t8.onnx `
  --threads 8 --iterations 5 --output C:\models\bonsai-2-2b-benchmark.json

python benchmark_t8.py --mode matmul `
  --model-dir C:\models\bonsai-2-27b `
  --rows 1 8 24 --threads 8 --iterations 5 `
  --output C:\models\bonsai-2-27b-matmul-benchmark.json
```

The model benchmark rejects non-finite outputs and checks logit agreement (`rtol=3e-4`, `atol=3e-3`), identical greedy next tokens and
KL divergence at most `1e-5`. Timings exclude loading and warmup, alternate Q2/T8 order and report medians
and all samples. Both formats use float32 activations, CPU EP and `ORT_ENABLE_ALL`; Q2 uses the existing
`accuracy_level=0` behavior, not an int8 activation approximation.
Thread-pool spinning is disabled so the inactive session does not compete with the session being measured.

The reported model steps/second are **fixed-window recomputation**, not cached decode throughput or prefill
tokens/second. The matrix benchmark uses actual checkpoint weights for the MLP, recurrent projection and output head,
excluding Hadamard transforms equally from both formats.

### Measured Windows ARM64 results

Snapdragon X X1E80100, native ARM64 ONNX Runtime 1.32.0 built from this branch, CPU EP,
8 threads, float32 activations, sequence length 24, 21 prompt tokens. Each format received
3 warmups and 7 alternating timed iterations; numbers below are medians, excluding session loading.
Sizes include the graph and all referenced external data and use decimal GB.

| Model | Q2 size | T8 size | Q2 step | T8 step | Q2/T8 speedup |
|---|---:|---:|---:|---:|---:|
| 2B | 0.669 GB | 0.562 GB | 2.807 s | 0.815 s | 3.44x |
| 27B | 7.682 GB | 6.002 GB | 93.207 s | 60.367 s | 1.54x |

These are fixed-window recomputation steps, **not cached tokens/second**. The 27B timed samples ranged
from 88.554 to 98.779 seconds for Q2 and 58.974 to 62.179 seconds for T8; earlier runs varied substantially,
so compare the alternating samples within a run rather than treating absolute timings as hardware guarantees.
Maximum Q2/T8 logit differences were `1.12e-5` (2B) and `4.33e-4` (27B), with matching greedy tokens.
All 151 matrices in 2B and 401 matrices in 27B passed packed-code round-trip validation.
T8 matrix storage fell about 23.8%; whole-model savings were 16.0% and 21.9%, respectively.

The actual-weight matrix benchmark also tested MLP up/down, recurrent QKV, and output-head matrices
at `M=1,8,24` for both model sizes (24 cases). All numerical checks passed; the largest absolute output
difference was `1.20e-6`. These matrix timings exclude Hadamard transforms and should not be substituted
for end-to-end generation throughput.

### Validation scope

The ARM64 tests cover every valid T8 byte and lane, padding, dynamic weights, batched activations, invalid
encodings, and both 16-head and 48-head recurrent projections. Nonzero recurrent outputs are also compared with
the independent PyTorch gated-delta reference. Conversion verifies the packed codes of every matrix.

Full-model validation additionally compared both ONNX formats with eager PyTorch Qwen3.5 using the checkpoint's
original weights dequantized in float32. This checks the adapted language-only converter, not native MLX inference,
multimodal behavior, a model-quality evaluation, or cached generation. On the France prompt above, the fresh 2B
graphs and eager reference return `Paris`; the 27B graphs and eager reference all return `afs`. The latter is not a
correct answer, and format agreement must not be interpreted as validation of the 27B model's answer quality.

## Experimental runtime weight layouts

The default `MatMulTernary` kernel remains unchanged. Two opt-in CPU prepacking paths consume the same
T8 v1 graphs; no model conversion or on-disk format change is required:

- `session.ternary.prepack=current` (default): original plane-major layout and kernel.
- `session.ternary.prepack=tiled`: reorder each 5-row/128-input group into eight adjacent NEON vectors,
  `[row_unit, K/128, 8, 4]`. Packed weights remain 1.725 bits/weight including FP16 scales, before row padding.
- `session.ternary.prepack=masks`: decode once into nonzero/sign bitplanes,
  `[row_unit, K/128, 5, 2, 4]`. Packed weights use 2.125 bits/weight including scales, about 23.2% more than T8.

For either experimental layout, `session.ternary.row_tile=2` shares decoding and mask extraction between
two activation rows; the default is `1`. The current kernel only supports row tile `1`.
Constant planes are packed through ONNX Runtime's initialization-time `PrePack` hook. Dynamic planes, or
sessions with prepacking disabled, are packed per invocation instead, and that work is included in inference.
The packed sizes above describe weight buffers, not measured total session memory.

```python
settings = onnxruntime.SessionOptions()
settings.add_session_config_entry("session.ternary.prepack", "masks")
settings.add_session_config_entry("session.ternary.row_tile", "2")
```

Compare both layouts, with and without two-row reuse, against the current kernel:

```powershell
python benchmark_t8_layouts.py --mode matmul `
  --model-dir C:\models\bonsai-2-2b --rows 1 8 24 --threads 8 `
  --warmup 3 --iterations 10 --output C:\models\layout-matmul-2b.json

python benchmark_t8_layouts.py --mode model `
  --model-dir C:\models\bonsai-2-2b --model C:\models\bonsai-2-2b-t8\model.onnx `
  --threads 8 --warmup 1 --iterations 5 --output C:\models\layout-model-2b.json
```

The layout benchmark requires bit-identical FP32 outputs, rotates measurement order, disables thread-pool
spinning, and records every sample plus initialization times. Packing is excluded from timed inference for
constant weights. Matrix tests use the checkpoint's actual MLP up/down, recurrent QKV, and output-head weights;
model timings remain fixed-window recomputation rather than cached decoding.

### Measured layout results

On Snapdragon X X1E80100, Windows ARM64, eight CPU threads and FP32 activations, all experimental outputs
were bit-identical to the current kernel: 24 actual-weight matrix shapes and both complete direct-T8 models.
The 14 native regression tests and 12 Python tests also passed, including dynamic weights, disabled prepacking,
empty activations, padding and odd two-row tails.

Whole-model medians below use a 24-token window containing the same 21-token prompt. The 2B run used one warmup
and five timed samples per variant; the more expensive 27B run used one warmup and three timed samples and
tested only the two-row variants against current. Speedups refer to the current kernel in the same run,
not the earlier Q2/T8 measurements.

| Runtime variant | 2B median (ms) | 2B speedup | 27B median (ms) | 27B speedup |
|---|---:|---:|---:|---:|
| Current | 3026.087 | 1.000x | 49050.671 | 1.000x |
| Tiled, one row | 3108.241 | 0.974x | Not measured | Not measured |
| Masks, one row | 2711.846 | 1.116x | Not measured | Not measured |
| Tiled, two-row reuse | 2200.528 | 1.375x | 34350.285 | 1.428x |
| Masks, two-row reuse | 2056.497 | 1.471x | 32872.387 | 1.492x |

For 27B, the timed ranges were 45.966-49.954 s (current), 34.014-35.171 s (tiled/two-row) and
30.487-34.064 s (masks/two-row). Matrix samples also showed substantial timing variation; small differences
around 1.00x should not be interpreted as a reliable gain.

Matrix results are geometric means of the four actual-weight matrices' current/variant median ratios,
with three warmups and ten timed samples per variant:

| Checkpoint | Activation rows M | Tiled | Masks | Tiled/two-row | Masks/two-row |
|---|---:|---:|---:|---:|---:|
| 2B | 1 | 1.036x | 1.066x | 0.952x | 1.068x |
| 2B | 8 | 1.039x | 1.112x | 1.481x | 1.585x |
| 2B | 24 | 0.995x | 1.122x | 1.417x | 1.524x |
| 27B | 1 | 1.021x | 1.129x | 0.985x | 1.075x |
| 27B | 8 | 1.016x | 1.125x | 1.427x | 1.513x |
| 27B | 24 | 1.006x | 1.153x | 1.473x | 1.494x |

Two-row reuse provides the main improvement for prefill; it offers no reuse when M=1. Tiling alone did not
give a consistent speedup. Prefer tiled/two-row when memory is important, or masks/two-row for the best
measured complete-model latency. Neither is enabled by default.

Tiled packing retains the encoded plane size. Masks require 25% more plane-buffer memory: an additional
94,041,088 bytes for the direct 2B model (about 89.7 MiB), or 1,280,012,288 bytes for direct 27B (about 1.19 GiB).
This is not a total resident-memory measurement. On-disk graphs and weights remain unchanged.
Initialization times are recorded, but their fixed session-creation order and filesystem cache state mean
they are not an isolated packing-cost benchmark.

The raw reports are `layout-matmul-2b.json`, `layout-matmul-27b.json`, `layout-model-2b.json` and
`layout-model-27b.json` in the local model-output directory. To reproduce the reduced 27B model run:

```powershell
python benchmark_t8_layouts.py --mode model `
  --model-dir C:\models\bonsai-2-27b --model C:\models\bonsai-2-27b-t8\model.onnx `
  --variants current tiled_reuse2 masks_reuse2 --threads 8 `
  --warmup 1 --iterations 3 --output C:\models\layout-model-27b.json
```
