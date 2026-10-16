# OpenVLA H100 CPU/CUDA drift investigation

This is an investigation, **not a verified CUDA kernel fix**. The component
diagnostic below does not replace an independent original-model reference or
reproduce the complete three-component qualification by itself.

## Observed full-model discrepancy

The FP32 ONNX OpenVLA package exported by
[`rui-ren/mobius@f049872`](https://github.com/rui-ren/mobius/tree/f049872a1e240a360e9120e606abf5fe2a75c627) runs through
ORT 1.31.0 (`402979ac1`) on one H100 80 GB HBM3, SM90, Linux. TF32 is disabled
and graph optimizations are `ORT_ENABLE_ALL`. Against the pinned original CPU
fixture, eight of 8,849,664 prefill logits exceed the combined
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` criterion.
Maximum absolute error is `0.00018551945686340332`.

Separate full-model experiments produced the following observations. Their
weights, request tensors and independent fixture are not bundled with this PR;
the component diagnostic below does not independently reproduce these counts.

| Control | Prefill failures against unchanged original CPU fixture |
|---|---:|
| Unchanged ORT CUDA package | 8 |
| Unchanged original-class Torch CUDA FP32, TF32 disabled | 7 |
| Vision MATH attention instead of efficient attention | 4 |
| Source Q-prescaling plus primitive attention | 3 |
| Frozen source embeddings supplied to the decoder | 0 |

The eight failing coordinates are `[0,14,31913]`, `[0,49,31815]`,
`[0,49,31849]`, `[0,119,31799]`, `[0,122,31809]`, `[0,122,31817]`,
`[0,243,31854]` and `[0,254,31936]`. They are image-token positions.
Matching the seven final actions does not clear the all-prefill numerical gate.

This does not establish an ORT-specific defect: original Torch CUDA also differs
from the frozen CPU fixture. It motivates locating the first vision divergence
with behavior-preserving captures and then comparing identical operator operands.

Neither fused `SkipLayerNormalization` nor `EmbedLayerNormalization` from
[microsoft/onnxruntime#33357](https://github.com/microsoft/onnxruntime/pull/33357)
executes in the observed package. It has 98 distinct ordinary vision LayerNorms
and 65 decoder RMSNorms. Replaying 102 original-source LayerNorm operand sets
through ordinary CUDA LayerNormalization passes the same combined arithmetic
bounds against centered FP64 calculations (maximum absolute error
`1.6502302599974428e-05`). These are source operands, not qualified captures from
the ORT candidate. Final-projection local rounding at the failed coordinates is
at most `2.6945027893354734e-06`, too small to explain the full discrepancy.
FP64 was only an offline arithmetic oracle, not an inference substitution.

## Component diagnostic

Use a GPU-enabled ORT environment with NumPy. Select and coordinate exclusive
use of one otherwise idle physical GPU externally; this script is not a resource
allocator. It neither installs drivers nor silently switches to another device.

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python onnxruntime/python/tools/transformers/models/openvla/compare_provider_outputs.py \
  --model /absolute/package/vision_encoder/model.onnx \
  --inputs /absolute/vision-inputs.npz \
  --output-dir /absolute/fresh-provider-comparison
```

The NPZ contains exactly the graph's input names, with their original dtypes
and shapes, captured from the request being investigated. Pickle is disabled.
External initializer files must remain alongside the supplied model with their
original relative paths. Pin their hashes separately: the reported graph hash
explicitly excludes external tensor data. No graph or weights are modified.

The script runs the same inputs through CPU and CUDA sessions, with TF32
disabled and `ORT_ENABLE_ALL`, and saves named-output mappings to separate NPY
files, effective provider options, native build information and profile hashes.
CPU is a diagnostic comparator, **not an independent source oracle**.
CPU-placed nodes are explicitly listed for review; inspect their input/output
types to distinguish integer shape work from floating-point fallback.
Unavailable CUDA, initialization fallback or absence of CUDA kernels fails
explicitly. Runtime fallback is disabled.

To inspect intermediates, separately prepare an identical graph with selected
additional outputs, for example using existing activation-debugging tools, and
pass `--instrumented-model /absolute/instrumented/model.onnx`.
The script first requires **every original output to remain bitwise identical
within each provider**. Even a one-ULP or signed-zero change rejects the capture;
instrumentation can change optimization and fusion. Additional comparisons are
suppressed on rejection. Captured output order is the graph's declared output
order, not a proven topological first-failure order.

Exit codes: **0** means the requested diagnostic comparisons are within their
bounds; **2** means comparisons exceed those bounds, not that a kernel defect
is established; **3** rejects behavior-changing instrumentation. Other input or
runtime errors raise explicitly. Reports never confer model qualification.
The default floating-point diagnostic bounds are `atol=rtol=1e-4`; integer and bool
outputs compare exactly. These CLI options do not
change any external frozen qualification policy.

The complete eight-logit failure still requires the pinned three-component
package, unchanged independent fixture and original own-path prefill sequence;
the component command alone does not reproduce it. The unresolved work is a
small matched-input kernel reproducer, not an unverified normalization patch.
Model weights, input tensors and native profiles are not bundled here.
The sanitized [component observation](h100_observation.json) records an actual
run of this tool: all original outputs stayed bitwise unchanged under four
additional taps, and all five CPU/CUDA output comparisons passed. The input to
the first tapped LayerNorm already differed between providers, so those captures
do not establish a normalization defect.

Generated local reports contain file paths and may contain device identifiers.
Review and redact them before publishing. The committed observation excludes
local paths, machine identifiers, weights and input tensors.

## Tests

Using an environment with NumPy, ONNX and ORT installed, run outside the checkout
so its unbuilt `onnxruntime` package does not shadow the installed wheel:

```bash
cd /tmp
python -m unittest discover \
  -s /absolute/onnxruntime/onnxruntime/test/python/transformers \
  -p test_openvla_provider_drift.py -v
```

Tests include real CPU execution with external initializers and an output named
`file`, combined tolerance, exact large integer comparison, signed zero,
nonfinite/mismatched outputs and rejection of behavior-changing captures.
