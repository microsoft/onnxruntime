# Packed-Format P0 References

Standalone, CPU-only test/reference modules. They are not a production Python API,
QMoE schema, kernel, exporter, or claim of existing QMoE ABI support. Generic
1..8-bit format utilities do not enable those widths in any operator.

The specification source is the P0 and Section 2/7.3 contract in
`docs/design/int3-qmoe-end-to-end-delivery-plan.md` at commit
`2b1185e24a5a79c56fd01226ad81b016450d0ef3`.

## Contents

- `reference.py`: row-local LSB-first unsigned uint8 codecs, checked sizes,
  separate canonical-padding validation, the proposed INT3 offset-4 numerical
  profile, and deterministic fixture RTN.
- `golden_vectors.json`: handwritten byte and numerical literals, including
  row/expert boundaries, packed block zero points, and a tiny predetermined-route
  SwiGLU MoE with interleaved gate/up FC1 and independently oriented FC2.
- `onnx_reference.py`: fixed-shape standard opset-18 unpack and FLOAT decode
  graphs, with per-row sentinel padding and explicit empty-axis shapes.
- `test_reference.py`: independent per-bit scalar codecs, reproducible randomized
  checks, fixed-vector regressions, malformed-input tests, and ONNX checker plus
  ReferenceEvaluator comparisons. No source from the draft proposal is used.

## Run

Requires `python3`, `numpy`, `onnx`, and `pytest`. Optional `ml_dtypes` enables
BF16 tests; only those tests skip when it is unavailable. No installed ORT or GPU
is needed. From this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --rootdir=. --confcutdir=. test_reference.py
```

The collection boundaries prevent pytest from importing the enclosing native
ORT Python package. The modules are intentionally imported directly from this
standalone directory rather than through `onnxruntime`.

## Numerical Contract

Generic unpack requires an exact byte count but ignores high padding bits.
`validate_canonical_padding` checks those bits separately. The INT3 validator
requires rank-3 weights, matching FP32/FP16/available BF16 block scales, finite
nonnegative scales, block sizes 32/64/128, canonical weight and zero-point
padding, and all-4 explicit zero-point codes. Weights unpack with count K; packed
zero points unpack with count B = ceil(K/block_size). Reference-only tails and
empty dimensions are accepted without claiming current operator support.

INT3 decoding casts unsigned codes to FLOAT before subtraction and gathers scales
and explicit zeros with floor(k/block_size). The ONNX graphs express that
decomposition, not input validation: call `validate_int3` before evaluating a
profile graph. Graph inputs have fixed checked shapes; they are semantic oracles
for small fixtures, not production weight materialization or dynamic-shape APIs.

RTN uses float64 reductions/division, rounds scales to the requested final storage
dtype first, then performs nearest-even rounding and signed clipping to [-4,3].
Zero and scale-underflow blocks produce scale 0 and code 4. `RtnResult` records
underflow flags and per-block maximum absolute reconstruction error in float64.
Nonfinite weights and scale-storage overflow are rejected without dtype promotion.

## External Interoperability

The optional `test_external_proposal_interoperability` compares 56 width/shape
cases with the actual Pack/Unpack Python references from ONNX PR #8576 at revision
`0a15c0c82254ffabf2f5675005cc5cfc8086d9b8`. It is skipped by default, requires an
explicit trusted checkout, and neither downloads nor vendors proposal source.
Set `ONNX_PACK_PROPOSAL_CHECKOUT` to that clean pinned checkout and run the same
test command. The test checks HEAD; keep the reference source unmodified.
This tests the proposal's Python decoder, not its draft operator in released ORT.

## Checkpoint Representation Audit

The local Olive audit used revision
`a9290ab9bf668cae5a02ad7473c8cb0f5476d35e` on 2026-10-15. At that revision,
`olive/common/quant/utils.py` restricts `WeightQuantizer` and `pack_to_uint8` to
2/4/8 bits, and the byte packer uses `packing_factor = 8 // bits`. The local
GPTQ pass imports that `WeightQuantizer`. `olive/constants.py` does not expose
a three-bit `PrecisionBits` option. These inspected paths therefore do not
establish a native INT3 checkpoint source compatible with this proposal.

No INT3 checkpoint converter is claimed here. A future importer must establish
the source code mapping, bit/word order, tensor orientation, group indexing,
scales, zero points, and any permutation before converting to canonical bytes.
It must verify gate/up interleaving and FC2 orientation against shared fixtures,
not simply relabel checkpoint bytes. External GPTQ implementations and GGUF
Q3_K/IQ3 representations were not audited; unknown bytes must not be reused.
This is a scoped audit of local Olive paths, not a claim that no external
quantizer supports INT3.

## Scope Gaps

This package does not prove existing provider behavior, modify or validate real
QMoE schemas, or qualify accuracy/performance. Semantic agreement comes from
manual literals, independently implemented scalar codecs, standard-operator
graphs, and optional pinned external reference checks. Owner approval and format
freeze, checkpoint conversion/exporter integration, and provider qualification
remain separate delivery gates. This PR supplies P0 review tools and evidence;
it does not declare the reviewed format frozen or P0 jointly approved.