# Direct shared INT4, INT3, and INT2 QMoE layouts

CPU and CUDA QMoE can consume the same immutable weight and scale banks for the
integer profiles below. CPU QMoE calls reusable projection kernels that can also
serve a later offload worker. This change supplies the layout and kernel
foundation; session placement, file-cache policy, and asynchronous offload are
separate work.

| Property | SM80 INT4 | Row-local INT3 | SM80 INT2 |
| --- | --- | --- | --- |
| `quant_type` | `int` | `int` | `int` |
| `expert_weight_bits` | 4 | 3 | 2 |
| `weights_prepacked` | Explicit `1` | Explicit `0` | Explicit `1` |
| Shared CPU/CUDA block sizes | 32 | 32, 64, 128 | 32, 64, 128 |
| Weight code interpretation | Unsigned nibble minus 8 | Unsigned three-bit code minus 4 | Unsigned two-bit code minus 2 |
| Physical weight ordering | SM80 `fpA_intB`, `ColumnMajorTileInterleave<64,4>` | LSB-first bit stream, independently packed for every output row | SM80 `fpA_intB`, `ColumnMajorTileInterleave<64,8>` |
| Serialized scale ordering | `[E,N,K/block_size]` | `[E,N,K/block_size]` | `[E,N,K/block_size]` |
| Activation/output dtype | FP16 or BF16; CPU also supports FP32 | FP16 or BF16 | FP16 or BF16; CPU also supports FP32 |

These operator profiles use uniform FC1/FC2 widths, `activation_type="swiglu"`,
`swiglu_fusion=1`, and no sparse mixer. Separate FC3, explicit zero points, global
scales, and activation scales are outside the shared operator profile. Optional
FC1/FC2 biases and `router_weights` retain their mathematical meanings. Supplied
`router_weights` select contribution weights directly; they do not change which
experts are selected from `router_probs`.

## Logical dimensions and physical storage

FC1 is logically `[E,2*I,H]` and FC2 is `[E,H,I]`. Scale tensors identify these
dimensions independently of weight shape aliases. INT4 accepts both canonical
`[E,N,K/2]` metadata and the existing CUDA `[E,K,N/2]` metadata; both identify the
same SM80 byte payload. INT4 requires `K` divisible by 64 and `N` divisible by
four. Its nominal shape does not describe its linear byte addressing.

INT3 weights are `[E,N,ceil(3*K/8)]`, with low-order codes first in each row.
Eight codes occupy three bytes. The supported block sizes ensure byte-aligned
rows and complete quantization groups. Scales remain in source row order; GPU
slots do not require a scale transpose.

For the candidates with `E=512,H=2560,I=640`, the source banks are:

| Bank | INT4 block32 shape | INT3 block64 shape | INT2 block64 shape |
| --- | --- | --- | --- |
| FC1 weights | `[512,2560,640]`, SM80 payload | `[512,1280,960]` | `[512,2560,320]`, SM80 payload |
| FC1 scales | BF16 `[512,1280,80]` | BF16 `[512,1280,40]` | BF16 `[512,1280,40]` |
| FC2 weights | `[512,640,1280]`, SM80 payload | `[512,2560,240]` | `[512,640,640]`, SM80 payload |
| FC2 scales | BF16 `[512,2560,20]` | BF16 `[512,2560,10]` | BF16 `[512,2560,10]` |
| Bytes per expert | 2,764,800 | 1,996,800 | 1,382,400 |

The inspected `qmoe_top16_44_33_b64` candidate contains 16 INT4 and 32 INT3
layers, and the `qmoe_2b375_x02_y14_z32_dp23_b64` candidate contains 2 INT4,
14 INT3, and 32 INT2 layers. All use top-10 routing. No model or external-data
rewrite is required.

## CPU shared-layout kernel interface

`qmoe_shared_layout.h` describes immutable expert views (`Int4ExpertView`,
`Int3ExpertView`, `Int2ExpertView`, `Nvfp4ExpertView`) and range kernels
(`Int4Fc1SwiGluRange`, `Int4Fc2Range`, `Int3Fc1SwiGluRange`, `Int3Fc2Range`,
`Int2Fc1SwiGluRange`, `Int2Fc2Range`, `Nvfp4Fc1SwiGluRange`, `Nvfp4Fc2Range`)
implemented across `qmoe_shared_layout.cc`,
`qmoe_shared_layout_int4_avx512.cc`, `qmoe_shared_layout_int4_avx2.cc`,
`qmoe_shared_layout_int2_avx512.cc`, `qmoe_shared_layout_int2_avx2.cc`,
`qmoe_shared_layout_int3_avx512.cc`, `qmoe_shared_layout_int3_avx2.cc`, and
`qmoe_shared_layout_fp4_avx512.cc`.
It performs no weight/scale conversion during initialization or execution; the
caller owns activation and output workspace. INT2 applies block scales before
activation products. Extreme BF16 scales use wider scalar intermediates to avoid
overflow before multiplication by small activations. INT2 rounds FC1 projections to
the boundary dtype before adding bias and evaluating SwiGLU in FP32, matching CUDA
decode and packed prefill. FC1 ranges must contain complete gate/up pairs (even
bounds); invalid ranges return false without writing output.

Scalar, runtime-dispatched AVX2, and runtime-dispatched AVX-512 implementations
read the same banks. Code extraction, activation permutation, scale decoding,
and boundary rounding happen inside the consuming kernel. No alternate weight
matrices or panels are materialized.

## Integer weight scaling

The session option `ep.cpu.qmoe_deferred_scale` currently applies only to the
shared-layout CPU INT3/INT4 kernels, including both projections. It does not
affect NVFP4/MXFP4, legacy CPU QMoE kernels, or CUDA kernels.

- `"0"` (default): round each dequantized weight to the activation type before
  multiplying by the activation, then accumulate in FP32.
- `"1"`: accumulate decoded integer codes times activations in FP32, then apply
  the block scale without rounding each weight. This can improve throughput and
  preserve more dequantization precision, but results can differ substantially
  under cancellation and when experts move between CPU and GPU.

```python
options = onnxruntime.SessionOptions()
options.add_session_config_entry("ep.cpu.qmoe_deferred_scale", "1")
```

This controls weight dequantization only. Bias handling, activation/output
rounding, and SwiGLU projection boundaries are independent of this option.
Low-level CPU offload callers select the same policy through
`Int3ExpertView::deferred_scale` or `Int4ExpertView::deferred_scale`.

## Validation

- `qmoe_shared_layout_test.cc` checks independent layout fixtures, scalar, AVX2,
  and AVX-512 results, candidate dimensions, byte extents/strides, and read-only
  guarded buffers. INT4 FP16 boundary tests compare F16C rounding against the
  scalar path for partial ranges, including vector tails and multiple rows.
- `moe_test.cc` checks prepacked INT2 CPU operator execution against independently
  packed multi-expert banks and a natural-order reference, including both weight
  shapes, FP32/FP16/BF16, optional biases, and explicit routing weights used as supplied, independently of normalization settings.
- `test_qmoe_shared_direct.py` checks CPU/CUDA operator execution, dynamic banks,
  prepacking on/off, non-square aliases, mixed INT4/INT3 graphs, and 512-expert
  top-10 routing.
- Set `ORT_QMOE_CANDIDATE_MODEL` to the candidate's `model.onnx` to test exact
  expert slices without loading the entire external-data package.
- With a CUPTI-enabled build, set `ORT_QMOE_REQUIRE_CUDA_PROFILE=1` and
  `ORT_QMOE_REQUIRE_CUDA=1` to require direct device-kernel observations and reject
  repack, transpose, and dense-dequantization launches in the tested path.

Run Python tests against the freshly built runtime from a directory outside the
source package. Explicit CUDA tests disable CPU fallback.
