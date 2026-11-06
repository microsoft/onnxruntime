# Tensor layout and prepacking

**Owns:** Logical versus physical shapes, vector components, buffer views,
quantized storage strides, and packed-weight lifecycle.
**Update when:** Layouts, padding, view accessors, dispatch operands, or prepack
ownership change. Shader reuse belongs in cache-identity.

- Track the logical ONNX shape separately from packed/padded dimensions and
  strides. Validate rank, channels/groups, bias, and output geometry using the
  logical contract. Preserve original shape metadata after prepacking; do not
  reinterpret a permuted filter's dimensions as ONNX dimensions.
- Identify which operand a preparation step transforms and when: constant
  initializer prepack, lazy constant cache, and runtime activation preparation
  have different lifetimes. Check dynamic and disabled-prepack fallbacks,
  session reuse, failure cleanup, and constant ownership through capture.
- Promote rank-1 MatMul operands to matrix views before matrix-oriented
  dispatch; restore the ONNX output rank. Check scalar/broadcast Gemm bias
  independently of matrix rank.
- Vector widths must respect per-group channel boundaries and output layout.
  Packed u32 input does not imply vec4 output. Audit scalar/vec2/vec4 tails,
  odd N/K, rank-specific layouts, bit widths, scales, zero points, and physical
  packed-row strides in every generic and hardware fallback.
- Follow views through `GetByOffset`/`SetByOffset`/`GetByIndices`, owner binding,
  offsets, and segmented bindings. Direct pointer, subgroup-matrix, or atomic
  paths can bypass ordinary accessor offsets. Check byte versus element units,
  alignment, writable aliasing, and view lifetime.
- Validate initializer, dynamic-weight, and disabled-prepack cases with
  observable prepack counts or equivalent path evidence. Cover partial tiles,
  group boundaries, and all newly used row/column waves with distinguishable
  data. Measure initialization separately from steady inference.

Source landmarks: `nn/conv.cc`, `nn/conv_transpose.cc`, `math/matmul.cc`,
`shader_variable.cc`, and contrib `quantization/matmul_nbits*` under the WebGPU
provider roots.
Apply [cache identity](cache-identity.md) when layout/components change codegen;
test distinct specializations in one shared context and both invocation orders.
