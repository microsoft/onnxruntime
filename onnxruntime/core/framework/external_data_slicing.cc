// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_slicing.h"

#include <algorithm>
#include <cstring>

#include "core/common/safeint.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace utils {

namespace {

// Element size in bytes for non-packed dtypes. For packed INT4/UINT4 returns 0
// and the caller must use packed-aware sizing (2 elements per byte).
size_t ElementByteSizeOrZeroForPacked(int32_t data_type) {
  using ONNX_NAMESPACE::TensorProto;
  switch (data_type) {
    case TensorProto::FLOAT:
    case TensorProto::INT32:
    case TensorProto::UINT32:
      return 4;
    case TensorProto::DOUBLE:
    case TensorProto::INT64:
    case TensorProto::UINT64:
    case TensorProto::COMPLEX64:
      return 8;
    case TensorProto::COMPLEX128:
      return 16;
    case TensorProto::FLOAT16:
    case TensorProto::BFLOAT16:
    case TensorProto::INT16:
    case TensorProto::UINT16:
      return 2;
    case TensorProto::INT8:
    case TensorProto::UINT8:
    case TensorProto::BOOL:
#if !defined(DISABLE_FLOAT8_TYPES)
    case TensorProto::FLOAT8E4M3FN:
    case TensorProto::FLOAT8E4M3FNUZ:
    case TensorProto::FLOAT8E5M2:
    case TensorProto::FLOAT8E5M2FNUZ:
#endif
      return 1;
    case TensorProto::INT4:
    case TensorProto::UINT4:
      return 0;  // packed
    default:
      return 0;  // unsupported here
  }
}

bool IsPackedInt4(int32_t data_type) {
  return data_type == ONNX_NAMESPACE::TensorProto::INT4 ||
         data_type == ONNX_NAMESPACE::TensorProto::UINT4;
}

// Computes the linear element count of a shape (product of dims) using SafeInt
// to surface overflow as an error.
common::Status ElementCount(gsl::span<const int64_t> shape, int64_t& out) {
  SafeInt<int64_t> total = 1;
  for (auto d : shape) {
    if (d <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "shape dim must be > 0, got ", d);
    }
    total *= d;
  }
  out = total;
  return common::Status::OK();
}

// Computes the byte size of a tensor with `shape` for `data_type`. Handles INT4/UINT4 packing.
common::Status TensorByteSize(int32_t data_type, gsl::span<const int64_t> shape, size_t& out_bytes) {
  int64_t elems = 0;
  ORT_RETURN_IF_ERROR(ElementCount(shape, elems));
  if (IsPackedInt4(data_type)) {
    // ceil(elems / 2)
    SafeInt<size_t> packed = static_cast<size_t>(elems);
    packed = (packed + 1) / 2;
    out_bytes = packed;
    return common::Status::OK();
  }
  const size_t elem_size = ElementByteSizeOrZeroForPacked(data_type);
  if (elem_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "ReadSlicedExternalTensor: unsupported data_type ", data_type);
  }
  SafeInt<size_t> bytes = static_cast<size_t>(elems);
  bytes *= elem_size;
  out_bytes = bytes;
  return common::Status::OK();
}

}  // namespace

common::Status ComputeSourceTensorByteSize(const ExternalDataInfo& info,
                                           int32_t data_type,
                                           size_t& out_bytes) {
  if (!info.HasSliceSpec()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ComputeSourceTensorByteSize requires a slice spec");
  }
  return TensorByteSize(data_type, info.GetSourceShape(), out_bytes);
}

common::Status ReadSlicedExternalTensor(const Env& env,
                                        const std::filesystem::path& file_path,
                                        const ExternalDataInfo& info,
                                        int32_t data_type,
                                        gsl::span<const int64_t> tensor_proto_dims,
                                        gsl::span<char> dst_buffer) {
  if (!info.HasSliceSpec()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ReadSlicedExternalTensor called without a slice spec");
  }

  const auto& source_shape = info.GetSourceShape();
  const auto& slice_starts = info.GetSliceStarts();
  const auto& slice_sizes = info.GetSliceSizes();
  const size_t rank = source_shape.size();

  // Slice rank/values were validated by ExternalDataInfo::Create. Re-check the bits
  // that depend on the caller-supplied tensor_proto_dims and dtype.
  if (tensor_proto_dims.size() != rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReadSlicedExternalTensor: TensorProto.dims rank ",
                           tensor_proto_dims.size(), " does not match source_shape rank ", rank);
  }
  for (size_t d = 0; d < rank; ++d) {
    if (tensor_proto_dims[d] != slice_sizes[d]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ReadSlicedExternalTensor: TensorProto.dims[", d, "]=",
                             tensor_proto_dims[d], " does not match slice_sizes[", d, "]=",
                             slice_sizes[d]);
    }
  }

  const bool packed = IsPackedInt4(data_type);
  const size_t elem_size = packed ? 0 : ElementByteSizeOrZeroForPacked(data_type);
  if (!packed && elem_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "ReadSlicedExternalTensor: unsupported data_type ", data_type);
  }

  // For packed dtypes the source's innermost dim must be even so each "row" is
  // a whole number of bytes. Slicing the innermost dim is allowed only when
  // both `slice_starts[innermost]` and `slice_sizes[innermost]` are even, so
  // the slice begins and ends on a byte boundary (no bit-level shifting
  // needed). Outer-axis-only slicing always satisfies this trivially.
  if (packed) {
    if (source_shape[rank - 1] % 2 != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "ReadSlicedExternalTensor: packed INT4/UINT4 source_shape's innermost dim must be even, got ",
                             source_shape[rank - 1]);
    }
    const bool inner_whole = (slice_starts[rank - 1] == 0 &&
                              slice_sizes[rank - 1] == source_shape[rank - 1]);
    if (!inner_whole) {
      if ((slice_starts[rank - 1] % 2) != 0 || (slice_sizes[rank - 1] % 2) != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "ReadSlicedExternalTensor: slicing the innermost dim of packed INT4/UINT4 tensors "
                               "requires both slice_start and slice_size to be even (byte-aligned). "
                               "Source innermost dim=", source_shape[rank - 1],
                               ", slice_start=", slice_starts[rank - 1],
                               ", slice_size=", slice_sizes[rank - 1]);
      }
    }
  }

  // Find the largest contiguous inner block: largest k such that for all d in [k, rank),
  // slice_starts[d] == 0 AND slice_sizes[d] == source_shape[d]. The chunk we copy per
  // iteration spans dims [k, rank).
  //
  // Special case: if the innermost dim is only partially sliced (allowed for packed
  // INT4/UINT4 when byte-aligned, see the guard above), the chunk MUST include that
  // partial innermost dim -- but cannot extend to outer fully-taken dims, because the
  // gap between rows breaks source contiguity. We bias k to never exceed rank-1 in
  // that case so the chunk = exactly `slice_sizes[rank-1]` elements per iteration.
  size_t k = rank;
  while (k > 0) {
    const size_t d = k - 1;
    if (slice_starts[d] == 0 && slice_sizes[d] == source_shape[d]) {
      --k;
    } else {
      break;
    }
  }
  if (k == rank && rank > 0) {
    // Innermost dim is partially sliced. Make the chunk cover just that (partial)
    // innermost dim; outer dims are iterated.
    k = rank - 1;
  }

  // Compute byte size of one chunk (= contiguous span across dims [k, rank)).
  int64_t chunk_elems_i64 = 1;
  for (size_t d = k; d < rank; ++d) {
    SafeInt<int64_t> v = chunk_elems_i64;
    v *= slice_sizes[d];
    chunk_elems_i64 = v;
  }
  size_t chunk_bytes;
  if (packed) {
    // For packed dtypes the chunk always spans the innermost dim (possibly partially).
    // The guard above ensures slice_sizes[rank-1] is either the full source dim
    // (which is required-even) OR is itself even (byte-aligned partial slice). So
    // chunk_elems is even and chunk_bytes is exact.
    chunk_bytes = static_cast<size_t>(chunk_elems_i64 / 2);
  } else {
    SafeInt<size_t> cb = static_cast<size_t>(chunk_elems_i64);
    cb *= elem_size;
    chunk_bytes = cb;
  }

  // Per-dim byte strides into the SOURCE tensor (row-major). Inner-most stride first
  // for non-packed; for packed the innermost dim is whole so we skip it.
  // stride_bytes[d] = product(source_shape[d+1..rank-1]) * elem_size_bytes.
  // For packed: stride_bytes[d] = product(source_shape[d+1..rank-1]) * (1/2 byte per elem) -- but since
  // packed innermost dim is whole, our outer-loop iteration never multiplies a stride that
  // sits "inside" the packed dim. The simplest implementation: compute strides in ELEMENTS
  // and convert chunk-start offsets to bytes once per chunk.
  std::vector<int64_t> source_elem_strides(rank, 1);
  for (size_t d = rank; d > 1;) {
    --d;
    SafeInt<int64_t> s = source_elem_strides[d];
    s *= source_shape[d];
    if (d == 0) break;
    source_elem_strides[d - 1] = s;
  }

  // Total bytes we'll write must equal dst_buffer.size() exactly.
  size_t dst_total_bytes = 0;
  ORT_RETURN_IF_ERROR(TensorByteSize(data_type, slice_sizes, dst_total_bytes));
  if (dst_buffer.size() != dst_total_bytes) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ReadSlicedExternalTensor: dst buffer size ", dst_buffer.size(),
                           " does not match slice byte size ", dst_total_bytes);
  }

  // Source-tensor start in the file (bytes). The optional `length` (info.GetLength()) was
  // already validated by the caller (if non-zero) to equal the source byte size.
  const FileOffsetType source_byte_offset = info.GetOffset();

  // Outer iteration over dims [0, k). For each multi-index (i_0, ..., i_{k-1}), the
  // source element index in dim d is (slice_starts[d] + i_d). Convert to a byte
  // offset using source_elem_strides; for packed dtypes, divide by 2 at the end
  // (guaranteed exact since the innermost packed dim is whole and even).
  std::vector<int64_t> idx(k, 0);

  // Pre-compute element offset contributions from slice_starts in ALL dims (not just
  // the iterated dims [0, k)). For fully-taken dims (d in [k, rank)) slice_starts[d]
  // is 0 so the contribution is 0; for a partially-sliced innermost dim
  // slice_starts[rank-1] is non-zero and MUST be folded into the base offset since
  // the outer loop never touches it.
  //
  // We'll maintain `current_source_elem_offset` incrementally to avoid an
  // O(rank) recompute per chunk.
  int64_t current_source_elem_offset = 0;
  for (size_t d = 0; d < rank; ++d) {
    SafeInt<int64_t> add = source_elem_strides[d];
    add *= slice_starts[d];
    current_source_elem_offset = static_cast<int64_t>(SafeInt<int64_t>(current_source_elem_offset) + add);
  }

  size_t dst_cursor = 0;
  // Total chunk count = product(slice_sizes[0..k-1]). When k == 0 the slice IS one chunk.
  while (true) {
    // Compute byte offset of this chunk within the source tensor and read it.
    int64_t src_byte_in_tensor;
    if (packed) {
      // Exact: current_source_elem_offset is always a multiple of source_shape[rank-1] which is even.
      src_byte_in_tensor = current_source_elem_offset / 2;
    } else {
      SafeInt<int64_t> b = current_source_elem_offset;
      b *= static_cast<int64_t>(elem_size);
      src_byte_in_tensor = b;
    }

    SafeInt<FileOffsetType> read_offset = source_byte_offset;
    read_offset += src_byte_in_tensor;

    if (dst_cursor + chunk_bytes > dst_buffer.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ReadSlicedExternalTensor: internal error, chunk overrun");
    }

    ORT_RETURN_IF_ERROR(env.ReadFileIntoBuffer(
        file_path.c_str(),
        read_offset,
        chunk_bytes,
        gsl::make_span(dst_buffer.data() + dst_cursor, chunk_bytes)));

    dst_cursor += chunk_bytes;

    // Increment outer multi-index (row-major: rightmost-first).
    if (k == 0) {
      break;
    }
    bool overflow_all = true;
    for (size_t d = k; d-- > 0;) {
      ++idx[d];
      // Bump source elem offset by one source stride at this dim.
      current_source_elem_offset = static_cast<int64_t>(
          SafeInt<int64_t>(current_source_elem_offset) + source_elem_strides[d]);
      if (idx[d] < slice_sizes[d]) {
        overflow_all = false;
        break;
      }
      // Wrap this dim back to 0: subtract slice_sizes[d] * stride[d] from offset.
      SafeInt<int64_t> sub = source_elem_strides[d];
      sub *= slice_sizes[d];
      current_source_elem_offset = static_cast<int64_t>(
          SafeInt<int64_t>(current_source_elem_offset) - sub);
      idx[d] = 0;
    }
    if (overflow_all) {
      break;
    }
  }

  if (dst_cursor != dst_buffer.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReadSlicedExternalTensor: wrote ", dst_cursor,
                           " bytes but expected ", dst_buffer.size());
  }

  return common::Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
