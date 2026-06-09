// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace utils {

// Materializes (gathers) a multi-dimensional slice of a tensor stored in an external
// file into a caller-supplied contiguous destination buffer.
//
// The slice spec comes from `info` (GetSourceShape/GetSliceStarts/GetSliceSizes).
// `info.GetOffset()` is the byte offset to the START of the SOURCE (un-sliced) tensor
// in the file.
//
// `tensor_proto_dims` must equal `info.GetSliceSizes()`.
//
// `dst_buffer` must be sized to exactly hold the sliced tensor: for non-packed
// dtypes that's `product(slice_sizes) * sizeof(elem)`; for INT4/UINT4 it is
// `ceil(product(slice_sizes) / 2)` bytes.
//
// For packed INT4/UINT4 dtypes the innermost dim must be sliced as a whole
// (slice_starts.back() == 0 AND slice_sizes.back() == source_shape.back()) and
// source_shape.back() must be even. Otherwise this function returns an error.
common::Status ReadSlicedExternalTensor(const Env& env,
                                        const std::filesystem::path& file_path,
                                        const ExternalDataInfo& info,
                                        int32_t data_type,
                                        gsl::span<const int64_t> tensor_proto_dims,
                                        gsl::span<char> dst_buffer);

// Returns the byte size of the SOURCE tensor described by `info` (un-sliced) for
// the given data type. Used by callers to validate the optional `length` entry
// against the source tensor when a slice spec is present.
common::Status ComputeSourceTensorByteSize(const ExternalDataInfo& info,
                                           int32_t data_type,
                                           size_t& out_bytes);

}  // namespace utils
}  // namespace onnxruntime
