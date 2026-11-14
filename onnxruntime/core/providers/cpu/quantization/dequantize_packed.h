// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <limits>
#include <string>

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cpu/quantization/dequantize_packed_impl.h"

namespace onnxruntime {
namespace dequantize_packed {

struct Attributes {
  template <typename Info>
  explicit Attributes(const Info& info) {
    std::string name;
    ORT_ENFORCE(info.GetAttr("format", &name).IsOK() && FindFormat(name, format),
                "DequantizePacked: unsupported format '", name,
                "'. Use one of the 27 fixed, case-sensitive gguf: identifiers from the ONNX specification.");
    ORT_ENFORCE(info.GetAttrs("shape", shape).IsOK() && !shape.empty(),
                "DequantizePacked: supply a shape attribute with rank at least one.");
    for (const int64_t dim : shape) {
      ORT_ENFORCE(dim >= 0, "DequantizePacked: shape contains negative dimension ", dim,
                  ". Supply nonnegative literal dimensions; inferred dimensions are not supported.");
    }
    ORT_ENFORCE(shape.back() % format.elements == 0,
                "DequantizePacked: last shape dimension ", shape.back(), " must be a multiple of ",
                format.elements, " for ", name, ", including empty tensors.");
    const auto dtype = info.template GetAttrOrDefault<int64_t>("output_dtype", 1);
    ORT_ENFORCE(dtype == 1 || dtype == 10 || dtype == 16 || dtype == 11,
                "DequantizePacked: output_dtype must be FLOAT (1), FLOAT16 (10), BFLOAT16 (16), or DOUBLE (11).");
    output_dtype = static_cast<int>(dtype);
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return;

    int64_t count = 1;
    for (const int64_t dim : shape) {
      ORT_ENFORCE(count <= std::numeric_limits<int64_t>::max() / dim,
                  "DequantizePacked: output element count exceeds int64 range. Reduce the logical shape.");
      count *= dim;
    }
    elements = count;
    blocks = count / format.elements;
    ORT_ENFORCE(blocks <= std::numeric_limits<int64_t>::max() / format.bytes,
                "DequantizePacked: packed byte count exceeds int64 range. Reduce the logical shape.");
    bytes = blocks * format.bytes;
    const size_t element_bytes = output_dtype == 11 ? 8 : (output_dtype == 1 ? 4 : 2);
    ORT_ENFORCE(static_cast<uint64_t>(elements) <= std::numeric_limits<size_t>::max() / element_bytes,
                "DequantizePacked: output allocation exceeds addressable memory. Reduce the logical shape.");
  }

  Status ValidateInput(const TensorShape& input_shape) const {
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 1,
                      "DequantizePacked: x must be a rank-one UINT8 tensor of packed payload bytes.");
    ORT_RETURN_IF_NOT(input_shape[0] == bytes,
                      "DequantizePacked: expected exactly ", bytes, " packed bytes for the supplied format and shape; got ",
                      input_shape[0], ". Supply the tensor payload without padding or container headers.");
    return Status::OK();
  }

  Format format{};
  TensorShapeVector shape;
  int output_dtype = 1;
  int64_t elements = 0;
  int64_t blocks = 0;
  int64_t bytes = 0;
};

}  // namespace dequantize_packed
}  // namespace onnxruntime
