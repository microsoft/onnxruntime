// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "js_export.h"

#include "core/framework/op_kernel.h"

const void* JsepOutput(void* context, int index, void* data) {
  uint32_t* data_offset = reinterpret_cast<uint32_t*>(data);
  uint32_t dim = *data_offset++;
  size_t dim_size = static_cast<size_t>(dim);
  std::vector<int64_t> dims;
  dims.reserve(dim_size);
  dims.resize(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    dims[i] = static_cast<int64_t>(*data_offset++);
  }

  LOGF_DEFAULT(VERBOSE, "JsepOutput(%d, %s)", index, onnxruntime::TensorShape(dims).ToString().c_str());

  auto output = reinterpret_cast<onnxruntime::OpKernelContext*>(context)->Output(index, onnxruntime::TensorShape(dims));
  auto r = output->DataRaw();

  LOGF_DEFAULT(VERBOSE, "JsepOutput -- data=%zu", (size_t)(r));
  return r;
}
