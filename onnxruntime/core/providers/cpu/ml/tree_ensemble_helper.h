// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace ml {

template <typename T>
Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<T>& data) {
  ONNX_NAMESPACE::TensorProto proto;
  auto result = info.GetAttr(name, &proto);

  SafeInt<int64_t> n_elements(1);
  for (auto dim : proto.dims()) {
    n_elements *= dim;
  }

  if (proto.dims().empty()) {
    return Status::OK();
  }

  const SafeInt<size_t> tensor_size(n_elements);
  data.clear();
  data.resize(tensor_size);

  result = utils::UnpackTensor<T>(proto, std::filesystem::path(), data.data(), tensor_size);
  ORT_ENFORCE(result.IsOK(), "TreeEnsemble could not unpack tensor attribute ", name);

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
