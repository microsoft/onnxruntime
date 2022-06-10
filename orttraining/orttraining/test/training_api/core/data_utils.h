// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace training {
namespace test {

template <typename T>
void OrtValueToVec(const OrtValue& val, std::vector<T>& output) {
  const Tensor& tensor = val.Get<Tensor>();
  int64_t num_elem = tensor.Shape().Size();
  const T* val_ptr = tensor.template Data<T>();
  output.assign(val_ptr, val_ptr + num_elem);
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
