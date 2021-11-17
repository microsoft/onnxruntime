// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_tensor.h"

namespace onnxruntime {
#ifdef DEBUG_BEAM_SEARCH

#ifdef NDEBUG
bool g_enable_tensor_dump = false;
#else
bool g_enable_tensor_dump = true;
#endif

void DumpOrtValue(const char* name, const OrtValue& value) {
  if (!g_enable_tensor_dump)
    return;
  std::cout << std::string(name) << "\n";
  const Tensor& tensor = value.Get<Tensor>();
  MLDataType dataType = tensor.DataType();
  if (dataType == DataTypeImpl::GetType<float>()) {
    DumpTensor<float>(nullptr, tensor);
  } else if (dataType == DataTypeImpl::GetType<int32_t>()) {
    DumpTensor<int32_t>(nullptr, tensor);
  } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
    DumpTensor<int64_t>(nullptr, tensor);
  } else {
    std::cout << "not float/int32/int64";
  }
}

void ConfigureTensorDump(bool enable) {
  g_enable_tensor_dump = enable;
}
#endif

}  // namespace onnxruntime