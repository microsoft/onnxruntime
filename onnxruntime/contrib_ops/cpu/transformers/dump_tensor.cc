// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_tensor.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

namespace dump_tensor_env_vars {
constexpr const char* kDumpBeamSearch = "ORT_DUMP_BEAM_SEARCH";
}

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

void ConfigureTensorDump() {
  const auto parsed = ParseEnvironmentVariable<bool>(dump_tensor_env_vars::kDumpBeamSearch);
  if (parsed.has_value()) {
    g_enable_tensor_dump = *parsed;
  }
}

void DisableTensorDump() {
  g_enable_tensor_dump = false;
}

void DumpString(const char* name, int index, bool end_line) {
  if (!g_enable_tensor_dump)
    return;
  std::cout << std::string(name) << "[" << index << "]";

  if (end_line) {
    std::cout << std::endl;
  }
}

void DumpString(const char* name, std::string value, bool end_line) {
  if (!g_enable_tensor_dump)
    return;

  std::cout << std::string(name) << "=" << value;

  if (end_line) {
    std::cout << std::endl;
  }
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime