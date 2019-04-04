// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "core/framework/ml_value.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#define RETURN_IF_FAIL(expr)                                \
  do {                                                      \
    auto status = (expr);                                   \
    if ((!status.IsOK())) {                                 \
      printf("Fail: %s \n", status.ErrorMessage().c_str()); \
      return -1;                                            \
    }                                                       \
  } while (0);

namespace onnxruntime {
namespace training {

class TrainingUtil {
 public:
  template <typename T>
  static void CreateMLValue(AllocatorPtr alloc,
                            const std::vector<int64_t>& dims,
                            const std::vector<T>& value,
                            MLValue* p_mlvalue) {
    TensorShape shape(dims);
    auto location = alloc->Info();
    auto element_type = DataTypeImpl::GetType<T>();
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
    if (value.size() > 0) {
      memcpy(buffer, &value[0], element_type->Size() * shape.Size());
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                buffer,
                                                                location);
    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

  static AllocatorPtr GetCpuAllocator() {
    static CPUExecutionProviderInfo info;
    static CPUExecutionProvider cpu_provider(info);
    return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
  }
};
}  // namespace training
}  // namespace onnxruntime
