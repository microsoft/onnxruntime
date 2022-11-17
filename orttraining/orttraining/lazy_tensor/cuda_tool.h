// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA
#include <string>
#include <vector>
#include <memory>
#include "core/framework/execution_providers.h"

namespace onnxruntime {
namespace lazytensor {

class NvtxRange {
 public:
  NvtxRange(const char* name);
  NvtxRange(const std::string& name);
  ~NvtxRange();
};

// Class holding the CUDA EPs (one unique EP per device)
// shared by all sessions.
class CUDAExecutionProviderPool {
 public:
  static CUDAExecutionProviderPool& GetInstance() {
    static CUDAExecutionProviderPool instance;
    return instance;
  }

  std::shared_ptr<IExecutionProvider> GetExecutionProvider(const int device_id) {
    return cuda_execution_providers_.at(device_id);
  }

 private:
  CUDAExecutionProviderPool() {
    Initialize();
  };
  ~CUDAExecutionProviderPool() = default;
  CUDAExecutionProviderPool(const CUDAExecutionProviderPool&) = delete;
  CUDAExecutionProviderPool& operator=(const CUDAExecutionProviderPool&) = delete;
  void Initialize();

  std::vector<std::shared_ptr<IExecutionProvider>> cuda_execution_providers_;
};

}  // namespace lazytensor
}  // namespace onnxruntime
#endif
