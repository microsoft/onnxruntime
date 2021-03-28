// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/env.h"
#define LOAD_PYOP_SYM(n, v, m) ORT_ENFORCE(Env::Default().GetSymbolFromLibrary(handle_, n, reinterpret_cast<void**>(&v)) == Status::OK(), m)
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#ifdef _WIN32
#include <Windows.h>
#else
#define HMODULE void*
#endif

namespace onnxruntime {
using OnnxAttrs = std::unordered_map<std::string, std::string>;

class PyOpLibProxy {
 public:
  static PyOpLibProxy& GetInstance();
  void ReleaseInstance(void*);
  bool InvokePythonFunc(void*,
                        const char*,
                        const std::vector<OrtValue*>&,
                        std::vector<std::unique_ptr<char[]>>&,
                        std::vector<int32_t>&,
                        std::vector<std::vector<int64_t>>&,
                        std::function<void(const char*)>);

  bool InvokePythonFunc(const char* module,
                        const char* function,
                        const std::vector<const OrtValue*>& inputs,
                        std::vector<void*>& outputs);

  bool InvokePythonAutoGradFunc(void* function,
                                const std::vector<const OrtValue*>&,
                                std::vector<void*>& outputs);

  bool InvokePythonAutoGradFunc(void*,
                                const char*,
                                const std::vector<OrtValue*>&,
                                std::vector<void*>& outputs,
                                std::function<void(const char*)>);
  const char* GetLastErrorMessage(std::string&);
  void* NewInstance(const char*, const char*, const OnnxAttrs&);
  void* NewInstance(void* pyClass);
  bool Initialized() const { return initialized_; };
  int32_t GetGil() const;
  void PutGil(int32_t) const;

 private:
  PyOpLibProxy();
  ~PyOpLibProxy();
  bool initialized_ = false;
};

}  // namespace onnxruntime
