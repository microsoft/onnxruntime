// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "./dll_safe.h"
#include "./my_ort.h"

struct OrtApi;
struct OrtKernelContext;

namespace vaip_core {
class CustomOp {
 public:
  VAIP_DLL_SPEC CustomOp();
  VAIP_DLL_SPEC virtual ~CustomOp();

 public:
  virtual void Compute(const OrtApi* api, OrtKernelContext* context) const = 0;
};

class ExecutionProvider {
 public:
  VAIP_DLL_SPEC ExecutionProvider();
  virtual ~ExecutionProvider();

 public:
  virtual DllSafe<std::vector<std::string>> get_meta_def_inputs() const = 0;
  virtual DllSafe<std::vector<std::string>> get_meta_def_outputs() const = 0;
  virtual DllSafe<std::vector<std::string>> get_meta_def_nodes() const = 0;
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_constant_initializer() const = 0;
  virtual std::unique_ptr<CustomOp> compile() const = 0;
};

}  // namespace vaip_core
