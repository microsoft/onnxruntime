// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/dll_safe.h"
#include "vaip/vaip_ort_api.h"
#include "vaip/custom_op.h"
#include "onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.h"
#include <cstdlib>
#include <iostream>
using namespace std;

namespace onnxruntime_vitisai_ep {
static void my_abort() {
  cerr << "please install VitisAI package." << endl;
  abort();
}
using namespace vaip_core;
void initialize_onnxruntime_vitisai_ep(OrtApiForVaip* /*api*/, std::vector<OrtCustomOpDomain*>& /*domain*/) {
  my_abort();
  return;
}  // namespace onnxruntime_vitisai_ep
DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>
compile_onnx_model_3(const std::string& /*model_path*/, const Graph& /*graph*/,
                     const char* /*json_config*/) {
  if (1) {  // suppress dead code warning
    my_abort();
  }
  return DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>();
}

}  // namespace onnxruntime_vitisai_ep
