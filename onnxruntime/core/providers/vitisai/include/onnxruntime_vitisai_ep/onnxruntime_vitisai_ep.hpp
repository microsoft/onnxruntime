// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <filesystem>
#include <vector>
#if defined(_WIN32)
#if ONNXRUNTIME_VITISAI_EP_EXPORT_DLL == 1
#define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __declspec(dllexport)
#else
#define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __declspec(dllimport)
#endif
#else
#define ONNXRUNTIME_VITISAI_EP_DLL_SPEC __attribute__((visibility("default")))
#endif

#ifndef USE_VITISAI
#define USE_VITISAI /* mimic VITISAI EP in ORT */
#endif

namespace vaip_core {
class ExecutionProvider;
struct OrtApiForVaip;
template <typename T>
class DllSafe;
}  // namespace vaip_core
namespace onnxruntime {
class Graph;
}
struct OrtCustomOpDomain;
namespace onnxruntime_vitisai_ep {

ONNXRUNTIME_VITISAI_EP_DLL_SPEC void
initialize_onnxruntime_vitisai_ep(vaip_core::OrtApiForVaip* api,
                                  std::vector<OrtCustomOpDomain*>& ret_domain);
ONNXRUNTIME_VITISAI_EP_DLL_SPEC
vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>
compile_onnx_model_3(const std::string& model_path,
                     const onnxruntime::Graph& graph, const char* json_config);
ONNXRUNTIME_VITISAI_EP_DLL_SPEC
int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);
}  // namespace onnxruntime_vitisai_ep

extern "C" ONNXRUNTIME_VITISAI_EP_DLL_SPEC const vaip_core::OrtApiForVaip*
get_the_global_api();
