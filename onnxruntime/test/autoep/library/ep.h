// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

class ExampleEpFactory;
struct MulKernel;

/// <summary>
/// Example EP that can compile a single Mul operator.
/// </summary>
class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  ExampleEp(ExampleEpFactory& factory, const std::string& name,
            const OrtSessionOptions& session_options, const OrtLogger& logger);

  ~ExampleEp();

  std::unordered_map<std::string, std::unique_ptr<MulKernel>>& Kernels() {
    return kernels;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL ExampleEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                              OrtEpGraphSupportInfo* graph_support_info);
  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos);
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos);

  ExampleEpFactory& factory_;
  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels;
};
