// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <memory>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

class VADMBackend : public IBackend {
 public:
  VADMBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
              GlobalContext& global_context,
              const SubGraphContext& subgraph_context);

  void Infer(OrtKernelContext* context) override;

 private:
  void StartAsyncInference(Ort::KernelContext& context,
                           size_t batch_slice_idx, size_t infer_req_idx);

  void CompleteAsyncInference(Ort::KernelContext& context,
                              size_t batch_slice_idx, size_t infer_req_idx,
                              size_t batch_size);

  GlobalContext& global_context_;
  SubGraphContext subgraph_context_;
  std::shared_ptr<OVNetwork> ie_cnn_network_;
  std::map<std::string, std::shared_ptr<ngraph::Node>> const_outputs_map_;
  std::vector<OVInferRequestPtr> infer_requests_;
  size_t num_inf_reqs_;
  mutable std::mutex compute_lock_;
};
}  // namespace openvino_ep
}  // namespace onnxruntime
