// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

class IBackend {
 public:
  virtual void Infer(OrtKernelContext* context) = 0;
  virtual ov::CompiledModel& GetOVCompiledModel() = 0;
};

class BackendFactory {
 public:
  static std::shared_ptr<IBackend>
  MakeBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
              GlobalContext& global_context,
              const SubGraphContext& subgraph_context,
              EPCtxHandler& ctx_handle);
};

}  // namespace openvino_ep
}  // namespace onnxruntime
