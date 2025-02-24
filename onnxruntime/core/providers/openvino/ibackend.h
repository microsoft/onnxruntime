// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <istream>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

class IBackend {
 public:
  virtual void Infer(OrtKernelContext* context) = 0;
  virtual ov::CompiledModel& GetOVCompiledModel() = 0;
  virtual ~IBackend() = default;
};
using ptr_stream_t = std::unique_ptr<std::istream>;
class BackendFactory {
 public:
  static std::shared_ptr<IBackend>
  MakeBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
              SessionContext& session_context,
              const SubGraphContext& subgraph_context,
              SharedContext& shared_context,
              ptr_stream_t& model_stream);
};

}  // namespace openvino_ep
}  // namespace onnxruntime
