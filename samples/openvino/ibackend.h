// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnx {
class ModelProto;
}

namespace onnxruntime {
namespace openvino_ep {

class IBackend {
 public:
  virtual void Infer(OrtKernelContext* context) = 0;
};

class BackendFactory {
 public:
  static std::shared_ptr<IBackend>
  MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
              GlobalContext& global_context,
              const SubGraphContext& subgraph_context);
};

}  // namespace openvino_ep
}  // namespace onnxruntime
