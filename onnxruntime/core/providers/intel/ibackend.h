// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace intel_ep {

class IBackend{
  public:
  virtual void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;
};

} // namespace intel_ep
} // namespace onnxruntime
