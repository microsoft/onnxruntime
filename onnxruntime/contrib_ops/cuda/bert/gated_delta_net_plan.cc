// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"

#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

const char* EngineName(Engine e) {
  switch (e) {
    case Engine::kChunked:
      return "chunked";
    case Engine::kChunkedSplit:
      return "chunked_split";
    case Engine::kRecurrent:
      return "recurrent";
    case Engine::kCudnn:
      return "cudnn";
    default:
      return "auto";
  }
}

Engine EngineFromName(const std::string& name) {
  if (name == "chunked") return Engine::kChunked;
  if (name == "chunked_split") return Engine::kChunkedSplit;
  if (name == "recurrent") return Engine::kRecurrent;
  if (name == "cudnn") return Engine::kCudnn;
  return Engine::kAuto;
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
