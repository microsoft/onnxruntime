// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"

#include <algorithm>
#include <cctype>
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

const char* Bf16PrecisionName(Bf16Precision p) {
  switch (p) {
    case Bf16Precision::kFp32:
      return "fp32";
    case Bf16Precision::kFp16:
      return "fp16";
    case Bf16Precision::kBf16:
      return "bf16";
    default:
      return "fp32";
  }
}

bool TryParseBf16Precision(const std::string& name, Bf16Precision* out) {
  if (name == "fp32") {
    *out = Bf16Precision::kFp32;
    return true;
  }
  if (name == "fp16") {
    *out = Bf16Precision::kFp16;
    return true;
  }
  if (name == "bf16") {
    *out = Bf16Precision::kBf16;
    return true;
  }
  return false;
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
