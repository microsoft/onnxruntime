// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_call.h"
#include "core/framework/float16.h"

namespace onnxruntime {
namespace cann {

#define CANN_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CANN_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CANN error executing ", #expr))

#define CANN_GRAPH_RETURN_IF_ERROR(expr)         \
  ORT_RETURN_IF_ERROR(CANN_GRAPH_CALL(expr)      \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CANN Graph error executing ", #expr))

template <typename ElemType>
struct Constants {
  static const ElemType Zero;
  static const ElemType One;
};

template <typename T>
class ToCannType {
 public:
  static T FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToCannType<MLFloat16> {
 public:
  static MLFloat16 FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MLFloat16*>(&h);
  }
};

}  // namespace cann
}  // namespace onnxruntime
