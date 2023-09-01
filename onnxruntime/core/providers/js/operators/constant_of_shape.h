// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/generator/constant_of_shape_base.h"

namespace onnxruntime {
namespace js {

template <typename T>
class ConstantOfShape : public JsKernel, public ConstantOfShapeBase<> {
 public:
  ConstantOfShape(const OpKernelInfo& info) : JsKernel(info), ConstantOfShapeBase(info) {
    auto size = sizeof(T);
    void* value_ptr = GetValuePtr();
    // The value of `dataType` must be the same as `DataType` in ./js/web/lib/wasm/wasm-common.ts
    switch (size) {
      // TODO: support sizeof(int8_t), sizeof(int16_t) and sizeof(int64_t)
      case sizeof(int32_t):
      {
        int32_t value = *(reinterpret_cast<int32_t*>(value_ptr));
        JSEP_INIT_KERNEL_ATTRIBUTE(ConstantOfShape, ({"value" : Number($1), "dataType" : Number($2)}),
                                   static_cast<int32_t>(value), static_cast<int32_t>(6));
        break;
      }
      default:
      {
        ORT_THROW("Unsupported value attribute datatype with size: ", size);
      }
    }
  }
};

}  // namespace js
}  // namespace onnxruntime
