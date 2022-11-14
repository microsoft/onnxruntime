// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>

#include "core/framework/op_kernel.h"
#include "core/providers/js/js_execution_provider.h"

struct pthreadpool;

namespace onnxruntime {
namespace js {

#define JSEP_INIT_KERNEL(x) EM_ASM({ Module.jsepCreateKernel(#x, $0, undefined); }, this)
#define JSEP_INIT_KERNEL_ATTRIBUTE(x, attr, ...) EM_ASM({ Module.jsepCreateKernel(#x, $0, attr); }, this, __VA_ARGS__)

#define JSEP_KERNEL_IMPL(classname, x)                       \
class classname : public JsKernel {                          \
public:                                                      \
    classname(const OpKernelInfo& info) : JsKernel(info) {   \
        JSEP_INIT_KERNEL(x);                                 \
    }                                                        \
};

#define JSEP_KERNEL_TYPED_IMPL(classname, x)                 \
template<typename T>                                         \
class classname : public JsKernel {                          \
public:                                                      \
    classname(const OpKernelInfo& info) : JsKernel(info) {   \
        JSEP_INIT_KERNEL(x);                                 \
    }                                                        \
};

#define JSEP_CLASS_IMPL_ATTRIBUTE(classname, x, attr_pre, attr, ...)       \
class classname : public JsKernel {                                        \
public:                                                                    \
    classname(const OpKernelInfo& info) : JsKernel(info) {                 \
        attr_pre                                                           \
        JSEP_INIT_KERNEL_ATTRIBUTE(x, attr, __VA_ARGS__);                  \
    }                                                                      \
};

#define JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(classname, x, attr_name, default_value, ...) \
    JSEP_CLASS_IMPL_ATTRIBUTE(classname, x, , ({#attr_name:$1}), static_cast<double>(info.GetAttrOrDefault<float>(#attr_name, 1.0)))

#define JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT(classname, x, attr_name, ...) \
    JSEP_CLASS_IMPL_ATTRIBUTE(classname, x,                           \
        float value;                                                  \
        ORT_ENFORCE(info.GetAttr<float>(#attr_name, &value)); ,       \
        , ({#attr_name:$1}), static_cast<double>(value))

class JsKernel : public OpKernel {
 public:
  explicit JsKernel(const OpKernelInfo& info)
      : OpKernel(info) {}
  virtual ~JsKernel() {
      EM_ASM({ Module.jsepReleaseKernel($0); }, this);
  }

  Status Compute(OpKernelContext* context) const override {
      AllocatorPtr alloc;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceCPUAllocator(&alloc));

      //
      // temp_data_format (every item is (u)int32_t):
      //    context_prt | input_count | [input_data_0] ... [input_data_N-1]
      //
      // input_data_format:
      //    type | data_ptr | dim_size | dim[0] ... dim[N-1]
      //
      size_t temp_data_size = sizeof(size_t) * 2;
      for (int i = 0; i < context->InputCount(); i++) {
        temp_data_size += sizeof(size_t) * (3 + context->Input<Tensor>(i)->Shape().NumDimensions());
      }
      uint32_t *p_inputs_data = reinterpret_cast<uint32_t*>(alloc->Alloc(temp_data_size));
      p_inputs_data[0] = reinterpret_cast<uint32_t>(context);
      p_inputs_data[1] = static_cast<uint32_t>(context->InputCount());
      size_t index = 2;
      for (int i = 0; i < context->InputCount(); i++) {
        p_inputs_data[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->GetElementType());
        p_inputs_data[index++] = reinterpret_cast<uint32_t>(context->Input<Tensor>(i)->DataRaw());
        p_inputs_data[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->Shape().NumDimensions());
        for (size_t d = 0; d < context->Input<Tensor>(i)->Shape().NumDimensions(); d++) {
          p_inputs_data[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->Shape()[d]);
        }
      }

      // printf("temp data size: %zu. Data: ", temp_data_size);
      // for (int i=0; i < (int)temp_data_size/4;i++) {
      //   printf("%u ", p_inputs_data[i]);
      // }
      // printf("\n");

      int status = EM_ASM_INT({ return Module.jsepRun($0, $1); }, this, p_inputs_data);

      // printf("outputs = %d. Y.data=%zu\n", context->OutputCount(), (size_t)(context->Output<Tensor>(0)->DataRaw()));

      alloc->Free(p_inputs_data);
      if (status == 0) {
        return Status::OK();
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to run JSEP kernel");
      }
  }
};
}  // namespace js
}  // namespace onnxruntime
