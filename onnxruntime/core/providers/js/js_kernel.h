// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>

#ifndef NDEBUG
#include <sstream>
#endif

#include "core/framework/op_kernel.h"
#include "core/providers/js/js_execution_provider.h"

struct pthreadpool;

namespace onnxruntime {
namespace js {

// This macro is defined to bypass the code format from clang-format, which will overwrite "=>" into "= >"
// We can use it to write JS inline code with arrow functions.

// clang-format off
#define JS_ARROW =>
// clang-format on

#define JSEP_INIT_KERNEL(optype) EM_ASM({ Module.jsepCreateKernel(#optype, $0, undefined); }, this)
#define JSEP_INIT_KERNEL_ATTRIBUTE(optype, attr, ...) EM_ASM({ Module.jsepCreateKernel(#optype, $0, attr); }, this, __VA_ARGS__)

#define JSEP_KERNEL_IMPL(classname, optype)                \
  class classname : public JsKernel {                      \
   public:                                                 \
    classname(const OpKernelInfo& info) : JsKernel(info) { \
      JSEP_INIT_KERNEL(optype);                            \
    }                                                      \
  };

#define JSEP_KERNEL_TYPED_IMPL(classname, optype)          \
  template <typename T>                                    \
  class classname : public JsKernel {                      \
   public:                                                 \
    classname(const OpKernelInfo& info) : JsKernel(info) { \
      JSEP_INIT_KERNEL(optype);                            \
    }                                                      \
  };

#define JSEP_CLASS_IMPL_ATTRIBUTE(classname, optype, attr_pre, attr, ...) \
  class classname : public JsKernel {                                     \
   public:                                                                \
    classname(const OpKernelInfo& info) : JsKernel(info) {                \
      attr_pre                                                            \
          JSEP_INIT_KERNEL_ATTRIBUTE(optype, attr, __VA_ARGS__);          \
    }                                                                     \
  };

#define JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_DEFAULT(classname, optype, attr_name, default_value, ...) \
  JSEP_CLASS_IMPL_ATTRIBUTE(classname, optype, , ({#attr_name : $1}), static_cast<double>(info.GetAttrOrDefault<float>(#attr_name, default_value)))

#define JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT_2_DEFAULT(classname, optype, attr_name_1, default_value_1, attr_name_2, default_value_2, ...) \
  JSEP_CLASS_IMPL_ATTRIBUTE(classname, optype, , ({#attr_name_1 : $1, #attr_name_2 : $2}),                                            \
                            static_cast<double>(info.GetAttrOrDefault<float>(#attr_name_1, default_value_1)),                         \
                            static_cast<double>(info.GetAttrOrDefault<float>(#attr_name_2, default_value_2)))

#define JSEP_CLASS_IMPL_ATTRIBUTE_FLOAT(classname, optype, attr_name, ...)         \
  JSEP_CLASS_IMPL_ATTRIBUTE(classname, optype,                                     \
                            float value;                                           \
                            ORT_ENFORCE(info.GetAttr<float>(#attr_name, &value));, \
                                                                                 , ({#attr_name : $1}), static_cast<double>(value))

// TODO:
// class JsMultiProgramKernel : public OpKernel { /* TBD */ };

class JsKernel : public OpKernel {
 public:
  explicit JsKernel(const OpKernelInfo& info)
      : OpKernel(info) {}
  ~JsKernel() override {
    EM_ASM({ Module.jsepReleaseKernel($0); }, this);
  }

  Status SerializeKernelContext(OpKernelContext* context, AllocatorPtr alloc, void* custom_data_ptr, size_t custom_data_size, void** ptr) const {
    //
    // temp_data_format (every item is (u)int32_t):
    //    context_ptr | input_count | custom_data_ptr | custom_data_size | [input_data_0] ... [input_data_N-1]
    //
    // input_data_format:
    //    type | data_ptr | dim_size | dim[0] ... dim[N-1]
    //
    size_t temp_data_size = sizeof(size_t) * 4;
    for (int i = 0; i < context->InputCount(); i++) {
      temp_data_size += sizeof(size_t) * (3 + context->Input<Tensor>(i)->Shape().NumDimensions());
    }
    uint32_t* p_serialized_kernel_context = reinterpret_cast<uint32_t*>(alloc->Alloc(temp_data_size));
    if (p_serialized_kernel_context == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate memory for serialized kernel context.");
    }

    p_serialized_kernel_context[0] = reinterpret_cast<uint32_t>(context);
    p_serialized_kernel_context[1] = static_cast<uint32_t>(context->InputCount());
    p_serialized_kernel_context[2] = reinterpret_cast<uint32_t>(custom_data_ptr);
    p_serialized_kernel_context[3] = static_cast<uint32_t>(custom_data_size);
    size_t index = 4;
    for (int i = 0; i < context->InputCount(); i++) {
      p_serialized_kernel_context[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->GetElementType());
      p_serialized_kernel_context[index++] = reinterpret_cast<uint32_t>(context->Input<Tensor>(i)->DataRaw());
      p_serialized_kernel_context[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->Shape().NumDimensions());
      for (size_t d = 0; d < context->Input<Tensor>(i)->Shape().NumDimensions(); d++) {
        p_serialized_kernel_context[index++] = static_cast<uint32_t>(context->Input<Tensor>(i)->Shape()[d]);
      }
    }

#ifndef NDEBUG
    std::ostringstream os;
    os << "temp data size: " << temp_data_size << ". Data:";
    size_t temp_data_count = temp_data_size >> 2;
    for (size_t i = 0; i < temp_data_count; i++) {
      os << " " << p_serialized_kernel_context[i];
    }
    LOGS_DEFAULT(VERBOSE) << os.str();
#endif

    *ptr = p_serialized_kernel_context;
    return Status::OK();
  }

  virtual Status SerializeCustomData(OpKernelContext* context, AllocatorPtr alloc, void** ptr, size_t* size) const {
    // default implementation: no custom data
    //
    // a subclass can override this method to serialize custom data. following is an example:
    //
    // // STEP.1 - calculate the size of the custom data
    // size_t bytes_required = 2 * sizeof(int32_t);
    //
    // // STEP.2 - allocate memory for the custom data
    // void* p_custom_data = alloc->Alloc(bytes_required);
    //
    // // STEP.3 - validate the memory allocation
    // if (p_custom_data == nullptr) {
    //   return Status(ONNXRUNTIME, FAIL, "failed to allocate memory for the custom data");
    // }
    //
    // // STEP.4 - serialize the custom data
    // int32_t* p_int32 = reinterpret_cast<int32_t*>(p_custom_data);
    // p_int32[0] = 100;
    // p_int32[1] = 200;
    //
    // // STEP.5 - assign output parameter "ptr" and "size"
    // *ptr = p_custom_data;
    // *size = bytes_required;
    //
    // return Status::OK();
    //
    return Status::OK();
  }

  virtual Status ComputeInternal(OpKernelContext* context) const {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceCPUAllocator(&alloc));

    void* p_custom_data = nullptr;
    size_t custom_data_size = 0;
    ORT_RETURN_IF_ERROR(SerializeCustomData(context, alloc, &p_custom_data, &custom_data_size));

    void* p_serialized_kernel_context = nullptr;
    auto status = SerializeKernelContext(context, alloc, p_custom_data, custom_data_size, &p_serialized_kernel_context);
    if (!status.IsOK()) {
      if (p_custom_data != nullptr) {
        alloc->Free(p_custom_data);
      }
      return status;
    }

    int status_code = EM_ASM_INT({ return Module.jsepRun($0, $1); }, this, reinterpret_cast<int32_t>(p_serialized_kernel_context));

    LOGS_DEFAULT(VERBOSE) << "outputs = " << context->OutputCount() << ". Y.data="
                          << (size_t)(context->Output<Tensor>(0)->DataRaw()) << ".";

    alloc->Free(p_serialized_kernel_context);
    if (p_custom_data != nullptr) {
      alloc->Free(p_custom_data);
    }

    if (status_code == 0) {
      return Status::OK();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to run JSEP kernel");
    }
  }

  Status Compute(OpKernelContext* context) const override {
    return ComputeInternal(context);
  }
};
}  // namespace js
}  // namespace onnxruntime
