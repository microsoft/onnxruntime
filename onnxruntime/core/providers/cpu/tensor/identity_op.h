// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/common/common.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/framework/op_kernel.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/utils.h"

namespace onnxruntime {

template <bool is_dropout>
class IdentityOp final : public OpKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* input_type_proto = Node().InputDefs()[0]->TypeAsProto();

    const auto* input_ort_value = context->GetInputOrtValue(0);

#if !defined(DISABLE_OPTIONAL_TYPE)
    // Only Optional type can be None (i.e.) not have data
    if (input_type_proto->has_optional_type() && !input_ort_value->IsAllocated()) {
      // We can't rely on the input OrtValue containing type information
      // as it could be a main graph input which will be missing the type
      // in the corresponding OrtValue for the "None" case because
      // the user doesn't provide any input for the "None" case.
      ORT_RETURN_IF_ERROR(utils::OutputOptionalWithoutDataHelper(*input_type_proto, context, 0));
      return Status::OK();
    }
#else
    ORT_UNUSED_PARAMETER(input_type_proto);
#endif

    if (input_ort_value->IsTensor()) {
      const auto* X = &input_ort_value->Get<Tensor>();
      const TensorShape& shape = X->Shape();
      Tensor* Y = context->Output(0, shape);
      auto X_type = X->DataType();

      const void* source = X->DataRaw(X_type);
      void* target = Y->MutableDataRaw(X_type);
      //If source and target pointers are not equal, we need to copy the data.
      if (target != source) {
        if (!X->IsDataTypeString()) {
          memcpy(target, source, shape.Size() * X_type->Size());
        } else {
          // handle std::string
          const auto* src = X->Data<std::string>();
          auto* dst = Y->MutableData<std::string>();
          std::copy(src, src + shape.Size(), dst);
        }
      }

      if (is_dropout) {
        Tensor* mask = context->Output(1, shape);
        // a 'nullptr' returned would make it an unused optional output
        if (mask != nullptr) {
          // Opset 7 differs with Opset 10 in that the type of the 'mask'
          // output is tied with the type of the input in Opset 7 whereas
          // the type of 'mask' in Opset 10 is 'bool' always
          // so we have a common solution
          void* mask_data = mask->MutableDataRaw();
          // In 'test'/'inference' mode, there are no input values dropped out
          // so fill the buffer with 0/false
          memset(mask_data, 0, mask->SizeInBytes());
        }
      }
    } else {  // Has to be TensorSeq
      const auto* X = &input_ort_value->Get<TensorSeq>();
      TensorSeq* output = context->Output<TensorSeq>(0);

      // Check if the output is an alias of the input
      // If so, there is nothing else to be done.
      // Equivalent of checking if two buffer pointers are
      // different before copying over the contents while
      // processing Tensors.
      if (X != output) {
        output->SetType(X->DataType());

        AllocatorPtr alloc;
        auto status = context->GetTempSpaceAllocator(&alloc);
        if (!status.IsOK()) {
          ORT_THROW("Unable to get an allocator");
        }
        std::vector<Tensor> tensors;
        for (auto it = X->begin(), end = X->end(); it != end; ++it) {
          Tensor tmp(it->DataType(), onnxruntime::TensorShape(it->Shape()), alloc);
          size_t bytes = it->SizeInBytes();
          memcpy(tmp.MutableDataRaw(), it->DataRaw(), bytes);
          tensors.push_back(std::move(tmp));
        }

        output->SetElements(std::move(tensors));
      }
    }

    return Status::OK();
  }
};

}  //namespace onnxruntime
