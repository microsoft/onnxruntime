// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/tensor/utils.h"
#include "orttraining/training_ops/cpu/optimizer/common.h"

namespace onnxruntime {
namespace contrib {

Status CopyIfNotSameCPUBuffer(OpKernelContext* ctx, size_t number_of_values,
                              const TensorSeq* src_values, TensorSeq* dest_values) {
  if (src_values != dest_values) {
    AllocatorPtr alloc;
    ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK(),
                "CPU CopyIfNotSameBuffer for tensor sequence: Unable to get an allocator.");

    dest_values->SetType(src_values->DataType());
    dest_values->Reserve(number_of_values);
    for (size_t input_idx = 0; input_idx < number_of_values; ++input_idx) {
      const Tensor& source_tensor = src_values->Get(input_idx);
      auto target_tensor = std::make_unique<Tensor>(source_tensor.DataType(),
                                                    source_tensor.Shape(),
                                                    alloc);

      CopyCpuTensor(&source_tensor, target_tensor.get());

      auto ml_tensor = DataTypeImpl::GetType<Tensor>();
      OrtValue target_ort_value(target_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

      dest_values->Add(target_ort_value);  // Add will check for type consistency
    }
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
