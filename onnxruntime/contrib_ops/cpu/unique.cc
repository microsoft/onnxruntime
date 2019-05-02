// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "unique.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include <unordered_set>

namespace onnxruntime {
namespace contrib {

Status Unique::Compute(OpKernelContext* ctx) const {
  const Tensor& sequence_0 = *(ctx->Input<Tensor>(0));          // sequence_0: [sequence_length, 1]

  const TensorShape& sequence_0_shape = sequence_0.Shape();

  int64_t seq_len_0 = sequence_0_shape[0];

  const int32_t* seq_0_ptr = sequence_0.Data<int32_t>();

  std::unordered_set<int32_t> seq_set;
  for (int64_t i = 0; i < seq_len_0; i++)
  {
      seq_set.insert(seq_0_ptr[i]);
  }

  int64_t output_size = static_cast<int64_t>(seq_set.size());

  TensorShape Y_dims{output_size};
  Tensor* Y0 = ctx->Output(/*index*/ 0, Y_dims);

  int32_t* result_ptr = Y0->MutableData<int32_t>();
  size_t count=0;
  for (const auto& elem: seq_set) {
      result_ptr[count++] = elem;
  }

  TensorShape Y_1_dims{seq_len_0};
  Tensor* Y1 = ctx->Output(/*index*/ 1, Y_1_dims);

  int32_t* result_1_ptr = Y1->MutableData<int32_t>();

  for (int32_t i = 0; i < seq_len_0; i++)
  {
      for (int32_t j = 0; j < output_size; j++)
      {
          if (result_ptr[j] == seq_0_ptr[i])
          {
               result_1_ptr[i] = j;
               break;
          }
      }
  }

  return Status::OK();
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    Unique,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Unique);

}  // namespace contrib
}  // namespace onnxruntime
