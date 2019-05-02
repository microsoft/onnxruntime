// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dense_intersection.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include <unordered_set>

namespace onnxruntime {
namespace contrib {

Status DenseIntersection::ValidateInputShape(const TensorShape& sequence_0_shape, const TensorShape& sequence_1_shape) const {
  if (sequence_0_shape[1] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Sequence shape[1] is not 1 for sequence 0, it is: ", sequence_0_shape[1]);
  }
  if (sequence_1_shape[1] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Sequence shape[1] is not 1 for sequence 1, it is: ", sequence_0_shape[1]);
  }
  return Status::OK();
}

Status DenseIntersection::Compute(OpKernelContext* ctx) const {
  const Tensor& sequence_0 = *(ctx->Input<Tensor>(0));          // sequence_0: [sequence_length, 1]
  const Tensor& sequence_1 = *(ctx->Input<Tensor>(1));          // sequence_0: [sequence_length, 1]

  const TensorShape& sequence_0_shape = sequence_0.Shape();
  const TensorShape& sequence_1_shape = sequence_1.Shape();

  ORT_RETURN_IF_ERROR(ValidateInputShape(sequence_0_shape, sequence_1_shape));

  int64_t seq_len_0 = sequence_0_shape[0];
  int64_t seq_len_1 = sequence_1_shape[0];

  const int64_t* seq_0_ptr = sequence_0.Data<int64_t>();
  const int64_t* seq_1_ptr = sequence_1.Data<int64_t>();

  std::unordered_set<int64_t> seq_set;
  for (int64_t i = 0; i < seq_len_0; i++)
  {
      seq_set.insert(seq_0_ptr[i]);
  }
  std::unordered_set<int64_t> seq_set_final;
  for (int64_t i = 0; i < seq_len_1; i++)
  {
      if (seq_set.find(seq_1_ptr[i]) != seq_set.end())
      {
          seq_set_final.insert(seq_1_ptr[i]);
      }
  }
  int64_t output_size = static_cast<int64_t>(seq_set_final.size());

  TensorShape Y_dims{output_size, 2};
  Tensor* Y = ctx->Output(/*index*/ 0, Y_dims);

  int64_t* result_ptr = Y->MutableData<int64_t>();
  size_t count=0;
  for (const auto& elem: seq_set_final) {
      result_ptr[count++] = elem;
      result_ptr[count++] = 0;
  }

  return Status::OK();
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    DenseIntersection,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    DenseIntersection);

}  // namespace contrib
}  // namespace onnxruntime
