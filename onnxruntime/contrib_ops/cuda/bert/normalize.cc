// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "normalize.h"
#include "core/framework/tensorprotoutils.h"
#include "normalize_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Normalize,                                                  \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Normalize);

REGISTER_KERNEL_TYPED(float)

static void FetchDataFromTensor(TensorProto& t_proto, std::vector<float>& value) {
  ORT_ENFORCE(t_proto.has_data_type());
  ORT_ENFORCE(TensorProto::DataType_IsValid(t_proto.data_type()));
  const auto tensor_type = static_cast<TensorProto_DataType>(t_proto.data_type());
  const void* const raw_data = t_proto.has_raw_data() ? t_proto.raw_data().data() : nullptr;
  const size_t raw_data_len = t_proto.has_raw_data() ? t_proto.raw_data().size() : 0;

  int64_t expected_size = 1;
  for( int d = 0; d < t_proto.dims_size(); d++ ) expected_size *= t_proto.dims()[ d ];
  value.resize( expected_size );
  auto unpack_status = utils::UnpackTensor(t_proto, raw_data, raw_data_len, value.data(), expected_size );
  ORT_ENFORCE(unpack_status.IsOK(), "Value attribute unpacking failed:", unpack_status.ErrorMessage());
}

Normalize::Normalize(const OpKernelInfo& info) : CudaKernel(info) {
  TensorProto t_proto;
  info.GetAttr<TensorProto>("gamma", &t_proto);
  FetchDataFromTensor( t_proto, gamma_ );

  info.GetAttr<TensorProto>("beta", &t_proto );
  FetchDataFromTensor( t_proto, beta_ );

  gamma_data_ = GetScratchBuffer<float>(gamma_.size());
  CUDA_CALL_THROW(cudaMemcpy(gamma_data_.get(), gamma_.data(), sizeof(float) * gamma_.size(), cudaMemcpyHostToDevice));

  beta_data_ = GetScratchBuffer<float>(beta_.size());
  CUDA_CALL_THROW(cudaMemcpy(beta_data_.get(), beta_.data(), sizeof(float) * beta_.size(), cudaMemcpyHostToDevice));
}

Status Normalize::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto dims = X->Shape().GetDims();

  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input is expected to have 3 dimensions, got ", dims.size());
  }

  Tensor* Y = context->Output(0, X->Shape());

  launchNormalizeKernel(X->template Data<float>(), Y->template MutableData<float>(), gamma_data_.get(), beta_data_.get(), static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
