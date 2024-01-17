// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define DequantizeLinearBlockWise operator, it is basically
// dequantize input tensor and unpack it into float/half tensor.
//
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

#include "dequantize_blockwise.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class DequantizeLinearBlockWise final : public CudaKernel {
 public:
  DequantizeLinearBlockWise(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(K_ * N_ > 0, "K and N must be greater than 0.");
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("axis", &axis_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<std::string>("packing", &packing_));
    if (packing_ == "default") {
      ORT_ENFORCE(nbits_ == 4,
                  "Only 4b quantization is supported for DequantizeLinearBlockWise op,"
                  " additional bits support is planned.");
    } else if (packing_ == "gptq") {
      ORT_ENFORCE(axis_ == 0, "axis_ should be 0 for gptq packing.");
      ORT_ENFORCE(nbits_ > 1 && nbits_ < 8, "nbits_ should be in range of 2-8.");
    } else if (packing_ == "hqq") {
      ORT_ENFORCE(axis_ == 0, "axis_ should be 0 for hqq packing.");
      ORT_ENFORCE(nbits_ == 4, "nbits_ should be in range of 2-8.");
    } else {
      ORT_THROW("Unsupported packing type: ", packing_);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeInternalGPTQ(OpKernelContext* context) const;
  Status ComputeInternalHQQ(OpKernelContext* context) const;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t axis_;
  int64_t nbits_;
  std::string packing_;
};

template <typename T>
Status DequantizeLinearBlockWise<T>::ComputeInternalGPTQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  const auto* input_gidx = ctx->Input<Tensor>(3);
  const auto& weight_shape = input_qweight->Shape();

  auto OutputShape = TensorShape({in_features, out_features});

  Tensor* Y = ctx->Output(0, OutputShape);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  auto fp16_weight_shape = weight_shape;
  fp16_weight_shape[0] = in_features;

  const auto* zero_point = input_zeros && input_zeros->DataRaw() ? input_zeros->DataRaw() : nullptr;
  if (input_gidx && input_gidx->Shape().Size() > 1) {
    GPTQPacking::DequantWeightNbitGidx(Stream(ctx), input_qweight->Data<int32_t>(),
                                       input_scale->Data<MLFloat16>(),
                                       static_cast<const int32_t*>(zero_point),
                                       input_gidx->Data<int32_t>(),
                                       Y->MutableData<MLFloat16>(),
                                       in_features, weight_shape[1], nbits_, groupsize);
  } else {
    GPTQPacking::DequantWeightNbit(Stream(ctx), input_qweight->Data<int32_t>(),
                                   input_scale->Data<MLFloat16>(),
                                   static_cast<const uint32_t*>(zero_point),
                                   Y->MutableData<MLFloat16>(),
                                   in_features, weight_shape[1], nbits_, groupsize);
  }
  return Status::OK();
}

template <typename T>
Status DequantizeLinearBlockWise<T>::ComputeInternalHQQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  // const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  const auto& weight_shape = input_qweight->Shape();
  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;
  auto OutputShape = TensorShape({in_features, N_});

  Tensor* Y = ctx->Output(0, OutputShape);

  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  auto fp16_weight_shape = weight_shape;
  fp16_weight_shape[0] = in_features;

  GPTQPacking::DequantWeightNbit(Stream(ctx), input_qweight->Data<int32_t>(),
                                 input_scale->Data<MLFloat16>(),
                                 static_cast<const CudaT*>(input_zeros->DataRaw()),
                                 Y->MutableData<MLFloat16>(),
                                 in_features, weight_shape[1], nbits_, groupsize);

  return Status::OK();
}

template <typename T>
Status DequantizeLinearBlockWise<T>::ComputeInternal(OpKernelContext* ctx) const {
  if (packing_ == "gptq") {
    return this->ComputeInternalGPTQ(ctx);
  }
  if (packing_ == "hqq") {
    return this->ComputeInternalHQQ(ctx);
  }
  const Tensor* b = ctx->Input<Tensor>(0);
  const Tensor* scales = ctx->Input<Tensor>(1);
  const Tensor* zero_points = ctx->Input<Tensor>(2);

  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  typedef typename ToCudaType<T>::MappedType CudaT;

  TensorShape b_shape({N_, K_});

  Tensor* Y = ctx->Output(0, b_shape);
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;

  if (axis_ == 1) {
    // column-wise block
    ORT_RETURN_IF_ERROR(Dequantize4Bits(
        reinterpret_cast<CudaT*>(Y->MutableDataRaw()),
        blob_data,
        reinterpret_cast<const CudaT*>(scales_data),
        zero_points_data,
        SafeInt<int>(K_padded),
        SafeInt<int>(N_),
        SafeInt<int>(block_size_),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
  } else {
    // row-wise block
    K_padded = K_;

    ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
        reinterpret_cast<CudaT*>(Y->MutableDataRaw()),
        blob_data,
        reinterpret_cast<const CudaT*>(scales_data),
        zero_points_data,
        SafeInt<int>(block_size_),
        axis_,
        SafeInt<int>(K_),
        SafeInt<int>(N_),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));

#if 0
  cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  T* b_data_cpu = new T[K_ * N_];
  cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);
  delete[] b_data_cpu;
#endif
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DequantizeLinearBlockWise,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    DequantizeLinearBlockWise<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DequantizeLinearBlockWise,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int32_t>()}),
    DequantizeLinearBlockWise<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
