// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose.h"

namespace onnxruntime {
namespace rocm {

// Op Set 11 for ConvTranspose only update document to clearify default dilations and strides value.
// which are already covered by op set 11 cpu version, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T, false>);                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T, false>);

REGISTER_KERNEL_TYPED(float)
// not yet supported in MIOpen
// REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T, bool NHWC>
Status ConvTranspose<T, NHWC>::ComputeInternal(OpKernelContext* context) const {
  return DoConvTranspose(context, false);
}

template <typename T, bool NHWC>
Status ConvTranspose<T, NHWC>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  auto x_dims = x_shape.AsShapeVector();
  auto x_data = reinterpret_cast<const HipT*>(X->Data<T>());

  auto x_dimensions = X->Shape().NumDimensions();
  if (x_dimensions < 3 || x_dimensions > 5) {
    // TODO: the error message should tell which operator raises it.
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must be 3-, 4- or 5-dimensional.",
                           " X: ", X->Shape().ToString().c_str());
  }
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  auto w_dims = w_shape.AsShapeVector();
  auto w_data = reinterpret_cast<const HipT*>(W->Data<T>());

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;

  HipT* y_data = nullptr;
  if (x_dimensions == 3) {
    x_dims.insert(x_dims.begin() + 2, 1);
    w_dims.insert(w_dims.begin() + 2, 1);
  }

  {
    std::lock_guard<OrtMutex> lock(s_.mutex);
    // TODO: add a global cache if need to handle cases for multiple frames running simultaneously with different batch_size
    bool input_dims_changed = (s_.last_x_dims.AsShapeVector() != x_dims);
    bool w_dims_changed = (s_.last_w_dims.AsShapeVector() != w_dims);
    if (input_dims_changed || w_dims_changed) {
      if (input_dims_changed)
        s_.last_x_dims = gsl::make_span(x_dims);

      if (w_dims_changed) {
        s_.last_w_dims = gsl::make_span(w_dims);
        s_.cached_benchmark_bwd_results.clear();
      }

      ConvTransposeAttributes::Prepare p;
      ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, dynamic_padding));

      auto y_dims = p.Y->Shape().AsShapeVector();
      if (x_dimensions == 3) {
        y_dims.insert(y_dims.begin() + 2, 1);
        p.kernel_shape.insert(p.kernel_shape.begin(), 1);
        p.pads.insert(p.pads.begin(), 0);
        p.pads.insert(p.pads.begin() + 2, 0);
        p.strides.insert(p.strides.begin(), 1);
        p.dilations.insert(p.dilations.begin(), 1);
      }
      s_.y_dims = gsl::make_span(y_dims);

      if (w_dims_changed) {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, MiopenTensor::GetDataType<HipT>()));
      }

      // Special case when there is a dim value of 0 in the shape.
      // Return only after we have cached the following for subsequent runs :
      // 1) `w_dims` in the `w_desc`
      // 2) `y_dims` in s_.y_dims
      if (p.Y->Shape().Size() == 0) {
        return Status::OK();
      }

      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims, MiopenTensor::GetDataType<HipT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims, MiopenTensor::GetDataType<HipT>()));

      miopenConvolutionMode_t mode = miopenConvolution;
      ORT_RETURN_IF_ERROR(s_.conv_desc.Set(p.kernel_shape.size(), p.pads, p.strides, p.dilations,
                                           gsl::narrow_cast<int>(conv_transpose_attrs_.group),
                                           mode, MiopenTensor::GetDataType<HipT>()));

      if (has_bias) {
        const auto& b_shape = p.B->Shape();
        ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
        std::vector<int64_t> b_dims(2 + p.kernel_shape.size());
        b_dims[0] = 1;           // N
        b_dims[1] = b_shape[0];  // C
        for (size_t i = 0; i < p.kernel_shape.size(); i++)
          b_dims[2 + i] = 1;

        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, MiopenTensor::GetDataType<HipT>()));
      }

      y_data = reinterpret_cast<HipT*>(p.Y->MutableData<T>());

      if (!s_.cached_benchmark_bwd_results.contains(x_dims)) {
        IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize, context->GetComputeStream());

        miopenConvAlgoPerf_t perf;
        int algo_count = 1;
        MIOPEN_RETURN_IF_ERROR(miopenFindConvolutionBackwardDataAlgorithm(
            GetMiopenHandle(context),
            s_.x_tensor,
            x_data,
            s_.w_desc,
            w_data,
            s_.conv_desc,
            s_.y_tensor,
            y_data,
            1,
            &algo_count,
            &perf,
            algo_search_workspace.get(),
            AlgoSearchWorkspaceSize,
            false));
        s_.cached_benchmark_bwd_results.insert(x_dims, {perf.bwd_data_algo, perf.memory});
      }

      const auto& perf = s_.cached_benchmark_bwd_results.at(x_dims);
      s_.bwd_data_algo = perf.bwd_data_algo;
      s_.workspace_bytes = perf.memory;
    }

    // The following block will be executed in case there has been no change in the shapes of the
    // input and the filter compared to the previous run
    if (!y_data) {
      auto y_dims = s_.y_dims.AsShapeVector();
      if (x_dimensions == 3) {
        y_dims.erase(y_dims.begin() + 2);
      }
      Tensor* Y = context->Output(0, TensorShape(y_dims));
      y_data = reinterpret_cast<HipT*>(Y->MutableData<T>());

      // Bail out early if one of the output dimensions is zero.
      if (Y->Shape().Size() == 0) {
        return Status::OK();
      }
    }

    const auto alpha = Consts<HipT>::One;
    const auto beta = Consts<HipT>::Zero;

    IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes, context->GetComputeStream());

    MIOPEN_RETURN_IF_ERROR(
        miopenConvolutionBackwardData(
            GetMiopenHandle(context),
            &alpha,
            s_.x_tensor,
            x_data,
            s_.w_desc,
            w_data,
            s_.conv_desc,
            s_.bwd_data_algo,
            &beta,
            s_.y_tensor,
            y_data,
            workspace.get(),
            s_.workspace_bytes));

    if (has_bias) {
      const Tensor* B = dynamic_padding ? context->Input<Tensor>(3) : context->Input<Tensor>(2);
      auto b_data = reinterpret_cast<const HipT*>(B->Data<T>());
      MIOPEN_RETURN_IF_ERROR((miopenConvolutionForwardBias(GetMiopenHandle(context), &alpha, s_.b_tensor, b_data, &beta, s_.y_tensor, y_data)));
    }
  }

  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
