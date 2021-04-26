// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      ConvGrad,                                                                 \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    cudnnDataType_t dataType,
    const std::vector<int64_t>& input_dim, const std::vector<int64_t>& weight_dim,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation,
    int64_t groups, bool deterministic, bool allow_tf32) {
  memset(params, 0, sizeof(ConvolutionParams));
  params->device_id = at::cuda::current_device();
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  params->input_dim = input_dim.size();
  // params->memory_format = input.suggest_memory_format();
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int)input_dim[i];
    params->weight_size[i] = (int)weight_dim[i];
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(
      args.handle,
      args.w_desc,
      args.o_desc,
      args.c_desc,
      args.i_desc,
      algo,
      sz);
}

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(
      args.handle,
      args.i_desc,
      args.o_desc,
      args.c_desc,
      args.w_desc,
      algo,
      sz);
}

template <typename algo_t>
size_t getMaxWorkspaceSize(const ConvolutionArgs& args,
                           const algo_t* algo, int n_algo) {
  size_t max_ws_size = 0;

  // TODO: get maximum available size from memory areana

  size_t free, total;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
  // Assuming 10% of fragmentation
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);

  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = getWorkspaceSize(args, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > free)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

// TODO: Use something less heavy duty than a big honking mutex
template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<std::vector<int64_t>, T, vector_hash<int64_t>> map;

  bool find(const std::vector<int64_t>& x_dims, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(x_dims);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const std::vector<int64_t>& x_dims, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[x_dims] = results;
  }
};

// BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos;

template <typename T>
Status ConvGrad<T>::getBwdDataAlgoPerf(const std::vector<int64_t>& x_dims, const ConvolutionArgs& args, int cudnn_conv_algo_search,
                                       const void* w, const void* dy, void* dx, cudnnConvolutionBwdDataAlgoPerf_t& perf) const {
  // cudnnConvolutionBwdDataAlgoPerf_t perf;
  if (bwd_data_algos.find(x_dims, &perf)) {
    return Status::OK();
  }

  static const cudnnConvolutionBwdDataAlgo_t kAllBwdDataAlgo[] = {
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
  };

  ORT_ENFORCE(cudnn_conv_algo_search > -1 && cudnn_conv_algo_search < 3,
              "cudnn_conv_algo_search should be 0, 1 or 2, but got ", cudnn_conv_algo_search);

  int algo_count = 1;
  if (cudnn_conv_algo_search == 0) {  // EXHAUSTIVE
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    size_t max_ws_size = getMaxWorkspaceSize(args, kAllBwdDataAlgo, num_algos);
    IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(max_ws_size);
    CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
        args.handle,
        args.w_desc, w,
        args.o_desc, dy,
        args.c_desc,
        args.i_desc, dx,
        1,            // requestedAlgoCount
        &algo_count,  // returnedAlgoCount
        &perf,
        algo_search_workspace.get(),
        max_ws_size));
  } else if (cudnn_conv_algo_search == 1) {  // HEURISTIC
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        args.handle,
        args.w_desc,
        args.o_desc,
        args.c_desc,
        args.i_desc,
        1,            // requestedAlgoCount
        &algo_count,  // returnedAlgoCount
        &perf));
  } else {  // DEFAULT
    perf.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args, perf.algo, &perf.memory));
    if (args.data_type == CUDNN_DATA_HALF) {
      perf.mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perf.mathType = CUDNN_DEFAULT_MATH;
    }
  }

  bwd_data_algos.insert(x_dims, perf);

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::getBwdFilterAlgoPerf(const std::vector<int64_t>& x_dims, const ConvolutionArgs& args, int cudnn_conv_algo_search,
                                         const void* x, const void* dy, void* dw, cudnnConvolutionBwdFilterAlgoPerf_t& perf) const {
  // cudnnConvolutionBwdFilterAlgoPerf_t perf;
  if (bwd_filter_algos.find(x_dims, &perf)) {
    return Status::OK();
  }

  static const cudnnConvolutionBwdFilterAlgo_t kAllBwdFilterAlgo[] = {
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
  };

  ORT_ENFORCE(cudnn_conv_algo_search > -1 && cudnn_conv_algo_search < 3,
              "cudnn_conv_algo_search should be 0, 1 or 2, but got ", cudnn_conv_algo_search);

  int algo_count = 1;
  if (cudnn_conv_algo_search == 0) {  // EXHAUSTIVE
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    size_t max_ws_size = getMaxWorkspaceSize(args, kAllBwdFilterAlgo, num_algos);
    IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(max_ws_size);
    CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        args.handle,
        args.i_desc, x,
        args.o_desc, dy,
        args.c_desc,
        args.w_desc, dw,
        1,            // requestedAlgoCount
        &algo_count,  // returnedAlgoCount
        &perf,
        algo_search_workspace.get(),
        max_ws_size));
  } else if (cudnn_conv_algo_search == 1) {  // HEURISTIC
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        args.handle,
        args.i_desc,
        args.o_desc,
        args.c_desc,
        args.w_desc,
        1,            // requestedAlgoCount
        &algo_count,  // returnedAlgoCount
        &perf));
  } else {  // DEFAULT
    perf.algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args, perf.algo, &perf.memory));
    if (args.data_type == CUDNN_DATA_HALF) {
      perf.mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perf.mathType = CUDNN_DEFAULT_MATH;
    }
  }

  bwd_filter_algos.insert(x_dims, perf);
  return Status::OK();
}

// TODO: we can cache the descriptors, and only update if the input shape changes
template <typename T>
Status ConvGrad<T>::PrepareArgs(const Tensor& input, const Tensor& output, const Tensor& weight, const Tensor* bias) const {
  const TensorShape& i_shape = input.Shape();
  std::vector<int64_t> i_dims = i_shape.GetDims();

  const TensorShape& o_shape = output.Shape();
  std::vector<int64_t> o_dims = o_shape.GetDims();

  const TensorShape& w_shape = weight.Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();

  // Update Attributes
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&input, &weight));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
  auto rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(rank * 2, 0);
  }

  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(rank, 1);
  }

  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(rank, 1);
  }

  // cudnn only takes 4D or 5D input, so pad dimensions if needed
  if (rank < 2) {
    i_dims.push_back(1);
    o_dims.push_back(1);
    w_dims.push_back(1);

    pads.insert(pads.begin() + rank, 0);
    pads.insert(pads.end(), 0);
    kernel_shape.push_back(1);
    strides.push_back(1);
    dilations.push_back(1);
  }

  args_.handle = CudnnHandle();
  args_.data_type = CudnnTensor::GetDataType<CudaT>();
  ORT_RETURN_IF_ERROR(args_.i_desc.Set(i_dims, args_.data_type));
  ORT_RETURN_IF_ERROR(args_.o_desc.Set(o_dims, args_.data_type));
  ORT_RETURN_IF_ERROR(args_.w_desc.Set(w_dims, args_.data_type));
  ORT_RETURN_IF_ERROR(args_.c_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                       gsl::narrow_cast<int>(conv_attrs_.group),
                                       CUDNN_CROSS_CORRELATION, args_.data_type));
  setConvolutionParams(&args_.params, i_dims, w_dims, pads, strides, dilations,
                       conv_attrs_.group, false, false);

  if (bias) {
    const TensorShape& b_shape = bias->Shape();
    ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
    std::vector<int64_t> b_dims(2 + kernel_shape.size(), 1);
    b_dims[1] = b_shape[0];
    ORT_RETURN_IF_ERROR(args_.b_desc.Set(b_dims, args_.data_type));
  }

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  const int64_t M = W->Shape()[0];

  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {M});

  ORT_RETURN_IF_ERROR(PrepareArgs(*X, *dY, *W, dB));

  ORT_RETURN_IF_ERROR(ComputeWeightGradient(dW, dY, X));
  ORT_RETURN_IF_ERROR(ComputeInputGradient(dX, dY, W));
  ORT_RETURN_IF_ERROR(ComputeBiasGradient(dB, dY));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeWeightGradient(Tensor* dW, const Tensor* dY, const Tensor* X) const {
  if (dW == nullptr) return Status::OK();

  void* dw_data = dW->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();
  const void* x_data = X->template Data<T>();

  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  // TODO: implement the algoritm search

  const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
  int cudnn_conv_algo_search = cuda_ep->GetCudnnConvAlgoSearch();

  cudnnConvolutionBwdFilterAlgoPerf_t perf;
  ORT_RETURN_IF_ERROR(getBwdFilterAlgoPerf(x_dims, args_, cudnn_conv_algo_search,
                                           x_data, dy_data, dw_data, perf));

  // cudnnConvolutionBwdFilterAlgoPerf_t perf;
  // perf.algo = kDefaultConvBwdFilterAlgo;f
  // if (args_.data_type == CUDNN_DATA_HALF) {
  //   perf.mathType = CUDNN_TENSOR_OP_MATH;
  // } else {
  //   perf.mathType = CUDNN_DEFAULT_MATH;
  // }
  // CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args_, perf.algo, &perf.memory));
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.c_desc, perf.mathType));

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(perf.memory);

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardFilter(
          args_.handle,
          &one, args_.i_desc, x_data,
          args_.o_desc, dy_data,
          args_.c_desc, perf.algo, workspace.get(), perf.memory,
          &zero, args_.w_desc, dw_data));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInputGradient(Tensor* dX, const Tensor* dY, const Tensor* W) const {
  if (dX == nullptr) return Status::OK();

  void* dx_data = dX->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();
  const void* w_data = W->template Data<T>();

  const TensorShape& x_shape = dX->Shape();
  const auto& x_dims = x_shape.GetDims();

  const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
  int cudnn_conv_algo_search = cuda_ep->GetCudnnConvAlgoSearch();

  cudnnConvolutionBwdDataAlgoPerf_t perf;
  ORT_RETURN_IF_ERROR(getBwdDataAlgoPerf(x_dims, args_, cudnn_conv_algo_search,
                                         w_data, dy_data, dx_data, perf));

  // TODO: implement the algoritm search
  // cudnnConvolutionBwdDataAlgoPerf_t perf;
  // perf.algo = kDefaultConvBwdDataAlgo;
  // if (args_.data_type == CUDNN_DATA_HALF) {
  //   perf.mathType = CUDNN_TENSOR_OP_MATH;
  // } else {
  //   perf.mathType = CUDNN_DEFAULT_MATH;
  // }
  // CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args_, perf.algo, &perf.memory));
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.c_desc, perf.mathType));

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(perf.memory);

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardData(
          args_.handle,
          &one, args_.w_desc, w_data,
          args_.o_desc, dy_data,
          args_.c_desc, perf.algo, workspace.get(), perf.memory,
          &zero, args_.i_desc, dx_data));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeBiasGradient(Tensor* dB, const Tensor* dY) const {
  if (dB == nullptr) return Status::OK();

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  void* db_data = dB->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardBias(
          args_.handle,
          &one, args_.o_desc, dy_data,
          &zero, args_.b_desc, db_data));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime