// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/ort_mutex.h"

// The AlgoPerfCache and AlgoSearch here for Conv/ConvGrad is referenced on PyTorch's implementation
// from aten/src/ATen/native/cudnn/Conv_v7.cpp.

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ConvGrad, kMSDomain, 1, T, kCudaExecutionProvider,                                   \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                ConvGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

using T_BwdDataPerf = cudnnConvolutionBwdDataAlgoPerf_t;
using T_BwdDataAlgo = cudnnConvolutionBwdDataAlgo_t;
using T_BwdFilterPerf = cudnnConvolutionBwdFilterAlgoPerf_t;
using T_BwdFilterAlgo = cudnnConvolutionBwdFilterAlgo_t;

cudnnStatus_t GetWorkspaceSize(const ConvArgs& args, T_BwdDataAlgo algo, size_t* workspace_size) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(args.handle, args.w_desc, args.y_tensor, args.conv_desc,
                                                      args.x_tensor, algo, workspace_size);
}

cudnnStatus_t GetWorkspaceSize(const ConvArgs& args, T_BwdFilterAlgo algo, size_t* workspace_size) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(args.handle, args.x_tensor, args.y_tensor, args.conv_desc,
                                                        args.w_desc, algo, workspace_size);
}

template <typename T_Algo>
size_t GetMaxWorkspaceSize(const ConvArgs& args, const T_Algo* algo, int n_algo) {
  // Calling cudaMemGetInfo is not ideal, but our cuda allocator doesn't have a way to get this info.
  size_t free, total;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
  // Assuming 10% of fragmentation.
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);
  size_t max_workspace_size = 0;
  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t status;
    size_t workspace_size;
    status = GetWorkspaceSize(args, algo[i], &workspace_size);
    if (CUDNN_STATUS_SUCCESS != status || workspace_size == 0 || workspace_size < max_workspace_size ||
        workspace_size > free)
      continue;
    max_workspace_size = workspace_size;
  }

  return max_workspace_size;
}

template <typename T_Perf>
std::vector<T_Perf> GetValidAlgorithms(const T_Perf* perf_results, int n_algo) {
  std::vector<T_Perf> result;
  result.reserve(n_algo);
  for (int i = 0; i < n_algo; i++) {
    T_Perf perf = perf_results[i];
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      result.emplace_back(perf);
    }
  }
  ORT_ENFORCE(result.size() > 0, "No valid convolution algorithms available in CuDNN");
  // TODO: This is a cuDNN bug that gave wrong results in certain strided convolution gradient setups
  // when cuDNN version < 7.5. Need to add handling for such special case.
  return result;
}

struct ConvParamsHash {
  // ConvParams must be a POD because we read out its memory constant as char* when hashing.
  static_assert(std::is_pod<ConvParams>::value, "ConvParams is not POD");
  size_t operator()(const ConvParams& conv_params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&conv_params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(ConvParams)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

struct ConvParamsEqual {
  // ConvParams must be a POD because we read out its memory constant as char* when hashing.
  static_assert(std::is_pod<ConvParams>::value, "ConvParams is not POD");
  bool operator()(const ConvParams& a, const ConvParams& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(ConvParams)) == 0;
  }
};

template <typename T_Perf>
struct AlgoPerfCache {
  mutable OrtMutex mutex;
  std::unordered_map<ConvParams, T_Perf, ConvParamsHash, ConvParamsEqual> map;

  bool Find(const ConvParams& params, T_Perf* result) {
    std::lock_guard<OrtMutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *result = it->second;
    return true;
  }

  void Insert(const ConvParams& params, const T_Perf& algo_perf) {
    std::lock_guard<OrtMutex> guard(mutex);
    map[params] = algo_perf;
  }
};

// TODO: Currently we use global AlgoPerfCache for ConvGrad only. Conv's perf cache is till per node.
// Need to apply such global cache for Conv, and move some shared code from here to conv.h/cc.
AlgoPerfCache<T_BwdDataPerf> bwd_data_algos;
AlgoPerfCache<T_BwdFilterPerf> bwd_filter_algos;

template <typename T_Perf>
struct AlgoSearch {};

template <>
struct AlgoSearch<T_BwdDataPerf> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static AlgoPerfCache<T_BwdDataPerf>& Cache() { return bwd_data_algos; }
  static Status FindAlgorithms(const ConvArgs& args, const CUDAExecutionProvider* provider, const AllocatorPtr& allocator,
                               std::vector<T_BwdDataPerf>& perf_results) {
    static const T_BwdDataAlgo algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    ORT_ENFORCE(sizeof(algos) / sizeof(algos[0]) == num_algos, "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<T_BwdDataPerf[]> candidates = std::make_unique<T_BwdDataPerf[]>(num_algos);
    if (args.params.algo_mode == OrtCudnnConvAlgoSearchHeuristic) {
      CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(args.handle, args.w_desc, args.y_tensor,
                                                                        args.conv_desc, args.x_tensor, num_algos,
                                                                        &perf_count, candidates.get()));
    } else if (args.params.algo_mode == OrtCudnnConvAlgoSearchExhaustive) {
      size_t max_workspace_size = provider->GetCudnnConvUseMaxWorkspace() ? GetMaxWorkspaceSize(args, algos, num_algos)
                                                                          : AlgoSearchWorkspaceSize;
      // Use GetTransientScratchBuffer() so the workspace can be freed instead of cached.
      // Because the benchmarking uses a huge amount of memory, e.g. a few GBs.
      IAllocatorUniquePtr<void> workspace = max_workspace_size == 0 ? nullptr : IAllocator::MakeUniquePtr<void>(allocator, max_workspace_size, true);
      CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle, args.w_desc, args.w_data, args.y_tensor, args.dy_data, args.conv_desc, args.x_tensor,
          args.dx_data, num_algos, &perf_count, candidates.get(), workspace.get(), max_workspace_size));
    } else {
      ORT_ENFORCE(false, "Algo mode should be EXHAUSTIVE (0) or HEURISTIC (1), but got ", args.params.algo_mode);
    }
    perf_results = GetValidAlgorithms<T_BwdDataPerf>(candidates.get(), perf_count);
    return Status::OK();
  }
};

template <>
struct AlgoSearch<T_BwdFilterPerf> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  static AlgoPerfCache<T_BwdFilterPerf>& Cache() { return bwd_filter_algos; }
  static Status FindAlgorithms(const ConvArgs& args, const CUDAExecutionProvider* provider, const AllocatorPtr& allocator,
                               std::vector<T_BwdFilterPerf>& perf_results) {
    static const T_BwdFilterAlgo algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };

    // NOTE: - 1 because ALGO_WINOGRAD is not implemented.
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    ORT_ENFORCE(sizeof(algos) / sizeof(algos[0]) == num_algos, "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<T_BwdFilterPerf[]> candidates = std::make_unique<T_BwdFilterPerf[]>(num_algos);
    int perf_count;
    if (args.params.algo_mode == OrtCudnnConvAlgoSearchHeuristic) {
      CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(args.handle, args.x_tensor, args.y_tensor,
                                                                          args.conv_desc, args.w_desc, num_algos,
                                                                          &perf_count, candidates.get()));
    } else if (args.params.algo_mode == OrtCudnnConvAlgoSearchExhaustive) {
      size_t max_workspace_size = provider->GetCudnnConvUseMaxWorkspace() ? GetMaxWorkspaceSize(args, algos, num_algos)
                                                                          : AlgoSearchWorkspaceSize;
      // Use GetTransientScratchBuffer() so the workspace can be freed instead of cached.
      // Because the benchmarking uses a huge amount of memory, e.g. a few GBs.
      IAllocatorUniquePtr<void> workspace = max_workspace_size == 0 ? nullptr : IAllocator::MakeUniquePtr<void>(allocator, max_workspace_size, true);
      CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle, args.x_tensor, args.x_data, args.y_tensor, args.dy_data, args.conv_desc, args.w_desc,
          args.dw_data, num_algos, &perf_count, candidates.get(), workspace.get(), max_workspace_size));
    } else {
      ORT_ENFORCE(false, "Algo mode should be EXHAUSTIVE (0) or HEURISTIC (1), but got ", args.params.algo_mode);
    }
    perf_results = GetValidAlgorithms<T_BwdFilterPerf>(candidates.get(), perf_count);
    return Status::OK();
  }
};

template <typename T_Perf>
class AlgoIterator {
 public:
  AlgoIterator(const ConvArgs& args) : args_(args) {}

  static Status OnlyDefaultAlgorithm(const ConvArgs& args, std::vector<T_Perf>& perf_results) {
    perf_results.resize(1);
    perf_results[0].algo = AlgoSearch<T_Perf>::DEFAULT_ALGO;
    if (args.params.data_type == CUDNN_DATA_HALF) {
      perf_results[0].mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perf_results[0].mathType = CUDNN_DEFAULT_MATH;
    }
    CUDNN_RETURN_IF_ERROR(GetWorkspaceSize(args, perf_results[0].algo, &(perf_results[0].memory)));
    return Status::OK();
  }

  Status TryAll(const CUDAExecutionProvider* provider, const AllocatorPtr& allocator, std::function<Status(const T_Perf& perf)> f) {
    auto& cache = AlgoSearch<T_Perf>::Cache();

    if (T_Perf algo_perf; cache.Find(args_.params, &algo_perf) && f(algo_perf) == Status::OK()) {
      return Status::OK();
    }

    std::vector<T_Perf> perf_results;
    ORT_RETURN_IF_ERROR(args_.params.algo_mode == OrtCudnnConvAlgoSearchDefault
                            ? OnlyDefaultAlgorithm(args_, perf_results)
                            : AlgoSearch<T_Perf>::FindAlgorithms(args_, provider, allocator, perf_results));
    for (auto& algo_perf : perf_results) {
      if (f(algo_perf) == Status::OK()) {
        cache.Insert(args_.params, algo_perf);
        return Status::OK();
      }
    }
    ORT_ENFORCE(false, "Unable to find a valid cuDNN algorithm to run convolution.");
    return Status::OK();
  }

 private:
  const ConvArgs& args_;
};

template <typename T>
Status ConvGrad<T>::PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX,
                                Tensor* dW, cudnnHandle_t cudnn_handle) const {
  const TensorShape& x_shape = x.Shape();
  auto x_dims = x_shape.AsShapeVector();
  args_.x_data = reinterpret_cast<const CudaT*>(x.template Data<T>());

  const TensorShape& dy_shape = dY.Shape();
  auto dy_dims = dy_shape.AsShapeVector();
  args_.dy_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());

  const TensorShape& w_shape = w.Shape();
  auto w_dims = w_shape.AsShapeVector();
  args_.w_data = reinterpret_cast<const CudaT*>(w.template Data<T>());

  args_.db_data = dB ? reinterpret_cast<CudaT*>(dB->template MutableData<T>()) : nullptr;
  args_.dx_data = dX ? reinterpret_cast<CudaT*>(dX->template MutableData<T>()) : nullptr;
  args_.dw_data = dW ? reinterpret_cast<CudaT*>(dW->template MutableData<T>()) : nullptr;

  bool x_dims_changed = (args_.last_x_dims != x_dims);
  bool w_dims_changed = (args_.last_w_dims != w_dims);
  if (x_dims_changed || w_dims_changed) {
    if (x_dims_changed) args_.last_x_dims = x_dims;
    if (w_dims_changed) args_.last_w_dims = w_dims;

    // Update Attributes
    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&x, &w));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
    auto rank = kernel_shape.size();

    ConvPadVector pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(rank * 2, 0);
    }

    TensorShapeVector dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(rank, 1);
    }

    TensorShapeVector strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(rank, 1);
    }

    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

    // cuDNN only takes 4D or 5D x tensor, so pad dimensions if needed.
    if (rank < 2) {
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims.insert(x_dims.begin() + 2, 1);
        dy_dims.insert(dy_dims.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims.push_back(1);
        dy_dims.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    memset(&args_.params, 0, sizeof(ConvParams));
    args_.params.device_id = static_cast<int8_t>(cuda_ep->GetDeviceId());
    args_.params.data_type = CudnnTensor::GetDataType<CudaT>();
    args_.params.input_dim = static_cast<uint8_t>(x_dims.size());
    for (size_t i = 0; i < x_dims.size(); i++) {
      args_.params.input_size[i] = static_cast<int>(x_dims[i]);
      args_.params.weight_size[i] = static_cast<int>(w_dims[i]);
    }
    for (size_t i = 0; i < rank; i++) {
      args_.params.padding[i] = static_cast<int>(pads[i]);
      args_.params.padding[i + rank] = static_cast<int>(pads[i + rank]);
      args_.params.stride[i] = static_cast<int>(strides[i]);
      args_.params.dilation[i] = static_cast<int>(dilations[i]);
    }
    args_.params.groups = conv_attrs_.group;
    int algo_mode = cuda_ep->GetCudnnConvAlgo();
    ORT_ENFORCE(algo_mode > -1 && algo_mode < 3,
                "Algo mode should be EXHAUSTIVE (0), HEURISTIC (1) or DEFAULT (2), but got ", algo_mode);
    args_.params.algo_mode = algo_mode;

    args_.handle = cudnn_handle;
    ORT_RETURN_IF_ERROR(args_.w_desc.Set(w_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.x_tensor.Set(x_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.y_tensor.Set(dy_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                            gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                            args_.params.data_type));

    if (dB) {
      const TensorShape& db_shape = dB->Shape();
      ORT_RETURN_IF_NOT(db_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> db_dims(2 + kernel_shape.size(), 1);
      db_dims[1] = db_shape[0];
      ORT_RETURN_IF_ERROR(args_.b_tensor.Set(db_dims, CudnnTensor::GetDataType<CudaT>()));
    }
  }

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);
  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {W->Shape()[0]});
  ORT_RETURN_IF_ERROR(PrepareArgs(*X, *dY, *W, dB, dX, dW, GetCudnnHandle(context)));
  if (dX) ORT_RETURN_IF_ERROR(ComputeInputGradient(context->GetComputeStream()));
  if (dW) ORT_RETURN_IF_ERROR(ComputeWeightGradient(context->GetComputeStream()));
  if (dB) ORT_RETURN_IF_ERROR(ComputeBiasGradient());
  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInputGradient(onnxruntime::Stream* stream) const {
  return AlgoIterator<T_BwdDataPerf>(args_).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_BwdDataPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardData(
            args_.handle, &one, args_.w_desc, args_.w_data, args_.y_tensor, args_.dy_data, args_.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args_.x_tensor, args_.dx_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvGrad<T>::ComputeWeightGradient(onnxruntime::Stream* stream) const {
  return AlgoIterator<T_BwdFilterPerf>(args_).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_BwdFilterPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardFilter(
            args_.handle, &one, args_.x_tensor, args_.x_data, args_.y_tensor, args_.dy_data, args_.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args_.w_desc, args_.dw_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvGrad<T>::ComputeBiasGradient() const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardBias(args_.handle, &one, args_.y_tensor, args_.dy_data, &zero,
                                                     args_.b_tensor, args_.db_data));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
