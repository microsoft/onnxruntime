// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_shared.h"

#include "core/platform/ort_mutex.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::cuda {

namespace {

cudnnStatus_t GetWorkspaceSize(const ConvArgs& args, T_BwdDataAlgo algo, size_t* workspace_size) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(args.handle, args.w_desc, args.y_tensor, args.conv_desc,
                                                      args.x_tensor, algo, workspace_size);
}

cudnnStatus_t GetWorkspaceSize(const ConvArgs& args, T_BwdFilterAlgo algo, size_t* workspace_size) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(args.handle, args.x_tensor, args.y_tensor, args.conv_desc,
                                                        args.w_desc, algo, workspace_size);
}

cudnnStatus_t GetWorkspaceSize(const ConvArgs& args, T_FwdAlgo algo, size_t* workspace_size) {
  return cudnnGetConvolutionForwardWorkspaceSize(args.handle, args.x_tensor, args.w_desc, args.conv_desc,
                                                 args.y_tensor, algo, workspace_size);
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

// TODO: Currently we use global AlgoPerfCache for ConvGrad and ConvTransposeGrad only.
// Conv's perf cache is still per node.
// Need to apply such global cache for Conv, and move some shared code from here to conv.h/cc.
AlgoPerfCache<T_BwdDataPerf> bwd_data_algos;
AlgoPerfCache<T_BwdFilterPerf> bwd_filter_algos;
AlgoPerfCache<T_FwdPerf> fwd_algos;

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
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
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
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");

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

template <>
struct AlgoSearch<T_FwdPerf> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static AlgoPerfCache<T_FwdPerf>& Cache() { return fwd_algos; }
  static Status FindAlgorithms(const ConvArgs& args, const CUDAExecutionProvider* provider, const AllocatorPtr& allocator,
                               std::vector<T_FwdPerf>& perf_results) {
    static const T_FwdAlgo algos[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };

    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");

    std::unique_ptr<T_FwdPerf[]> candidates = std::make_unique<T_FwdPerf[]>(num_algos);
    int perf_count;
    if (args.params.algo_mode == OrtCudnnConvAlgoSearchHeuristic) {
      CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(args.handle, args.x_tensor, args.w_desc,
                                                                   args.conv_desc, args.y_tensor, num_algos,
                                                                   &perf_count, candidates.get()));
    } else if (args.params.algo_mode == OrtCudnnConvAlgoSearchExhaustive) {
      size_t max_workspace_size = provider->GetCudnnConvUseMaxWorkspace() ? GetMaxWorkspaceSize(args, algos, num_algos)
                                                                          : AlgoSearchWorkspaceSize;
      // Use GetTransientScratchBuffer() so the workspace can be freed instead of cached.
      // Because the benchmarking uses a huge amount of memory, e.g. a few GBs.
      IAllocatorUniquePtr<void> workspace = max_workspace_size == 0
                                                ? nullptr
                                                : IAllocator::MakeUniquePtr<void>(allocator, max_workspace_size, true);
      CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
          args.handle, args.x_tensor, args.x_data, args.w_desc, args.w_data, args.conv_desc, args.y_tensor,
          args.y_data, num_algos, &perf_count, candidates.get(), workspace.get(), max_workspace_size));
    } else {
      ORT_ENFORCE(false, "Algo mode should be EXHAUSTIVE (0) or HEURISTIC (1), but got ", args.params.algo_mode);
    }
    perf_results = GetValidAlgorithms<T_FwdPerf>(candidates.get(), perf_count);
    return Status::OK();
  }
};

}  // namespace

size_t ConvParamsHash::operator()(const ConvParams& conv_params) const {
  auto ptr = reinterpret_cast<const uint8_t*>(&conv_params);
  uint32_t value = 0x811C9DC5;
  for (int i = 0; i < static_cast<int>(sizeof(ConvParams)); ++i) {
    value ^= ptr[i];
    value *= 0x01000193;
  }
  return static_cast<size_t>(value);
}

bool ConvParamsEqual::operator()(const ConvParams& a, const ConvParams& b) const {
  auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
  auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
  return memcmp(ptr1, ptr2, sizeof(ConvParams)) == 0;
}

template <typename T_Perf>
Status AlgoIterator<T_Perf>::OnlyDefaultAlgorithm(const ConvArgs& args, std::vector<T_Perf>& perf_results, bool use_tf32) {
  perf_results.resize(1);
  perf_results[0].algo = AlgoSearch<T_Perf>::DEFAULT_ALGO;
  if (args.params.data_type == CUDNN_DATA_HALF) {
    perf_results[0].mathType = CUDNN_TENSOR_OP_MATH;
  } else if (args.params.data_type == CUDNN_DATA_FLOAT && !use_tf32) {
    perf_results[0].mathType = CUDNN_FMA_MATH;
  } else {
    perf_results[0].mathType = CUDNN_DEFAULT_MATH;
  }
  CUDNN_RETURN_IF_ERROR(GetWorkspaceSize(args, perf_results[0].algo, &(perf_results[0].memory)));
  return Status::OK();
}

template <typename T_Perf>
Status AlgoIterator<T_Perf>::TryAll(const CUDAExecutionProvider* provider, const AllocatorPtr& allocator,
                                    std::function<Status(const T_Perf& perf)> f) {
  auto& cache = AlgoSearch<T_Perf>::Cache();

  if (T_Perf algo_perf; cache.Find(args_.params, &algo_perf) && f(algo_perf) == Status::OK()) {
    return Status::OK();
  }

  std::vector<T_Perf> perf_results;
  ORT_RETURN_IF_ERROR(args_.params.algo_mode == OrtCudnnConvAlgoSearchDefault
                          ? OnlyDefaultAlgorithm(args_, perf_results, provider->UseTF32())
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

template class AlgoIterator<T_BwdDataPerf>;
template class AlgoIterator<T_BwdFilterPerf>;
template class AlgoIterator<T_FwdPerf>;

}  // namespace onnxruntime::cuda
