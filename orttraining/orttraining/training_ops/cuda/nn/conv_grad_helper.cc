// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad_helper.h"

namespace onnxruntime {
namespace cuda {

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
std::vector<T_Perf> GetValidAlgorithms(T_Perf* perf_results, int n_algo) {
  std::vector<T_Perf> result;
  result.reserve(n_algo);
  for (int i = 0; i < n_algo; i++) {
    T_Perf perf = perf_results[i];
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      result.emplace_back(perf);
    }
  }
  ORT_ENFORCE(result.size() > 0, "No valid convolution algorithms available in CuDNN");
  return result;
}

AlgoPerfCache<T_BwdDataPerf> bwd_data_algos;
AlgoPerfCache<T_BwdFilterPerf> bwd_filter_algos;

template <>
struct AlgoSearch<T_BwdDataPerf> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static AlgoPerfCache<T_BwdDataPerf>& Cache() { return bwd_data_algos; }
  static Status FindAlgorithms(const ConvArgs& args, const CUDAExecutionProvider* provider,
                               std::vector<T_BwdDataPerf>& perf_results) {
    static const T_BwdDataAlgo algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    ORT_ENFORCE(sizeof(algos) / sizeof(algos[0]) == num_algos, "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<T_BwdDataPerf[]> candidates(new T_BwdDataPerf[num_algos]);
    if (args.params.algo_mode == OrtCudnnConvAlgoSearch::HEURISTIC) {
      CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(args.handle, args.w_desc, args.y_tensor,
                                                                        args.conv_desc, args.x_tensor, num_algos,
                                                                        &perf_count, candidates.get()));
    } else if (args.params.algo_mode == OrtCudnnConvAlgoSearch::EXHAUSTIVE) {
      size_t max_workspace_size = GetMaxWorkspaceSize(args, algos, num_algos);
      // Use IAllocator's Reserve so the workspace can be freed instead of cached. Because the benchmarking uses a huge
      // amount of memory, e.g. a few GBs.
      IAllocatorUniquePtr<void> workspace = provider->GetScratchBuffer<void>(max_workspace_size, true);
      CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle, args.w_desc, args.w_data, args.y_tensor, args.dy_data, args.conv_desc, args.x_tensor,
          args.dx_data, num_algos, &perf_count, candidates.get(), workspace.get(), max_workspace_size));
    } else {
      ORT_ENFORCE(false, "Algo mode should be 0, 1 or 2, but got ", args.params.algo_mode);
    }
    perf_results = GetValidAlgorithms<T_BwdDataPerf>(candidates.get(), perf_count);
    return Status::OK();
  }
};

template <>
struct AlgoSearch<T_BwdFilterPerf> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  static AlgoPerfCache<T_BwdFilterPerf>& Cache() { return bwd_filter_algos; }
  static Status FindAlgorithms(const ConvArgs& args, const CUDAExecutionProvider* provider,
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
    std::unique_ptr<T_BwdFilterPerf[]> candidates(new T_BwdFilterPerf[num_algos]);
    int perf_count;
    if (args.params.algo_mode == OrtCudnnConvAlgoSearch::HEURISTIC) {
      CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(args.handle, args.x_tensor, args.y_tensor,
                                                                          args.conv_desc, args.w_desc, num_algos,
                                                                          &perf_count, candidates.get()));
    } else if (args.params.algo_mode == OrtCudnnConvAlgoSearch::EXHAUSTIVE) {
      size_t max_workspace_size = GetMaxWorkspaceSize(args, algos, num_algos);
      // Use IAllocator's Reserve so the workspace can be freed instead of cached. Because the benchmarking uses a huge
      // amount of memory, e.g. a few GBs.
      IAllocatorUniquePtr<void> workspace = provider->GetScratchBuffer<void>(max_workspace_size, true);
      CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle, args.x_tensor, args.x_data, args.y_tensor, args.dy_data, args.conv_desc, args.w_desc,
          args.dw_data, num_algos, &perf_count, candidates.get(), workspace.get(), max_workspace_size));
    } else {
      ORT_ENFORCE(false, "Algo mode should be 0, 1 or 2, but got ", args.params.algo_mode);
    }
    perf_results = GetValidAlgorithms<T_BwdFilterPerf>(candidates.get(), perf_count);
    return Status::OK();
  }
};

template <typename T_Perf>
Status AlgoIterator<T_Perf>::TryAll(const CUDAExecutionProvider* provider,
                                    std::function<Status(const T_Perf& perf)> f) {
  auto& cache = AlgoSearch<T_Perf>::Cache();
  T_Perf algo_perf;
  if (cache.Find(args_.params, &algo_perf) && f(algo_perf) == Status::OK()) {
    return Status::OK();
  }

  std::vector<T_Perf> perf_results;
  ORT_RETURN_IF_ERROR(args_.params.algo_mode == OrtCudnnConvAlgoSearch::DEFAULT
                          ? OnlyDefaultAlgorithm(args_, perf_results)
                          : AlgoSearch<T_Perf>::FindAlgorithms(args_, provider, perf_results));
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

}  // namespace cuda
}  // namespace onnxruntime
