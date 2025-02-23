/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4100)
#pragma warning(disable : 4101)  // equivalent to GCC's -Wunused-variable
#pragma warning(disable : 4189)  // equivalent to GCC's -Wunused-but-set-variable
#endif

#include "contrib_ops/cuda/llm/common/cublasMMWrapper.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <string>
#include <vector>

#include "contrib_ops/cuda/llm/gemm_runner.h"

#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/gemmUtils.h"
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/cudaCoreGemm.h"
#include "contrib_ops/cuda/llm/common/debugUtils.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "contrib_ops/cuda/llm/runtime/nv_infer_runtime.h"
#include "contrib_ops/cuda/llm/runtime/iBuffer.h"
#include "contrib_ops/cuda/llm/gemm_profiler.cc"

#include <cassert>

using namespace onnxruntime::llm::nvinfer1;
using namespace onnxruntime::llm::common;

namespace onnxruntime::llm {

using CublasGemmWrapper = onnxruntime::llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class CublasLtGemmPluginProfiler
    : public GemmPluginProfiler<cublasLtMatmulHeuristicResult_t, CublasGemmWrapperPtr, GemmIdCublas, GemmIdCublasHash> {
 public:
  using Config = cublasLtMatmulHeuristicResult_t;

  void setTranspose(bool transposeA, bool transposeB) {
    transA_ = transposeA;
    transB_ = transposeB;
  }

  void setPadLd(int padLda, int padLdb, int padLdc) {
    pad_lda_ = padLda;
    pad_ldb_ = padLdb;
    pad_ldc_ = padLdc;
  }

  void setInputType(nvinfer1::DataType type) {
    input_dtype_ = type;
  }

  void setOutputType(nvinfer1::DataType type) {
    output_dtype_ = type;
  }

 protected:
  void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

  void computeTmpSize(size_t maxM, size_t n, size_t k) override;

  bool checkTactic(int m, int n, int k, Config const& tactic) const override;

  std::vector<Config> getTactics(int m, int n, int k) const override;

 private:
  bool transA_;
  bool transB_;
  int pad_lda_;
  int pad_ldb_;
  int pad_ldc_;

  nvinfer1::DataType input_dtype_;
  nvinfer1::DataType output_dtype_;

  static constexpr size_t ALIGNMENT = 256;
};

void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb,
                      int& m, int& n, int& k, int& lda, int& ldb, int& ldc,
                      bool transA, bool transB, int M, int N, int K, int pad_lda, int pad_ldb, int pad_ldc) {
  transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  m = N;
  n = M;
  k = K;
  lda = transB ? K + pad_ldb : N + pad_ldb;
  ldb = transA ? M + pad_lda : K + pad_lda;
  ldc = N + pad_ldc;
}

void runGemm(int const M, int const N, int const K,
             bool const transA, bool const transB,
             int const pad_lda, int const pad_ldb, int const pad_ldc,
             nvinfer1::DataType const /*input_dtype*/,
             CublasGemmWrapperPtr const& cublasWrapperPtr,
             void const* act,
             void const* weight,
             float const alpha,
             void* output,
             std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic,
             void* workspace,
             cudaStream_t stream) {
  if (M == 0 || N == 0 || K == 0)
    return;

  cublasWrapperPtr->setStream(stream);
  cublasWrapperPtr->setWorkspace(workspace);

  cublasOperation_t transa, transb;
  int m, n, k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K, pad_lda, pad_ldb, pad_ldc);

  cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
  cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, alpha, 0.0f, heuristic);
  cublasWrapperPtr->destroyDescriptors();
}

void CublasLtGemmPluginProfiler::runTactic(
    int m, int n, int k,
    CublasLtGemmPluginProfiler::Config const& tactic,
    char* workspace,
    cudaStream_t const& stream) {
  size_t dataSize = sizeof(half);
  if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
    dataSize = sizeof(float);
  }

  void* actPtr = reinterpret_cast<void*>(workspace);
  void* weightPtr = reinterpret_cast<void*>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(actPtr), m * k * dataSize, ALIGNMENT));
  void* outputPtr = reinterpret_cast<void*>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(weightPtr), n * k * dataSize, ALIGNMENT));
  char* workspacePtr = reinterpret_cast<char*>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(outputPtr), m * (n + pad_ldc_) * dataSize, ALIGNMENT));
  runGemm(m, n, k, transA_, transB_, pad_lda_, pad_ldb_, pad_ldc_,
          input_dtype_, mRunner, actPtr, weightPtr, 1.0f, outputPtr, {tactic}, workspacePtr, stream);
}

bool CublasLtGemmPluginProfiler::checkTactic(int m, int n, int k, Config const& tactic) const {
  cublasOperation_t transa, transb;
  int M = m, N = n, K = k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA_, transB_, M, N, K, pad_lda_, pad_ldb_, pad_ldc_);

  mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

  auto const checkResult = mRunner->checkTactic(transa, transb, m, n, k, lda, ldb, ldc, tactic.algo);

  mRunner->destroyDescriptors();

  return checkResult;
}

void CublasLtGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k) {
  size_t dataSize = getDTypeSize(input_dtype_);
  size_t outputDataSize = getDTypeSize(output_dtype_);

  std::vector<size_t> workspaces = {
      maxM * k * dataSize,                     // A
      n * k * dataSize,                        // B
      maxM * (n + pad_ldc_) * outputDataSize,  // C
      CUBLAS_WORKSPACE_SIZE                    // workspace
  };
  size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), ALIGNMENT);
  setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<CublasLtGemmPluginProfiler::Config> CublasLtGemmPluginProfiler::getTactics(int M, int N, int K) const {
  cublasOperation_t transa, transb;
  int m, n, k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA_, transB_, M, N, K, pad_lda_, pad_ldb_, pad_ldc_);

  mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
  auto const heruistics = mRunner->getTactics(transa, transb, m, n, k, lda, ldb, ldc);
  mRunner->destroyDescriptors();

  return heruistics;
}

class GemmRunner::Impl {
 public:
  void Initialize(int pad_lda,
                  int pad_ldb,
                  int pad_ldc,
                  bool transA,
                  bool transB,
                  int output_dtype,
                  float alpha);

  void Run(const onnxruntime::Tensor* X, const onnxruntime::Tensor* W, onnxruntime::Tensor* Y, void* workspace,
           cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublasLt_handle);

  void Configure(const onnxruntime::TensorShape& min_x, const onnxruntime::TensorShape& max_x, const onnxruntime::TensorShape& w);

  TensorShape GetOutputShape(const onnxruntime::TensorShape& x, const onnxruntime::TensorShape& w) const;

  GemmPluginProfilerManager<CublasLtGemmPluginProfiler> gemmPluginProfileManager;
  std::shared_ptr<CublasLtGemmPluginProfiler> plugin_profiler_;
  static thread_local CublasGemmWrapperPtr cublas_wrapper_;

 private:
  bool transA_;
  bool transB_;
  int pad_lda_;
  int pad_ldb_;
  int pad_ldc_;
  nvinfer1::DataType output_dtype_;
  float alpha_{1.f};

  GemmDims profile_dims{};
  GemmIdCublas gemm_id{};
};

GemmRunner::GemmRunner() : pImpl(new GemmRunner::Impl()) {
}

void GemmRunner::Initialize(int pad_lda, int pad_ldb, int pad_ldc, bool transA, bool transB, int output_dtype, float alpha) {
  pImpl->Initialize(pad_lda, pad_ldb, pad_ldc, transA, transB, output_dtype, alpha);
}

std::unique_ptr<IGemmRunner> GemmRunner::Create(int pad_lda, int pad_ldb, int pad_ldc, bool transA, bool transB, int output_dtype, float alpha) {
#ifdef _MSC_VER
  auto runner = std::make_unique<GemmRunner>();
#else
  // Linux build has error using make_unique: invalid application of ‘sizeof’ to incomplete type.
  std::unique_ptr<IGemmRunner> runner;
  runner.reset(new GemmRunner());
#endif
  reinterpret_cast<GemmRunner*>(runner.get())->Initialize(pad_lda, pad_ldb, pad_ldc, transA, transB, output_dtype, alpha);
  return runner;
}

size_t GemmRunner::GetWorkspaceSize() const {
  return CUBLAS_WORKSPACE_SIZE;
}

void GemmRunner::Run(const onnxruntime::Tensor* X, const onnxruntime::Tensor* W, onnxruntime::Tensor* Y,
                     void* workspace,
                     cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublasLt_handle) {
  pImpl->Run(X, W, Y, workspace, stream, cublas_handle, cublasLt_handle);
}

void GemmRunner::Configure(const TensorShape& min_x, const TensorShape& max_x, const TensorShape& w) {
  pImpl->Configure(min_x, max_x, w);
}

TensorShape GemmRunner::GetOutputShape(TensorShape const& x, TensorShape const& w) const {
  return pImpl->GetOutputShape(x, w);
}

TensorShape GemmRunner::Impl::GetOutputShape(TensorShape const& x, TensorShape const& w) const {
  std::vector<int64_t> output_shape;
  const auto num_dims_x = x.NumDimensions();
  const auto num_dims_w = w.NumDimensions();
  output_shape.reserve(num_dims_x + num_dims_w - 2);
  if (transA_) {
    for (size_t i = 1; i < num_dims_x; ++i) {
      output_shape[i - 1] = x.GetDims()[i];
    }
  } else {
    for (size_t i = 0; i + 1 < num_dims_x; ++i) {
      output_shape[i] = x.GetDims()[i];
    }
  }
  if (transB_) {
    for (size_t i = 0; i + 1 < num_dims_w; ++i) {
      output_shape[num_dims_x - 1 + i] = w.GetDims()[i] + pad_ldc_;
    }
  } else {
    for (size_t i = 1; i < num_dims_w; ++i) {
      output_shape[num_dims_x - 2 + i] = w.GetDims()[i] + pad_ldc_;
    }
  }
  return TensorShape(output_shape);
}

// ONNX_NAMESPACE::TensorProto_DataType in_type
nvinfer1::DataType GetTrtDataType(int dtype) {
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return nvinfer1::DataType::kFLOAT;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return nvinfer1::DataType::kHALF;
#ifdef ENABLE_BF16
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return nvinfer1::DataType::kBF16;
#endif
#ifdef ENABLE_FP8
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      return nvinfer1::DataType::kFP8;
#endif
    default:
      ORT_THROW("Not implemented for ", dtype);
  }
}


void GemmRunner::Impl::Initialize(int pad_lda, int pad_ldb, int pad_ldc, bool transA, bool transB, int output_dtype, float alpha) {
  pad_lda_ = pad_lda;
  pad_ldb_ = pad_ldb;
  pad_ldc_ = pad_ldc;
  transA_ = transA;
  transB_ = transB;
  output_dtype_ = GetTrtDataType(output_dtype);
  alpha_ = alpha;
  plugin_profiler_ = std::make_shared<CublasLtGemmPluginProfiler>();
  plugin_profiler_->setOutputType(output_dtype_);
  plugin_profiler_->setPadLd(pad_lda_, pad_ldb_, pad_ldc_);
}

cudaDataType_t GetOutputCublasType(int out_type) {
  switch (out_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return CUDA_R_32F;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return CUDA_R_16F;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return CUDA_R_16BF;
    default:
      ORT_THROW("Not supported output data type for cuBLAS ", out_type);
  }
}

nvinfer1::DataType SetGemmConfig(
    CublasGemmWrapperPtr& cublas_wrapper,
    int in_type,     //ONNX_NAMESPACE::TensorProto_DataType& in_type,
    cudaDataType_t output_cublas_type) {
  nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
  switch (in_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      cublas_wrapper->setFP32GemmConfig();
      input_type = nvinfer1::DataType::kFLOAT;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      cublas_wrapper->setFP16GemmConfig(output_cublas_type);
      input_type = nvinfer1::DataType::kHALF;
      break;
#ifdef ENABLE_BF16
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      cublas_wrapper->setBF16GemmConfig(output_cublas_type);
      input_type = nvinfer1::DataType::kBF16;
      break;
#endif
#ifdef ENABLE_FP8
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      cublas_wrapper->setBF8GemmConfig(output_cublas_type);
      input_type = nvinfer1::DataType::kFP8;
      break;
#endif
    default:
      ORT_THROW("Not implemented for ", in_type);
  }

  return input_type;
}

inline int64_t computeMDimension(bool transA, const TensorShape& shape) {
  return transA ? shape.SizeFromDimension(1) : shape.SizeToDimension(shape.NumDimensions() - 1);
}

inline int64_t computeNDimension(bool transB, const TensorShape& shape) {
  return transB ? shape.SizeToDimension(shape.NumDimensions() - 1) : shape.SizeFromDimension(1);
}

void GemmRunner::Impl::Configure(const TensorShape& min_x, const TensorShape& max_x, const TensorShape& w) {
  auto const min_M = computeMDimension(transA_, min_x);
  auto const max_M = computeMDimension(transA_, max_x);
  auto const N = computeNDimension(transB_, w);
  auto const K = static_cast<int64_t>(transA_ ? max_x.GetDims()[0] : max_x.GetDims()[max_x.NumDimensions() - 1]);

  if (!profile_dims.isInitialized()) {
    profile_dims = {min_M, max_M, N, K};
  }

  gemm_id.n = N;
  gemm_id.k = K;
}

// Initialize the static thread_local member variable.
thread_local CublasGemmWrapperPtr GemmRunner::Impl::cublas_wrapper_ = nullptr;

void GemmRunner::Impl::Run(const onnxruntime::Tensor* X, const onnxruntime::Tensor* W, onnxruntime::Tensor* Y,
                           void* workspace, cudaStream_t stream,
                           cublasHandle_t cublas_handle, cublasLtHandle_t cublasLt_handle) {
  int const pad_m = transA_ ? pad_lda_ : 0;
  int const pad_n = transB_ ? 0 : pad_ldb_;
  int const pad_k = transA_ ? 0 : pad_lda_;
  const int64_t M = computeMDimension(transA_, X->Shape()) - pad_m;
  const int64_t N = computeNDimension(transB_, W->Shape()) - pad_n;
  const int64_t K = transA_ ? X->Shape().GetDims()[0] - pad_k : X->Shape().GetDims()[X->Shape().NumDimensions() - 1] - pad_k;

  // cublas_wrapper_ is thread_local so there is no need to use mutex here.
  if (cublas_wrapper_ == nullptr) {
    cublas_wrapper_ = std::make_shared<CublasMMWrapper>(
      std::make_shared<cublasHandle_t>(cublas_handle),
      std::make_shared<cublasLtHandle_t>(cublasLt_handle), stream, workspace);
  }

  cudaDataType_t output_cublas_type = GetOutputCublasType(Y->GetElementType());
  auto input_type = SetGemmConfig(cublas_wrapper_, X->GetElementType(), output_cublas_type);

  if (gemm_id.n == 0) {  // not configured yet.
    const TensorShape& min_x = X->Shape();
    const TensorShape& max_x = X->Shape();
    Configure(min_x, max_x, W->Shape());
  }

  plugin_profiler_ = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true, /* skip */ true);
  plugin_profiler_->setTranspose(transA_, transB_);

  plugin_profiler_->setOutputType(output_dtype_);

  plugin_profiler_->setPadLd(pad_lda_, pad_ldb_, pad_ldc_);

  gemm_id = GemmIdCublas(profile_dims.n, profile_dims.k, input_type, transA_, transB_, output_dtype_);

  plugin_profiler_->profileTactics(cublas_wrapper_, input_type, profile_dims, gemm_id);

  std::string mnkStr = "MNK={" + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + "}";
  {
    std::string const activationStr = "GEMM layer's activation before GEMM with " + mnkStr;
    TLLM_CHECK_DEBUG_WITH_INFO(
        onnxruntime::llm::runtime::utils::tensorHasInvalid(M, K, input_type, X->DataRaw(), stream, activationStr) == false,
        "Found invalid number (NaN or Inf) in " + activationStr);
  }

  bool cudaKernelFinished = false;
  bool no_pad_dim = pad_m == 0 && pad_n == 0 && pad_k == 0 && pad_ldc_ == 0;

  // TODO: support fp8 (E4M3)
  bool cudaKernelSupportType = input_type == nvinfer1::DataType::kHALF ||
                               input_type == nvinfer1::DataType::kFLOAT ||
                               input_type == nvinfer1::DataType::kBF16;

  bool use_fp8 = input_type == nvinfer1::DataType::kFP8;

  // TODO: sub tensor matmul is not supported in fp8 gemm cuda kernel
  if (M <= 4 && N <= 128000 && use_fp8 && no_pad_dim && cudaKernelSupportType) {
    onnxruntime::llm::common::QuantMode quantMode;  // quantization is not used.
    onnxruntime::llm::kernels::cuda_core_gemm::Params params(
        X->DataRaw(),
        W->DataRaw(),
        alpha_,
        Y->MutableDataRaw(),
        M, N, K, quantMode,
        nvinfer1::DataType::kFP8, output_dtype_);
    cudaKernelFinished = onnxruntime::llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
  } else if (M <= 6 && N <= 128000 && !use_fp8 && no_pad_dim && cudaKernelSupportType) {
    onnxruntime::llm::common::QuantMode quantMode;
    onnxruntime::llm::kernels::cuda_core_gemm::Params params(
        X->DataRaw(),
        W->DataRaw(),
        alpha_,
        Y->MutableDataRaw(),
        M, N, K, quantMode,
        input_type, output_dtype_);
    cudaKernelFinished = onnxruntime::llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
  }

  if (!cudaKernelFinished) {
    auto bestTactic = plugin_profiler_->getBestConfig(M, gemm_id);
    runGemm(M, N, K, transA_, transB_, pad_lda_, pad_ldb_, pad_ldc_, input_type, cublas_wrapper_,
            X->DataRaw(),
            W->DataRaw(),
            alpha_,
            Y->MutableDataRaw(), bestTactic, workspace, stream);
  }

  {
    std::string const outputStr = "GEMM layer's output after GEMM with " + mnkStr;
    TLLM_CHECK_DEBUG_WITH_INFO(
        onnxruntime::llm::runtime::utils::tensorHasInvalid(M, N + pad_ldc_, input_type, Y->MutableDataRaw(), stream, outputStr) == false,
        "Found invalid number (NaN or Inf) in " + outputStr);
  }
}

}  // namespace onnxruntime::llm

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
