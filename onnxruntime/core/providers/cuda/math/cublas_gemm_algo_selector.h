#pragma once

#include "core/providers/cuda/cuda_common.h"
#include <functional>
#include <assert.h>
#include <stdlib.h>
using namespace std::chrono;

namespace onnxruntime {
namespace cuda {

static const int MaxGemmAlgoParameters = 5;

struct GemmAlgoKey {
  int parameters[MaxGemmAlgoParameters];
  std::size_t hash;

  GemmAlgoKey(int m, int n, int k, int batch_size, const onnxruntime::NodeIndex node_index) {
    parameters[0] = m;
    parameters[1] = n;
    parameters[2] = k;
    parameters[3] = batch_size;
    parameters[4] = static_cast<int>(node_index);
    hash = CalculateHash();
  }

  bool operator==(const GemmAlgoKey& other) const {
    return 0 == memcmp(parameters, other.parameters, sizeof(int) * MaxGemmAlgoParameters);
  }

  std::size_t GetHash() const { return hash; }

  std::size_t CalculateHash() const {
    using h = std::hash<int>;

    std::size_t hash = 17;
    for (int i = 0; i < MaxGemmAlgoParameters; i++) {
      hash *= 31;
      hash += h()(parameters[i]);
    }
    return hash;
  }
};

struct GemmAlgoKeyHasher {
  std::size_t operator()(const GemmAlgoKey& key) const {
    return key.GetHash();
  }
};

struct GemmAlgoKeyEqualer {
  std::size_t operator()(const GemmAlgoKey& key1, const GemmAlgoKey& key2) const {
    return key1 == key2;
  }
};

struct GemmAlgoPerformance {
  GemmAlgoPerformance() : average_latency(10000), success_times(0), failed_times(0) {}
  double average_latency;
  int success_times;
  int failed_times;

  // A score reflects latency and success rate. Smaller is better.
  double LatencyScore() {
    if (failed_times == 0)
      return average_latency;
    return (10000 * failed_times + average_latency * success_times) / (success_times + failed_times);
  }

  void AddResult(cublasStatus_t status, double latency) {
    if (status == CUBLAS_STATUS_SUCCESS) {
      assert(latency > 0);
      if (success_times == 0) {
        average_latency = latency;
      } else {
        average_latency = (average_latency * success_times + latency) / (success_times + 1);
      }
      success_times++;
    } else {
      failed_times++;
    }
  }
};

struct GemmAlgoValue {
  GemmAlgoValue(){}

  GemmAlgoValue(bool use_tensor_core, const GemmAlgoKey& key, int runs_per_algo) {
    int first = static_cast<int>(use_tensor_core ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);
    int last = static_cast<int>(use_tensor_core ? CUBLAS_GEMM_ALGO15_TENSOR_OP : CUBLAS_GEMM_ALGO23);
    Initialize(first, last, key, runs_per_algo);
  }

  void Initialize(int first, int last, const GemmAlgoKey& key, int runs_per_algo){
    first_algo_ = first;
    last_algo_ = last;
    total_runs_ = 0;
    best_index_ = -1;
    algo_results_.resize(TotalAlgos());
    max_runs_ = TotalAlgos() * runs_per_algo;
    memcpy(key_parameters_, key.parameters, sizeof(int) * MaxGemmAlgoParameters);
  }

  int TotalAlgos() {
    return last_algo_ - first_algo_ + 1;
  }

  int GetNext(bool& is_stable) {
    is_stable = (best_index_ >= 0);
    if (is_stable) {
      return first_algo_ + best_index_;
    }

    // Schedule by round robin, in a order like:
    //    CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_ALGO1 ... CUBLAS_GEMM_ALGO23
    // The default is first one in one loop.
    int next_algo = total_runs_ % TotalAlgos() + first_algo_;
    total_runs_++;
    return next_algo;
  }

  void AddResult(int algo, cublasStatus_t status, double latency) {
    assert(algo >= first_algo_ && algo <= last_algo_);
    algo_results_[algo - first_algo_].AddResult(status, latency);

    if (total_runs_ == max_runs_) {
      best_index_ = 0;
      auto best_score = algo_results_[best_index_].LatencyScore();
      for (int i = 1; i < TotalAlgos(); i++) {
        auto score = algo_results_[i].LatencyScore();
        if (score < best_score) {
          best_index_ = i;
          best_score = score;
        }
      }
      LOGS_DEFAULT(VERBOSE) << " m=" << key_parameters_[0]
                            << " n=" << key_parameters_[1]
                            << " k=" << key_parameters_[2]
                            << " b=" << key_parameters_[3]
                            << " node_index=" << key_parameters_[4]
                            << " first_algo=" << first_algo_
                            << " last_algo=" << last_algo_
                            << " best_index=" << best_index_
                            << " total_runs=" << total_runs_
                            << " max_runs=" << max_runs_;
      for (size_t i = 0; i < algo_results_.size(); i++) {
        const GemmAlgoPerformance& r = algo_results_[i];
        LOGS_DEFAULT(VERBOSE) << " Algo=" << i
                              << " Latency=" << r.average_latency
                              << " Success=" << r.success_times
                              << " Failure=" << r.failed_times;
      }
    }
  }

  std::vector<GemmAlgoPerformance> algo_results_;
  int best_index_;
  int first_algo_;
  int last_algo_;
  int total_runs_;
  int max_runs_;
  int key_parameters_[MaxGemmAlgoParameters];
};

int GetIntegerFromEnv(const char* name, int default_value=0) {
  char* val = nullptr;
#if _MSC_VER
  size_t len;
  if (_dupenv_s(&val, &len, name)) {
    // Something went wrong
    val = nullptr;
  }
#else
  val = getenv(name);
#endif

  if (nullptr != val) {
    return atoi(val);
  }

  return default_value;
}

class CublasGemmAlgoSelector final {
 public:
  CublasGemmAlgoSelector() {
    enable_gemm_profiler = GetIntegerFromEnv("ORT_GEMM_PROFILER") > 0;

    gemm_profiler_runs_per_algo = GetIntegerFromEnv("ORT_GEMM_PROFILER_RUNS");
    if (gemm_profiler_runs_per_algo <= 0) {
      gemm_profiler_runs_per_algo = 20;
    }
  }

  bool IsProfilerEnabled() { return enable_gemm_profiler; }

  cublasGemmAlgo_t GetDefault(bool use_tensor_core) {
    return use_tensor_core ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
  }

  cublasGemmAlgo_t GetNext(bool use_tensor_core, const GemmAlgoKey& key, bool& is_stable) {
    is_stable = false;

    auto iter = gemm_algo_.find(key);
    if (iter != gemm_algo_.end()) {
      int next_algo = iter->second.GetNext(is_stable);
      return static_cast<cublasGemmAlgo_t>(next_algo);
    }

    GemmAlgoValue value(use_tensor_core, key, gemm_profiler_runs_per_algo);
    int next_algo = value.GetNext(is_stable);
    gemm_algo_[key] = value;
    return static_cast<cublasGemmAlgo_t>(next_algo);
  }

  void AddProfileResult(const GemmAlgoKey& key, cublasGemmAlgo_t algo, cublasStatus_t status, double latency) {
    auto iter = gemm_algo_.find(key);
    if (iter != gemm_algo_.end()) {
      iter->second.AddResult(algo, status, latency);
    }
  }

 private:
  bool enable_gemm_profiler;
  int gemm_profiler_runs_per_algo;
  std::unordered_map<GemmAlgoKey, GemmAlgoValue, GemmAlgoKeyHasher, GemmAlgoKeyEqualer> gemm_algo_;
};

#define PROFILE_GEMM(use_tensor_core, gemm_type)                                        \
  do {                                                                                  \
    ::onnxruntime::TimePoint start_time = std::chrono::high_resolution_clock::now();    \
    status = cublasGemmEx(handle,                                                       \
                          transa, transb,                                               \
                          m, n, k,                                                      \
                          alpha,                                                        \
                          A, gemm_type, lda,                                            \
                          B, gemm_type, ldb,                                            \
                          beta,                                                         \
                          C, gemm_type, ldc,                                            \
                          gemm_type,                                                    \
                          algo);                                                        \
    ::onnxruntime::TimePoint end_time = std::chrono::high_resolution_clock::now();      \
    std::chrono::duration<double> duration = end_time - start_time;                     \
    double latency = duration.count();                                                  \
    gemm_algo_selector->AddProfileResult(key, algo, status, latency);                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                              \
      algo = gemm_algo_selector->GetNext(use_tensor_core, key, is_stable);              \
    }                                                                                   \
  } while (status != CUBLAS_STATUS_SUCCESS && algo != default_algo);

inline cublasStatus_t cublasGemmAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const double* alpha, const double* A, int lda, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
  cublasStatus_t status;

  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(false);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, 0, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(false, key, is_stable);
    if (!is_stable) {
      PROFILE_GEMM(false, CUDA_R_64F)
    } else {
      status = cublasGemmEx(handle,
                            transa, transb,
                            m, n, k,
                            alpha,
                            A, CUDA_R_64F, lda,
                            B, CUDA_R_64F, ldb,
                            beta,
                            C, CUDA_R_64F, ldc,
                            CUDA_R_64F,
                            algo);
    }
  } else {
    status = cublasGemmEx(handle,
                          transa, transb,
                          m, n, k,
                          alpha,
                          A, CUDA_R_64F, lda,
                          B, CUDA_R_64F, ldb,
                          beta,
                          C, CUDA_R_64F, ldc,
                          CUDA_R_64F,
                          default_algo);
  }

  return status;
}

inline cublasStatus_t cublasGemmAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
  cublasStatus_t status;

  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(false);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, 0, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(false, key, is_stable);
    if (!is_stable) {  // need profiling
      PROFILE_GEMM(false, CUDA_R_32F)
    } else {
      status = cublasGemmEx(handle,
                            transa, transb,
                            m, n, k,
                            alpha,
                            A, CUDA_R_32F, lda,
                            B, CUDA_R_32F, ldb,
                            beta,
                            C, CUDA_R_32F, ldc,
                            CUDA_R_32F,
                            algo);
    }
  } else {
    status = cublasGemmEx(handle,
                          transa, transb,
                          m, n, k,
                          alpha,
                          A, CUDA_R_32F, lda,
                          B, CUDA_R_32F, ldb,
                          beta,
                          C, CUDA_R_32F, ldc,
                          CUDA_R_32F,
                          default_algo);
  }
  return status;
}

inline cublasStatus_t cublasGemmAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const half* alpha, const half* A, int lda,
    const half* B, int ldb, const half* beta, half* C, int ldc) {
  cublasStatus_t status;

  // Allow to use Tensor Core operations whenever possible.
  CublasMathModeSetter helper(handle, CUBLAS_TENSOR_OP_MATH);
  bool use_tensor_core = (onnxruntime::cuda::DeviceProp().GetDeviceProps().major >= 7);
  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(use_tensor_core);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, 0, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(use_tensor_core, key, is_stable);
    if (!is_stable) {  // need profiling
      PROFILE_GEMM(use_tensor_core, CUDA_R_16F)
    } else {
      status = cublasGemmEx(handle,
                            transa, transb,
                            m, n, k,
                            alpha,
                            A, CUDA_R_16F, lda,
                            B, CUDA_R_16F, ldb,
                            beta,
                            C, CUDA_R_16F, ldc,
                            CUDA_R_16F,
                            algo);
    }
  } else {
    status = cublasGemmEx(handle,
                          transa, transb,
                          m, n, k,
                          alpha,
                          A, CUDA_R_16F, lda,
                          B, CUDA_R_16F, ldb,
                          beta,
                          C, CUDA_R_16F, ldc,
                          CUDA_R_16F,
                          default_algo);
  }

  return status;
}

#define PROFILE_GEMM_BATCH(use_tensor_core, gemm_type)                                   \
  do {                                                                                   \
    ::onnxruntime::TimePoint start_time = std::chrono::high_resolution_clock::now();     \
    status = cublasGemmBatchedEx(handle,                                                 \
                                 transa, transb,                                         \
                                 m, n, k,                                                \
                                 alpha,                                                  \
                                 reinterpret_cast<const void**>(Aarray), gemm_type, lda, \
                                 reinterpret_cast<const void**>(Barray), gemm_type, ldb, \
                                 beta,                                                   \
                                 reinterpret_cast<void**>(Carray), gemm_type, ldc,       \
                                 batchCount,                                             \
                                 gemm_type,                                              \
                                 algo);                                                  \
    ::onnxruntime::TimePoint end_time = std::chrono::high_resolution_clock::now();       \
    std::chrono::duration<double> duration = end_time - start_time;                      \
    double latency = duration.count();                                                   \
    gemm_algo_selector->AddProfileResult(key, algo, status, latency);                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
      algo = gemm_algo_selector->GetNext(use_tensor_core, key, is_stable);               \
    }                                                                                    \
  } while (status != CUBLAS_STATUS_SUCCESS && algo != default_algo);

inline cublasStatus_t cublasGemmBatchedAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float* alpha, const float** Aarray, int lda, const float** Barray, int ldb,
    const float* beta, float** Carray, int ldc, int batchCount) {
  cublasStatus_t status;

  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(false);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, batchCount, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(false, key, is_stable);
    if (!is_stable) {  // need profiling
      PROFILE_GEMM_BATCH(false, CUDA_R_32F)
    } else {
      status = cublasGemmBatchedEx(handle,
                                   transa, transb,
                                   m, n, k,
                                   alpha,
                                   reinterpret_cast<const void**>(Aarray), CUDA_R_32F, lda,
                                   reinterpret_cast<const void**>(Barray), CUDA_R_32F, ldb,
                                   beta,
                                   reinterpret_cast<void**>(Carray), CUDA_R_32F, ldc,
                                   batchCount,
                                   CUDA_R_32F,
                                   algo);
    }
  } else {
    status = cublasGemmBatchedEx(handle,
                                 transa, transb,
                                 m, n, k,
                                 alpha,
                                 reinterpret_cast<const void**>(Aarray), CUDA_R_32F, lda,
                                 reinterpret_cast<const void**>(Barray), CUDA_R_32F, ldb,
                                 beta,
                                 reinterpret_cast<void**>(Carray), CUDA_R_32F, ldc,
                                 batchCount,
                                 CUDA_R_32F,
                                 default_algo);
  }

  return status;
}
inline cublasStatus_t cublasGemmBatchedAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const double* alpha, const double** Aarray, int lda, const double** Barray, int ldb,
    const double* beta, double** Carray, int ldc, int batchCount) {
  cublasStatus_t status;

  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(false);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, batchCount, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(false, key, is_stable);
    if (!is_stable) {  // need profiling
      PROFILE_GEMM_BATCH(false, CUDA_R_64F)
    } else {
      status = cublasGemmBatchedEx(handle,
                                   transa, transb,
                                   m, n, k,
                                   alpha,
                                   reinterpret_cast<const void**>(Aarray), CUDA_R_64F, lda,
                                   reinterpret_cast<const void**>(Barray), CUDA_R_64F, ldb,
                                   beta,
                                   reinterpret_cast<void**>(Carray), CUDA_R_64F, ldc,
                                   batchCount,
                                   CUDA_R_64F,
                                   algo);
    }
  } else {
    status = cublasGemmBatchedEx(handle,
                                 transa, transb,
                                 m, n, k,
                                 alpha,
                                 reinterpret_cast<const void**>(Aarray), CUDA_R_64F, lda,
                                 reinterpret_cast<const void**>(Barray), CUDA_R_64F, ldb,
                                 beta,
                                 reinterpret_cast<void**>(Carray), CUDA_R_64F, ldc,
                                 batchCount,
                                 CUDA_R_64F,
                                 default_algo);
  }

  return status;
}

inline cublasStatus_t cublasGemmBatchedAlgoHelper(
    CublasGemmAlgoSelector* gemm_algo_selector,
    const onnxruntime::NodeIndex node_index,
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const half* alpha, const half** Aarray, int lda, const half** Barray, int ldb,
    const half* beta, half** Carray, int ldc, int batchCount) {
  cublasStatus_t status;

  // Allow to use Tensor Core operations whenever possible.
  CublasMathModeSetter helper(handle, CUBLAS_TENSOR_OP_MATH);
  bool use_tensor_core = (onnxruntime::cuda::DeviceProp().GetDeviceProps().major >= 7);
  cublasGemmAlgo_t default_algo = gemm_algo_selector->GetDefault(use_tensor_core);
  if (gemm_algo_selector->IsProfilerEnabled()) {
    bool is_stable = false;
    GemmAlgoKey key(m, n, k, batchCount, node_index);
    cublasGemmAlgo_t algo = gemm_algo_selector->GetNext(use_tensor_core, key, is_stable);
    if (!is_stable) {  // need profiling
      PROFILE_GEMM_BATCH(use_tensor_core, CUDA_R_16F)
    } else {
      status = cublasGemmBatchedEx(handle,
                                   transa, transb,
                                   m, n, k,
                                   alpha,
                                   reinterpret_cast<const void**>(Aarray), CUDA_R_16F, lda,
                                   reinterpret_cast<const void**>(Barray), CUDA_R_16F, ldb,
                                   beta,
                                   reinterpret_cast<void**>(Carray), CUDA_R_16F, ldc,
                                   batchCount,
                                   CUDA_R_16F,
                                   algo);
    }
  } else {
    status = cublasGemmBatchedEx(handle,
                                 transa, transb,
                                 m, n, k,
                                 alpha,
                                 reinterpret_cast<const void**>(Aarray), CUDA_R_16F, lda,
                                 reinterpret_cast<const void**>(Barray), CUDA_R_16F, ldb,
                                 beta,
                                 reinterpret_cast<void**>(Carray), CUDA_R_16F, ldc,
                                 batchCount,
                                 CUDA_R_16F,
                                 default_algo);
  }

  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
