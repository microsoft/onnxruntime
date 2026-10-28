// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "unit_lower_triangular_solve.h"
#include "unit_lower_triangular_solve_impl.h"

#include <climits>
#include <mutex>
#include <utility>
#include <cuda.h>

#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"

namespace onnxruntime::contrib::cuda {
namespace {

void LogCleanup(const Status& status) {
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "UnitLowerTriangularSolve cleanup: " << status.ErrorMessage();
  }
}

// Separate fixed-stream handles avoid changing shared GEMM handle state and
// keep cuBLAS workspace ownership independent of concurrent streams.
struct SolveHandle {
  SolveHandle(CUcontext context, cudaStream_t stream) : context(context), stream(stream) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SolveHandle);

  Status Initialize() {
    CUBLAS_RETURN_IF_ERROR(cublasCreate(&handle));
    CUBLAS_RETURN_IF_ERROR(cublasSetStream(handle, stream));
    CUBLAS_RETURN_IF_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_RETURN_IF_ERROR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
    return Status::OK();
  }

  Status Record() {
    const auto status = cudaEventRecord(completion, stream);
    fence_failed = status != cudaSuccess;
    recorded = recorded || status == cudaSuccess;
    return CUDA_CALL(status);
  }

  ~SolveHandle() {
    if (cuCtxPushCurrent(context) != CUDA_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "UnitLowerTriangularSolve could not restore its CUDA context for cleanup.";
      return;
    }
    // The final event also covers submissions that returned an error. Do not
    // destroy workspace still referenced by a queued cuBLAS kernel.
    auto retired = cudaSuccess;
    if (fence_failed) {
      retired = cudaDeviceSynchronize();
    } else if (recorded) {
      retired = cudaEventSynchronize(completion);
    }
    LogCleanup(CUDA_CALL(retired));
    if (retired == cudaSuccess) {
      if (handle) LogCleanup(CUBLAS_CALL(cublasDestroy(handle)));
      if (completion) LogCleanup(CUDA_CALL(cudaEventDestroy(completion)));
    }
    CUcontext previous = nullptr;
    if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "UnitLowerTriangularSolve could not restore the previous CUDA context.";
    }
  }

  CUcontext context;
  cudaStream_t stream;
  cublasHandle_t handle = nullptr;
  cudaEvent_t completion = nullptr;
  std::mutex mutex;
  bool recorded = false;
  bool fence_failed = false;
};

using HandleKey = std::pair<CUcontext, unsigned long long>;

Status AcquireHandle(HandleKey key, cudaStream_t stream,
                     std::shared_ptr<SolveHandle>& result) {
  static std::mutex mutex;
  static InlinedHashMap<HandleKey, std::weak_ptr<SolveHandle>> registry;
  std::lock_guard<std::mutex> lock(mutex);
  for (auto it = registry.begin(); it != registry.end();) {
    if (it->second.expired()) {
      registry.erase(it++);
    } else {
      ++it;
    }
  }
  auto found = registry.find(key);
  if (found != registry.end()) {
    result = found->second.lock();
    if (result) return Status::OK();
  }
  result = std::make_shared<SolveHandle>(key.first, stream);
  ORT_RETURN_IF_ERROR(result->Initialize());
  registry[key] = result;
  return Status::OK();
}

struct Submission {
  explicit Submission(SolveHandle& handle) : handle(handle), lock(handle.mutex) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Submission);
  ~Submission() {
    if (!finished) LogCleanup(handle.Record());
  }
  Status Finish() {
    ORT_RETURN_IF_ERROR(handle.Record());
    finished = true;
    return Status::OK();
  }
  SolveHandle& handle;
  std::unique_lock<std::mutex> lock;
  bool finished = false;
};

}  // namespace

struct UnitLowerTriangularSolve::Impl {
  Impl() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Impl);

  Status GetHandle(cudaStream_t stream, std::shared_ptr<SolveHandle>& result) {
    CUcontext context = nullptr;
    const auto status = cuCtxGetCurrent(&context);
    ORT_RETURN_IF(status != CUDA_SUCCESS || context == nullptr, "CUDA context unavailable: ", status);
    // Opaque stream pointers can be reused after destruction; the per-thread
    // default-stream sentinel also has the same value in different threads.
    // CUDA stream IDs distinguish both cases without rebinding a live handle.
    unsigned long long stream_id = 0;
#if CUDART_VERSION >= 13000
    CUDA_RETURN_IF_ERROR(cudaStreamGetId(stream, &stream_id));
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "UnitLowerTriangularSolve requires CUDA 13.0 or later.");
#endif
    const HandleKey key{context, stream_id};
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = handles.find(key);
    if (found != handles.end()) {
      result = found->second;
      return Status::OK();
    }
    ORT_RETURN_IF_ERROR(AcquireHandle(key, stream, result));
    handles.emplace(key, result);
    return Status::OK();
  }
  std::mutex mutex;
  InlinedHashMap<HandleKey, std::shared_ptr<SolveHandle>> handles;
};

UnitLowerTriangularSolve::UnitLowerTriangularSolve(const OpKernelInfo& info)
    : CudaKernel(info), impl_(std::make_unique<Impl>()) {}
UnitLowerTriangularSolve::~UnitLowerTriangularSolve() = default;

#if CUDART_VERSION >= 13000
ONNX_OPERATOR_KERNEL_EX(
    UnitLowerTriangularSolve, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    UnitLowerTriangularSolve);
#endif

Status UnitLowerTriangularSolve::ComputeInternal(OpKernelContext* context) const {
  const auto* a = context->Input<Tensor>(0);
  const auto* b = context->Input<Tensor>(1);
  const auto& as = a->Shape();
  const auto& bs = b->Shape();
  const auto rank = as.NumDimensions();
  ORT_RETURN_IF_NOT(rank >= 2 && rank == bs.NumDimensions(), "A and B require equal rank >= 2.");
  const auto m = as[rank - 1];
  const auto n = bs[rank - 1];
  ORT_RETURN_IF_NOT(as[rank - 2] == m, "A must be square.");
  ORT_RETURN_IF_NOT(bs[rank - 2] == m, "B row count must match A.");
  for (size_t i = 0; i + 2 < rank; ++i) {
    ORT_RETURN_IF_NOT(as[i] == bs[i], "Leading dimensions must match exactly; no broadcasting.");
  }
  const auto batch = as.SizeToDimension(rank - 2);
  ORT_RETURN_IF_NOT(m <= INT_MAX && n <= INT_MAX && batch <= INT_MAX,
                    "cuBLAS dimensions and batch must fit int32.");
  auto* x = context->Output(0, bs);
  if (batch == 0 || m == 0 || n == 0) return Status::OK();

  const size_t ac = SafeInt<size_t>(batch) * m * m;
  const size_t bc = SafeInt<size_t>(batch) * m * n;
  const size_t bytes = (SafeInt<size_t>(ac) + bc) * sizeof(float);
  const size_t pointer_bytes = SafeInt<size_t>(batch) * sizeof(float*);
  const size_t aligned = (SafeInt<size_t>(bytes) + alignof(float*) - 1) / alignof(float*) * alignof(float*);
  const size_t scratch_bytes = SafeInt<size_t>(aligned) + SafeInt<size_t>(pointer_bytes) * 2;
  auto* ort_stream = GetComputeStream(context);
  const auto stream = Stream(context);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
  ORT_RETURN_IF(capture != cudaStreamCaptureStatusNone,
                "UnitLowerTriangularSolve does not support CUDA graph capture.");
  std::shared_ptr<SolveHandle> handle;
  ORT_RETURN_IF_ERROR(impl_->GetHandle(stream, handle));
  Submission submission(*handle);
  // Keep the column-major matrices adjacent, with aligned device pointer
  // tables. ORT's stream-aware allocator owns scratch reuse and retirement.
  auto scratch = GetScratchBuffer<uint8_t>(scratch_bytes, ort_stream);
  auto* ca = reinterpret_cast<float*>(scratch.get());
  auto* cb = ca + ac;
  auto* ap = reinterpret_cast<const float**>(scratch.get() + aligned);
  auto* bp = reinterpret_cast<float**>(scratch.get() + aligned + pointer_bytes);
  const int rows = static_cast<int>(m);
  const int columns = static_cast<int>(n);
  const int batches = static_cast<int>(batch);
  ORT_RETURN_IF_ERROR(UnitLowerTriangularSolveToColumn(stream, a->Data<float>(), ca, ac, rows, rows));
  ORT_RETURN_IF_ERROR(UnitLowerTriangularSolveToColumn(stream, b->Data<float>(), cb, bc, rows, columns));
  ORT_RETURN_IF_ERROR(UnitLowerTriangularSolvePointers(stream, ca, cb, ap, bp,
                                                       batches, rows, columns));
  const float alpha = 1;
  CUBLAS_RETURN_IF_ERROR(cublasStrsmBatched(handle->handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT, rows, columns, &alpha,
                                            ap, rows, bp, rows, batches));
  ORT_RETURN_IF_ERROR(UnitLowerTriangularSolveFromColumn(stream, cb, x->MutableData<float>(),
                                                         bc, rows, columns));
  return submission.Finish();
}

}  // namespace onnxruntime::contrib::cuda
