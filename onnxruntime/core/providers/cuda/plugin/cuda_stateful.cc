// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_stateful.h"
#include "cuda_greedy.h"

#include <atomic>
#include <memory>

#include "cuda_plugin_utils.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"

namespace onnxruntime::cuda_plugin {
namespace {

struct Context {
  std::atomic<size_t> references{1};
  int device_id;
  cudaStream_t stream;
  Context(int device, cudaStream_t value) : device_id(device), stream(value) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Context);
};

void ORT_API_CALL Retain(void* pointer) noexcept {
  ++static_cast<Context*>(pointer)->references;
}

void ORT_API_CALL Release(void* pointer) noexcept {
  auto* context = static_cast<Context*>(pointer);
  if (--context->references == 0) delete context;
}

void CopyRows(void* destination, size_t destination_pitch, const void* source, size_t source_pitch,
              size_t width, size_t rows, cudaStream_t stream) {
  if (width != 0) {
    for (size_t row = 0; row < rows; ++row) {
      const size_t destination_offset = SafeInt<size_t>(row) * destination_pitch;
      const size_t source_offset = SafeInt<size_t>(row) * source_pitch;
      PL_CUDA_CALL_THROW(cudaMemcpyAsync(static_cast<char*>(destination) + destination_offset,
                                         static_cast<const char*>(source) + source_offset,
                                         width, cudaMemcpyDeviceToDevice, stream));
    }
  }
}

OrtStatus* Work(const Context& context, const OrtStatefulCacheOperationRequest& request,
                OrtStatefulCacheOperationResult& result, Ort::Value& storage) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  const auto& spec = *request.spec;
  const auto operation = request.operation;
  const int64_t incoming =
      request.value ? Ort::ConstValue(request.value).GetTensorTypeAndShapeInfo().GetShape()[spec.sequence_axis] : 0;
  result = {sizeof(result), nullptr, request.length, request.capacity, 0, 0, 0};
  if (operation == ORT_STATEFUL_CACHE_SHARE || operation == ORT_STATEFUL_CACHE_RESET) {
    OrtValue* retained = nullptr;
    Ort::ThrowOnError(request.retain_value(
        operation == ORT_STATEFUL_CACHE_SHARE ? request.value : request.storage, &retained));
    storage = Ort::Value(retained);
    result.length = operation == ORT_STATEFUL_CACHE_SHARE ? incoming : 0;
    result.capacity = operation == ORT_STATEFUL_CACHE_SHARE ? incoming : request.capacity;
    result.shared_updates = operation == ORT_STATEFUL_CACHE_SHARE ? 1 : 0;
    return nullptr;
  }

  result.length = operation == ORT_STATEFUL_CACHE_INITIALIZE ? 0
                  : operation == ORT_STATEFUL_CACHE_RESERVE  ? request.length
                  : operation == ORT_STATEFUL_CACHE_APPEND
                      ? static_cast<int64_t>(SafeInt<int64_t>(request.length) + incoming)
                      : incoming;
  result.capacity = operation == ORT_STATEFUL_CACHE_INITIALIZE || operation == ORT_STATEFUL_CACHE_RESERVE
                        ? request.requested_capacity
                        : std::max(request.capacity, result.length);
  InlinedVector<int64_t> shape(spec.shape, spec.shape + spec.rank);
  shape[spec.sequence_axis] = result.capacity;
  SafeInt<int64_t> elements = result.capacity;
  for (size_t i = 0; i < spec.rank; ++i) {
    if (i != static_cast<size_t>(spec.sequence_axis)) elements *= spec.shape[i];
  }
  const size_t bytes = SafeInt<size_t>(static_cast<int64_t>(elements)) *
                       (spec.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 2);
  storage = Ort::Value::CreateTensor(request.allocator, shape.data(), shape.size(), spec.element_type);
  void* destination = storage.GetTensorMutableData<void>();
  ORT_ENFORCE(bytes == 0 || destination != nullptr, "CUDA KV cache allocation failed");
  result.allocations = bytes != 0 ? 1 : 0;
  if (operation == ORT_STATEFUL_CACHE_APPEND) {
    SafeInt<size_t> rows = 1;
    SafeInt<size_t> inner = spec.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 2;
    for (size_t i = 0; i < spec.rank; ++i) {
      if (i < static_cast<size_t>(spec.sequence_axis)) rows *= spec.shape[i];
      if (i > static_cast<size_t>(spec.sequence_axis)) inner *= spec.shape[i];
    }
    const size_t old_row = inner * request.length;
    const size_t added_row = inner * incoming;
    const size_t new_row = inner * result.length;
    CopyRows(destination, new_row, Ort::ConstValue(request.logical).GetTensorRawData(), old_row,
             old_row, rows, context.stream);
    CopyRows(static_cast<char*>(destination) + old_row, new_row,
             Ort::ConstValue(request.value).GetTensorRawData(), added_row, added_row, rows, context.stream);
    result.bytes_copied = SafeInt<size_t>(rows) * new_row;
  } else if (operation != ORT_STATEFUL_CACHE_INITIALIZE) {
    const Ort::ConstValue source(operation == ORT_STATEFUL_CACHE_RESERVE ? request.logical : request.value);
    result.bytes_copied = source.GetTensorSizeInBytes();
    CopyRows(destination, result.bytes_copied, source.GetTensorRawData(), result.bytes_copied,
             result.bytes_copied, 1, context.stream);
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL Cache(void* pointer, const OrtStatefulCacheOperationRequest* request,
                              OrtStatefulCacheOperationResult* out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  ORT_ENFORCE(request->struct_size >= offsetof(OrtStatefulCacheOperationRequest, retain_value) +
                                          sizeof(request->retain_value) &&
                  request->retain_value,
              "CUDA cache requires the retaining cache-request extension");
  auto& context = *static_cast<Context*>(pointer);
  int previous = 0;
  PL_CUDA_CALL_THROW(cudaGetDevice(&previous));
  PL_CUDA_CALL_THROW(cudaSetDevice(context.device_id));
  Ort::Value storage(nullptr);
  OrtStatefulCacheOperationResult result{};
  Ort::Status status(Work(context, *request, result, storage));
  // Complete even failed work before its temporary allocations can be released.
  Ort::Status completion = StatusFromCudaError(cudaStreamSynchronize(context.stream));
  Ort::Status restore = StatusFromCudaError(cudaSetDevice(previous));
  Ort::Status pending = StatusFromCudaError(cudaGetLastError());
  Ort::ThrowOnError(std::move(status));
  Ort::ThrowOnError(std::move(completion));
  Ort::ThrowOnError(std::move(restore));
  Ort::ThrowOnError(std::move(pending));
  result.storage = storage.release();
  *out = result;
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* GreedyWork(const Context& context, const OrtStatefulGreedyRequest& request,
                      Ort::Value& scratch, GpuGreedyResult& result) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  ORT_ENFORCE(request.struct_size >= offsetof(OrtStatefulGreedyRequest, allocator) + sizeof(request.allocator) &&
                  request.allocator,
              "CUDA greedy requires the selected Session scratch allocator extension");
  const Ort::ConstValue logits(request.logits);
  const auto metadata = logits.GetTensorTypeAndShapeInfo();
  const auto shape = metadata.GetShape();
  const size_t width = SafeInt<size_t>(shape.back());
  const int64_t row = request.has_position ? request.position : shape.size() == 3 ? shape[1] - 1
                                                                                  : 0;
  const size_t begin = SafeInt<size_t>(row) * width;
  const size_t end = SafeInt<size_t>(begin) + width;
  const int64_t scratch_shape[]{kGreedyWorkspaceBytes};
  scratch = Ort::Value::CreateTensor<uint8_t>(request.allocator, scratch_shape, 1);
  void* workspace = scratch.GetTensorMutableData<void>();
  ORT_ENFORCE(workspace, "CUDA greedy scratch allocation failed");
  const GpuGreedyResult* device_result = nullptr;
  PL_CUDA_RETURN_IF_ERROR(LaunchGreedy(context.stream, logits.GetTensorRawData(),
                                       metadata.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                       metadata.GetElementCount(), begin, end, workspace, device_result));
  PL_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&result, device_result, sizeof(result),
                                          cudaMemcpyDeviceToHost, context.stream));
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL Greedy(void* pointer, const OrtStatefulGreedyRequest* request, int64_t* out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  auto& context = *static_cast<Context*>(pointer);
  int previous = 0;
  PL_CUDA_CALL_THROW(cudaGetDevice(&previous));
  PL_CUDA_CALL_THROW(cudaSetDevice(context.device_id));
  Ort::Value scratch(nullptr);
  GpuGreedyResult result{};
  Ort::Status status(GreedyWork(context, *request, scratch, result));
  Ort::Status completion = StatusFromCudaError(cudaStreamSynchronize(context.stream));
  Ort::Status restore = StatusFromCudaError(cudaSetDevice(previous));
  Ort::Status pending = StatusFromCudaError(cudaGetLastError());
  Ort::ThrowOnError(std::move(status));
  Ort::ThrowOnError(std::move(completion));
  Ort::ThrowOnError(std::move(restore));
  Ort::ThrowOnError(std::move(pending));
  ORT_ENFORCE(!result.has_nan, "Greedy sampling does not accept NaN logits");
  *out = result.token;
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

}  // namespace

OrtStatus* RegisterCudaStatefulProvider(OrtEp* ep, int device_id, cudaStream_t stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  std::unique_ptr<Context, decltype(&Release)> context(new Context(device_id, stream), &Release);
  const OrtStatefulProviderDescriptor descriptor{
      sizeof(descriptor),
      {sizeof(OrtStatefulProviderCapabilities), 1, (uint64_t{1} << (ORT_STATEFUL_SAMPLE_GREEDY + 1)) - 1,
       (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
       (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
       uint32_t{1} << OrtMemoryInfoDeviceType_GPU, device_id},
      context.get(),
      &Cache,
      &Greedy,
      &Retain,
      &Release};
  return Ort::GetEpApi().RegisterStatefulProvider(ep, &descriptor);
  EXCEPTION_TO_STATUS_END
}

}  // namespace onnxruntime::cuda_plugin
