// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/runtime_workload_collector.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <atomic>
#include <limits>
#include <new>
#include <utility>

#include "core/common/safeint.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace {

void IncrementDropCount(uint64_t& count) {
  if (count != std::numeric_limits<uint64_t>::max()) {
    ++count;
  }
}

bool PayloadSize(std::string_view graph_path, std::string_view provider,
                 const MatMulNBitsRuntimeMetadata& metadata, size_t& bytes) {
  size_t dimensions;
  size_t shape_bytes;
  bytes = sizeof(RuntimeWorkloadRecord);
  return SafeAdd(metadata.input_shape.size(), metadata.output_shape.size(), dimensions) &&
         SafeMultiply(dimensions, sizeof(int64_t), shape_bytes) &&
         SafeAdd(bytes, shape_bytes, bytes) &&
         SafeAdd(bytes, graph_path.size(), bytes) &&
         SafeAdd(bytes, provider.size(), bytes) &&
         SafeAdd(bytes, size_t{2}, bytes);
}

}  // namespace

RuntimeWorkloadCollector::RuntimeWorkloadCollector(uint64_t session_id, uint64_t run_id,
                                                   RuntimeWorkloadLimits limits)
    : session_id_(session_id), run_id_(run_id), limits_(limits) {}

Status RuntimeWorkloadCollector::RecordMatMulNBits(std::string_view graph_path, size_t node_index,
                                                   std::string_view provider,
                                                   const MatMulNBitsRuntimeMetadata& metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (records_.size() >= limits_.max_records) {
    stats_.collection_complete = false;
    IncrementDropCount(stats_.record_limit_drops);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector record limit reached.");
  }

  size_t bytes;
  if (!PayloadSize(graph_path, provider, metadata, bytes) ||
      bytes > limits_.max_payload_bytes - stats_.retained_payload_bytes) {
    stats_.collection_complete = false;
    IncrementDropCount(stats_.payload_limit_drops);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector payload limit reached.");
  }

  Status status;
  ORT_TRY {
    RuntimeWorkloadRecord record{records_.size(), std::string(graph_path), node_index,
                                 std::string(provider), metadata};
    records_.push_back(std::move(record));
    stats_.retained_payload_bytes += bytes;
    return Status::OK();
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() {
      stats_.collection_complete = false;
      IncrementDropCount(stats_.allocation_failure_drops);
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload collector allocation failed.");
    });
  }
  return status;
}

Status RuntimeWorkloadCollector::GetSnapshot(RuntimeWorkloadSnapshot& output) const {
  std::lock_guard<std::mutex> lock(mutex_);
  Status status;
  ORT_TRY {
    RuntimeWorkloadSnapshot snapshot{session_id_, run_id_, records_, stats_, run_state_, limits_};
    output = std::move(snapshot);
    return Status::OK();
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Runtime workload snapshot allocation failed.");
    });
  }
  return status;
}

bool RuntimeWorkloadCollector::BeginRun(uint64_t session_id) {
  static std::atomic<uint64_t> next_run_id{1};
  std::lock_guard<std::mutex> lock(mutex_);
  if (run_state_ != RuntimeWorkloadRunState::NotStarted || !records_.empty()) {
    stats_.collection_complete = false;
    IncrementDropCount(stats_.run_reuse_rejections);
    return false;
  }
  session_id_ = session_id;
  run_id_ = next_run_id.fetch_add(1, std::memory_order_relaxed);
  run_state_ = RuntimeWorkloadRunState::Running;
  return true;
}

void RuntimeWorkloadCollector::EndRun(bool succeeded) {
  std::lock_guard<std::mutex> lock(mutex_);
  run_state_ = succeeded ? RuntimeWorkloadRunState::Succeeded : RuntimeWorkloadRunState::Failed;
}

void RuntimeWorkloadCollector::RecordCollectionFailure(bool allocation_failure) {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.collection_complete = false;
  IncrementDropCount(allocation_failure ? stats_.allocation_failure_drops : stats_.metadata_failure_drops);
}

namespace {
std::string GraphPath(const Graph& graph) {
  const auto* parent = graph.ParentNode();
  if (parent == nullptr) {
    return "main";
  }
  for (const auto& [attribute, subgraph] : parent->GetAttributeNameToSubgraphMap()) {
    if (subgraph == &graph) {
      return GraphPath(*graph.ParentGraph()) + "/" + std::to_string(parent->Index()) + "/" + attribute;
    }
  }
  ORT_THROW("Runtime workload subgraph identity is unavailable.");
}
}  // namespace

void RuntimeWorkloadRecorder::RecordCudaMatMulNBits(OpKernelContextInternal& context) const {
  if (node->GetExecutionProviderType() != kCudaExecutionProvider ||
      node->Domain() != kMSDomain || node->OpType() != "MatMulNBits" || node->SinceVersion() != 1) {
    return;
  }
  ORT_TRY {
    const auto* input = context.Input<Tensor>(0);
    const auto* output_value = context.GetOutputMLValue(0);
    ORT_ENFORCE(input != nullptr && output_value != nullptr && output_value->IsTensor(),
                "CUDA MatMulNBits workload tensors are unavailable.");
    const auto& output = output_value->Get<Tensor>();
    const auto& attributes = node->GetAttributes();
    const int64_t k = attributes.at("K").i();
    const int64_t n = attributes.at("N").i();
    const auto& shape = input->Shape();
    ORT_ENFORCE(shape.NumDimensions() > 0 && shape.GetDims().back() == k,
                "CUDA MatMulNBits workload input geometry is invalid.");
    const int64_t m = shape.SizeToDimension(shape.NumDimensions() - 1);
    const bool empty = output.Shape().Size() == 0;
    const auto& cuda_route = context.GetMatMulNBitsCudaRoute();
    RecordMatMulNBits(shape, output.Shape(), m, n, k, 1,
                      context.Input<Tensor>(5) != nullptr, input->GetElementType(),
                      SafeInt<uint32_t>(attributes.at("bits").i()),
                      SafeInt<uint32_t>(attributes.at("block_size").i()),
                      empty        ? std::optional{MatMulNBitsRuntimeRoute::EmptyOutput}
                      : cuda_route ? std::optional{cuda_route->route}
                                   : std::nullopt,
                      empty        ? std::optional{MatMulNBitsRuntimeComputeMode::NotExecuted}
                      : cuda_route ? std::optional{MatMulNBitsRuntimeComputeMode::CudaFloatingPoint}
                                   : std::nullopt,
                      cuda_route ? std::optional{cuda_route->reconstruction_k} : std::nullopt);
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() { collector->RecordCollectionFailure(true); });
  }
  ORT_CATCH(const std::exception&) {
    ORT_HANDLE_EXCEPTION([&]() { collector->RecordCollectionFailure(false); });
  }
}

void RuntimeWorkloadRecorder::RecordMatMulNBits(
    const TensorShape& input_shape, const TensorShape& output_shape,
    int64_t m, int64_t n, int64_t k, size_t batch_count, bool bias_present,
    int32_t input_element_type, uint32_t bits, uint32_t block_size,
    std::optional<MatMulNBitsRuntimeRoute> route, std::optional<MatMulNBitsRuntimeComputeMode> mode,
    std::optional<int64_t> reconstruction_k) const {
  ORT_TRY {
    MatMulNBitsRuntimeMetadata metadata;
    metadata.input_shape.assign(input_shape.GetDims().begin(), input_shape.GetDims().end());
    metadata.output_shape.assign(output_shape.GetDims().begin(), output_shape.GetDims().end());
    metadata.m = m;
    metadata.n = n;
    metadata.k = k;
    metadata.batch_count = batch_count;
    metadata.bias_present = bias_present;
    metadata.input_element_type = input_element_type;
    metadata.bits = bits;
    metadata.block_size = block_size;
    metadata.route = route;
    metadata.compute_mode = mode;
    metadata.reconstruction_k = reconstruction_k;
    ORT_IGNORE_RETURN_VALUE(collector->RecordMatMulNBits(
        GraphPath(*graph), node->Index(), node->GetExecutionProviderType(), metadata));
  }
  ORT_CATCH(const std::bad_alloc&) {
    ORT_HANDLE_EXCEPTION([&]() { collector->RecordCollectionFailure(true); });
  }
  ORT_CATCH(const std::exception&) {
    ORT_HANDLE_EXCEPTION([&]() { collector->RecordCollectionFailure(false); });
  }
}

}  // namespace onnxruntime

#endif
