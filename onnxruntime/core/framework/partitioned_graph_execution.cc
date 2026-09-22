// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/partitioned_graph_execution.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <thread>

#include "core/framework/bfc_arena.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

namespace onnxruntime {

#ifdef ORT_ENABLE_STREAM
namespace {
struct TensorSignature {
  explicit TensorSignature(const Tensor& tensor, bool remember_contents = false)
      : address(tensor.DataRaw()), shape(tensor.Shape()), type(tensor.DataType()), device(tensor.Location().device) {
    if (remember_contents && tensor.SizeInBytes() != 0) {
      const auto* bytes = static_cast<const char*>(tensor.DataRaw());
      contents.assign(bytes, bytes + tensor.SizeInBytes());
    }
  }

  Status Check(const OrtValue& value) const {
    ORT_RETURN_IF_NOT(value.IsTensor(), "Partitioned CUDA graphs require tensor values.");
    const auto& tensor = value.Get<Tensor>();
    ORT_RETURN_IF(tensor.DataRaw() != address || tensor.Shape() != shape ||
                      tensor.DataType() != type || tensor.Location().device != device,
                  "Partitioned CUDA graph tensor address, shape, type, or device changed. Use a new gpu_graph_id.");
    ORT_RETURN_IF(!contents.empty() &&
                      std::memcmp(contents.data(), tensor.DataRaw(), contents.size()) != 0,
                  "A CPU control input to a captured CUDA partition changed. Use a new gpu_graph_id.");
    return Status::OK();
  }

  const void* address;
  TensorShape shape;
  MLDataType type;
  OrtDevice device;
  InlinedVector<char> contents;
};

struct Partition {
  bool capture;
  InlinedVector<NodeIndex> nodes;
};

struct CapturedPartition {
  CapturedPartition(int id, gsl::span<const AllocatorPtr> allocators)
      : graph_id(id), scratch(allocators) {}
  int graph_id;
  ArenaAllocationCapture scratch;
  InlinedVector<std::pair<int, TensorSignature>> signatures;
};

struct CapturedRun {
  explicit CapturedRun(const SessionState& session_state) : streams(&session_state) {}
  // Scratch and streams outlive the execution frame.
  DeviceStreamCollectionHolder streams;
  InlinedVector<std::unique_ptr<CapturedPartition>> partitions;
  std::unique_ptr<ExecutionFrame> frame;
  InlinedVector<int> feed_indices;
  InlinedVector<int> fetch_indices;
  InlinedVector<TensorSignature> feed_signatures;
  bool ready{false};
};
}  // namespace

struct PartitionedGraphExecution::Impl {
  Impl(const SessionState& state, IExecutionProvider& ep) : session_state(state), provider(ep) {
    const auto& graph = session_state.GetGraphViewer();
    for (NodeIndex index : graph.GetNodesInTopologicalOrder(state.GetSessionOptions().execution_order)) {
      const auto& node = *graph.GetNode(index);
      bool capture = node.GetExecutionProviderType() == provider.Type() && !utils::IsMemcpyNode(node);
      if (partitions.empty() || partitions.back().capture != capture) {
        partitions.push_back({capture, {}});
      }
      partitions.back().nodes.push_back(index);
    }
    for (const auto& [device, allocator] : state.GetAllocators()) {
      if (device.Type() == OrtDevice::GPU) {
        ORT_ENFORCE(allocator->AsArena() != nullptr,
                    "Partitioned CUDA capture requires a CUDA arena allocator.");
        allocators.push_back(allocator);
      }
    }
    ORT_ENFORCE(!allocators.empty(), "Partitioned CUDA capture requires a CUDA arena.");
  }

  Status Compute(CapturedRun& state, const Partition& partition, const RunOptions& options,
                 const logging::Logger& logger) {
    const auto& plan = *session_state.GetExecutionPlan();
    for (NodeIndex index : partition.nodes) {
      ORT_RETURN_IF(options.terminate, "Partitioned CUDA graph execution was terminated.");
      auto* kernel = session_state.GetKernel(index);
      ORT_RETURN_IF(kernel->IsAsync(), "Partitioned CUDA capture does not support asynchronous host kernels.");
      auto* stream = state.streams.p_->GetStream(plan.node_stream_map_[index]);
      OpKernelContextInternal context(session_state, *state.frame, *kernel, logger, options.terminate, stream);
      ORT_RETURN_IF_ERROR(kernel->Compute(&context));
      if (!partition.capture && utils::IsMemcpyNode(kernel->Node())) {
        // Copies stay outside capture. Complete D2H before CPU reads, and H2D before host buffers can be reused.
        ORT_RETURN_IF_ERROR(provider.Sync());
      }
    }
    return Status::OK();
  }

  Status SaveSignatures(CapturedRun& state, const Partition& partition, CapturedPartition& captured) {
    for (NodeIndex index : partition.nodes) {
      const auto& node = *session_state.GetGraphViewer().GetNode(index);
      int offset = state.frame->GetNodeOffset(index);
      const size_t input_count = node.InputDefs().size() + node.ImplicitInputDefs().size();
      const size_t value_count = input_count + node.OutputDefs().size();
      for (size_t i = 0; i < value_count; ++i) {
        int value_offset = offset + narrow<int>(i);
        const auto* value = state.frame->GetNodeInputOrOutputMLValue(value_offset);
        if (value == nullptr) {
          continue;
        }
        ORT_RETURN_IF_NOT(value->IsTensor() && !value->Get<Tensor>().IsDataTypeString(),
                          "Partitioned CUDA capture supports only non-string tensor values.");
        const auto& tensor = value->Get<Tensor>();
        bool host_input = i < input_count && tensor.Location().device.Type() == OrtDevice::CPU;
        ORT_RETURN_IF(i >= input_count && tensor.Location().device.Type() == OrtDevice::CPU,
                      "Partitioned CUDA capture does not support CUDA kernels producing host outputs.");
        captured.signatures.emplace_back(value_offset, TensorSignature(tensor, host_input));
      }
    }
    return Status::OK();
  }

  Status Execute(CapturedRun& state, const RunOptions& options, const logging::Logger& logger) {
    // Allocate outputs, then retain scratch before capture. Plugin warm-ups use graph ID -1
    // because the plugin may otherwise start capturing before either preparation pass finishes.
    // Its configurable warm-up count is subsequently honored with a bounded capture retry loop.
    const bool plugin = provider.GetOrtEp() != nullptr;
    const int max_capture_attempts = plugin ? 8 : 1;
    for (int pass = state.ready ? 2 : 0; pass < 2 + max_capture_attempts; ++pass) {
      for (size_t i = 0; i < partitions.size(); ++i) {
        const auto& partition = partitions[i];
        ORT_RETURN_IF(options.terminate, "Partitioned CUDA graph execution was terminated.");
        if (!partition.capture) {
          failed = true;
          ORT_RETURN_IF_ERROR(Compute(state, partition, options, logger));
          failed = false;
          continue;
        }
        auto& captured = *state.partitions[i];
        if (pass >= 2 && provider.IsGraphCaptured(captured.graph_id)) {
          for (const auto& [offset, signature] : captured.signatures) {
            const auto* value = state.frame->GetNodeInputOrOutputMLValue(offset);
            ORT_RETURN_IF_NOT(value != nullptr, "A captured partition value is missing.");
            ORT_RETURN_IF_ERROR(signature.Check(*value));
          }
          LOGS(logger, INFO) << "Replaying CUDA partition " << i << " with internal graph id " << captured.graph_id;
          failed = true;
          ORT_RETURN_IF_ERROR(provider.ReplayGraph(captured.graph_id, true));
          failed = false;
          continue;
        }
        ORT_RETURN_IF(state.ready, "CUDA partition graph is missing.");

        RunOptions partition_options;
        const int capture_id = plugin && pass < 2 ? -1 : captured.graph_id;
        ORT_RETURN_IF_ERROR(partition_options.config_options.AddConfigEntry(
            kOrtRunOptionsConfigCudaGraphAnnotation, std::to_string(capture_id).c_str()));
        failed = true;
        ORT_RETURN_IF_ERROR(provider.OnRunStart(partition_options));
        bool ended = false;
        auto end_run = gsl::finally([&]() {
          captured.scratch.Cancel();
          if (!ended) {
            // End even an invalidated capture so subsequent sessions can use the CUDA stream.
            ORT_TRY {
              auto status = provider.OnRunEnd(true, partition_options);
              if (!status.IsOK()) {
                LOGS(logger, ERROR) << status.ErrorMessage();
              }
            }
            ORT_CATCH(const std::exception& ex) {
              ORT_HANDLE_EXCEPTION([&]() { LOGS(logger, ERROR) << ex.what(); });
            }
          }
        });
        if (pass != 0) {
          ORT_RETURN_IF_ERROR(captured.scratch.Begin(pass >= 2));
        }
        ORT_RETURN_IF_ERROR(Compute(state, partition, options, logger));
        if (pass != 0) {
          ORT_RETURN_IF_ERROR(captured.scratch.End());
        }
        ended = true;
        ORT_RETURN_IF_ERROR(provider.OnRunEnd(true, partition_options));
        const bool captured_graph = provider.IsGraphCaptured(captured.graph_id);
        ORT_RETURN_IF(captured_graph && pass < 2,
                      "Unexpected CUDA partition capture warm-up behavior.");
        if (captured_graph) {
          ORT_RETURN_IF_ERROR(SaveSignatures(state, partition, captured));
        }
        failed = false;
      }
      failed = true;
      ORT_RETURN_IF_ERROR(state.streams.p_->CleanUp(true));
      failed = false;
      if (pass >= 2 &&
          std::all_of(state.partitions.begin(), state.partitions.end(),
                      [&](const auto& captured) { return !captured || provider.IsGraphCaptured(captured->graph_id); })) {
        state.ready = true;
        return Status::OK();
      }
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA partition capture did not complete after ",
                           max_capture_attempts, " capture attempts.");
  }

  Status Run(const RunOptions& options, int graph_id, FeedsFetchesManager& manager,
             gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches, const logging::Logger& logger) {
    ORT_RETURN_IF(failed, "A previous partitioned CUDA execution failed; recreate the session.");
    ORT_RETURN_IF(options.only_execute_path_to_fetches || options.sync_stream != nullptr,
                  "Partitioned CUDA capture does not support partial execution or per-run stream overrides.");
    ORT_RETURN_IF_NOT(options.config_options.GetConfigOrDefault(
                                                kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "")
                          .empty(),
                      "Arena shrinking is not supported with partitioned CUDA capture.");
    ORT_RETURN_IF(owner_thread != std::thread::id{} && owner_thread != std::this_thread::get_id(),
                  "Partitioned CUDA graphs must be run on the thread that captured them.");
    const auto& info = manager.GetFeedsFetchesInfo();
    ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(session_state, manager));
    const auto& output_copies = manager.GetFetchesDeviceCopyInfo();
    for (size_t i = 0; i < output_copies.size(); ++i) {
      const auto& expected_device = output_copies[i].source_device;
      if (!fetches.empty() && fetches[i].IsAllocated()) {
        ORT_RETURN_IF_NOT(fetches[i].IsTensor() &&
                              fetches[i].Get<Tensor>().Location().device == expected_device,
                          "Bind partitioned CUDA graph outputs on their producing device using IOBinding.");
      } else {
        ORT_RETURN_IF(output_copies[i].target_device != expected_device,
                      "Bind partitioned CUDA graph outputs on their producing device using IOBinding.");
      }
    }
    auto it = runs.find(graph_id);
    if (it == runs.end()) {
      const auto& copies = manager.GetFeedsDeviceCopyInfo();
      for (size_t i = 0; i < feeds.size(); ++i) {
        ORT_RETURN_IF_NOT(feeds[i].IsTensor() && !feeds[i].Get<Tensor>().IsDataTypeString(),
                          "Partitioned CUDA capture requires non-string tensor inputs.");
        ORT_RETURN_IF(feeds[i].Get<Tensor>().Location().device != copies[i].target_device,
                      "Bind partitioned CUDA graph inputs on their consuming device using IOBinding.");
      }
      auto state = std::make_unique<CapturedRun>(session_state);
      state->feed_indices.assign(info.feeds_mlvalue_idxs.begin(), info.feeds_mlvalue_idxs.end());
      state->fetch_indices.assign(info.fetches_mlvalue_idxs.begin(), info.fetches_mlvalue_idxs.end());
      for (const auto& feed : feeds) {
        state->feed_signatures.emplace_back(feed.Get<Tensor>());
      }
      state->frame = std::make_unique<ExecutionFrame>(
          info.feeds_mlvalue_idxs, feeds, info.fetches_mlvalue_idxs, fetches,
          std::unordered_map<size_t, IExecutor::CustomAllocator>{}, state->streams.p_.get(), session_state);
      for (const auto& partition : partitions) {
        if (partition.capture) {
          ORT_RETURN_IF(next_graph_id == std::numeric_limits<int>::max(), "Too many CUDA partition graphs.");
          state->partitions.push_back(std::make_unique<CapturedPartition>(next_graph_id++, allocators));
        } else {
          state->partitions.push_back(nullptr);
        }
      }
      it = runs.emplace(graph_id, std::move(state)).first;
    }
    auto& state = *it->second;
    ORT_RETURN_IF(state.feed_indices != info.feeds_mlvalue_idxs || state.fetch_indices != info.fetches_mlvalue_idxs,
                  "Partitioned CUDA graph input/output names or order changed. Use a new gpu_graph_id.");
    for (size_t i = 0; i < feeds.size(); ++i) {
      ORT_RETURN_IF_ERROR(state.feed_signatures[i].Check(feeds[i]));
    }
    std::vector<OrtValue> retained_fetches;
    ORT_RETURN_IF_ERROR(state.frame->GetOutputs(retained_fetches));
    for (size_t i = 0; i < fetches.size(); ++i) {
      if (fetches[i].IsAllocated() && retained_fetches[i].IsAllocated()) {
        ORT_RETURN_IF_ERROR(TensorSignature(retained_fetches[i].Get<Tensor>()).Check(fetches[i]));
      }
    }

    owner_thread = std::this_thread::get_id();
    auto invalidate_incomplete_capture = gsl::finally([&]() {
      if (!state.ready) {
        failed = true;
      }
    });
    ORT_RETURN_IF_ERROR(Execute(state, options, logger));
    ORT_RETURN_IF_ERROR(state.frame->GetOutputs(fetches));
    return Status::OK();
  }

  const SessionState& session_state;
  IExecutionProvider& provider;
  InlinedVector<Partition> partitions;
  InlinedVector<AllocatorPtr> allocators;
  InlinedHashMap<int, std::unique_ptr<CapturedRun>> runs;
  std::thread::id owner_thread;
  int next_graph_id{0};
  bool failed{false};
};
#else
struct PartitionedGraphExecution::Impl {};
#endif

PartitionedGraphExecution::PartitionedGraphExecution(const SessionState& state, IExecutionProvider& provider)
#ifdef ORT_ENABLE_STREAM
    : impl_(std::make_unique<Impl>(state, provider))
#endif
{
#ifndef ORT_ENABLE_STREAM
  ORT_UNUSED_PARAMETER(state);
  ORT_UNUSED_PARAMETER(provider);
#endif
}

PartitionedGraphExecution::~PartitionedGraphExecution() = default;

Status PartitionedGraphExecution::Run(const RunOptions& options, int graph_id, FeedsFetchesManager& manager,
                                      gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                                      const logging::Logger& logger) {
#ifdef ORT_ENABLE_STREAM
  return impl_->Run(options, graph_id, manager, feeds, fetches, logger);
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(graph_id);
  ORT_UNUSED_PARAMETER(manager);
  ORT_UNUSED_PARAMETER(feeds);
  ORT_UNUSED_PARAMETER(fetches);
  ORT_UNUSED_PARAMETER(logger);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Partitioned CUDA capture requires stream support.");
#endif
}
}  // namespace onnxruntime
