// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sequential_executor.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <thread>
#include <vector>
#include <sstream>
#include <algorithm>
#include <mutex>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/stream_execution_context.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/utils.h"
#include "core/platform/env.h"

#if defined DEBUG_NODE_INPUTS_OUTPUTS
#include "core/framework/debug_node_inputs_outputs_utils.h"
#endif

#ifdef ENABLE_NVTX_PROFILE
// This header is for profile using Nvidia's visual profilier.
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif

// #define TRACE_EXECUTION

// Define this symbol to create Concurrency Visualizer markers.
// See https://docs.microsoft.com/en-us/visualstudio/profiling/concurrency-visualizer-sdk
// You will need to install Concurrency Visualizer and add the SDK to the project that compiles this file
// via Analyze->Concurrency Visualizer->Add SDK to Project...
// #define CONCURRENCY_VISUALIZER
#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
using namespace Concurrency;
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include <Windows.h>
#include "core/platform/tracing.h"
namespace {
LARGE_INTEGER OrtGetPerformanceFrequency() {
  LARGE_INTEGER v;
  // On systems that run Windows XP or later, the QueryPerformanceFrequency function will always succeed
  // and will thus never return zero.
  (void)QueryPerformanceFrequency(&v);
  return v;
}

LARGE_INTEGER perf_freq = OrtGetPerformanceFrequency();
}  // namespace
#endif

namespace onnxruntime {

namespace {

constexpr size_t kOpLatencyMovingAverageWindow = 32;

struct OpLatencyMovingAverageState {
  std::deque<uint64_t> recent_run_avg_latency_us;
  uint64_t rolling_total_latency_us{0};
  uint64_t cumulative_total_latency_us{0};
  uint64_t total_runs_seen{0};
};

struct OpLatencyMovingAverageSnapshot {
  uint64_t moving_avg_latency_us{0};
  uint64_t running_avg_latency_us{0};
  size_t window_sample_count{0};
  uint64_t total_runs_seen{0};
};

class OpLatencyMovingAverageTracker {
 public:
  OpLatencyMovingAverageSnapshot Update(const SessionState* session_state,
                                        const std::string& op_name,
                                        uint64_t run_avg_latency_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& session_stats = moving_average_by_session_[session_state];
    auto& state = session_stats[op_name];

    state.recent_run_avg_latency_us.push_back(run_avg_latency_us);
    state.rolling_total_latency_us += run_avg_latency_us;
    state.cumulative_total_latency_us += run_avg_latency_us;
    state.total_runs_seen += 1;

    if (state.recent_run_avg_latency_us.size() > kOpLatencyMovingAverageWindow) {
      state.rolling_total_latency_us -= state.recent_run_avg_latency_us.front();
      state.recent_run_avg_latency_us.pop_front();
    }

    const auto sample_count = state.recent_run_avg_latency_us.size();
    return OpLatencyMovingAverageSnapshot{
        sample_count == 0 ? 0 : (state.rolling_total_latency_us / sample_count),
        state.total_runs_seen == 0 ? 0 : (state.cumulative_total_latency_us / state.total_runs_seen),
        sample_count,
        state.total_runs_seen};
  }

 private:
  std::mutex mutex_;
  InlinedHashMap<const SessionState*, InlinedHashMap<std::string, OpLatencyMovingAverageState>> moving_average_by_session_;
};

OpLatencyMovingAverageTracker& GetOpLatencyMovingAverageTracker() {
  static OpLatencyMovingAverageTracker tracker;
  return tracker;
}

}  // namespace

static void CalculateTotalInputSizes(const OpKernelContextInternal* op_kernel_context,
                                     const onnxruntime::OpKernel* p_op_kernel,
                                     size_t& input_activation_sizes, size_t& input_parameter_sizes,
                                     const std::string& node_name, std::string& input_type_shape) {
  // Calculate total input sizes for this operation.
  std::stringstream ss;
  ss << "[";
  int added_type_shapes = 0;
  input_activation_sizes = 0;
  input_parameter_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  const int input_count = op_kernel_context->InputCount();
  for (auto i = 0; i < input_count; i++) {
    const OrtValue* p_input = op_kernel_context->GetInputMLValue(i);
    if (p_input != nullptr && p_input->IsAllocated() && p_input->IsTensor()) {
      const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
      const Tensor* p_tensor = nullptr;
      bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
      if (!is_param) {
        p_tensor = &(p_input->Get<Tensor>());
      }
      size_t tensor_size = p_tensor->SizeInBytes();

#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = p_tensor->Shape();
      size_t element_size = p_tensor->DataType()->Size();
      LOGS(logger, INFO) << node_name << " input[" << i << "]"
                         << " is_param=" << is_param
                         << " size=" << tensor_size
                         << " shape=" << tensor_shape.ToString()
                         << " element_size=" << element_size
                         << "\n";
#endif
      if (is_param) {
        input_parameter_sizes += tensor_size;
      } else {
        input_activation_sizes += tensor_size;
      }
      auto shape_str = p_tensor->Shape().ToString();
      ss << (added_type_shapes++ > 0 ? "," : "")
         << "{\"" << DataTypeImpl::ToString(p_tensor->DataType()) << "\":["
         << shape_str.substr(1, shape_str.size() - 2) << "]}";
    }
  }
  ss << "]";
  input_type_shape = ss.str();
}

static bool TryGetConvRuntimeShape(OpKernelContextInternal* op_kernel_context,
                                   const onnxruntime::OpKernel* p_op_kernel,
                                   int64_t& cin, int64_t& cout,
                                   int64_t& h, int64_t& w,
                                   int64_t& kh, int64_t& kw) {
  const auto& op_name = p_op_kernel->KernelDef().OpName();
  if (op_name != "Conv" && op_name != "FusedConv") {
    return false;
  }

  if (op_kernel_context->InputCount() < 2 || op_kernel_context->OutputCount() < 1) {
    return false;
  }

  const OrtValue* input_ort_value = op_kernel_context->GetInputMLValue(0);
  const OrtValue* weight_ort_value = op_kernel_context->GetInputMLValue(1);
  const OrtValue* output_ort_value = op_kernel_context->GetOutputMLValue(0);

  if (input_ort_value == nullptr || weight_ort_value == nullptr || output_ort_value == nullptr ||
      !input_ort_value->IsTensor() || !weight_ort_value->IsTensor() || !output_ort_value->IsTensor()) {
    return false;
  }

  const auto& input_tensor = input_ort_value->Get<Tensor>();
  const auto& weight_tensor = weight_ort_value->Get<Tensor>();
  const auto& output_tensor = output_ort_value->Get<Tensor>();

  const auto& input_shape = input_tensor.Shape();
  const auto& weight_shape = weight_tensor.Shape();
  const auto& output_shape = output_tensor.Shape();

  if (input_shape.NumDimensions() != 4 || weight_shape.NumDimensions() != 4 || output_shape.NumDimensions() != 4) {
    return false;
  }

  cin = input_shape[1];
  cout = output_shape[1];
  h = input_shape[2];
  w = input_shape[3];
  kh = weight_shape[2];
  kw = weight_shape[3];
  return true;
}

static std::string MakeLatencyProfileOpName(OpKernelContextInternal* op_kernel_context,
                                            const onnxruntime::OpKernel* p_op_kernel) {
  int64_t cin = 0;
  int64_t cout = 0;
  int64_t h = 0;
  int64_t w = 0;
  int64_t kh = 0;
  int64_t kw = 0;

  if (TryGetConvRuntimeShape(op_kernel_context, p_op_kernel, cin, cout, h, w, kh, kw)) {
    return MakeString(p_op_kernel->KernelDef().OpName(), "_",
                      kh, "x", kw,
                      "_Cin=", cin,
                      "_Cout=", cout,
                      "_H=", h,
                      "_W=", w);
  }

  return p_op_kernel->KernelDef().OpName();
}

static bool IsPointwiseConvLatencyStatsEnabled() {
  static const bool is_enabled = []() {
    const std::string env_value = Env::Default().GetEnvironmentVar("ORT_ENABLE_POINTWISE_CONV_LATENCY_STATS");
    return env_value.empty() || env_value != "0";
  }();

  return is_enabled;
}

class KernelScope;

#ifdef CONCURRENCY_VISUALIZER
std::string ComposeSeriesName(const GraphViewer& graph_viewer) {
  char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
  if (graph_viewer.IsSubgraph()) {
    auto s = graph_viewer.ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
    std::copy(s.cbegin(), s.cend(), series_name);
  }
  return series_name;
}
#endif

class SessionScope {
 public:
  friend class KernelScope;
  SessionScope(const SessionState& session_state, const ExecutionFrame& frame, profiling::Profiler* run_profiler)
      : session_state_(session_state), run_profiler_(run_profiler)
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
        ,
        frame_(frame)
#endif
#ifdef CONCURRENCY_VISUALIZER
        ,
        series_(ComposeSeriesName(session_state.GetGraphViewer()))
#endif
#ifdef ENABLE_NVTX_PROFILE
        ,
        session_tag_(profile::Context::GetInstance().GetThreadTagOrDefault(std::this_thread::get_id())),
        forward_range_("Batch-" + session_tag_ + " Forward", profile::Color::White),
        backward_range_("Batch-" + session_tag_ + " Backward", profile::Color::Black)
#endif
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
        ,
        dump_context_{session_state_.GetGraphExecutionCounter(), 0}
#endif
  {
    session_start_ = StartProfilingIfEnabled();
    session_wall_clock_begin_time_ = std::chrono::steady_clock::now();

    auto& logger = session_state_.Logger();
    VLOGS(logger, 0) << "Begin execution";
    const SequentialExecutionPlan& seq_exec_plan = *session_state_.GetExecutionPlan();
    const auto& exec_plan_vec = seq_exec_plan.execution_plan;
    VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

// Enable TRACE_EXECUTION compile flag to dump execution plan
#if defined(TRACE_EXECUTION)
    std::cout << std::make_pair(&seq_exec_plan, &session_state) << std::endl;
#endif
#if defined(ORT_MINIMAL_BUILD) || !defined(ORT_MEMORY_PROFILE)
    ORT_UNUSED_PARAMETER(frame);
#endif
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionScope);

  ~SessionScope() {
#ifdef ENABLE_NVTX_PROFILE
    // Make sure forward Range object call Begin and End.
    if (!forward_range_.IsBeginCalled()) {
      forward_range_.Begin();
    }
    if (!forward_range_.IsEndCalled()) {
      forward_range_.End();
    }
    // Make sure backward Range object call Begin and End.
    if (!backward_range_.IsBeginCalled()) {
      backward_range_.Begin();
    }
    if (!backward_range_.IsEndCalled()) {
      backward_range_.End();
    }
#endif

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    if (flush_memory_info_) {
      session_state_.GetMemoryProfiler()->CreateEvents(
          "dynamic activations_" + std::to_string(session_state_.GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
          session_state_.GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::DynamicActivation, "", 0);
      session_state_.GetMemoryProfiler()->Clear();
    }
#endif

    StopProfilingIfEnabled(profiling::SESSION_EVENT, "SequentialExecutor::Execute", session_start_);

    if (IsPointwiseConvLatencyStatsEnabled()) {
      LogOpLatencyStatsWarning();
    }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    auto& logger = session_state_.Logger();
    for (auto i : frame_.GetStaticMemorySizeInfo()) {
      LOGS(logger, VERBOSE) << "[Memory] ExecutionFrame statically allocates "
                            << i.second << " bytes for " << i.first << std::endl;
    }

    for (auto i : frame_.GetDynamicMemorySizeInfo()) {
      LOGS(logger, VERBOSE) << "[Memory] ExecutionFrame dynamically allocates "
                            << i.second << " bytes for " << i.first << std::endl;
    }
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    dump_analysis_.PrintToStdOut(session_state_.GetGraphViewer().ModelPath().string());
#endif
  }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  void SetFlushMemoryInfoFlag(bool flush_memory_info) {
    flush_memory_info_ = flush_memory_info;
  }
#endif

  bool IsRunProfilingEnabled() const {
    return run_profiler_ && run_profiler_->IsEnabled();
  }

  void RecordKernelLatency(uint64_t latency_us) {
    total_kernel_latency_us_.fetch_add(latency_us, std::memory_order_relaxed);
  }

  void RecordOpLatency(const std::string& op_name, uint64_t latency_us) {
    if (!IsPointwiseConvLatencyStatsEnabled()) {
      return;
    }

    std::lock_guard<std::mutex> lock(op_latency_stats_mutex_);
    auto& stats = op_latency_stats_[op_name];
    stats.count += 1;
    stats.total_latency_us += latency_us;
    stats.min_latency_us = std::min(stats.min_latency_us, latency_us);
    stats.max_latency_us = std::max(stats.max_latency_us, latency_us);
  }

  void StopProfilingIfEnabled(profiling::EventCategory category,
                              const std::string& event_name,
                              const TimePoint& start_time,
                              InlinedHashMap<std::string, std::string> event_args = {}) {
    const bool session_profiling_enabled = session_state_.Profiler().IsEnabled();
    const bool run_profiling_enabled = IsRunProfilingEnabled();

    if (session_profiling_enabled) {
      session_state_.Profiler().EndTimeAndRecordEvent(category,
                                                      event_name,
                                                      start_time,
                                                      std::move(event_args));
    } else if (run_profiling_enabled) {
      run_profiler_->EndTimeAndRecordEvent(category,
                                           event_name,
                                           start_time,
                                           std::move(event_args));
    }
  }

  TimePoint StartProfilingIfEnabled() {
    const bool session_profiling_enabled = session_state_.Profiler().IsEnabled();
    const bool run_profiling_enabled = IsRunProfilingEnabled();

    if (session_profiling_enabled) {
      return session_state_.Profiler().Start();
    } else if (run_profiling_enabled) {
      return run_profiler_->Start();
    }
    return TimePoint{};
  }

 private:
  struct OpLatencyStats {
    uint64_t count{0};
    uint64_t total_latency_us{0};
    uint64_t min_latency_us{std::numeric_limits<uint64_t>::max()};
    uint64_t max_latency_us{0};
  };

  struct OpLatencyLogEntry {
    std::string op_name;
    OpLatencyStats current_run_stats;
    uint64_t current_run_avg_latency_us{0};
    OpLatencyMovingAverageSnapshot moving_average_snapshot;
  };

  void LogOpLatencyStatsWarning() const {
    InlinedVector<OpLatencyLogEntry> op_stats;
    uint64_t total_tracked_conv_kernel_latency_us = 0;
    uint64_t total_tracked_conv_kernel_invocations = 0;

    {
      std::lock_guard<std::mutex> lock(op_latency_stats_mutex_);
      if (op_latency_stats_.empty()) {
        return;
      }

      op_stats.reserve(op_latency_stats_.size());
      for (const auto& kvp : op_latency_stats_) {
        const auto& stats = kvp.second;
        const uint64_t current_run_avg_latency_us = stats.count == 0 ? 0 : (stats.total_latency_us / stats.count);

        op_stats.push_back(OpLatencyLogEntry{
            kvp.first,
            stats,
            current_run_avg_latency_us,
          GetOpLatencyMovingAverageTracker().Update(&session_state_, kvp.first, current_run_avg_latency_us)});

        total_tracked_conv_kernel_latency_us += kvp.second.total_latency_us;
        total_tracked_conv_kernel_invocations += kvp.second.count;
      }
    }

    const uint64_t total_run_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                    std::chrono::steady_clock::now() - session_wall_clock_begin_time_)
                                                                    .count());
    const uint64_t total_kernel_latency_us = total_kernel_latency_us_.load(std::memory_order_relaxed);
    const uint64_t framework_overhead_us =
        total_run_latency_us > total_kernel_latency_us ? (total_run_latency_us - total_kernel_latency_us) : 0;

    std::sort(op_stats.begin(), op_stats.end(),
              [](const auto& left, const auto& right) {
                if (left.current_run_avg_latency_us != right.current_run_avg_latency_us) {
                  return left.current_run_avg_latency_us > right.current_run_avg_latency_us;
                }
                if (left.current_run_stats.total_latency_us != right.current_run_stats.total_latency_us) {
                  return left.current_run_stats.total_latency_us > right.current_run_stats.total_latency_us;
                }
                return left.op_name < right.op_name;
              });

    auto& logger = session_state_.Logger();
    LOGS(logger, WARNING) << "[Latency] run_total_us=" << total_run_latency_us
                          << " total_kernel_us=" << total_kernel_latency_us
                          << " framework_overhead_us=" << framework_overhead_us;
    LOGS(logger, WARNING) << "[Latency] kernel latency summary by op/shape: unique_entries="
                          << op_stats.size() << ", kernel_invocations="
                          << total_tracked_conv_kernel_invocations << ", total_kernel_time_us="
                          << total_tracked_conv_kernel_latency_us;

    for (const auto& entry : op_stats) {
      const auto& op_name = entry.op_name;
      const auto& stats = entry.current_run_stats;
      LOGS(logger, WARNING) << "[Latency] op=" << op_name
                            << " count=" << stats.count
                            << " total_us=" << stats.total_latency_us
                            << " avg_us=" << entry.current_run_avg_latency_us
                            << " moving_avg_us=" << entry.moving_average_snapshot.moving_avg_latency_us
                            << " running_avg_us=" << entry.moving_average_snapshot.running_avg_latency_us
                            << " moving_avg_window=" << entry.moving_average_snapshot.window_sample_count
                            << " moving_avg_runs_seen=" << entry.moving_average_snapshot.total_runs_seen
                            << " min_us=" << stats.min_latency_us
                            << " max_us=" << stats.max_latency_us;
    }
  }

  const SessionState& session_state_;
  profiling::Profiler* run_profiler_;
  TimePoint session_start_;
  std::chrono::steady_clock::time_point session_wall_clock_begin_time_;
  std::atomic<uint64_t> total_kernel_latency_us_{0};
  mutable std::mutex op_latency_stats_mutex_;
  InlinedHashMap<std::string, OpLatencyStats> op_latency_stats_;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  const ExecutionFrame& frame_;
#endif
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Whether memory profiler need create events and flush to file.
  // For partial graph run, when the last subgraph of the whole graph is executing, we need flush to file.
  bool flush_memory_info_ = true;
#endif

#ifdef CONCURRENCY_VISUALIZER
  diagnostic::marker_series series_;
#endif

#ifdef ENABLE_NVTX_PROFILE
  const std::string session_tag_;
  profile::NvtxRangeCreator forward_range_;
  profile::NvtxRangeCreator backward_range_;
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  utils::NodeDumpContext dump_context_;
  utils::NodeDumpAnalysis dump_analysis_;
#endif
};

class KernelScope {
 public:
  KernelScope(SessionScope& session_scope,
              OpKernelContextInternal& kernel_context,
              const OpKernel& kernel)
      : session_scope_(session_scope),
        session_state_(session_scope_.session_state_),
        kernel_context_(kernel_context),
        kernel_(kernel)
#ifdef CONCURRENCY_VISUALIZER
        ,
        span_(session_scope_.series_, "%s.%d", kernel_.Node().OpType().c_str(), kernel_.Node().Index())
#endif
#ifdef ENABLE_NVTX_PROFILE
        ,
        node_compute_range_(MakeString(kernel_.Node().OpType(),
                                       ".",
                                       kernel_.Node().Index(),
                                       "(",
                                       kernel_.Node().Name(),
                                       ")"),
                            profile::Color::Yellow)
#endif
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
        ,
        dump_context_{
            session_scope_.dump_context_.iteration, kernel_.Node().Index()}
#endif
  {
    wall_clock_begin_time_ = std::chrono::steady_clock::now();

#ifdef CONCURRENCY_VISUALIZER
    session_scope_.series_.write_flag(kernel_.Node().Name().c_str());
#endif

#ifdef ENABLE_NVTX_PROFILE
    {
      auto& node = kernel_.Node();
      profile::NvtxRangeCreator& forward_range = session_scope_.forward_range_;
      profile::NvtxRangeCreator& backward_range = session_scope_.backward_range_;
      if (node.Description() != "Backward pass" && !forward_range.IsBeginCalled()) {
        // Start timing forward pass when encountering the first forward node.
        forward_range.Begin();
      } else if (node.Description() == "Backward pass" && !backward_range.IsBeginCalled() &&
                 forward_range.IsBeginCalled()) {
        // Start timing backward pass when encountering the first backward node.
        // In the meanwhile, forward range ends.
        forward_range.End();
        backward_range.Begin();
      }
    }
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_start;
    QueryPerformanceCounter(&kernel_start);
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeInputs(dump_context_, kernel_context_, kernel_.Node(), session_state_, session_scope_.dump_analysis_);
#endif

    const bool session_profiling_enabled = session_state_.Profiler().IsEnabled();
    const bool run_profiling_enabled = session_scope_.IsRunProfilingEnabled();

    if (session_profiling_enabled || run_profiling_enabled) {
      auto& node = kernel.Node();
      node_name_ = node.Name().empty() ? MakeString(node.OpType(), "_", node.Index()) : node.Name();
      concurrency::ThreadPool::StartProfiling(session_state_.GetThreadPool());
      VLOGS(session_state_.Logger(), 1) << "Computing kernel: " << node_name_;

      kernel_begin_time_ = session_scope_.StartProfilingIfEnabled();

      CalculateTotalInputSizes(&kernel_context, &kernel_,
                               input_activation_sizes_, input_parameter_sizes_,
                               node_name_, input_type_shape_);
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelScope);

  ~KernelScope() {
    const auto wall_clock_end_time = std::chrono::steady_clock::now();
    const auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                wall_clock_end_time - wall_clock_begin_time_)
                                .count();

    if (latency_us > 0) {
      session_scope_.RecordKernelLatency(static_cast<uint64_t>(latency_us));
      session_scope_.RecordOpLatency(MakeLatencyProfileOpName(&kernel_context_, &kernel_),
                                     static_cast<uint64_t>(latency_us));
    }

#ifdef ENABLE_NVTX_PROFILE
    node_compute_range_.End();
#endif

    const bool session_profiling_enabled = session_state_.Profiler().IsEnabled();
    const bool run_profiling_enabled = session_scope_.IsRunProfilingEnabled();

    if (session_profiling_enabled || run_profiling_enabled) {
      {
        InlinedHashMap<std::string, std::string> event_args = {
            {"op_name", kernel_.KernelDef().OpName()},
            {"provider", kernel_.KernelDef().Provider()},
            {"node_index", std::to_string(kernel_.Node().Index())},
        };

        event_args.insert_or_assign("activation_size", std::to_string(input_activation_sizes_));
        event_args.insert_or_assign("parameter_size", std::to_string(input_parameter_sizes_));
        event_args.insert_or_assign("thread_scheduling_stats",
                                    concurrency::ThreadPool::StopProfiling(session_state_.GetThreadPool()));

        session_scope_.StopProfilingIfEnabled(profiling::NODE_EVENT,
                                              node_name_ + "_kernel_time",
                                              kernel_begin_time_,
                                              std::move(event_args));
      }
    }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_stop;
    QueryPerformanceCounter(&kernel_stop);
    LARGE_INTEGER elapsed;
    elapsed.QuadPart = kernel_stop.QuadPart - kernel_start.QuadPart;
    elapsed.QuadPart *= 1000000;
    elapsed.QuadPart /= perf_freq.QuadPart;
    // Log an event
    TraceLoggingWrite(telemetry_provider_handle,  // handle to my provider
                      "OpEnd",                    // Event Name that should uniquely identify your event.
                      TraceLoggingValue(p_op_kernel->KernelDef().OpName().c_str(), "op_name"),
                      TraceLoggingValue(elapsed.QuadPart, "time"));
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeOutputs(dump_context_, kernel_context_, kernel_.Node(), session_state_, session_scope_.dump_analysis_);
#endif
  }  //~KernelScope

 private:
  std::chrono::steady_clock::time_point wall_clock_begin_time_;
  TimePoint kernel_begin_time_;
  SessionScope& session_scope_;
  const SessionState& session_state_;
  std::string node_name_;
  OpKernelContextInternal& kernel_context_;
  const OpKernel& kernel_;

  size_t input_activation_sizes_{};
  size_t input_parameter_sizes_{};
  size_t total_output_sizes_{};
  std::string input_type_shape_;

#ifdef CONCURRENCY_VISUALIZER
  diagnostic::span span_;
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator node_compute_range_;
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  utils::NodeDumpContext dump_context_;
#endif
};

onnxruntime::Status ExecuteKernel(StreamExecutionContext& ctx,
                                  NodeIndex idx,
                                  size_t stream_idx,
                                  const bool& terminate_flag,
                                  SessionScope& session_scope) {
  auto* p_kernel = ctx.GetSessionState().GetKernel(idx);
  if (p_kernel->KernelDef().OpName() == "YieldOp") {
    // Do not execute YieldOp (it is an no-op anyways).
    // Decrement the reference count of tensors that are not needed beyond this point.
    // REVIEW(codemzs): The current model assumes the intermediate tensors that are exported
    // as graph outputs are owned by ORT, the risk of caller freeing the tensor or manipulating tensor
    // memory lingers while the tensor is used downstream after the export.
    ctx.RecycleNodeInputs(idx);
    return Status::OK();
  }
  // TODO: set terminate flag from run_option
  OpKernelContextInternal kernel_ctx(ctx.GetSessionState(),
                                     ctx.GetExecutionFrame(),
                                     *p_kernel,
                                     ctx.GetLogger(),
                                     terminate_flag,
                                     ctx.GetDeviceStream(stream_idx));
  onnxruntime::Status status;
  auto& logger = ctx.GetLogger();
  if (p_kernel->IsAsync()) {
    ORT_THROW("Async Kernel Support is not implemented yet.");
  } else {
    KernelScope kernel_scope(session_scope, kernel_ctx, *p_kernel);
    ORT_TRY {
#ifdef ENABLE_TRAINING
      // AllocateInputsContiguously - is only required for NCCL kernels
      // can be moved under USE_NCCL
      if (p_kernel->KernelDef().AllocateInputsContiguously()) {
        ORT_RETURN_IF_ERROR(utils::VerifyInputTensorsAllocatedContiguously(&kernel_ctx));
      }

      // This is most probably deprecated code and is causing unnecessary complexity.
      // Can be removed.
      // Cache lookup. Currently we only cache single-output nodes,
      // to keep memory overhead impact in check. Hence we only look in cache
      // if the current node has one output.
      bool reuse_cached_value = false;
      std::string cached_arg_name;
      auto& cache = ctx.GetOrtValueCache();
      if (cache != nullptr) {
        if (p_kernel->Node().OutputDefs().size() == 1) {
          cached_arg_name = p_kernel->Node().OutputDefs()[0]->Name();
          if (cache.get()->count(cached_arg_name)) {  // found arg in cache_
            VLOGS(logger, 1) << "Found OrtValue in cache for arg: " << cached_arg_name;
            reuse_cached_value = true;
          }
        }
      }
      if (!reuse_cached_value) {
        status = p_kernel->Compute(&kernel_ctx);
      } else {
        status = kernel_ctx.SetOutputMLValue(0, cache.get()->at(cached_arg_name));
      }
#else
      status = p_kernel->Compute(&kernel_ctx);

#if !defined(ORT_MINIMAL_BUILD)
      auto* node_stats_recorder = ctx.GetSessionState().GetNodeStatsRecorder();
      if (node_stats_recorder != nullptr) {
        const auto& node = p_kernel->Node();
        const OpKernelInfo& op_kernel_info = p_kernel->Info();
        const auto input_defs = node.InputDefs();

        // Lets first check if any inputs are initializers,
        // if so we need to account for their memory usage.
        SafeInt<int64_t> initializers_size = 0;
        SafeInt<size_t> input_sizes = 0;
        for (int i = 0, lim = kernel_ctx.InputCount(); i < lim; ++i) {
          // Need to get ort_value_index for each input.
          const OrtValue* p_input = kernel_ctx.GetInputMLValue(i);
          if (p_input != nullptr && p_input->IsAllocated() && p_input->IsTensor()) {
            const auto& input_name = input_defs[i]->Name();
            if (node_stats_recorder->ShouldAccountFor(input_name)) {
              const Tensor* p_tensor = nullptr;
              const bool is_constant = op_kernel_info.TryGetConstantInput(i, &p_tensor);
              if (!is_constant) {
                p_tensor = &p_input->Get<Tensor>();
              }
              input_sizes += p_tensor->SizeInBytes();
            }
          }
        }

        // Get outputs and see if anything were allocated dynamically
        const auto output_defs = node.OutputDefs();
        SafeInt<size_t> total_dynamic_sizes = 0;
        const auto& exec_frame = ctx.GetExecutionFrame();
        for (int i = 0, lim = kernel_ctx.OutputCount(); i < lim; ++i) {
          const OrtValue* p_output = kernel_ctx.GetOutputMLValue(i);
          if (p_output != nullptr && p_output->IsAllocated() && p_output->IsTensor()) {
            int ort_value_index = kernel_ctx.GetOrtValueIndexForOutput(i);
            auto maybe_val = exec_frame.GetOrtValueDynamicAllocation(ort_value_index);
            if (maybe_val.has_value() && node_stats_recorder->ShouldAccountFor(output_defs[i]->Name())) {
              total_dynamic_sizes += *maybe_val;
            }
          }
        }

        NodeAllocationStats node_stats;
        node_stats.input_sizes = static_cast<size_t>(input_sizes);
        node_stats.initializers_sizes = static_cast<size_t>(initializers_size);
        node_stats.total_dynamic_sizes = total_dynamic_sizes;

        // Get the temporary allocations
        AllocatorStats temp_stats;
        if (kernel_ctx.GetAllocatorStats(temp_stats)) {
          node_stats.total_temp_allocations = narrow<size_t>(temp_stats.total_allocated_bytes);
        }

        // Record node allocation stats
        const std::string name = IResourceAccountant::MakeUniqueNodeName(node);
        node_stats_recorder->ReportNodeStats(name, node_stats);
      }
#endif
#endif
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
  }
  if (!status.IsOK()) {
    std::ostringstream ss;
    const auto& node = p_kernel->Node();
    ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
       << "' Status Message: " << status.ErrorMessage();
    // If the computation failed, we still can record the memory consumption
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    ctx.GetSessionState().GetMemoryProfiler()->CreateEvents(
        "dynamic activations_" + std::to_string(ctx.GetSessionState().GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
        ctx.GetSessionState().GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::DynamicActivation, "", 0);
#endif
    const auto msg_string = ss.str();
    LOGS(logger, ERROR) << msg_string;
    return Status(status.Category(), status.Code(), msg_string);
  }

  ctx.RecycleNodeInputs(idx);
  VLOGS(logger, 0) << "stream " << stream_idx << " launch kernel with idx " << idx;
  return Status::OK();
}

onnxruntime::Status ExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                                   const DeviceStreamCollection* device_streams,
#endif
                                   const bool& terminate_flag,
                                   const bool only_execute_path_to_fetches,
                                   bool single_thread_mode,
                                   profiling::Profiler* run_profiler) {
  auto* execution_plan = session_state.GetExecutionPlan();
  VLOGS(logger, 0) << "Number of streams: " << execution_plan->execution_plan.size();
  int32_t valid_streams = 0;
  for (auto& stream : execution_plan->execution_plan) {
    if (stream && stream->steps_.size() > 0)
      valid_streams++;
  }

  // prepare the execution context, notifications got initialized.
#ifdef ORT_ENABLE_STREAM
  StreamExecutionContext ctx(session_state,
                             valid_streams,
                             execution_plan->notification_owner_stream,
                             execution_plan->num_barriers,
                             device_streams,
                             feed_mlvalue_idxs,
                             feeds,
                             fetch_mlvalue_idxs,
                             fetches,
                             fetch_allocators,
                             logger,
                             single_thread_mode);
#else
  StreamExecutionContext ctx(session_state,
                             valid_streams,
                             feed_mlvalue_idxs,
                             feeds,
                             fetch_mlvalue_idxs,
                             fetches,
                             fetch_allocators,
                             logger,
                             single_thread_mode);
#endif
#ifdef ENABLE_TRAINING
  if (only_execute_path_to_fetches) {
    auto* node_to_execute = session_state.GetToBeExecutedRange(fetch_mlvalue_idxs);
    ctx.SetNodeToExecute(node_to_execute);
  }
#else
  ORT_UNUSED_PARAMETER(only_execute_path_to_fetches);
#endif

  SessionScope session_scope(session_state, ctx.GetExecutionFrame(), run_profiler);

  auto* tp = single_thread_mode ? nullptr : session_state.GetInterOpThreadPool();

  for (size_t i = 0; i < execution_plan->execution_plan.size(); ++i) {
    if (execution_plan->execution_plan[i]->steps_.empty()) {
      // execution context is initialized with number of valid streams
      // for invalid stream (0 steps), it doesn't count in number of tasks
      // so don't need to invoke CompleteTask here
      // ctx.CompleteTask();
    } else {
      concurrency::ThreadPool::Schedule(tp, [i, &ctx, &terminate_flag, &session_scope]() {
        RunSince(i, ctx, session_scope, terminate_flag, 0);
      });
    }
  }

  ctx.WaitAll();
  ORT_RETURN_IF_ERROR(ctx.TaskStatus());
  ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GetOutputs(fetches));
  if (ctx.GetExecutionFrame().HasMemoryPatternPlanner()) {
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
    }

    if (all_tensors) {
      MemoryPatternGroup mem_patterns;
      ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GeneratePatterns(mem_patterns));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(feeds, std::move(mem_patterns)));
    }
  }

  return Status::OK();
}

#ifdef ENABLE_TRAINING
onnxruntime::Status PartialExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                          std::vector<OrtValue>& feeds,
                                          gsl::span<const int> fetch_mlvalue_idxs,
                                          std::vector<OrtValue>& fetches,
                                          const std::unordered_map<size_t, IExecutor::CustomAllocator>&
                                              fetch_allocators,
                                          const logging::Logger& logger,
                                          const DeviceStreamCollection* device_streams,
                                          const bool& terminate_flag,
                                          bool single_thread_mode,
                                          PartialGraphExecutionState& state,
                                          const OrtValueCachePtr& cache,
                                          int32_t partial_graph_index) {
  // Be noted: feeds will be std::move to ctx, so it will be empty after this function.
  auto& ctx = state.GetExecutionContext(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches,
                                        fetch_allocators, session_state, logger, device_streams);

  auto* plan = session_state.GetExecutionPlan();

  ctx.SetCurrentRange(&state.GetProgramRegions(session_state));

  SessionScope session_scope(session_state, ctx.GetExecutionFrame(), nullptr);

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Only flush memory info for the 2nd partial graph execution (since ORTModule runs this function twice).
  session_scope.SetFlushMemoryInfoFlag(partial_graph_index == 1);
#else
  ORT_UNUSED_PARAMETER(partial_graph_index);
#endif

  ctx.SetOrtValueCache(cache);

  auto* tp = single_thread_mode ? nullptr : session_state.GetInterOpThreadPool();

  for (size_t i = 0; i < plan->execution_plan.size(); ++i) {
    if (!plan->execution_plan[i]->steps_.empty()) {
      concurrency::ThreadPool::Schedule(tp, [i, &ctx, &terminate_flag, &session_scope]() {
        auto* range = ctx.GetCurrentRange();
        size_t start = !range ? 0 : range->stream_pc_range[i].first;
        RunSince(i, ctx, session_scope, terminate_flag, start);
      });
    }
  }

  if (!single_thread_mode) {
    ctx.WaitAll();
  }

  ORT_RETURN_IF_ERROR(ctx.TaskStatus());
  ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GetOutputs(fetches));
  return Status::OK();
}
#endif

}  // namespace onnxruntime
