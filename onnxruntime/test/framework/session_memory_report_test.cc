// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "asserts.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace test {

ONNX_OPERATOR_SCHEMA(MemoryBackingReportTestOp)
    .SetDoc("Test operator for session memory diagnostics.")
    .Attr("workspace_bytes", "Declared bytes; -1 means unavailable.", ONNX_NAMESPACE::AttributeProto::INT, int64_t{128})
    .Attr("prepack", "Whether to replace the initializer with kernel-owned packed storage.",
          ONNX_NAMESPACE::AttributeProto::INT, int64_t{0})
    .Input(0, "input", "Test input.", "tensor(int32)")
    .Output(0, "output", "Test output.", "tensor(int32)");

class MemoryBackingReportTestKernel final : public OpKernel {
 public:
  explicit MemoryBackingReportTestKernel(const OpKernelInfo& info)
      : OpKernel(info),
        workspace_bytes_(info.GetAttrOrDefault<int64_t>("workspace_bytes", 128)),
        prepack_(info.GetAttrOrDefault<int64_t>("prepack", 0) != 0) {}

  Status Compute(OpKernelContext*) const override {
    return Status::OK();
  }

  Status PrePack(const Tensor&, int, AllocatorPtr allocator, bool& is_packed, PrePackedWeights*) override {
    is_packed = prepack_;
    if (prepack_) {
      packed_ = IAllocator::MakeUniquePtr<void>(std::move(allocator), 128);
    }
    return Status::OK();
  }

  Status DeclareWorkspaceRequirements(gsl::span<const WorkspaceInputShape>,
                                      InlinedVector<WorkspaceRequirement>& requirements) const override {
    requirements.clear();
    if (workspace_bytes_ >= 0) {
      requirements.push_back({static_cast<size_t>(workspace_bytes_), 0, 0});
    }
    return Status::OK();
  }

 private:
  const int64_t workspace_bytes_;
  const bool prepack_;
  IAllocatorUniquePtr<void> packed_;
};

namespace {

ONNX_NAMESPACE::TypeProto ReportTensorType(bool dynamic = false) {
  ONNX_NAMESPACE::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  auto* dimension = type.mutable_tensor_type()->mutable_shape()->add_dim();
  if (dynamic) {
    dimension->set_dim_param("N");
  } else {
    dimension->set_dim_value(4);
  }
  return type;
}

void AddReportInitializer(Graph& graph) {
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name("input");
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  initializer.add_dims(4);
  for (int i = 0; i < 4; ++i) {
    initializer.add_int32_data(i);
  }
  graph.AddInitializedTensor(initializer);
}

void WithReportSession(const std::function<NodeWorkspaceReservationMap(Graph&)>& build,
                       SessionOptions options, const std::function<void(const SessionState&)>& inspect,
                       bool enabled = true) {
  Model model("memory_backing_report", false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  auto reservations = build(graph);
  ASSERT_STATUS_OK(graph.Resolve());
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  if (enabled) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableMemoryBackingReport, "1"));
  }

  ExecutionProviders providers;
  ASSERT_STATUS_OK(providers.Add(
      kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(options.enable_cpu_mem_arena))));
  DataTransferManager transfers;
  ExternalDataLoaderManager loaders;
  profiling::Profiler profiler;
  SessionState state(graph, providers, nullptr, nullptr, transfers, loaders,
                     DefaultLoggingManager().DefaultLogger(), profiler, options);
  WorkspaceReservationMap graph_reservations;
  graph_reservations.emplace(&graph, std::move(reservations));
  state.SetWorkspaceReservations(std::move(graph_reservations));

  KernelRegistryManager registry_manager;
  ASSERT_STATUS_OK(registry_manager.RegisterKernels(providers));
  auto registry = std::make_shared<KernelRegistry>();
  auto definition = KernelDefBuilder().SetName("MemoryBackingReportTestOp").Provider(kCpuExecutionProvider).SinceVersion(1).Build();
  ASSERT_STATUS_OK(registry->Register(KernelCreateInfo(
      std::move(definition),
      [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) -> Status {
        kernel = std::make_unique<MemoryBackingReportTestKernel>(info);
        return Status::OK();
      })));
  registry_manager.RegisterKernelRegistry(registry);
  ASSERT_STATUS_OK(state.FinalizeSessionState(ORT_TSTR(""), registry_manager));
  inspect(state);
}

NodeWorkspaceReservationMap AddReportChain(Graph& graph) {
  const auto type = ReportTensorType();
  auto& input = graph.GetOrCreateNodeArg("input", &type);
  auto& intermediate = graph.GetOrCreateNodeArg("intermediate", &type);
  auto& output = graph.GetOrCreateNodeArg("output", &type);
  auto& first = graph.AddNode("first", "MemoryBackingReportTestOp", "", {&input}, {&intermediate});
  first.AddAttribute("workspace_bytes", int64_t{128});
  auto& second = graph.AddNode("second", "MemoryBackingReportTestOp", "", {&intermediate}, {&output});
  second.AddAttribute("workspace_bytes", int64_t{256});
  graph.SetOutputs({&output});
  return {};
}

bool HasUncoveredTerm(const SessionMemoryReport& report, const std::string& text) {
  return std::any_of(report.uncovered_terms.begin(), report.uncovered_terms.end(),
                     [&](const std::string& term) { return term.find(text) != std::string::npos; });
}

}  // namespace

TEST(SessionMemoryReportTest, OrderedScratchLifetimesDoNotSumSequentialNodes) {
  SessionOptions options;
  options.enable_mem_reuse = false;
  WithReportSession(AddReportChain, options, [](const SessionState& state) {
    const auto* report = state.GetMemoryBackingReport();
    ASSERT_NE(report, nullptr);
    EXPECT_TRUE(report->ordered_lifetimes);
    const auto& device = report->devices.at(OrtDevice());
    // The second node retains input, intermediate and output, plus its own workspace.
    EXPECT_EQ(device.modeled_run_live_peak_bytes, 3 * 4 * sizeof(int32_t) + 256);
    EXPECT_FALSE(device.initialization_backing_peak_bytes.has_value());
    EXPECT_FALSE(device.run_backing_peak_bytes.has_value());
    EXPECT_TRUE(HasUncoveredTerm(*report, "retained pools"));
  });
}

TEST(SessionMemoryReportTest, ParallelScratchLifetimesConservativelyOverlap) {
  SessionOptions options;
  options.execution_mode = ExecutionMode::ORT_PARALLEL;
  options.enable_mem_reuse = false;
  WithReportSession(AddReportChain, options, [](const SessionState& state) {
    const auto* report = state.GetMemoryBackingReport();
    ASSERT_NE(report, nullptr);
    EXPECT_FALSE(report->ordered_lifetimes);
    EXPECT_EQ(report->devices.at(OrtDevice()).modeled_run_live_peak_bytes,
              3 * 4 * sizeof(int32_t) + 128 + 256);
    EXPECT_TRUE(HasUncoveredTerm(*report, "not GPU completion"));
  });
}

TEST(SessionMemoryReportTest, PlannerAliasesChargeSharedInputAndInitializerOnce) {
  for (const bool initializer : {false, true}) {
    WithReportSession(
        [initializer](Graph& graph) {
          const auto type = ReportTensorType();
          auto& input = graph.GetOrCreateNodeArg("input", &type);
          auto& intermediate = graph.GetOrCreateNodeArg("intermediate", &type);
          auto& output = graph.GetOrCreateNodeArg("output", &type);
          graph.AddNode("first", "Identity", "", {&input}, {&intermediate});
          graph.AddNode("second", "Identity", "", {&intermediate}, {&output});
          graph.SetOutputs({&output});
          if (initializer) {
            AddReportInitializer(graph);
          }
          return NodeWorkspaceReservationMap{};
        },
        SessionOptions{}, [initializer](const SessionState& state) {
          const auto* report = state.GetMemoryBackingReport();
          ASSERT_NE(report, nullptr);
          // The intermediate aliases the input; the graph output requires a separate allocation.
          EXPECT_EQ(report->allocations.size(), 2U);
          const auto& device = report->devices.at(OrtDevice());
          EXPECT_EQ(device.modeled_run_live_peak_bytes, 2 * 4 * sizeof(int32_t));
          EXPECT_EQ(device.initializers_before_kernel_creation_bytes, initializer ? 4 * sizeof(int32_t) : 0U);
          EXPECT_EQ(device.initializers_after_prepack_bytes, initializer ? 4 * sizeof(int32_t) : 0U); });
  }
}

TEST(SessionMemoryReportTest, InitializerSnapshotPrecedesSourceReleaseAndPackedDestination) {
  WithReportSession(
      [](Graph& graph) {
        const auto type = ReportTensorType();
        auto& input = graph.GetOrCreateNodeArg("input", &type);
        auto& output = graph.GetOrCreateNodeArg("output", &type);
        auto& node = graph.AddNode("packed", "MemoryBackingReportTestOp", "", {&input}, {&output});
        node.AddAttribute("prepack", int64_t{1});
        graph.SetOutputs({&output});
        AddReportInitializer(graph);
        WorkspaceEstimateSelection estimate;
        estimate.bytes = 128;
        estimate.source = WorkspaceEstimateSource::kEstimator;
        estimate.persistent_prepack_bytes = 128;
        estimate.initialization_scratch_bytes = 64;
        return NodeWorkspaceReservationMap{{node.Index(), estimate}};
      },
      SessionOptions{}, [](const SessionState& state) {
        const auto* report = state.GetMemoryBackingReport();
        ASSERT_NE(report, nullptr);
        const auto& device = report->devices.at(OrtDevice());
        EXPECT_EQ(device.initializers_before_kernel_creation_bytes, 4 * sizeof(int32_t));
        EXPECT_EQ(device.initializers_after_prepack_bytes, 0U);
        EXPECT_EQ(device.l1_persistent_prepack_estimate_bytes, 128U);
        EXPECT_EQ(device.l1_initialization_scratch_estimate_bytes, 64U);
        EXPECT_EQ(device.modeled_run_live_peak_bytes, 4 * sizeof(int32_t) + 128);
        EXPECT_TRUE(HasUncoveredTerm(*report, "Packed-weight allocation identities"));
        EXPECT_FALSE(device.initialization_backing_peak_bytes.has_value());
        const auto snapshots = state.GetMemoryBackingAllocatorSnapshots();
        ASSERT_FALSE(snapshots.empty());
        EXPECT_EQ(std::any_of(snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
                    return snapshot.reported_allocated_bytes.has_value() &&
                           *snapshot.reported_allocated_bytes > device.initializers_after_prepack_bytes;
                  }),
                  state.GetAllocator(OrtDevice())->Info().alloc_type == OrtArenaAllocator); });
}

TEST(SessionMemoryReportTest, ShapeHintsDoNotTurnUnknownTensorBackingIntoZeroOrProof) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsMaxShapeOverride, "input:[4]"));
  WithReportSession(
      [](Graph& graph) {
        const auto type = ReportTensorType(/*dynamic=*/true);
        auto& input = graph.GetOrCreateNodeArg("input", &type);
        auto& output = graph.GetOrCreateNodeArg("output", &type);
        graph.AddNode("dynamic", "MemoryBackingReportTestOp", "", {&input}, {&output});
        graph.SetOutputs({&output});
        return NodeWorkspaceReservationMap{};
      },
      options, [](const SessionState& state) {
        const auto* report = state.GetMemoryBackingReport();
        ASSERT_NE(report, nullptr);
        EXPECT_EQ(report->devices.at(OrtDevice()).modeled_run_live_peak_bytes, 128U);
        EXPECT_TRUE(HasUncoveredTerm(*report, "Unavailable tensor storage for value 'input'"));
        EXPECT_TRUE(HasUncoveredTerm(*report, "Unavailable tensor storage for value 'output'"));
        EXPECT_TRUE(std::any_of(report->allocations.begin(), report->allocations.end(),
                                [](const auto& allocation) { return !allocation.size_bytes.has_value(); })); });
}

TEST(SessionMemoryReportTest, ExplicitZeroWorkspaceIsDistinctFromUnavailable) {
  for (const int64_t workspace_bytes : {int64_t{0}, int64_t{-1}}) {
    WithReportSession(
        [workspace_bytes](Graph& graph) {
          const auto type = ReportTensorType();
          auto& input = graph.GetOrCreateNodeArg("input", &type);
          auto& output = graph.GetOrCreateNodeArg("output", &type);
          auto& node = graph.AddNode("zero_or_missing", "MemoryBackingReportTestOp", "", {&input}, {&output});
          node.AddAttribute("workspace_bytes", workspace_bytes);
          graph.SetOutputs({&output});
          return NodeWorkspaceReservationMap{};
        },
        SessionOptions{}, [workspace_bytes](const SessionState& state) {
          const auto* report = state.GetMemoryBackingReport();
          ASSERT_NE(report, nullptr);
          EXPECT_EQ(HasUncoveredTerm(*report, "Unavailable workspace declaration"), workspace_bytes < 0);
          EXPECT_EQ(report->devices.at(OrtDevice()).modeled_run_live_peak_bytes, 2 * 4 * sizeof(int32_t)); });
  }
}

TEST(SessionMemoryReportTest, DisabledReportingPreservesFinalizationWithoutCollectingMetadata) {
  WithReportSession(
      AddReportChain, SessionOptions{},
      [](const SessionState& state) { EXPECT_EQ(state.GetMemoryBackingReport(), nullptr); },
      /*enabled=*/false);
}

TEST(SessionMemoryReportTest, EnabledSessionRunKeepsRetainedBackingSeparateFromPayloads) {
  Model model("memory_report_run", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  const auto type = ReportTensorType();
  auto& input_arg = graph.GetOrCreateNodeArg("input", &type);
  auto& output_arg = graph.GetOrCreateNodeArg("output", &type);
  graph.AddNode("identity", "Identity", "", {&input_arg}, {&output_arg});
  graph.SetOutputs({&output_arg});
  ASSERT_STATUS_OK(graph.Resolve());
  std::string serialized;
  ASSERT_TRUE(model.ToProto().SerializeToString(&serialized));

  for (const bool enable_arena : {false, true}) {
    SCOPED_TRACE(enable_arena);
    SessionOptions options;
    options.enable_cpu_mem_arena = enable_arena;
    options.graph_optimization_level = TransformerLevel::Default;
    options.session_log_severity_level = 1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsEnableMemoryBackingReport, "1"));
    InferenceSession session{options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());

    OrtValue input;
    Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), TensorShape{4},
                         session.GetSessionState().GetAllocator(OrtDevice()), input);
    const std::array<int32_t, 4> expected{1, 2, 3, 4};
    std::copy(expected.begin(), expected.end(), input.GetMutable<Tensor>()->MutableData<int32_t>());
    NameMLValMap feeds{{"input", input}};
    std::vector<OrtValue> outputs;
    const std::array<std::string, 1> output_names{"output"};
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    ASSERT_EQ(outputs.size(), 1U);
    const auto values = outputs[0].Get<Tensor>().DataAsSpan<int32_t>();
    EXPECT_TRUE(std::equal(values.begin(), values.end(), expected.begin(), expected.end()));

    const auto& state = session.GetSessionState();
    const auto* report = state.GetMemoryBackingReport();
    ASSERT_NE(report, nullptr);
    const auto& device = report->devices.at(OrtDevice());
    EXPECT_EQ(device.modeled_run_live_peak_bytes, 2 * 4 * sizeof(int32_t));
    EXPECT_FALSE(device.run_backing_peak_bytes.has_value());
    const auto snapshots = state.GetMemoryBackingAllocatorSnapshots();
    ASSERT_FALSE(snapshots.empty());
    EXPECT_EQ(std::any_of(snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
                return snapshot.reported_allocated_bytes.has_value() &&
                       *snapshot.reported_allocated_bytes > device.modeled_run_live_peak_bytes;
              }),
              state.GetAllocator(OrtDevice())->Info().alloc_type == OrtArenaAllocator);
    if (state.GetAllocator(OrtDevice())->Info().alloc_type != OrtArenaAllocator) {
      for (const auto& snapshot : snapshots) {
        EXPECT_FALSE(snapshot.reported_allocated_bytes.has_value());
        EXPECT_FALSE(snapshot.reported_in_use_bytes.has_value());
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
