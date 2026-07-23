// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the segregated optional-capability interfaces on
// IExecutionProvider (see core/framework/execution_provider_capabilities.h).
//
// These exercise the capability-query mechanism with lightweight fake EPs and
// run entirely on CPU -- they require no GPU/NPU backend.

#include <memory>

#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"
#include "core/framework/execution_provider_capabilities.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr const char* kFakeEpType = "FakeCapabilityEp";

// An EP that supports no optional capability. Every query hook should return the
// base-class default of nullptr.
class PlainEp : public IExecutionProvider {
 public:
  PlainEp() : IExecutionProvider(kFakeEpType) {}
};

// An EP that supports graph capture, implemented via the segregated mix-in and
// surfaced through the GetGraphCaptureCapability() query hook.
class GraphCaptureEp : public IExecutionProvider, public IGraphCaptureCapability {
 public:
  GraphCaptureEp() : IExecutionProvider(kFakeEpType) {}

  IGraphCaptureCapability* GetGraphCaptureCapability() noexcept override { return this; }

  // IGraphCaptureCapability
  bool IsGraphCaptureEnabled() const override { return true; }
  bool IsGraphCaptured(int graph_annotation_id) const override { return graph_annotation_id == kCapturedId; }
  common::Status ReplayGraph(int /*graph_annotation_id*/, bool /*sync*/) override {
    ++replay_count_;
    return Status::OK();
  }
  common::Status ReleaseCapturedGraph(int /*graph_annotation_id*/) override { return Status::OK(); }
  OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const override {
    return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
  }

  static constexpr int kCapturedId = 7;
  int replay_count() const { return replay_count_; }

 private:
  int replay_count_ = 0;
};

// An EP that supports only tuning.
class TuningEp : public IExecutionProvider, public ITuningCapability {
 public:
  TuningEp() : IExecutionProvider(kFakeEpType) {}

  // Non-const hook (matches the base signature): returns the mix-in subobject
  // directly, with no const_cast.
  ITuningCapability* GetTuningCapability() noexcept override { return this; }

  // ITuningCapability::GetTuningContext() is pure virtual, so a successful call
  // necessarily dispatches to this override -- that alone proves routing through
  // the mix-in, with no mutable call-counter needed. A fake EP holds no real
  // tuning state, so it reports a null context.
  ITuningContext* GetTuningContext() const override { return nullptr; }
};

// An EP that supports only a data-layout preference.
class DataLayoutEp : public IExecutionProvider, public IDataLayoutCapability {
 public:
  DataLayoutEp() : IExecutionProvider(kFakeEpType) {}

  // Const hook returning a const pointer: no const_cast, returns the mix-in
  // subobject directly.
  const IDataLayoutCapability* GetDataLayoutCapability() const noexcept override { return this; }

  // IDataLayoutCapability
  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }
  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view /*domain*/,
                                                   std::string_view op_type,
                                                   DataLayout target_data_layout) const override {
    if (target_data_layout == DataLayout::NHWC && op_type == "Conv") {
      return true;
    }
    return std::nullopt;
  }
};

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// An EP that supports only subgraph compilation. Call markers prove the calls
// route to the concrete implementation through the mix-in.
class CompileEp : public IExecutionProvider, public ICompileCapability {
 public:
  CompileEp() : IExecutionProvider(kFakeEpType) {}

  ICompileCapability* GetCompileCapability() noexcept override { return this; }

  // ICompileCapability
  common::Status Compile(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& /*node_compute_funcs*/) override {
    last_fused_count_ = fused_nodes_and_graphs.size();
    ++compile_call_count_;
    return Status::OK();
  }
  std::string GetCompiledModelCompatibilityInfo(const GraphViewer& /*graph_viewer*/) const override {
    return kCompatInfo;
  }
  common::Status ValidateCompiledModelCompatibilityInfo(
      const std::string& compatibility_info,
      OrtCompiledModelCompatibility& model_compatibility) const override {
    model_compatibility = (compatibility_info == kCompatInfo)
                              ? OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL
                              : OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return Status::OK();
  }

  static constexpr const char* kCompatInfo = "fake-compat-v1";
  int compile_call_count() const { return compile_call_count_; }
  size_t last_fused_count() const { return last_fused_count_; }

 private:
  int compile_call_count_ = 0;
  size_t last_fused_count_ = 0;
};
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// An EP that implements two capabilities at once. Used to verify the hooks
// return the correct, independent mix-in subobjects under multiple inheritance.
class GraphCaptureAndDataLayoutEp : public IExecutionProvider,
                                    public IGraphCaptureCapability,
                                    public IDataLayoutCapability {
 public:
  GraphCaptureAndDataLayoutEp() : IExecutionProvider(kFakeEpType) {}

  IGraphCaptureCapability* GetGraphCaptureCapability() noexcept override { return this; }
  const IDataLayoutCapability* GetDataLayoutCapability() const noexcept override { return this; }

  // IGraphCaptureCapability
  bool IsGraphCaptureEnabled() const override { return true; }
  bool IsGraphCaptured(int /*graph_annotation_id*/) const override { return false; }
  common::Status ReplayGraph(int /*graph_annotation_id*/, bool /*sync*/) override { return Status::OK(); }
  common::Status ReleaseCapturedGraph(int /*graph_annotation_id*/) override { return Status::OK(); }
  OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const override {
    return OrtGraphCaptureNodeAssignmentPolicy_ALL_NODES_ON_EP;
  }

  // IDataLayoutCapability
  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }
  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view /*domain*/, std::string_view /*op_type*/,
                                                   DataLayout /*target_data_layout*/) const override {
    return std::nullopt;
  }
};

}  // namespace

// A capability-less EP returns nullptr from every query hook.
TEST(ExecutionProviderCapabilitiesTest, UnsupportedCapabilitiesReturnNull) {
  PlainEp ep;
  IExecutionProvider& base = ep;

  EXPECT_EQ(base.GetGraphCaptureCapability(), nullptr);
  EXPECT_EQ(base.GetTuningCapability(), nullptr);
  EXPECT_EQ(base.GetDataLayoutCapability(), nullptr);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  EXPECT_EQ(base.GetCompileCapability(), nullptr);
#endif
}

// An EP that supports graph capture is queryable through the base interface and
// usable through the narrow mix-in.
TEST(ExecutionProviderCapabilitiesTest, GraphCaptureCapabilityIsQueryableAndUsable) {
  GraphCaptureEp ep;
  IExecutionProvider& base = ep;

  IGraphCaptureCapability* gc = base.GetGraphCaptureCapability();
  ASSERT_NE(gc, nullptr) << "EP advertising graph capture must return a non-null capability.";

  EXPECT_TRUE(gc->IsGraphCaptureEnabled());
  EXPECT_TRUE(gc->IsGraphCaptured(GraphCaptureEp::kCapturedId));
  EXPECT_FALSE(gc->IsGraphCaptured(GraphCaptureEp::kCapturedId + 1));
  EXPECT_EQ(gc->GetGraphCaptureNodeAssignmentPolicy(),
            OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES);

  ASSERT_TRUE(gc->ReplayGraph(GraphCaptureEp::kCapturedId, /*sync*/ true).IsOK());
  EXPECT_TRUE(gc->ReleaseCapturedGraph(GraphCaptureEp::kCapturedId).IsOK());
  EXPECT_EQ(ep.replay_count(), 1) << "Replay must reach the concrete implementation.";
}

// An EP that supports a data-layout preference is queryable through the base
// interface and usable through the narrow mix-in.
TEST(ExecutionProviderCapabilitiesTest, DataLayoutCapabilityIsQueryableAndUsable) {
  DataLayoutEp ep;
  IExecutionProvider& base = ep;

  const IDataLayoutCapability* dl = base.GetDataLayoutCapability();
  ASSERT_NE(dl, nullptr) << "EP advertising a data-layout preference must return a non-null capability.";

  EXPECT_EQ(dl->GetPreferredLayout(), DataLayout::NHWC);

  const std::optional<bool> conv_decision = dl->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC);
  ASSERT_TRUE(conv_decision.has_value());
  EXPECT_TRUE(*conv_decision);

  // An op the EP has no opinion on leaves the decision to ORT (std::nullopt).
  EXPECT_FALSE(dl->ShouldConvertDataLayoutForOp("", "Add", DataLayout::NHWC).has_value());
}

// An EP that supports tuning is queryable through the base interface, and the
// call routes to the concrete implementation through the narrow mix-in.
TEST(ExecutionProviderCapabilitiesTest, TuningCapabilityIsQueryableAndUsable) {
  TuningEp ep;
  IExecutionProvider& base = ep;

  ITuningCapability* tc = base.GetTuningCapability();
  ASSERT_NE(tc, nullptr) << "EP advertising tuning must return a non-null capability.";

  // GetTuningContext() is pure virtual on the mix-in, so reaching this concrete
  // (null-context) result proves the call routed through the mix-in.
  EXPECT_EQ(tc->GetTuningContext(), nullptr);
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// An EP that supports subgraph compilation is queryable through the base
// interface and usable through the narrow mix-in.
TEST(ExecutionProviderCapabilitiesTest, CompileCapabilityIsQueryableAndUsable) {
  CompileEp ep;
  IExecutionProvider& base = ep;

  ICompileCapability* compile = base.GetCompileCapability();
  ASSERT_NE(compile, nullptr) << "EP advertising compilation must return a non-null capability.";

  std::vector<NodeComputeInfo> node_compute_funcs;
  ASSERT_TRUE(compile->Compile(/*fused_nodes_and_graphs*/ {}, node_compute_funcs).IsOK());
  EXPECT_EQ(ep.compile_call_count(), 1) << "Compile must reach the concrete implementation.";
  EXPECT_EQ(ep.last_fused_count(), 0u);

  OrtCompiledModelCompatibility match = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  ASSERT_TRUE(compile->ValidateCompiledModelCompatibilityInfo(CompileEp::kCompatInfo, match).IsOK());
  EXPECT_EQ(match, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);

  OrtCompiledModelCompatibility mismatch = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  ASSERT_TRUE(compile->ValidateCompiledModelCompatibilityInfo("other", mismatch).IsOK());
  EXPECT_EQ(mismatch, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// Capabilities are independent: an EP that implements only graph capture must
// not be mistaken for supporting tuning or compilation, and vice versa.
TEST(ExecutionProviderCapabilitiesTest, CapabilitiesAreIndependent) {
  GraphCaptureEp graph_ep;
  IExecutionProvider& graph_base = graph_ep;
  EXPECT_NE(graph_base.GetGraphCaptureCapability(), nullptr);
  EXPECT_EQ(graph_base.GetTuningCapability(), nullptr);
  EXPECT_EQ(graph_base.GetDataLayoutCapability(), nullptr);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  EXPECT_EQ(graph_base.GetCompileCapability(), nullptr);
#endif

  TuningEp tuning_ep;
  IExecutionProvider& tuning_base = tuning_ep;
  EXPECT_NE(tuning_base.GetTuningCapability(), nullptr);
  EXPECT_EQ(tuning_base.GetGraphCaptureCapability(), nullptr);
  EXPECT_EQ(tuning_base.GetDataLayoutCapability(), nullptr);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  EXPECT_EQ(tuning_base.GetCompileCapability(), nullptr);
#endif

  DataLayoutEp data_layout_ep;
  IExecutionProvider& data_layout_base = data_layout_ep;
  EXPECT_NE(data_layout_base.GetDataLayoutCapability(), nullptr);
  EXPECT_EQ(data_layout_base.GetGraphCaptureCapability(), nullptr);
  EXPECT_EQ(data_layout_base.GetTuningCapability(), nullptr);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  EXPECT_EQ(data_layout_base.GetCompileCapability(), nullptr);
#endif
}

// A single EP may implement multiple capabilities at once. Each hook must return
// the correct mix-in subobject (correct multiple-inheritance dispatch), and the
// unimplemented capabilities must remain null.
TEST(ExecutionProviderCapabilitiesTest, MultipleCapabilitiesCoexist) {
  GraphCaptureAndDataLayoutEp ep;
  IExecutionProvider& base = ep;

  IGraphCaptureCapability* gc = base.GetGraphCaptureCapability();
  const IDataLayoutCapability* dl = base.GetDataLayoutCapability();
  ASSERT_NE(gc, nullptr);
  ASSERT_NE(dl, nullptr);

  // Each hook routes to the correct subobject.
  EXPECT_TRUE(gc->IsGraphCaptureEnabled());
  EXPECT_EQ(dl->GetPreferredLayout(), DataLayout::NHWC);

  // Unimplemented capabilities remain null.
  EXPECT_EQ(base.GetTuningCapability(), nullptr);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  EXPECT_EQ(base.GetCompileCapability(), nullptr);
#endif
}

}  // namespace test
}  // namespace onnxruntime
