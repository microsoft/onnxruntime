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

  ITuningCapability* GetTuningCapability() const noexcept override {
    return const_cast<TuningEp*>(this);
  }

  // ITuningCapability. Returns nullptr (no real tuning state) -- the test only
  // needs to prove the call path is reachable through the mix-in.
  ITuningContext* GetTuningContext() const override { return nullptr; }
};

// An EP that supports only a data-layout preference.
class DataLayoutEp : public IExecutionProvider, public IDataLayoutCapability {
 public:
  DataLayoutEp() : IExecutionProvider(kFakeEpType) {}

  IDataLayoutCapability* GetDataLayoutCapability() const noexcept override {
    return const_cast<DataLayoutEp*>(this);
  }

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

  IDataLayoutCapability* dl = base.GetDataLayoutCapability();
  ASSERT_NE(dl, nullptr) << "EP advertising a data-layout preference must return a non-null capability.";

  EXPECT_EQ(dl->GetPreferredLayout(), DataLayout::NHWC);

  const std::optional<bool> conv_decision = dl->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC);
  ASSERT_TRUE(conv_decision.has_value());
  EXPECT_TRUE(*conv_decision);

  // An op the EP has no opinion on leaves the decision to ORT (std::nullopt).
  EXPECT_FALSE(dl->ShouldConvertDataLayoutForOp("", "Add", DataLayout::NHWC).has_value());
}

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

  DataLayoutEp data_layout_ep;
  IExecutionProvider& data_layout_base = data_layout_ep;
  EXPECT_NE(data_layout_base.GetDataLayoutCapability(), nullptr);
  EXPECT_EQ(data_layout_base.GetGraphCaptureCapability(), nullptr);
  EXPECT_EQ(data_layout_base.GetTuningCapability(), nullptr);
}

}  // namespace test
}  // namespace onnxruntime
