// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <gtest/gtest.h>

#include "core/providers/webgpu/ep/telemetry_aggregation.h"

namespace onnxruntime {
namespace test {

using webgpu::ep::telemetry::OpTypeCounter;
using webgpu::ep::telemetry::OpTypeTelemetryKey;
using webgpu::ep::telemetry::RunCounters;
using webgpu::ep::telemetry::SummaryIntervalScheduler;

TEST(WebGpuPluginTelemetryTest, OpTypeKeyBucketsCustomDomains) {
  EXPECT_EQ(OpTypeTelemetryKey("", "NonZero"), "ai.onnx:NonZero");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx", "NonZero"), "ai.onnx:NonZero");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx.ml", "TreeEnsemble"), "ai.onnx.ml:TreeEnsemble");
  EXPECT_EQ(OpTypeTelemetryKey("com.microsoft", "GroupQueryAttention"), "com.microsoft:GroupQueryAttention");
  EXPECT_EQ(OpTypeTelemetryKey("com.ms.internal.nhwc", "Conv"), "com.ms.internal.nhwc:Conv");
  // Operator names from other domains are model-defined and must not be reported.
  EXPECT_EQ(OpTypeTelemetryKey("com.contoso.secret", "ProjectX"), "custom");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx.contrib", "Tokenizer"), "custom");
}

TEST(WebGpuPluginTelemetryTest, OpTypeCounterOrdersByCountThenName) {
  OpTypeCounter counter;
  EXPECT_TRUE(counter.Empty());
  EXPECT_EQ(counter.ToString(10), "");

  counter.Add("", "NonZero");
  counter.Add("com.microsoft", "Foo");
  counter.Add("", "NonZero");
  counter.Add("", "Unique");
  counter.Add("com.microsoft", "Foo");
  counter.Add("", "NonZero");
  counter.Add("my.domain", "A");
  counter.Add("my.domain", "B");

  EXPECT_EQ(counter.DistinctCount(), 4u);
  EXPECT_EQ(counter.ToString(10), "ai.onnx:NonZero=3,com.microsoft:Foo=2,custom=2,ai.onnx:Unique=1");
  EXPECT_EQ(counter.ToString(2), "ai.onnx:NonZero=3,com.microsoft:Foo=2");
  EXPECT_EQ(counter.ToString(0), "");
}

TEST(WebGpuPluginTelemetryTest, OpTypeCounterBoundsDistinctKeys) {
  OpTypeCounter counter;
  const size_t total = OpTypeCounter::kMaxDistinctKeys + 10;
  for (size_t i = 0; i < total; ++i) {
    counter.Add("", "Op" + std::to_string(i));
  }

  // The overflow bucket is the last distinct key; it collects every operator that did not fit.
  EXPECT_EQ(counter.DistinctCount(), OpTypeCounter::kMaxDistinctKeys + 1);
  EXPECT_EQ(counter.ToString(1), "other=10");
}

TEST(WebGpuPluginTelemetryTest, SummaryIntervalDoublesToCap) {
  SummaryIntervalScheduler scheduler;
  EXPECT_EQ(scheduler.IntervalUs(), SummaryIntervalScheduler::kInitialIntervalUs);
  EXPECT_FALSE(scheduler.IsDue(SummaryIntervalScheduler::kInitialIntervalUs - 1));
  EXPECT_TRUE(scheduler.IsDue(SummaryIntervalScheduler::kInitialIntervalUs));

  scheduler.OnEmitted();
  EXPECT_EQ(scheduler.IntervalUs(), 2 * SummaryIntervalScheduler::kInitialIntervalUs);

  for (int i = 0; i < 64; ++i) {
    scheduler.OnEmitted();
  }
  EXPECT_EQ(scheduler.IntervalUs(), SummaryIntervalScheduler::kMaxIntervalUs);
  EXPECT_FALSE(scheduler.IsDue(SummaryIntervalScheduler::kMaxIntervalUs - 1));
}

TEST(WebGpuPluginTelemetryTest, RunCountersAccumulate) {
  RunCounters counters;
  EXPECT_FALSE(counters.HasData());

  counters.AddRun(100, true);
  counters.AddRun(300, false);
  counters.AddRun(200, true);
  EXPECT_TRUE(counters.HasData());
  EXPECT_EQ(counters.runs, 3u);
  EXPECT_EQ(counters.run_end_failures, 1u);
  EXPECT_EQ(counters.total_run_duration_us, 600);
  EXPECT_EQ(counters.max_run_duration_us, 300);

  counters.AddGraphReplay(true);
  counters.AddGraphReplay(false);
  EXPECT_EQ(counters.graph_replays, 2u);
  EXPECT_EQ(counters.graph_replay_failures, 1u);

  RunCounters start_failure_only;
  ++start_failure_only.run_start_failures;
  EXPECT_TRUE(start_failure_only.HasData());
}

}  // namespace test
}  // namespace onnxruntime
