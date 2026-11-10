// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <gtest/gtest.h>

#include "core/providers/webgpu/ep/telemetry_aggregation.h"
#include "core/providers/webgpu/ep/telemetry_privacy.h"

namespace onnxruntime {
namespace test {

using webgpu::ep::telemetry::BoundedUtf8;
using webgpu::ep::telemetry::IsNonFalseEnvironmentValue;
using webgpu::ep::telemetry::IsTrueEnvironmentValue;
using webgpu::ep::telemetry::kMaxStringBytes;
using webgpu::ep::telemetry::OpTypeCounter;
using webgpu::ep::telemetry::OpTypeTelemetryKey;
using webgpu::ep::telemetry::RunCounters;
using webgpu::ep::telemetry::ScrubPaths;
using webgpu::ep::telemetry::SummaryIntervalScheduler;

TEST(WebGpuPluginTelemetryTest, ScrubPathsRedactsToEndOfMessage) {
  EXPECT_EQ(ScrubPaths("no path here, n/a read/write"), "no path here, n/a read/write");
  EXPECT_EQ(ScrubPaths("failed to load C:\\Users\\First Last\\model.onnx: bad"), "failed to load [path]");
  EXPECT_EQ(ScrubPaths("open D:/models/x.onnx"), "open [path]");
  EXPECT_EQ(ScrubPaths("share \\\\server\\share\\m.onnx"), "share [path]");
  EXPECT_EQ(ScrubPaths("home ~/models/m.onnx"), "home [path]");
  EXPECT_EQ(ScrubPaths("posix /home/jane doe/m.onnx"), "posix [path]");
  EXPECT_EQ(ScrubPaths("relative alice\\models\\m.onnx"), "relative [path]");
  EXPECT_EQ(ScrubPaths("domain\\user only"), "domain\\user only");
}

TEST(WebGpuPluginTelemetryTest, ScrubPathsBoundsLongInput) {
  std::string long_message;
  while (long_message.size() < 2 * kMaxStringBytes) {
    long_message += "word ";
  }
  EXPECT_EQ(ScrubPaths(long_message).size(), kMaxStringBytes);
  // Input beyond the scrub window: an unseen suffix could turn the last token into a path, so it is redacted.
  EXPECT_EQ(ScrubPaths(std::string(20 * 1024, 'a')), "[path]");
  const std::string truncated = std::string(10, 'x') + " " + std::string(20 * 1024, 'y');
  EXPECT_EQ(ScrubPaths(truncated), std::string(10, 'x') + " [path]");
}

TEST(WebGpuPluginTelemetryTest, BoundedUtf8KeepsValidUtf8) {
  EXPECT_EQ(BoundedUtf8("abc"), "abc");
  EXPECT_EQ(BoundedUtf8("ab\xFF"
                        "c"),
            "ab?c");
  // A two-byte codepoint that does not fit is omitted entirely.
  EXPECT_EQ(BoundedUtf8("a\xC3\xA9", 2), "a");
  EXPECT_EQ(BoundedUtf8(std::string(2000, 'z')).size(), kMaxStringBytes);
}

TEST(WebGpuPluginTelemetryTest, EnvironmentValueParsing) {
  for (const char* value : {"1", "true", " YES ", "on", "y"}) {
    EXPECT_TRUE(IsTrueEnvironmentValue(value)) << value;
    EXPECT_TRUE(IsNonFalseEnvironmentValue(value)) << value;
  }
  for (const char* value : {"", "0", "false", "No", " off "}) {
    EXPECT_FALSE(IsTrueEnvironmentValue(value)) << value;
    EXPECT_FALSE(IsNonFalseEnvironmentValue(value)) << value;
  }
  EXPECT_FALSE(IsTrueEnvironmentValue("2"));
  EXPECT_TRUE(IsNonFalseEnvironmentValue("2"));
}

TEST(WebGpuPluginTelemetryTest, OpTypeKeyBucketsCustomOperators) {
  EXPECT_EQ(OpTypeTelemetryKey("", "NonZero", true), "ai.onnx:NonZero");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx", "NonZero", true), "ai.onnx:NonZero");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx.ml", "TreeEnsemble", true), "ai.onnx.ml:TreeEnsemble");
  EXPECT_EQ(OpTypeTelemetryKey("com.microsoft", "GroupQueryAttention", true), "com.microsoft:GroupQueryAttention");
  EXPECT_EQ(OpTypeTelemetryKey("com.ms.internal.nhwc", "Conv", true), "com.ms.internal.nhwc:Conv");
  // Operator names from other domains are model-defined and must not be reported.
  EXPECT_EQ(OpTypeTelemetryKey("com.contoso.secret", "ProjectX", true), "custom");
  EXPECT_EQ(OpTypeTelemetryKey("ai.onnx.contrib", "Tokenizer", true), "custom");
  // Custom operators can be registered in well-known domains; without a registered schema they are bucketed.
  EXPECT_EQ(OpTypeTelemetryKey("", "ContosoSecretFeatureExtractor", false), "custom");
  EXPECT_EQ(OpTypeTelemetryKey("com.microsoft", "ContosoOp", false), "custom");
}

TEST(WebGpuPluginTelemetryTest, OpTypeCounterOrdersByCountThenName) {
  OpTypeCounter counter;
  EXPECT_TRUE(counter.Empty());
  EXPECT_EQ(counter.ToString(10), "");

  counter.Add("", "NonZero", true);
  counter.Add("com.microsoft", "Foo", true);
  counter.Add("", "NonZero", true);
  counter.Add("", "Unique", true);
  counter.Add("com.microsoft", "Foo", true);
  counter.Add("", "NonZero", true);
  counter.Add("my.domain", "A", true);
  counter.Add("", "Secret", false);

  EXPECT_EQ(counter.DistinctCount(), 4u);
  EXPECT_EQ(counter.ToString(10), "ai.onnx:NonZero=3,com.microsoft:Foo=2,custom=2,ai.onnx:Unique=1");
  EXPECT_EQ(counter.ToString(2), "ai.onnx:NonZero=3,com.microsoft:Foo=2");
  EXPECT_EQ(counter.ToString(0), "");
}

TEST(WebGpuPluginTelemetryTest, OpTypeCounterBoundsDistinctKeys) {
  OpTypeCounter counter;
  const size_t total = OpTypeCounter::kMaxDistinctKeys + 10;
  for (size_t i = 0; i < total; ++i) {
    counter.Add("", "Op" + std::to_string(i), true);
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
