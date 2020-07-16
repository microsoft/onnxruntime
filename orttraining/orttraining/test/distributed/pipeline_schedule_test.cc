// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/models/runner/pipeline.h"

namespace onnxruntime {
namespace test {

void TestPipelineScheduler(const int num_batches, const int num_stages, std::vector<std::vector<int>> baseline_events) {
  onnxruntime::training::pipeline::PipelineScheduler schedule(num_batches, num_stages);
  for (int s = 0; s < num_stages; ++s) {
    for (int b = 0; b < num_batches; ++b) {
      const int forward_wait_before_recv = schedule.GetForwardWaitedEventBeforeRecv(b, s);
      const int forward_wait_after_recv = schedule.GetForwardWaitedEventAfterRecv(b, s);
      const int forward_record_before_send = schedule.GetForwardRecordedEventBeforeSend(b, s);
      const int forward_record_after_send = schedule.GetForwardRecordedEventAfterSend(b, s);
      const int backward_wait_before_recv = schedule.GetBackwardWaitedEventBeforeRecv(b, s);
      const int backward_wait_after_recv = schedule.GetBackwardWaitedEventAfterRecv(b, s);
      const int backward_record_before_send = schedule.GetBackwardRecordedEventBeforeSend(b, s);
      const int backward_record_after_send = schedule.GetBackwardRecordedEventAfterSend(b, s);

      EXPECT_EQ(forward_wait_before_recv, baseline_events[2 * s + 0][4 * b + 0]) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_wait_after_recv, baseline_events[2 * s + 0][4 * b + 1]) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_record_before_send, baseline_events[2 * s + 0][4 * b + 2]) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_record_after_send, baseline_events[2 * s + 0][4 * b + 3]) << " batch " << b << " stage " << s;

      EXPECT_EQ(backward_wait_before_recv, baseline_events[2 * s + 1][4 * b + 0]) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_wait_after_recv, baseline_events[2 * s + 1][4 * b + 1]) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_record_before_send, baseline_events[2 * s + 1][4 * b + 2]) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_record_after_send, baseline_events[2 * s + 1][4 * b + 3]) << " batch " << b << " stage " << s;
    }
  }
}

TEST(Pipeline, ScheduleB8S3) {
  const int num_batches = 8;
  const int num_stages = 3;

  // The event baselines at different stages are the same.
  // The first 4 events are for the first computation on that stage.
  // Similarly, the last 4 events are for the last computation on that stage.
  // Below, we add comments to indicate which computation the events associated with.

  // Format per line below:
  //   waited event before Recv, waited event after Recv, recorded event before Send, recorded event after Send.
  // The value "-1" means a NULL event; RecordEvent and WaitEvent do nothing for NULL events.
  // Note that the computation pattern is
  //   WaitEvent -> Recv -> WaitEvent -> FW/BW -> RecordEvent -> Send -> RecordEvent.
  std::vector<int> forward_baseline_events_stage0{
      -1, -1, 0, 1,    // FW00 @ stage 0
      0, 1, 2, 3,      // FW01
      2, 3, 4, 5,      // FW02
      6, 7, 8, 9,      // FW03
      10, 11, 12, 13,  // FW04
      14, 15, 16, 17,  // FW05
      18, 19, 20, 21,  // FW06
      22, 23, 24, 25,  // FW07
  };

  std::vector<int> backward_baseline_events_stage0{
      4, 5, 6, 7,      // BW00
      8, 9, 10, 11,    // BW01
      12, 13, 14, 15,  // BW02
      16, 17, 18, 19,  // BW03
      20, 21, 22, 23,  // BW04
      24, 25, 26, 27,  // BW05
      26, 27, 28, 29,  // BW06
      28, 29, 30, 31   // BW07
  };

  std::vector<int> forward_baseline_events_stage1{
      -1, -1, 0, 1,    // FW00 @ stage 1
      0, 1, 2, 3,      // FW01
      2, 3, 4, 5,      // FW02
      8, 9, 10, 11,    // FW03
      12, 13, 14, 15,  // FW04
      16, 17, 18, 19,  // FW05
      20, 21, 22, 23,  // FW06
      24, 25, 26, 27,  // FW07
  };

  std::vector<int> backward_baseline_events_stage1{
      4, 5, 6, 7,      // BW00
      6, 7, 8, 9,      // BW01
      10, 11, 12, 13,  // BW02
      14, 15, 16, 17,  // BW03
      18, 19, 20, 21,  // BW04
      22, 23, 24, 25,  // BW05
      26, 27, 28, 29,  // BW06
      28, 29, 30, 31   // BW07
  };

  std::vector<int> forward_baseline_events_stage2{
      -1, -1, 0, 1,    // FW00 @ stage 2
      2, 3, 4, 5,      // FW01
      6, 7, 8, 9,      // FW02
      10, 11, 12, 13,  // FW03
      14, 15, 16, 17,  // FW04
      18, 19, 20, 21,  // FW05
      22, 23, 24, 25,  // FW06
      26, 27, 28, 29,  // FW07
  };

  std::vector<int> backward_baseline_events_stage2{
      0, 1, 2, 3,      // BW00
      4, 5, 6, 7,      // BW01
      8, 9, 10, 11,    // BW02
      12, 13, 14, 15,  // BW03
      16, 17, 18, 19,  // BW04
      20, 21, 22, 23,  // BW05
      24, 25, 26, 27,  // BW06
      28, 29, 30, 31   // BW07
  };

  std::vector<std::vector<int>> baseline_events{
      forward_baseline_events_stage0, backward_baseline_events_stage0,
      forward_baseline_events_stage1, backward_baseline_events_stage1,
      forward_baseline_events_stage2, backward_baseline_events_stage2};

  TestPipelineScheduler(num_batches, num_stages, baseline_events);
}

}  // namespace test
}  // namespace onnxruntime