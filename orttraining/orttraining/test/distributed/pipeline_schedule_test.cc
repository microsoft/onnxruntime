// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/core/framework/pipeline.h"
#include "orttraining/core/framework/distributed_run_context.h"

namespace onnxruntime {
namespace test {

void TestPipelineScheduler(const int num_batches, const int num_stages, std::vector<std::vector<int>> baseline_events) {
  onnxruntime::training::pipeline::PipelineScheduler schedule(num_batches, num_stages, {0, 1, 2});
  for (int s = 0; s < num_stages; ++s) {
    for (int b = 0; b < num_batches; ++b) {
      const auto forward_recv_wait = schedule.GetForwardRecvWaitedEvent(b, s);
      const auto forward_recv_record = schedule.GetForwardRecvRecordedEvent(b, s);
      const auto forward_compute_wait = schedule.GetForwardComputeWaitedEvent(b, s);
      const auto forward_compute_record = schedule.GetForwardComputeRecordedEvent(b, s);
      const auto forward_send_wait = schedule.GetForwardSendWaitedEvent(b, s);
      const auto forward_send_record = schedule.GetForwardSendRecordedEvent(b, s);

      const auto backward_recv_wait = schedule.GetBackwardRecvWaitedEvent(b, s);
      const auto backward_recv_record = schedule.GetBackwardRecvRecordedEvent(b, s);
      const auto backward_compute_wait = schedule.GetBackwardComputeWaitedEvent(b, s);
      const auto backward_compute_record = schedule.GetBackwardComputeRecordedEvent(b, s);
      const auto backward_send_wait = schedule.GetBackwardSendWaitedEvent(b, s);
      const auto backward_send_record = schedule.GetBackwardSendRecordedEvent(b, s);

      const auto batch_stride = 6;
      const auto stage_stride = 2;
      EXPECT_EQ(forward_recv_wait, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 0)) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_recv_record, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 1)) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_compute_wait, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 2)) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_compute_record, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 3)) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_send_wait, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 4)) << " batch " << b << " stage " << s;
      EXPECT_EQ(forward_send_record, baseline_events.at(stage_stride * s + 0).at(batch_stride * b + 5)) << " batch " << b << " stage " << s;

      EXPECT_EQ(backward_recv_wait, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 0)) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_recv_record, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 1)) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_compute_wait, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 2)) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_compute_record, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 3)) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_send_wait, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 4)) << " batch " << b << " stage " << s;
      EXPECT_EQ(backward_send_record, baseline_events.at(stage_stride * s + 1).at(batch_stride * b + 5)) << " batch " << b << " stage " << s;
    }
  }
}

TEST(Pipeline, ScheduleB5S3) {
  const int num_batches = 5;
  const int num_stages = 3;

  // The event baselines at different stages are the same.
  // The first 4 events are for the first computation on that stage.
  // Similarly, the last 4 events are for the last computation on that stage.
  // Below, we add comments to indicate which computation the events associated with.

  // Each line below sequentially contains waited and recorded events in the following pattern.
  //   WaitEvent -> Recv -> RecordEvent -> WaitEvent -> FW/BW -> RecordEvent -> WaitEvent -> Send -> RecordEvent.
  std::vector<int> forward_baseline_events_stage0{
      -1, -1, -1, 0, 0, 1,     // None -> None -> None -> Wait -> FW00 -> Record -> Wait -> Send -> Record @ stage 0
      -1, -1, 1, 2, 2, 3,      //                                 FW01
      -1, -1, 3, 4, 4, 5,      //                                 FW02
      -1, -1, 7, 8, 8, 9,      //                                 FW03
      -1, -1, 10, 11, 11, 12,  //                                 FW04
  };

  std::vector<int> backward_baseline_events_stage0{
      5, 6, 6, 7, -1, -1,      // Wait -> Recv -> Record -> Wait -> BW00 -> Record -> None -> None -> None @ stage 0
      8, 9, 9, 10, -1, -1,     //                                   BW01
      11, 12, 12, 13, -1, -1,  //                                   BW02
      13, 14, 14, 15, -1, -1,  //                                   BW03
      15, 16, 16, 17, -1, -1   //                                   BW04
  };

  std::vector<int> forward_baseline_events_stage1{
      -1, 0, 0, 1, 1, 2,       // Wait -> Recv -> Record -> Wait -> FW00 -> Record -> Wait -> Send -> Record @ stage 1
      1, 2, 2, 3, 5, 6,        //                                   FW01
      3, 4, 4, 5, 8, 9,        //                                   FW02
      10, 11, 11, 12, 12, 13,  //                                   FW03
      14, 15, 15, 16, 16, 17   //                                   FW04
  };

  std::vector<int> backward_baseline_events_stage1{
      5, 6, 6, 7, 7, 8,        // Wait -> Recv -> Record -> Wait -> BW00 -> Record -> Wait -> Send -> Record @ stage 1
      8, 9, 9, 10, 10, 11,     //                                   BW01
      12, 13, 13, 14, 14, 15,  //                                   BW02
      16, 17, 17, 18, 18, 19,  //                                   BW03
      19, 20, 20, 21, 21, 22   //                                   BW04
  };

  std::vector<int> forward_baseline_events_stage2{
      -1, 0, 0, 1, -1, -1,    // Wait -> Recv -> Record -> Wait -> FW00 -> Record -> None -> None -> None @ stage 2
      2, 3, 3, 4, -1, -1,     //                                   FW01
      5, 6, 6, 7, -1, -1,     //                                   FW02
      8, 9, 9, 10, -1, -1,    //                                   FW03
      11, 12, 12, 13, -1, -1  //                                   FW04
  };

  std::vector<int> backward_baseline_events_stage2{
      -1, -1, 1, 2, 2, 3,      // None -> None -> None -> Wait -> BW00 -> Record -> Wait -> Send -> Record @ stage 2
      -1, -1, 4, 5, 5, 6,      //                                 BW01
      -1, -1, 7, 8, 8, 9,      //                                 BW02
      -1, -1, 10, 11, 11, 12,  //                                 BW03
      -1, -1, 13, 14, 14, 15   //                                 BW04
  };

  std::vector<std::vector<int>> baseline_events{
      forward_baseline_events_stage0, backward_baseline_events_stage0,
      forward_baseline_events_stage1, backward_baseline_events_stage1,
      forward_baseline_events_stage2, backward_baseline_events_stage2};

  TestPipelineScheduler(num_batches, num_stages, baseline_events);
}

}  // namespace test
}  // namespace onnxruntime