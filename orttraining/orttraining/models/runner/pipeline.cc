// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/models/runner/pipeline.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "gsl/gsl"
#include "core/framework/ml_value.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"

namespace onnxruntime {
namespace training {
namespace pipeline {

Slot::Slot() {
  type = Empty;
  batch_id = 0;
}

bool Slot::IsEmpty() const {
  return type == Empty;
}

bool Slot::IsForward() const {
  return type == Forward;
}

bool Slot::IsBackward() const {
  return type == Backward;
}

void Slot::Show() const {
  const int waited_event_id =
      waited_events.size() > 0 ? waited_events[0] : -1;
  const int waited_event_id_after_recv =
      waited_events.size() > 0 ? waited_events[1] : -1;
  const int recorded_event_id_before_send =
      recorded_events.size() > 0 ? recorded_events[0] : -1;
  const int recorded_event_id =
      recorded_events.size() > 0 ? recorded_events[1] : -1;
  switch (type) {
    case Empty:
      std::cout << "             ";
      break;
    case Forward:
      std::cout << " F" << batch_id << "("
                << waited_event_id << "," << waited_event_id_after_recv << ","
                << recorded_event_id_before_send << "," << recorded_event_id
                << ")"
                << " ";
      break;
    case Backward:
      std::cout << " B" << batch_id << "("
                << waited_event_id << "," << waited_event_id_after_recv << ","
                << recorded_event_id_before_send << "," << recorded_event_id
                << ")"
                << " ";
      break;
  }
}

PipelineSchedule::PipelineSchedule(int num_stages) {
  num_stages_ = num_stages;
  num_batches_ = 0;
}

void PipelineSchedule::Add(int batch_id) {
  ++num_batches_;

  // Expand table to accomonadate the new batch.
  const int required_max_time = 2 * num_stages_ + 2 * (num_batches_ - 1);
  const int current_max_time = static_cast<int>(table_.size());

  for (int t = current_max_time; t < required_max_time; ++t) {
    table_.push_back(std::vector<Slot>(num_stages_));
    batch_count_.push_back(0);
  }

  std::vector<int> forward_time(num_stages_, 0);

  // Insert forward.
  for (int s = 0; s < num_stages_; ++s) {
    for (int t = 0; t < required_max_time; ++t) {
      if (!table_[t][s].IsEmpty()) {
        // One slot cannot be occupied by two batches.
        continue;
      }

      if (s > 0 && t <= forward_time[s - 1]) {
        // Foward of the s-th stage must happen after Forward of (s-1)-th stage.
        // Note that forward_time[s] is the time slot of the s-th stage.
        continue;
      }

      if (batch_count_[t] >= num_stages_) {
        // At time t, the number of running batches is at maximum,
        // so we need to put this stage to another time slot.
        continue;
      }

      // The s-th stage happens at time t.
      forward_time[s] = t;
      // This s-th stage is forward pass of the batch_id-th batch.
      table_[t][s].type = Slot::Type::Forward;
      table_[t][s].batch_id = batch_id;

      auto events = SearchLastRecordedEvents(t, s);
      table_[t][s].waited_events = events;

      if (events.size() != 0) {
        table_[t][s].recorded_events = std::vector<int>{events[events.size() - 1] + 1,
                                                        events[events.size() - 1] + 2};
      } else {
        table_[t][s].recorded_events = std::vector<int>{0, 1};
      }

      for (int t_ = t + 1; t_ < required_max_time; ++t_) {
        if (table_[t_][s].IsEmpty()) {
          continue;
        }
        // Find the non-empty slot happens right before this slot.
        events = SearchLastRecordedEvents(t_, s);
        // Wait previously recorded events in the slot right before this slot table_[t_][s].
        table_[t_][s].waited_events = events;
        // Generate two new unique events; one for RecordEvent before Send and the other one
        // for RecordEvent after Send.
        table_[t_][s].recorded_events = std::vector<int>{events[events.size() - 1] + 1,
                                                         events[events.size() - 1] + 2};
      }

      break;
    }
  }

  // Insert backward.
  std::vector<int> backward_time(num_stages_, 0);
  for (int s = num_stages_ - 1; s > -1; --s) {
    for (int t = 0; t < required_max_time; ++t) {
      if (!table_[t][s].IsEmpty()) {
        continue;
      }

      if (s < num_stages_ - 1 && t <= backward_time[s + 1]) {
        continue;
      }

      if (t <= forward_time[num_stages_ - 1]) {
        continue;
      }

      if (batch_count_[t] >= num_stages_) {
        continue;
      }

      backward_time[s] = t;
      table_[t][s].type = Slot::Type::Backward;
      table_[t][s].batch_id = batch_id;

      auto events = SearchLastRecordedEvents(t, s);
      table_[t][s].waited_events = events;

      if (events.size() != 0) {
        table_[t][s].recorded_events = std::vector<int>{events[events.size() - 1] + 1,
                                                        events[events.size() - 1] + 2};
      } else {
        table_[t][s].recorded_events = std::vector<int>{/* first event id */ 0};
      }

      for (int t_ = t + 1; t_ < required_max_time; ++t_) {
        if (table_[t_][s].IsEmpty()) {
          continue;
        }
        events = SearchLastRecordedEvents(t_, s);
        table_[t_][s].waited_events = events;
        table_[t][s].recorded_events = std::vector<int>{events[events.size() - 1] + 1,
                                                        events[events.size() - 1] + 2};
      }

      break;
    }
  }

  for (int t = forward_time[0]; t <= forward_time[num_stages_ - 1]; ++t) {
    ++batch_count_[t];
  }

  for (int t = backward_time[num_stages_ - 1]; t <= backward_time[0]; ++t) {
    ++batch_count_[t];
  }
}

void PipelineSchedule::Add(int batch_id_begin, int batch_id_end) {
  for (int i = batch_id_begin; i < batch_id_end; ++i) {
    Add(i);
  }
}

int PipelineSchedule::GetForwardWaitedEventId(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsForward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.waited_events;
  }
  return events[0];
}

int PipelineSchedule::GetForwardWaitedEventIdAfterRecv(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsForward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.waited_events;
  }
  return events[1];
}

int PipelineSchedule::GetForwardRecordedEventIdBeforeSend(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsForward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.recorded_events;
  }
  return events[0];
}

int PipelineSchedule::GetForwardRecordedEventId(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsForward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.recorded_events;
  }
  return events[1];
}

int PipelineSchedule::GetBackwardWaitedEventId(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsBackward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.waited_events;
  }
  return events[0];
}

int PipelineSchedule::GetBackwardWaitedEventIdAfterRecv(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsBackward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.waited_events;
  }
  return events[1];
}

int PipelineSchedule::GetBackwardRecordedEventIdBeforeSend(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsBackward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.recorded_events;
  }
  return events[0];
}

int PipelineSchedule::GetBackwardRecordedEventId(int stage_id, int batch_id) const {
  std::vector<int> events = {-1, -1};
  for (size_t t = 0; t < table_.size(); ++t) {
    auto& slot = table_[t][stage_id];
    if (!slot.IsBackward()) {
      continue;
    }
    if (slot.batch_id != batch_id) {
      continue;
    }
    events = slot.recorded_events;
  }
  return events[1];
}

void PipelineSchedule::Show() const {
  const int num_slots = static_cast<int>(table_.size());

  for (int s = 0; s < num_stages_; ++s) {
    for (int t = 0; t < num_slots; ++t) {
      table_[t][s].Show();
      if (t == num_slots - 1) {
        std::cout << std::endl;
      }
    }
  }
}

std::vector<int> PipelineSchedule::SearchLastRecordedEvents(int time_id, int stage_id) const {
  std::vector<int> events;
  for (int t = time_id - 1; t >= 0; --t) {
    auto& slot = table_[t][stage_id];
    if (slot.IsEmpty()) {
      continue;
    }

    events = slot.recorded_events;
    break;
  }

  return events;
}

void PipelineWorkerPool::Join(size_t worker_id) {
  auto& worker = workers[worker_id];
  if (!worker.joinable())
    return;
  worker.join();
}

void PipelineWorkerPool::JoinAll() {
  for (size_t i = 0; i < workers.size(); ++i) {
    auto& worker = workers[i];
    if (!worker.joinable())
      continue;
    worker.join();
  };
}

}  // namespace pipeline
}  // namespace training
}  // namespace onnxruntime