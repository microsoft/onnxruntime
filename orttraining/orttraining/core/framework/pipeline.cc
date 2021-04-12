// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/pipeline.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <iomanip>

namespace onnxruntime {
namespace training {
namespace pipeline {

std::ostream& operator<<(std::ostream& stream, const PipelineTask& slot) {
  if (slot.pass == PipelineTask::Pass::Forward) {
    switch (slot.type) {
      case PipelineTask::Type::Compute:
        stream << "FW";
        break;
      case PipelineTask::Type::Send:
        stream << "FS";
        break;
      case PipelineTask::Type::Recv:
        stream << "FR";
        break;
      default:
        throw std::invalid_argument("Unsupported forwoard action.");
        break;
    }
  } else if (slot.pass == PipelineTask::Pass::Backward) {
    switch (slot.type) {
      case PipelineTask::Type::Compute:
        stream << "BW";
        break;
      case PipelineTask::Type::Send:
        stream << "BS";
        break;
      case PipelineTask::Type::Recv:
        stream << "BR";
        break;
      default:
        throw std::invalid_argument("Unsupported backward action.");
        break;
    }
  } else {
    throw std::invalid_argument("Unsupported pass type.");
  }
  std::ios state(nullptr);
  state.copyfmt(state);
  stream << std::setw(2) << std::setfill('0') << slot.batch;
  stream.copyfmt(state);
  return stream;
}

void PipelineSlot::AddSend(const int batch_id, const PipelineTask::Pass pass, const int upstream_time, const int upstream_stage, const int this_rank, const int peer_rank) {
  tasks_.push_back(PipelineTask{batch_id, PipelineTask::Type::Send, pass, upstream_time, upstream_stage, this_rank, peer_rank});
}
void PipelineSlot::AddRecv(const int batch_id, const PipelineTask::Pass pass, const int upstream_time, const int upstream_stage, const int this_rank, const int peer_rank) {
  tasks_.push_back(PipelineTask{batch_id, PipelineTask::Type::Recv, pass, upstream_time, upstream_stage, this_rank, peer_rank});
}
void PipelineSlot::AddCompute(const int batch_id, const PipelineTask::Pass pass, const int upstream_time, const int upstream_stage) {
  tasks_.push_back(PipelineTask{batch_id, PipelineTask::Type::Compute, pass, upstream_time, upstream_stage});
}

PipelineTask& PipelineSlot::operator[](int index) {
  return tasks_.at(index);
}

const PipelineTask& PipelineSlot::operator[](int index) const {
  return tasks_.at(index);
}

PipelineTask& PipelineSlot::GetFrontAction() {
  return tasks_.front();
}

const PipelineTask& PipelineSlot::GetFrontAction() const {
  return tasks_.front();
}

PipelineScheduler::PipelineScheduler() : num_stages_(0),
                                         num_batches_(0) {}

PipelineScheduler::PipelineScheduler(
    int num_batches,
    const int num_stages,
    const std::vector<int>& stage_id_to_rank_id_map) : num_stages_(num_stages),
                                                       num_batches_(num_batches),
                                                       stage_id_to_rank_id_map_(stage_id_to_rank_id_map) {
  if (stage_id_to_rank_id_map.size() != static_cast<size_t>(num_stages)) {
    throw std::invalid_argument("stage_id_to_rank_id_map should contain the MPI ranks from the first to the last pipeline stages");
  }

  CreateComputeSchedule();

  const size_t num_events_per_slot_compute_side = 2;
  std::vector<int> compute_default_events(num_events_per_slot_compute_side, -1);
  InsertEvents(compute_table_, num_events_per_slot_compute_side, compute_default_events);

  CreateFullSchedule();

  // Change the pipeline stage ID to actual MPI rank in Send and Recv.
  MapStageIdToMpiRank();

  const size_t num_events_per_slot_side = 1;
  std::vector<int> default_events(num_events_per_slot_side, -1);
  InsertEvents(compute_commute_table_, num_events_per_slot_side, default_events);
}

// Print this structure following a fixed-length format.
// It assumes there are at most 2 actions per slot.
std::ostream& operator<<(std::ostream& stream, const PipelineSlot& slot) {
  switch (slot.NumActions()) {
    case 0:
      // One empty action.
      stream << "    ";
      // Another empty action.
      stream << "    ";
      break;
    case 1:
      // Print an action.
      stream << slot[0];
      // One empty action.
      stream << "    ";
      break;
    case 2:
      // Print an action.
      stream << slot[0];
      // Print another action.
      stream << slot[1];
      break;
  }
  return stream;
}

void PipelineSlot::SetWaitedEvent(const std::vector<int> events) {
  waited_events_ = std::move(events);
}

std::vector<int> PipelineSlot::GetWaitedEvent() const {
  return waited_events_;
}

void PipelineSlot::SetRecordedEvent(const std::vector<int> events) {
  recorded_events_ = std::move(events);
}

std::vector<int> PipelineSlot::GetRecordedEvent() const {
  return recorded_events_;
}

std::ostream& operator<<(std::ostream& stream, PipelineScheduler const& schedule) {
  // print something from v to str, e.g: Str << v.getX();
  stream << "-------------View of Compute Schedule-------------" << std::endl;
  for (int s = 0; s < schedule.num_stages_; ++s) {
    for (size_t t = 0; t < schedule.compute_table_.size(); ++t) {
      stream << schedule.compute_table_.at(t).at(s);
    }
    stream << std::endl;
  }

  stream << "-------------View of Compute-commute Schedule-------------" << std::endl;
  for (int s = 0; s < schedule.num_stages_; ++s) {
    for (size_t t = 0; t < schedule.compute_commute_table_.size(); ++t) {
      stream << schedule.compute_commute_table_.at(t).at(s);
    }
    stream << std::endl;
  }

  return stream;
}

// Return time indexes of a given batch.
// i-th returned element is the time of batch_id's forward at stage i.
// previous_forward_time[s] is the time that the last forward happens on stage s.
std::vector<int> PipelineScheduler::FindForwardComputeTime(const std::vector<int> previous_forward_time) const {
  // forward_time[i]: i-th stage's forward time of batch_id.
  std::vector<int> forward_time(num_stages_, 0);

  for (int s = 0; s < num_stages_; ++s) {
    for (int t = previous_forward_time.at(s); t < static_cast<int>(compute_table_.size()); ++t) {
      if (!compute_table_.at(t).at(s).IsEmpty()) {
        // One slot cannot be occupied by two batches.
        continue;
      }

      if (s > 0 && t <= forward_time.at(s - 1)) {
        // Foward of the s-th stage must happen after forward of (s-1)-th stage.
        // Note that forward_time[s] is the time slot of the s-th stage.
        continue;
      }

      if (compute_batch_count_.at(t) >= num_stages_) {
        // At time t, the number of running batches is at maximum,
        // so we need to put this stage to another time slot.
        continue;
      }

      // For batch_id, its forward happens at time t on the s-th stage.
      forward_time.at(s) = t;

      break;
    }
  }

  return forward_time;
}

// Return time indexes of a given batch.
// i-th returned element is the time of batch_id's backward at stage i.
// forward_time[s] is the forward time for the given batch on stage s.
std::vector<int> PipelineScheduler::FindBackwardComputeTime(const std::vector<int> forward_time) const {
  std::vector<int> backward_time(num_stages_, 0);
  // For a specific batch, the last stage has the earliest backward computation.
  // Thus, the first loop reversely scans stages.
  for (int s = num_stages_ - 1; s > -1; --s) {
    for (int t = forward_time.at(s) + 1; t < static_cast<int>(compute_table_.size()); ++t) {
      if (!compute_table_.at(t).at(s).IsEmpty()) {
        continue;
      }

      if (s < num_stages_ - 1 && t <= backward_time.at(s + 1)) {
        continue;
      }

      if (compute_batch_count_.at(t) >= num_stages_) {
        continue;
      }

      backward_time.at(s) = t;
      break;
    }
  }
  return backward_time;
}

int PipelineScheduler::FindSendRecvTime(const int upstream_compute_time, const int upstream_stage, const int stage) const {
  int good_time = -1;
  // Search for a time to insert Recv and then Send in full table.
  // Recv is on slot's stage.
  // Send is on upstream slot's stage.
  for (int full_t = static_cast<int>(compute_commute_table_.size()) - 1; full_t > upstream_compute_time; --full_t) {
    bool is_good_time = true;
    for (int full_s = 0; full_s < num_stages_; ++full_s) {
      auto& candidate_slot = compute_commute_table_.at(full_t).at(full_s);

      if (candidate_slot.HasCompute()) {
        is_good_time = false;
        break;
      }

      if (candidate_slot.HasRecvFrom(upstream_stage)) {
        is_good_time = false;
        break;
      }

      if (candidate_slot.HasRendTo(stage)) {
        is_good_time = false;
        break;
      }
    }

    if (!is_good_time) {
      continue;
    }

    good_time = full_t;
    break;
  }
  return good_time;
}

void PipelineScheduler::CreateFullSchedule() {
  for (int t = 0; static_cast<size_t>(t) < compute_table_.size(); ++t) {
    // The last stage is compute, so we append one slot for commute.
    if (t != 0) {
      compute_commute_table_.push_back(std::vector<PipelineSlot>(num_stages_));
    }

    for (int s = 0; s < num_stages_; ++s) {
      // Read a slot from compute-only schedule.
      // We will build send-recv pair to connect this slot with its upstream slot.
      auto slot = compute_table_.at(t).at(s);

      if (slot.IsEmpty()) {
        // No need to insert Send and Recv for empty compute slot.
        continue;
      }

      // Ok, this slot is not empty but it must have exactly one action.
      if (slot.NumActions() != 1) {
        throw std::invalid_argument("In compute-only schedule, one slot can have only one action, whose type is Compute.");
      }

      // Get the only action in this slot.
      const auto action = slot.GetFrontAction();

      const int batch = action.batch;
      // Stage of upstream compute in compute-only table.
      const auto upstream_s = action.upstream_stage;
      // Time of upstream compute in compute-only table.
      const auto upstream_t = action.upstream_time;

      if (upstream_s < 0 && upstream_t < 0) {
        // This operator has no upstream, so we don't need to create Send and Recv.
        continue;
      }

      if (s == num_stages_ - 1 && action.IsBackward() && action.IsCompute()) {
        continue;
      }

      // Upstream of "slot".
      const auto upstream_slot = compute_table_.at(upstream_t).at(upstream_s);

      // Get the only action in upstream slot.
      const auto upstream_action = upstream_slot.GetFrontAction();

      // Time of upstream compute in full table.
      const auto upstream_compute_time = upstream_action.full_table_time;

      // Forward/backward information is sent by forward/backward Send.
      const PipelineTask::Pass recv_pass = action.IsForward() ? PipelineTask::Pass::Forward : PipelineTask::Pass::Backward;
      // Forward/backward information is received by forward/backward Send.
      const PipelineTask::Pass send_pass = upstream_action.IsForward() ? PipelineTask::Pass::Forward : PipelineTask::Pass::Backward;

      // Find a time index to insert send-recv pair in the schedule with compute and commute actions.
      const int good_time = FindSendRecvTime(upstream_compute_time, upstream_s, s);

      // Add Send and Recv to compute-commute schedule.
      // Send from upstream_s-th stage.
      // Recv at s-th stage.
      compute_commute_table_.at(good_time).at(upstream_s).AddSend(batch, send_pass, upstream_compute_time, upstream_s, upstream_s, s);
      compute_commute_table_.at(good_time).at(s).AddRecv(batch, recv_pass, good_time, s, s, upstream_s);
    }

    // Actions in compute_table_[t] are going be copied to full schedule.
    // For each action, we store its actual time and responsding etage in compute-only schedule.
    // The stored information may be carried to full schedule.
    for (int s = 0; s < num_stages_; ++s) {
      auto slot = compute_table_.at(t).at(s);
      for (int a = 0; a < static_cast<int>(slot.NumActions()); ++a) {
        auto& task = slot[a];
        task.full_table_time = static_cast<int>(compute_commute_table_.size());
        task.full_table_stage = s;
      }
    }

    // Copy compute actions from compute-only schedule to compute-commute schedule.
    compute_commute_table_.push_back(compute_table_.at(t));
  }
}

void PipelineScheduler::MapStageIdToMpiRank() {
  for (int t = 0; static_cast<size_t>(t) < compute_commute_table_.size(); ++t) {
    for (int s = 0; s < num_stages_; ++s) {
      // Slot at time t and pipeline stage s. It's a collection of tasks.
      auto& slot = compute_commute_table_.at(t).at(s);
      for (int k = 0; k < slot.Size(); ++k) {
        auto& task = slot[k];
        if (!task.IsCommute()) {
          continue;
        }

        // task.this_rank is actually the stage ID.
        // task.this_rank=0 means this stage is the first pipeline stage.
        // If data parallel is enabled, we may have multiple pipeline parallel groups.
        // The code below maps stage ID to MPI's world rank.
        task.this_rank = stage_id_to_rank_id_map_.at(task.this_rank);

        // Similarly, we do the mapping for peer's rank to know which process to communicate with
        // in runtime.
        task.peer_rank = stage_id_to_rank_id_map_.at(task.peer_rank);
      }
    }
  }
}

void PipelineScheduler::InsertEvents(std::vector<std::vector<PipelineSlot>>& schedule, const size_t num_events_per_slot, const std::vector<int> initial_events) {
  std::vector<std::vector<int>> last_recorded_events(num_stages_, initial_events);

  for (int t = 0; static_cast<size_t>(t) < schedule.size(); ++t) {
    for (int s = 0; s < num_stages_; ++s) {
      if (schedule.at(t).at(s).IsEmpty()) {
        continue;
      }
      schedule.at(t).at(s).SetWaitedEvent(last_recorded_events.at(s));

      // Create new recorded events. Their indexes should be greater than those of previous events.
      const auto max_event = std::max_element(last_recorded_events.at(s).begin(), last_recorded_events.at(s).end());
      std::vector<int> new_recorded_events;
      for (int i = 0; static_cast<size_t>(i) < num_events_per_slot; ++i) {
        new_recorded_events.push_back(*max_event + i + 1);
      }

      schedule.at(t).at(s).SetRecordedEvent(new_recorded_events);
      last_recorded_events.at(s) = schedule.at(t).at(s).GetRecordedEvent();
    }
  }
}

void PipelineScheduler::InsertForwardCompute(const int batch_id, const std::vector<int> forward_time) {
  // Occupy the time slots so that these slots won't be used in later iterations.
  for (int s = 0; s < num_stages_; ++s) {
    const auto batch_forward_time = forward_time.at(s);
    if (s == 0) {
      // The first forward compute has no upstream action.
      compute_table_.at(batch_forward_time).at(s).AddCompute(batch_id, PipelineTask::Pass::Forward);
    } else {
      // For other cases, forward at stage s happens after forward at stage s - 1.
      compute_table_.at(batch_forward_time).at(s).AddCompute(batch_id, PipelineTask::Pass::Forward, forward_time.at(s - 1), s - 1);
    }
  }
}

void PipelineScheduler::InsertBackwardCompute(const int batch_id, const std::vector<int> forward_time, const std::vector<int> backward_time) {
  // Occupy the time slots so that these slots won't be used in later iterations.
  const auto last_stage_index = num_stages_ - 1;
  for (int s = num_stages_ - 1; s >= 0; --s) {
    const auto batch_backward_time = backward_time.at(s);
    if (s == last_stage_index) {
      // The first backward (on the last pipeline stage) depends on the a forward on the last pipeline stage.
      compute_table_.at(batch_backward_time).at(s).AddCompute(batch_id, PipelineTask::Pass::Backward, forward_time.at(s), s);
    } else {
      // For other cases, backward at stage s depedns on the backward at stage s + 1.
      compute_table_.at(batch_backward_time).at(s).AddCompute(batch_id, PipelineTask::Pass::Backward, backward_time.at(s + 1), s + 1);
    }
  }
}

void PipelineScheduler::CreateComputeSchedule() {
  // Expand table to accomonadate the new batch.
  const int compute_max_time = 2 * num_stages_ + 2 * (num_batches_ - 1);

  compute_table_.resize(compute_max_time, std::vector<PipelineSlot>(num_stages_));
  compute_batch_count_.resize(compute_max_time);
  commute_batch_count_.resize(compute_max_time);

  std::vector<int> forward_time(num_stages_, 0);
  std::vector<int> backward_time(num_stages_, 0);
  for (int batch_id = 0; batch_id < num_batches_; ++batch_id) {
    // Find slot to insert forward compute.
    // The search on stage[s] starts at time forward_time[s].
    forward_time = FindForwardComputeTime(forward_time);

    // Insert forward Compute's at different stages based on forward_time found.
    InsertForwardCompute(batch_id, forward_time);

    // Find slot to insert backward compute.
    // The search on stage[s] starts at time forward_time[s].
    backward_time = FindBackwardComputeTime(forward_time);

    // Insert backward Compute's at different stages based on backward_time found.
    InsertBackwardCompute(batch_id, forward_time, backward_time);

    // Increase the batch count for whenever that batch stays in the pipeline.
    // For a specific batch, its starting/end time is the time it enters/leaves the first pipeline stage.
    for (int t_compute = forward_time.at(0); t_compute <= backward_time.at(0); ++t_compute) {
      ++compute_batch_count_.at(t_compute);
    }
  }
}

std::vector<int> PipelineScheduler::TryGetEvent(
    const bool is_waited_event,
    const int batch_id,
    const int stage_id,
    const PipelineTask::Pass pass,
    const PipelineTask::Type type,
    bool& is_found) const {
  is_found = false;

  // Go through slots of stage stage_id to find the requested action.
  for (int t = 0; static_cast<size_t>(t) < compute_commute_table_.size(); ++t) {
    const auto slot = compute_commute_table_.at(t).at(stage_id);
    for (int a = 0; static_cast<size_t>(a) < slot.NumActions(); ++a) {
      auto task = slot[a];
      if (task.batch != batch_id) {
        continue;
      }
      if (task.pass != pass) {
        continue;
      }
      if (task.type != type) {
        continue;
      }

      // Slot of the asked action is found, so we return its event.
      is_found = true;
      return is_waited_event ? slot.GetWaitedEvent() : slot.GetRecordedEvent();
    }
  }

  return std::vector<int>();
}

int PipelineScheduler::GetEventOrDefault(
    const bool is_waited_event,
    const int batch_id,
    const int stage_id,
    const PipelineTask::Pass pass,
    const PipelineTask::Type type) const {
  bool is_found = false;
  auto events = TryGetEvent(is_waited_event, batch_id, stage_id, pass, type, is_found);
  if (is_found) {
    return events.front();
  } else {
    return -1;
  }
}

// Forward Compute
int PipelineScheduler::GetForwardComputeWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Compute);
}

// Forward Compute
int PipelineScheduler::GetForwardComputeRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Compute);
}

// Backward Compute
int PipelineScheduler::GetBackwardComputeWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Compute);
}

// Backward Compute
int PipelineScheduler::GetBackwardComputeRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Compute);
}

// Forward Send.
int PipelineScheduler::GetForwardSendWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Send);
}

// Forward Send.
int PipelineScheduler::GetForwardSendRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Send);
}

// Backward Send.
int PipelineScheduler::GetBackwardSendWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Send);
}

// Backward Send.
int PipelineScheduler::GetBackwardSendRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Send);
}

// Forward Recv.
int PipelineScheduler::GetForwardRecvWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Recv);
}

// Forward Recv.
int PipelineScheduler::GetForwardRecvRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Forward, PipelineTask::Type::Recv);
}

// Backward Recv.
int PipelineScheduler::GetBackwardRecvWaitedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(true, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Recv);
}

// Backward Recv.
int PipelineScheduler::GetBackwardRecvRecordedEvent(const int batch_id, const int stage_id) const {
  return GetEventOrDefault(false, batch_id, stage_id, PipelineTask::Pass::Backward, PipelineTask::Type::Recv);
}

void PipelineWorkerPool::Join(size_t worker_id) {
  auto& worker = workers.at(worker_id);
  if (!worker.joinable())
    return;
  worker.join();
}

void PipelineWorkerPool::JoinAll() {
  for (size_t i = 0; i < workers.size(); ++i) {
    auto& worker = workers.at(i);
    if (!worker.joinable())
      continue;
    worker.join();
  };
}

}  // namespace pipeline
}  // namespace training
}  // namespace onnxruntime