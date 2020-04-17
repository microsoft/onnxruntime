// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "gsl/gsl"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
#include "core/framework/ml_value.h"

struct Slot {
  enum Type { Empty, Forward, Backward };

  Slot() {
    type = Empty;
    batch_id = 0;
  }

  bool IsEmpty() const {
    return type == Empty;
  }

  bool IsForward() const {
    return type == Forward;
  }

  bool IsBackward() const {
    return type == Backward;
  }

  void show() const {
    int wid = waited_events.size() > 0 ? waited_events[0] : -1;
    int rid = recorded_events.size() > 0 ? recorded_events[0] : -1;
    switch (type) {
    case Empty:
      std::cout << "         ";
      break;
    case Forward:
      std::cout << " F" << batch_id << "(" << wid << "," << rid << ")" << " ";
      break;
    case Backward:
      std::cout << " B" << batch_id << "(" << wid << "," << rid << ")" << " ";
      break;
    }
  }
  
  Type type;
  int batch_id;
  std::vector<int> waited_events;
  std::vector<int> recorded_events;
};

class PipelineSchedule {
public:
  PipelineSchedule() = default;

  PipelineSchedule(int num_stages) {
    num_stages_ = num_stages;
    num_batches_ = 0;
  }

  void add(int batch_id) {
    ++num_batches_;

    // Expand table to accomonadate the new batch.
    const int required_max_time = 2 * num_stages_ + 2 * (num_batches_ - 1);
    const int current_max_time = table_.size();

    for (int t = current_max_time; t < required_max_time; ++t) {
      table_.push_back(std::vector<Slot>(num_stages_));
      batch_count_.push_back(0);
    }

    std::vector<int> forward_time(num_stages_, 0);

    // Insert forward.
    for (int s = 0; s < num_stages_; ++s) {
      for (int t = 0; t < required_max_time; ++t) {
        if (!table_[t][s].IsEmpty()) {
          continue;
        }

        if (s > 0 && t <= forward_time[s - 1]) {
          continue;
        }

        if (batch_count_[t] >= num_stages_) {
          continue;
        }

        forward_time[s] = t;
        table_[t][s].type = Slot::Type::Forward;
        table_[t][s].batch_id = batch_id;

        auto events = search_last_recorded_events(t, s);
        table_[t][s].waited_events = events;

        if (events.size() != 0) {
          table_[t][s].recorded_events = std::vector<int>{ /* max event id */ events[events.size() - 1] + 1 };
        }
        else {
          table_[t][s].recorded_events = std::vector<int>{ /* first event id */ 0 };
        }

        for (int t_ = t + 1; t_ < required_max_time; ++t_) {
          if (table_[t_][s].IsEmpty())
            continue;
          auto events = search_last_recorded_events(t_, s);
          table_[t_][s].waited_events = events;
          table_[t_][s].recorded_events = std::vector<int>{ /* max event id */ events[events.size() - 1] + 1 };
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

        auto events = search_last_recorded_events(t, s);
        table_[t][s].waited_events = events;

        if (events.size() != 0) {
          table_[t][s].recorded_events = std::vector<int>{ /* max event id */ events[events.size() - 1] + 1 };
        }
        else {
          table_[t][s].recorded_events = std::vector<int>{ /* first event id */ 0 };
        }

        for (int t_ = t + 1; t_ < required_max_time; ++t_) {
          if (table_[t_][s].IsEmpty())
            continue;
          auto events = search_last_recorded_events(t_, s);
          table_[t_][s].waited_events = events;
          table_[t_][s].recorded_events = std::vector<int>{ /* max event id */ events[events.size() - 1] + 1 };
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

  void add(int batch_id_begin, int batch_id_end) {
    for (int i = batch_id_begin; i < batch_id_end; ++i) {
      add(i);
    }
  }

  int get_forward_waited_event_id(int stage_id, int batch_id) const {
    std::vector<int> events = {-1};
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
    // std::cout << "stage=" << stage_id << ", batch_id=" << batch_id << ", fw_wait=" << events[0] << std::endl;
    return events[0];
  }

  int get_forward_recorded_event_id(int stage_id, int batch_id) const {
    std::vector<int> events = {-1};
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
    // std::cout << "stage=" << stage_id << ", batch_id=" << batch_id << ", fw_record=" << events[0] << std::endl;
    return events[0];
  }

  int get_backward_waited_event_id(int stage_id, int batch_id) const {
    std::vector<int> events = {-1};
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
    // std::cout << "stage=" << stage_id << ", batch_id=" << batch_id << ", bw_waited=" << events[0] << std::endl;
    return events[0];
  }

  int get_backward_recorded_event_id(int stage_id, int batch_id) const {
    std::vector<int> events = {-1};
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
    // std::cout << "stage=" << stage_id << ", batch_id=" << batch_id << ", bw_recorded=" << events[0] << std::endl;
    return events[0];
  }

  void show() const {
    const int num_slots = table_.size();

    for (int s = 0; s < num_stages_; ++s) {
      for (int t = 0; t < num_slots; ++t) {
        table_[t][s].show();
        if (t == num_slots - 1) {
          std::cout << std::endl;
        }
      }
    }
  }

private:
  std::vector<int> search_last_recorded_events(int time_id, int stage_id) {
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

  std::vector<std::vector<Slot>> table_;
  std::vector<std::vector<int>> batch_time_;
  std::vector<int> batch_count_;
  int num_stages_;
  int num_batches_;
};

struct PipelineWorkerState {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;
  std::vector<std::string> fetch_names;
  std::vector<MLValue> fetches;
};

struct PipelineWorkerPool {
  PipelineWorkerPool(size_t num_workers) : workers(num_workers), worker_states(num_workers) {};
  std::vector<std::thread> workers;
  std::vector<PipelineWorkerState> worker_states;

  void join(size_t worker_id) {
    auto & worker = workers[worker_id];
    if (!worker.joinable())
      return;
    worker.join();
  }

  void join_all() {
    for(size_t i = 0; i < workers.size(); ++i) {
      auto& worker = workers[i];
      if (!worker.joinable())
        continue;
      worker.join();
    };
  }
};

struct PipelineContext {
  // Total number of pipeline stages. 
  size_t num_pipeline_stages;

  // Id of stage handled by this process. Currently, it matches the MPI's rank.
  size_t pipeline_stage_id;
  // The number of batches per pipeline run. Its value is
  // num_gradient_accumulation_steps - 1
  size_t num_pipeline_batches;
  // We only run pipeline on the first num_gradient_accumulation_steps - 1 batches.
  // The last batch runs optimizer and update the weights. 
  size_t num_gradient_accumulation_steps;

  // Name of scheduling event in graph's input list.
  // If an event name is an empty string, it means no event
  // should be waited or recorded.
  // [TODO] Add events for communication stages and computation stages independently.
  // std::string forward_comm_waited_event_name;
  // std::string forward_comm_recorded_event_name;
  // std::string forward_comp_waited_event_name;
  // std::string forward_comp_recorded_event_name;
  // std::string backward_comm_waited_event_name;
  // std::string backward_comm_recorded_event_name;
  // std::string backward_comp_waited_event_name;
  // std::string backward_comp_recorded_event_name;
  std::string forward_waited_event_name;
  std::string forward_recorded_event_name;
  std::string backward_waited_event_name;
  std::string backward_recorded_event_name;
};