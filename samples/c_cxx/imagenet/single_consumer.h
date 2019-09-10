// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <functional>

/**
 * A special FIFO that is restricted to have only one consumer
 * The consumer must return the previous borrowed item before taking the next
 */
template <typename ValueType>
class SingleConsumerFIFO {
 public:
  struct ListEntry {
    ValueType value;
    ListEntry* next = nullptr;
  };

 private:
  // fixed size
  ListEntry* values_;
  ListEntry* free_list_ = nullptr;
  // whenever free_list_ is nullptr, free_list_tail_ should equal to &free_list_;
  ListEntry** free_list_tail_ = &free_list_;
  bool is_consumer_running_ = false;
  size_t len_;
#ifndef NDEBUG
  size_t count_ = 0;
#endif
 public:
  explicit SingleConsumerFIFO(size_t len) : values_(new ListEntry[len]), len_(len) {}

  // destruct values earlier
  void Release() {
    delete[] values_;
    values_ = nullptr;
  }
  ~SingleConsumerFIFO() noexcept { delete[] values_; }

  template <typename T>
  void Init(const T& t) {
    for (size_t i = 0; i != len_; ++i) {
      t(values_[i].value);
    }
  }

  /**
   * Return a borrowed item
   * @param e a pointer returned from the Take() function
   * @return ID of the entry, in [0,len)
   */
  size_t Return(ListEntry* e) {
    is_consumer_running_ = false;
    return e - values_;
  }

  template <typename FUNC>
  void Put(size_t element_id, const FUNC& f) {
    assert(element_id < len_);
#ifndef NDEBUG
    ++count_;
#endif

    // printf("Append %zd to the free list\n", element_id);
    ListEntry* t = &values_[element_id];
    t->next = nullptr;
    (*free_list_tail_) = t;
    free_list_tail_ = &t->next;
    f(t->value);
  }

  ListEntry* Take() {
    if (is_consumer_running_) return nullptr;
    if (free_list_ == nullptr) {
      is_consumer_running_ = false;
      assert(count_ == 0);
      return nullptr;
    }
    auto input_tensor = free_list_;
    is_consumer_running_ = true;
    if ((free_list_ = free_list_->next) == nullptr) free_list_tail_ = &free_list_;
#ifndef NDEBUG
    --count_;
    assert(free_list_ != nullptr || count_ == 0);
#endif
    return input_tensor;
  }
};