#include "bits/stdc++.h"

#pragma once

template <typename Item>
class ResponseQueue {
 public:
  ResponseQueue() {}

  bool Empty() {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.empty();
  }

  Item Get() {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      cv_.wait(lk, [this] { return !queue_.empty(); });
    }
    auto res = std::move(queue_.front());
    queue_.pop_front();
    return res;
  }

  void Put(const Item& value) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(value);
    }
    cv_.notify_all();
  }

  void Put(Item&& value) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(std::move(value));
    }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
};
