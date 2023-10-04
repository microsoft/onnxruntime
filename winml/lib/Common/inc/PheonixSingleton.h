// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

template <typename T, typename... TArgs>
std::shared_ptr<T> PheonixSingleton(TArgs&&... args) {
  static std::weak_ptr<T> instance_;
  static std::mutex lock_;

  std::lock_guard<std::mutex> lock(lock_);
  if (auto instance = instance_.lock()) {
    return instance;
  }

  auto instance = std::make_shared<T>(std::forward<TArgs>(args)...);
  instance_ = instance;
  return instance;
}
