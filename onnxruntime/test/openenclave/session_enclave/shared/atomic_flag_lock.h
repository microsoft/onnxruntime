// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Copied from https://github.com/openenclave/openenclave/blob/master/tests/libcxx/host/threadArgs.h.

#pragma once

#include <atomic>
#include <mutex>

const uint64_t MAX_ENC_KEYS = 16;

class atomic_flag_lock {
 public:
  void lock() {
    while (_flag.test_and_set()) {
      continue;
    }
  }
  void unlock() {
    _flag.clear();
  }

 private:
  std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

using atomic_lock = std::unique_lock<atomic_flag_lock>;
