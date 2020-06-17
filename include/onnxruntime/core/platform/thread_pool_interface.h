#pragma once

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* Modifications Copyright (c) Microsoft. */
#include <functional>

namespace onnxruntime {

//Copied from Eigen
class ThreadPoolInterface {
 public:
  // Submits a closure to be run by a thread in the pool.
  virtual void Schedule(std::function<void()> fn) = 0;

  // If implemented, stop processing the closures that have been enqueued.
  // Currently running closures may still be processed.
  // If not implemented, does nothing.
  virtual void Cancel() {}

  // Returns the number of threads in the pool.
  virtual int NumThreads() const = 0;

  virtual ~ThreadPoolInterface() {}
};

}