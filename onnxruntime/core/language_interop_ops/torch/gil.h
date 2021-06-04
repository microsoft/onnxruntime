// copyright (c) microsoft corporation. all rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_TORCH_INTEROP

#pragma once
#include "core/language_interop_ops/torch/python_common.h"

// Holder of GIL
// (Global Interpreter Lock, https://wiki.python.org/moin/GlobalInterpreterLock)
// state. It automatically acquire the state upon creation and release the
// acquired state after being destroyed.
// This class is a standard design pattern for running Python function from
// non-Python-created threads.
// See https://docs.python.org/3/c-api/init.html#non-python-created-threads for details.
class GilGuard {
 public:
  GilGuard() : state_(PyGILState_Ensure()){};
  ~GilGuard() { PyGILState_Release(state_); };

 private:
  PyGILState_STATE state_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GilGuard);
};

#endif
