// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/rocm/backward_guard.h"

#include <iostream>
#include <thread>
#define PRE __FILE__ << ":" << __LINE__ << ":" << std::this_thread::get_id() << " "

namespace onnxruntime {

thread_local bool BackwardPassGuard::is_backward_pass_;

BackwardPassGuard::BackwardPassGuard() {
  //std::cerr << PRE << "BackwardPassGuard ctor pre " << is_backward_pass_ << std::endl;
  is_backward_pass_ = true;
  //std::cerr << PRE << "BackwardPassGuard ctor post " << is_backward_pass_ << std::endl;
}

BackwardPassGuard::~BackwardPassGuard() {
  //std::cerr << PRE << "BackwardPassGuard dtor pre " << is_backward_pass_ << std::endl;
  is_backward_pass_ = false;
  //std::cerr << PRE << "BackwardPassGuard dtor post " << is_backward_pass_ << std::endl;
}

bool BackwardPassGuard::is_backward_pass() {
  //std::cerr << PRE << "is_backward_pass_ " << is_backward_pass_ << std::endl;
  return is_backward_pass_;
}

}  // namespace onnxruntime
