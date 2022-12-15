//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {

struct ProgramRegion {
  size_t start_pc;
  size_t end_pc;

  InlinedVector<std::pair<size_t, size_t> > stream_pc_range;
};

}  // namespace onnxruntime
