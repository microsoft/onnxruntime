#pragma once
#include <vector>

namespace onnxruntime {

struct ProgramRegion {
  size_t start_pc;
  size_t end_pc;

  std::vector<std::pair<size_t, size_t> > stream_pc_range;
};

}  // namespace onnxruntime
