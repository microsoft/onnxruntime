#pragma once
#include <vector>

namespace onnxruntime {

struct ProgramRegion {
  size_t node_start;
  size_t node_end;

  std::vector<std::pair<size_t, size_t> > stream_pc_range;
};

}