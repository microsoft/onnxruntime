// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

template <int input_start, int output_start>
std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.push_back(std::pair<int, int>(input_start + i, output_start + i));
  }
  return aliases;
}

}  // namespace contrib
}  // namespace onnxruntime
