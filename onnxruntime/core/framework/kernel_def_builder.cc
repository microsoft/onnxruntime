// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_def_builder.h"
#include <unordered_set>
#include <string>

namespace onnxruntime {
namespace {
//assume start1 <= end1, start2 <= end2
inline bool AreIntervalsOverlap(int start1, int end1, int start2, int end2) {
  return start1 <= end2 && start2 <= end1;
}

template <typename T>
inline bool AreVectorsOverlap(const std::vector<T>& v1, const std::vector<T>& v2) {
  for (T type : v1) {
    if (std::find(v2.begin(), v2.end(), type) != v2.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace
bool KernelDef::IsConflict(const KernelDef& other) const {
  if (op_name_ != other.OpName() || provider_type_ != other.Provider())
    return false;
  int start = 0, end = 0;
  other.SinceVersion(&start, &end);
  if (!AreIntervalsOverlap(op_since_version_start_, op_since_version_end_, start, end))
    return false;
  //only one case they don't conflict:
  //There is a type_constraint, it exists in both hands, but they don't overlap
  //check types
  auto other_types = other.TypeConstraints();
  bool type_has_conflict = true;
  for (const auto& it : type_constraints_) {
    auto iter = other_types.find(it.first);
    if (iter != other_types.end()) {
      if (!AreVectorsOverlap(it.second, iter->second)) {
        type_has_conflict = false;
        break;
      }
    }
  }
  if (!type_has_conflict)
    return false;
  //if has type conflict, check if any other field has different
  //for example, we register two kernel with float type, but one is inplace, another is not.
  //check in-place
  if (inplace_map_.empty() && !other.MayInplace().empty())
    return false;
  for (auto& it : inplace_map_) {
    if (std::find(other.MayInplace().begin(), other.MayInplace().end(), it) == other.MayInplace().end())
      return false;
  }

  //check alias
  for (auto& it : alias_map_) {
    if (std::find(other.Alias().begin(), other.Alias().end(), it) == other.Alias().end())
      return false;
  }
  if (alias_map_.empty() && !other.Alias().empty())
    return false;

  //check memory type
  auto other_input_mem_types = other.InputMemoryType();
  for (auto it : input_memory_type_args_) {
    if (other_input_mem_types.count(it.first) && other_input_mem_types[it.first] == it.second)
      return false;
  }
  if (input_memory_type_args_.empty() && !other.InputMemoryType().empty())
    return false;

  auto other_output_mem_types = other.OutputMemoryType();
  for (auto it : output_memory_type_args_) {
    if (other_output_mem_types.count(it.first) && other_output_mem_types[it.first] == it.second)
      return false;
  }
  return !(output_memory_type_args_.empty() && !other.OutputMemoryType().empty());
}
}  // namespace onnxruntime
