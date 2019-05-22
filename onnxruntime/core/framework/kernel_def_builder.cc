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
  int start = 0;
  int end = 0;
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
  auto& other_input_mem_types = other.input_memory_type_args_;
  for (auto it : input_memory_type_args_) {
    if (other_input_mem_types.count(it.first) && other_input_mem_types.find(it.first)->second == it.second)
      return false;
  }
  if (input_memory_type_args_.empty() && !other.input_memory_type_args_.empty())
    return false;

  auto& other_output_mem_types = other.output_memory_type_args_;
  for (auto it : output_memory_type_args_) {
    if (other_output_mem_types.count(it.first) && other_output_mem_types.find(it.second)->second == it.second)
      return false;
  }
  return !(output_memory_type_args_.empty() && !other.output_memory_type_args_.empty());
}

KernelDefBuilder& KernelDefBuilder::SetName(const std::string& op_name) {
  kernel_def_->op_name_ = op_name;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetName(const char* op_name) {
  kernel_def_->op_name_ = std::string(op_name);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(const std::string& domain) {
  kernel_def_->op_domain_ = domain;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(const char* domain) {
  kernel_def_->op_domain_ = std::string(domain);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Provider(onnxruntime::ProviderType provider_type) {
  kernel_def_->provider_type_ = provider_type;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Provider(const char* provider_type) {
  kernel_def_->provider_type_ = std::string(provider_type);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const std::string& arg_name,
                                                   const std::vector<MLDataType>& supported_types) {
  kernel_def_->type_constraints_[arg_name] = supported_types;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name,
                                                   const std::vector<MLDataType>& supported_types) {
  return TypeConstraint(std::string(arg_name), supported_types);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const std::string& arg_name,
                                                   MLDataType supported_type) {
  kernel_def_->type_constraints_[arg_name] = std::vector<MLDataType>{supported_type};
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name,
                                                   MLDataType supported_type) {
  return TypeConstraint(std::string(arg_name), supported_type);
}

KernelDefBuilder& KernelDefBuilder::MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
  kernel_def_->inplace_map_ = inplaces;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::MayInplace(int input_index, int output_index) {
  // TODO: validate inputs.
  kernel_def_->inplace_map_.emplace_back(input_index, output_index);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Alias(const std::vector<std::pair<int, int>>& aliases) {
  kernel_def_->alias_map_ = aliases;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Alias(int input_index, int output_index) {
  kernel_def_->alias_map_.emplace_back(input_index, output_index);
  return *this;
}

}  // namespace onnxruntime
