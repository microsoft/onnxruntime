// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits.h>

#include "core/common/common.h"
#include "core/graph/basic_types.h"
#include "core/framework/data_types.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class KernelDefBuilder;

typedef std::map<size_t, OrtMemType> MemTypeMap;

class KernelDef {
 private:
  // note that input/output might be on CPU implicitly when the node is from CPU execution provider
  static inline bool MemTypeOnCpuExplicitly(OrtMemType mem_type) {
    return mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput;
  }

 public:
  explicit KernelDef() = default;

  const std::string& OpName() const {
    return op_name_;
  }

  const std::string& Domain() const {
    return op_domain_;
  }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const {
    *start = op_since_version_start_;
    *end = op_since_version_end_;
  }

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  const std::pair<int, int> SinceVersion() const {
    return std::pair<int, int>(op_since_version_start_, op_since_version_end_);
  }
#endif

  onnxruntime::ProviderType Provider() const {
    return provider_type_;
  }

  const std::unordered_map<std::string, std::vector<MLDataType>>& TypeConstraints() const {
    return type_constraints_;
  }

  const std::vector<std::pair<int, int>>& MayInplace() const {
    return inplace_map_;
  }

  const std::vector<std::pair<int, int>>& Alias() const {
    return alias_map_;
  }

  OrtMemType InputMemoryType(size_t input_index) const {
    auto it = input_memory_type_args_.find(input_index);
    if (it == input_memory_type_args_.end())
      return default_inputs_mem_type_;
    return it->second;
  }

  bool IsInputOnCpu(size_t input_index) const { return MemTypeOnCpuExplicitly(InputMemoryType(input_index)); }

  bool IsOutputOnCpu(size_t output_index) const { return MemTypeOnCpuExplicitly(OutputMemoryType(output_index)); }

  OrtMemType OutputMemoryType(size_t output_index) const {
    auto it = output_memory_type_args_.find(output_index);
    if (it == output_memory_type_args_.end())
      return default_outputs_mem_type_;
    return it->second;
  }

  int ExecQueueId() const {
    return exec_queue_id_;
  }

  bool IsConflict(const KernelDef& other) const;

  uint64_t GetHash() const noexcept {
    // if we need to support different hash versions we can update CalculateHash to take a version number
    // and calculate any non-default versions dynamically. we only use this during kernel lookup so
    // it's not performance critical
    return hash_;
  }

 private:
  friend class KernelDefBuilder;

  // call once the KernelDef has been built
  void CalculateHash() {
    // use name, start/end, domain, provider and the type constraints.
    // we wouldn't have two kernels that only differed by the inplace or alias info or memory types.
    // currently nothing sets exec_queue_id either (and would assumably be a runtime thing and not part of the base
    // kernel definition)
    hash_ = 0;  // reset in case this is called multiple times
    HashCombine(hash_, op_name_);
    HashCombine(hash_, op_since_version_start_);
    // If we include op_since_version_end_ the hash of an existing op changes when it's superseded.
    // e.g. Unsqueeze 11 had no end version until Unsqueeze 13, at which point the existing op is changed to have
    // an end version of 12. That would result in a new ORT build having a different hash for Unsqueeze 11 and a
    // previously serialized ORT format model wouldn't find the kernel. In order to select the kernel to include
    // in the ORT model the full OpSchema info is used, so it's safe to exclude op_since_version_end_ from the hash.
    // HashCombine(hash_, op_since_version_end_);
    HashCombine(hash_, op_domain_);
    HashCombine(hash_, provider_type_);
    for (const auto& key_value : type_constraints_) {
      HashCombine(hash_, key_value.first);
      for (const auto& data_type : key_value.second) {
        // need to construct a std::string so it doesn't hash the address of a const char*
        HashCombine(hash_, std::string(DataTypeImpl::ToString(data_type)));
      }
    }
  }

  // The operator name supported by <*this> kernel..
  std::string op_name_;

  // The operator since_version range supported by <*this> kernel.
  // A kernel could support an operator definition between <op_since_version_start>
  // and <op_since_version_end> (inclusive).
  int op_since_version_start_ = 1;
  int op_since_version_end_ = INT_MAX;

  // The operator domain supported by <*this> kernel.
  // Default to 'onnxruntime::kOnnxDomain'.
  // Please note the behavior of std::string("") and std::string() are different
  std::string op_domain_;

  // The type of the execution provider.
  std::string provider_type_;

  // The supported data types for inputs/outputs.
  // Key is input/output name defined in op schema, Value are supported types.
  std::unordered_map<std::string, std::vector<MLDataType>> type_constraints_;

  // An element <i, j> means that output j reuses the memory of input i.
  std::vector<std::pair<int, int>> inplace_map_;

  // An element <i, j> means that output j is an alias of input i.
  std::vector<std::pair<int, int>> alias_map_;

  // The memory types of inputs/outputs of this kernel
  MemTypeMap input_memory_type_args_;
  MemTypeMap output_memory_type_args_;

  // execution command queue id, 0 for default queue in execution provider
  int exec_queue_id_ = 0;
  // Default memory type for all inputs
  OrtMemType default_inputs_mem_type_{OrtMemTypeDefault};
  // Default memory type for all outputs
  OrtMemType default_outputs_mem_type_{OrtMemTypeDefault};

  // hash of kernel definition for lookup in minimal build
  uint64_t hash_ = 0;
};

class KernelDefBuilder {
 public:
  explicit KernelDefBuilder()
      : kernel_def_(new KernelDef()) {}

  KernelDefBuilder& SetName(const std::string& op_name);
  KernelDefBuilder& SetName(const char* op_name);

  KernelDefBuilder& SetDomain(const std::string& domain);
  KernelDefBuilder& SetDomain(const char* domain);

  /**
     This kernel supports operator definition since <since_version> (to latest).
  */
  KernelDefBuilder& SinceVersion(int since_version) {
    kernel_def_->op_since_version_start_ = since_version;
    return *this;
  }

  /**
     The start and end version should be set accordingly per version range for
     each domain registered in OpSchemaRegistry::DomainToVersionRange in
     \onnxruntime\onnxruntime\core\graph\op.h as below.
     Key: domain. Value: <lowest version, highest version> pair.
     std::unordered_map<std::string, std::pair<int, int>> map_;
  */
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    kernel_def_->op_since_version_start_ = since_version_start;
    kernel_def_->op_since_version_end_ = since_version_end;
    return *this;
  }

  /**
     The execution provider type of the kernel.
  */
  KernelDefBuilder& Provider(onnxruntime::ProviderType provider_type);
  KernelDefBuilder& Provider(const char* provider_type);

  /**
     Specify the set of types that this kernel supports. A further restriction
     of the set of types specified in the op schema.
     The arg name could be either op formal parameter name, say "X", or type
     argument name specified in op schema, say "T".
  */
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   const std::vector<MLDataType>& supported_types);
  KernelDefBuilder& TypeConstraint(const char* arg_name,
                                   const std::vector<MLDataType>& supported_types);

  /**
     Like TypeConstraint but supports just a single type.
  */
  KernelDefBuilder& TypeConstraint(const std::string& arg_name, MLDataType supported_type);
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type);

  /**
     Inplace mapping from inputs to outputs allowed.
     It means that uplayer runtime could do memory in-place optimization
     as it will not impact the correctness of this kernel.
  */
  KernelDefBuilder& MayInplace(const std::vector<std::pair<int, int>>& inplaces);
  KernelDefBuilder& MayInplace(int input_index, int output_index);

  /**
     Alias mapping from inputs to outputs. Different from Inplace that the
     content of the tensor is not changed. This is to take care of operators
     such as Identity and Reshape.
  */
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases);
  KernelDefBuilder& Alias(int input_index, int output_index);

  /**
     Specify that this kernel requires an input arg
     in certain memory type (instead of the default, device memory).
  */
  template <OrtMemType T>
  KernelDefBuilder& InputMemoryType(int input_index) {
    kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, T));
    return *this;
  }

  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, type));
    return *this;
  }

  /**
     Specify that this kernel requires input arguments
     in certain memory type (instead of the default, device memory).
  */
  template <OrtMemType T>
  KernelDefBuilder& InputMemoryType(const std::vector<int>& input_indexes) {
    for (auto input_index : input_indexes) {
      kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, T));
    }
    return *this;
  }

  /**
     Specify that this kernel provides an output arg
     in certain memory type (instead of the default, device memory).
  */
  template <OrtMemType T>
  KernelDefBuilder& OutputMemoryType(int output_index) {
    kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, T));
    return *this;
  }

  KernelDefBuilder& OutputMemoryType(OrtMemType type, int output_index) {
    kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, type));
    return *this;
  }

  /**
     Specify that this kernel provides an output arguments
     in certain memory type (instead of the default, device memory).
  */
  template <OrtMemType T>
  KernelDefBuilder& OutputMemoryType(const std::vector<int>& output_indexes) {
    for (auto output_index : output_indexes) {
      kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, T));
    }
    return *this;
  }

  /**
     Specify that this kernel runs on which execution queue in the provider
  */
  KernelDefBuilder& ExecQueueId(int queue_id) {
    kernel_def_->exec_queue_id_ = queue_id;
    return *this;
  }

  /**
  Specify the default inputs memory type, if not specified, it is DefaultMemory
  */
  KernelDefBuilder& SetDefaultInputsMemoryType(OrtMemType mem_type) {
    kernel_def_->default_inputs_mem_type_ = mem_type;
    return *this;
  }

  /**
  Specify the default outputs memory type, if not specified, it is DefaultMemory
  */
  KernelDefBuilder& SetDefaultOutputMemoryType(OrtMemType mem_type) {
    kernel_def_->default_outputs_mem_type_ = mem_type;
    return *this;
  }

  /**
     Return the kernel definition, passing ownership of the KernelDef to the caller
  */
  std::unique_ptr<KernelDef> Build() {
    kernel_def_->CalculateHash();
    return std::move(kernel_def_);
  }

 private:
  // we own the KernelDef until Build() is called.
  std::unique_ptr<KernelDef> kernel_def_;
};

}  // namespace onnxruntime
