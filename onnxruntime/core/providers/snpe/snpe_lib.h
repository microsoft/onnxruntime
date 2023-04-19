// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "DlSystem/DlError.hpp"
#include "core/providers/snpe/snpe_runtime_options.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

struct UserBufferAttribute {
 public:
  UserBufferAttribute(size_t size,
                      const std::vector<size_t>& buffer_strides,
                      zdl::DlSystem::UserBufferEncoding* const buffer_encoding) : buffer_size(size),
                                                                                  strides(buffer_strides),
                                                                                  user_buffer_encoding(buffer_encoding) {}

  size_t buffer_size;
  std::vector<size_t> strides;
  zdl::DlSystem::UserBufferEncoding* user_buffer_encoding;
};

using onnxruntime::common::Status;
class SnpeLib {
 public:
  SnpeLib() : buffer_type_(BufferType::ITENSOR) {}
  ~SnpeLib() {}

  Status SnpeProcess(const unsigned char* input,
                     size_t input_size,
                     unsigned char* output,
                     size_t output_size,
                     const std::unordered_map<std::string, size_t>& output_names_index);
  Status SnpeProcessMultipleOutput(const unsigned char* input,
                                   size_t input_size,
                                   size_t output_number,
                                   unsigned char* outputs[],
                                   size_t output_sizes[],
                                   const std::unordered_map<std::string, size_t>& output_names_index);
  Status SnpeProcessMultiInputsMultiOutputs(const unsigned char** inputs,
                                            const size_t* input_sizes,
                                            size_t input_number,
                                            unsigned char** outputs,
                                            const size_t* output_sizes,
                                            size_t output_number,
                                            const std::unordered_map<std::string, size_t>& output_names_index);
  Status SnpeProcessWithUserBuffer(const std::vector<std::string>& input_names,
                                   const unsigned char** inputs,
                                   size_t input_number,
                                   unsigned char** outputs,
                                   const std::unordered_map<std::string, size_t>& output_names_index);

  Status CheckInputsSize(const std::vector<std::string>& input_tensor_names,
                         const std::vector<int64_t>& input_sizes);

  Status InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                        const std::vector<std::string>& output_tensor_names,
                        const std::vector<std::string>& input_tensor_names,
                        const std::vector<int64_t>& input_sizes,
                        const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());

  Status Initialize(const char* dlcPath,
                    const std::vector<std::string>& output_layer_names,
                    const std::vector<std::string>& input_layer_names,
                    const std::vector<int64_t>& input_sizes,
                    const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());
  Status Initialize(const unsigned char* dlcData,
                    size_t size,
                    const std::vector<std::string>& output_layer_names,
                    const std::vector<std::string>& input_layer_names,
                    const std::vector<int64_t>& input_sizes,
                    const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());

  Status SetupUserBufferAttribute(const std::string& name);
  Status SetupUserBufferAttributes(const std::vector<std::string>& tensor_names);
  Status SetupInputTensors(const std::vector<std::string>& input_tensor_names);

 private:
  const char* GetSnpeErrorString() {
    return zdl::DlSystem::getLastErrorString();
  }

  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors_;
  zdl::DlSystem::TensorMap input_tensor_map_;

  BufferType buffer_type_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_input_buffers_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_output_buffers_;
  std::vector<std::unique_ptr<zdl::DlSystem::UserBufferEncoding>> user_buffer_encoding_;
  std::unordered_map<std::string, UserBufferAttribute> user_buffer_attr_table_;
};

std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data,
                                        size_t size,
                                        const std::unordered_map<std::string, std::string>& options,
                                        const std::vector<std::string>& output_layer_names,
                                        const std::vector<std::string>& input_layer_names,
                                        const std::vector<int64_t>& input_sizes,
                                        BufferType& buffer_type);

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
