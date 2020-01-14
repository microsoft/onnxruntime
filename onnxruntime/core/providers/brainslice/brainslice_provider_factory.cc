// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/brainslice/brainslice_provider_factory.h"
#include <atomic>
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct BrainSliceExecutionProviderFactory : IExecutionProviderFactory {
  BrainSliceExecutionProviderFactory(uint32_t ip, bool load_firmware, const char* instr_path, const char* data_path, const char* schema_path) 
    : ip_(ip),
    lead_firmware_(load_firmware),
    instr_path_(instr_path),
    data_path_(data_path),
    schema_path_(schema_path){}
  ~BrainSliceExecutionProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  uint32_t ip_;
  bool lead_firmware_;
  std::string instr_path_;
  std::string data_path_;
  std::string schema_path_;
};

std::unique_ptr<IExecutionProvider> BrainSliceExecutionProviderFactory::CreateProvider() {
  fpga::FPGAInfo info = {ip_, lead_firmware_, instr_path_.c_str(), data_path_.c_str(), schema_path_.c_str()};
  return onnxruntime::make_unique<brainslice::BrainSliceExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_BrainSlice(uint32_t ip, bool load_firmware, const char* instr_path, const char* data_path, const char* schema_path) {
  return std::make_shared<onnxruntime::BrainSliceExecutionProviderFactory>(ip, load_firmware, instr_path, data_path, schema_path);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_BrainSlice, _In_ OrtSessionOptions* options, uint32_t ip, bool load_firmware, _In_ const char* instr_path, _In_ const char* data_path, _In_ const char* schema_path) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_BrainSlice(ip, load_firmware, instr_path, data_path, schema_path));
  return nullptr;
}
