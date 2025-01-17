// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "TestCase.h"

class OnnxModelInfo : public TestModelInfo {
 private:
  std::string node_name_;
  // Due to performance, the opset version is get from directory name, so it's nominal
  std::string onnx_nominal_opset_vesion_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> input_value_info_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> output_value_info_;
  std::unordered_map<std::string, int64_t> domain_to_version_;
  const std::filesystem::path model_url_;

#if !defined(ORT_MINIMAL_BUILD)
  void InitOnnxModelInfo(const std::filesystem::path& model_url);
#endif

  void InitOrtModelInfo(const std::filesystem::path& model_url);

 public:
  OnnxModelInfo(const std::filesystem::path& path, bool is_ort_model = false);
  bool HasDomain(const std::string& name) const {
    return domain_to_version_.find(name) != domain_to_version_.end();
  }

  int64_t GetONNXOpSetVersion() const {
    auto iter = domain_to_version_.find("");
    return iter == domain_to_version_.end() ? -1 : iter->second;
  }

  const std::filesystem::path& GetModelUrl() const override { return model_url_; }
  std::string GetNominalOpsetVersion() const override { return onnx_nominal_opset_vesion_; }

  const std::string& GetNodeName() const override { return node_name_; }

  const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const override {
    return &input_value_info_[i];
  }

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return &output_value_info_[i];
  }

  int GetInputCount() const override { return static_cast<int>(input_value_info_.size()); }
  int GetOutputCount() const override { return static_cast<int>(output_value_info_.size()); }
  const std::string& GetInputName(size_t i) const override { return input_value_info_[i].name(); }
  const std::string& GetOutputName(size_t i) const override { return output_value_info_[i].name(); }
};
