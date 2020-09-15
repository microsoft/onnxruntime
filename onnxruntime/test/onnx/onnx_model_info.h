// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "TestCase.h"

// This is a temporary solution to enable onnx_test_runner for ort minimal build and ort file format
// It is tracked by Product Backlog Item 895235: Re-work onnx_test_runner/perf_test for ort/onnx model
class BaseModelInfo : public TestModelInfo {
 protected:
  std::string node_name_;
  std::string onnx_commit_tag_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> input_value_info_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> output_value_info_;
  std::unordered_map<std::string, int64_t> domain_to_version_;
  const std::basic_string<PATH_CHAR_TYPE> model_url_;

 public:
  BaseModelInfo(_In_ const PATH_CHAR_TYPE* model_url) : model_url_(model_url) {}
  bool HasDomain(const std::string& name) const {
    return domain_to_version_.find(name) != domain_to_version_.end();
  }

  int64_t GetONNXOpSetVersion() const {
    auto iter = domain_to_version_.find("");
    return iter == domain_to_version_.end() ? -1 : iter->second;
  }

  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_url_.c_str(); }
  std::string GetModelVersion() const override { return onnx_commit_tag_; }

  const std::string& GetNodeName() const override { return node_name_; }
  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return &output_value_info_[i];
  }
  int GetInputCount() const override { return static_cast<int>(input_value_info_.size()); }
  int GetOutputCount() const override { return static_cast<int>(output_value_info_.size()); }
  const std::string& GetInputName(size_t i) const override { return input_value_info_[i].name(); }
  const std::string& GetOutputName(size_t i) const override { return output_value_info_[i].name(); }
};

#if !defined(ORT_MINIMAL_BUILD)
class OnnxModelInfo : public BaseModelInfo {
 public:
  OnnxModelInfo(_In_ const PATH_CHAR_TYPE* model_url);
};
#else
class OrtModelInfo : public BaseModelInfo {
 public:
  OrtModelInfo(_In_ const PATH_CHAR_TYPE* model_url);
};
#endif
