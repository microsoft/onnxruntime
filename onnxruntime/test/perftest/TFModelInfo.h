// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "TestCase.h"
#include <string>
#include <vector>

class TFModelInfo : public TestModelInfo {
 public:
  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_url_.c_str(); }

  const std::string& GetNodeName() const override { return node_name_; }
  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t) const override { return nullptr; }

  int GetInputCount() const override;
  int GetOutputCount() const override;
  const std::string& GetInputName(size_t i) const override;
  const std::string& GetOutputName(size_t i) const override;
  ~TFModelInfo() override = default;

  static std::unique_ptr<TestModelInfo> Create(_In_ const PATH_CHAR_TYPE* model_url);

 private:
  TFModelInfo() = default;
  std::basic_string<PATH_CHAR_TYPE> model_url_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string node_name_;
};
