// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TFModelInfo.h"

#include <memory>

#include <core/platform/env.h>

std::unique_ptr<TestModelInfo> TFModelInfo::Create(_In_ const PATH_CHAR_TYPE* model_url) {
  auto* model_info = new TFModelInfo{};
  std::unique_ptr<TestModelInfo> ret(model_info);

  model_info->model_url_ = model_url;
  std::basic_string<PATH_CHAR_TYPE> meta_file_path = model_url;
  meta_file_path.append(ORT_TSTR(".meta"));
  const onnxruntime::Env& env = onnxruntime::Env::Default();
  size_t len;
  auto status = env.GetFileLength(meta_file_path.c_str(), len);
  if (!status.IsOK()) {
    ORT_THROW(status.ErrorMessage());
  }
  std::string file_content;
  file_content.resize(len);
  auto buffer_span = gsl::make_span(&file_content[0], file_content.size());
  status = onnxruntime::Env::Default().ReadFileIntoBuffer(meta_file_path.c_str(), 0, len, buffer_span);
  if (!status.IsOK()) {
    ORT_THROW(status.ErrorMessage());
  }
  // this string is not null terminated
  std::istringstream is{file_content};

  std::string line;
  while (std::getline(is, line)) {
    size_t line_len = 0;
    if (!line.empty() && line.back() == '\n') {
      line_len = line.length() - 1;
      if (line_len > 0 && line[line_len - 1] == '\r') {
        --line_len;
      }
      line.resize(line_len);
    }
    if (line.empty()) continue;
    if (line.compare(0, 6, "input=") == 0) {
      model_info->input_names_.push_back(line.substr(6));
    } else if (line.compare(0, 7, "output=") == 0) {
      model_info->output_names_.push_back(line.substr(7));
    } else {
      ORT_THROW("unknown line:", line.size());
    }
  }

  return ret;
}

int TFModelInfo::GetInputCount() const { return static_cast<int>(input_names_.size()); }
int TFModelInfo::GetOutputCount() const { return static_cast<int>(output_names_.size()); }
const std::string& TFModelInfo::GetInputName(size_t i) const { return input_names_[i]; }
const std::string& TFModelInfo::GetOutputName(size_t i) const { return output_names_[i]; }
