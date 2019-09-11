// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TFModelInfo.h"
#include <core/platform/env.h>

TestModelInfo* TFModelInfo::Create(_In_ const PATH_CHAR_TYPE* model_url) {
  TFModelInfo* ret = new TFModelInfo();
  ret->model_url_ = model_url;
  std::basic_string<PATH_CHAR_TYPE> meta_file_path = model_url;
  meta_file_path.append(ORT_TSTR(".meta"));
  void* p = nullptr;
  size_t len = 0;
  onnxruntime::OrtCallback b;
  auto st = onnxruntime::Env::Default().ReadFileAsString(meta_file_path.c_str(), 0, p, len, b);
  if (!st.IsOK()) {
    ORT_THROW(st.ErrorMessage());
  }
  // this string is not null terminated
  std::string filecontent(reinterpret_cast<char*>(p), len);
  std::istringstream is(filecontent);

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
      ret->input_names_.push_back(line.substr(6));
    } else if (line.compare(0, 7, "output=") == 0) {
      ret->output_names_.push_back(line.substr(7));
    } else {
      ORT_THROW("unknow line:", line.size());
    }
  }

  if (b.f) b.f(b.param);

  return ret;
}

int TFModelInfo::GetInputCount() const { return static_cast<int>(input_names_.size()); }
int TFModelInfo::GetOutputCount() const { return static_cast<int>(output_names_.size()); }
const std::string& TFModelInfo::GetInputName(size_t i) const { return input_names_[i]; }
const std::string& TFModelInfo::GetOutputName(size_t i) const { return output_names_[i]; }
