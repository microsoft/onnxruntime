// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx_model_info.h"
#include "core/platform/env.h"
#include "re2/re2.h"
#include "pb_helper.h"

using namespace onnxruntime;
static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;

template <typename T>
static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info,
                                     std::vector<T>& out) {
  for (int i = 0; i != input_value_info.size(); ++i) {
    out.push_back(input_value_info[i]);
  }
}

OnnxModelInfo::OnnxModelInfo(_In_ const PATH_CHAR_TYPE* model_url) : model_url_(model_url) {
  // parse model
  int model_fd;
  auto st = Env::Default().FileOpenRd(model_url, model_fd);
  if (!st.IsOK()) {
    ORT_THROW(st.ErrorMessage());
  }

  ONNX_NAMESPACE::ModelProto model_pb;
  ::google::protobuf::io::FileInputStream input(model_fd, protobuf_block_size_in_bytes);
  const bool parse_result = model_pb.ParseFromZeroCopyStream(&input) && input.GetErrno() == 0;
  if (!parse_result) {
    (void)Env::Default().FileClose(model_fd);
    ORT_THROW("Failed to load model because protobuf parsing failed.");
  }
  (void)Env::Default().FileClose(model_fd);
  {
    const RE2::Anchor re2_anchor = RE2::UNANCHORED;
    const std::string model_url_string = ToMBString(model_url);
    re2::StringPiece text(model_url_string);
    re2::StringPiece submatch;
    re2::RE2 regex("onnx[0-9a-z]{3}", re2::RE2::Options());  //e.g. onnx141, onnx150, onnxtip
    if (!regex.ok()) {
      ORT_THROW("Failed to parse regex: onnx[0-9a-z]{3}");
    }
    bool match = regex.Match(text, 0, text.length(), re2_anchor, &submatch, 1);
    if (match) {
      onnx_commit_tag_.assign(submatch.data(), submatch.length());
    } else {
      onnx_commit_tag_ = TestModelInfo::unknown_version;
    }
  }
  for (const auto& opset : model_pb.opset_import()) {
    std::string s = opset.domain();
    if (s == "ai.onnx") s = "";
    domain_to_version_[s] = opset.version();
  }
  const ONNX_NAMESPACE::GraphProto& graph = model_pb.graph();
  if (graph.node().size() == 1) {
    node_name_ = graph.node()[0].op_type();
  }
  std::unordered_set<std::string> initializer_names;
  for (const auto& init : graph.initializer()) {
    if (!init.has_name()) continue;
    initializer_names.insert(init.name());
  }
  //Ignore the inputs that are already in initializers
  for (const auto& p : graph.input()) {
    if (!p.has_name()) ORT_THROW("input without name??");
    if (initializer_names.find(p.name()) == initializer_names.end()) input_value_info_.push_back(p);
  }
  RepeatedPtrFieldToVector(graph.output(), output_value_info_);
}
