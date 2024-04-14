// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx_model_info.h"

#include <fstream>

#include "pb_helper.h"
#include "re2/re2.h"

#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/platform/env.h"

using namespace onnxruntime;

OnnxModelInfo::OnnxModelInfo(const std::filesystem::path& model_url, bool is_ort_model)
    : model_url_(model_url) {
  if (is_ort_model) {
    InitOrtModelInfo(model_url);
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    InitOnnxModelInfo(model_url);
#else
    ORT_THROW("onnx model is not supported in this build");
#endif
  }
}

#if !defined(ORT_MINIMAL_BUILD)

static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;
template <typename T>
static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info,
                                     std::vector<T>& out) {
  for (int i = 0; i != input_value_info.size(); ++i) {
    out.push_back(input_value_info[i]);
  }
}

void OnnxModelInfo::InitOnnxModelInfo(const std::filesystem::path& model_url) {  // parse model
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
    std::ostringstream oss;
    oss << "Failed to load model from " << model_url << " because protobuf parsing failed.";
    ORT_THROW(oss.str());
  }
  (void)Env::Default().FileClose(model_fd);
  {
    const RE2::Anchor re2_anchor = RE2::UNANCHORED;
    const std::string model_url_string = ToUTF8String(model_url);
    re2::StringPiece text(model_url_string);
    re2::StringPiece submatch;
    re2::RE2 regex_op("opset[0-9a-z]{1,2}", re2::RE2::Options());  // e.g. opset14, opset15

    bool match = regex_op.Match(text, 0, text.length(), re2_anchor, &submatch, 1);
    if (match) {
      onnx_nominal_opset_vesion_.assign(submatch.data(), submatch.length());
    } else {
      onnx_nominal_opset_vesion_ = TestModelInfo::unknown_version;
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
  // Ignore the inputs that are already in initializers
  for (const auto& p : graph.input()) {
    if (!p.has_name()) ORT_THROW("input without name??");
    if (initializer_names.find(p.name()) == initializer_names.end()) input_value_info_.push_back(p);
  }
  RepeatedPtrFieldToVector(graph.output(), output_value_info_);
}

#endif  // #if !defined(ORT_MINIMAL_BUILD)

void OnnxModelInfo::InitOrtModelInfo(const std::filesystem::path& model_url) {
  std::vector<uint8_t> bytes;
  size_t num_bytes = 0;
  const auto model_location = ToWideString(model_url);
  ORT_THROW_IF_ERROR(Env::Default().GetFileLength(model_location.c_str(), num_bytes));
  bytes.resize(num_bytes);
  std::ifstream bytes_stream(model_location, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(bytes.data()), num_bytes);

  // TODO, verify it is a valid ort format
  // TODO, version matches the ORT version
  const auto* fbs_session = fbs::GetInferenceSession(bytes.data());
  if (nullptr == fbs_session)
    ORT_THROW("InferenceSession is null. Invalid ORT format model.");

  const auto* fbs_model = fbs_session->model();
  if (nullptr == fbs_model)
    ORT_THROW("Missing Model. Invalid ORT format model.");

  const auto* fbs_graph = fbs_model->graph();
  if (nullptr == fbs_graph)
    ORT_THROW("Missing Graph. Invalid ORT format model.");

  std::unordered_map<std::string, int> _opset_import;
  ORT_THROW_IF_ERROR(fbs::utils::LoadOpsetImportOrtFormat(fbs_model->opset_import(), _opset_import));
  for (const auto& entry : _opset_import)
    domain_to_version_[entry.first] = entry.second;

  // Load all node args from fbs_graph
  std::unordered_map<std::string, ONNX_NAMESPACE::ValueInfoProto> _node_args;
  auto fbs_node_args = fbs_graph->node_args();
  if (fbs_node_args) {
    _node_args.reserve(fbs_node_args->size());
    for (const auto* fbs_value_info : *fbs_node_args) {
      if (nullptr == fbs_value_info)
        ORT_THROW("NodeArg is missing. Invalid ORT format model.");
      ONNX_NAMESPACE::ValueInfoProto node_arg_info;
      ORT_THROW_IF_ERROR(fbs::utils::LoadValueInfoOrtFormat(*fbs_value_info, node_arg_info));
      // NodeArg ctor is private, cannot use make_unique
      _node_args[fbs_value_info->name()->str()] = std::move(node_arg_info);
    }
  }

  if (fbs_graph->nodes() && fbs_graph->nodes()->size() == 1) {
    const auto* node = fbs_graph->nodes()->Get(0);
    if (!node)
      ORT_THROW("Missing Node. Invalid ORT format model.");

    if (!node->op_type())
      ORT_THROW("Missing op_type. Invalid ORT format model.");

    node_name_ = node->op_type()->str();
  }
  // Load input and output node args
  auto add_node_args = [&](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* fbs_node_args,
                           std::vector<ONNX_NAMESPACE::ValueInfoProto>& node_args) -> Status {
    if (fbs_node_args != nullptr) {
      node_args.reserve(fbs_node_args->size());
      for (const auto* fbs_node_arg_name : *fbs_node_args) {
        if (!fbs_node_arg_name)
          ORT_THROW("NodeArg Name is missing. Invalid ORT format model.");
        node_args.push_back(_node_args.at(fbs_node_arg_name->str()));
      }
    }
    return Status::OK();
  };

  ORT_THROW_IF_ERROR(add_node_args(fbs_graph->inputs(), input_value_info_));
  ORT_THROW_IF_ERROR(add_node_args(fbs_graph->outputs(), output_value_info_));
}
