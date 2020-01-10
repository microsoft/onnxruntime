// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "winml_adapter_model.h"

#include "core/session/winml_adapter_c_api.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/session/winml_adapter_apis.h"
#include "error_code_helper.h"

#include <io.h>
#include <fcntl.h>
#include "google/protobuf/io/zero_copy_stream_impl.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;
//
//static std::vector<const char*> GetAllNodeOutputs(const onnx::ModelProto& model_proto) {
//  std::vector<const char*> nodes_outputs;
//  auto& graph = model_proto.graph();
//  auto& nodes = graph.node();
//  for (auto& node : nodes) {
//    for (auto& node_output : node.output()) {
//      nodes_outputs.push_back(node_output.c_str());
//    }
//  }
//  return nodes_outputs;
//}

static std::vector<const char*> GetInitializers(const onnx::ModelProto& model_proto) {
  std::vector<const char*> initializers;
  auto& graph = model_proto.graph();
  auto& graph_initializers = graph.initializer();
  for (auto& initializer : graph_initializers) {
    initializers.push_back(initializer.name().c_str());
  }
  return initializers;
}

static std::vector<const onnx::ValueInfoProto*> GetInputsWithoutInitializers(const onnx::ModelProto& model_proto) {
  auto initializers = GetInitializers(model_proto);

  std::vector<const onnx::ValueInfoProto*> inputs_without_initializers;
  auto& graph = model_proto.graph();
  auto& inputs = graph.input();
  for (auto& input : inputs) {
    if (input.has_name() && input.has_type()) {
      auto found_it = std::find_if(
          std::begin(initializers),
          std::end(initializers),
          [&](auto& initializer) {
            return std::strcmp(initializer, input.name().c_str()) == 0;
          });

      auto is_initializer = found_it != std::end(initializers);
      if (!is_initializer) {
        inputs_without_initializers.push_back(&input);
      }
    }
  }
  return inputs_without_initializers;
}

static std::vector<const onnx::ValueInfoProto*> GetOutputs(const onnx::ModelProto& model_proto) {
  std::vector<const onnx::ValueInfoProto*> outputs_with_name;
  auto& graph = model_proto.graph();
  auto& outputs = graph.output();
  for (auto& output : outputs) {
    if (output.has_name() && output.has_type()) {
      outputs_with_name.push_back(&output);
    }
  }
  return outputs_with_name;
}

class ModelInfo {
 public:
  ModelInfo(const onnx::ModelProto* model_proto) {
    Initialize(model_proto);
  }

 public:
  // model metadata
  std::string author_;
  std::string name_;
  std::string domain_;
  std::string description_;
  int64_t version_;
  std::vector<std::pair<std::string, std::string>> model_metadata_;
  //wfc::IVector<winml::ILearningModelFeatureDescriptor> input_features_;
  //wfc::IVector<winml::ILearningModelFeatureDescriptor> output_features_;

 private:
  void Initialize(const onnx::ModelProto* model_proto) {
    // metadata
    for (auto& prop : model_proto->metadata_props()) {
      model_metadata_.push_back(std::make_pair(prop.key(), prop.value()));
    }

    //WinML::FeatureDescriptorFactory builder(model_metadata_);

    // Create inputs
    auto inputs = GetInputsWithoutInitializers(*model_proto);
    //input_features_ = builder.CreateDescriptorsFromValueInfoProtos(inputs);

    // Create outputs
    auto outputs = ::GetOutputs(*model_proto);
    //output_features_ = builder.CreateDescriptorsFromValueInfoProtos(outputs);

    // author
    auto has_producer_name = model_proto->has_producer_name();
    author_ = has_producer_name
                  ? model_proto->producer_name()
                  : "";

    // domain
    auto has_domain = model_proto->has_domain();
    domain_ = has_domain
                  ? model_proto->domain()
                  : "";

    // name
    auto has_graph = model_proto->has_graph();
    auto graph_has_name = model_proto->graph().has_name();
    auto is_name_available = has_graph && graph_has_name;
    name_ = is_name_available
                ? model_proto->graph().name()
                : "";

    // description
    auto has_description = model_proto->has_doc_string();
    description_ = has_description
                       ? model_proto->doc_string()
                       : "";

    // version
    auto has_version = model_proto->has_model_version();
    version_ = has_version
                   ? model_proto->model_version()
                   : 0;
  }
};

OrtModel::OrtModel(std::unique_ptr<onnx::ModelProto>&& model_proto) : model_proto_(std::move(model_proto)),
                                                                      model_info_(std::make_unique<ModelInfo>(model_proto_.get())) {
}

// factory methods for creating an ort model from a path
static OrtStatus* CreateModelProto(const char* path, std::unique_ptr<onnx::ModelProto>& out) {
  int file_descriptor;
  _set_errno(0);  // clear errno
  _sopen_s(
      &file_descriptor,
      path,
      O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
      _SH_DENYWR,
      _S_IREAD | _S_IWRITE);

  errno_t err = 0;
  _get_errno(&err);
  if (err == ENOENT) {
    return OrtApis::CreateStatus(ORT_NO_SUCHFILE, "Model file not found!");
  }

  if (0 > file_descriptor) {
    return OrtApis::CreateStatus(ORT_NO_SUCHFILE, "Model file not found!");
  }

  google::protobuf::io::FileInputStream stream(file_descriptor);
  stream.SetCloseOnDelete(true);

  auto model_proto = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());

  auto parse_succeeded = model_proto->ParseFromZeroCopyStream(&stream);
  if (!parse_succeeded) {
    return OrtApis::CreateStatus(ORT_INVALID_PROTOBUF, "Failed to parse model file!");
  }

  out = std::move(model_proto);

  return S_OK;
}

OrtStatus* OrtModel::CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model) {
  ORT_UNUSED_PARAMETER(len);

  std::unique_ptr<onnx::ModelProto> model_proto;

  if (auto status = CreateModelProto(path, model_proto)) {
    return status;
  }

  *model = new (std::nothrow) OrtModel(std::move(model_proto));
  if (*model == nullptr) {
    return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Engine failed to create a model!");
  }

  return nullptr;
}

OrtStatus* OrtModel::CreateOrtModelFromData(void* data, size_t len, OrtModel** model) {
  auto model_proto = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());
 
  auto parse_succeeded = model_proto->ParseFromArray(data, static_cast<int>(len));
  if (!parse_succeeded) {
    return OrtApis::CreateStatus(ORT_INVALID_PROTOBUF, "Failed to parse model stream!");
  }

  *model = new (std::nothrow) OrtModel(std::move(model_proto));
  if (*model == nullptr) {
    return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Engine failed to create a model!");
  }

  return nullptr;
}

const ModelInfo* OrtModel::UseModelInfo() const {
  return model_info_.get();
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromPath, const char* model_path, size_t size, OrtModel** out) {
  if (auto status = OrtModel::CreateOrtModelFromPath(model_path, size, out)) {
    return status;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromData, void* data, size_t size, OrtModel** out) {
  if (auto status = OrtModel::CreateOrtModelFromData(data, size, out)) {
    return status;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetAuthor, const OrtModel* model, const char** const author, size_t* len) {
  *author = model->UseModelInfo()->author_.c_str();
  *len = model->UseModelInfo()->author_.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetName, const OrtModel* model, const char** const name, size_t* len) {
  *name = model->UseModelInfo()->name_.c_str();
  *len = model->UseModelInfo()->name_.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetDomain, const OrtModel* model, const char** const domain, size_t* len) {
  *domain = model->UseModelInfo()->domain_.c_str();
  *len = model->UseModelInfo()->domain_.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetDescription, const OrtModel* model, const char** const description, size_t* len) {
  *description = model->UseModelInfo()->description_.c_str();
  *len = model->UseModelInfo()->description_.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetVersion, const OrtModel* model, int64_t* version) {
  *version = model->UseModelInfo()->version_;
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetMetadataCount, const OrtModel* model, size_t* count) {
  *count = model->UseModelInfo()->model_metadata_.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ModelGetMetadata, const OrtModel* model, size_t count, const char** const key,
                    size_t* key_len, const char** const value, size_t* value_len) {
  *key = model->UseModelInfo()->model_metadata_[count].first.c_str();
  *key_len = model->UseModelInfo()->model_metadata_[count].first.size();
  *value = model->UseModelInfo()->model_metadata_[count].second.c_str();
  *value_len = model->UseModelInfo()->model_metadata_[count].second.size();
  return nullptr;
}


ORT_API(void, winmla::ReleaseModel, OrtModel* ptr) {
  delete ptr;
}