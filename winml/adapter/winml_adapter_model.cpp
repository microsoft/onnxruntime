// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "winml_adapter_model.h"

#include "winml_adapter_c_api.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/common/common.h"

#include <io.h>
#include <fcntl.h>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "core/framework/onnxruntime_typeinfo.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

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
  std::vector<const onnx::ValueInfoProto*> input_features_;
  std::vector<const onnx::ValueInfoProto*> output_features_;
  bool requires_float16_support_;

 private:
  void Initialize(const onnx::ModelProto* model_proto) {
    for (auto& prop : model_proto->metadata_props()) {
      model_metadata_.push_back(std::make_pair(prop.key(), prop.value()));
    }

    input_features_ = GetInputsWithoutInitializers(*model_proto);
    output_features_ = ::GetOutputs(*model_proto);

    auto has_producer_name = model_proto->has_producer_name();
    author_ = has_producer_name ? model_proto->producer_name() : "";

    auto has_domain = model_proto->has_domain();
    domain_ = has_domain ? model_proto->domain() : "";

    auto has_graph = model_proto->has_graph();
    auto graph_has_name = model_proto->graph().has_name();
    auto is_name_available = has_graph && graph_has_name;
    name_ = is_name_available ? model_proto->graph().name() : "";

    auto has_description = model_proto->has_doc_string();
    description_ = has_description ? model_proto->doc_string() : "";

    auto has_version = model_proto->has_model_version();
    version_ = has_version ? model_proto->model_version() : 0;
  }
};

OrtModel::OrtModel(std::unique_ptr<onnx::ModelProto> model_proto) : model_proto_(std::move(model_proto)),
                                                                    model_info_(std::make_unique<ModelInfo>(model_proto_.get())) {
}

// factory methods for creating an ort model from a path
static OrtStatus* CreateModelProto(const char* path, std::unique_ptr<onnx::ModelProto>& out) {
  int file_descriptor;

  auto path_str = std::string(path);
  auto wide_path = onnxruntime::ToWideString(path_str);
  
  _set_errno(0);  // clear errno
  _wsopen_s(
      &file_descriptor,
      wide_path.c_str(),
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

  return OrtModel::CreateOrtModelFromProto(std::move(model_proto), model);
}

OrtStatus* OrtModel::CreateOrtModelFromData(void* data, size_t len, OrtModel** model) {
  auto model_proto = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());

  auto parse_succeeded = model_proto->ParseFromArray(data, static_cast<int>(len));
  if (!parse_succeeded) {
    return OrtApis::CreateStatus(ORT_INVALID_PROTOBUF, "Failed to parse model stream!");
  }

  return OrtModel::CreateOrtModelFromProto(std::move(model_proto), model);
}

OrtStatus* OrtModel::CreateOrtModelFromProto(std::unique_ptr<onnx::ModelProto>&& model_proto, OrtModel** model) {
  *model = new (std::nothrow) OrtModel(std::move(model_proto));
  if (*model == nullptr) {
    return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Engine failed to create a model!");
  }

  return nullptr;
}

const ModelInfo* OrtModel::UseModelInfo() const {
  return model_info_.get();
}

const ONNX_NAMESPACE::ModelProto* OrtModel::UseModelProto() const {
  return model_proto_.get();
}

std::unique_ptr<onnx::ModelProto> OrtModel::DetachModelProto() {
  return std::move(model_proto_);
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromPath, const char* model_path, size_t size, OrtModel** out) {
  API_IMPL_BEGIN
  if (auto status = OrtModel::CreateOrtModelFromPath(model_path, size, out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromData, void* data, size_t size, OrtModel** out) {
  API_IMPL_BEGIN
  if (auto status = OrtModel::CreateOrtModelFromData(data, size, out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CloneModel, const OrtModel* in, OrtModel** out) {
  API_IMPL_BEGIN
  auto model_proto_copy = std::make_unique<onnx::ModelProto>(*in->UseModelProto());
  if (auto status = OrtModel::CreateOrtModelFromProto(std::move(model_proto_copy), out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetAuthor, const OrtModel* model, const char** const author, size_t* len) {
  API_IMPL_BEGIN
  *author = model->UseModelInfo()->author_.c_str();
  *len = model->UseModelInfo()->author_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetName, const OrtModel* model, const char** const name, size_t* len) {
  API_IMPL_BEGIN
  *name = model->UseModelInfo()->name_.c_str();
  *len = model->UseModelInfo()->name_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetDomain, const OrtModel* model, const char** const domain, size_t* len) {
  API_IMPL_BEGIN
  *domain = model->UseModelInfo()->domain_.c_str();
  *len = model->UseModelInfo()->domain_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetDescription, const OrtModel* model, const char** const description, size_t* len) {
  API_IMPL_BEGIN
  *description = model->UseModelInfo()->description_.c_str();
  *len = model->UseModelInfo()->description_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetVersion, const OrtModel* model, int64_t* version) {
  API_IMPL_BEGIN
  *version = model->UseModelInfo()->version_;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetMetadataCount, const OrtModel* model, size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->model_metadata_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetMetadata, const OrtModel* model, size_t count, const char** const key,
                    size_t* key_len, const char** const value, size_t* value_len) {
  API_IMPL_BEGIN
  *key = model->UseModelInfo()->model_metadata_[count].first.c_str();
  *key_len = model->UseModelInfo()->model_metadata_[count].first.size();
  *value = model->UseModelInfo()->model_metadata_[count].second.c_str();
  *value_len = model->UseModelInfo()->model_metadata_[count].second.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetInputCount, const OrtModel* model, size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->input_features_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetOutputCount, const OrtModel* model, size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->output_features_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetInputName, const OrtModel* model, size_t index,
                    const char** input_name, size_t* count) {
  API_IMPL_BEGIN
  *input_name = model->UseModelInfo()->input_features_[index]->name().c_str();
  *count = model->UseModelInfo()->input_features_[index]->name().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetOutputName, const OrtModel* model, size_t index,
                    const char** output_name, size_t* count) {
  API_IMPL_BEGIN
  *output_name = model->UseModelInfo()->output_features_[index]->name().c_str();
  *count = model->UseModelInfo()->output_features_[index]->name().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetInputDescription, const OrtModel* model, size_t index,
                    const char** input_description, size_t* count) {
  API_IMPL_BEGIN
  *input_description = model->UseModelInfo()->input_features_[index]->doc_string().c_str();
  *count = model->UseModelInfo()->input_features_[index]->doc_string().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetOutputDescription, const OrtModel* model, size_t index,
                    const char** output_description, size_t* count) {
  API_IMPL_BEGIN
  *output_description = model->UseModelInfo()->output_features_[index]->doc_string().c_str();
  *count = model->UseModelInfo()->output_features_[index]->doc_string().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetInputTypeInfo, const OrtModel* model, size_t index, OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  if (auto status = OrtTypeInfo::FromTypeProto(&model->UseModelInfo()->input_features_[index]->type(), type_info)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetOutputTypeInfo, const OrtModel* model, size_t index, OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  if (auto status = OrtTypeInfo::FromTypeProto(&model->UseModelInfo()->output_features_[index]->type(), type_info)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelEnsureNoFloat16, const OrtModel* model) {
  API_IMPL_BEGIN
  auto model_info = model->UseModelInfo();
  auto model_proto = model->UseModelProto();
  auto& graph = model_proto->graph();

  // The model will not contain fp16 operations if:
  // 1. The model has no fp16 inputs
  // 2. The model has no fp16 initializers
  // 3. The model does not create any fp16 intermediary tensors via the Cast (to float16) operator
  // 4. The model does not have any fp16 outputs

  // 1. Ensure that The model has no fp16 inputs
  for (auto input : model_info->input_features_) {
    auto& type = input->type();
    if (type.value_case() == ONNX_NAMESPACE::TypeProto::kTensorType) {
      auto& tensor_type = type.tensor_type();
      if (tensor_type.elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
        std::stringstream error_message;
        error_message << "The model contains a 16-bit input ("
                      << input->name()
                      << "), but the current device does not support 16-bit float.";
        return OrtApis::CreateStatus(ORT_INVALID_GRAPH, error_message.str().c_str());
      }
    }
  }

  // 2. Ensure that the model has no fp16 initializers
  for (int i = 0; i < graph.node_size(); i++) {
    auto node = graph.node(i);
    if (node.op_type() == "Cast" && node.domain().empty()) {
      for (int attribIndex = 0; attribIndex < node.attribute_size(); attribIndex++) {
        auto attribute = node.attribute(attribIndex);
        if (attribute.name() == "to") {
          if (attribute.i() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16) {
            std::stringstream error_message;
            error_message << "The model contains a 16-bit input ("
                          << node.name().c_str()
                          << "), but the current device does not support 16-bit float.";
            return OrtApis::CreateStatus(ORT_INVALID_GRAPH, error_message.str().c_str());
          }
        }
      }
    }
  }

  // 3. Ensure that the model does not create any fp16 intermediary
  //    tensors via the Cast (to float16) operator
  for (int i = 0; i < graph.initializer_size(); i++) {
    auto initializer = graph.initializer(i);
    if (initializer.data_type() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16) {
      std::stringstream error_message;
      error_message << "The model contains a 16-bit input ("
                    << initializer.name().c_str()
                    << "), but the current device does not support 16-bit float.";
      return OrtApis::CreateStatus(ORT_INVALID_GRAPH, error_message.str().c_str());
    }
  }

  // 4. Ensure that the model does not have any fp16 outputs
  for (auto output : model_info->output_features_) {
    auto& type = output->type();
    if (type.value_case() == ONNX_NAMESPACE::TypeProto::kTensorType) {
      auto& tensor_type = type.tensor_type();
      if (tensor_type.elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
        std::stringstream error_message;
        error_message << "The model contains a 16-bit input ("
                      << output->name()
                      << "), but the current device does not support 16-bit float.";
        return OrtApis::CreateStatus(ORT_INVALID_GRAPH, error_message.str().c_str());
      }
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API(void, winmla::ReleaseModel, OrtModel* ptr) {
  delete ptr;
}