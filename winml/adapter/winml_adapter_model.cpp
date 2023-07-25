// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "adapter/pch.h"

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

#include "onnx/defs/schema.h"
#include "core/framework/tensor_type_and_shape.h"

#include "onnx/onnx-ml.pb.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

static std::vector<const char*> GetInitializers(const ONNX_NAMESPACE::ModelProto& model_proto) {
  std::vector<const char*> initializers;
  auto& graph = model_proto.graph();
  auto& graph_initializers = graph.initializer();
  for (auto& initializer : graph_initializers) {
    initializers.push_back(initializer.name().c_str());
  }
  return initializers;
}

static std::vector<const ONNX_NAMESPACE::ValueInfoProto*> GetInputsWithoutInitializers(
  const ONNX_NAMESPACE::ModelProto& model_proto
) {
  auto initializers = GetInitializers(model_proto);

  std::vector<const ONNX_NAMESPACE::ValueInfoProto*> inputs_without_initializers;
  auto& graph = model_proto.graph();
  auto& inputs = graph.input();
  for (auto& input : inputs) {
    if (input.has_name() && input.has_type()) {
      auto found_it = std::find_if(std::begin(initializers), std::end(initializers), [&](auto& initializer) {
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

static std::vector<const ONNX_NAMESPACE::ValueInfoProto*> GetOutputs(const ONNX_NAMESPACE::ModelProto& model_proto) {
  std::vector<const ONNX_NAMESPACE::ValueInfoProto*> outputs_with_name;
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
  ModelInfo(const ONNX_NAMESPACE::ModelProto* model_proto) { Initialize(model_proto); }

 public:
  // model metadata
  std::string author_;
  std::string name_;
  std::string domain_;
  std::string description_;
  int64_t version_;
  std::vector<std::pair<std::string, std::string>> model_metadata_;
  std::vector<const ONNX_NAMESPACE::ValueInfoProto*> input_features_;
  std::vector<const ONNX_NAMESPACE::ValueInfoProto*> output_features_;
  bool requires_float16_support_;

 private:
  void Initialize(const ONNX_NAMESPACE::ModelProto* model_proto) {
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

OrtModel::OrtModel(std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto)
  : model_proto_(std::move(model_proto)),
    model_info_(std::make_unique<ModelInfo>(model_proto_.get())) {
}

// factory methods for creating an ort model from a path
static OrtStatus* CreateModelProto(const char* path, std::unique_ptr<ONNX_NAMESPACE::ModelProto>& out) {
  int file_descriptor;

  auto path_str = std::string(path);
  auto wide_path = onnxruntime::ToWideString(path_str);

  _set_errno(0);  // clear errno
  _wsopen_s(
    &file_descriptor, wide_path.c_str(), O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE
  );

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

  auto model_proto = std::unique_ptr<ONNX_NAMESPACE::ModelProto>(new ONNX_NAMESPACE::ModelProto());

  auto parse_succeeded = model_proto->ParseFromZeroCopyStream(&stream);
  if (!parse_succeeded) {
    return OrtApis::CreateStatus(ORT_INVALID_PROTOBUF, "Failed to parse model file!");
  }

  out = std::move(model_proto);

  return S_OK;
}

OrtStatus* OrtModel::CreateEmptyModel(int64_t opset, OrtModel** model) {
  auto model_proto = std::unique_ptr<ONNX_NAMESPACE::ModelProto>(new ONNX_NAMESPACE::ModelProto());
  auto opsetimportproto = model_proto->add_opset_import();
  opsetimportproto->set_version(opset);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  return OrtModel::CreateOrtModelFromProto(std::move(model_proto), model);
}

OrtStatus* OrtModel::CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model) {
  ORT_UNUSED_PARAMETER(len);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto;

  if (auto status = CreateModelProto(path, model_proto)) {
    return status;
  }

  return OrtModel::CreateOrtModelFromProto(std::move(model_proto), model);
}

OrtStatus* OrtModel::CreateOrtModelFromData(void* data, size_t len, OrtModel** model) {
  auto model_proto = std::unique_ptr<ONNX_NAMESPACE::ModelProto>(new ONNX_NAMESPACE::ModelProto());

  auto parse_succeeded = model_proto->ParseFromArray(data, static_cast<int>(len));
  if (!parse_succeeded) {
    return OrtApis::CreateStatus(ORT_INVALID_PROTOBUF, "Failed to parse model stream!");
  }

  return OrtModel::CreateOrtModelFromProto(std::move(model_proto), model);
}

OrtStatus* OrtModel::CreateOrtModelFromProto(
  std::unique_ptr<ONNX_NAMESPACE::ModelProto>&& model_proto, OrtModel** model
) {
  *model = new (std::nothrow) OrtModel(std::move(model_proto));
  if (*model == nullptr) {
    return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Engine failed to create a model!");
  }

  return nullptr;
}

const ModelInfo* OrtModel::UseModelInfo() const {
  return model_info_.get();
}

ONNX_NAMESPACE::ModelProto* OrtModel::UseModelProto() const {
  return model_proto_.get();
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto> OrtModel::DetachModelProto() {
  return std::move(model_proto_);
}

void OrtModel::RefreshModelInfo() {
  auto new_info = std::make_unique<ModelInfo>(model_proto_.get());
  model_info_->author_ = std::move(new_info->author_);
  model_info_->description_ = std::move(new_info->description_);
  model_info_->domain_ = std::move(new_info->domain_);
  model_info_->input_features_ = std::move(new_info->input_features_);
  model_info_->model_metadata_ = std::move(new_info->model_metadata_);
  model_info_->name_ = std::move(new_info->name_);
  model_info_->output_features_ = std::move(new_info->output_features_);
  model_info_->requires_float16_support_ = std::move(new_info->requires_float16_support_);
  model_info_->version_ = std::move(new_info->version_);
}

ORT_API_STATUS_IMPL(
  winmla::CreateModelFromPath, _In_ const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out
) {
  API_IMPL_BEGIN
  if (auto status = OrtModel::CreateOrtModelFromPath(model_path, size, out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromData, _In_opt_ void* data, _In_ size_t size, _Outptr_ OrtModel** out) {
  API_IMPL_BEGIN
  if (auto status = OrtModel::CreateOrtModelFromData(data, size, out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CloneModel, _In_ const OrtModel* in, _Outptr_ OrtModel** out) {
  API_IMPL_BEGIN
  auto model_proto_copy = std::make_unique<ONNX_NAMESPACE::ModelProto>(*in->UseModelProto());
  if (auto status = OrtModel::CreateOrtModelFromProto(std::move(model_proto_copy), out)) {
    return status;
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SaveModel, _In_ const OrtModel* in, _In_ const wchar_t* const file_name, _In_ size_t len) {
  API_IMPL_BEGIN
  int fd;
  std::wstring file_path = file_name;
  Status status = onnxruntime::Env::Default().FileOpenWr(file_path, fd);
  if (fd < 0) {
    return OrtApis::CreateStatus(ORT_NO_SUCHFILE, "File not found!");
  }

  auto model_proto = in->UseModelProto();
  google::protobuf::io::FileOutputStream output(fd);
  const bool success = model_proto->SerializeToZeroCopyStream(&output) && output.Flush();
  if (!success) {
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, "Failed to serialize model!");
  }
  output.Close();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetAuthor, _In_ const OrtModel* model, _Out_ const char** const author, _Out_ size_t* len
) {
  API_IMPL_BEGIN
  *author = model->UseModelInfo()->author_.c_str();
  *len = model->UseModelInfo()->author_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetName, _In_ const OrtModel* model, _Out_ const char** const name, _Out_ size_t* len
) {
  API_IMPL_BEGIN
  *name = model->UseModelInfo()->name_.c_str();
  *len = model->UseModelInfo()->name_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelSetName, _In_ const OrtModel* model, _In_ const char* const name) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();
  ONNX_NAMESPACE::GraphProto& graph = *model_proto->mutable_graph();
  graph.set_name(name);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetDomain, _In_ const OrtModel* model, _Out_ const char** const domain, _Out_ size_t* len
) {
  API_IMPL_BEGIN
  *domain = model->UseModelInfo()->domain_.c_str();
  *len = model->UseModelInfo()->domain_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetDescription, _In_ const OrtModel* model, _Out_ const char** const description, _Out_ size_t* len
) {
  API_IMPL_BEGIN
  *description = model->UseModelInfo()->description_.c_str();
  *len = model->UseModelInfo()->description_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetVersion, _In_ const OrtModel* model, _Out_ int64_t* version) {
  API_IMPL_BEGIN
  *version = model->UseModelInfo()->version_;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetMetadataCount, _In_ const OrtModel* model, _Out_ size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->model_metadata_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetMetadata,
  _In_ const OrtModel* model,
  _In_ size_t count,
  _Out_ const char** const key,
  _Out_ size_t* key_len,
  _Out_ const char** const value,
  _Out_ size_t* value_len
) {
  API_IMPL_BEGIN
  *key = model->UseModelInfo()->model_metadata_[count].first.c_str();
  *key_len = model->UseModelInfo()->model_metadata_[count].first.size();
  *value = model->UseModelInfo()->model_metadata_[count].second.c_str();
  *value_len = model->UseModelInfo()->model_metadata_[count].second.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetInputCount, _In_ const OrtModel* model, _Out_ size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->input_features_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelGetOutputCount, _In_ const OrtModel* model, _Out_ size_t* count) {
  API_IMPL_BEGIN
  *count = model->UseModelInfo()->output_features_.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetInputName,
  _In_ const OrtModel* model,
  _In_ size_t index,
  _Out_ const char** input_name,
  _Out_ size_t* count
) {
  API_IMPL_BEGIN
  *input_name = model->UseModelInfo()->input_features_[index]->name().c_str();
  *count = model->UseModelInfo()->input_features_[index]->name().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetOutputName,
  _In_ const OrtModel* model,
  _In_ size_t index,
  _Out_ const char** output_name,
  _Out_ size_t* count
) {
  API_IMPL_BEGIN
  *output_name = model->UseModelInfo()->output_features_[index]->name().c_str();
  *count = model->UseModelInfo()->output_features_[index]->name().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetInputDescription,
  _In_ const OrtModel* model,
  _In_ size_t index,
  _Out_ const char** input_description,
  _Out_ size_t* count
) {
  API_IMPL_BEGIN
  *input_description = model->UseModelInfo()->input_features_[index]->doc_string().c_str();
  *count = model->UseModelInfo()->input_features_[index]->doc_string().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetOutputDescription,
  _In_ const OrtModel* model,
  _In_ size_t index,
  _Out_ const char** output_description,
  _Out_ size_t* count
) {
  API_IMPL_BEGIN
  *output_description = model->UseModelInfo()->output_features_[index]->doc_string().c_str();
  *count = model->UseModelInfo()->output_features_[index]->doc_string().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetInputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info
) {
  API_IMPL_BEGIN
  auto info = OrtTypeInfo::FromTypeProto(model->UseModelInfo()->input_features_[index]->type());
  *type_info = info.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetOutputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info
) {
  API_IMPL_BEGIN
  auto info = OrtTypeInfo::FromTypeProto(model->UseModelInfo()->output_features_[index]->type());
  *type_info = info.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::ModelEnsureNoFloat16, _In_ const OrtModel* model) {
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
        error_message << "The model contains a 16-bit input (" << input->name()
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
          if (attribute.i() == ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT16) {
            std::stringstream error_message;
            error_message << "The model contains a 16-bit input (" << node.name().c_str()
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
    if (initializer.data_type() == ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT16) {
      std::stringstream error_message;
      error_message << "The model contains a 16-bit input (" << initializer.name().c_str()
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
        error_message << "The model contains a 16-bit input (" << output->name()
                      << "), but the current device does not support 16-bit float.";
        return OrtApis::CreateStatus(ORT_INVALID_GRAPH, error_message.str().c_str());
      }
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CreateModel, _In_ int64_t opset, _Outptr_ OrtModel** out) {
  API_IMPL_BEGIN
  return OrtModel::CreateEmptyModel(opset, out);
  API_IMPL_END
}

static ONNX_NAMESPACE::TensorProto_DataType ONNXTensorElementDataTypeToTensorProto_DataType(
  ONNXTensorElementDataType type
) {
  switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return ONNX_NAMESPACE::TensorProto_DataType_STRING;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
    default:
      return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
}

static void CreateTypeProto_Tensor(
  ONNX_NAMESPACE::TypeProto_Tensor* mutable_tensor_type,
  const char* const name,
  const int64_t* shape,
  size_t shape_len,
  ONNX_NAMESPACE::TensorProto_DataType data_type
) {
  mutable_tensor_type->set_elem_type(data_type);

  size_t dim_param = 0;
  for (size_t i = 0; i < shape_len; i++) {
    if (shape[i] == -1) {
      std::ostringstream str;
      str << name << dim_param++;
      mutable_tensor_type->mutable_shape()->add_dim()->set_dim_param(str.str().c_str(), 1);
    } else {
      mutable_tensor_type->mutable_shape()->add_dim()->set_dim_value(shape[i]);
    }
  }

  if (shape_len > 0) {
    mutable_tensor_type->mutable_shape()->mutable_dim(0)->set_denotation("DATA_BATCH");
  }
}

ORT_API_STATUS_IMPL(
  winmla::ModelAddInput, _In_ OrtModel* model, _In_ const char* const input_name, _In_ OrtTypeInfo* info
) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();
  ONNX_NAMESPACE::GraphProto& graph = *model_proto->mutable_graph();
  ONNX_NAMESPACE::ValueInfoProto& input = *graph.add_input();
  input.set_name(input_name);

  if (info->type == ONNXType::ONNX_TYPE_TENSOR) {
    auto num_dims = info->data->shape.NumDimensions();
    CreateTypeProto_Tensor(
      input.mutable_type()->mutable_tensor_type(),
      input_name,
      (num_dims == 0) ? nullptr : &info->data->shape[0],
      num_dims,
      ONNXTensorElementDataTypeToTensorProto_DataType(info->data->type)
    );
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelAddConstantInput,
  _In_ OrtModel* model,
  _In_ const char* const input_name,
  _In_ OrtTypeInfo* info,
  _In_ OrtValue* value
) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();
  ONNX_NAMESPACE::GraphProto& graph = *model_proto->mutable_graph();
  ONNX_NAMESPACE::TensorProto& input = *graph.add_initializer();
  input.set_name(input_name);

  auto num_dims = info->data->shape.NumDimensions();
  for (size_t i = 0; i < num_dims; i++) {
    input.add_dims(info->data->shape[i]);
  }

  input.set_data_type(ONNXTensorElementDataTypeToTensorProto_DataType(info->data->type));
  auto tensor = value->GetMutable<onnxruntime::Tensor>();
  input.set_raw_data(tensor->DataRaw(), tensor->SizeInBytes());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelAddOutput, _In_ OrtModel* model, _In_ const char* const output_name, _In_ OrtTypeInfo* info
) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();
  ONNX_NAMESPACE::GraphProto& graph = *model_proto->mutable_graph();
  ONNX_NAMESPACE::ValueInfoProto& output = *graph.add_output();
  output.set_name(output_name);

  if (info->type == ONNXType::ONNX_TYPE_TENSOR) {
    CreateTypeProto_Tensor(
      output.mutable_type()->mutable_tensor_type(),
      output_name,
      &info->data->shape[0],
      info->data->shape.NumDimensions(),
      ONNXTensorElementDataTypeToTensorProto_DataType(info->data->type)
    );
  }
  return nullptr;
  API_IMPL_END
}

static const onnx::OpSchema* GetSchema(const char* const op_type, int64_t opset, const char* const op_domain) {
  std::string domain = onnx::ONNX_DOMAIN;
  if (op_domain) {
    domain = op_domain;
  }

  auto registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  return registry->GetSchema(op_type, static_cast<int>(opset), domain);
}

ORT_API_STATUS_IMPL(
  winmla::ModelAddOperator,
  _In_ OrtModel* model,
  _In_ const char* const op_type,
  _In_ const char* const op_name,
  _In_ int64_t opset,
  _In_ const char* const op_domain,
  _In_ const char* const* input_names,
  _In_ size_t num_inputs,
  _In_ const char* const* output_names,
  _In_ size_t num_outputs,
  _In_ const char* const* attribute_names,
  _In_ OrtValue** attribute_values,
  _In_ size_t num_attributes
) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();
  ONNX_NAMESPACE::GraphProto& graph = *model_proto->mutable_graph();
  onnx::NodeProto& node = *graph.add_node();
  node.set_op_type(op_type);
  node.set_name(op_name);
  node.set_domain(op_domain);

  auto schema = GetSchema(op_type, opset, op_domain);
  auto all_attributes = schema->attributes();

  for (size_t i = 0; i < num_attributes; i++) {
    auto tensor = attribute_values[i]->GetMutable<onnxruntime::Tensor>();

    auto attr = node.add_attribute();
    attr->set_name(attribute_names[i]);
    auto& schema_attribute_definition = all_attributes.at(attribute_names[i]);
    attr->set_type(schema_attribute_definition.type);

    switch (schema_attribute_definition.type) {
      case onnx::AttributeProto_AttributeType_INT: {
        if (tensor->Shape().Size() != 1) {
          return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Expected a single int64 value!");
        }
        auto raw_data = tensor->DataRaw();
        attr->set_i(*reinterpret_cast<const int64_t*>(raw_data));
        break;
      }
      case onnx::AttributeProto_AttributeType_FLOAT: {
        if (tensor->Shape().Size() != 1) {
          return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Expected a single float value!");
        }
        auto raw_data = tensor->DataRaw();
        attr->set_f(*reinterpret_cast<const float*>(raw_data));
        break;
      }
      case onnx::AttributeProto_AttributeType_STRING: {
        if (tensor->Shape().Size() != 1) {
          return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Expected a single string value!");
        }
        auto raw_data = tensor->DataRaw();
        attr->set_s(*reinterpret_cast<const std::string*>(raw_data));
        break;
      }
      case onnx::AttributeProto_AttributeType_INTS: {
        auto raw_data = tensor->DataRaw();
        for (int j = 0; j < tensor->Shape().Size(); j++) {
          attr->add_ints(*(reinterpret_cast<const int64_t*>(raw_data) + j));
        }
        break;
      }
      case onnx::AttributeProto_AttributeType_FLOATS: {
        auto raw_data = tensor->DataRaw();
        for (int j = 0; j < tensor->Shape().Size(); j++) {
          attr->add_floats(*(reinterpret_cast<const float*>(raw_data) + j));
        }
        break;
      }
      case onnx::AttributeProto_AttributeType_TENSOR: {
        auto tensor_proto = attr->add_tensors();
        auto prim_type = tensor->DataType()->AsPrimitiveDataType();
        if (prim_type == nullptr) {
          return OrtApis::CreateStatus(ORT_ENGINE_ERROR, "Undefined tensor type!");
        }
        tensor_proto->set_data_type(prim_type->GetDataType());
        tensor_proto->set_raw_data(tensor->DataRaw(), tensor->SizeInBytes());
        break;
      }
    }
  }

  for (size_t i = 0; i < num_inputs; i++) {
    auto name = input_names[i];
    if (name != nullptr) {
      node.add_input(name);
    } else {
      node.add_input();
    }
  }

  for (size_t i = 0; i < num_outputs; i++) {
    auto name = output_names[i];
    if (name != nullptr) {
      node.add_output(name);
    } else {
      node.add_output("unused");
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::ModelGetOpsetVersion, _In_ OrtModel* model, _In_ const char* const domain, _Out_ int32_t* version
) {
  API_IMPL_BEGIN
  auto model_proto = model->UseModelProto();

  *version = -1;
  auto size = static_cast<int>(model_proto->opset_import_size());
  for (int i = 0; i < size; i++) {
    auto& current_opset = model_proto->opset_import(i);
    auto& current_domain = current_opset.domain();
    if (_strnicmp(domain, current_domain.c_str(), current_domain.size()) == 0) {
      *version = static_cast<int32_t>(current_opset.version());
      break;
    }
  }

  return nullptr;
  API_IMPL_END
}

ORT_API(void, winmla::ReleaseModel, OrtModel* ptr) {
  delete ptr;
}

#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"

ORT_API_STATUS_IMPL(
  winmla::CreateTensorTypeInfo,
  _In_ const int64_t* dim_values,
  size_t dim_count,
  ONNXTensorElementDataType type,
  _Out_ OrtTypeInfo** ort_type_info
) {
  API_IMPL_BEGIN
  auto tensor_shape = onnxruntime::TensorShape(dim_values, dim_count);
  auto type_and_shape = OrtTensorTypeAndShapeInfo::GetTensorShapeAndTypeHelper(type, std::move(tensor_shape), nullptr);
  *ort_type_info = OrtTypeInfo::MakePtr(ONNX_TYPE_TENSOR, std::move(type_and_shape)).release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CreateSequenceTypeInfo, _Out_ OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::CreateMapTypeInfo, _Out_ OrtTypeInfo** type_info) {
  API_IMPL_BEGIN
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::OperatorGetNumInputs,
  _In_ const char* const op_type,
  _In_ int64_t opset,
  _In_ const char* const op_domain,
  _Out_ size_t* num_inputs
) {
  API_IMPL_BEGIN
  auto schema = GetSchema(op_type, opset, op_domain);
  *num_inputs = schema->inputs().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::OperatorGetInputName,
  _In_ const char* const op_type,
  _In_ int64_t opset,
  _In_ const char* const op_domain,
  _In_ size_t index,
  _Out_ const char** const name
) {
  API_IMPL_BEGIN
  auto schema = GetSchema(op_type, opset, op_domain);
  *name = schema->inputs().at(index).GetName().c_str();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::OperatorGetNumOutputs,
  _In_ const char* const op_type,
  _In_ int64_t opset,
  _In_ const char* const op_domain,
  _Out_ size_t* num_outputs
) {
  API_IMPL_BEGIN
  auto schema = GetSchema(op_type, opset, op_domain);
  *num_outputs = schema->outputs().size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::OperatorGetOutputName,
  _In_ const char* const op_type,
  _In_ int64_t opset,
  _In_ const char* const op_domain,
  _In_ size_t index,
  _Out_ const char** const name
) {
  API_IMPL_BEGIN
  auto schema = GetSchema(op_type, opset, op_domain);
  *name = schema->outputs().at(index).GetName().c_str();
  return nullptr;
  API_IMPL_END
}
#include "core/platform/threadpool.h"
#include "core/platform/env.h"

ORT_API_STATUS_IMPL(
  winmla::CreateThreadPool, _In_ ThreadPoolType type, _In_ OrtThreadPoolOptions* options, _Outptr_ OrtThreadPool** out
) {
  API_IMPL_BEGIN
  OrtThreadPoolParams params = {};
  params.thread_pool_size = options->thread_pool_size;
  params.auto_set_affinity = options->auto_set_affinity;
  params.allow_spinning = options->allow_spinning;
  params.dynamic_block_base_ = options->dynamic_block_base_;
  params.stack_size = options->stack_size;
  params.name = options->name;
  params.set_denormal_as_zero = options->set_denormal_as_zero;

  auto unique_tp = onnxruntime::concurrency::CreateThreadPool(
    &onnxruntime::Env::Default(), params, (onnxruntime::concurrency::ThreadPoolType)type
  );
  *out = reinterpret_cast<OrtThreadPool*>(unique_tp.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, winmla::ReleaseThreadPool, OrtThreadPool* ptr) {
  delete reinterpret_cast<onnxruntime::concurrency::ThreadPool*>(ptr);
}

ORT_API_STATUS_IMPL(
  winmla::JoinModels,
  _In_ OrtModel* first_model,
  _In_ OrtModel* second_model,
  _In_ const char* const* output_names,
  _In_ const char* const* input_names,
  size_t num_linkages,
  bool promote_unlinked_outputs,
  _In_ const char* const join_node_prefix
) {
  API_IMPL_BEGIN

  std::string second_model_prefix = join_node_prefix;
  auto first_model_proto = first_model->UseModelProto();
  auto second_model_proto = second_model->DetachModelProto();

  // Remove old outputs
  if (promote_unlinked_outputs) {
    // Copy the output of the first model
    auto first_outputs = first_model_proto->graph().output();

    // Clear all outputs
    first_model_proto->mutable_graph()->mutable_output()->Clear();

    // Add back output
    for (int i = first_outputs.size() - 1; i >= 0; i--) {
      auto& output = first_outputs.at(i);
      auto output_name = output.name();

      auto found_it = std::find_if(output_names, output_names + num_linkages, [output_name](auto& name) {
        return std::strcmp(name, output_name.c_str()) == 0;
      });
      if (found_it == (output_names + num_linkages)) {
        // if output.name() is not found in the linkages, it is unlinked, and it should be promoted
        auto& promoted_output = *first_model_proto->mutable_graph()->add_output();
        promoted_output = std::move(output);
      }
    }
  } else {
    // remove all first model outputs
    first_model_proto->mutable_graph()->mutable_output()->Clear();
  }

  // add all model outputs from the second model
  for (int i = 0; i < second_model_proto->graph().output_size(); i++) {
    auto& other_output = *second_model_proto->mutable_graph()->mutable_output(i);
    *other_output.mutable_name() = second_model_prefix + other_output.name();
    auto& output = *first_model_proto->mutable_graph()->add_output();
    output = std::move(other_output);
  }

  // loop through second model inputs and promote the unlinked ones to the main model inputs
  for (int i = 0; i < second_model_proto->graph().input_size(); i++) {
    auto& other_input = *second_model_proto->mutable_graph()->mutable_input(i);
    auto old_name = other_input.name();
    *other_input.mutable_name() = second_model_prefix + old_name;

    auto found_it = std::find_if(input_names, input_names + num_linkages, [old_name](auto& name) {
      return std::strcmp(name, old_name.c_str()) == 0;
    });
    bool is_linked =
      found_it != (input_names + num_linkages);  // figure out if other_input.name() exists in the output_names mapped
    if (!is_linked) {
      auto& input = *first_model_proto->mutable_graph()->add_input();
      input = std::move(other_input);
    }
  }

  // add all initializers
  for (int i = 0; i < second_model_proto->graph().initializer_size(); i++) {
    auto& other_initializer = *second_model_proto->mutable_graph()->mutable_initializer(i);
    *other_initializer.mutable_name() = second_model_prefix + other_initializer.name();
    auto& initializer = *first_model_proto->mutable_graph()->add_initializer();
    initializer = std::move(other_initializer);
  }

  // add all nodes
  for (int i = 0; i < second_model_proto->graph().node_size(); i++) {
    auto& other_node = *second_model_proto->mutable_graph()->mutable_node(i);
    if (0 != strcmp(other_node.name().c_str(), "")) {
      *other_node.mutable_name() = second_model_prefix + other_node.name();
    }
    for (int j = 0; j < other_node.input_size(); j++) {
      *other_node.mutable_input(j) = second_model_prefix + other_node.input(j);
    }
    for (int j = 0; j < other_node.output_size(); j++) {
      *other_node.mutable_output(j) = second_model_prefix + other_node.output(j);
    }
    auto& node = *first_model_proto->mutable_graph()->add_node();
    node = std::move(other_node);
  }

  // WinML+RT API only supports opset 7 and above models.
  // In practice this number is always overwritten by the for loop below which will find the actual opset version.
  int64_t opset = 7;
  for (int i = 0; i < second_model_proto->opset_import_size(); i++) {
    auto mutable_opset_import = second_model_proto->mutable_opset_import(i);
    auto domain = mutable_opset_import->has_domain() ? mutable_opset_import->domain() : std::string("");
    auto version = mutable_opset_import->version();

    // does the domain exist in the first model?
    auto found_it = std::find_if(
      first_model_proto->mutable_opset_import()->begin(),
      first_model_proto->mutable_opset_import()->end(),
      [&domain](auto& mutable_opset_import) {
        auto first_model_domain = mutable_opset_import.has_domain() ? mutable_opset_import.domain() : std::string("");
        return 0 == strcmp(first_model_domain.c_str(), domain.c_str());
      }
    );
    if (found_it != first_model_proto->mutable_opset_import()->end()) {
      found_it->set_version(std::max(found_it->version(), version));
      if (0 == strcmp(domain.c_str(), "")) {
        opset = found_it->version();
      }
    }
  }

  // add identity ops to rename all of the first model outputs to secondmodel inputs with prefix for each linkage
  for (int i = 0; i < num_linkages; i++) {
    auto op_output_name = second_model_prefix + *(input_names + i);
    const char* const op_output_name_const_str = op_output_name.c_str();
    std::string name = "IdentityTo";
    name += second_model_prefix + *(input_names + i);
    ModelAddOperator(
      first_model,
      "Identity",
      name.c_str(),
      opset,
      "",
      (output_names + i),
      1,
      &op_output_name_const_str,
      1,
      nullptr,
      nullptr,
      0
    );
  }
  first_model->RefreshModelInfo();

  return nullptr;
  API_IMPL_END
}
