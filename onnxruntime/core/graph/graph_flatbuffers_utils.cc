// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/graph/graph.h>
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/tensorprotoutils.h"
#include "graph_flatbuffers_utils.h"
#include "flatbuffers/flatbuffers.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::experimental;

namespace onnxruntime {
namespace experimental {
namespace utils {

#if !defined(ORT_MINIMAL_BUILD)

template <typename DimsFieldType>
inline flatbuffers::Offset<flatbuffers::Vector<int64_t>>
SaveDims(flatbuffers::FlatBufferBuilder& builder, const DimsFieldType& dims) {
  std::vector<int64_t> dims_data(dims.size());
  std::copy(dims.cbegin(), dims.cend(), dims_data.begin());
  return builder.CreateVector(dims_data);
}

Status SaveInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                const TensorProto& initializer,
                                const Path& model_path,
                                flatbuffers::Offset<fbs::Tensor>& fbs_tensor) {
  auto name = SaveStringToOrtFormat(builder, initializer.has_name(), initializer.name());
  auto doc_string = SaveStringToOrtFormat(builder, initializer.has_doc_string(), initializer.doc_string());
  auto dims = SaveDims(builder, initializer.dims());

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> string_data;
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> raw_data;

  auto src_type = initializer.data_type();
  const bool has_string_data = src_type == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  if (has_string_data) {
    std::vector<std::string> string_data_vec(initializer.string_data().size());
    std::copy(initializer.string_data().cbegin(), initializer.string_data().cend(), string_data_vec.begin());
    string_data = builder.CreateVectorOfStrings(string_data_vec);
  } else {
    std::unique_ptr<uint8_t[]> unpacked_tensor;
    size_t tensor_byte_size = 0;
    ORT_RETURN_IF_ERROR(
        onnxruntime::utils::UnpackInitializerData(initializer, model_path, unpacked_tensor, tensor_byte_size));
    raw_data = builder.CreateVector(unpacked_tensor.get(), tensor_byte_size);
  }

  fbs::TensorBuilder tb(builder);
  tb.add_name(name);
  tb.add_doc_string(doc_string);
  tb.add_dims(dims);
  tb.add_data_type(static_cast<fbs::TensorDataType>(src_type));
  if (has_string_data)
    tb.add_string_data(string_data);
  else
    tb.add_raw_data(raw_data);
  fbs_tensor = tb.Finish();
  return Status::OK();
}

Status SaveSparseInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      const ONNX_NAMESPACE::SparseTensorProto& initializer,
                                      const Path& model_path,
                                      flatbuffers::Offset<fbs::SparseTensor>& fbs_sparse_tensor) {
  // values
  const auto& values = initializer.values();
  flatbuffers::Offset<fbs::Tensor> values_off;
  ORT_RETURN_IF_ERROR(SaveInitializerOrtFormat(builder, values, model_path, values_off));

  // Indicies
  const auto& indicies = initializer.indices();
  flatbuffers::Offset<fbs::Tensor> indicies_off;
  ORT_RETURN_IF_ERROR(SaveInitializerOrtFormat(builder, indicies, model_path, indicies_off));

  // Shape
  auto shape = SaveDims(builder, initializer.dims());

  fbs::SparseTensorBuilder stb(builder);
  stb.add_values(values_off);
  stb.add_indices(indicies_off);
  stb.add_dims(shape);

  fbs_sparse_tensor = stb.Finish();

  return Status::OK();
}

#define GET_FBS_ATTR(BUILDER, TYPE, DATA_NAME, DATA) \
  fbs::AttributeBuilder attr_builder(BUILDER);       \
  attr_builder.add_name(name);                       \
  attr_builder.add_doc_string(doc_string);           \
  attr_builder.add_type(TYPE);                       \
  attr_builder.add_##DATA_NAME(DATA);                \
  fbs_attr = attr_builder.Finish();

#define GET_DATA_VEC(TYPE, NAME, SRC_DATA) \
  std::vector<TYPE> NAME(SRC_DATA.size()); \
  std::copy(SRC_DATA.cbegin(), SRC_DATA.cend(), NAME.begin());

Status SaveAttributeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                              const AttributeProto& attr_proto,
                              flatbuffers::Offset<fbs::Attribute>& fbs_attr,
                              const Path& model_path,
                              const onnxruntime::Graph* subgraph) {
  auto name = SaveStringToOrtFormat(builder, attr_proto.has_name(), attr_proto.name());
  auto doc_string = SaveStringToOrtFormat(builder, attr_proto.has_doc_string(), attr_proto.doc_string());
  auto type = static_cast<fbs::AttributeType>(attr_proto.type());
  switch (type) {
    case fbs::AttributeType::FLOAT: {
      GET_FBS_ATTR(builder, type, f, attr_proto.f());
    } break;
    case fbs::AttributeType::INT: {
      GET_FBS_ATTR(builder, type, i, attr_proto.i());
    } break;
    case fbs::AttributeType::STRING: {
      auto s = builder.CreateString(attr_proto.s());
      GET_FBS_ATTR(builder, type, s, s);
    } break;
    case fbs::AttributeType::TENSOR: {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          experimental::utils::SaveInitializerOrtFormat(builder, attr_proto.t(), model_path, fbs_tensor));
      GET_FBS_ATTR(builder, type, t, fbs_tensor);
    } break;
    case fbs::AttributeType::GRAPH: {
      ORT_RETURN_IF(nullptr == subgraph, "Graph attribute value was null. Invalid ORT format model.");
      flatbuffers::Offset<fbs::Graph> fbs_graph;
      ORT_RETURN_IF_ERROR(subgraph->SaveToOrtFormat(builder, fbs_graph));
      GET_FBS_ATTR(builder, type, g, fbs_graph);
    } break;
    case fbs::AttributeType::FLOATS: {
      GET_DATA_VEC(float, floats_vec_, attr_proto.floats());
      auto floats = builder.CreateVector(floats_vec_);
      GET_FBS_ATTR(builder, type, floats, floats);
    } break;
    case fbs::AttributeType::INTS: {
      GET_DATA_VEC(int64_t, ints_vec_, attr_proto.ints());
      auto ints = builder.CreateVector(ints_vec_);
      GET_FBS_ATTR(builder, type, ints, ints);
    } break;
    case fbs::AttributeType::STRINGS: {
      GET_DATA_VEC(std::string, strings_vec_, attr_proto.strings());
      auto strings = builder.CreateVectorOfStrings(strings_vec_);
      GET_FBS_ATTR(builder, type, strings, strings);
    } break;
    case fbs::AttributeType::TENSORS: {
      std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors_vec;
      fbs_tensors_vec.reserve(attr_proto.tensors().size());
      for (const auto& tensor : attr_proto.tensors()) {
        flatbuffers::Offset<fbs::Tensor> fbs_tensor;
        ORT_RETURN_IF_ERROR(
            experimental::utils::SaveInitializerOrtFormat(builder, tensor, model_path, fbs_tensor));
        fbs_tensors_vec.push_back(fbs_tensor);
      }
      auto tensors = builder.CreateVector(fbs_tensors_vec);
      GET_FBS_ATTR(builder, type, tensors, tensors);
    } break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "SaveAttributeOrtFormat: Unsupported attribute type: ", fbs::EnumNameAttributeType(type));
      break;
  }

  return Status::OK();
}

#undef GET_FBS_ATTR
#undef GET_DATA_VEC

#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)

Status LoadInitializerOrtFormat(const fbs::Tensor& fbs_tensor,
                                TensorProto& initializer) {
  initializer.Clear();

  LOAD_STR_FROM_ORT_FORMAT(initializer, name, fbs_tensor.name());
  LOAD_STR_FROM_ORT_FORMAT(initializer, doc_string, fbs_tensor.doc_string());

  auto fbs_dims = fbs_tensor.dims();
  ORT_RETURN_IF(nullptr == fbs_dims, "Missing dimensions for initializer. Invalid ORT format model.");
  initializer.mutable_dims()->Add(fbs_dims->cbegin(), fbs_dims->cend());

  auto fbs_data_type = fbs_tensor.data_type();
  initializer.set_data_type(static_cast<int32_t>(fbs_data_type));
  if (fbs_data_type == fbs::TensorDataType::STRING) {
    auto fbs_str_data = fbs_tensor.string_data();
    ORT_RETURN_IF(nullptr == fbs_str_data, "Missing string data for initializer. Invalid ORT format model.");
    auto mutable_str_data = initializer.mutable_string_data();
    mutable_str_data->Reserve(fbs_str_data->size());
    for (const auto* fbs_str : *fbs_str_data) {
      mutable_str_data->Add(fbs_str->str());
    }
  } else {
    const auto* fbs_raw_data = fbs_tensor.raw_data();
    ORT_RETURN_IF(nullptr == fbs_raw_data, "Missing raw data for initializer. Invalid ORT format model.");

    // fbs_raw_data is uint8_t vector, so the size is byte size
    initializer.set_raw_data(fbs_raw_data->Data(), fbs_raw_data->size());
  }

  return Status::OK();
}

Status LoadSparseInitializerOrtFormat(const fbs::SparseTensor& fbs_sparse_tensor,
                                      SparseTensorProto& initializer) {
  SparseTensorProto loaded_initializer;
  auto fbs_values_tensor = fbs_sparse_tensor.values();
  ORT_RETURN_IF(nullptr == fbs_values_tensor, "Missing values for sparse initializer. Invalid ORT format model.");
  auto* values_tensor = loaded_initializer.mutable_values();
  ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_values_tensor, *values_tensor));
  ORT_RETURN_IF(values_tensor->name().empty(), "Missing name for SparseTensor initializer. Invalid ORT format model.");

  auto fbs_indicies_tensor = fbs_sparse_tensor.indices();
  ORT_RETURN_IF(nullptr == fbs_indicies_tensor, "Missing indicies for sparse initializer: ", "'", values_tensor->name(), "'",
                "Invalid ORT format model.");
  auto* indicies_tensor = loaded_initializer.mutable_indices();
  ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_indicies_tensor, *indicies_tensor));

  auto fbs_dims = fbs_sparse_tensor.dims();
  ORT_RETURN_IF(nullptr == fbs_dims, "Missing dims for sparse initializer: ", "'", values_tensor->name(), "'",
                "Invalid ORT format model.");
  loaded_initializer.mutable_dims()->Add(fbs_dims->cbegin(), fbs_dims->cend());

  swap(loaded_initializer, initializer);
  return Status::OK();
}

Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                              ONNX_NAMESPACE::AttributeProto& attr_proto,
                              std::unique_ptr<onnxruntime::Graph>& sub_graph,
                              Graph& graph, Node& node,
                              const logging::Logger& logger) {
  attr_proto.Clear();
  LOAD_STR_FROM_ORT_FORMAT(attr_proto, name, fbs_attr.name());
  LOAD_STR_FROM_ORT_FORMAT(attr_proto, doc_string, fbs_attr.doc_string());

  auto type = static_cast<AttributeProto_AttributeType>(fbs_attr.type());
  attr_proto.set_type(type);
  switch (type) {
    case AttributeProto_AttributeType_FLOAT: {
      attr_proto.set_f(fbs_attr.f());
    } break;
    case AttributeProto_AttributeType_INT: {
      attr_proto.set_i(fbs_attr.i());
    } break;
    case AttributeProto_AttributeType_STRING: {
      auto fbs_str = fbs_attr.s();
      ORT_RETURN_IF(nullptr == fbs_str, "Null string attribute. Invalid ORT format model.");
      attr_proto.set_s(fbs_str->str());
    } break;
    case AttributeProto_AttributeType_TENSOR: {
      auto fbs_tensor = fbs_attr.t();
      ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor attribute. Invalid ORT format model.");
      ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *attr_proto.mutable_t()));
    } break;
    case AttributeProto_AttributeType_GRAPH: {
      // If the attribute type is a graph, we will create an empty graph in attr_proto so that the ONNX checker
      // is happy in a full build, and deserialize the ORT Graph instance into the 'graph' param.
      auto fbs_graph = fbs_attr.g();
      ORT_RETURN_IF(nullptr == fbs_graph, "Null graph attribute. Invalid ORT format model.");
      attr_proto.mutable_g()->set_name("Empty graph proto from deserialization of ORT format model");
      ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, graph, node, logger, sub_graph));
    } break;
    case AttributeProto_AttributeType_FLOATS: {
      auto fbs_floats = fbs_attr.floats();
      ORT_RETURN_IF(nullptr == fbs_floats, "Null floats attribute. Invalid ORT format model.");
      auto floats = attr_proto.mutable_floats();
      floats->Reserve(fbs_floats->size());
      floats->Add(fbs_floats->cbegin(), fbs_floats->cend());
    } break;
    case AttributeProto_AttributeType_INTS: {
      auto fbs_ints = fbs_attr.ints();
      ORT_RETURN_IF(nullptr == fbs_ints, "Null ints attribute. Invalid ORT format model.");
      auto* ints = attr_proto.mutable_ints();
      ints->Reserve(fbs_ints->size());
      ints->Add(fbs_ints->cbegin(), fbs_ints->cend());
    } break;
    case AttributeProto_AttributeType_STRINGS: {
      auto fbs_strings = fbs_attr.strings();
      ORT_RETURN_IF(nullptr == fbs_strings, "Null strings attribute. Invalid ORT format model.");
      auto* strings = attr_proto.mutable_strings();
      strings->Reserve(fbs_strings->size());
      for (const auto* fbs_str : *fbs_strings) {
        ORT_RETURN_IF(nullptr == fbs_str, "Null string in strings attribute. Invalid ORT format model.");
        strings->Add(fbs_str->str());
      }
    } break;
    case AttributeProto_AttributeType_TENSORS: {
      auto fbs_tensors = fbs_attr.tensors();
      ORT_RETURN_IF(nullptr == fbs_tensors, "Null tensors attribute. Invalid ORT format model.");
      auto* tensors = attr_proto.mutable_tensors();
      tensors->Reserve(fbs_tensors->size());
      for (const auto* fbs_tensor : *fbs_tensors) {
        ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor in tensors attribute. Invalid ORT format model.");
        ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *tensors->Add()));
      }
    } break;

    default:
      break;
  }

  return Status::OK();
}

#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
