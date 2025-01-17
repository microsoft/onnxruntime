// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <filesystem>

#include "core/common/flatbuffers.h"

#include "core/common/status.h"
#include "core/graph/ort_format_load_options.h"
#include "core/framework/tensor.h"

namespace ONNX_NAMESPACE {
class AttributeProto;
class TensorProto;

#if !defined(DISABLE_SPARSE_TENSORS)
class SparseTensorProto;
#endif  // !defined(DISABLE_SPARSE_TENSORS)
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

class Graph;
class Node;
class Path;

namespace logging {
class Logger;
}

namespace fbs {
struct Attribute;
struct Tensor;

#if !defined(DISABLE_SPARSE_TENSORS)
struct SparseTensor;
#endif  // !defined(DISABLE_SPARSE_TENSORS)

namespace utils {

/// <summary>
/// Delegate to write initializer data to an external file/buffer.
/// Data should be aligned to an appropriate boundary for the data type and or any potential mmap'd usage of the file.
/// `data_type` is value returned by TensorProto::data_type() and is a value from onnx::TensorTypeProto_DataType.
/// The function is not called for onnx::TensorTypeProto_DataType_STRING.
/// The function should set `offset` to the start of the data in the external file/buffer.
/// </summary>
using ExternalDataWriter = std::function<Status(int32_t data_type, gsl::span<const uint8_t> bytes, uint64_t& offset)>;

// inverse to ExternalDataWriter.
// The reader should write bytes to the output_buffer which is pre-allocated with the correct size.
using ExternalDataReader = std::function<Status(uint64_t offset, gsl::span<uint8_t> output_buffer)>;

/// <summary>
/// Minimum number of bytes for data to be written as external data.
/// </summary>
/// <remarks>arbitrary choice to keep small values local. adjust as needed. consider if it needs to be configurable.
/// </remarks>
constexpr uint32_t kMinimumSizeForExternalData = 64;

/// <summary>
/// Save an initializer to an ORT format flatbuffer.
/// </summary>
/// <param name="builder">Builder to write initializer with.</param>
/// <param name="initializer">Initializer to serialize</param>
/// <param name="model_path">Model path. Used if TensorProto has external data.</param>
/// <param name="fbs_tensor">Tensor in flatbuffer.</param>
/// <param name="external_writer">Optional delegate to write the initializer data to an external file
/// if the initializer contains kMinimumSizeForExternalData bytes or more, and not string data.</param>
Status SaveInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::TensorProto& initializer,
    const std::filesystem::path& model_path, flatbuffers::Offset<fbs::Tensor>& fbs_tensor,
    const ExternalDataWriter& external_writer = nullptr);

#if !defined(DISABLE_SPARSE_TENSORS)
Status SaveSparseInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::SparseTensorProto& initializer,
    const std::filesystem::path& model_path, flatbuffers::Offset<fbs::SparseTensor>& fbs_sparse_tensor);
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Convert a given AttributeProto into fbs::Attribute
// Note, we current do not support graphs, and sparse_tensor(s)
//       If the attribute type is a graph, we need to use the supplied Graph instance,
//       instead of the GraphProto in attr_proto
Status SaveAttributeOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::AttributeProto& attr_proto,
    flatbuffers::Offset<fbs::Attribute>& fbs_attr, const std::filesystem::path& model_path,
    const onnxruntime::Graph* subgraph);

/// <summary>
/// Load an initializer from an ORT format flatbuffer.
/// </summary>
/// <param name="fbs_tensor">Flatbuffer Tensor</param>
/// <param name="initializer">TensorProto to load data into</param>
/// <param name="load_options">ORT format load options</param>
/// <param name="external_data_reader">Optional delegate to read from external data file.</param>
/// <returns>Status</returns>
Status LoadInitializerOrtFormat(const fbs::Tensor& fbs_tensor,
                                ONNX_NAMESPACE::TensorProto& initializer,
                                const OrtFormatLoadOptions& load_options,
                                const ExternalDataReader& external_data_reader = nullptr);

#if !defined(DISABLE_SPARSE_TENSORS)
Status LoadSparseInitializerOrtFormat(const fbs::SparseTensor& fbs_sparse_tensor,
                                      ONNX_NAMESPACE::SparseTensorProto& initializer,
                                      const OrtFormatLoadOptions& load_options);
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Load a give fbs::Attribute into AttributeProto
// Note, If the attribute type is a graph, we will leave an empty graph in attr_proto,
//       and set the deserialized Graph to the param graph
Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                              ONNX_NAMESPACE::AttributeProto& attr_proto,
                              std::unique_ptr<onnxruntime::Graph>& sub_graph,
                              onnxruntime::Graph& graph, onnxruntime::Node& node,
                              const OrtFormatLoadOptions& load_options,
                              const logging::Logger& logger);

#ifdef ENABLE_TRAINING_APIS

/// @brief Save an ORT Tensor to a flatbuffer tensor
/// @param[in] tensor_name Name of the tensor
/// @param[in] ort_tensor ORT tensor to serialize to a flatbuffer tensor
/// @param[in] builder flatbuffer builder to use for creating the flatbuffer tensor
/// @param[out] fbs_tensor flatbuffer tensor to serialize the ORT tensor to
/// @param[out] external_data_writer Optional delegate to write the tensor data to an external file
/// @return Status indicating success or providing error information
Status SaveOrtTensorOrtFormat(
    const std::string& tensor_name, const onnxruntime::Tensor& ort_tensor,
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<fbs::Tensor>& fbs_tensor,
    ExternalDataWriter external_data_writer = nullptr);

/// @brief Load an ORT tensor from a flatbuffer tensor
/// @param[in] fbs_tensor flatbuffer tensor to load the ORT tensor from
/// @param[in] allocator Allocator to use for creating the ORT tensor
/// @param[out] tensor_name Name of the tensor
/// @param[out] ort_tensor ORT tensor to load the flatbuffer tensor into
/// @param[in] external_data_reader Optional delegate to read from an external data file
/// @return Status indicating success or providing error information
Status LoadOrtTensorOrtFormat(const fbs::Tensor& fbs_tensor, const AllocatorPtr allocator,
                              std::string& tensor_name, onnxruntime::Tensor& ort_tensor,
                              const ExternalDataReader& external_data_reader = nullptr);

#endif

}  // namespace utils
}  // namespace fbs
}  // namespace onnxruntime
