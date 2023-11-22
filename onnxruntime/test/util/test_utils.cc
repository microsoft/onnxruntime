// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_utils.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/ort_value.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/framework/tensorprotoutils.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {
void VerifyOutput(const std::string& output_name,
                  const Tensor& expected_tensor,
                  const Tensor& tensor,
                  float fp32_abs_err) {
  ASSERT_TRUE(SpanEq(expected_tensor.Shape().GetDims(), tensor.Shape().GetDims()));
  ASSERT_EQ(expected_tensor.GetElementType(), tensor.GetElementType());
  auto element_type = expected_tensor.GetElementType();
  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<uint32_t>(), tensor.DataAsSpan<uint32_t>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<int32_t>(), tensor.DataAsSpan<int32_t>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<int64_t>(), tensor.DataAsSpan<int64_t>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<uint8_t>(), tensor.DataAsSpan<uint8_t>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<int8_t>(), tensor.DataAsSpan<int8_t>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      EXPECT_TRUE(SpanEq(expected_tensor.DataAsSpan<bool>(), tensor.DataAsSpan<bool>()))
          << " mismatch for " << output_name;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      EXPECT_THAT(expected_tensor.DataAsSpan<float>(),
                  ::testing::Pointwise(::testing::FloatNear(fp32_abs_err), tensor.DataAsSpan<float>()));
      break;
    }
    default:
      ORT_THROW("Unhandled data type. Please add 'case' statement for ", element_type);
  }
}

static void VerifyOutputs(const std::vector<std::string>& output_names,
                          const std::vector<OrtValue>& expected_fetches,
                          const std::vector<OrtValue>& fetches,
                          const EPVerificationParams& params) {
  ASSERT_EQ(expected_fetches.size(), fetches.size());

  for (size_t i = 0, end = expected_fetches.size(); i < end; ++i) {
    auto& ltensor = expected_fetches[i].Get<Tensor>();
    auto& rtensor = fetches[i].Get<Tensor>();
    VerifyOutput(output_names[i], ltensor, rtensor, params.fp32_abs_err);
  }
}

int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type) {
  int count = 0;

  for (const auto& node : current_graph.Nodes()) {
    if (node.GetExecutionProviderType() == ep_type) {
      ++count;
    }

    if (node.ContainsSubgraph()) {
      for (const auto& entry : node.GetSubgraphs()) {
        count += CountAssignedNodes(*entry, ep_type);
      }
    }
  }

  return count;
}

void VerifyEPNodeAssignment(const Graph& graph, const std::string& provider_type,
                            ExpectedEPNodeAssignment assignment) {
  const auto provider_node_count = CountAssignedNodes(graph, provider_type);
  if (assignment == ExpectedEPNodeAssignment::All) {
    // Verify the entire graph is assigned to the EP
    ASSERT_EQ(provider_node_count, graph.NumberOfNodes()) << "Not all nodes were assigned to " << provider_type;
  } else if (assignment == ExpectedEPNodeAssignment::None) {
    // or none of the graph
    ASSERT_EQ(provider_node_count, 0) << "Some nodes were assigned to " << provider_type;
  } else {
    // or some of the graph
    ASSERT_GT(provider_node_count, 0) << "No nodes were assigned to " << provider_type;
  }
}

static gsl::span<const std::byte> GetModelBytes(ModelPathOrBytes model_path_or_bytes,
                                                std::vector<std::byte>& byte_buffer_out) {
  if (const auto* model_bytes = std::get_if<gsl::span<const std::byte>>(&model_path_or_bytes);
      model_bytes != nullptr) {
    byte_buffer_out = std::vector<std::byte>{};
    return *model_bytes;
  }

  const auto model_path = std::get<std::basic_string_view<ORTCHAR_T>>(model_path_or_bytes);

  std::vector<std::byte> byte_buffer{};
  std::ifstream stream{std::basic_string<ORTCHAR_T>{model_path},
                       std::ios::in | std::ios::binary | std::ios::ate};
  ORT_ENFORCE(stream, "Failed to open file.");
  const auto num_bytes = narrow<size_t>(stream.tellg());
  byte_buffer.resize(num_bytes);
  stream.seekg(0);
  ORT_ENFORCE(stream.read(reinterpret_cast<char*>(byte_buffer.data()), num_bytes), "Failed to read file.");

  byte_buffer_out = std::move(byte_buffer);
  return gsl::span<const std::byte>(byte_buffer_out);
}

void RunAndVerifyOutputsWithEP(ModelPathOrBytes model_path_or_bytes, std::string_view log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params,
                               const std::function<void(SessionOptions&)>& session_options_updater) {
  std::vector<std::byte> model_data_buffer{};
  const auto model_data = GetModelBytes(model_path_or_bytes, model_data_buffer);

  SessionOptions so;
  if (session_options_updater) {
    session_options_updater(so);
  }

  so.session_logid = log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  //
  // get expected output from CPU EP
  //
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());

  const auto& graph = session_object.GetGraph();
  const auto& outputs = graph.GetOutputs();

  // fetch all outputs
  std::vector<std::string> output_names;
  output_names.reserve(outputs.size());
  for (const auto* node_arg : outputs) {
    if (node_arg->Exists()) {
      output_names.push_back(node_arg->Name());
    }
  }

  std::vector<OrtValue> expected_fetches;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &expected_fetches));

  auto provider_type = execution_provider->Type();  // copy string so the std::move doesn't affect us

  //
  // get output with EP enabled
  //
  InferenceSessionWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.RegisterExecutionProvider(std::move(execution_provider)));
  ASSERT_STATUS_OK(session_object2.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object2.Initialize());

  const auto& graph2 = session_object2.GetGraph();
  ASSERT_NO_FATAL_FAILURE(VerifyEPNodeAssignment(graph2, provider_type, params.ep_node_assignment));

  // Run with EP and verify the result
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object2.Run(run_options, feeds, output_names, &fetches));
  VerifyOutputs(output_names, expected_fetches, fetches, params);

  if (params.graph_verifier) {
    (*params.graph_verifier)(graph2);
  }
}

void TestModelLoad(ModelPathOrBytes model_path_or_bytes,
                   std::unique_ptr<IExecutionProvider> execution_provider,
                   const std::function<void(const Graph&)>& check_graph) {
  std::vector<std::byte> model_data_buffer{};
  const auto model_data = GetModelBytes(model_path_or_bytes, model_data_buffer);

  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(execution_provider)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());
  if (check_graph) {
    check_graph(session_object.GetGraph());
  }
}

void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1,
                        const ONNX_NAMESPACE::TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
  auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
  for (int i = 0; i < min_dims; ++i) {
    auto dim1 = shape1->dim(i);
    auto dim2 = shape2->dim(i);
    EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
    if (dim1.has_dim_value()) {
      EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
    }
    EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
    if (dim1.has_dim_param()) {
      EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
    }
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
void SparseIndicesChecker(const ONNX_NAMESPACE::TensorProto& indices_proto, gsl::span<const int64_t> expected_indicies) {
  using namespace ONNX_NAMESPACE;
  Path model_path;
  std::vector<uint8_t> unpack_buffer;
  gsl::span<const int64_t> ind_span;
  std::vector<int64_t> converted_indices;
  TensorShape ind_shape(indices_proto.dims().data(), indices_proto.dims().size());
  const auto elements = narrow<size_t>(ind_shape.Size());
  const bool has_raw_data = indices_proto.has_raw_data();
  switch (indices_proto.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      if (has_raw_data) {
        const auto& rd = indices_proto.raw_data();
        ASSERT_EQ(rd.size(), elements * sizeof(int64_t));
        ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
        ind_span = ReinterpretAsSpan<const int64_t>(gsl::make_span(unpack_buffer));
      } else {
        ind_span = gsl::make_span(indices_proto.int64_data().data(), indices_proto.int64_data_size());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      if (has_raw_data) {
        const auto& rd = indices_proto.raw_data();
        ASSERT_EQ(rd.size(), elements * sizeof(int32_t));
        ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpack_buffer));
        converted_indices.insert(converted_indices.cend(), int32_span.begin(), int32_span.end());
      } else {
        converted_indices.insert(converted_indices.cend(), indices_proto.int32_data().cbegin(), indices_proto.int32_data().cend());
      }
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      ASSERT_TRUE(has_raw_data);
      const auto& rd = indices_proto.raw_data();
      ASSERT_EQ(rd.size(), elements * sizeof(int16_t));
      ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
      auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpack_buffer));
      converted_indices.insert(converted_indices.cend(), int16_span.begin(), int16_span.end());
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ASSERT_TRUE(has_raw_data);
      const auto& rd = indices_proto.raw_data();
      ASSERT_EQ(rd.size(), elements);
      ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
      auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpack_buffer));
      converted_indices.insert(converted_indices.cend(), int8_span.begin(), int8_span.end());
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    default:
      ASSERT_TRUE(false);
  }
  ASSERT_TRUE(SpanEq(ind_span, expected_indicies));
}

#endif  // DISABLE_SPARSE_TENSORS

}  // namespace test
}  // namespace onnxruntime
