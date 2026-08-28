// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

// These tests require direct access to both real ORT internals (Model, Graph, GraphViewer)
// and QNN EP builder internals (QnnModelWrapper, QnnTensorWrapper). This is only possible
// when QNN EP is built as a static library, because the shared library build redefines
// ORT types as opaque wrappers in provider_api.h / provider_wrappedtypes.h.
#if !defined(ORT_MINIMAL_BUILD) && BUILD_QNN_EP_STATIC_LIB

#include "core/graph/model.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

// Helper to create a minimal QnnModelWrapper for unit testing.
// AddTensorWrapper does not invoke any QNN SDK functions, so we can use
// null handles and a zeroed-out interface struct.
struct QnnModelWrapperTestContext {
  std::unique_ptr<onnxruntime::Model> model;
  std::unique_ptr<GraphViewer> graph_viewer;
  QNN_INTERFACE_VER_TYPE qnn_interface;
  Qnn_BackendHandle_t backend_handle;
  std::unordered_map<std::string, size_t> input_index_map;
  std::unordered_map<std::string, size_t> output_index_map;

  QnnModelWrapperTestContext() : qnn_interface(QNN_INTERFACE_VER_TYPE_INIT),
                                 backend_handle(nullptr) {
    model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
    Graph& graph = model->MainGraph();
    graph_viewer = std::make_unique<GraphViewer>(graph);
  }

  std::unique_ptr<QnnModelWrapper> CreateWrapper(const ModelSettings& settings) {
    return std::make_unique<QnnModelWrapper>(
        *graph_viewer,
        DefaultLoggingManager().DefaultLogger(),
        qnn_interface,
        backend_handle,
        input_index_map,
        output_index_map,
        QnnBackendType::HTP,
        settings);
  }
};

}  // namespace

// Verifies that when htp_shared_memory is disabled (default), the mem type of a
// graph input tensor remains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphInput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is enabled, a graph input tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphInput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, a graph output tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, an intermediate (native) tensor
// that is neither a graph input nor output retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_IntermediateTensor_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  // "intermediate0" is NOT in input_index_map or output_index_map.

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("intermediate0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("intermediate0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is disabled, a graph output tensor
// retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphOutput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that both graph input and output tensors get MEMHANDLE when
// htp_shared_memory is enabled, within the same wrapper instance.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_BothInputAndOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper input_tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  QnnTensorWrapper output_tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(input_tensor)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(output_tensor)));

  const auto& stored_input = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored_input.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);

  const auto& stored_output = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored_output.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that adding a duplicate tensor (same name) returns true
// and does not overwrite the existing entry.
TEST(QnnModelWrapperTest, AddTensorWrapper_DuplicateTensor_ReturnsTrueWithoutOverwrite) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor1("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor1)));

  // Attempt to add another tensor with the same name
  QnnTensorWrapper tensor2("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                           QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 112, 112});
  EXPECT_TRUE(wrapper->AddTensorWrapper(std::move(tensor2)));

  // Should still have the original data type
  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(stored.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// Verifies that adding a tensor with an empty name returns false.
TEST(QnnModelWrapperTest, AddTensorWrapper_EmptyName_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;

  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  EXPECT_FALSE(wrapper->AddTensorWrapper(std::move(tensor)));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && BUILD_QNN_EP_STATIC_LIB
