// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "asserts.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "orttraining/core/session/tensor_helper.h"
#include "test/util/include/default_providers.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"

namespace onnxruntime {
#ifdef USE_CUDA
void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count);
#endif

namespace test {

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

// Create ML value.
OrtValue CreateTensorValue(const std::vector<int64_t>& shape, const std::vector<float>& initializer, const bool allocate_on_gpu) {
#ifdef USE_CUDA
  auto cpu_allocator = allocate_on_gpu ? DefaultCudaExecutionProvider()->GetAllocator(0, OrtMemTypeDefault) : TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
#else
  ORT_ENFORCE(allocate_on_gpu != true);
  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
#endif
  auto element_type = onnxruntime::DataTypeImpl::GetType<float>();

  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      onnxruntime::TensorShape(shape),
      cpu_allocator);

  if (!allocate_on_gpu) {
    memcpy(p_tensor->MutableData<float>(), initializer.data(), initializer.size() * sizeof(float));
  } else {
#ifdef USE_CUDA
    cudaMemcpy_HostToDevice(p_tensor->MutableData<float>(), initializer.data(), initializer.size() * sizeof(float));
#else
    ORT_THROW("Cannot use CUDA function when ORT is not built with CUDA.");
#endif
  }

  OrtValue value;
  value.Init(p_tensor.release(),
             onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
             onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

  return value;
}

std::vector<float> CreateVector(const OrtValue& value) {
  const onnxruntime::Tensor& tensor = value.Get<onnxruntime::Tensor>();
  std::vector<float> vector(tensor.Shape().Size());
  memcpy(vector.data(), tensor.Data<float>(), tensor.Shape().Size() * sizeof(float));
  return vector;
}

void CreateFakeGraph(onnxruntime::Graph& graph) {
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg x_def("X", &float_tensor);
  onnxruntime::NodeArg y_def("Y", &float_tensor);

  auto& node = graph.AddNode("MyNode", "Identity", "A identity operator", {&x_def}, {&y_def});
  node.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);

  ASSERT_TRUE(graph.Resolve().IsOK());
}

void InitializeSession(onnxruntime::InferenceSession& session, onnxruntime::Model& model) {
  // Construct an un-initialized session object.

  // Convert the format of model so that the session can load it.
  std::stringstream buffer;
  model.ToProto().SerializeToOstream(&buffer);

  // Load model.
  ASSERT_STATUS_OK(session.Load(buffer));

// Initialize the session.
#if defined(USE_CUDA)
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#endif
  ASSERT_STATUS_OK(session.Initialize());
}

void CompareVector(const std::vector<float>& result, const std::vector<float>& expected) {
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result.at(i), expected.at(i));
  }
}

TEST(PipelineParallel, FloatTensorSlice2d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  std::vector<float> vector{1.0f, -2.0f, -3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f};
  auto value = CreateTensorValue({2, 4}, vector, false);

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  // # of Slices = 2

  auto sliced_vector = CreateVector(training::SliceTensor(value, 1, 0, 2, session));
  CompareVector(sliced_vector, {5.0f, 6.0f, 7.0f, 8.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 0, 0, 2, session));
  CompareVector(sliced_vector, {1.0, -2.0f, -3.0f, 4.0f});

  // # of Slices = 4

  sliced_vector = CreateVector(training::SliceTensor(value, 0, 1, 4, session));
  CompareVector(sliced_vector, {1.0f, 5.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 1, 4, session));
  CompareVector(sliced_vector, {-2.0f, 6.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 2, 1, 4, session));
  CompareVector(sliced_vector, {-3.0f, 7.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 3, 1, 4, session));
  CompareVector(sliced_vector, {4.0f, 8.0f});
}

TEST(PipelineParallel, FloatTensorSlice1d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  std::vector<float> vector{1.0f, -2.0f, -3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f};
  auto value = CreateTensorValue({8}, vector, false);

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  // # of Slices = 1

  auto sliced_vector = CreateVector(training::SliceTensor(value, 0, 0, 1, session));
  CompareVector(sliced_vector, {1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  // # of Slices = 2

  sliced_vector = CreateVector(training::SliceTensor(value, 0, 0, 2, session));
  CompareVector(sliced_vector, {1.0f, -2.0f, -3.0f, 4.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 0, 2, session));
  CompareVector(sliced_vector, {5.0f, 6.0f, 7.0f, 8.0f});

  // # of Slices = 4

  sliced_vector = CreateVector(training::SliceTensor(value, 0, 0, 4, session));
  CompareVector(sliced_vector, {1.0f, -2.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 0, 4, session));
  CompareVector(sliced_vector, {-3.0f, 4.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 2, 0, 4, session));
  CompareVector(sliced_vector, {5.0f, 6.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 3, 0, 4, session));
  CompareVector(sliced_vector, {7.0f, 8.0f});

  // # of Slices = 8

  sliced_vector = CreateVector(training::SliceTensor(value, 0, 0, 8, session));
  CompareVector(sliced_vector, {1.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 0, 8, session));
  CompareVector(sliced_vector, {-2.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 2, 0, 8, session));
  CompareVector(sliced_vector, {-3.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 3, 0, 8, session));
  CompareVector(sliced_vector, {4.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 4, 0, 8, session));
  CompareVector(sliced_vector, {5.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 5, 0, 8, session));
  CompareVector(sliced_vector, {6.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 6, 0, 8, session));
  CompareVector(sliced_vector, {7.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 7, 0, 8, session));
  CompareVector(sliced_vector, {8.0f});
}

TEST(PipelineParallel, FloatTensorSlice3d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  std::vector<float> vector{1.0f, -2.0f, -3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f};
  auto value = CreateTensorValue({2, 2, 2}, vector, false);

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  // # of Slices = 2

  auto sliced_vector = CreateVector(training::SliceTensor(value, 0, 1, 2, session));
  CompareVector(sliced_vector, {1.0f, -2.0f, 5.0f, 6.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 1, 2, session));
  CompareVector(sliced_vector, {-3.0f, 4.0f, 7.0f, 8.0f});
}

#ifdef USE_CUDA
TEST(PipelineParallel, FloatTensorSlice3dGpu) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  std::vector<float> vector{1.0f, -2.0f, -3.0f, 4.0f,
                            5.0f, 6.0f, 7.0f, 8.0f};
  auto value = CreateTensorValue({2, 2, 2}, vector, true);

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  // # of Slices = 2

  auto sliced_vector = CreateVector(training::SliceTensor(value, 0, 1, 2, session));
  CompareVector(sliced_vector, {1.0f, -2.0f, 5.0f, 6.0f});

  sliced_vector = CreateVector(training::SliceTensor(value, 1, 1, 2, session));
  CompareVector(sliced_vector, {-3.0f, 4.0f, 7.0f, 8.0f});
}
#endif

TEST(PipelineParallel, FloatTensorConcat1d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  std::vector<float> vector0{0.f, 1.f};
  std::vector<float> vector1{2.f, 3.f};
  std::vector<float> vector2{4.f, 5.f};

  std::vector<OrtValue> values;
  for (auto& vector : {vector0, vector1, vector2}) {
    auto value = CreateTensorValue({2}, vector, false);
    values.push_back(value);
  }

  auto result_vector = CreateVector(training::ConcatenateTensors(values, 0, session));
  CompareVector(result_vector, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
}

TEST(PipelineParallel, FloatTensorConcat2d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  std::vector<float> vector0{0.f, 1.f, 2.f, 3.f};
  std::vector<float> vector1{4.f, 5.f, 6.f, 7.f};
  std::vector<float> vector2{8.f, 9.f, 10.f, 11.f};

  std::vector<OrtValue> values;
  for (auto& vector : {vector0, vector1, vector2}) {
    auto value = CreateTensorValue({2, 2}, vector, false);
    values.push_back(value);
  }

  auto result_vector_axis_0 = CreateVector(training::ConcatenateTensors(values, 0, session));
  CompareVector(result_vector_axis_0, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});

  auto result_vector_axis_1 = CreateVector(training::ConcatenateTensors(values, 1, session));
  CompareVector(result_vector_axis_1, {0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 2.f, 3.f, 6.f, 7.f, 10.f, 11.f});
}

TEST(PipelineParallel, FloatTensorConcat3d) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  std::vector<float> vector0{0.f, 1.f, 2.f, 3.f};
  std::vector<float> vector1{4.f, 5.f, 6.f, 7.f};
  std::vector<float> vector2{8.f, 9.f, 10.f, 11.f};

  std::vector<OrtValue> values;
  for (auto& vector : {vector0, vector1, vector2}) {
    auto value = CreateTensorValue({2, 1, 2}, vector, false);
    values.push_back(value);
  }

  auto result_vector_axis_0 = CreateVector(training::ConcatenateTensors(values, 0, session));
  CompareVector(result_vector_axis_0, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});

  auto result_vector_axis_1 = CreateVector(training::ConcatenateTensors(values, 1, session));
  CompareVector(result_vector_axis_1, {0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 2.f, 3.f, 6.f, 7.f, 10.f, 11.f});

  auto result_vector_axis_2 = CreateVector(training::ConcatenateTensors(values, 2, session));
  CompareVector(result_vector_axis_2, {0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 2.f, 3.f, 6.f, 7.f, 10.f, 11.f});
}

TEST(PipelineParallel, FloatTensorConcat3dMore) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  std::vector<float> vector0{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> vector1{8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  std::vector<float> vector2{16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f};

  std::vector<OrtValue> values;
  for (auto& vector : {vector0, vector1, vector2}) {
    auto value = CreateTensorValue({2, 2, 2}, vector, false);
    values.push_back(value);
  }

  auto result_vector_axis_0 = CreateVector(training::ConcatenateTensors(values, 0, session));
  CompareVector(result_vector_axis_0, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                                       8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
                                       16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f});

  auto result_vector_axis_1 = CreateVector(training::ConcatenateTensors(values, 1, session));
  CompareVector(result_vector_axis_1, {0.f, 1.f, 2.f, 3.f, 8.f, 9.f, 10.f, 11.f, 16.f, 17.f, 18.f, 19.f,
                                       4.f, 5.f, 6.f, 7.f, 12.f, 13.f, 14.f, 15.f, 20.f, 21.f, 22.f, 23.f});

  auto result_vector_axis_2 = CreateVector(training::ConcatenateTensors(values, 2, session));
  CompareVector(result_vector_axis_2, {0.f, 1.f, 8.f, 9.f, 16.f, 17.f, 2.f, 3.f, 10.f, 11.f, 18.f, 19.f,
                                       4.f, 5.f, 12.f, 13.f, 20.f, 21.f, 6.f, 7.f, 14.f, 15.f, 22.f, 23.f});
}

#ifdef USE_CUDA
TEST(PipelineParallel, FloatTensorConcat1dGpu) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());

  CreateFakeGraph(model->MainGraph());

  onnxruntime::SessionOptions so;
  onnxruntime::InferenceSession session(so, GetEnvironment());
  InitializeSession(session, *model);

  std::vector<float> vector0{0.f, 1.f};
  std::vector<float> vector1{2.f, 3.f};
  std::vector<float> vector2{4.f, 5.f};

  std::vector<OrtValue> values;
  for (auto& vector : {vector0, vector1, vector2}) {
    auto value = CreateTensorValue({2}, vector, true);
    values.push_back(value);
  }

  auto result_vector = CreateVector(training::ConcatenateTensors(values, 0, session));
  CompareVector(result_vector, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
}
#endif

}  // namespace test
}  // namespace onnxruntime
