// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_training_cxx_api.h"

#include "orttraining/training_api/checkpoint.h"

#include "orttraining/test/training_api/core/data_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime::training::test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")
#define ORT_FORMAT_MODEL_FOLDER ORT_TSTR("testdata/training_api/ort_format/")

TEST(TrainingCApiTest, SaveCheckpoint) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(), checkpoint_state, model_uri);

  auto test_dir = ORT_TSTR("save_checkpoint_dir");
  if (Env::Default().FolderExists(test_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(test_dir).IsOK());
  }
  onnxruntime::test::TemporaryDirectory tmp_dir{test_dir};
  PathString checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("new_checkpoint.ckpt"))};

  Ort::CheckpointState::SaveCheckpoint(checkpoint_state, checkpoint_path);

  Ort::CheckpointState new_checkpoint_state = Ort::CheckpointState::LoadCheckpoint(checkpoint_path);
  Ort::TrainingSession new_training_session = Ort::TrainingSession(env, Ort::SessionOptions(),
                                                                   new_checkpoint_state, model_uri);
}

TEST(TrainingCApiTest, LoadCheckpointFromBuffer) {
  Ort::Env env;
  size_t num_bytes = 0;
  PathString checkpoint_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(Env::Default().GetFileLength(checkpoint_path.c_str(), num_bytes));
  std::vector<uint8_t> checkpoint_bytes(num_bytes);

  std::ifstream bytes_stream(checkpoint_path, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(checkpoint_bytes.data()), num_bytes);

  ASSERT_TRUE(bytes_stream);

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpointFromBuffer(checkpoint_bytes);

  auto test_dir = ORT_TSTR("save_checkpoint_dir");
  if (Env::Default().FolderExists(test_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(test_dir).IsOK());
  }
  onnxruntime::test::TemporaryDirectory tmp_dir{test_dir};
  PathString new_checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("new_checkpoint.ckpt"))};

  Ort::CheckpointState::SaveCheckpoint(checkpoint_state, new_checkpoint_path);

  Ort::CheckpointState new_checkpoint_state = Ort::CheckpointState::LoadCheckpoint(new_checkpoint_path);
}

TEST(TrainingCApiTest, AddIntProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  int64_t value = 365 * 24;

  checkpoint_state.AddProperty("hours in a year", value);

  auto property = checkpoint_state.GetProperty("hours in a year");

  ASSERT_EQ(std::get<int64_t>(property), value);
}

TEST(TrainingCApiTest, AddFloatProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  float value = 3.14f;

  checkpoint_state.AddProperty("pi", value);

  auto property = checkpoint_state.GetProperty("pi");

  ASSERT_EQ(std::get<float>(property), value);
}

TEST(TrainingCApiTest, AddStringProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  std::string value("onnxruntime");

  checkpoint_state.AddProperty("framework", value);

  auto property = checkpoint_state.GetProperty("framework");

  ASSERT_EQ(std::get<std::string>(property), value);
}

TEST(TrainingCApiTest, InputNames) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(), checkpoint_state, model_uri);

  const auto input_names = training_session.InputNames(true);
  ASSERT_EQ(input_names.size(), 2U);
  ASSERT_EQ(input_names.front(), "input-0");
  ASSERT_EQ(input_names.back(), "labels");
}

TEST(TrainingCApiTest, OutputNames) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(), checkpoint_state, model_uri);

  const auto output_names = training_session.OutputNames(true);
  ASSERT_EQ(output_names.size(), 1U);
  ASSERT_EQ(output_names.front(), "onnx::loss::21273");
}

TEST(TrainingCApiTest, ToBuffer) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(), checkpoint_state, model_uri);

  Ort::Value buffer = training_session.ToBuffer(true);

  ASSERT_TRUE(buffer.IsTensor());
  auto tensor_info = buffer.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  ASSERT_EQ(shape.size(), 1U);
  ASSERT_EQ(shape.front(), static_cast<int64_t>(397510));

  buffer = training_session.ToBuffer(false);

  ASSERT_TRUE(buffer.IsTensor());
  tensor_info = buffer.GetTensorTypeAndShapeInfo();
  shape = tensor_info.GetShape();
  ASSERT_EQ(shape.size(), 1U);
  ASSERT_EQ(shape.front(), static_cast<int64_t>(397510));
}

TEST(TrainingCApiTest, FromBuffer) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(), checkpoint_state, model_uri);

  OrtValue* buffer_impl = std::make_unique<OrtValue>().release();
  GenerateRandomInput(std::array<int64_t, 1>{397510}, *buffer_impl);

  Ort::Value buffer(buffer_impl);

  training_session.FromBuffer(buffer);
}

TEST(TrainingCApiTest, RegisterCustomOps) {
  auto model_uri = MODEL_FOLDER "custom_ops/training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "custom_ops/checkpoint");
  Ort::SessionOptions session_options;

#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.so"));
#endif

#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  Ort::TrainingSession training_session = Ort::TrainingSession(env, session_options, checkpoint_state, model_uri);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<int64_t> input_shape{3, 5};

  std::vector<float> x{1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                       6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                       11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  ort_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, x.data(),
                                                   x.size() * sizeof(float),
                                                   input_shape.data(), input_shape.size(),
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

  std::vector<float> y{15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                       10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                       5.5f, 4.4f, 3.3f, 2.2f, 1.1f};
  ort_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, y.data(),
                                                   y.size() * sizeof(float),
                                                   input_shape.data(), input_shape.size(),
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

  std::vector<int64_t> labels{0, 8, 4};
  std::vector<int64_t> labels_shape{3};
  ort_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels.data(),
                                                   labels.size() * sizeof(int64_t),
                                                   labels_shape.data(), labels_shape.size(),
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

  auto loss = training_session.TrainStep(ort_inputs);
  ASSERT_EQ(loss.size(), 1U);
  ASSERT_TRUE(loss.front().IsTensor());
}

TEST(TrainingCApiTest, LoadModelsAndCreateSession) {
  auto model_path = MODEL_FOLDER "training_model.onnx";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env,
                                                               Ort::SessionOptions(),
                                                               checkpoint_state,
                                                               model_path);
}

TEST(TrainingCApiTest, LoadModelsAndCreateSession_ORTFormat) {
  auto train_model_path = ORT_FORMAT_MODEL_FOLDER "training_model.ort";
  auto eval_train_model_path = ORT_FORMAT_MODEL_FOLDER "eval_model.ort";
  auto optimizer_model_path = ORT_FORMAT_MODEL_FOLDER "optimizer_model.ort";

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(ORT_FORMAT_MODEL_FOLDER "checkpoint");
  Ort::TrainingSession training_session = Ort::TrainingSession(env,
                                                               Ort::SessionOptions(),
                                                               checkpoint_state,
                                                               train_model_path,
                                                               eval_train_model_path,
                                                               optimizer_model_path);
}

TEST(TrainingCApiTest, LoadONNXModelsFromBuffer) {
  auto model_path = MODEL_FOLDER "training_model.onnx";
  size_t model_data_len = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(model_path, model_data_len));
  std::vector<uint8_t> train_model_data(model_data_len);
  std::ifstream bytes_stream(model_path, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(train_model_data.data()), model_data_len);
  ASSERT_TRUE(train_model_data.size() == model_data_len);

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(env,
                                                               Ort::SessionOptions(),
                                                               checkpoint_state,
                                                               train_model_data);
}

TEST(TrainingCApiTest, LoadORTFormatModelsFromBuffer) {
  auto train_model_path = ORT_FORMAT_MODEL_FOLDER "training_model.ort";
  auto eval_model_path = ORT_FORMAT_MODEL_FOLDER "eval_model.ort";
  auto optimizer_model_path = ORT_FORMAT_MODEL_FOLDER "optimizer_model.ort";
  size_t model_data_len = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(train_model_path, model_data_len));
  std::vector<uint8_t> train_model_data(model_data_len);
  {
    std::ifstream bytes_stream(train_model_path, std::ifstream::in | std::ifstream::binary);
    bytes_stream.read(reinterpret_cast<char*>(train_model_data.data()), model_data_len);
    ASSERT_TRUE(train_model_data.size() == model_data_len);
  }

  model_data_len = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(eval_model_path, model_data_len));
  std::vector<uint8_t> eval_model_data(model_data_len);
  {
    std::ifstream bytes_stream(eval_model_path, std::ifstream::in | std::ifstream::binary);
    bytes_stream.read(reinterpret_cast<char*>(eval_model_data.data()), model_data_len);
    ASSERT_TRUE(eval_model_data.size() == model_data_len);
  }

  model_data_len = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(optimizer_model_path, model_data_len));
  std::vector<uint8_t> optimizer_model_data(model_data_len);
  {
    std::ifstream bytes_stream(optimizer_model_path, std::ifstream::in | std::ifstream::binary);
    bytes_stream.read(reinterpret_cast<char*>(optimizer_model_data.data()), model_data_len);
    ASSERT_TRUE(optimizer_model_data.size() == model_data_len);
  }

  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(ORT_FORMAT_MODEL_FOLDER "checkpoint");
  Ort::TrainingSession training_session = Ort::TrainingSession(env, Ort::SessionOptions(),
                                                               checkpoint_state, train_model_data,
                                                               eval_model_data, optimizer_model_data);
}

TEST(TrainingCApiTest, LoadModelsFromBufferThrows) {
  Ort::Env env;
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  try {
    std::vector<uint8_t> train_model_data;
    Ort::TrainingSession training_session = Ort::TrainingSession(env,
                                                                 Ort::SessionOptions(),
                                                                 checkpoint_state,
                                                                 train_model_data);
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(),
                testing::HasSubstr("Training Session Creation failed. Train model data cannot be NULL."));
  }
}
}  // namespace onnxruntime::training::test
