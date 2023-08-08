// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/framework/ort_value.h"
#include "core/graph/model.h"

#include "core/platform/env.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/optimizer/test_optimizer_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

void RandomFillFloatVector(const TensorShapeVector& shape, std::vector<float>& data) {
  static RandomValueGenerator random{1234};
  data = random.Gaussian<float>(shape, 0.0f, 0.25f);
}

void RandomFillHalfVector(const TensorShapeVector& shape, std::vector<MLFloat16>& data) {
  std::vector<float> data_float(TensorShape(shape).Size());
  RandomFillFloatVector(shape, data_float);
  std::transform(data_float.begin(), data_float.end(), data.begin(),
                 [](float value) { return MLFloat16(value); });
}

void RandomMasks(int64_t batch, int64_t sequence_length, std::vector<int64_t>& data) {
  static RandomValueGenerator random{5678};
  const std::vector<int64_t> num_count_to_random{batch};
  std::vector<int64_t> random_seq_lens = random.Uniform<int64_t>(num_count_to_random, 0, sequence_length);
  data.resize(batch * sequence_length);  // fill with zeros first.
  for (int64_t i = 0; i < batch; ++i) {
    for (int64_t j = 0; j < sequence_length; ++j) {
      if (j > random_seq_lens[i]) {
        break;
      }

      data[i * sequence_length + j] = 1;
    }
  }
}

void RunModelWithData(const PathString& model_uri, const std::string session_log_id,
                      const std::string& provider_type, const InputContainer& input_container,
                      const std::vector<std::string>& output_names,
                      std::vector<OrtValue>& run_results) {
  SessionOptions so;
  // we don't want any transformation here.
  so.graph_optimization_level = TransformerLevel::Default;
  so.session_logid = session_log_id;

  InferenceSession session_object{so, GetEnvironment()};
  std::unique_ptr<IExecutionProvider> execution_provider;
  if (provider_type == onnxruntime::kCpuExecutionProvider)
    execution_provider = DefaultCpuExecutionProvider();
  else if (provider_type == onnxruntime::kCudaExecutionProvider)
    execution_provider = DefaultCudaExecutionProvider();
  else if (provider_type == onnxruntime::kRocmExecutionProvider)
    execution_provider = DefaultRocmExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  Status st;
  ASSERT_TRUE((st = session_object.Load(model_uri)).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st.ErrorMessage();

  NameMLValMap feeds;
  input_container.ToInputMap(feeds);

  // Now run
  RunOptions run_options;
  st = session_object.Run(run_options, feeds, output_names, &run_results);

  ASSERT_TRUE(st.IsOK()) << "RunModelWithData run graph failed with error: " << st.ErrorMessage();
}

}  // namespace test
}  // namespace onnxruntime
