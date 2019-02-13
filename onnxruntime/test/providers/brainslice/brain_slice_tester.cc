#include "test/providers/brainslice/brain_slice_tester.h"
#include "core/session/inference_session.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void CheckTensor(const Tensor& expected_tensor, const Tensor& output_tensor, double rtol, double atol) {
  ORT_ENFORCE(expected_tensor.Shape() == output_tensor.Shape(),
                      "Expected output shape [" + expected_tensor.Shape().ToString() +
                          "] did not match run output shape [" +
                          output_tensor.Shape().ToString() + "]");

  ASSERT_TRUE(expected_tensor.DataType() == DataTypeImpl::GetType<float>()) << "Compare with non float number is not supported yet. ";
  auto expected = expected_tensor.Data<float>();
  auto output = output_tensor.Data<float>();
  for (auto i = 0; i < expected_tensor.Shape().Size(); ++i) {
    double diff = fabs(expected[i] - output[i]);
    ASSERT_TRUE(diff <= (atol + rtol * expected[i]));
  }
}

void BrainSliceTestor::CompareWithCPU(double rtol, double atol) {
#ifndef NDEBUG
  run_called_ = true;
#endif
  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  InferenceSession cpu_session_object{so};

  // first run with cpu
  std::stringstream s1;
  p_model->ToProto().SerializeToOstream(&s1);
  status = cpu_session_object.Load(s1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = cpu_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  std::vector<MLValue> cpu_fetches;
  status = cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  // run with fpga
  auto brainslice_execution_provider = DefaultBrainSliceExecutionProvider();
  InferenceSession bs_session_object{so};
  EXPECT_TRUE(bs_session_object.RegisterExecutionProvider(std::move(brainslice_execution_provider)).IsOK());

  auto p_model_2 = BuildGraph();
  auto& graph_2 = p_model_2->MainGraph();

  status = graph_2.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  std::stringstream s2;
  p_model_2->ToProto().SerializeToOstream(&s2);

  status = bs_session_object.Load(s2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = bs_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  std::vector<MLValue> bs_fetches;
  status = bs_session_object.Run(run_options, feeds, output_names, &bs_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  //compare
  ASSERT_TRUE(cpu_fetches.size() == bs_fetches.size());
  for (auto i = 0; i < cpu_fetches.size(); i++) {
    if (cpu_fetches[i].IsTensor() && bs_fetches[i].IsTensor()) {
      CheckTensor(cpu_fetches[i].Get<Tensor>(), bs_fetches[i].Get<Tensor>(), rtol, atol);
	}
  }
}

}
}