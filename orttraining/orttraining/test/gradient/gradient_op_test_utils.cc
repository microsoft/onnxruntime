// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_op_test_utils.h"
#include "core/session/inference_session.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/graph/gradient_config.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// TODO: Refactor this so that we dont build gradient graph in every run
void GradientOpTester::Run(
    int output_index_to_use_as_loss,
    int data_index_of_output,
    ExpectResult expect_result,
    const std::string& expected_failure_string,
    const std::unordered_set<std::string>& /*excluded_provider_types*/,
    const RunOptions* run_options,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  try {
#ifndef NDEBUG
    run_called_ = true;
#endif
    fetches_.clear();

    std::unordered_map<std::string, int> extra_domain_to_version{{kMSDomain, 1}, {kOnnxDomain, 9}};
    bool cacheEnabled = cached_model_ != nullptr;
    auto p_model = !cacheEnabled ? BuildGraph(extra_domain_to_version) : cached_model_;
    auto& graph = p_model->MainGraph();

    Status status = Status::OK();
    if (!cacheEnabled) {
      if (expect_result == ExpectResult::kExpectFailure) {
        // capture possible exceptions from shape inference for invalid testcase
        try {
          status = graph.Resolve();
        } catch (const std::exception& ex) {
          status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
        }
      } else {
        status = graph.Resolve();
      }

      if (!status.IsOK()) {
        if (expect_result == ExpectResult::kExpectFailure) {
          EXPECT_TRUE(!status.IsOK());
          EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
        } else {
          LOGS_DEFAULT(ERROR) << "Resolve failed with status: " << status.ErrorMessage();
          EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
        }
      }

      // TODO: We will need finer control over both inputs and ouptuts
      // Not all inputs/outputs reqiures/have a gradient, e.g.index in gather
      std::unordered_set<std::string> weights_to_train;
      for (size_t i = 0; i < input_data_.size(); i++) {
        if (input_infos_[i].has_gradient) {
          weights_to_train.insert(input_data_[i].def_.Name());
        }
      }

      std::unordered_set<std::string> dy_values;
      for (size_t i = 0; i < output_data_.size(); i++) {
        if (output_infos_[i].has_gradient) {
          dy_values.insert(output_data_[i].def_.Name());
        }
      }

      training::GradientGraphConfiguration gradient_graph_config;
      gradient_graph_config.set_gradients_as_graph_outputs = true;
      training::GradientGraphBuilder grad_graph_builder(&graph,
                                                        dy_values,
                                                        weights_to_train,
                                                        "",
                                                        gradient_graph_config,
                                                        logging::LoggingManager::DefaultLogger());
      status = grad_graph_builder.Build();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }

    // Hookup the inputs and outputs
    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names, output_index_to_use_as_loss, data_index_of_output);

    // Add new inputs to the graph.
    std::vector<const NodeArg*> new_input_args = graph.GetInputs();  // Make a copy of existing input args.
    auto input_size = new_input_args.size();
    for (const auto& feed : feeds) {
      const auto* input_arg = graph.GetNodeArg(feed.first);
      EXPECT_TRUE(input_arg != nullptr);
      if (std::find(new_input_args.begin(), new_input_args.end(), input_arg) == new_input_args.end()) {
        new_input_args.emplace_back(input_arg);
      }
    }
    if (new_input_args.size() != input_size) {
      graph.SetInputs(new_input_args);  // By setting this, Graph::SetGraphInputsOutputs could infer the input as expected.
      graph.SetGraphResolveNeeded();
      graph.SetGraphProtoSyncNeeded();

      status = graph.Resolve();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }

    // Run the model
    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    static const std::string all_provider_types[] = {
        kCpuExecutionProvider,
        kCudaExecutionProvider,
        kRocmExecutionProvider,
        kDnnlExecutionProvider,
        kNupharExecutionProvider,
        kTensorrtExecutionProvider,
    };
    bool has_run = false;


    if (execution_providers) {
      for (auto& entry : *execution_providers) {
        if (entry->Type() == kDmlExecutionProvider) {
          so.enable_mem_pattern = false;
          so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
          break;
        }
      }

      onnxruntime::training::TrainingSession session_object{so, GetEnvironment()};

      ASSERT_TRUE(!execution_providers->empty())
          << "Empty execution providers vector.";
      std::string provider_types;

      for (auto& entry : *execution_providers) {
        provider_types += entry->Type() + ":";

        std::unique_ptr<IExecutionProvider> execution_provider;
        if (entry->Type() == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (entry->Type() == onnxruntime::kCudaExecutionProvider)
          execution_provider = DefaultCudaExecutionProvider();
        else if (entry->Type() == onnxruntime::kDnnlExecutionProvider)
          execution_provider = DefaultDnnlExecutionProvider(1);
        else if (entry->Type() == onnxruntime::kNupharExecutionProvider)
          execution_provider = DefaultNupharExecutionProvider();
        else if (entry->Type() == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(execution_provider)));
      }

      has_run = true;

      fetches_ = ExecuteModel<onnxruntime::training::TrainingSession>(
          *p_model, session_object, expect_result, expected_failure_string,
          run_options, feeds, output_names, provider_types);

    } else {
      for (const std::string& provider_type : all_provider_types) {
        std::unique_ptr<IExecutionProvider> execution_provider;
        if (provider_type == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (provider_type == onnxruntime::kCudaExecutionProvider)
          execution_provider = DefaultCudaExecutionProvider();
        else if (provider_type == onnxruntime::kDnnlExecutionProvider)
          execution_provider = DefaultDnnlExecutionProvider();
        else if (provider_type == onnxruntime::kNupharExecutionProvider)
          execution_provider = DefaultNupharExecutionProvider();
        else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        else if (provider_type == onnxruntime::kRocmExecutionProvider)
          execution_provider = DefaultRocmExecutionProvider();
        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        bool valid = true;

        // set execution provider for all nodes in the graph
        for (auto& node : graph.Nodes()) {
          if (node.OpType() == kConstant)
            continue;

          //if node is not registered for the provider, skip
          node.SetExecutionProviderType(provider_type);
          auto reg = execution_provider->GetKernelRegistry();
          const KernelCreateInfo* kci;
          auto st = reg->TryFindKernel(node, execution_provider->Type(), &kci);
          if (!st.IsOK()) {
            auto* node_func = node.GetFunctionBody();
            if (!node_func) {
              valid = false;
            } else {
              for (auto& sub_node : node_func->Body().Nodes()) {
                if (sub_node.OpType() != "Constant") {
                  auto sub_reg = execution_provider->GetKernelRegistry();
                  const KernelCreateInfo* sub_kci;
                  st = sub_reg->TryFindKernel(sub_node, execution_provider->Type(), &sub_kci);
                  if (!st.IsOK()) {
                    valid = false;
                    break;
                  }
                }
              }
            }
            if (!valid)
              break;
          }
        }

        if (!valid)
          continue;

        has_run = true;
        onnxruntime::training::TrainingSession session_object{so, GetEnvironment()};

        EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

        fetches_ = ExecuteModel<onnxruntime::training::TrainingSession>(*p_model, session_object, expect_result, expected_failure_string, run_options,
                                                                        feeds, output_names, provider_type);
      }
    }
    EXPECT_TRUE(has_run) << "No registered execution providers were able to run the model.";

  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}

void GradientOpTester::FillFeedsAndOutputNames(std::unordered_map<std::string, MLValue>& feeds,
                                               std::vector<std::string>& output_names,
                                               int output_index_to_use_as_loss,
                                               int data_index_of_output) {
  OpTester::FillFeedsAndOutputNames(feeds, output_names);
  output_names.clear();  //ignore output names

  // add gradients as output instead
  for (size_t i = 0; i < input_data_.size(); ++i) {
    if (!input_infos_[i].has_gradient) {
      continue;
    }
    output_names.push_back(input_data_[i].def_.Name() + "_grad");
  }

  // Append gradient names and values to feeds
  std::vector<Data> gradient_data;
  for (size_t i = 0; i < output_data_.size(); i++) {
    if (!output_infos_[i].has_gradient) {
      continue;
    }
    auto shape = output_data_[i].data_.Get<Tensor>().Shape();
    std::vector<float> values(shape.Size(), 0.0);
    if (output_index_to_use_as_loss == static_cast<int>(i)) {
      values[data_index_of_output] = 1.0;  //set only one value to one to construct jacobian matrix
    }
    AddData<float>(gradient_data, (output_data_[i].def_.Name() + "_grad").c_str(), shape.GetDims(), values.data(), values.size(), true);
  }

  for (size_t i = 0; i < gradient_data.size(); ++i) {
    feeds[gradient_data[i].def_.Name()] = gradient_data[i].data_;
  }
}

}  // namespace test
}  // namespace onnxruntime
