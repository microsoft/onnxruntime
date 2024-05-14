// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/test/gradient/gradient_op_test_utils.h"

#include "gmock/gmock.h"

#include "core/framework/kernel_type_str_resolver.h"
#include "core/session/inference_session.h"

#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/graph/gradient_config.h"

#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void GradientOpTester::Run(int output_index_to_use_as_loss,
                           int data_index_of_output,
                           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  SetTestFunctionCalled();

  try {
    std::unordered_map<std::string, int> extra_domain_to_version{{kMSDomain, 1}, {kOnnxDomain, 9}};

    ORT_ENFORCE(cached_model_, "BuildAndCacheModel must be called first");
    Model& model = *cached_model_;
    Graph& graph = model.MainGraph();

    // Hookup the inputs and outputs
    std::unordered_map<std::string, OrtValue> feeds;
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

      ASSERT_STATUS_OK(graph.Resolve());
    }

    // Run the model
    SessionOptions so;
    so.session_logid = Op();

    static const std::string all_provider_types[] = {
        kCpuExecutionProvider,
        kCudaExecutionProvider,
        kRocmExecutionProvider,
        kDnnlExecutionProvider,
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

      onnxruntime::InferenceSession session_object{so, GetEnvironment()};

      ASSERT_TRUE(!execution_providers->empty()) << "Empty execution providers vector.";
      std::string provider_types;

      for (auto& entry : *execution_providers) {
        provider_types += entry->Type() + ":";

        std::unique_ptr<IExecutionProvider> execution_provider;
        if (entry->Type() == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (entry->Type() == onnxruntime::kCudaExecutionProvider)
          execution_provider = DefaultCudaExecutionProvider();
        else if (entry->Type() == onnxruntime::kDnnlExecutionProvider)
          execution_provider = DefaultDnnlExecutionProvider();
        else if (entry->Type() == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(execution_provider)));
      }

      has_run = true;

      ExecuteModel<onnxruntime::InferenceSession>(
          model, session_object, ExpectResult::kExpectSuccess, "", nullptr, feeds, output_names, provider_types);
    } else {
      for (const std::string& provider_type : all_provider_types) {
        std::unique_ptr<IExecutionProvider> execution_provider;
        if (provider_type == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (provider_type == onnxruntime::kCudaExecutionProvider)
          execution_provider = DefaultCudaExecutionProvider();
        else if (provider_type == onnxruntime::kDnnlExecutionProvider)
          execution_provider = DefaultDnnlExecutionProvider();
        else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        else if (provider_type == onnxruntime::kRocmExecutionProvider)
          execution_provider = DefaultRocmExecutionProvider();
        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        bool valid = true;

        OpSchemaKernelTypeStrResolver kernel_type_str_resolver{};

        // set execution provider for all nodes in the graph
        for (auto& node : graph.Nodes()) {
          if (node.OpType() == kConstant)
            continue;

          // if node is not registered for the provider, skip
          node.SetExecutionProviderType(provider_type);

          // provider types that don't use the KernelRegistry
          if (provider_type == onnxruntime::kDnnlExecutionProvider) {
            continue;
          }

          auto reg = execution_provider->GetKernelRegistry();
          const KernelCreateInfo* kci;
          auto st = reg->TryFindKernel(node, execution_provider->Type(), kernel_type_str_resolver, &kci);
          if (!st.IsOK()) {
            // The goal here is unclear. It seems best to leave it to the Session
            // creation to figure out whether the model can be executed using some
            // valid execution-provider. Removed the logic here for partially inlining
            // functions, as function-inlining requires other pre-conditions like
            // Graph::Resolve etc, and it appears it is not being used anyway.
            if (!node.CanBeInlined()) {
              valid = false;
              break;
            }
          }
        }

        if (!valid)
          continue;

        has_run = true;
        onnxruntime::InferenceSession session_object{so, GetEnvironment()};

        EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

        ExecuteModel<onnxruntime::InferenceSession>(
            model, session_object, ExpectResult::kExpectSuccess, "", nullptr, feeds, output_names, provider_type);
      }
    }

    EXPECT_TRUE(has_run) << "No registered execution providers were able to run the model.";
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}

void GradientOpTester::FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                                               std::vector<std::string>& output_names,
                                               int output_index_to_use_as_loss,
                                               int data_index_of_output) {
  // this method is for usage when these two properties are set
  assert(input_infos_ && output_infos_);

  const auto& input_infos = *input_infos_;
  const auto& output_infos = *output_infos_;

  OpTester::FillFeedsAndOutputNames(feeds, output_names);
  output_names.clear();  // ignore output names

  const auto& input_data = GetInputData();
  const auto& output_data = GetOutputData();

  // add gradients as output instead
  for (size_t i = 0; i < input_data.size(); ++i) {
    if (!input_infos[i].has_gradient) {
      continue;
    }
    output_names.push_back(input_data[i].def.Name() + "_grad");
  }

  // Append gradient names and values to feeds
  std::vector<Data> gradient_data;
  for (size_t i = 0; i < output_data.size(); i++) {
    if (!output_infos[i].has_gradient) {
      continue;
    }

    auto shape = output_data[i].data.Get<Tensor>().Shape();
    std::vector<float> values(shape.Size(), 0.0);
    if (output_index_to_use_as_loss == static_cast<int>(i)) {
      values[data_index_of_output] = 1.0;  // set only one value to one to construct jacobian matrix
    }

    AddData<float>(gradient_data, (output_data[i].def.Name() + "_grad").c_str(), shape.AsShapeVector(),
                   values.data(), values.size(), true);
  }

  for (size_t i = 0; i < gradient_data.size(); ++i) {
    feeds[gradient_data[i].def.Name()] = gradient_data[i].data;
  }
}

}  // namespace test
}  // namespace onnxruntime
