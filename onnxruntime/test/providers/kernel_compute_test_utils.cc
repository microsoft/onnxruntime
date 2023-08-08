// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_STRIDED_TENSORS

#include "test/providers/kernel_compute_test_utils.h"

#include "core/framework/execution_providers.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void KernelComputeTester::Run(std::unordered_set<int> strided_outputs) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  auto cpu_ep_type = cpu_ep->Type();
  DataTransferManager dtm;
  auto cpu_transfer = cpu_ep->GetDataTransfer();
  ASSERT_STATUS_OK(dtm.RegisterDataTransfer(std::move(cpu_transfer)));
  ExecutionProviders execution_providers;
  ASSERT_STATUS_OK(execution_providers.Add(cpu_ep_type, std::move(cpu_ep)));
  auto ep_type = cpu_ep_type;
#ifdef USE_CUDA
  if (provider_ == kCudaExecutionProvider) {
    auto cuda_ep = DefaultCudaExecutionProvider();
    ep_type = cuda_ep->Type();
    auto cuda_transfer = cuda_ep->GetDataTransfer();
    ASSERT_STATUS_OK(dtm.RegisterDataTransfer(std::move(cuda_transfer)));
    ASSERT_STATUS_OK(execution_providers.Add(ep_type, std::move(cuda_ep)));
  }
#endif
#ifdef USE_ROCM
  if (provider_ == kRocmExecutionProvider) {
    auto rocm_ep = DefaultRocmExecutionProvider();
    ep_type = rocm_ep->Type();
    auto rocm_transfer = rocm_ep->GetDataTransfer();
    ASSERT_STATUS_OK(dtm.RegisterDataTransfer(std::move(rocm_transfer)));
    ASSERT_STATUS_OK(execution_providers.Add(ep_type, std::move(rocm_ep)));
  }
#endif

  Model model("test", false, ModelMetaData(), ORT_TSTR(""), IOnnxRuntimeOpSchemaRegistryList(),
              {{domain_, opset_version_}}, {}, DefaultLoggingManager().DefaultLogger());

  std::vector<NodeArg*> input_args;
  std::unordered_map<std::string, OrtValue> initializer_map;
  for (auto& data : input_data_) {
    input_args.emplace_back(&data.def_);
    const auto& name = data.def_.Name();
    // If running on CPU or input is CPU input, use the OrtValue from input_data_ directly.
    // Otherwise, we need to create a new OrtValue on target device.
    if (provider_ == kCpuExecutionProvider || data.is_cpu_data_) {
      initializer_map[name] = data.value_;
    }
#if defined(USE_CUDA) || defined(USE_ROCM)
    if ((provider_ == kCudaExecutionProvider || provider_ == kRocmExecutionProvider) && !data.is_cpu_data_) {
      OrtValue gpu_value;
      const Tensor& tensor = data.value_.Get<Tensor>();
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                           execution_providers.Get(ep_type)->CreatePreferredAllocators()[0], gpu_value,
                           tensor.Strides());
      ASSERT_STATUS_OK(dtm.CopyTensor(tensor, *gpu_value.GetMutable<Tensor>()));
      initializer_map[name] = gpu_value;
    }
#endif
  }

  std::vector<NodeArg*> output_args;
  for (auto& data : output_data_) {
    output_args.emplace_back(&data.def_);
  }

  Graph& graph = model.MainGraph();
  auto& node = graph.AddNode("node", op_, op_, input_args, output_args, nullptr, domain_);
  for (auto& add_attribute_fn : add_attribute_funcs_) {
    add_attribute_fn(node);
  }
  ASSERT_STATUS_OK(graph.Resolve());

  node.SetExecutionProviderType(ep_type);
  OptimizerExecutionFrame::Info info({&node}, initializer_map, graph.ModelPath(), *execution_providers.Get(ep_type),
                                     [](std::string const&) { return false; });
  const KernelCreateInfo* kernel_create_info = nullptr;
  ASSERT_STATUS_OK(info.TryFindKernel(&node, &kernel_create_info));
  ASSERT_TRUE(kernel_create_info);

  // If the tester specifies output i is strided tensor output, we need to check output i for this kernel is
  // MayStridedOutput from the KernelDef, and create an OrtValue as output by sharing the data buffer from
  // its corresponding input. Otherwise, create an OrtValue with contiguous tensor.
  const auto& may_strided_outputs_map = kernel_create_info->kernel_def->MayStridedOutput();
  std::vector<OrtValue> outputs;
  for (size_t i = 0; i < output_data_.size(); ++i) {
    OrtValue output;
    const Tensor& tensor = output_data_[i].value_.Get<Tensor>();
    if (strided_outputs.find(static_cast<int>(i)) != strided_outputs.end()) {
      bool is_may_strided_output = false;
      for (auto& pair : may_strided_outputs_map) {
        if (pair.second == static_cast<int>(i)) {
          Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                               initializer_map[input_data_[static_cast<size_t>(pair.first)].def_.Name()]
                                   .GetMutable<Tensor>()
                                   ->MutableDataRaw(),
                               execution_providers.Get(ep_type)->CreatePreferredAllocators()[0]->Info(), output);
          is_may_strided_output = true;
          break;
        }
      }
      ASSERT_TRUE(is_may_strided_output);
    } else {
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                           execution_providers.Get(ep_type)->CreatePreferredAllocators()[0], output);
    }
    outputs.emplace_back(output);
  }

  auto kernel = info.CreateKernel(&node);
  ASSERT_TRUE(kernel);

  std::vector<int> fetch_mlvalue_idxs;
  for (const auto* node_out : node.OutputDefs()) {
    fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
  }

  // Execute the kernel and fetch outputs.
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 6387)
#endif
  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs, outputs);
  OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, nullptr, DefaultLoggingManager().DefaultLogger());
#ifdef _WIN32
#pragma warning(pop)
#endif

  ASSERT_STATUS_OK(kernel->Compute(&op_kernel_context));
  ASSERT_STATUS_OK(frame.GetOutputs(outputs));

  // Check the outputs.
  for (size_t i = 0; i < output_data_.size(); ++i) {
    if (strided_outputs.find(static_cast<int>(i)) != strided_outputs.end()) {
      // If the output tensor is strided tensor, check that it shares the data buffer from corresponding input,
      // and the strides is same as expected.
      bool is_may_strided_output = false;
      for (auto& pair : may_strided_outputs_map) {
        if (pair.second == static_cast<int>(i)) {
          EXPECT_EQ(outputs[i].Get<Tensor>().DataRaw(),
                    initializer_map[input_data_[static_cast<size_t>(pair.first)].def_.Name()].Get<Tensor>().DataRaw());
          EXPECT_EQ(outputs[i].Get<Tensor>().Strides(), output_data_[i].value_.Get<Tensor>().Strides());
          is_may_strided_output = true;
          break;
        }
      }
      ASSERT_TRUE(is_may_strided_output);
    } else {
      // If it's contiguous output, check if the data is same as expected. If the output is on GPU, copy them to CPU
      // for comparison.
      OrtValue cpu_value;
      if (provider_ == kCpuExecutionProvider) {
        cpu_value = outputs[i];
      } else {
        const Tensor& tensor = outputs[i].Get<Tensor>();
        Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(),
                             execution_providers.Get(cpu_ep_type)->CreatePreferredAllocators()[0], cpu_value,
                             tensor.Strides());
        ASSERT_STATUS_OK(dtm.CopyTensor(tensor, *cpu_value.GetMutable<Tensor>()));
      }

      CheckOrtValuesAreEqual(output_data_[i].def_.Name(), output_data_[i].value_, cpu_value, {}, provider_);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
