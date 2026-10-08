// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"

namespace onnxruntime::test {

class WebGpuGraphCaptureTester final : public OpTester {
 public:
  using OpTester::OpTester;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuGraphCaptureTester);

  template <typename T>
  void RunWithBoundCapture(std::unique_ptr<IExecutionProvider> provider,
                           const SessionOptions& options, const TensorShape& shape,
                           gsl::span<const T> expected) {
    SetTestFunctionCalled();
    const std::string model = BuildModel().ToProto().SerializeAsString();
    auto* capture_provider = provider.get();
    InferenceSession session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const auto allocators = capture_provider->CreatePreferredAllocators();
    ASSERT_FALSE(allocators.empty());
    const auto allocator = allocators.back();
    ASSERT_NE(allocator->Info().device.Type(), OrtDevice::CPU);
    std::unique_ptr<IOBinding> binding;
    ASSERT_STATUS_OK(session.NewIOBinding(&binding));
    std::unordered_map<std::string, OrtValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);
    ASSERT_EQ(output_names.size(), 1u);
    for (const auto& [name, value] : feeds) {
      const auto& source = value.Get<Tensor>();
      OrtValue input;
      Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, input);
      if (source.Shape().Size() != 0) {
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(source, *input.GetMutable<Tensor>()));
      }
      ASSERT_STATUS_OK(binding->BindInput(name, input));
    }
    OrtValue output;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, allocator, output);
    ASSERT_STATUS_OK(binding->BindOutput(output_names[0], output));
    ASSERT_STATUS_OK(binding->SynchronizeInputs());
    Tensor cpu_output(DataTypeImpl::GetType<T>(), shape, CPUAllocator::DefaultInstance());
    ASSERT_EQ(static_cast<size_t>(shape.Size()), expected.size());
    for (int run = 0; run < 4; ++run) {
      if (!expected.empty()) {
        std::fill_n(cpu_output.MutableData<T>(), expected.size(), T{-99.0f});
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_output, *output.GetMutable<Tensor>()));
      }
      ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
      ASSERT_STATUS_OK(binding->SynchronizeOutputs());
      ASSERT_EQ(binding->GetOutputs()[0].Get<Tensor>().Shape(), shape);
      if (!expected.empty()) {
        ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(output.Get<Tensor>(), cpu_output));
        for (size_t index = 0; index < expected.size(); ++index) {
          EXPECT_EQ(cpu_output.Data<T>()[index], expected[index]) << "run=" << run << " index=" << index;
        }
      }
    }
    EXPECT_TRUE(capture_provider->IsGraphCaptured(0));
  }
};

}  // namespace onnxruntime::test
