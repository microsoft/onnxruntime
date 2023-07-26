// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Implements a custom operator kernel which counts the number of calls to Compute(), but otherwise is a no-op.
//

#pragma once

#include "test.h"

template <typename T>
struct NullShapeInferrer : winrt::implements<NullShapeInferrer<T>, IMLOperatorShapeInferrer> {
  STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept {
    WINML_EXPECT_NO_THROW(OperatorHelper::ShapeInferenceFunction<T>(context));
    return S_OK;
  }
};

struct NullOperator : winrt::implements<NullOperator, IMLOperatorKernel> {
  NullOperator(std::atomic<uint32_t>* callCount) : m_callCount(callCount) {}

  STDMETHOD(Compute)(IMLOperatorKernelContext* context) {
    winrt::com_ptr<IMLOperatorTensor> outputTensor;
    WINML_EXPECT_HRESULT_SUCCEEDED(context->GetOutputTensor(0, outputTensor.put()));

    ++(*m_callCount);
    return S_OK;
  }

 private:
  std::atomic<uint32_t>* m_callCount;
};

struct NullOperatorFactory : winrt::implements<NullOperatorFactory, IMLOperatorKernelFactory> {
  NullOperatorFactory(std::atomic<uint32_t>* callCount) : m_callCount(callCount) {}

  STDMETHOD(CreateKernel)(IMLOperatorKernelCreationContext* context, IMLOperatorKernel** kernel) {
    ORT_UNUSED_PARAMETER(context);
    auto op = winrt::make<NullOperator>(m_callCount);
    op.copy_to(kernel);
    return S_OK;
  }

  static MLOperatorEdgeDescription CreateEdgeDescriptor(MLOperatorEdgeType type, MLOperatorTensorDataType dataType) {
    ORT_UNUSED_PARAMETER(type);
    MLOperatorEdgeDescription desc;
    desc.edgeType = MLOperatorEdgeType::Tensor;
    desc.tensorDataType = dataType;
    return desc;
  }

  static void RegisterKernel(
    const char* name,
    const char* domain,
    int versionSince,
    winrt::com_ptr<IMLOperatorRegistry> registry,
    winrt::com_ptr<IMLOperatorShapeInferrer> shapeInferrer,
    std::atomic<uint32_t>* callCount
  ) {
    MLOperatorKernelDescription kernelDescription;
    kernelDescription.domain = domain;
    kernelDescription.name = name;
    kernelDescription.minimumOperatorSetVersion = versionSince;
    kernelDescription.executionType = MLOperatorExecutionType::D3D12;

    MLOperatorEdgeTypeConstrant typeConstraint;
    typeConstraint.typeLabel = "T";
    std::vector<MLOperatorEdgeDescription> allowedEdges{
      CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Double),
      CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Float),
      CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Float16)};
    typeConstraint.allowedTypes = allowedEdges.data();
    typeConstraint.allowedTypeCount = static_cast<uint32_t>(allowedEdges.size());

    std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{typeConstraint};
    kernelDescription.typeConstraints = typeConstraints.data();
    kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

    kernelDescription.defaultAttributes = nullptr;
    kernelDescription.defaultAttributeCount = 0;
    kernelDescription.options = MLOperatorKernelOptions::None;
    kernelDescription.executionOptions = 0;

    auto factory = winrt::make<NullOperatorFactory>(callCount);

    WINML_EXPECT_HRESULT_SUCCEEDED(
      registry->RegisterOperatorKernel(&kernelDescription, factory.get(), shapeInferrer.get())
    );
  }

 private:
  std::atomic<uint32_t>* m_callCount;
};
