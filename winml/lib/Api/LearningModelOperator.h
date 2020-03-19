#pragma once

#include "LearningModelOperator.g.h"
#include "TensorFeatureDescriptor.h"
#include "iengine.h"
#include "LearningModelBuilder.h"
#include "LearningModelInputs.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct LearningModelOperator : LearningModelOperatorT<LearningModelOperator>
    {
        LearningModelOperator() = delete;
        LearningModelOperator(hstring const& type, hstring const& name);

        more::LearningModelOperator Then(more::LearningModelOperator const& next_operator);
        more::LearningModelOperator Then(more::LearningModelOperator const& next_operator, more::LearningModelOperatorResolutionPolicy const& policy);
        more::LearningModelBuilder ConnectToOutputs();
        more::LearningModelBuilder ConnectToOutputs(more::LearningModelOperatorResolutionPolicy const& policy);
        more::LearningModelOperator SetAttribute(hstring const& name, wf::IInspectable const& value);
        wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Inputs();
        wfc::IMapView<hstring, hstring> InputMapping();
        wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Attributes();
        wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Outputs();
        hstring Name();

        void SetBuilder(more::LearningModelBuilder& builder);
        void GetBuilder(more::LearningModelBuilder& builder);
        void JoinAfterInternal(wfc::IVectorView<winml::ILearningModelFeatureDescriptor>& input_decs);
        void JoinAfter(more::LearningModelInputs const& inputs);
        void JoinAfter(more::LearningModelOperator const& previous_operator);

        void SetAttributeInternal(const char* const name, wf::IInspectable const& /*inspectable*/);

        static more::LearningModelOperator Gemm();
        static more::LearningModelOperator Gemm(hstring const& name);

        private:
        more::LearningModelBuilder builder_;
        winrt::hstring name_;
        winrt::hstring type_;

        wfc::IVector<winml::ILearningModelFeatureDescriptor> inputs_;
        wfc::IVector<winml::ILearningModelFeatureDescriptor> outputs_;
        wfc::IMap<winrt::hstring, winrt::hstring> input_mapping_;

        std::unordered_map<std::string, winml::ILearningModelFeatureDescriptor> attributes_;
        std::unordered_map<std::string, winrt::com_ptr<WinML::IValue>> attribute_values_;
    };
}

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation
{
    struct LearningModelOperator : LearningModelOperatorT<LearningModelOperator, implementation::LearningModelOperator>
    {
    };
}
