#pragma once

#include "LearningModelOperator.g.h"
#include "TensorFeatureDescriptor.h"
#include "iengine.h"
#include "LearningModelBuilder.h"
#include "LearningModelInputs.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelOperator : LearningModelOperatorT<LearningModelOperator>
{
    LearningModelOperator() = delete;
    LearningModelOperator(hstring const& type, hstring const& name);

    winml_experimental::LearningModelOperator Then(winml_experimental::LearningModelOperator const& next_operator);
    winml_experimental::LearningModelOperator Then(winml_experimental::LearningModelOperator const& next_operator, winml_experimental::LearningModelOperatorResolutionPolicy const& policy);
    winml_experimental::LearningModelBuilder ConnectToOutputs();
    winml_experimental::LearningModelBuilder ConnectToOutputs(winml_experimental::LearningModelOperatorResolutionPolicy const& policy);
    winml_experimental::LearningModelOperator SetAttribute(hstring const& name, wf::IInspectable const& value);
    wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Inputs();
    wfc::IMapView<hstring, hstring> InputMapping();
    wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Attributes();
    wfc::IVectorView<winml::ILearningModelFeatureDescriptor> Outputs();
    hstring Name();

    void SetBuilder(winml_experimental::LearningModelBuilder& builder);
    void GetBuilder(winml_experimental::LearningModelBuilder& builder);
    void JoinAfterInternal(wfc::IVectorView<winml::ILearningModelFeatureDescriptor>& input_decs);
    void JoinAfter(winml_experimental::LearningModelInputs const& inputs);
    void JoinAfter(winml_experimental::LearningModelOperator const& previous_operator);

    void SetAttributeInternal(const char* const name, wf::IInspectable const& /*inspectable*/);

    static winml_experimental::LearningModelOperator Gemm();
    static winml_experimental::LearningModelOperator Gemm(hstring const& name);

    private:
    winml_experimental::LearningModelBuilder builder_;
    winrt::hstring name_;
    winrt::hstring type_;

    wfc::IVector<winml::ILearningModelFeatureDescriptor> inputs_;
    wfc::IVector<winml::ILearningModelFeatureDescriptor> outputs_;
    wfc::IMap<winrt::hstring, winrt::hstring> input_mapping_;

    std::unordered_map<std::string, winml::ILearningModelFeatureDescriptor> attributes_;
    std::unordered_map<std::string, winrt::com_ptr<_winml::IValue>> attribute_values_;
};

} // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelOperator : LearningModelOperatorT<LearningModelOperator, implementation::LearningModelOperator>
{
};

}
