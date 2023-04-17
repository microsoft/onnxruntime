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
    LearningModelOperator(hstring const& type);
    LearningModelOperator(hstring const& type, hstring const& domain);
    
    winml_experimental::LearningModelOperator SetName(hstring const& name);
    winml_experimental::LearningModelOperator SetInput(hstring const& operator_input_name, hstring const& input_name);
    winml_experimental::LearningModelOperator SetConstant(hstring const& operator_input_name, wf::IInspectable const& value);
    winml_experimental::LearningModelOperator SetOutput(hstring const& operator_output_name, hstring const& output_name);
    winml_experimental::LearningModelOperator SetAttribute(hstring const& name, wf::IInspectable const& value);
    hstring Name();
    hstring Type();
    hstring Domain();

    wfc::IMap<winrt::hstring, winrt::hstring> InputMapping();
    wfc::IMap<winrt::hstring, wf::IInspectable> ConstantInputMapping();
    wfc::IMap<winrt::hstring, winrt::hstring> OutputMapping();
    wfc::IMap<winrt::hstring, wf::IInspectable> AttributeMap();

private:
    winrt::hstring name_;
    winrt::hstring domain_;
    winrt::hstring type_;

    wfc::IMap<winrt::hstring, wf::IInspectable> attribute_values_;
    wfc::IMap<winrt::hstring, wf::IInspectable> constant_input_mapping_;
    wfc::IMap<winrt::hstring, winrt::hstring> input_mapping_;
    wfc::IMap<winrt::hstring, winrt::hstring> output_mapping_;
};

} // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelOperator : LearningModelOperatorT<LearningModelOperator, implementation::LearningModelOperator>
{
};

}
