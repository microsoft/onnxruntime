#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelOperator.h"

namespace WINML_EXPERIMENTALP {

static uint32_t c_operator_index = 0;

LearningModelOperator::LearningModelOperator(hstring const& type) : LearningModelOperator(type, L"") {
}

LearningModelOperator::LearningModelOperator(hstring const& type, hstring const& domain)
  : domain_(domain),
    type_(type) {
  constant_input_mapping_ = winrt::single_threaded_map<winrt::hstring, wf::IInspectable>();
  input_mapping_ = winrt::single_threaded_map<winrt::hstring, winrt::hstring>();
  output_mapping_ = winrt::single_threaded_map<winrt::hstring, winrt::hstring>();
  attribute_values_ = winrt::single_threaded_map<winrt::hstring, wf::IInspectable>();

  SetName(L"");
}

winml_experimental::LearningModelOperator LearningModelOperator::SetName(hstring const& name) {
  if (name.empty()) {
    std::wostringstream name_stream;
    name_stream << type_.c_str() << "_" << c_operator_index++;
    name_ = name_stream.str().c_str();
  } else {
    name_ = name;
  }
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetInput(
  hstring const& operator_input_name, hstring const& input_name
) {
  // TODO Validate against allowed operator input NAMES. The types are not deduced.
  input_mapping_.Insert(operator_input_name, input_name);
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetConstant(
  hstring const& operator_input_name, wf::IInspectable const& value
) {
  // TODO Validate against allowed operator input NAMES. The types are not deduced.
  auto constant_name = name_ + L"." + operator_input_name;
  input_mapping_.Insert(operator_input_name, constant_name);
  constant_input_mapping_.Insert(constant_name, value);
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetOutput(
  hstring const& operator_output_name, hstring const& output_name
) {
  // TODO Validate against allowed operator output NAMES. The types are not deduced.
  output_mapping_.Insert(operator_output_name, output_name);
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetAttribute(
  hstring const& name, Windows::Foundation::IInspectable const& value
) {
  attribute_values_.Insert(name, value);
  return *this;
}

hstring LearningModelOperator::Name() {
  return name_;
}

hstring LearningModelOperator::Type() {
  return type_;
}

hstring LearningModelOperator::Domain() {
  return domain_;
}

wfc::IMap<winrt::hstring, winrt::hstring> LearningModelOperator::InputMapping() {
  return input_mapping_;
}

wfc::IMap<winrt::hstring, wf::IInspectable> LearningModelOperator::ConstantInputMapping() {
  return constant_input_mapping_;
}

wfc::IMap<winrt::hstring, winrt::hstring> LearningModelOperator::OutputMapping() {
  return output_mapping_;
}

wfc::IMap<winrt::hstring, wf::IInspectable> LearningModelOperator::AttributeMap() {
  return attribute_values_;
}

}  // namespace WINML_EXPERIMENTALP
