#include "pch.h"
#include "LearningModelOperator.h"

namespace WINML_EXPERIMENTALP {

static uint32_t c_operator_index = 0;

LearningModelOperator::LearningModelOperator(hstring const& type, hstring const& name) : LearningModelOperator(type, name, L"")
{}

LearningModelOperator::LearningModelOperator(hstring const& type, hstring const& name, hstring const& domain) :
    type_(type),
    name_(name),
    domain_(domain) {
  input_mapping_ = winrt::single_threaded_map<winrt::hstring, winrt::hstring>();
  output_mapping_ = winrt::single_threaded_map<winrt::hstring, winrt::hstring>();

  if (name_.empty()) {
    std::wostringstream name_stream;
    name_stream << type_.c_str() << "_" << c_operator_index;
    name_ = name_stream.str().c_str();
  }
}

winml_experimental::LearningModelOperator LearningModelOperator::SetInput(
    hstring const& operator_input_name, hstring const& input_name) {

  // TODO Validate against allowed operator input NAMES. The types are not deduced.
  input_mapping_.Insert(operator_input_name, input_name);
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetOutput(
    hstring const& operator_output_name, hstring const& output_name) {
  // TODO Validate against allowed operator output NAMES. The types are not deduced.
  output_mapping_.Insert(operator_output_name, output_name);
  return *this;
}

winml_experimental::LearningModelOperator LearningModelOperator::SetAttribute(hstring const& name, Windows::Foundation::IInspectable const& value) {
  // TODO Validate against allowed operator attribute NAMES. The types are not deduced.
 // attributes_[name] = inspectable;

  /*
    auto featureValue = _winml::CreateFeatureValueFromInspectable(WinML::BindingType::kInput, inspectable, found_it->second);
    // Validate that the feature value is compatible with the descriptor
    _winml::VerifyFeatureValueCompatibleWithDescriptor(featureValue, found_it->second);
    
    auto spLotusValueProvider = featureValue.as<_winml::ILotusValueProviderPrivate>();

    ////////
    //////// TODO: Need to create a fake IEngine that is not backed by a cpu session but no model in order to generate the appropriate values here.
    ////////

    // Create the Binding Context to pass to the feature value
    _winml::BindingContext context{
        _winml::BindingType::kInput,
        nullptr,
        found_it->second,
        nullptr,
        {}  // SubresourceId is set by callee
    };

    // Get the bound tensor
    winrt::com_ptr<_winml::IValue> value;
    spLotusValueProvider->GetValue(context, value.put());

    attribute_values_[name] = value;

    */
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

wfc::IMap<winrt::hstring, winrt::hstring> LearningModelOperator::InputMapping(){
  return input_mapping_;
}

wfc::IMap<winrt::hstring, winrt::hstring> LearningModelOperator::OutputMapping() {
  return output_mapping_;
}

std::unordered_map<std::string, winrt::com_ptr<_winml::IValue>> LearningModelOperator::AttributeMap() {
  return attribute_values_;
}

}  // namespace WINML_EXPERIMENTALP