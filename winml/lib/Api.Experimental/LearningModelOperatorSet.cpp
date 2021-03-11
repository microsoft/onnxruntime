#include "pch.h"
#include "LearningModelOperatorSet.h"
#include "LearningModelOperator.h"

#include "..\Api\inc\ILotusValueProviderPrivate.h"

namespace WINML_EXPERIMENTALP {

LearningModelOperatorSet::LearningModelOperatorSet(winml_experimental::LearningModelBuilder builder) :
    builder_(builder),
    operators_(winrt::single_threaded_vector<winml_experimental::LearningModelOperator>())
{
}

winml_experimental::LearningModelBuilder LearningModelOperatorSet::Add(winml_experimental::LearningModelOperator const& op)
{
  auto operator_private = op.as<winml_experimentalp::LearningModelOperator>();
  auto constant_input_map = operator_private->ConstantInputMapping();
  auto input_map = operator_private->InputMapping();
  auto output_map = operator_private->OutputMapping();
  auto attribute_map = operator_private->AttributeMap();

  auto operator_name = _winml::Strings::UTF8FromHString(operator_private->Name());
  auto operator_type = _winml::Strings::UTF8FromHString(operator_private->Type());
  auto operator_domain = _winml::Strings::UTF8FromHString(operator_private->Domain());

  std::vector<std::string> operator_input_names(input_map.Size());
  std::vector<std::string> actual_input_names(input_map.Size());
  std::vector<const char*> raw_operator_input_names(input_map.Size());
  std::vector<const char*> raw_actual_input_names(input_map.Size());
  int i = 0;
  for (auto kvp : input_map) {
    operator_input_names[i] = _winml::Strings::UTF8FromHString(kvp.Key());
    actual_input_names[i] = _winml::Strings::UTF8FromHString(kvp.Value());
    raw_operator_input_names[i] = operator_input_names[i].c_str();
    raw_actual_input_names[i] = actual_input_names[i].c_str();
    i++;
  }

  std::vector<std::string> operator_output_names(output_map.Size());
  std::vector<std::string> actual_output_names(output_map.Size());
  std::vector<const char*> raw_operator_output_names(output_map.Size());
  std::vector<const char*> raw_actual_output_names(output_map.Size());
  i = 0;
  for (auto kvp : output_map) {
    operator_output_names[i] = _winml::Strings::UTF8FromHString(kvp.Key());
    actual_output_names[i] = _winml::Strings::UTF8FromHString(kvp.Value());
    raw_operator_output_names[i] = operator_output_names[i].c_str();
    raw_actual_output_names[i] = actual_output_names[i].c_str();
    i++;
  }

  // Create the Binding Context to pass to the feature value
  _winml::BindingContext context{
      _winml::BindingType::kInput,
      builder_.as<winml_experimentalp::LearningModelBuilder>()->InertSession(),
      nullptr,
      nullptr,
      {}  // SubresourceId is set by callee
  };

  std::vector<std::string> attribute_names(attribute_map.Size());
  std::vector<const char*> raw_attribute_names(attribute_map.Size());
  std::vector<winrt::com_ptr<_winml::IValue>> attribute_values(attribute_map.Size());
  std::vector<_winml::IValue*> raw_attribute_values(attribute_map.Size());
  i = 0;
  for (auto kvp : attribute_map) {
    attribute_names[i] = _winml::Strings::UTF8FromHString(kvp.Key());
    auto default_value_value_provider = kvp.Value().as<_winml::ILotusValueProviderPrivate>();
    default_value_value_provider->GetValue(context, attribute_values[i].put());

    raw_attribute_names[i] = attribute_names[i].c_str();
    raw_attribute_values[i] = attribute_values[i].get();
    i++;
  }

  auto builder = builder_.as<winml_experimentalp::LearningModelBuilder>();
  WINML_THROW_IF_FAILED(builder->UseModel()->AddOperator(
      operator_type.c_str(),
      operator_name.c_str(),
      operator_domain.c_str(),
      raw_operator_input_names.data(), raw_actual_input_names.data(), input_map.Size(),
      raw_operator_output_names.data(), raw_actual_output_names.data(), output_map.Size(),
      raw_attribute_names.data(), raw_attribute_values.data(), attribute_map.Size()));

  // Add constants
  for (auto kvp : constant_input_map) {
    builder_.Inputs().AddConstant(kvp.Key(), kvp.Value());
  }

  return builder_;
}

}
