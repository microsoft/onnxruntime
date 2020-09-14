#include "pch.h"
#include "LearningModelOperatorSet.h"
#include "LearningModelOperator.h"

namespace WINML_EXPERIMENTALP {

LearningModelOperatorSet::LearningModelOperatorSet(winml_experimental::LearningModelBuilder builder) :
    builder_(builder),
    operators_(winrt::single_threaded_vector<winml_experimental::LearningModelOperator>())
{
}

winml_experimental::LearningModelBuilder LearningModelOperatorSet::Add(winml_experimental::LearningModelOperator const& op)
{
  auto operator_private = op.as<winml_experimentalp::LearningModelOperator>();
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

  auto builder = builder_.as<winml_experimentalp::LearningModelBuilder>();
  WINML_THROW_IF_FAILED(builder->UseModel()->AddOperator(
      operator_type.c_str(),
      operator_name.c_str(),
      operator_domain.c_str(),
      raw_operator_input_names.data(), raw_actual_input_names.data(), input_map.Size(),
      raw_operator_output_names.data(), raw_actual_output_names.data(), output_map.Size()));

  return builder_;
}

}
