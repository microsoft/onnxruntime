#include "pch.h"
#include "LearningModelOperator.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation {
static uint32_t c_operator_index = 0;

LearningModelOperator::LearningModelOperator(hstring const& type, hstring const& name) : builder_(nullptr),
                                                                                         type_(type),
                                                                                         name_(name) {
  if (name_.empty()) {
    std::wostringstream name_stream;
    name_stream << type_.c_str() << "_" << c_operator_index;
    name_ = name_stream.str().c_str();
  }
}

void LearningModelOperator::SetBuilder(more::LearningModelBuilder& builder) {
  builder_ = builder;
}

void LearningModelOperator::GetBuilder(more::LearningModelBuilder& builder) {
  builder = builder_;
}

void LearningModelOperator::JoinAfterInternal(wfc::IVectorView<winml::ILearningModelFeatureDescriptor>& input_decs) {
  auto operator_name = WinML::Strings::UTF8FromHString(name_);
  auto operator_type = WinML::Strings::UTF8FromHString(type_);

  auto builder = builder_.as<morep::LearningModelBuilder>();

  // Expect that the Outputs of the current node are fully inferred already!!!!

  // No need to get the next operators acceptable input types as they are known in
  // the onnx schema defs in onnxruntime.dll, only need to reference them via the next_ops name.

  // Map current op output names to next_operator inputs
  wfc::IVector<winml::ILearningModelFeatureDescriptor> resolved_inputs;
  wfc::IMap<winrt::hstring, winrt::hstring> mapping;

  WINML_THROW_IF_FAILED(builder->UseModel()->ResolveOperatorInputs(
      operator_type.c_str(),
      input_decs,
      resolved_inputs,
      mapping));

  // Determine outputs
  wfc::IVector<winml::ILearningModelFeatureDescriptor> resolved_outputs;
  WINML_THROW_IF_FAILED(builder->UseModel()->InferOperatorOutputs(
      operator_name.c_str(),
      resolved_inputs,
      resolved_outputs));

  // Add the operator inputs and outputs
  std::vector<std::string> input_names(inputs_.Size());
  std::transform(begin(inputs_),
                 end(inputs_),
                 std::begin(input_names),
                 [](auto input) {
                   return WinML::Strings::UTF8FromHString(input.Name());
                 });

  std::vector<const char*> inputs(input_names.size());
  std::transform(std::begin(input_names),
                 std::end(input_names),
                 std::begin(inputs),
                 [](auto input) {
                   return input.c_str();
                 });

  std::vector<std::string> output_names(outputs_.Size());
  std::transform(begin(outputs_),
                 end(outputs_),
                 std::begin(output_names),
                 [operator_name](auto output) {
                   return operator_name + "." + WinML::Strings::UTF8FromHString(output.Name());
                 });

  std::vector<const char*> outputs(output_names.size());
  std::transform(std::begin(output_names),
                 std::end(output_names),
                 std::begin(outputs),
                 [](auto output) {
                   return output.c_str();
                 });

  WINML_THROW_IF_FAILED(builder->UseModel()->AddOperator(
      operator_type.c_str(),
      operator_name.c_str(),
      inputs.data(), inputs.size(),
      outputs.data(), outputs.size()));
}

void LearningModelOperator::JoinAfter(more::LearningModelInputs const& inputs) {
  auto input_descriptors = inputs.as<morep::LearningModelInputs>()->Inputs().GetView();
  return JoinAfterInternal(input_descriptors);
}

void LearningModelOperator::JoinAfter(more::LearningModelOperator const& previous_operator) {
  auto output_descriptors = previous_operator.Outputs();
  return JoinAfterInternal(output_descriptors);
}

void LearningModelOperator::SetAttributeInternal(const char* const name, wf::IInspectable const& /*inspectable*/) {
  auto found_it = attributes_.find(name);
  if (found_it == std::end(attributes_)) {
    throw;
  }
  /*
    auto featureValue = WinML::CreateFeatureValueFromInspectable(WinML::BindingType::kInput, inspectable, found_it->second);
    // Validate that the feature value is compatible with the descriptor
    WinML::VerifyFeatureValueCompatibleWithDescriptor(featureValue, found_it->second);
    
    auto spLotusValueProvider = featureValue.as<WinML::ILotusValueProviderPrivate>();

    ////////
    //////// TODO: Need to create a fake IEngine that is not backed by a cpu session but no model in order to generate the appropriate values here.
    ////////

    // Create the Binding Context to pass to the feature value
    WinML::BindingContext context{
        WinML::BindingType::kInput,
        nullptr,
        found_it->second,
        nullptr,
        {}  // SubresourceId is set by callee
    };

    // Get the bound tensor
    winrt::com_ptr<WinML::IValue> value;
    spLotusValueProvider->GetValue(context, value.put());

    attribute_values_[name] = value;

    */
}

more::LearningModelOperator LearningModelOperator::Then(more::LearningModelOperator const& next_operator) {
  auto operator_impl = next_operator.as<morep::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

more::LearningModelOperator LearningModelOperator::Then(more::LearningModelOperator const& next_operator, more::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  auto operator_impl = next_operator.as<morep::LearningModelOperator>();
  operator_impl->SetBuilder(builder_);
  operator_impl->JoinAfter(*this);
  return next_operator;
}

more::LearningModelBuilder LearningModelOperator::ConnectToOutputs() {
  return builder_;
}

more::LearningModelBuilder LearningModelOperator::ConnectToOutputs(more::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  return builder_;
}

more::LearningModelOperator LearningModelOperator::SetAttribute(hstring const& name, Windows::Foundation::IInspectable const& value) {
  SetAttributeInternal(WinML::Strings::UTF8FromHString(name).c_str(), value);
  return *this;
}

Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> LearningModelOperator::Inputs() {
  return inputs_.GetView();
}

Windows::Foundation::Collections::IMapView<hstring, hstring> LearningModelOperator::InputMapping() {
  return input_mapping_.GetView();
}

Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> LearningModelOperator::Attributes() {
  return nullptr;
}

Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor> LearningModelOperator::Outputs() {
  return outputs_.GetView();
}

hstring LearningModelOperator::Name() {
  return name_;
}

more::LearningModelOperator LearningModelOperator::Gemm() {
  return winrt::make<morep::LearningModelOperator>(L"Gemm", winrt::hstring());
}

more::LearningModelOperator LearningModelOperator::Gemm(hstring const& name) {
  return winrt::make<morep::LearningModelOperator>(L"Gemm", name);
}
}  // namespace winrt::Windows::AI::MachineLearning::More::implementation
