#include "pch.h"
#include "LearningModelInputs.h"
#include "LearningModelOperator.h"
#include "LearningModelSession.h"
#include "LearningModelBuilder.h"
#include "TensorFeatureDescriptor.h"

#include "..\Api\inc\ILotusValueProviderPrivate.h"

namespace WINML_EXPERIMENTALP {

LearningModelInputs::LearningModelInputs(winml_experimental::LearningModelBuilder builder) : builder_(builder),
                                                                                      input_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      input_default_values_(winrt::single_threaded_vector<wf::IInspectable>()),
                                                                                      constant_descriptors_(winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>()),
                                                                                      constant_values_(winrt::single_threaded_vector<wf::IInspectable>()) {
}

winml_experimental::LearningModelBuilder LearningModelInputs::AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant) {
  // Perform model update inside the builder
  auto model = builder_.as<winml_experimentalp::LearningModelBuilder>()->UseModel();
  auto descriptor_provider = input.as<_winml::IDescriptorInfoProvider>();
  auto input_name = _winml::Strings::UTF8FromHString(input.Name());
  winrt::com_ptr<_winml::IValue> default_value_ivalue;

  if (default_value) {
    auto default_value_value_provider = default_value.as<_winml::ILotusValueProviderPrivate>();
    // Create the Binding Context to pass to the feature value
    _winml::BindingContext context{
        _winml::BindingType::kInput,
        builder_.as<winml_experimentalp::LearningModelBuilder>()->InertSession(),
        nullptr,
        nullptr,
        {}  // SubresourceId is set by callee
    };
    default_value_value_provider->GetValue(context, default_value_ivalue.put());
  }

  model->AddModelInput(input_name.c_str(), descriptor_provider.get(), is_constant, default_value_ivalue.get());

  return builder_;
}

winml_experimental::LearningModelBuilder LearningModelInputs::Add(winml::ILearningModelFeatureDescriptor const& input) {
  return AddInput(input, nullptr, false);
}

winml_experimental::LearningModelBuilder LearningModelInputs::Add(hstring const& input_name, hstring const& input_description, Windows::Foundation::IInspectable const& default_value) {
  if (auto tensor = default_value.try_as<winml::ITensor>()) {
    auto shape = tensor.Shape();
    std::vector<int64_t> shape_vector(begin(shape), end(shape));
    auto descriptor = winrt::make<winmlp::TensorFeatureDescriptor>(input_name, input_description, tensor.TensorKind(), shape_vector);
    return AddInput(descriptor, default_value, false);
  }
  WINML_THROW_HR(E_UNEXPECTED);
}

winml_experimental::LearningModelBuilder LearningModelInputs::AddConstant(hstring const& input_name, Windows::Foundation::IInspectable const& value) {
  if (auto tensor = value.try_as<winml::ITensor>()) {
    winrt::hstring no_description_for_constants = L"";
    auto shape = tensor.Shape();
    std::vector<int64_t> shape_vector(begin(shape), end(shape));
    auto descriptor = winrt::make<winmlp::TensorFeatureDescriptor>(input_name, no_description_for_constants, tensor.TensorKind(), shape_vector);
    return AddInput(descriptor, value, true);
  }
  WINML_THROW_HR(E_UNEXPECTED);
}

}  // namespace WINML_EXPERIMENTALP