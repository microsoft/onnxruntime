#include "pch.h"
#include "LearningModelBuilder.h"
#include "LearningModel.h"

#include "LearningModelInputs.h"
#include "LearningModelOutputs.h"
#include "OnnxruntimeProvider.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation {

LearningModelBuilder::LearningModelBuilder() : inputs_(nullptr), outputs_(nullptr) {
  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(engine_factory_->CreateEmptyModel(model_.put()));
  inputs_ = winrt::make<morep::LearningModelInputs>(*this);
  outputs_ = winrt::make<morep::LearningModelOutputs>(*this);
}

LearningModelBuilder::LearningModelBuilder(LearningModelBuilder& builder) : inputs_(builder.inputs_),
                                                                            outputs_(builder.outputs_) {
}

more::LearningModelInputs LearningModelBuilder::Inputs() {
  return inputs_;
}

more::LearningModelOutputs LearningModelBuilder::Outputs() {
  return outputs_;
}

winml::LearningModel LearningModelBuilder::CreateModel() {
  return winrt::make<winmlp::LearningModel>(engine_factory_.get(), model_.get(), nullptr);
}

more::LearningModelBuilder LearningModelBuilder::Create() {
  return winrt::make<LearningModelBuilder>();
}

more::LearningModelOperator LearningModelBuilder::AfterAll(more::LearningModelOperator const& /*target*/, wfc::IVectorView<more::LearningModelOperator> const& /*input_operators*/) {
  throw hresult_not_implemented();
}

more::LearningModelOperator LearningModelBuilder::AfterAll(more::LearningModelOperator const& /*target*/, wfc::IVectorView<more::LearningModelOperator> const& /*input_operators*/, more::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  throw hresult_not_implemented();
}

WinML::IModel* LearningModelBuilder::UseModel() {
  return model_.get();
}

}  // namespace winrt::Windows::AI::MachineLearning::More::implementation
