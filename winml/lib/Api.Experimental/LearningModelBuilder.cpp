#include "pch.h"
#include "LearningModelBuilder.h"
#include "LearningModel.h"

#include "LearningModelInputs.h"
#include "LearningModelOutputs.h"
#include "OnnxruntimeProvider.h"

namespace WINML_EXPERIMENTALP {

LearningModelBuilder::LearningModelBuilder() : inputs_(nullptr), outputs_(nullptr) {
  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(engine_factory_->CreateEmptyModel(model_.put()));
  inputs_ = winrt::make<winml_experimentalp::LearningModelInputs>(*this);
  outputs_ = winrt::make<winml_experimentalp::LearningModelOutputs>(*this);
}

LearningModelBuilder::LearningModelBuilder(LearningModelBuilder& builder) : inputs_(builder.inputs_),
                                                                            outputs_(builder.outputs_) {
}

winml_experimental::LearningModelInputs LearningModelBuilder::Inputs() {
  return inputs_;
}

winml_experimental::LearningModelOutputs LearningModelBuilder::Outputs() {
  return outputs_;
}

winml::LearningModel LearningModelBuilder::CreateModel() {
  return winrt::make<winmlp::LearningModel>(engine_factory_.get(), model_.get(), nullptr);
}

winml_experimental::LearningModelBuilder LearningModelBuilder::Create() {
  return winrt::make<LearningModelBuilder>();
}

winml_experimental::LearningModelOperator LearningModelBuilder::AfterAll(winml_experimental::LearningModelOperator const& /*target*/, wfc::IVectorView<winml_experimental::LearningModelOperator> const& /*input_operators*/) {
  throw hresult_not_implemented();
}

winml_experimental::LearningModelOperator LearningModelBuilder::AfterAll(winml_experimental::LearningModelOperator const& /*target*/, wfc::IVectorView<winml_experimental::LearningModelOperator> const& /*input_operators*/, winml_experimental::LearningModelOperatorResolutionPolicy const& /*policy*/) {
  throw hresult_not_implemented();
}

_winml::IModel* LearningModelBuilder::UseModel() {
  return model_.get();
}

}  // namespace WINML_EXPERIMENTALP
