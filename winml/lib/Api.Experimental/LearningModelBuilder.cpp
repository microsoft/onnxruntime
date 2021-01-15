#include "pch.h"
#include "LearningModelBuilder.h"
#include "LearningModel.h"

#include "LearningModelInputs.h"
#include "LearningModelOutputs.h"
#include "LearningModelOperatorSet.h"
#include "OnnxruntimeProvider.h"

namespace WINML_EXPERIMENTALP {

LearningModelBuilder::LearningModelBuilder() : inputs_(nullptr), outputs_(nullptr), operators_(nullptr) {
  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(engine_factory_->CreateEmptyModel(model_.put()));
  inputs_ = winrt::make<winml_experimentalp::LearningModelInputs>(*this);
  outputs_ = winrt::make<winml_experimentalp::LearningModelOutputs>(*this);
  operators_ = winrt::make<winml_experimentalp::LearningModelOperatorSet>(*this);
}

LearningModelBuilder::LearningModelBuilder(LearningModelBuilder& builder) : inputs_(builder.inputs_),
                                                                            outputs_(builder.outputs_),
                                                                            operators_(builder.operators_) {
}

winml_experimental::LearningModelInputs LearningModelBuilder::Inputs() {
  return inputs_;
}

winml_experimental::LearningModelOutputs LearningModelBuilder::Outputs() {
  return outputs_;
}

winml_experimental::LearningModelOperatorSet LearningModelBuilder::Operators() {
  return operators_;
}

winml::LearningModel LearningModelBuilder::CreateModel() {
  com_ptr<_winml::IModel> model_clone;
  model_->CloneModel(model_clone.put());
  return winrt::make<winmlp::LearningModel>(engine_factory_.get(), model_clone.get(), nullptr);
}

void LearningModelBuilder::Save(const winrt::hstring& file_name) {
  model_->SaveModel(file_name.c_str(), file_name.size());
}

winml_experimental::LearningModelBuilder LearningModelBuilder::Create() {
  return winrt::make<LearningModelBuilder>();
}

_winml::IModel* LearningModelBuilder::UseModel() {
  return model_.get();
}

}  // namespace WINML_EXPERIMENTALP
