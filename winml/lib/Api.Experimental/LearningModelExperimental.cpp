#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelExperimental.h"
#include "LearningModel.h"
#include "LearningModelJoinOptions.h"

namespace WINML_EXPERIMENTALP {

LearningModelExperimental::LearningModelExperimental(Microsoft::AI::MachineLearning::LearningModel const& model)
  : model_(model) {
}

winml::LearningModel LearningModelExperimental::JoinModel(
  winml::LearningModel const& other, winml_experimental::LearningModelJoinOptions const& options
) {
  telemetry_helper.LogApiUsage("LearningModelExperimental::JoinModel");

  auto modelp = model_.as<winmlp::LearningModel>();
  auto optionsp = options.as<winml_experimentalp::LearningModelJoinOptions>();

  modelp->JoinModel(
    other,
    optionsp->GetLinkages(),
    optionsp->PromoteUnlinkedOutputsToFusedOutputs(),
    optionsp->CloseModelOnJoin(),
    optionsp->JoinedNodePrefix()
  );

  return model_;
}

void LearningModelExperimental::Save(hstring const& file_name) {
  telemetry_helper.LogApiUsage("LearningModelExperimental::Save");
  auto modelp = model_.as<winmlp::LearningModel>();
  modelp->SaveToFile(file_name);
}

void LearningModelExperimental::SetName(hstring const& model_name) {
  auto modelp = model_.as<winmlp::LearningModel>();
  modelp->SetName(model_name);
}

}  // namespace WINML_EXPERIMENTALP
