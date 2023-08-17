#pragma once
#include "LearningModelExperimental.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelExperimental : LearningModelExperimentalT<LearningModelExperimental> {
  LearningModelExperimental() = default;

  LearningModelExperimental(Microsoft::AI::MachineLearning::LearningModel const& model);
  Microsoft::AI::MachineLearning::LearningModel JoinModel(
    Microsoft::AI::MachineLearning::LearningModel const& other,
    Microsoft::AI::MachineLearning::Experimental::LearningModelJoinOptions const& options
  );

  void Save(hstring const& file_name);

  void SetName(hstring const& model_name);

 private:
  Microsoft::AI::MachineLearning::LearningModel model_;
};

}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelExperimental
  : LearningModelExperimentalT<LearningModelExperimental, implementation::LearningModelExperimental> {};

}  // namespace WINML_EXPERIMENTAL::factory_implementation
