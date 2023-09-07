#pragma once
#include "LearningModelJoinOptions.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelJoinOptions : LearningModelJoinOptionsT<LearningModelJoinOptions> {
  LearningModelJoinOptions();

  bool PromoteUnlinkedOutputsToFusedOutputs();
  void PromoteUnlinkedOutputsToFusedOutputs(bool value);
  winrt::hstring JoinedNodePrefix();
  void JoinedNodePrefix(hstring const& join_prefix);

  bool CloseModelOnJoin();
  void CloseModelOnJoin(bool value);
  void Link(hstring const& firstModelOutput, hstring const& secondModelInput);

 public:
  const std::unordered_map<std::string, std::string>& GetLinkages();

 private:
  bool promote_unlinked_outputs_ = true;
  bool promote_unlinked_inputs_ = true;
  bool close_model_on_link_ = false;

  std::unordered_map<std::string, std::string> linkages_;
  std::string join_prefix_;
};
}  // namespace WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelJoinOptions
  : LearningModelJoinOptionsT<LearningModelJoinOptions, implementation::LearningModelJoinOptions> {};

}  // namespace WINML_EXPERIMENTAL::factory_implementation
