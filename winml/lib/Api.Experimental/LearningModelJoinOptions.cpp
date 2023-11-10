#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelJoinOptions.h"

namespace WINML_EXPERIMENTALP {

LearningModelJoinOptions::LearningModelJoinOptions() {
  GUID guid;
  WINML_THROW_IF_FAILED(CoCreateGuid(&guid));

  OLECHAR* guidString;
  WINML_THROW_IF_FAILED(StringFromCLSID(guid, &guidString));

  join_prefix_ = winrt::to_string(winrt::hstring(guidString));
  // ensure memory is freed
  ::CoTaskMemFree(guidString);
}

bool LearningModelJoinOptions::PromoteUnlinkedOutputsToFusedOutputs() {
  return promote_unlinked_outputs_;
}
void LearningModelJoinOptions::PromoteUnlinkedOutputsToFusedOutputs(bool value) {
  promote_unlinked_outputs_ = value;
}
winrt::hstring LearningModelJoinOptions::JoinedNodePrefix() {
  return winrt::to_hstring(join_prefix_);
}
void LearningModelJoinOptions::JoinedNodePrefix(hstring const& join_prefix) {
  join_prefix_ = winrt::to_string(join_prefix);
}
bool LearningModelJoinOptions::CloseModelOnJoin() {
  return close_model_on_link_;
}
void LearningModelJoinOptions::CloseModelOnJoin(bool value) {
  close_model_on_link_ = value;
}
void LearningModelJoinOptions::Link(hstring const& firstModelOutput, hstring const& secondModelInput) {
  linkages_[winrt::to_string(firstModelOutput)] = winrt::to_string(secondModelInput);
}
const std::unordered_map<std::string, std::string>& LearningModelJoinOptions::GetLinkages() {
  return linkages_;
}

}// namespace WINML_EXPERIMENTALP
