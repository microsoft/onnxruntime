#include "core/optimizer/optimizer_utils.h"
#include "core/graph/constants.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace onnxruntime {

// Allow certain domains/ops. We don't know anything about unknown domains/ops (e.g. custom ops),
// so we have to assume that they are not deterministic, to be on the safe side.
// We could also allow other known domains (kMSDomain, kMSNchwcDomain, kMSFeaturizersDomain),
// as long as we verify which of their operations are non-deterministic and add them in the map below.
static const std::unordered_map<std::string, std::unordered_set<std::string>> kNonDeterministicOps =
{
  {kOnnxDomain, {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"}},
};

bool IsOperationDeterministic(const std::string& domain, const std::string& op) {
  auto itDomain = kNonDeterministicOps.find(domain);
  if (itDomain == kNonDeterministicOps.end()) {
    // Unknown domain. Assume the op is not deterministic.
    return false;
  }

  return itDomain->second.count(op) == 0;
}

}  // namespace onnxruntime
