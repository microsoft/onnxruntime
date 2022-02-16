#include "core/framework/static_kernel_def_hashes.h"

namespace onnxruntime {
std::optional<HashValue> GetHashValueFromStaticKernelHashMap(const std::string& op_type, int since_version) {
  // Layout tranformer can add new nodes to the graph.
  // Since layout transformation can happen in an extended build, if these nodes are not picked up and compiled by
  // NNAPI or other compiling EPs then we need a way to get the hashes for these nodes. Since the infrastructure
  // as well as op_schema required to generate these hashes is not available in an extended minimal build,
  // we maintain a static map of nodes to hash value. This hash value can then be used to retireive the
  // kernel for the given op.
  static std::unordered_map<std::string, HashValue> static_kernel_hashes{
      {"Transpose_1", 4324835766923221184ULL},
      {"Transpose_13", 17267477159887372848ULL},
      {"Squeeze_1", 12889825108950034784ULL},
      {"Squeeze_11", 14725795030460042064ULL},
      {"Squeeze_13", 16122603335179721968ULL},
      {"UnSqueeze_1", 15964030255371555232ULL},
      {"UnSqueeze_11", 16989589986691430224ULL},
      {"UnSqueeze_13", 9466011545409597224ULL},
      {"Gather_1", 625186873870077080ULL},
      {"Gather_11", 11761559382112736008ULL},
      {"Gather_13", 7462749543760614528ULL},
      {"Identity_1", 18001636502361632792ULL},
      {"Identity_13", 16879814636194901248ULL},
      {"Identity_14", 16515685968327103576ULL},
      {"Identity_16", 17661628575887109792ULL},
  };

  auto key = op_type + "_" + std::to_string(since_version);
  auto iter = static_kernel_hashes.find(key);
  if (iter != static_kernel_hashes.end()) {
    return iter->second;
  }

  return std::nullopt;
}
}