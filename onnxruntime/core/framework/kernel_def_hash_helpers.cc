// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_def_hash_helpers.h"

#include "core/framework/data_types_internal.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace utils {
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

// special case for node NHWC optimizer may insert when running in minimal build
std::optional<HashValue> GetInternalNhwcOpHash(const Node& node) {
  if (node.Domain() == kMSDomain) {
    const auto& op_type = node.OpType();
    const auto& input_0_type = *node.InputDefs()[0]->TypeAsProto();

    if (op_type == "QLinearConv") {
      // first input is a tensor. could be uint8 or int8
      bool is_uint8 = input_0_type.tensor_type().elem_type() == utils::ToTensorProtoElementType<uint8_t>();
      return is_uint8 ? 16835965565578160400ULL : 10904143578341560456ULL;
    } else if (op_type == "NhwcMaxPool") {
      // first input is a tensor. could be uint8 or int8
      bool is_uint8 = input_0_type.tensor_type().elem_type() == utils::ToTensorProtoElementType<uint8_t>();
      return is_uint8 ? 8512357837341844248ULL : 11773579655431087496ULL;
    }
  }

  return std::nullopt;
}

void UpdateHashForBackwardsCompatibility(HashValue& hash) {
  // map of old hash to new hash if we were forced to break backwards compatibility for a kernel registration
  //
  // If we need to update the hash for an existing registration, an entry needs to be added here to map the
  // old hash to the new. This should rarely be required as historically the only need for it was fixing
  // kernel registrations with invalid type constraints. Please carefully read through the information at the top of
  // onnxruntime/test/providers/kernel_def_hash_test.cc regarding how/when hashes might change and the best way to
  // address that.
  static const std::unordered_map<HashValue, HashValue> hashes{
      // old                   new                          domain, operator, opset[, type]
      {2832535737534577496ULL, 16708009824840936392ULL},    // kOnnxDomain, Dropout, 7
      {12198479371038564912ULL, 1718418059112844640ULL},    // kOnnxDomain, Scan, 9
      {2560955351529676608ULL, 3668627007850399040ULL},     // kOnnxDomain, Scan, 11
      {10232409728231027688ULL, 5212043150202938416ULL},    // kOnnxDomain, Not, 1
      {11912523891622051440ULL, 10225383741733918632ULL},   // kOnnxDomain, RoiAlign, 10, float
      {18084231515768318048ULL, 17022700455473327752ULL},   // kOnnxDomain, RoiAlign, 10, double
      {14033689580222898712ULL, 634727773751317256ULL},     // kOnnxDomain, GatherND, 11
      {646512416908411600ULL, 3064028185911332496ULL},      // kOnnxDomain, GatherND, 12
      {15019893097608892000ULL, 11311962292460032936ULL},   // kOnnxDomain, GatherND, 13
      {14259324427750852648ULL, 7767393334034626736ULL},    // kOnnxDomain, StringNormalizer, 10
                                                            // contrib ops
      {7642430665819070720ULL, 8620498355864235632ULL},     // kMSDomain, CropAndResize, 1
      {15019666093341768288ULL, 11924582339825775592ULL}};  // kMSDomain, GridSample, 1

  auto iter = hashes.find(hash);
  if (iter != hashes.cend()) {
    // hash was updated in newer version of ORT kernel registrations
    hash = iter->second;
  }
}

}  // namespace utils
}  // namespace onnxruntime
