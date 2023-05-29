// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Note: This file should be self-contained, i.e., no implementation in a .cc file.
// This is to allow it to be used from code that doesn't otherwise have any dependencies on the ORT optimizers.

#pragma once

#include <array>

#include "core/graph/constants.h"
#include "core/graph/op_identifier.h"

namespace onnxruntime {

// This is a list of ops and their versions which layout transformations can potentially add to the graph.
// This is needed in minimal build since opschema is not available.
inline constexpr std::array kLayoutTransformationPotentiallyAddedOps = {
    // Note: these region_begin/end markers are used by tools/ci_build/reduce_op_kernels.py
    // @@region_begin(extended_minimal_build_required_kernels)@@

    // kOnnxDomain ops
    OpIdentifierWithStringViews{kOnnxDomain, "Gather", 1},
    OpIdentifierWithStringViews{kOnnxDomain, "Gather", 11},
    OpIdentifierWithStringViews{kOnnxDomain, "Gather", 13},
    OpIdentifierWithStringViews{kOnnxDomain, "Identity", 1},
    OpIdentifierWithStringViews{kOnnxDomain, "Identity", 13},
    OpIdentifierWithStringViews{kOnnxDomain, "Identity", 14},
    OpIdentifierWithStringViews{kOnnxDomain, "Identity", 16},
    OpIdentifierWithStringViews{kOnnxDomain, "Identity", 19},
    OpIdentifierWithStringViews{kOnnxDomain, "Squeeze", 1},
    OpIdentifierWithStringViews{kOnnxDomain, "Squeeze", 11},
    OpIdentifierWithStringViews{kOnnxDomain, "Squeeze", 13},
    OpIdentifierWithStringViews{kOnnxDomain, "Transpose", 1},
    OpIdentifierWithStringViews{kOnnxDomain, "Transpose", 13},
    OpIdentifierWithStringViews{kOnnxDomain, "Unsqueeze", 1},
    OpIdentifierWithStringViews{kOnnxDomain, "Unsqueeze", 11},
    OpIdentifierWithStringViews{kOnnxDomain, "Unsqueeze", 13},

#if !defined(DISABLE_CONTRIB_OPS)
    // kMSDomain ops
    OpIdentifierWithStringViews{kMSDomain, "NhwcMaxPool", 1},
    OpIdentifierWithStringViews{kMSDomain, "QLinearConv", 1},
#endif  // !defined(DISABLE_CONTRIB_OPS)

    // @@region_end(extended_minimal_build_required_kernels)@@
};

namespace detail {
// std::is_sorted is not constexpr in C++17, so use our own constexpr version for now
template <typename It, typename Compare>
constexpr bool IsSorted(It begin, It end, Compare cmp) {
  if (begin == end) return true;
  It curr = begin, next = begin;
  while (++next != end) {
    if (cmp(*next, *curr)) return false;
    curr = next;
  }
  return true;
}
}  // namespace detail

static_assert(detail::IsSorted(kLayoutTransformationPotentiallyAddedOps.begin(),
                               kLayoutTransformationPotentiallyAddedOps.end(),
                               std::less<OpIdentifierWithStringViews>{}),
              "kLayoutTransformationPotentiallyAddedOps entries must be in sorted order.");

}  // namespace onnxruntime
