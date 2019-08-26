// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#include "SampleAdd.h"

namespace Microsoft {
namespace Featurizer {
namespace SampleAdd {

// ----------------------------------------------------------------------
// |
// |  Transformer
// |
// ----------------------------------------------------------------------
Transformer::Transformer(std::uint16_t delta) :
    _delta(delta) {
}

Transformer::return_type Transformer::transform(arg_type const &arg) const /*override*/ {
    return _delta + arg;
}

// ----------------------------------------------------------------------
// |
// |  Estimator
// |
// ----------------------------------------------------------------------
Estimator & Estimator::fit_impl(apache_arrow const &data) /*override*/ {
    _accumulated_delta += static_cast<std::uint16_t>(data);
    return *this;
}

Estimator::TransformerUniquePtr Estimator::commit_impl(void) /*override*/ {
  return std::make_unique<SampleAdd::Transformer>(static_cast<std::uint16_t>(_accumulated_delta));
}

} // namespace SampleAdd
} // namespace Featurizer
} // namespace Microsoft
