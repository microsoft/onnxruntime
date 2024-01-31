// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Span.hpp"

namespace MILBlob {
namespace Util {

/**
    reinterpret_casts the underlying pointer in Span<SourceT> to Span<TargetT>. Callers are responsible for ensuring
    that SourceT can be interpreted as TargetT in a meaningful way as there are neither compile- nor run-time safety
    guards in place.
*/

template <typename TargetT, typename SourceT>
Span<TargetT> SpanCast(Span<SourceT> span)
{
    auto ptr = reinterpret_cast<TargetT*>(span.Data());
    auto size = (span.Size() * sizeof(SourceT)) / sizeof(TargetT);
    return Span<TargetT>(ptr, size);
}

}  // namespace Util
}  // namespace MILBlob
