/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Defines the structs involved in issuing a query and responding to
/// a query.
/// \details These are broken out into their own files since they are exposed
/// as part of the C API and we want to avoid C clients including C++ code.

#ifndef MLPERF_LOADGEN_QUERY_SAMPLE_H_
#define MLPERF_LOADGEN_QUERY_SAMPLE_H_

#include <stddef.h>
#include <stdint.h>

namespace mlperf {

/// \addtogroup LoadgenAPI
/// @{

/// \brief Represents a unique identifier for a sample of an issued query.
/// \details As currently implemented, the id is a pointer to an internal
/// loadgen struct whose value will never be zero/null.
typedef uintptr_t ResponseId;
constexpr ResponseId kResponseIdReserved = 0;

/// \brief An index into the QuerySampleLibrary corresponding to a
/// single sample.
typedef size_t QuerySampleIndex;

/// \brief Represents the smallest unit of input inference can run on.
/// A query consists of one or more samples.
struct QuerySample {
  ResponseId id;
  QuerySampleIndex index;
};

/// \brief Represents a single response to QuerySample
struct QuerySampleResponse {
  ResponseId id;
  uintptr_t data;
  size_t size;  ///< Size in bytes.
};

/// \brief A latency in nanoseconds, as recorded by the loadgen.
typedef int64_t QuerySampleLatency;

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_QUERY_SAMPLE_H_
