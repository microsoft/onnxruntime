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
/// \brief Defines the QuerySampleLibrary interface.

#ifndef MLPERF_LOADGEN_QUERY_SAMPLE_LIBRARY_H
#define MLPERF_LOADGEN_QUERY_SAMPLE_LIBRARY_H

#include <memory>
#include <vector>
#include <string>

#include "query_sample.h"

namespace mlperf {

/// \addtogroup LoadgenAPI
/// @{

/// \brief The interface a client implements to coordinate with the loadgen
/// which samples should be loaded.
class QuerySampleLibrary {
 public:
  virtual ~QuerySampleLibrary() {}

  /// \brief A human readable name for the model.
  virtual const std::string& Name() const = 0;

  /// \brief Total number of samples in library.
  virtual size_t TotalSampleCount() = 0;

  /// \brief The number of samples that are guaranteed to fit in RAM.
  virtual size_t PerformanceSampleCount() = 0;

  /// \brief Loads the requested query samples into memory.
  /// \details Paired with calls to UnloadSamplesFromRam.
  /// In the MultiStream scenarios:
  ///   * Samples will appear more than once.
  ///   * SystemUnderTest::IssueQuery will only be called with a set of samples
  ///     that are neighbors in the vector of samples here, which helps
  ///     SUTs that need the queries to be contiguous.
  /// In all other scenarios:
  ///   * A previously loaded sample will not be loaded again.
  virtual void LoadSamplesToRam(
      const std::vector<QuerySampleIndex>& samples) = 0;

  /// \brief Unloads the requested query samples from memory.
  /// \details In the MultiStream scenarios:
  ///   * Samples may be unloaded the same number of times they were loaded;
  ///     however, if the implementation de-dups loaded samples rather than
  ///     loading samples into contiguous memory, it may unload a sample the
  ///     first time they see it unloaded without a refcounting scheme, ignoring
  ///     subsequent unloads. A refcounting scheme would also work, but is not
  ///     a requirement.
  /// In all other scenarios:
  ///   * A previously unloaded sample will not be unloaded again.
  virtual void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) = 0;
};

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_QUERY_SAMPLE_LIBRARY_H
