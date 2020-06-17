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
/// \brief Declares the version-related strings that will be defined in
/// a version_generated.cc as created by version_generator.py.

#ifndef MLPERF_LOADGEN_VERSION_H
#define MLPERF_LOADGEN_VERSION_H

#include <string>

namespace mlperf {

// Non-generated.
void LogLoadgenVersion();

// Definitions generated at compile time.
const std::string& LoadgenVersion();
const std::string& LoadgenGitRevision();
const std::string& LoadgenBuildDateLocal();
const std::string& LoadgenBuildDateUtc();
const std::string& LoadgenGitCommitDate();
const std::string& LoadgenGitStatus();
const std::string& LoadgenGitLog();
const std::string& LoadgenSha1OfFiles();

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_VERSION_H
