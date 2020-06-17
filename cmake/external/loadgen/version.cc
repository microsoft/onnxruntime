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
/// \brief Non-generated version logic.

#include "version.h"

#include "logging.h"

namespace mlperf {

void LogLoadgenVersion() {
  LogDetail([](AsyncDetail &detail) {
    detail("LoadgenVersionInfo:");
    detail("version : " + LoadgenVersion() + " @ " + LoadgenGitRevision());
    detail("build_date_local : " + LoadgenBuildDateLocal());
    detail("build_date_utc   : " + LoadgenBuildDateUtc());
    detail("git_commit_date  : " + LoadgenGitCommitDate());
    detail("git_log :\n\n" + LoadgenGitLog() + "\n");
    detail("git_status :\n\n" + LoadgenGitStatus() + "\n");
    if (!LoadgenGitStatus().empty() && LoadgenGitStatus() != "NA") {
      detail.Error("Loadgen built with uncommitted changes!");
    }
    detail("SHA1 of files :\n\n" + LoadgenSha1OfFiles() + "\n");
  });
}

}  // namespace mlperf
