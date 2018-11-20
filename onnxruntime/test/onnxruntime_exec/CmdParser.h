//
// Copyright (c) Microsoft Corporation.  All rights reserved.
// Licensed under the MIT License.
//

#pragma once

#include <string>
#include <map>

class CmdParser {
 public:
  CmdParser(int argc, const char* argsv[]) {
    if (argc > 2) {
      for (int i = 1; i < argc; i += 2) {
        cmd_map_.insert({argsv[i], argsv[i + 1]});
      }
    }
  }

  const std::string* GetCommandArg(const std::string& option) const {
    auto value = cmd_map_.find(option);
    if (value != cmd_map_.cend()) {
      return &value->second;
    }

    return nullptr;
  }

 private:
  std::map<const std::string, const std::string> cmd_map_;
};
