#pragma once
#include <string>

#include <cassert>
#include <cstdbool>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"
#include "onnxruntime_config.h"
#ifdef HAVE_GLOB_H
#include <glob.h>
#include <fnmatch.h>
namespace onnxruntime {

// The two input strings must be null terminated
inline bool FnmatchSimple(std::basic_string_view<ORTCHAR_T> pattern_str, std::basic_string_view<ORTCHAR_T> input_string) {
  return fnmatch(pattern_str.data(), input_string.data(), FNM_PATHNAME | FNM_PERIOD) == 0;
}

inline std::vector<std::filesystem::path> SimpleGlob(const std::filesystem::path& dir, std::basic_string_view<ORTCHAR_T> pattern_string) {
  std::ostringstream oss;
  oss << dir.string() << "/" << pattern_string;
  std::string path_string = oss.str();
  std::vector<std::filesystem::path> ret;
  glob_t glob_state;
  memset(&glob_state, 0, sizeof(glob_t));
  int glob_ret = glob(path_string.c_str(), 0, nullptr, &glob_state);
  if (glob_ret != GLOB_NOMATCH) {
    for (size_t i = 0; glob_state.gl_pathv[i] != nullptr; ++i) {
      ret.push_back(glob_state.gl_pathv[i]);
    }
  }
  globfree(&glob_state);
  if (glob_ret == GLOB_NOMATCH || glob_ret == 0) {
    return ret;
  }
  ORT_THROW("glob error");
}

}  // namespace onnxruntime

#else
namespace onnxruntime {
class FnMatchInternalState {
 public:
  const ORTCHAR_T* pattern = nullptr;
  const ORTCHAR_T* string = nullptr;
};

inline bool FnMatchRec(const ORTCHAR_T* p, const ORTCHAR_T* n, const ORTCHAR_T* string_end,
                       FnMatchInternalState& ends) {
  for (ORTCHAR_T c; (c = *p) != '\0'; ++p, ++n) {
    if (c == '?') {
      if (n == string_end) return false;
    } else if (c == '*') {
      ends.pattern = p;
      ends.string = n;
      return true;
    } else {
      if (n == string_end || c != (*n)) return false;
    }
  }
  return n == string_end;
}

// The two input strings must be null terminated and they should not contain a path separator.
// TODO: support the FNM_PERIOD flag.
inline bool FnmatchSimple(std::basic_string_view<ORTCHAR_T> pattern_str, std::basic_string_view<ORTCHAR_T> input_string) {
  const ORTCHAR_T* string_end = input_string.data() + input_string.size();
  const ORTCHAR_T *p = pattern_str.data(), *n = input_string.data();
  ORTCHAR_T c;

  while ((c = *p++) != '\0') {
    {
      if (c == '?') {
        if (n == string_end) return false;
      } else if (c == '*') {
        for (c = *p++; c == '?' || c == '*'; c = *p++) {
          if (c == '?') {
            if (n == string_end)
              return false;
            else
              ++n;
          }
        }
        assert(c == *(p - 1));
        if (c == '\0') return true;

        FnMatchInternalState end;

        end.pattern = nullptr;
        for (--p; n < string_end; ++n) {
          if (((ORTCHAR_T)*n) == c && FnMatchRec(p + 1, n + 1, string_end, end)) {
            if (end.pattern == nullptr) return true;
            break;
          }
        }
        if (end.pattern != nullptr) {
          p = end.pattern;
          n = end.string;
          continue;
        }

        return false;
      } else {
        if (n == string_end || c != *n) return false;
      }
    }
    ++n;
  }

  return n == string_end;
}

inline std::vector<std::filesystem::path> SimpleGlob(const std::filesystem::path& dir, std::basic_string_view<ORTCHAR_T> pattern_string) {
  std::vector<std::filesystem::path> ret;
  if (!std::filesystem::exists(dir)) return ret;
  for (auto const& dir_entry : std::filesystem::directory_iterator(dir)) {
    if (!dir_entry.is_regular_file()) continue;
    const std::filesystem::path& path = dir_entry.path();
    std::filesystem::path fname = path.filename();
    if (fname.empty() || fname.native()[0] == ORT_TSTR('.')) {
      // Ignore hidden files.
      continue;
    }
    if (FnmatchSimple(pattern_string, fname.native())) {
      ret.push_back(path);
    }
  }
  return ret;
}

}  // namespace onnxruntime

#endif
