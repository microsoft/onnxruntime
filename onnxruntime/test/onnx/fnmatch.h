#pragma once
#include <string>

#include <cassert>
#include <cstdbool>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class FnMatchInternalState {
 public:
  const ORTCHAR_T* pattern = nullptr;
  const ORTCHAR_T* string = nullptr;
};

inline bool FnMatchRec(const ORTCHAR_T*  p, const ORTCHAR_T*  n, const ORTCHAR_T*  string_end,
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
    std::filesystem::path fname= path.filename();
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