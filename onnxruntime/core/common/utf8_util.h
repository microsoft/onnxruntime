// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
namespace utf8_util {

// Returns the number of bytes in the utf8 character
// by analyzing its leading byte
inline bool utf8_bytes(unsigned char ch, size_t& len) {
  if ((ch & 0x80) == 0) {
    len = 1;
    return true;
  }
  if ((ch & 0xE0) == 0xC0) {
    len = 2;
    return true;
  }
  unsigned int result = (ch & 0xF0);
  if (result == 0xE0) {
    len = 3;
    return true;
  }
  if (result == 0xF0) {
    len = 4;
    return true;
  }
  return false;
}

// Computes length of the utf8 string in characters
inline bool utf8_len(const unsigned char* s, size_t bytes, size_t& len) {
  size_t result = 0;
  while (bytes > 0) {
    size_t char_bytes = 0;
    bool valid = utf8_bytes(*s, char_bytes);
    if (!valid || bytes < char_bytes) {
      return false;
    }
    bytes -= char_bytes;
    s += char_bytes;
    ++result;
  }
  len = result;
  return true;
}

inline bool utf8_validate(const unsigned char* s, size_t len, size_t& utf8_chars) {
  size_t utf8_len = 0;
  size_t idx = 0;
  while (idx < len) {
    size_t bytes = 0;
    auto ch = s[idx];
    if (utf8_bytes(ch, bytes)) {
      switch (bytes) {
        case 1:
          break;
        case 2: {
          if (++idx >= len || s[idx] < 0x80u || s[idx] > 0xBFu) {
            return false;
          }
        } break;  // 2
        case 3: {
          auto ch1 = s[idx];
          switch (ch1) {
            case 0xE0u:
              if (++idx >= len || s[idx] < 0xA0u || s[idx] > 0xBFu) {
                return false;
              }
              break;
            case 0xEDu:
              if (++idx >= len || s[idx] < 0x80u || s[idx] > 0x9Fu) {
                return false;
              }
              break;
            default: {
              if ((ch1 >= 0xE1u && ch1 <= 0xECu) ||
                  (ch1 >= 0xEEu && ch1 <= 0xEFu)) {
                if (++idx >= len || s[idx] < 0x80u || s[idx] > 0xBFu) {
                  return false;
                }
              } else {
                return false;
              }
            } break;
          }
          // validate byte 3
          if (++idx >= len || s[idx] < 0x80u || s[idx] > 0xBFu) {
            return false;
          }
        } break;  // 3
        case 4: {
          auto ch1 = s[idx];
          switch (ch1) {
            case 0xF0u: {
              if (++idx >= len || s[idx] < 0x90u || s[idx] > 0xBFu) {
                return false;
              }
            } break;
            case 0xF4u: {
              if (++idx >= len || s[idx] < 0x80u || s[idx] > 0x8Fu) {
                return false;
              }
            } break;
            default: {
              if (ch1 >= 0xF1u && ch1 <= 0xF3u) {
                if (++idx >= len || s[idx] < 0x80u || s[idx] > 0xBFu) {
                  return false;
                }
              } else {
                return false;
              }
            } break;
          }
          // validate bytes 3 and 4
          size_t stop = idx + 2;
          while (idx < stop) {
            if (++idx >= len || s[idx] < 0x80u || s[idx] > 0xBFu) {
              return false;
            }
          }
        } break;  // 4
        default:
          // no chars longer than 4
          return false;
      }  // switch bytes
      ++idx;
      ++utf8_len;
    } else {
      return false;
    }
  }
  // End index must match
  // the end of the last byte sequence.
  if (idx != len) {
    return false;
  }
  utf8_chars = utf8_len;
  return true;
}

}  // namespace utf8_util
}  // namespace onnxruntime
