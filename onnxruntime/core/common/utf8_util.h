// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
namespace utf8_util {

// For functions decode/utf_validate
// Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
// See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.

const uint32_t UTF8_ACCEPT = 0;
const uint32_t UTF8_REJECT = 1;

uint32_t decode(uint32_t* state, uint32_t* codep, uint32_t byte);

inline bool utf8_validate(const unsigned char* s, size_t len, size_t& utf8_chars) {
  size_t utf8_len = 0;
  uint32_t state = 0;
  uint32_t codepoint = 0;
  for (size_t idx = 0; idx < len; ++idx) {
    if (UTF8_ACCEPT == decode(&state, &codepoint, s[idx])) {
      ++utf8_len;
    }
  }
  utf8_chars = utf8_len;
  return state == UTF8_ACCEPT;
}

// Returns the number of bytes in the utf8 character
// by analyzing its leading byte
inline size_t utf8_bytes(unsigned char ch) {
  if ((ch & 0x80) == 0) return 1;
  if ((ch & 0xE0) == 0xC0) return 2;
  unsigned int result = (ch & 0xF0);
  if (result == 0xE0) return 3;
  if (result == 0xF0) return 4;
  ONNXRUNTIME_ENFORCE(false, "utf8_bytes failed");
}

}  // namespace utf8_util
}  // namespace onnxruntime
