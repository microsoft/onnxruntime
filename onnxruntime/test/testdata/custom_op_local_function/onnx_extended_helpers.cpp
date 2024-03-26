#include "onnx_extended_helpers.h"
#include <algorithm>
#include <float.h>
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>

namespace onnx_extended_helpers {

std::string Version() {
  auto s = MakeString("onnx-extended", 1, 1.1, 1.1f, "de", std::vector<int>{1},
                      std::vector<float>{1.1f});
  auto s2 = MakeString("Unable to allocate ", 5, " bytes on GPU.");
  return s + s2;
}

StringStream::StringStream() {}
StringStream::~StringStream() {}
StringStream &StringStream::append_string(const std::string &) { return *this; }
StringStream &StringStream::append_uint16(const uint16_t &) { return *this; }
StringStream &StringStream::append_uint32(const uint32_t &) { return *this; }
StringStream &StringStream::append_uint64(const uint64_t &) { return *this; }
StringStream &StringStream::append_int16(const int16_t &) { return *this; }
StringStream &StringStream::append_int32(const int32_t &) { return *this; }
StringStream &StringStream::append_int64(const int64_t &) { return *this; }
StringStream &StringStream::append_float(const float &) { return *this; }
StringStream &StringStream::append_double(const double &) { return *this; }
StringStream &StringStream::append_char(const char &) { return *this; }
StringStream &StringStream::append_charp(const char *) { return *this; }
std::string StringStream::str() { return std::string(); }

class StringStreamStd : public StringStream {
public:
  StringStreamStd() : StringStream() {}
  virtual ~StringStreamStd() {}
  virtual StringStream &append_uint16(const uint16_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_uint32(const uint32_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_uint64(const uint64_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_int16(const int16_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_int32(const int32_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_int64(const int64_t &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_float(const float &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_double(const double &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_char(const char &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_string(const std::string &obj) {
    stream_ << obj;
    return *this;
  }
  virtual StringStream &append_charp(const char *obj) {
    stream_ << obj;
    return *this;
  }
  virtual std::string str() { return stream_.str(); }

private:
  std::stringstream stream_;
};

StringStream *StringStream::NewStream() { return new StringStreamStd(); }

std::vector<std::string> SplitString(const std::string &input, char delimiter) {
  std::vector<std::string> parts;
  std::string::size_type start = 0;
  std::string::size_type end = input.find(delimiter);

  std::size_t n = 0;
  for (auto it : input) {
    if (it == delimiter)
      ++n;
  }
  parts.reserve(n + 1);

  while (end != std::string::npos) {
    parts.emplace_back(input.substr(start, end - start));
    start = end + 1;
    end = input.find(delimiter, start);
  }

  parts.emplace_back(input.substr(start));
  return parts;
}

void MakeStringInternal(StringStream &) {}

void MakeStringInternalElement(StringStream &ss, const std::string &t) { ss.append_string(t); }

void MakeStringInternalElement(StringStream &ss, const char *t) { ss.append_charp(t); }

void MakeStringInternalElement(StringStream &ss, const char &t) { ss.append_char(t); }

void MakeStringInternalElement(StringStream &ss, const uint16_t &t) { ss.append_uint16(t); }

void MakeStringInternalElement(StringStream &ss, const uint32_t &t) { ss.append_uint32(t); }
void MakeStringInternalElement(StringStream &ss, const uint64_t &t) { ss.append_uint64(t); }

void MakeStringInternalElement(StringStream &ss, const int16_t &t) { ss.append_int16(t); }

void MakeStringInternalElement(StringStream &ss, const int32_t &t) { ss.append_int32(t); }

void MakeStringInternalElement(StringStream &ss, const int64_t &t) { ss.append_int64(t); }

void MakeStringInternalElement(StringStream &ss, const float &t) { ss.append_float(t); }

void MakeStringInternalElement(StringStream &ss, const double &t) { ss.append_double(t); }

void MakeStringInternalElement(StringStream &ss, const std::vector<uint16_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_uint16(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<uint32_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_uint32(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<uint64_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_uint64(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<int16_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_int16(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<int32_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_int32(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<int64_t> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_int64(it);
  }
}

void MakeStringInternalElement(StringStream &ss, const std::vector<float> &t) {
  for (auto it : t) {
    ss.append_charp("x");
    ss.append_float(it);
  }
}

} // namespace onnx_extended_helpers
