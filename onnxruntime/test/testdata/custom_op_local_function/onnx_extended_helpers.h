#pragma once

#include <algorithm>
#include <float.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onnx_extended_helpers {

std::string Version();

class StringStream {
public:
  StringStream();
  virtual ~StringStream();
  virtual StringStream &append_uint16(const uint16_t &obj);
  virtual StringStream &append_uint32(const uint32_t &obj);
  virtual StringStream &append_uint64(const uint64_t &obj);
  virtual StringStream &append_int16(const int16_t &obj);
  virtual StringStream &append_int32(const int32_t &obj);
  virtual StringStream &append_int64(const int64_t &obj);
  virtual StringStream &append_float(const float &obj);
  virtual StringStream &append_double(const double &obj);
  virtual StringStream &append_char(const char &obj);
  virtual StringStream &append_string(const std::string &obj);
  virtual StringStream &append_charp(const char *obj);
  virtual std::string str();
  static StringStream *NewStream();
};

std::vector<std::string> SplitString(const std::string &input, char delimiter);

void MakeStringInternalElement(StringStream &ss, const char *t);

void MakeStringInternalElement(StringStream &ss, const std::string &t);

void MakeStringInternalElement(StringStream &ss, const char &t);

void MakeStringInternalElement(StringStream &ss, const uint16_t &t);
void MakeStringInternalElement(StringStream &ss, const uint32_t &t);
void MakeStringInternalElement(StringStream &ss, const uint64_t &t);

void MakeStringInternalElement(StringStream &ss, const int16_t &t);
void MakeStringInternalElement(StringStream &ss, const int32_t &t);
void MakeStringInternalElement(StringStream &ss, const int64_t &t);

void MakeStringInternalElement(StringStream &ss, const float &t);

void MakeStringInternalElement(StringStream &ss, const double &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<uint16_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<uint32_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<uint64_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<int16_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<int32_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<int64_t> &t);

void MakeStringInternalElement(StringStream &ss, const std::vector<float> &t);

void MakeStringInternal(StringStream &ss);

template <typename T, typename... Args>
inline void MakeStringInternal(StringStream &ss, const T &t) {
  MakeStringInternalElement(ss, t);
}

template <typename T, typename... Args>
inline void MakeStringInternal(StringStream &ss, const T &t, const Args &...args) {
  MakeStringInternalElement(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args> inline std::string MakeString(const Args &...args) {
  StringStream *ss = StringStream::NewStream();
  MakeStringInternal(*ss, args...);
  std::string res = ss->str();
  delete ss;
  return res;
}

#if !defined(_THROW_DEFINED)
#define EXT_THROW(...)                                                                         \
  throw std::runtime_error(onnx_extended_helpers::MakeString(                                  \
      "[onnx-extended] ", onnx_extended_helpers::MakeString(__VA_ARGS__)));
#define _THROW_DEFINED
#endif

#if !defined(_ENFORCE_DEFINED)
#define EXT_ENFORCE(cond, ...)                                                                 \
  if (!(cond))                                                                                 \
    throw std::runtime_error(onnx_extended_helpers::MakeString(                                \
        "`", #cond, "` failed. ",                                                              \
        onnx_extended_helpers::MakeString("[onnx-extended] ",                                  \
                                          onnx_extended_helpers::MakeString(__VA_ARGS__))));
#define _ENFORCE_DEFINED
#endif

} // namespace onnx_extended_helpers
