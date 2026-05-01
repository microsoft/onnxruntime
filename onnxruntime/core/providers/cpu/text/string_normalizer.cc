// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_STRING_TYPE)

#include "string_normalizer.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#ifdef _MSC_VER
#include <Windows.h>
#include <locale.h>
#endif  // _MSC_VER

#include <locale>
#include <functional>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    StringNormalizer,
    10,
    KernelDefBuilder()
        .TypeConstraint("X", DataTypeImpl::GetTensorType<std::string>()),
    StringNormalizer);

namespace string_normalizer {

// Manual UTF-8 <-> wchar_t (UTF-32 on non-Windows) converter.
// Replaces the deprecated std::codecvt_utf8<wchar_t>.
class Utf8ConverterGeneric {
 public:
  size_t ComputeRequiredSizeToUtf8(const std::wstring& wstr) const {
    if (wstr.empty()) {
      return 0;
    }

    size_t result = 0;
    for (wchar_t wc : wstr) {
      char32_t cp = static_cast<char32_t>(wc);
      if (cp <= 0x7F) {
        result += 1;
      } else if (cp <= 0x7FF) {
        result += 2;
      } else if (cp <= 0xFFFF) {
        result += 3;
      } else if (cp <= 0x10FFFF) {
        result += 4;
      } else {
        ORT_THROW("Invalid Unicode codepoint U+", std::hex, static_cast<uint32_t>(cp));
      }
    }
    return result;
  }

  Status ConvertToUtf8(const std::wstring& wstr, std::string& str) const {
    if (wstr.empty()) {
      str.clear();
      return Status::OK();
    }

    char* dest = str.data();
    char* dest_end = dest + str.size();

    for (wchar_t wc : wstr) {
      char32_t cp = static_cast<char32_t>(wc);
      if (cp <= 0x7F) {
        if (dest >= dest_end) break;
        *dest++ = static_cast<char>(cp);
      } else if (cp <= 0x7FF) {
        if (dest + 1 >= dest_end) break;
        *dest++ = static_cast<char>(0xC0 | (cp >> 6));
        *dest++ = static_cast<char>(0x80 | (cp & 0x3F));
      } else if (cp <= 0xFFFF) {
        if (dest + 2 >= dest_end) break;
        *dest++ = static_cast<char>(0xE0 | (cp >> 12));
        *dest++ = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *dest++ = static_cast<char>(0x80 | (cp & 0x3F));
      } else if (cp <= 0x10FFFF) {
        if (dest + 3 >= dest_end) break;
        *dest++ = static_cast<char>(0xF0 | (cp >> 18));
        *dest++ = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        *dest++ = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        *dest++ = static_cast<char>(0x80 | (cp & 0x3F));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Invalid Unicode codepoint during UTF-8 conversion");
      }
    }

    str.resize(static_cast<size_t>(dest - str.data()));
    return Status::OK();
  }

  Status ComputeRequiredSizeToWideChar(const std::string& str, size_t& wchars) {
    if (str.empty()) {
      wchars = 0;
      return Status::OK();
    }

    size_t result = 0;
    const auto* src = reinterpret_cast<const unsigned char*>(str.data());
    const auto* src_end = src + str.size();

    while (src < src_end) {
      size_t byte_len = 0;
      if ((*src & 0x80) == 0) {
        byte_len = 1;
      } else if ((*src & 0xE0) == 0xC0) {
        byte_len = 2;
      } else if ((*src & 0xF0) == 0xE0) {
        byte_len = 3;
      } else if ((*src & 0xF8) == 0xF0) {
        byte_len = 4;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Invalid UTF-8 lead byte at offset ",
                               static_cast<size_t>(src - reinterpret_cast<const unsigned char*>(str.data())));
      }

      if (static_cast<size_t>(src_end - src) < byte_len) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Truncated UTF-8 sequence at offset ",
                               static_cast<size_t>(src - reinterpret_cast<const unsigned char*>(str.data())));
      }

      src += byte_len;
      ++result;
    }

    wchars = result;
    return Status::OK();
  }

  Status ConvertToWideChar(const std::string& str, std::wstring& wstr) {
    if (str.empty()) {
      wstr.clear();
      return Status::OK();
    }

    const auto* src = reinterpret_cast<const unsigned char*>(str.data());
    const auto* src_end = src + str.size();
    wchar_t* dest = wstr.data();

    while (src < src_end) {
      char32_t cp = 0;
      size_t byte_len = 0;

      if ((*src & 0x80) == 0) {
        cp = *src;
        byte_len = 1;
      } else if ((*src & 0xE0) == 0xC0) {
        byte_len = 2;
        if (static_cast<size_t>(src_end - src) < 2) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Truncated UTF-8 sequence");
        }
        cp = (static_cast<char32_t>(src[0] & 0x1F) << 6) |
             static_cast<char32_t>(src[1] & 0x3F);
      } else if ((*src & 0xF0) == 0xE0) {
        byte_len = 3;
        if (static_cast<size_t>(src_end - src) < 3) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Truncated UTF-8 sequence");
        }
        cp = (static_cast<char32_t>(src[0] & 0x0F) << 12) |
             (static_cast<char32_t>(src[1] & 0x3F) << 6) |
             static_cast<char32_t>(src[2] & 0x3F);
      } else if ((*src & 0xF8) == 0xF0) {
        byte_len = 4;
        if (static_cast<size_t>(src_end - src) < 4) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Truncated UTF-8 sequence");
        }
        cp = (static_cast<char32_t>(src[0] & 0x07) << 18) |
             (static_cast<char32_t>(src[1] & 0x3F) << 12) |
             (static_cast<char32_t>(src[2] & 0x3F) << 6) |
             static_cast<char32_t>(src[3] & 0x3F);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid UTF-8 lead byte");
      }

      *dest++ = static_cast<wchar_t>(cp);
      src += byte_len;
    }

    wstr.resize(static_cast<size_t>(dest - wstr.data()));
    return Status::OK();
  }

  std::wstring from_bytes(const std::string& s) {
    std::wstring result;

    size_t wchars = 0;
    ORT_THROW_IF_ERROR(ComputeRequiredSizeToWideChar(s, wchars));

    result.resize(wchars);
    ORT_THROW_IF_ERROR(ConvertToWideChar(s, result));
    return result;
  }
};

// We need to specialize for MS as there is
// a std::locale creation bug that affects different
// environments in a different way
#ifdef _MSC_VER

class Locale {
 public:
  explicit Locale(const std::string& name)
      : loc_(nullptr) {
    loc_ = _create_locale(LC_CTYPE, name.c_str());
    if (loc_ == nullptr) {
      ORT_THROW("Failed to construct locale with name:",
                name, ":", ":Please, install necessary language-pack-XX and configure locales");
    }
  }

  ~Locale() {
    if (loc_ != nullptr) {
      _free_locale(loc_);
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Locale);

  void ChangeCase(StringNormalizer::CaseAction caseaction,
                  std::wstring& wstr) const {
    assert(caseaction != StringNormalizer::NONE);
    if (caseaction == StringNormalizer::LOWER) {
      std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                     [this](wchar_t ch) { return ::_towlower_l(ch, loc_); });
    } else {
      std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                     [this](wchar_t ch) { return ::_towupper_l(ch, loc_); });
    }
  }

 private:
  _locale_t loc_;
};

class Utf8ConverterWindows {
 public:
  size_t ComputeRequiredSizeToUtf8(const std::wstring& wstr) const {
    if (wstr.empty()) {
      return 0;
    }

    int ret = WideCharToMultiByte(CP_UTF8,
                                  0,
                                  wstr.data(),
                                  narrow<int>(wstr.length()),  // We specify the length so no trailing zero terminator
                                  NULL,
                                  0,      // indicates we need the buffer size.
                                  NULL,   // Must be NULL for UTF-8
                                  NULL);  // Must be NULL for UTF-8

    // Failed. This is unlikely since the original UTF-8 to wchar_t succeeded.
    // So we throw.
    if (ret == 0) {
      const auto error_code = GetLastError();
      ORT_THROW("WideCharToMultiByte failed errcode = ",
                error_code, " - ",
                std::system_category().message(error_code));
    }

    return narrow<size_t>(ret);
  }

  Status ConvertToUtf8(const std::wstring& wstr, std::string& dest) const {
    if (wstr.empty()) {
      dest.clear();
      return Status::OK();
    }

    const int ret = WideCharToMultiByte(CP_UTF8, 0,
                                        wstr.data(),
                                        narrow<int>(wstr.length()),
                                        dest.data(),
                                        narrow<int>(dest.length()),
                                        nullptr,
                                        nullptr);

    if (ret == 0) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WideCharToMultiByte failed errcode = ",
                             error_code, " - ",
                             std::system_category().message(error_code));
    }

    dest.resize(narrow<size_t>(ret));
    return Status::OK();
  }

  Status ComputeRequiredSizeToWideChar(const std::string& str, size_t& wchars) {
    if (str.empty()) {
      wchars = 0;
      return Status::OK();
    }

    const int ret = MultiByteToWideChar(CP_UTF8,
                                        MB_ERR_INVALID_CHARS,
                                        str.data(),
                                        narrow<int>(str.length()),
                                        nullptr,
                                        0);

    if (ret == 0) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MultiByteToWideChar failed errcode = ",
                             error_code, " - ",
                             std::system_category().message(error_code));
    }

    wchars = narrow<size_t>(ret);
    return Status::OK();
  }

  Status ConvertToWideChar(const std::string& str, std::wstring& wstr) {
    if (str.empty()) {
      // Preserve the buffer for re-use, just set size to 0
      wstr.clear();
      return Status::OK();
    }

    const int ret = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                        str.data(),
                                        narrow<int>(str.length()),
                                        wstr.data(),
                                        narrow<int>(wstr.length()));

    if (ret == 0) {
      const auto error_code = GetLastError();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MultiByteToWideChar failed errcode = ",
                             error_code, " - ",
                             std::system_category().message(error_code));
    }

    wstr.resize(narrow<size_t>(ret));
    return Status::OK();
  }

  // Used only in the constructor to initialize stop_words
  std::wstring from_bytes(const std::string& s) {
    size_t size_required = 0;
    ORT_THROW_IF_ERROR(ComputeRequiredSizeToWideChar(s, size_required));
    std::wstring result;
    result.resize(size_required);
    ORT_THROW_IF_ERROR(ConvertToWideChar(s, result));
    return result;
  }
};

const std::string default_locale("en-US");

using Utf8Converter = Utf8ConverterWindows;

#else  // _MSC_VER

class Locale {
 public:
  explicit Locale(const std::string& name) {
    ORT_TRY {
      loc_ = std::locale(name.c_str());
    }
    ORT_CATCH(const std::runtime_error& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        ORT_THROW("Failed to construct locale with name:",
                  name, ":", e.what(), ":Please, install necessary language-pack-XX and configure locales");
      });
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Locale);

  void ChangeCase(StringNormalizer::CaseAction caseaction,
                  std::wstring& wstr) const {
    assert(caseaction != StringNormalizer::NONE);
    if (caseaction == StringNormalizer::LOWER) {
      std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                     [this](wchar_t ch) { return std::tolower(ch, loc_); });
    } else {
      std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                     [this](wchar_t ch) { return std::toupper(ch, loc_); });
    }
  }

 private:
  std::locale loc_;
};

using Utf8Converter = Utf8ConverterGeneric;

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE || TARGET_OS_SIMULATOR
const std::string default_locale("en-US.UTF-8");
#else
const std::string default_locale("en_US.UTF-8");  // Other kinds of Apple Platforms including MacOS, etc
#endif
#else
const std::string default_locale("en_US.UTF-8");  // All non-MS and not Apple
#endif

#endif  // _MSC_VER
}  // namespace string_normalizer

using namespace string_normalizer;

StringNormalizer::StringNormalizer(const OpKernelInfo& info) : OpKernel(info) {
  int64_t iscasesensitive = 0;
  Status status = info.GetAttr("is_case_sensitive", &iscasesensitive);
  ORT_ENFORCE(status.IsOK(), "attribute is_case_sensitive is not set");
  is_case_sensitive_ = iscasesensitive != 0;

  std::string case_change_action;
  status = info.GetAttr("case_change_action", &case_change_action);
  ORT_ENFORCE(status.IsOK(), "attribute case_change_action is not set");
  if (case_change_action == "LOWER") {
    case_change_action_ = LOWER;
  } else if (case_change_action == "UPPER") {
    case_change_action_ = UPPER;
  } else if (case_change_action == "NONE") {
    case_change_action_ = NONE;
  } else {
    ORT_ENFORCE(false, "attribute case_change_action has invalid value");
  }

  locale_name_ = info.GetAttrOrDefault("locale", default_locale);

  std::vector<std::string> stop_words = info.GetAttrsOrDefault<std::string>("stopwords");
  if (is_case_sensitive_) {
    stopwords_.reserve(stop_words.size());
    for (std::string& s : stop_words) {
      stopwords_.insert(std::move(s));
    }
  } else {
    Locale locale(locale_name_);
    Utf8Converter converter;
    wstopwords_.reserve(stop_words.size());
    for (std::string& s : stop_words) {
      std::wstring wstr = converter.from_bytes(s);
      locale.ChangeCase(compare_caseaction_, wstr);
      wstopwords_.insert(std::move(wstr));
    }
  }
}

Status StringNormalizer::Compute(OpKernelContext* ctx) const {
  using namespace string_normalizer;

  auto X = ctx->Input<Tensor>(0);
  auto input_dims = X->Shape().GetDims();
  auto input_span = X->DataAsSpan<std::string>();

  TensorShapeVector output_shape;
  int64_t C = 0;
  if (input_dims.size() == 1) {
    if (input_dims[0] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Single dimension value must be greater than 0");
    }
    C = input_dims[0];
  } else if (input_dims.size() == 2) {
    if (input_dims[0] != 1 || input_dims[1] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input dimensions are either[C > 0] or [1][C > 0] allowed");
    }
    output_shape.push_back(1);
    C = input_dims[1];
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either[C > 0] or [1][C > 0] allowed");
  }

  // Special case, no filtering and no case change
  if (case_change_action_ == NONE &&
      ((is_case_sensitive_ && stopwords_.empty()) ||
       (!is_case_sensitive_ && wstopwords_.empty()))) {
    output_shape.push_back(C);
    auto output_tensor = ctx->Output(0, output_shape);
    auto const output_data = output_tensor->MutableData<std::string>();
    std::copy(input_span.begin(), input_span.end(), output_data);
    return Status::OK();
  }

  // We need to know the result dimension, and for that we need to filter
  // the words first. If comparison mode is case sensitive, we just go ahead
  // and compare with the original strings. Otherwise, we need to convert the string
  // to widechar, lowercase it and then compare. Case-insensitive comparison is complicated
  // for UTF-8 and requires additional dependency.

  Locale locale(locale_name_);
  Utf8Converter converter;

  // Determine whether we need wchar conversion at all.
  // We need it if: (a) case change is requested, or (b) case-insensitive filtering.
  const bool needs_wchar = (case_change_action_ != NONE) || !is_case_sensitive_;

  size_t max_wide_buffer_len = 0;
  if (needs_wchar) {
    // UTF-8 byte count is an upper bound on wchar_t count: each codepoint requires
    // at least 1 byte but produces exactly 1 wchar_t (UTF-32) or at most 2 (UTF-16).
    // This avoids a full UTF-8 decode pass just to compute buffer sizes.
    for (const auto& s : input_span) {
      max_wide_buffer_len = std::max(max_wide_buffer_len, s.size());
    }
  }

  // Reuse reserved space
  std::wstring wchar_buffer;
  if (needs_wchar) {
    wchar_buffer.reserve(max_wide_buffer_len);
  }

  // Output everything and change case as required
  auto output_no_filtering = [&](const TensorShape& output_shape) {
    auto output_tensor = ctx->Output(0, output_shape);
    auto const output_data = output_tensor->MutableData<std::string>();
    for (size_t i = 0, lim = input_span.size(); i < lim; ++i) {
      const std::string& s = input_span[i];
      wchar_buffer.resize(max_wide_buffer_len);
      ORT_RETURN_IF_ERROR(converter.ConvertToWideChar(s, wchar_buffer));
      locale.ChangeCase(case_change_action_, wchar_buffer);

      auto& dest = output_data[i];
      size_t utf8_buffer_len = converter.ComputeRequiredSizeToUtf8(wchar_buffer);
      dest.resize(utf8_buffer_len);
      ORT_RETURN_IF_ERROR(converter.ConvertToUtf8(wchar_buffer, dest));
    }
    return Status::OK();
  };

  auto output_filtered = [&](const TensorShape& output_shape, gsl::span<const size_t> filtered_indices) {
    auto output_tensor = ctx->Output(0, output_shape);
    auto output_data = output_tensor->MutableData<std::string>();
    for (size_t i : filtered_indices) {
      const std::string& s = input_span[i];
      if (case_change_action_ != NONE) {
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWideChar(s, wchar_buffer));
        locale.ChangeCase(case_change_action_, wchar_buffer);

        auto& dest = *output_data++;
        size_t utf8_buffer_len = converter.ComputeRequiredSizeToUtf8(wchar_buffer);
        dest.resize(utf8_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToUtf8(wchar_buffer, dest));
      } else {
        *output_data++ = s;
      }
    }
    return Status::OK();
  };

  // Output filtered strings with pre-computed wide forms (avoids double conversion).
  auto output_filtered_with_wide = [&](const TensorShape& output_shape,
                                       gsl::span<const size_t> filtered_indices,
                                       InlinedVector<std::wstring>& wide_forms) {
    auto output_tensor = ctx->Output(0, output_shape);
    auto output_data = output_tensor->MutableData<std::string>();
    for (size_t idx = 0; idx < filtered_indices.size(); ++idx) {
      if (case_change_action_ != NONE) {
        // wide_forms were lowercased for comparison; re-convert from original and apply target case
        const std::string& s = input_span[filtered_indices[idx]];
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWideChar(s, wchar_buffer));
        locale.ChangeCase(case_change_action_, wchar_buffer);

        auto& dest = *output_data++;
        size_t utf8_buffer_len = converter.ComputeRequiredSizeToUtf8(wchar_buffer);
        dest.resize(utf8_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToUtf8(wchar_buffer, dest));
      } else {
        *output_data++ = input_span[filtered_indices[idx]];
      }
    }
    return Status::OK();
  };

  Status status;

  if (is_case_sensitive_) {
    if (stopwords_.empty()) {
      assert(case_change_action_ != NONE);
      output_shape.push_back(C);
      status = output_no_filtering(output_shape);
    } else {
      // Case-sensitive filtering: direct string compare, no wchar needed for comparison.
      InlinedVector<size_t> filtered_strings_indices;
      filtered_strings_indices.reserve(input_span.size());

      for (size_t i = 0, lim = input_span.size(); i < lim; ++i) {
        const std::string& s = input_span[i];
        if (stopwords_.count(s) == 0) {
          filtered_strings_indices.push_back(i);
        }
      }

      const int64_t filtered_count = std::max<int64_t>(1, narrow<int64_t>(filtered_strings_indices.size()));
      output_shape.push_back(filtered_count);
      status = output_filtered(output_shape, filtered_strings_indices);
    }
  } else {
    if (wstopwords_.empty()) {
      assert(case_change_action_ != NONE);
      output_shape.push_back(C);
      status = output_no_filtering(output_shape);
    } else {
      // Case insensitive filtering: convert to wchar_t and case-fold for comparison.
      // If case_change_action_ matches compare_caseaction_, we can reuse the wide form
      // directly for output, avoiding a second conversion.
      const bool can_reuse_wide = (case_change_action_ == compare_caseaction_);

      InlinedVector<size_t> filtered_strings_indices;
      InlinedVector<std::wstring> filtered_wide_forms;
      filtered_strings_indices.reserve(input_span.size());
      if (can_reuse_wide) {
        filtered_wide_forms.reserve(input_span.size());
      }

      for (size_t i = 0, lim = input_span.size(); i < lim; ++i) {
        const std::string& s = input_span[i];
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWideChar(s, wchar_buffer));
        locale.ChangeCase(compare_caseaction_, wchar_buffer);
        if (wstopwords_.count(wchar_buffer) == 0) {
          filtered_strings_indices.push_back(i);
          if (can_reuse_wide) {
            filtered_wide_forms.push_back(wchar_buffer);
          }
        }
      }

      const int64_t filtered_count = std::max<int64_t>(1, narrow<int64_t>(filtered_strings_indices.size()));
      output_shape.push_back(filtered_count);

      if (can_reuse_wide && case_change_action_ != NONE) {
        // Reuse the already case-converted wide forms directly for output
        auto output_tensor = ctx->Output(0, output_shape);
        auto output_data = output_tensor->MutableData<std::string>();
        for (size_t idx = 0; idx < filtered_strings_indices.size(); ++idx) {
          const std::wstring& wide = filtered_wide_forms[idx];
          auto& dest = *output_data++;
          size_t utf8_buffer_len = converter.ComputeRequiredSizeToUtf8(wide);
          dest.resize(utf8_buffer_len);
          ORT_RETURN_IF_ERROR(converter.ConvertToUtf8(wide, dest));
        }
        status = Status::OK();
      } else {
        status = output_filtered_with_wide(output_shape, filtered_strings_indices, filtered_wide_forms);
      }
    }
  }

  return status;
}
}  // namespace onnxruntime

#endif  // !defined(DISABLE_STRING_TYPE)
