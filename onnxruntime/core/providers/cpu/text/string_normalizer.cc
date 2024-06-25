// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalizer.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
// Used below HAS_DEPRECATED_DECLARATIONS
#include "onnxruntime_config.h"

#ifdef _MSC_VER
#include <locale.h>
#endif  // _MSC_VER

#include <codecvt>
#include <locale>
#include <functional>

#if defined(__GNUC__)
// Allow deprecated-declarations warning - std::codecvt_utf8 is deprecatedd
#if defined(HAS_DEPRECATED_DECLARATIONS)
#pragma GCC diagnostic warning "-Wdeprecated-declarations"
#endif  // defined(HAS_DEPRECATED_DECLARATIONS)
#endif  // defined(__GNUC__)

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    StringNormalizer,
    10,
    KernelDefBuilder()
        .TypeConstraint("X", DataTypeImpl::GetTensorType<std::string>()),
    StringNormalizer);

namespace string_normalizer {

// codecvt_utf8 is deprecated, we will want to replace it with our class
class Utf8ConverterGeneric {
 public:
  size_t ComputeRequiredSizeToUtf8(const std::wstring& wstr) const {
    if (wstr.empty()) {
      return 0;
    }

    size_t result = 0;
    std::mbstate_t state = std::mbstate_t();

    const wchar_t* src = wstr.data();
    const wchar_t* src_end = src + wstr.length();

    char dummy_dest[128] = {0};

    char* char_next = dummy_dest;
    const wchar_t* wchar_next = src;

    size_t converted = 0;

    std::codecvt_base::result ret_code = std::codecvt_base::ok;

    // Continue while we exhaust the sequence
    while (converted < wstr.length()) {
      ret_code = converter_.out(state,
                                wchar_next,
                                src_end,
                                wchar_next,
                                std::begin(dummy_dest),
                                std::end(dummy_dest),
                                char_next);
      result += (char_next - dummy_dest);
      converted = (wchar_next - src);

      if (ret_code != std::codecvt_base::partial &&
          ret_code != std::codecvt_base::ok) {
        break;
      }
    }

    ORT_ENFORCE(ret_code != std::codecvt_base::noconv, "Conversion is expected");

    if (ret_code != std::codecvt_base::ok) {
      ORT_THROW("Failed to compute size for UTF-8. Converted only first: ",
                converted, " codepoints out of: ", wstr.length());
    }

    return result;
  }

  // We assume the caller pre-allocated the correct length
  Status ConvertToUtf8(const std::wstring& wstr, std::string& str) const {
    if (wstr.empty()) {
      str.clear();
      return Status::OK();
    }

    std::mbstate_t state = std::mbstate_t();

    const wchar_t* src = wstr.data();
    const wchar_t* src_end = src + wstr.length();

    char* dest = str.data();
    char* dest_end = dest + str.length();

    char* char_next = dest;
    const wchar_t* wchar_next = src;

    std::codecvt_base::result ret_code = converter_.out(state,
                                                        src,
                                                        src_end,
                                                        wchar_next,
                                                        dest,
                                                        dest_end,
                                                        char_next);

    if (ret_code != std::codecvt_base::ok) {
      size_t converted = narrow<size_t>(wchar_next - wstr.data());
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to convert to UTF-8. Converted only first: ",
                             converted, " codepoints out of: ", wstr.length());
    }

    str.resize(char_next - dest);

    return Status::OK();
  }

  Status ComputeRequiredSizeToWideChar(const std::string& str, size_t& wchars) {
    if (str.empty()) {
      wchars = 0;
      return Status::OK();
    }

    size_t result = 0;
    std::mbstate_t state = std::mbstate_t();

    const char* src = str.data();
    const char* src_end = src + str.length();

    wchar_t dummy_dest[128] = {0};
    const char* char_next = src;
    wchar_t* wchar_next = dummy_dest;

    size_t converted = 0;

    std::codecvt_base::result ret_code = std::codecvt_base::ok;
    while (converted < str.length()) {
      ret_code = converter_.in(state,
                               char_next,
                               src_end,
                               char_next,
                               std::begin(dummy_dest),
                               std::end(dummy_dest),
                               wchar_next);
      result += (wchar_next - dummy_dest);
      converted = (char_next - src);

      if (ret_code != std::codecvt_base::partial &&
          ret_code != std::codecvt_base::ok) {
        break;
      }
    }

    ORT_ENFORCE(ret_code != std::codecvt_base::noconv, "Conversion is expected");

    if (ret_code != std::codecvt_base::ok) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to compute buffer size for wchar_t. Converted only first: ",
                             converted, " bytes out of: ", str.length(),
                             " Source: ", src);
    }

    wchars = result;
    return Status::OK();
  }

  // We assume the destination buffer is preallocated correctly
  Status ConvertToWideChar(const std::string& str, std::wstring& wstr) {
    if (str.empty()) {
      // Preserve the buffer for re-use, just set size to 0
      wstr.clear();
      return Status::OK();
    }

    std::mbstate_t state = std::mbstate_t();
    const char* src = str.data();
    const char* src_end = src + str.length();

    wchar_t* dest = wstr.data();
    wchar_t* dest_end = dest + wstr.length();

    const char* char_next = src;
    wchar_t* wchar_next = dest;

    std::codecvt_base::result ret_code = converter_.in(state,
                                                       src,
                                                       src_end,
                                                       char_next,
                                                       dest,
                                                       dest_end,
                                                       wchar_next);

    if (ret_code != std::codecvt_base::ok) {
      size_t converted = narrow<size_t>(char_next - str.data());
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to convert to wchar_t. Converted only first: ",
                             converted, " bytes out of: ", str.length(),
                             " Source: ", src);
    }

    wstr.resize(wchar_next - dest);

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

 private:
#if __cplusplus >= 202002L
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  std::codecvt_utf8<wchar_t> converter_;
#if __cplusplus >= 202002L
#pragma GCC diagnostic pop
#endif
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

#if defined(__APPLE__) || defined(__ANDROID__)

using Utf8Converter = Utf8ConverterGeneric;

#else

using Utf8Converter = Utf8ConverterGeneric;

#endif

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

  // Compute the largest widestring buffer needed.
  size_t max_wide_buffer_len = 0;
  for (const auto& s : input_span) {
    size_t wchars = 0;
    // Checks for invalid UTF-8 characters on Windows
    ORT_RETURN_IF_ERROR(converter.ComputeRequiredSizeToWideChar(s, wchars));
    max_wide_buffer_len = std::max(max_wide_buffer_len, wchars);
  }

  // Reuse reserved space
  std::wstring wchar_buffer;
  wchar_buffer.reserve(max_wide_buffer_len);

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

  Status status;

  if (is_case_sensitive_) {
    if (stopwords_.empty()) {
      assert(case_change_action_ != NONE);
      output_shape.push_back(C);
      status = output_no_filtering(output_shape);
    } else {
      // we need to filter
      InlinedVector<size_t> filtered_strings_indices;
      filtered_strings_indices.reserve(input_span.size());

      for (size_t i = 0, lim = input_span.size(); i < lim; ++i) {
        const std::string& s = input_span[i];
        if (stopwords_.count(s) == 0) {
          filtered_strings_indices.push_back(i);
        }
      }

      // According to the spec, if all strings are filtered out
      // the output must have a shape of {1} with a single empty string.
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
      // Case insensitive filtering is performed by converting the input strings
      // to compare_caseaction_. For that we convert to wchar_t UNICODE.
      // Otherwise, we need to pull ICU library on all platforms.
      InlinedVector<size_t> filtered_strings_indices;
      filtered_strings_indices.reserve(input_span.size());
      for (size_t i = 0, lim = input_span.size(); i < lim; ++i) {
        const std::string& s = input_span[i];
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWideChar(s, wchar_buffer));
        locale.ChangeCase(compare_caseaction_, wchar_buffer);
        if (wstopwords_.count(wchar_buffer) == 0) {
          filtered_strings_indices.push_back(i);
        }
      }

      // According to the spec, if all strings are filtered out
      // the output must have a shape of {1} with a single empty string.
      const int64_t filtered_count = std::max<int64_t>(1, narrow<int64_t>(filtered_strings_indices.size()));
      output_shape.push_back(filtered_count);
      status = output_filtered(output_shape, filtered_strings_indices);
    }
  }

  return status;
}
}  // namespace onnxruntime
