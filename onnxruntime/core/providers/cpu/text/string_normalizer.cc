// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalizer.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "onnxruntime_config.h"

#ifdef _MSC_VER
#include <codecvt>
#include <locale.h>
#elif defined(__APPLE__) || defined(__ANDROID__)
#include <codecvt>
#else
#include <limits>
#include <iconv.h>

#endif  // _MSC_VER

#include <locale>
#include <functional>
#include <unordered_set>

#if defined(__GNUC__)
// Allow deprecated-declarations warning - std::wstring_convert is deprecated.
// TODO find a suitable replacement
// Note: GNU libiconv (e.g., on Apple platforms) is not suitable due to its LGPL license.
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
const std::string conv_error("Conversion Error");
const std::wstring wconv_error(L"Conversion Error");

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

// using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;

class Utf8Converter {
 public:
  Utf8Converter() = default;

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

  Status ConvertToWchar(const std::string& str, std::wstring& wstr) {
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
    ORT_THROW_IF_ERROR(ConvertToWchar(s, result));
    return result;
  }
};

const std::string default_locale("en-US");

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

using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t> >;

#else

// All others (not Windows, Apple, or Android)
class Utf8Converter {
 public:
  Utf8Converter() = default;

  std::wstring from_bytes(const std::string& s) const {
    std::wstring result;
    if (s.empty()) {
      return result;
    }
    // Order of arguments is to, from
    auto icvt = iconv_open("WCHAR_T", "UTF-8");
    // CentOS is not happy with -1
    if (std::numeric_limits<iconv_t>::max() == icvt) {
      return wconv_error;
    }

    char* iconv_in = const_cast<char*>(s.c_str());
    size_t iconv_in_bytes = s.length();
    // Temporary buffer assumes 1 byte to 1 wchar_t
    // to make sure it is enough.
    const size_t buffer_len = iconv_in_bytes * sizeof(wchar_t);
    auto buffer = std::make_unique<char[]>(buffer_len);
    char* iconv_out = buffer.get();
    size_t iconv_out_bytes = buffer_len;
    auto ret = iconv(icvt, &iconv_in, &iconv_in_bytes, &iconv_out, &iconv_out_bytes);
    if (static_cast<size_t>(-1) == ret) {
      result = wconv_error;
    } else {
      size_t converted_bytes = buffer_len - iconv_out_bytes;
      assert((converted_bytes % sizeof(wchar_t)) == 0);
      result.assign(reinterpret_cast<const wchar_t*>(buffer.get()), converted_bytes / sizeof(wchar_t));
    }
    iconv_close(icvt);
    return result;
  }

  std::string to_bytes(const std::wstring& wstr) const {
    std::string result;
    if (wstr.empty()) {
      return result;
    }
    // Order of arguments is to, from
    auto icvt = iconv_open("UTF-8", "WCHAR_T");
    // CentOS is not happy with -1
    if (std::numeric_limits<iconv_t>::max() == icvt) {
      return conv_error;
    }

    // I hope this does not modify the incoming buffer
    wchar_t* non_const_in = const_cast<wchar_t*>(wstr.c_str());
    char* iconv_in = reinterpret_cast<char*>(non_const_in);
    size_t iconv_in_bytes = wstr.length() * sizeof(wchar_t);
    // Temp buffer, assume every code point converts into 3 bytes, this should be enough
    // We do not convert terminating zeros
    const size_t buffer_len = wstr.length() * 3;
    auto buffer = std::make_unique<char[]>(buffer_len);

    char* iconv_out = buffer.get();
    size_t iconv_out_bytes = buffer_len;
    auto ret = iconv(icvt, &iconv_in, &iconv_in_bytes, &iconv_out, &iconv_out_bytes);
    if (static_cast<size_t>(-1) == ret) {
      result = conv_error;
    } else {
      size_t converted_len = buffer_len - iconv_out_bytes;
      result.assign(buffer.get(), converted_len);
    }
    iconv_close(icvt);
    return result;
  }
};

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

StringNormalizer::StringNormalizer(const OpKernelInfo& info) : OpKernel(info),
                                                               is_case_sensitive_(true),
                                                               case_change_action_(NONE),
                                                               compare_caseaction_(NONE) {
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

  compare_caseaction_ = (case_change_action_ == UPPER) ? UPPER : LOWER;
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
    std::copy(X->Data<std::string>(), X->Data<std::string>() + C, output_data);
    return Status::OK();
  }

  // We need to know the result dimension first, and for that we need to filter
  // the words first. If the compare with stop words is case insensitive, we
  // can go ahead and compare the input strings straight away. Otherwise, we need
  // Upper case or lowercase them to compare with wide stop words.

  Locale locale(locale_name_);
  Utf8Converter converter;

  // We need to change case either because of filtering or because of the
  // case change requested.
  // Compute the largest widestring buffer needed.
  size_t max_wide_buffer_len = 0;
  auto input_span = X->DataAsSpan<std::string>();
  for (const auto& s : input_span) {
    size_t wchars;
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
    for (size_t i = 0, lim = narrow<size_t>(C); i < lim; ++i) {
      const std::string& s = input_span[i];
      wchar_buffer.resize(max_wide_buffer_len);
      ORT_RETURN_IF_ERROR(converter.ConvertToWchar(s, wchar_buffer));
      locale.ChangeCase(case_change_action_, wchar_buffer);

      auto& dest = output_data[i];
      size_t utf8_buffer_len = converter.ComputeRequiredSizeToUtf8(wchar_buffer);
      dest.resize(utf8_buffer_len);
      ORT_RETURN_IF_ERROR(converter.ConvertToUtf8(wchar_buffer, dest));
    }
    return Status::OK();
  };

  auto output_filtered = [&](const TensorShape& output_shape, gsl::span<const size_t> filtered_indices) {
    // According to the spec, if all strings are filtered out
    // the output must have a shape of {1} with a single empty string.
    auto output_tensor = ctx->Output(0, output_shape);
    auto output_data = output_tensor->MutableData<std::string>();
    for (size_t i : filtered_indices) {
      const std::string& s = input_span[i];
      if (case_change_action_ != NONE) {
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWchar(s, wchar_buffer));
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
      InlinedVector<size_t> filtered_strings_indecies;
      filtered_strings_indecies.reserve(C);

      for (size_t i = 0, lim = narrow<size_t>(C); i < lim; ++i) {
        const std::string& s = input_span[i];
        if (stopwords_.count(s) == 0) {
          filtered_strings_indecies.push_back(i);
        }
      }

      const int64_t filtered_count = std::max(1LL, narrow<int64_t>(filtered_strings_indecies.size()));
      output_shape.push_back(filtered_count);
      status = output_filtered(output_shape, filtered_strings_indecies);
    }
  } else {
    if (wstopwords_.empty()) {
      assert(case_change_action_ != NONE);
      output_shape.push_back(C);
      status = output_no_filtering(output_shape);
    } else {
      // Case insensitive filtering is performed by converting the input strings
      // to the same case. For that we convert to wchar_t UNICODE.
      // Otherwise, we need to pull ICU library on all platforms
      InlinedVector<size_t> filtered_strings_indecies;
      filtered_strings_indecies.reserve(C);
      for (size_t i = 0, lim = narrow<size_t>(C); i < lim; ++i) {
        const std::string& s = input_span[i];
        wchar_buffer.resize(max_wide_buffer_len);
        ORT_RETURN_IF_ERROR(converter.ConvertToWchar(s, wchar_buffer));
        locale.ChangeCase(compare_caseaction_, wchar_buffer);
        if (wstopwords_.count(wchar_buffer) == 0) {
          filtered_strings_indecies.push_back(i);
        }
      }

      const int64_t filtered_count = std::max(1LL, narrow<int64_t>(filtered_strings_indecies.size()));
      output_shape.push_back(filtered_count);
      status = output_filtered(output_shape, filtered_strings_indecies);
    }
  }

  return status;
}
}  // namespace onnxruntime
