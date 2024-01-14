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

using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;

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

using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;

#else

// All others (not Windows, Apple, or Android)
class Utf8Converter {
 public:
  Utf8Converter(const std::string&, const std::wstring&) {}

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

template <class ForwardIter>
Status CopyCaseAction(ForwardIter first, ForwardIter end, OpKernelContext* ctx,
                      const Locale& loc,
                      Utf8Converter& converter,
                      size_t N, size_t C,
                      StringNormalizer::CaseAction caseaction) {
  std::vector<int64_t> output_dims;
  if (N == 1) {
    output_dims.push_back(1);
  }

  // Empty output case
  if (C == 0) {
    output_dims.push_back(1);
    TensorShape output_shape(output_dims);
    // This will create one empty string
    ctx->Output(0, output_shape);
    return Status::OK();
  }

  output_dims.push_back(C);

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->MutableData<std::string>();

  size_t output_idx = 0;
  while (first != end) {
    auto& s = *first;
    if (caseaction == StringNormalizer::LOWER || caseaction == StringNormalizer::UPPER) {
      std::wstring wstr = converter.from_bytes(s);
      if (wstr == wconv_error) {
        // Please do not include the input text in the error message as it could
        // be deemed as a compliance violation by teams using this operator
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Input contains invalid utf8 chars");
      }
      // In place transform
      loc.ChangeCase(caseaction, wstr);
      *(output_data + output_idx) = converter.to_bytes(wstr);
    } else {
      assert(caseaction == StringNormalizer::NONE);
      // Simple copy or move if the iterator points to a non-const string
      *(output_data + output_idx) = std::move(s);
    }
    ++output_idx;
    ++first;
  }
  return Status::OK();
}
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

  if (!is_case_sensitive_) {
    // Convert stop words to a case which can help us preserve the case of filtered strings
    compare_caseaction_ = (case_change_action_ == UPPER) ? UPPER : LOWER;
  }

  locale_name_ = info.GetAttrOrDefault("locale", default_locale);
  Locale locale(locale_name_);
  Utf8Converter converter(conv_error, wconv_error);

  std::vector<std::string> swords = info.GetAttrsOrDefault<std::string>("stopwords");
  for (auto& sw : swords) {
    ORT_ENFORCE(!sw.empty(), "Empty stopwords not allowed");
    if (is_case_sensitive_) {
      auto p = stopwords_.insert(std::move(sw));
      ORT_ENFORCE(p.second, "Duplicate stopwords not allowed");
    } else {
      std::wstring wstr = converter.from_bytes(sw);
      ORT_ENFORCE(wstr != wconv_error, "Stopword contains invalid utf8 chars");
      locale.ChangeCase(compare_caseaction_, wstr);
      auto p = wstopwords_.insert(std::move(wstr));
      ORT_ENFORCE(p.second, "Duplicate stopwords not allowed");
    }
  }
}

Status StringNormalizer::Compute(OpKernelContext* ctx) const {
  using namespace string_normalizer;

  auto X = ctx->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  auto input_dims = X->Shape().GetDims();

  size_t N = 0;
  size_t C = 0;
  if (input_dims.size() == 1) {
    if (input_dims[0] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Single dimension value must be greater than 0");
    }
    C = onnxruntime::narrow<size_t>(input_dims[0]);
  } else if (input_dims.size() == 2) {
    if (input_dims[0] != 1 || input_dims[1] < 1) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Input dimensions are either[C > 0] or [1][C > 0] allowed");
    }
    N = 1;
    C = onnxruntime::narrow<size_t>(input_dims[1]);
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either[C > 0] or [1][C > 0] allowed");
  }

  Status status;
  Locale locale(locale_name_);
  Utf8Converter converter(conv_error, wconv_error);
  auto* const input_data = X->Data<std::string>();
  using StrRef = std::reference_wrapper<const std::string>;
  if (is_case_sensitive_) {
    if (!stopwords_.empty()) {
      InlinedVector<StrRef> filtered_strings;
      filtered_strings.reserve(C);
      auto first = input_data;
      auto const last = input_data + C;
      while (first != last) {
        const std::string& s = *first;
        if (0 == stopwords_.count(s)) {
          filtered_strings.push_back(std::cref(s));
        }
        ++first;
      }
      status = CopyCaseAction(filtered_strings.cbegin(), filtered_strings.cend(), ctx, locale, converter,
                              N, filtered_strings.size(), case_change_action_);
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, locale, converter, N, C, case_change_action_);
    }
  } else {
    if (!wstopwords_.empty()) {
      // Filter input. When no case action is required
      // we simply store original string references.
      // Otherwise, we store converted strings.
      InlinedVector<StrRef> filtered_orignal_strings;
      InlinedVector<std::string> filtered_cased_strings;
      filtered_orignal_strings.reserve(C);
      filtered_cased_strings.reserve(C);
      auto first = input_data;
      auto const last = input_data + C;
      while (first != last) {
        const std::string& s = *first;
        std::wstring wstr = converter.from_bytes(s);
        if (wstr == wconv_error) {
          // Please do not include the input text in the error message as it could
          // be deemed as a compliance violation by teams using this operator
          return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                        "Input contains invalid utf8 chars");
        }
        locale.ChangeCase(compare_caseaction_, wstr);
        if (0 == wstopwords_.count(wstr)) {
          if (case_change_action_ == NONE) {
            filtered_orignal_strings.push_back(std::cref(s));
          } else {
            filtered_cased_strings.push_back(converter.to_bytes(wstr));
          }
        }
        ++first;
      }
      if (case_change_action_ == NONE) {
        status = CopyCaseAction(filtered_orignal_strings.cbegin(), filtered_orignal_strings.cend(), ctx, locale, converter,
                                N, filtered_orignal_strings.size(), NONE);
      } else {
        status = CopyCaseAction(filtered_cased_strings.begin(), filtered_cased_strings.end(), ctx, locale, converter,
                                N, filtered_cased_strings.size(), NONE);
      }
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, locale, converter, N, C, case_change_action_);
    }
  }
  return status;
}
}  // namespace onnxruntime
