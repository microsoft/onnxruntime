// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalizer.h"
#include "onnx/defs/schema.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

#include <codecvt>
#include <locale>
#include <functional>
#include <unordered_set>

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    StringNormalizer,
    1,
    string,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    contrib::StringNormalizer);

namespace string_normalizer {
const std::string conv_error("Conversion Error");
const std::wstring wconv_error(L"Conversion Error");
// performs tolower/toupper in-place
inline void ChangeCase(const std::locale& loc, StringNormalizer::CaseAction caseaction,
                       std::wstring& wstr) {
  assert(caseaction != StringNormalizer::NONE);
  if (caseaction == StringNormalizer::LOWER) {
    std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                   [&loc](wchar_t ch) { return std::tolower(ch, loc); });
  } else {
    std::transform(wstr.begin(), wstr.end(), wstr.begin(),
                   [&loc](wchar_t ch) { return std::toupper(ch, loc); });
  }
}

template <class ForwardIter>
Status CopyCaseAction(ForwardIter first, ForwardIter end, OpKernelContext* ctx,
                      const std::locale& loc,
                      std::wstring_convert<std::codecvt_utf8<wchar_t>>& converter,
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
    auto output_ten = ctx->Output(0, output_shape);
    auto output_default = output_ten->template MutableData<std::string>();
    new (output_default) std::string();
    return Status::OK();
  }

  output_dims.push_back(C);

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  auto const output_data = output_tensor->template MutableData<std::string>();

  size_t output_idx = 0;
  while (first != end) {
    auto& s = *first;
    if (caseaction == StringNormalizer::LOWER || caseaction == StringNormalizer::UPPER) {
      std::wstring wstr = converter.from_bytes(s);
      if (wstr == wconv_error) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                      "Input contains invalid utf8 chars at: " + static_cast<const std::string&>(s));
      }
      // In place transform
      ChangeCase(loc, caseaction, wstr);
      new (output_data + output_idx) std::string(converter.to_bytes(wstr));
    } else {
      assert(caseaction == StringNormalizer::NONE);
      // Simple copy or move if the iterator points to a non-const string
      new (output_data + output_idx) std::string(std::move(s));
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
                                                               casechangeaction_(NONE),
                                                               compare_caseaction_(NONE) {
  int64_t iscasesensitive = 0;
  Status status = info.GetAttr("is_case_sensitive", &iscasesensitive);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute is_case_sensitive is not set");
  is_case_sensitive_ = iscasesensitive != 0;

  std::string casechangeaction;
  status = info.GetAttr("casechangeaction", &casechangeaction);
  ONNXRUNTIME_ENFORCE(status.IsOK(), "attribute caseaction is not set");
  if (casechangeaction == "LOWER") {
    casechangeaction_ = LOWER;
  } else if (casechangeaction == "UPPER") {
    casechangeaction_ = UPPER;
  } else if (casechangeaction == "NONE") {
    casechangeaction_ = NONE;
  } else {
    ONNXRUNTIME_ENFORCE(false, "attribute casechangeaction has invalid value");
  }

  if (!is_case_sensitive_) {
    // Convert stop words to a case which can help us preserve the case of filtered strings
    compare_caseaction_ = (casechangeaction_ == UPPER) ? UPPER : LOWER;
  }

  locale_ = info.GetAttrOrDefault("locale", std::string("en_US"));
  std::locale locale(locale_);
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter(conv_error, wconv_error);

  std::vector<std::string> swords = info.GetAttrsOrDefault<std::string>("stopwords");
  for (const auto& sw : swords) {
    ONNXRUNTIME_ENFORCE(!sw.empty(), "Empty stopwords not allowed");
    if (is_case_sensitive_) {
      auto p = stopwords_.insert(sw);
      ONNXRUNTIME_ENFORCE(p.second, "Duplicate stopwords not allowed");
    } else {
      std::wstring wstr = converter.from_bytes(sw);
      ONNXRUNTIME_ENFORCE(wstr != wconv_error, "Stopword contains invalid utf8 chars");
      ChangeCase(locale, compare_caseaction_, wstr);
      auto p = wstopwords_.insert(wstr);
      ONNXRUNTIME_ENFORCE(p.second, "Duplicate stopwords not allowed");
    }
  }
}

Status StringNormalizer::Compute(OpKernelContext* ctx) const {
  using namespace string_normalizer;

  auto X = ctx->Input<Tensor>(0);
  auto& input_dims = X->Shape().GetDims();

  size_t N = 0;
  size_t C = 0;
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
    N = 1;
    C = input_dims[1];
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input dimensions are either[C > 0] or [1][C > 0] allowed");
  }

  Status status;
  std::locale locale(locale_);
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter(conv_error, wconv_error);
  auto const input_data = X->template Data<std::string>();
  using StrRef = std::reference_wrapper<const std::string>;
  if (is_case_sensitive_) {
    if (!stopwords_.empty()) {
      std::vector<StrRef> filtered_strings;
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
                              N, filtered_strings.size(), casechangeaction_);
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, locale, converter, N, C, casechangeaction_);
    }
  } else {
    if (!wstopwords_.empty()) {
      // Filter input. When no case action is required
      // we simply store original string references.
      // Otherwise, we store converted strings.
      std::vector<StrRef> filtered_orignal_strings;
      std::vector<std::string> filtered_cased_strings;
      filtered_orignal_strings.reserve(C);
      filtered_cased_strings.reserve(C);
      auto first = input_data;
      auto const last = input_data + C;
      while (first != last) {
        const std::string& s = *first;
        std::wstring wstr = converter.from_bytes(s);
        if (wstr == wconv_error) {
          return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                        "Input contains invalid utf8 chars at: " + s);
        }
        ChangeCase(locale, compare_caseaction_, wstr);
        if (0 == wstopwords_.count(wstr)) {
          if (casechangeaction_ == NONE) {
            filtered_orignal_strings.push_back(std::cref(s));
          } else {
            filtered_cased_strings.push_back(converter.to_bytes(wstr));
          }
        }
        ++first;
      }
      if (casechangeaction_ == NONE) {
        status = CopyCaseAction(filtered_orignal_strings.cbegin(), filtered_orignal_strings.cend(), ctx, locale, converter,
                                N, filtered_orignal_strings.size(), NONE);
      } else {
        status = CopyCaseAction(filtered_cased_strings.begin(), filtered_cased_strings.end(), ctx, locale, converter,
                                N, filtered_cased_strings.size(), NONE);
      }
    } else {
      // Nothing to filter. Copy input to output and change case if needed
      status = CopyCaseAction(input_data, input_data + C, ctx, locale, converter, N, C, casechangeaction_);
    }
  }
  return status;
}
}  // namespace contrib
}  // namespace onnxruntime
