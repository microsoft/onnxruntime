// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

#include "core/session/onnxruntime_c_api.h"
#include "TestCase.h"

namespace onnxruntime {
namespace test {

std::unique_ptr<std::set<BrokenTest>> GetBrokenTests(const std::string& provider_name);

std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name);

std::unique_ptr<std::unordered_set<std::basic_string<ORTCHAR_T>>> GetAllDisabledTests(const std::basic_string_view<ORTCHAR_T>& provider_name);

using ORT_STRING_VIEW = std::basic_string_view<ORTCHAR_T>;

static ORT_STRING_VIEW opset7 = ORT_TSTR("opset7");
static ORT_STRING_VIEW opset8 = ORT_TSTR("opset8");
static ORT_STRING_VIEW opset9 = ORT_TSTR("opset9");
static ORT_STRING_VIEW opset10 = ORT_TSTR("opset10");
static ORT_STRING_VIEW opset11 = ORT_TSTR("opset11");
static ORT_STRING_VIEW opset12 = ORT_TSTR("opset12");
static ORT_STRING_VIEW opset13 = ORT_TSTR("opset13");
static ORT_STRING_VIEW opset14 = ORT_TSTR("opset14");
static ORT_STRING_VIEW opset15 = ORT_TSTR("opset15");
static ORT_STRING_VIEW opset16 = ORT_TSTR("opset16");
static ORT_STRING_VIEW opset17 = ORT_TSTR("opset17");
static ORT_STRING_VIEW opset18 = ORT_TSTR("opset18");
// TODO: enable opset19 tests
// static ORT_STRING_VIEW opset19 = ORT_TSTR("opset19");

static ORT_STRING_VIEW provider_name_cpu = ORT_TSTR("cpu");
static ORT_STRING_VIEW provider_name_tensorrt = ORT_TSTR("tensorrt");
#ifdef USE_MIGRAPHX
static ORT_STRING_VIEW provider_name_migraphx = ORT_TSTR("migraphx");
#endif
static ORT_STRING_VIEW provider_name_openvino = ORT_TSTR("openvino");
static ORT_STRING_VIEW provider_name_cuda = ORT_TSTR("cuda");
#ifdef USE_ROCM
static ORT_STRING_VIEW provider_name_rocm = ORT_TSTR("rocm");
#endif
static ORT_STRING_VIEW provider_name_dnnl = ORT_TSTR("dnnl");
// For any non-Android system, NNAPI will only be used for ort model converter
#if defined(USE_NNAPI) && defined(__ANDROID__)
static ORT_STRING_VIEW provider_name_nnapi = ORT_TSTR("nnapi");
#endif
#ifdef USE_RKNPU
static ORT_STRING_VIEW provider_name_rknpu = ORT_TSTR("rknpu");
#endif
#ifdef USE_ACL
static ORT_STRING_VIEW provider_name_acl = ORT_TSTR("acl");
#endif
#ifdef USE_ARMNN
static ORT_STRING_VIEW provider_name_armnn = ORT_TSTR("armnn");
#endif

static ORT_STRING_VIEW provider_name_qnn = ORT_TSTR("qnn");
static ORT_STRING_VIEW provider_name_dml = ORT_TSTR("dml");

}  // namespace test
}  // namespace onnxruntime
