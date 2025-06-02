// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#pragma once

// Some tests fail when DNNL is used. Skip them for now.
#if defined(USE_DNNL)
#define DNNL_GTEST_SKIP() GTEST_SKIP() << "Skipping test when DNNL is used."
#else
#define DNNL_GTEST_SKIP()
#endif

namespace onnxruntime {
namespace test {
bool DnnlHasBF16Support();
}  // namespace test
}  // namespace onnxruntime
