// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_plugin_provider_interfaces.h"

#include "gsl/gsl"
#include "gtest/gtest.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

// Helper class to access public ORT APIs.
struct ApiPtrs {
  ApiPtrs() : ort_api{::OrtGetApiBase()->GetApi(ORT_API_VERSION)},
              ep_api{ort_api->GetEpApi()} {
  }

  const gsl::not_null<const ::OrtApi*> ort_api;
  const gsl::not_null<const ::OrtEpApi*> ep_api;
};

// Normally, a plugin EP would be implemented in a separate library.
// The `test_plugin_ep` namespace contains a local implementation intended for unit testing.
namespace test_plugin_ep {

struct TestOrtEp : ::OrtEp, ApiPtrs {
  TestOrtEp() : ::OrtEp{}, ApiPtrs{} {
    ort_version_supported = ORT_API_VERSION;

    GetName = GetNameImpl;

    // Individual tests should fill out the other function pointers as needed.
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* /*this_ptr*/) {
    constexpr const char* ep_name = "TestOrtEp";
    return ep_name;
  }
};

// This factory doesn't do anything other than implement ReleaseEp().
// It is only used to create the UniqueOrtEp that is required by PluginExecutionProvider.
struct TestOrtEpFactory : ::OrtEpFactory {
  TestOrtEpFactory() : ::OrtEpFactory{} {
    ort_version_supported = ORT_API_VERSION;
    ReleaseEp = ReleaseEpImpl;
  }

  static void ORT_API_CALL ReleaseEpImpl(::OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    delete static_cast<TestOrtEp*>(ep);
  }
};

static TestOrtEpFactory g_test_ort_ep_factory{};

struct MakeTestOrtEpResult {
  std::unique_ptr<IExecutionProvider> ep;  // the IExecutionProvider wrapping the TestOrtEp
  gsl::not_null<TestOrtEp*> ort_ep;        // the wrapped TestOrtEp, owned by `ep`
};

// Creates an IExecutionProvider that wraps a TestOrtEp.
// The TestOrtEp is also exposed so that tests can manipulate its function pointers directly.
MakeTestOrtEpResult MakeTestOrtEp() {
  auto ort_ep_raw = std::make_unique<TestOrtEp>().release();
  auto ort_ep = UniqueOrtEp(ort_ep_raw, OrtEpDeleter{g_test_ort_ep_factory});
  auto ort_session_options = Ort::SessionOptions{};
  auto ep = std::make_unique<PluginExecutionProvider>(std::move(ort_ep),
                                                      *static_cast<const OrtSessionOptions*>(ort_session_options));
  auto result = MakeTestOrtEpResult{std::move(ep), ort_ep_raw};
  return result;
}

}  // namespace test_plugin_ep

TEST(PluginExecutionProviderTest, GetPreferredLayout) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->GetPreferredDataLayout = nullptr;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

  {
    auto prefer_nhwc_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) -> ::OrtStatus* {
      *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = prefer_nhwc_fn;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto invalid_layout_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) -> ::OrtStatus* {
      *preferred_data_layout = static_cast<OrtEpDataLayout>(-1);
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = invalid_layout_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }

  {
    auto failing_fn = [](OrtEp* this_ptr, OrtEpDataLayout* /*preferred_data_layout*/) -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "I can't decide what data layout I prefer.");
    };
    ort_ep->GetPreferredDataLayout = failing_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

TEST(PluginExecutionProviderTest, ShouldConvertNodeLayoutToNhwc) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->ShouldConvertNodeLayoutToNhwc = nullptr;
    ASSERT_EQ(ep->ShouldConvertNodeLayoutToNhwc("", "Conv"), std::nullopt);
  }

  {
    auto custom_nhwc_op_determination = [](OrtEp* /*this_ptr*/,
                                           const char* /*node_domain*/,
                                           const char* node_op_type,
                                           int* should_convert) -> ::OrtStatus* {
      if (node_op_type == std::string_view{"Conv"}) {
        *should_convert = 1;
      } else if (node_op_type == std::string_view{"BatchNormalization"}) {
        *should_convert = 0;
      } else {
        *should_convert = -1;
      }
      return nullptr;
    };
    ort_ep->ShouldConvertNodeLayoutToNhwc = custom_nhwc_op_determination;

    std::optional<bool> should_convert = ep->ShouldConvertNodeLayoutToNhwc("", "Conv");
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, true);

    should_convert = ep->ShouldConvertNodeLayoutToNhwc("", "BatchNormalization");
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, false);

    should_convert = ep->ShouldConvertNodeLayoutToNhwc("", "GridSample");
    ASSERT_EQ(should_convert, std::nullopt);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto failing_fn = [](OrtEp* this_ptr,
                         const char* /*node_domain*/,
                         const char* /*node_op_type*/,
                         int* /*should_convert*/) -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL,
                                                "To convert to NHWC or not to convert to NHWC...");
    };
    ort_ep->ShouldConvertNodeLayoutToNhwc = failing_fn;
    ASSERT_THROW(ep->ShouldConvertNodeLayoutToNhwc("", "Conv"), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

}  // namespace onnxruntime::test
