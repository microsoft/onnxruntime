// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

namespace {

struct RouterGateUpCase {
  int64_t rows;
  int64_t k;
  int64_t router;
  int64_t inter;
  float limit = 0.0f;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool weight_is_initializer = true;
};

template <typename T>
float RoundToT(float v) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(v).ToFloat();
  } else {
    return MLFloat16(v).ToFloat();
  }
}

template <typename T>
std::vector<T> ToT(const std::vector<float>& v) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(v);
  } else {
    return ToFloat16(v);
  }
}

float SwiGluRef(float g, float l, float limit, float alpha, float beta) {
  if (limit > 0.0f) {
    g = std::min(g, limit);
    l = std::min(std::max(l, -limit), limit);
  }
  return g / (1.0f + std::exp(-alpha * g)) * (l + beta);
}

// Inputs rounded to T and the outputs expected from MatMul + Split + SwiGLU, computed in double.
struct RouterGateUpData {
  std::vector<float> x;
  std::vector<float> w;
  std::vector<float> router;
  std::vector<float> activated;
};

template <typename T>
RouterGateUpData MakeRouterGateUpData(const RouterGateUpCase& c) {
  const int64_t n = c.router + 2 * c.inter;
  std::mt19937 generator(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  RouterGateUpData data;
  data.x.resize(static_cast<size_t>(c.rows * c.k));
  data.w.resize(static_cast<size_t>(c.k * n));
  for (float& v : data.x) v = RoundToT<T>(dist(generator));
  for (float& v : data.w) v = RoundToT<T>(dist(generator) * 0.1f);

  data.router.resize(static_cast<size_t>(c.rows * c.router));
  data.activated.resize(static_cast<size_t>(c.rows * c.inter));
  for (int64_t m = 0; m < c.rows; ++m) {
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int64_t kk = 0; kk < c.k; ++kk) {
      const double xv = data.x[static_cast<size_t>(m * c.k + kk)];
      for (int64_t j = 0; j < n; ++j) y[static_cast<size_t>(j)] += xv * data.w[static_cast<size_t>(kk * n + j)];
    }
    for (int64_t j = 0; j < c.router; ++j) {
      data.router[static_cast<size_t>(m * c.router + j)] = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(j)]));
    }
    for (int64_t j = 0; j < c.inter; ++j) {
      const float gate = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(c.router + j)]));
      const float up = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(c.router + c.inter + j)]));
      data.activated[static_cast<size_t>(m * c.inter + j)] =
          RoundToT<T>(SwiGluRef(gate, up, c.limit, c.alpha, c.beta));
    }
  }
  return data;
}

template <typename T>
void RunRouterGateUpProjTest(const RouterGateUpCase& c, const char* activation = nullptr) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA is not available";
  }

  const int64_t n = c.router + 2 * c.inter;
  const RouterGateUpData data = MakeRouterGateUpData<T>(c);

  OpTester tester("RouterGateUpProj", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("router_size", c.router);
  if (activation != nullptr) tester.AddAttribute<std::string>("activation", activation);
  tester.AddAttribute<float>("limit", c.limit);
  tester.AddAttribute<float>("activation_alpha", c.alpha);
  tester.AddAttribute<float>("activation_beta", c.beta);
  tester.AddInput<T>("X", {c.rows, c.k}, ToT<T>(data.x));
  tester.AddInput<T>("W", {c.k, n}, ToT<T>(data.w), c.weight_is_initializer);
  // 16-bit outputs can differ by one ulp when the fp32 accumulation order differs from the reference.
  tester.AddOutput<T>("router", {c.rows, c.router}, ToT<T>(data.router), false, 0.02f, 0.02f);
  tester.AddOutput<T>("activated", {c.rows, c.inter}, ToT<T>(data.activated), false, 0.02f, 0.02f);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

// Runs the operator in an IOBinding session where X and W are views that start `x_offset` / `w_offset` elements (2
// bytes each) into larger device buffers, and the outputs are prefilled with a value no result can equal. A view at
// an odd offset is a valid tensor that vector loads in the fused kernel cannot read directly, and a launch that
// writes nothing leaves the prefilled value behind.
template <typename T>
void RunRouterGateUpProjViewTest(const RouterGateUpCase& c, size_t x_offset, size_t w_offset) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA is not available";
  }

  constexpr float kPrefill = -77.0f;
  constexpr int kElemType = std::is_same_v<T, BFloat16> ? ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
                                                        : ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  const int64_t n = c.router + 2 * c.inter;
  const RouterGateUpData data = MakeRouterGateUpData<T>(c);

  const std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMSDomain, 1}};
  Model model("router_gate_up_proj_view", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  auto make_arg = [&](const char* name, std::initializer_list<int64_t> dims) -> NodeArg& {
    ONNX_NAMESPACE::TypeProto type;
    type.mutable_tensor_type()->set_elem_type(kElemType);
    for (const int64_t dim : dims) type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    return graph.GetOrCreateNodeArg(name, &type);
  };
  NodeArg& x_arg = make_arg("X", {c.rows, c.k});
  NodeArg& w_arg = make_arg("W", {c.k, n});
  NodeArg& router_arg = make_arg("router", {c.rows, c.router});
  NodeArg& activated_arg = make_arg("activated", {c.rows, c.inter});
  NodeAttributes attributes = {
      {"router_size", utils::MakeAttribute("router_size", c.router)},
      {"limit", utils::MakeAttribute("limit", c.limit)},
      {"activation_alpha", utils::MakeAttribute("activation_alpha", c.alpha)},
      {"activation_beta", utils::MakeAttribute("activation_beta", c.beta)},
  };
  Node& node = graph.AddNode("router_gate_up_proj", "RouterGateUpProj", "view test", {&x_arg, &w_arg},
                             {&router_arg, &activated_arg}, &attributes, onnxruntime::kMSDomain);
  node.SetExecutionProviderType(kCudaExecutionProvider);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_bytes;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));
  std::stringstream model_stream(model_bytes);
  SessionOptions session_options;
  session_options.session_logid = "RouterGateUpProjViewTest";
  InferenceSession session(session_options, GetEnvironment());
  IExecutionProvider* ep = cuda_ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  AllocatorPtr device_alloc;
  for (const auto& allocator : ep->CreatePreferredAllocators()) {
    const OrtMemoryInfo& info = allocator->Info();
    if (info.device.Type() == OrtDevice::GPU && info.mem_type == OrtMemTypeDefault) {
      device_alloc = session.GetAllocator(info);
    }
  }
  ASSERT_NE(device_alloc, nullptr);
  const AllocatorPtr cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto copy_to_device = [&](const std::vector<T>& host, const TensorShape& shape, OrtValue& device_value) {
    Tensor host_tensor(DataTypeImpl::GetType<T>(), shape, const_cast<T*>(host.data()), cpu_alloc->Info());
    ASSERT_STATUS_OK(ep->GetDataTransfer()->CopyTensor(host_tensor, *device_value.GetMutable<Tensor>()));
  };
  // `backing` owns the device buffer; `view` is the tensor that starts `offset` elements into it.
  auto make_view = [&](const std::vector<float>& values, const TensorShape& shape, size_t offset, OrtValue& backing,
                       OrtValue& view) {
    const int64_t count = static_cast<int64_t>(offset) + shape.Size() + 8;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape({count}), device_alloc, backing);
    auto* base = static_cast<char*>(backing.GetMutable<Tensor>()->MutableDataRaw());
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, base + offset * sizeof(T), device_alloc->Info(), view);
    copy_to_device(ToT<T>(values), shape, view);
  };
  auto make_prefilled = [&](const TensorShape& shape, OrtValue& value) {
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, device_alloc, value);
    copy_to_device(ToT<T>(std::vector<float>(static_cast<size_t>(shape.Size()), kPrefill)), shape, value);
  };

  OrtValue x_backing, x_view, w_backing, w_view, router_value, activated_value;
  make_view(data.x, TensorShape({c.rows, c.k}), x_offset, x_backing, x_view);
  make_view(data.w, TensorShape({c.k, n}), w_offset, w_backing, w_view);
  make_prefilled(TensorShape({c.rows, c.router}), router_value);
  make_prefilled(TensorShape({c.rows, c.inter}), activated_value);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("X", x_view));
  ASSERT_STATUS_OK(binding->BindInput("W", w_view));
  ASSERT_STATUS_OK(binding->BindOutput("router", router_value));
  ASSERT_STATUS_OK(binding->BindOutput("activated", activated_value));
  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options, *binding));

  auto check = [&](const char* name, const OrtValue& device_value, const std::vector<float>& expected) {
    const Tensor& device_tensor = device_value.Get<Tensor>();
    Tensor host_tensor(DataTypeImpl::GetType<T>(), device_tensor.Shape(), cpu_alloc);
    ASSERT_STATUS_OK(ep->GetDataTransfer()->CopyTensor(device_tensor, host_tensor));
    const auto actual = host_tensor.DataAsSpan<T>();
    ASSERT_EQ(actual.size(), expected.size());
    int mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      const float value = actual[i].ToFloat();
      // Negated comparison so that NaN fails too.
      if (!(std::abs(value - expected[i]) <= 0.02f + 0.02f * std::abs(expected[i]))) {
        ADD_FAILURE() << name << "[" << i << "] = " << value << ", expected " << expected[i];
        if (++mismatches == 5) return;
      }
    }
  };
  check("router", router_value, data.router);
  check("activated", activated_value, data.activated);
}

}  // namespace

TEST(RouterGateUpProjTest, FusedSingleRow) { RunRouterGateUpProjTest<BFloat16>({1, 512, 128, 64}); }
TEST(RouterGateUpProjTest, FusedOddRows) { RunRouterGateUpProjTest<BFloat16>({3, 1024, 256, 128}); }
TEST(RouterGateUpProjTest, FusedEightRows) { RunRouterGateUpProjTest<BFloat16>({8, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, FusedClampAndBeta) {
  RunRouterGateUpProjTest<BFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f});
}
TEST(RouterGateUpProjTest, FusedDynamicWeight) {
  RunRouterGateUpProjTest<BFloat16>({2, 512, 128, 64, 0.0f, 1.0f, 0.0f, false});
}
TEST(RouterGateUpProjTest, FusedExplicitActivation) { RunRouterGateUpProjTest<BFloat16>({2, 512, 128, 64}, "swiglu"); }
TEST(RouterGateUpProjTest, GenericManyRows) { RunRouterGateUpProjTest<BFloat16>({40, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, GenericUnalignedK) { RunRouterGateUpProjTest<BFloat16>({4, 200, 48, 24}); }

TEST(RouterGateUpProjTest, Float16FusedSingleRow) { RunRouterGateUpProjTest<MLFloat16>({1, 512, 128, 64}); }
TEST(RouterGateUpProjTest, Float16FusedEightRows) { RunRouterGateUpProjTest<MLFloat16>({8, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, Float16FusedClampAndBeta) {
  RunRouterGateUpProjTest<MLFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f});
}
TEST(RouterGateUpProjTest, Float16GenericManyRows) { RunRouterGateUpProjTest<MLFloat16>({40, 1024, 128, 64}); }

// An empty reduction projects to zeros, as MatMul does; the schema accepts K = 0.
TEST(RouterGateUpProjTest, ZeroReduction) { RunRouterGateUpProjTest<BFloat16>({1, 0, 128, 64}); }
TEST(RouterGateUpProjTest, ZeroReductionDynamicWeightWithBeta) {
  RunRouterGateUpProjTest<BFloat16>({3, 0, 256, 128, 0.5f, 1.702f, 1.0f, false});
}
TEST(RouterGateUpProjTest, Float16ZeroReduction) { RunRouterGateUpProjTest<MLFloat16>({2, 0, 128, 64}); }

// X, W and the outputs are views into larger device buffers (IOBinding). The fused kernel reads W as 16-byte and X as
// 4-byte vectors, so a view that is aligned only to 2, 4 or 8 bytes must take the generic path and still be correct.
TEST(RouterGateUpProjTest, ViewAlignedAtOffset) {
  RunRouterGateUpProjViewTest<BFloat16>({3, 1024, 256, 128}, /*x_offset=*/2, /*w_offset=*/8);
}
TEST(RouterGateUpProjTest, ViewPrefilledOutputs) {
  RunRouterGateUpProjViewTest<BFloat16>({8, 2560, 512, 640}, /*x_offset=*/0, /*w_offset=*/0);
}
TEST(RouterGateUpProjTest, ViewMisalignedX) {
  RunRouterGateUpProjViewTest<BFloat16>({3, 1024, 256, 128}, /*x_offset=*/1, /*w_offset=*/0);
}
TEST(RouterGateUpProjTest, ViewMisalignedW) {
  RunRouterGateUpProjViewTest<BFloat16>({3, 1024, 256, 128}, /*x_offset=*/0, /*w_offset=*/1);
}
TEST(RouterGateUpProjTest, ViewWAlignedTo8BytesOnly) {
  RunRouterGateUpProjViewTest<BFloat16>({3, 1024, 256, 128}, /*x_offset=*/0, /*w_offset=*/4);
}
TEST(RouterGateUpProjTest, Float16ViewMisalignedX) {
  RunRouterGateUpProjViewTest<MLFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f}, /*x_offset=*/1, /*w_offset=*/0);
}
TEST(RouterGateUpProjTest, Float16ViewMisalignedW) {
  RunRouterGateUpProjViewTest<MLFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f}, /*x_offset=*/0, /*w_offset=*/1);
}

}  // namespace test
}  // namespace onnxruntime
