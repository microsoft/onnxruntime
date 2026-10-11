// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kv_cache.h"
#include "core/framework/ort_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "asserts.h"
#include "gtest/gtest.h"

#include <cmath>
#include <sstream>

namespace onnxruntime::test {
namespace {
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TypeProto;

InlinedVector<float> ReadKeys(const KVCache& cache) {
  const auto& o = cache.Options();
  const int64_t length = cache.End() - cache.Begin();
  InlinedVector<float> result(static_cast<size_t>(o.batch_size * o.num_kv_heads * length * o.key_head_size));
  cache.Visit([&](const KVCacheBlock& b) {
    for (int64_t h = 0; h < o.batch_size * o.num_kv_heads; ++h) {
      std::copy_n(b.keys + h * b.head_stride * o.key_head_size, b.count * o.key_head_size,
                  result.data() + (h * length + b.position - cache.Begin()) * o.key_head_size);
    }
  });
  return result;
}

TypeProto CacheProto(int64_t window = 0) {
  TypeProto type;
  auto* c = type.mutable_kv_cache_type();
  c->set_elem_type(TensorProto::FLOAT);
  c->mutable_batch_size()->set_dim_value(2);
  c->mutable_num_kv_heads()->set_dim_value(2);
  c->mutable_key_head_size()->set_dim_value(3);
  c->mutable_value_head_size()->set_dim_value(2);
  c->set_sliding_window(window);
  return type;
}

TypeProto TensorProtoType(int dtype, std::initializer_list<int64_t> sizes) {
  TypeProto type;
  auto* t = type.mutable_tensor_type();
  t->set_elem_type(dtype);
  for (auto n : sizes) {
    auto* dim = t->mutable_shape()->add_dim();
    if (n >= 0) dim->set_dim_value(n);
  }
  return type;
}

std::unique_ptr<InferenceSession> Session(int64_t window, bool causal = true,
                                          bool mask = false, bool positions = false,
                                          bool fail_later = false, float softcap = 0,
                                          int mask_dtype = TensorProto::FLOAT, float scale = 0.75f) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  options.inter_op_param.thread_pool_size = 1;
  auto session = std::make_unique<InferenceSession>(options, GetEnvironment());
  Model model("kv_cache_test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 29}}, {}, DefaultLoggingManager().DefaultLogger(), ModelOptions{false, true});
  auto& graph = model.MainGraph();
  auto qt = TensorProtoType(TensorProto::FLOAT, {2, 4, -1, 3});
  auto kt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 3});
  auto vt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 2});
  auto yt = TensorProtoType(TensorProto::FLOAT, {2, 4, -1, 2});
  auto ct = CacheProto(window);
  auto& q = graph.GetOrCreateNodeArg("q", &qt);
  auto& k = graph.GetOrCreateNodeArg("k", &kt);
  auto& v = graph.GetOrCreateNodeArg("v", &vt);
  auto& cache = graph.GetOrCreateNodeArg("cache", &ct);
  auto& y = graph.GetOrCreateNodeArg("y", &yt);
  auto& next = graph.GetOrCreateNodeArg("next", &ct);
  std::vector<NodeArg*> inputs{&q, &k, &v, &cache};
  std::vector<NodeArg*> graph_inputs = inputs;
  if (mask || positions) {
    auto mt = TensorProtoType(mask_dtype, {2, 1, -1, -1});
    auto& m = graph.GetOrCreateNodeArg(mask ? "mask" : "", mask ? &mt : nullptr);
    inputs.push_back(&m);
    if (mask) graph_inputs.push_back(&m);
  }
  if (positions) {
    auto pt = TensorProtoType(TensorProto::INT64, {2, -1});
    auto& p = graph.GetOrCreateNodeArg("positions", &pt);
    inputs.push_back(&p);
    graph_inputs.push_back(&p);
  }
  auto& node = graph.AddNode("attention", "AttentionWithCache", "", inputs, {&y, &next});
  node.AddAttribute("is_causal", static_cast<int64_t>(causal));
  node.AddAttribute("sliding_window", window);
  node.AddAttribute("softcap", softcap);
  node.AddAttribute("scale", scale);
  std::vector<NodeArg*> graph_outputs{&y, &next};
  if (fail_later) {
    // This depends on attention, so its runtime failure is necessarily later.
    auto shape_type = TensorProtoType(TensorProto::INT64, {1});
    auto& shape = graph.GetOrCreateNodeArg("bad_shape", &shape_type);
    graph_inputs.push_back(&shape);
    auto out_type = TensorProtoType(TensorProto::FLOAT, {-1});
    auto& failed = graph.GetOrCreateNodeArg("failed", &out_type);
    graph.AddNode("later_failure", "Reshape", "", {&y, &shape}, {&failed});
    graph_outputs.push_back(&failed);
  }
  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);
  ORT_THROW_IF_ERROR(graph.Resolve());
  std::istringstream stream(model.ToProto().SerializeAsString());
  ORT_THROW_IF_ERROR(session->Load(stream, false));
  ORT_THROW_IF_ERROR(session->Initialize());
  return session;
}

OrtValue Floats(std::initializer_list<int64_t> shape, const std::vector<float>& data) {
  OrtValue value;
  CreateMLValue<float>(std::make_shared<CPUAllocator>(), shape, gsl::make_span(data), &value);
  return value;
}

std::vector<float> Payload(int64_t heads, int64_t tokens, int64_t width, int64_t start, float factor) {
  std::vector<float> result(static_cast<size_t>(2 * heads * tokens * width));
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t d = 0; d < width; ++d) {
          result[static_cast<size_t>(((b * heads + h) * tokens + t) * width + d)] =
              std::sin(static_cast<float>(b * 31 + h * 7 + (start + t) * 3 + d)) * factor;
        }
      }
    }
  }
  return result;
}

std::vector<float> DenseExpected(int64_t start, int64_t tokens, int64_t window, bool causal, float softcap,
                                 const std::vector<float>* mask = nullptr, float scale = 0.75f) {
  const int64_t end = start + tokens;
  auto q = Payload(4, tokens, 3, start, 2);
  auto k = Payload(2, end, 3, 0, 3);
  auto v = Payload(2, end, 2, 0, 1);
  std::vector<float> y(static_cast<size_t>(2 * 4 * tokens * 2));
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t h = 0; h < 4; ++h) {
      for (int64_t t = 0; t < tokens; ++t) {
        std::vector<double> scores(static_cast<size_t>(end), -INFINITY);
        double maximum = -INFINITY;
        for (int64_t p = 0; p < end; ++p) {
          if ((causal && p > start + t) || (window && p < start + t - window + 1)) continue;
          double score = 0;
          for (int64_t d = 0; d < 3; ++d) {
            score += static_cast<double>(q[static_cast<size_t>(((b * 4 + h) * tokens + t) * 3 + d)]) *
                     k[static_cast<size_t>(((b * 2 + h / 2) * end + p) * 3 + d)];
          }
          score *= scale;
          if (softcap) score = softcap * std::tanh(score / softcap);
          if (mask) score += (*mask)[static_cast<size_t>((b * tokens + t) * end + p)];
          scores[static_cast<size_t>(p)] = score;
          maximum = std::max(maximum, score);
        }
        if (maximum == -INFINITY) continue;
        double sum = 0, values[2]{};
        for (int64_t p = 0; p < end; ++p) {
          const double weight = std::exp(scores[static_cast<size_t>(p)] - maximum);
          sum += weight;
          for (int d = 0; d < 2; ++d) {
            values[d] += weight * v[static_cast<size_t>(((b * 2 + h / 2) * end + p) * 2 + d)];
          }
        }
        for (int d = 0; d < 2; ++d) {
          y[static_cast<size_t>(((b * 4 + h) * tokens + t) * 2 + d)] = static_cast<float>(values[d] / sum);
        }
      }
    }
  }
  return y;
}

std::vector<OrtValue> RunCache(InferenceSession& session, const OrtValue& cache, int64_t start, int64_t tokens,
                               const std::vector<float>* mask = nullptr, bool positions = false) {
  NameMLValMap feeds{
      {"q", Floats({2, 4, tokens, 3}, Payload(4, tokens, 3, start, 2))},
      {"k", Floats({2, 2, tokens, 3}, Payload(2, tokens, 3, start, 3))},
      {"v", Floats({2, 2, tokens, 2}, Payload(2, tokens, 2, start, 1))},
      {"cache", cache}};
  if (mask) feeds["mask"] = Floats({2, 1, tokens, start + tokens}, *mask);
  if (positions) {
    std::vector<int64_t> data(static_cast<size_t>(2 * tokens));
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t t = 0; t < tokens; ++t) data[static_cast<size_t>(b * tokens + t)] = start + t;
    }
    CreateMLValue<int64_t>(std::make_shared<CPUAllocator>(), {2, tokens}, gsl::make_span(data), &feeds["positions"]);
  }
  std::vector<OrtValue> fetches;
  ORT_THROW_IF_ERROR(session.Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next"}, &fetches));
  return fetches;
}

void AssertNear(const Tensor& actual, const std::vector<float>& expected) {
  ASSERT_EQ(actual.Shape().Size(), static_cast<int64_t>(expected.size()));
  for (size_t i = 0; i < expected.size(); ++i) ASSERT_NEAR(actual.Data<float>()[i], expected[i], 2e-6) << i;
}

}  // namespace

TEST(KVCacheTest, NativeTypeOwnershipAndStorage) {
  for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged, KVCacheStorage::Ring}) {
    const int64_t window = storage == KVCacheStorage::Ring ? 3 : 0;
    auto empty = CreateKVCache({2, 2, 3, 2, storage, 2, window});
    auto k = Payload(2, 5, 3, 0, 1), v = Payload(2, 5, 2, 0, 1);
    auto staged = empty->Append(k, v, 5);
    EXPECT_EQ(empty->End(), 0);
    EXPECT_EQ(staged->End(), 5);
    EXPECT_EQ(staged->Begin(), 0);  // Do not evict earlier queries in a long chunk.
    auto committed = staged->Seal();
    EXPECT_EQ(committed->Begin(), window ? 2 : 0);
    auto before = ReadKeys(*committed);
    auto next = committed->Append(Payload(2, 4, 3, 5, 1), Payload(2, 4, 2, 5, 1), 4)->Seal();
    EXPECT_EQ(next->End(), 9);
    EXPECT_EQ(ReadKeys(*committed), before);
    EXPECT_EQ(committed->End(), 5);
    EXPECT_THROW(committed->Append({}, {}, 1), OnnxRuntimeException);
  }
  auto proto = CacheProto();
  EXPECT_EQ(DataTypeImpl::TypeFromProto(proto), DataTypeImpl::GetType<KVCacheValue>());
  OrtValue value;
  CreateKVCacheOrtValue({2, 2, 3, 2}, value);
  EXPECT_FALSE(value.IsTensor());
  OrtValue shared = value;
  value = OrtValue{};
  EXPECT_EQ(shared.Get<KVCacheValue>().cache->End(), 0);
  EXPECT_EQ(OrtTypeInfo::FromOrtValue(shared)->type, ONNX_TYPE_KV_CACHE);
  EXPECT_EQ(OrtTypeInfo::FromTypeProto(proto)->Clone()->type, ONNX_TYPE_KV_CACHE);
  ASSERT_STATUS_OK(ValidateKVCacheType(*shared.Get<KVCacheValue>().cache, proto));
  proto.mutable_kv_cache_type()->mutable_batch_size()->set_dim_value(3);
  EXPECT_FALSE(ValidateKVCacheType(*shared.Get<KVCacheValue>().cache, proto).IsOK());
}

TEST(KVCacheTest, PagedVersionsShareUnchangedPages) {
  auto empty = CreateKVCache({2, 2, 3, 2, KVCacheStorage::Paged, 2});
  auto first = empty->Append(Payload(2, 5, 3, 0, 1), Payload(2, 5, 2, 0, 1), 5)->Seal();
  InlinedVector<const float*> pages;
  first->Visit([&](const KVCacheBlock& b) {
    EXPECT_LE(b.count, 2);
    pages.push_back(b.keys);
  });
  auto second = first->Append(Payload(2, 3, 3, 5, 1), Payload(2, 3, 2, 5, 1), 3)->Seal();
  size_t index = 0;
  second->Visit([&](const KVCacheBlock& b) {
    EXPECT_LE(b.count, 2);
    if (index < pages.size()) EXPECT_EQ(b.keys, pages[index]);
    ++index;
  });
  EXPECT_EQ(index, 5u);
}

TEST(KVCacheTest, PrefillChunkDecodeGQAAndRingWrap) {
  for (int64_t window : {0, 1, 4}) {
    for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged, KVCacheStorage::Ring}) {
      if (window == 0 && storage == KVCacheStorage::Ring) continue;
      auto session = Session(window);
      OrtValue cache;
      CreateKVCacheOrtValue({2, 2, 3, 2, storage, 2, window}, cache);
      int64_t start = 0;
      for (int64_t count : {3, 7, 1, 2, 1}) {
        auto old = cache;
        auto old_keys = ReadKeys(*old.Get<KVCacheValue>().cache);
        auto result = RunCache(*session, cache, start, count);
        AssertNear(result[0].Get<Tensor>(), DenseExpected(start, count, window, true, 0));
        EXPECT_EQ(old.Get<KVCacheValue>().cache->End(), start);
        EXPECT_EQ(ReadKeys(*old.Get<KVCacheValue>().cache), old_keys);
        cache = result[1];
        start += count;
        EXPECT_EQ(cache.Get<KVCacheValue>().cache->End(), start);
        EXPECT_EQ(cache.Get<KVCacheValue>().cache->Begin(), window ? std::max<int64_t>(0, start - window) : 0);
      }
    }
  }
}

TEST(KVCacheTest, GemmaWindow512ChunkCrossingPhysicalWrap) {
  auto session = Session(512);
  OrtValue cache;
  CreateKVCacheOrtValue({2, 2, 3, 2, KVCacheStorage::Ring, 16, 512}, cache);
  int64_t start = 0;
  for (int64_t tokens : {500, 24, 3, 520, 1}) {
    auto previous = cache;
    auto result = RunCache(*session, cache, start, tokens);
    AssertNear(result[0].Get<Tensor>(), DenseExpected(start, tokens, 512, true, 0));
    EXPECT_EQ(previous.Get<KVCacheValue>().cache->End(), start);
    start += tokens;
    cache = result[1];
    EXPECT_EQ(cache.Get<KVCacheValue>().cache->Begin(), std::max<int64_t>(0, start - 512));
    cache.Get<KVCacheValue>().cache->Visit([](const KVCacheBlock& block) {
      EXPECT_EQ(block.head_stride, 512);
    });
  }
  EXPECT_GT(start, 1024);
}

TEST(KVCacheTest, LogicalMaskSoftcapFullyMaskedAndPositions) {
  for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged}) {
    auto session = Session(0, false, true, true, false, 1.25f);
    OrtValue cache;
    CreateKVCacheOrtValue({2, 2, 3, 2, storage, 2}, cache);
    std::vector<float> mask(2 * 3 * 3, 0);
    for (size_t i = 0; i < 3; ++i) mask[i] = -INFINITY;
    mask[5] = -2;
    auto result = RunCache(*session, cache, 0, 3, &mask, true);
    AssertNear(result[0].Get<Tensor>(), DenseExpected(0, 3, 0, false, 1.25f, &mask));
    std::vector<float> later_mask(2 * 2 * 5, 0);
    later_mask[0] = -INFINITY;
    auto later = RunCache(*session, result[1], 3, 2, &later_mask, true);
    AssertNear(later[0].Get<Tensor>(), DenseExpected(3, 2, 0, false, 1.25f, &later_mask));
  }
}

TEST(KVCacheTest, WindowMaskUsesAbsolutePositionsAcrossWrap) {
  for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged, KVCacheStorage::Ring}) {
    auto session = Session(4, true, true, true, false, 1.25f);
    OrtValue cache;
    CreateKVCacheOrtValue({2, 2, 3, 2, storage, 2, 4}, cache);
    int64_t start = 0;
    for (int64_t tokens : {3, 7, 2}) {
      const int64_t end = start + tokens;
      std::vector<float> mask(static_cast<size_t>(2 * tokens * end), 0);
      for (int64_t b = 0; b < 2; ++b) {
        for (int64_t t = 0; t < tokens; ++t) {
          for (int64_t p = 0; p < end; ++p) {
            mask[static_cast<size_t>((b * tokens + t) * end + p)] =
                t == 0 || p % 3 == 0 ? -INFINITY : static_cast<float>(p) * -0.25f;
          }
        }
      }
      auto result = RunCache(*session, cache, start, tokens, &mask, true);
      AssertNear(result[0].Get<Tensor>(), DenseExpected(start, tokens, 4, true, 1.25f, &mask));
      cache = result[1];
      start = end;
    }
  }
}

TEST(KVCacheTest, LaterLayerFailurePreservesCallerInputs) {
  for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged, KVCacheStorage::Ring}) {
    auto session = Session(3, true, false, false, true);
    OrtValue cache;
    CreateKVCacheOrtValue({2, 2, 3, 2, storage, 2, 3}, cache);
    cache.GetMutable<KVCacheValue>()->cache =
        cache.Get<KVCacheValue>().cache->Append(Payload(2, 4, 3, 0, 3), Payload(2, 4, 2, 0, 1), 4)->Seal();
    auto before = ReadKeys(*cache.Get<KVCacheValue>().cache);
    NameMLValMap feeds{
        {"q", Floats({2, 4, 2, 3}, Payload(4, 2, 3, 4, 2))},
        {"k", Floats({2, 2, 2, 3}, Payload(2, 2, 3, 4, 3))},
        {"v", Floats({2, 2, 2, 2}, Payload(2, 2, 2, 4, 1))},
        {"cache", cache}};
    CreateMLValue<int64_t>(std::make_shared<CPUAllocator>(), {1}, {123}, &feeds["bad_shape"]);
    std::vector<OrtValue> fetches;
    auto status = session->Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next", "failed"}, &fetches);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(cache.Get<KVCacheValue>().cache->End(), 4);
    EXPECT_EQ(ReadKeys(*cache.Get<KVCacheValue>().cache), before);
    auto good = Session(3);
    auto retry = RunCache(*good, cache, 4, 2);
    AssertNear(retry[0].Get<Tensor>(), DenseExpected(4, 2, 3, true, 0));
  }
}

TEST(KVCacheTest, BooleanMaskAndExplicitZeroScale) {
  auto session = Session(0, true, true, false, false, 0, TensorProto::BOOL, 0);
  OrtValue cache;
  CreateKVCacheOrtValue({2, 2, 3, 2, KVCacheStorage::Paged, 2}, cache);
  NameMLValMap feeds{
      {"q", Floats({2, 4, 3, 3}, Payload(4, 3, 3, 0, 2))},
      {"k", Floats({2, 2, 3, 3}, Payload(2, 3, 3, 0, 3))},
      {"v", Floats({2, 2, 3, 2}, Payload(2, 3, 2, 0, 1))},
      {"cache", cache}};
  std::vector<bool> visible(18, true);
  visible[0] = visible[1] = visible[2] = false;
  visible[4] = false;
  std::vector<float> bias(18, 0);
  for (size_t i = 0; i < visible.size(); ++i)
    if (!visible[i]) bias[i] = -INFINITY;
  CreateMLValue<bool>(std::make_shared<CPUAllocator>(), AsSpan({int64_t{2}, int64_t{1}, int64_t{3}, int64_t{3}}),
                      visible, &feeds["mask"]);
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session->Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next"}, &fetches));
  AssertNear(fetches[0].Get<Tensor>(), DenseExpected(0, 3, 0, true, 0, &bias, 0));
}

TEST(KVCacheTest, RejectInvalidPositionsAndCacheGeometryWithoutMutation) {
  auto session = Session(0, true, false, true);
  OrtValue cache;
  CreateKVCacheOrtValue({2, 2, 3, 2, KVCacheStorage::Paged, 2}, cache);
  NameMLValMap feeds{
      {"q", Floats({2, 4, 1, 3}, Payload(4, 1, 3, 0, 2))},
      {"k", Floats({2, 2, 1, 3}, Payload(2, 1, 3, 0, 3))},
      {"v", Floats({2, 2, 1, 2}, Payload(2, 1, 2, 0, 1))},
      {"cache", cache}};
  CreateMLValue<int64_t>(std::make_shared<CPUAllocator>(), {2, 1}, {int64_t{1}, int64_t{0}}, &feeds["positions"]);
  std::vector<OrtValue> fetches;
  EXPECT_FALSE(session->Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next"}, &fetches).IsOK());
  EXPECT_EQ(cache.Get<KVCacheValue>().cache->End(), 0);
  OrtValue wrong;
  CreateKVCacheOrtValue({1, 2, 3, 2}, wrong);
  feeds["cache"] = wrong;
  fetches.clear();
  EXPECT_FALSE(session->Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next"}, &fetches).IsOK());
  EXPECT_EQ(wrong.Get<KVCacheValue>().cache->End(), 0);
}

TEST(KVCacheTest, CreationInspectionAndExportKernels) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.kv_cache.backend", "paged"));
  InferenceSession session(options, GetEnvironment());
  Model model("kv_helpers", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 29}}, {}, DefaultLoggingManager().DefaultLogger(), ModelOptions{false, true});
  auto& graph = model.MainGraph();
  auto ct = CacheProto();
  auto& cache = graph.GetOrCreateNodeArg("cache", &ct);
  auto& create = graph.AddNode("create", "KVCacheCreate", "", {}, {&cache});
  create.AddAttribute("elem_type", int64_t{TensorProto::FLOAT});
  create.AddAttribute("batch_size", int64_t{2});
  create.AddAttribute("num_kv_heads", int64_t{2});
  create.AddAttribute("key_head_size", int64_t{3});
  create.AddAttribute("value_head_size", int64_t{2});
  auto lt = TensorProtoType(TensorProto::INT64, {2});
  auto rt = TensorProtoType(TensorProto::INT64, {2, 2});
  auto& lengths = graph.GetOrCreateNodeArg("lengths", &lt);
  auto& ranges = graph.GetOrCreateNodeArg("ranges", &rt);
  graph.AddNode("describe", "KVCacheDescribe", "", {&cache}, {&lengths, &ranges});
  auto kt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 3});
  auto vt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 2});
  auto pt = TensorProtoType(TensorProto::INT64, {2, -1});
  auto& k = graph.GetOrCreateNodeArg("k", &kt);
  auto& v = graph.GetOrCreateNodeArg("v", &vt);
  auto& positions = graph.GetOrCreateNodeArg("positions", &pt);
  graph.AddNode("export", "KVCacheExport", "", {&cache}, {&k, &v, &positions});
  graph.SetInputs({});
  graph.SetOutputs({&cache, &lengths, &ranges, &k, &v, &positions});
  ASSERT_STATUS_OK(graph.Resolve());
  std::istringstream stream(model.ToProto().SerializeAsString());
  ASSERT_STATUS_OK(session.Load(stream, false));
  ASSERT_STATUS_OK(session.Initialize());
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(RunOptions{}, NameMLValMap{}, std::vector<std::string>{"cache", "lengths", "ranges", "k", "v", "positions"},
                               &fetches));
  EXPECT_EQ(fetches[0].Get<KVCacheValue>().cache->Options().storage, KVCacheStorage::Paged);
  for (auto i : {1u, 2u}) {
    for (auto value : fetches[i].Get<Tensor>().DataAsSpan<int64_t>()) EXPECT_EQ(value, 0);
  }
  EXPECT_EQ(fetches[3].Get<Tensor>().Shape(), TensorShape({2, 2, 0, 3}));
  EXPECT_EQ(fetches[4].Get<Tensor>().Shape(), TensorShape({2, 2, 0, 2}));
  EXPECT_EQ(fetches[5].Get<Tensor>().Shape(), TensorShape({2, 0}));
}

TEST(KVCacheTest, RejectCallerCacheAsPreallocatedFetch) {
  auto session = Session(0);
  OrtValue cache;
  CreateKVCacheOrtValue({2, 2, 3, 2}, cache);
  NameMLValMap feeds{
      {"q", Floats({2, 4, 1, 3}, Payload(4, 1, 3, 0, 2))},
      {"k", Floats({2, 2, 1, 3}, Payload(2, 1, 3, 0, 3))},
      {"v", Floats({2, 2, 1, 2}, Payload(2, 1, 2, 0, 1))},
      {"cache", cache}};
  std::vector<OrtValue> fetches{OrtValue{}, cache};
  EXPECT_FALSE(session->Run(RunOptions{}, feeds, std::vector<std::string>{"y", "next"}, &fetches).IsOK());
  EXPECT_EQ(cache.Get<KVCacheValue>().cache->End(), 0);
}

TEST(KVCacheTest, ExportRetainedRingPositionsWithoutRelabeling) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  InferenceSession session(options, GetEnvironment());
  Model model("export_ring", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 29}}, {}, DefaultLoggingManager().DefaultLogger(), ModelOptions{false, true});
  auto& graph = model.MainGraph();
  auto ct = CacheProto(3);
  auto& cache_arg = graph.GetOrCreateNodeArg("cache", &ct);
  auto kt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 3});
  auto vt = TensorProtoType(TensorProto::FLOAT, {2, 2, -1, 2});
  auto pt = TensorProtoType(TensorProto::INT64, {2, -1});
  auto& k = graph.GetOrCreateNodeArg("k", &kt);
  auto& v = graph.GetOrCreateNodeArg("v", &vt);
  auto& positions = graph.GetOrCreateNodeArg("positions", &pt);
  graph.AddNode("export", "KVCacheExport", "", {&cache_arg}, {&k, &v, &positions});
  graph.SetInputs({&cache_arg});
  graph.SetOutputs({&k, &v, &positions});
  ASSERT_STATUS_OK(graph.Resolve());
  std::istringstream stream(model.ToProto().SerializeAsString());
  ASSERT_STATUS_OK(session.Load(stream, false));
  ASSERT_STATUS_OK(session.Initialize());
  for (auto storage : {KVCacheStorage::Contiguous, KVCacheStorage::Paged, KVCacheStorage::Ring}) {
    OrtValue cache;
    CreateKVCacheOrtValue({2, 2, 3, 2, storage, 2, 3}, cache);
    cache.GetMutable<KVCacheValue>()->cache =
        cache.Get<KVCacheValue>().cache->Append(Payload(2, 5, 3, 0, 3), Payload(2, 5, 2, 0, 1), 5)->Seal();
    cache.GetMutable<KVCacheValue>()->cache =
        cache.Get<KVCacheValue>().cache->Append(Payload(2, 3, 3, 5, 3), Payload(2, 3, 2, 5, 1), 3)->Seal();
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, NameMLValMap{{"cache", cache}}, std::vector<std::string>{"k", "v", "positions"}, &fetches));
    AssertNear(fetches[0].Get<Tensor>(), Payload(2, 3, 3, 5, 3));
    AssertNear(fetches[1].Get<Tensor>(), Payload(2, 3, 2, 5, 1));
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t p = 0; p < 3; ++p) EXPECT_EQ(fetches[2].Get<Tensor>().Data<int64_t>()[b * 3 + p], 5 + p);
    }
    EXPECT_EQ(cache.Get<KVCacheValue>().cache->End(), 8);
  }
}

}  // namespace onnxruntime::test
