// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#if !defined(ORT_USE_EP_API_ADAPTERS) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#endif
#endif

namespace onnxruntime::test {
namespace {

using Shapes = std::array<std::vector<int64_t>, 4>;

std::unique_ptr<Model> MakeReplayModel(const Shapes& shapes) {
  auto model = std::make_unique<Model>(
      "gated_delta_net_state_replay", false, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), std::unordered_map<std::string, int>{{kMSDomain, 1}},
      std::vector<ONNX_NAMESPACE::FunctionProto>{}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();
  constexpr std::array<const char*, 4> names{
      "source_backing", "capsule_backing", "destination_backing", "metadata"};
  std::vector<NodeArg*> inputs;
  for (size_t index = 0; index < names.size(); ++index) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(index == 3 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                          : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (const auto dim : shapes[index]) {
      auto* dimension = tensor_type->mutable_shape()->add_dim();
      if (dim >= 0) {
        dimension->set_dim_value(dim);
      } else {
        dimension->set_dim_param("backing_extent");
      }
    }
    inputs.push_back(&graph.GetOrCreateNodeArg(names[index], &type));
  }
  auto& output = graph.GetOrCreateNodeArg("destination_out", nullptr);
  auto& node = graph.AddNode("replay", "GatedDeltaNetStateReplay", "", inputs, {&output}, nullptr, kMSDomain);
  node.SetExecutionProviderType(kWebGpuExecutionProvider);
  graph.SetOutputs({&output});
  return model;
}

TEST(GatedDeltaNetStateReplayShapeInferenceTest, DestinationShapeAndType) {
  auto model = MakeReplayModel(Shapes{{{20}, {30}, {40}, {11}}});
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
  const auto* output = model->MainGraph().GetOutputs()[0];
  ASSERT_EQ(output->TypeAsProto()->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ASSERT_EQ(output->Shape()->dim_size(), 1);
  EXPECT_EQ(output->Shape()->dim(0).dim_value(), 40);

  model = MakeReplayModel(Shapes{{{-1}, {-1}, {-1}, {11}}});
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
  EXPECT_EQ(model->MainGraph().GetOutputs()[0]->Shape()->dim(0).dim_param(), "backing_extent");
}

TEST(GatedDeltaNetStateReplayShapeInferenceTest, RejectsRanksAndMetadataLength) {
  for (size_t index = 0; index < 4; ++index) {
    auto shapes = Shapes{{{20}, {30}, {40}, {11}}};
    shapes[index] = {1, shapes[index][0]};
    auto model = MakeReplayModel(shapes);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(model->MainGraph().Resolve(), "inputs must have rank 1");
  }
  for (const int64_t size : {0, 10, 12}) {
    auto model = MakeReplayModel(Shapes{{{20}, {30}, {40}, {size}}});
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(model->MainGraph().Resolve(), "metadata must have shape [11]");
  }
}

#ifdef USE_WEBGPU

struct Geometry {
  int64_t hv, dv, dk, hk, capacity;
};

class ReplayHarness {
 public:
  Status Initialize(const Geometry& geometry,
                    std::unique_ptr<IExecutionProvider> provider = DefaultWebGpuExecutionProvider(),
                    bool shared_source_capsule = false) {
    ORT_RETURN_IF_NOT(provider, "WebGPU EP is unavailable.");
    provider_ = provider.get();
    const auto allocators = provider_->CreatePreferredAllocators();
    ORT_RETURN_IF_NOT(!allocators.empty(), "WebGPU EP has no preferred allocator.");
    const OrtMemoryInfo memory_info = allocators[0]->Info();

    const auto [hv, dv, dk, hk, capacity] = geometry;
    state_elements_ = static_cast<size_t>(hv * dv * dk);
    const int64_t row_elements = capacity * (hv + hk * dk + hv * dv);
    desc_ = {static_cast<int64_t>(state_elements_) + 3, static_cast<int64_t>(state_elements_) + 5,
             row_elements, row_elements + capacity * hv, row_elements + capacity * (hv + hk * dk),
             hv, dv, dk, hk, capacity, 1};
    source_.resize(2 * state_elements_ + 11);
    capsule_.resize(static_cast<size_t>(2 * row_elements + 7));
    shared_source_capsule_ = shared_source_capsule;
    if (shared_source_capsule_) {
      const auto source_extent = source_.size();
      const auto shared_extent = source_extent + capsule_.size();
      source_.resize(shared_extent);
      capsule_.resize(shared_extent);
      for (size_t field = 2; field < 5; ++field) {
        desc_[field] += static_cast<int64_t>(source_extent);
      }
    }
    destination_.assign(3 * state_elements_ + 17, -931.25f);
    for (size_t index = 0; index < source_.size(); ++index) {
      source_[index] = static_cast<float>(static_cast<int>(index % 113) - 56) / 97.0f;
    }

    auto model = MakeReplayModel(Shapes{{{static_cast<int64_t>(source_.size())},
                                         {static_cast<int64_t>(capsule_.size())},
                                         {static_cast<int64_t>(destination_.size())},
                                         {11}}});
    ORT_RETURN_IF_ERROR(model->MainGraph().Resolve());
    std::stringstream stream;
    model->ToProto().SerializeToOstream(&stream);
    SessionOptions session_options;
    session_options.graph_optimization_level = TransformerLevel::Default;
    session_ = std::make_unique<InferenceSession>(session_options, GetEnvironment());
    ORT_RETURN_IF_ERROR(session_->RegisterExecutionProvider(std::move(provider)));
    ORT_RETURN_IF_ERROR(session_->Load(stream));
    ORT_RETURN_IF_ERROR(session_->Initialize());
    allocator_ = session_->GetAllocator(memory_info);
    ORT_RETURN_IF_NOT(allocator_, "WebGPU allocator is unavailable.");
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(source_.size())},
                         allocator_, source_value_);
    if (shared_source_capsule_) {
      capsule_value_ = source_value_;
    } else {
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(capsule_.size())},
                           allocator_, capsule_value_);
    }
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(destination_.size())},
                         allocator_, destination_value_);
    Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape{11}, desc_.data(),
                         cpu_allocator_.Info(), metadata_value_);
    return Status::OK();
  }

  Status Prepare(int64_t kept_count) {
    desc_[10] = kept_count;
    std::fill(capsule_.begin(), capsule_.end(), std::numeric_limits<float>::quiet_NaN());
    const auto hv = desc_[5], dv = desc_[6], dk = desc_[7], hk = desc_[8];
    for (int64_t t = 0; t < kept_count; ++t) {
      for (int64_t h = 0; h < hv; ++h) {
        capsule_[static_cast<size_t>(desc_[2] + t * hv + h)] = 0.71f + 0.013f * static_cast<float>((t + h) % 12);
        for (int64_t v = 0; v < dv; ++v) {
          capsule_[static_cast<size_t>(desc_[4] + (t * hv + h) * dv + v)] =
              static_cast<float>((t * 11 + h * 7 + v) % 23 - 11) / 31.0f;
        }
      }
      for (int64_t h = 0; h < hk; ++h) {
        for (int64_t i = 0; i < dk; ++i) {
          capsule_[static_cast<size_t>(desc_[3] + (t * hk + h) * dk + i)] =
              static_cast<float>((t * 13 + h * 5 + i) % 29 - 14) / 37.0f;
        }
      }
    }
    if (shared_source_capsule_) {
      std::copy_n(source_.begin() + desc_[0], state_elements_, capsule_.begin() + desc_[0]);
      source_ = capsule_;
    }
    ORT_RETURN_IF_ERROR(Upload(source_, source_value_));
    ORT_RETURN_IF_ERROR(Upload(capsule_, capsule_value_));
    return Upload(destination_, destination_value_);
  }

  Status Run(bool alias_output = true, const OrtValue* source_override = nullptr,
             const OrtValue* capsule_override = nullptr) {
    std::unique_ptr<IOBinding> binding;
    ORT_RETURN_IF_ERROR(session_->NewIOBinding(&binding));
    ORT_RETURN_IF_ERROR(binding->BindInput("source_backing", source_override ? *source_override : source_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("capsule_backing", capsule_override ? *capsule_override : capsule_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("destination_backing", destination_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("metadata", metadata_value_));
    OrtValue separate_output;
    if (!alias_output) {
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), destination_value_.Get<Tensor>().Shape(),
                           allocator_, separate_output);
    }
    ORT_RETURN_IF_ERROR(binding->BindOutput("destination_out", alias_output ? destination_value_ : separate_output));
    return session_->Run(RunOptions{}, *binding);
  }

  std::vector<float> Reference() const {
    const auto hv = desc_[5], dv = desc_[6], dk = desc_[7], hk = desc_[8];
    std::vector<float> state(source_.begin() + desc_[0], source_.begin() + desc_[0] + state_elements_);
    // Token-major CPU oracle, independent of the shader's flattened per-element loop.
    for (int64_t t = 0; t < desc_[10]; ++t) {
      for (int64_t h = 0; h < hv; ++h) {
        const float decay = capsule_[static_cast<size_t>(desc_[2] + t * hv + h)];
        const int64_t key_head = h * hk / hv;
        for (int64_t v = 0; v < dv; ++v) {
          const float delta = capsule_[static_cast<size_t>(desc_[4] + (t * hv + h) * dv + v)];
          for (int64_t i = 0; i < dk; ++i) {
            auto& value = state[static_cast<size_t>((h * dv + v) * dk + i)];
            value *= decay;
            value += capsule_[static_cast<size_t>(desc_[3] + (t * hk + key_head) * dk + i)] * delta;
          }
        }
      }
    }
    return state;
  }

  void Verify() {
    const auto expected = Reference();
    std::vector<float> actual(destination_.size());
    ASSERT_STATUS_OK(Download(destination_value_, actual));
    const auto offset = static_cast<size_t>(desc_[1]);
    for (size_t index = 0; index < expected.size(); ++index) {
      ASSERT_TRUE(std::isfinite(actual[offset + index])) << index;
      constexpr float kAbsoluteTolerance = 1e-5f, kRelativeTolerance = 2e-5f;
      ASSERT_LE(std::abs(actual[offset + index] - expected[index]),
                kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[index]))
          << "state element " << index;
    }
    EXPECT_EQ(std::memcmp(actual.data(), destination_.data(), offset * sizeof(float)), 0);
    const auto tail = offset + expected.size();
    EXPECT_EQ(std::memcmp(actual.data() + tail, destination_.data() + tail,
                          (actual.size() - tail) * sizeof(float)),
              0);
    std::vector<float> source_copy(source_.size()), capsule_copy(capsule_.size());
    ASSERT_STATUS_OK(Download(source_value_, source_copy));
    ASSERT_STATUS_OK(Download(capsule_value_, capsule_copy));
    EXPECT_EQ(std::memcmp(source_copy.data(), source_.data(), source_.size() * sizeof(float)), 0);
    // Byte comparison is intentional for preservation checks, including poisoned NaN tails.
    EXPECT_EQ(std::memcmp(capsule_copy.data(), capsule_.data(), capsule_.size() * sizeof(float)), 0);
  }

  void VerifyDestinationUnchanged() {
    std::vector<float> actual(destination_.size());
    ASSERT_STATUS_OK(Download(destination_value_, actual));
    EXPECT_EQ(std::memcmp(actual.data(), destination_.data(), destination_.size() * sizeof(float)), 0);
  }

  std::array<int64_t, 11>& Descriptor() { return desc_; }

  Status RunWithAliasedDestination(size_t input_index) {
    OrtValue aliased_input;
    const auto& input = (input_index == 0 ? source_value_ : capsule_value_).Get<Tensor>();
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), input.Shape(),
                         destination_value_.GetMutable<Tensor>()->MutableDataRaw(),
                         input.Location(), aliased_input);
    return Run(true, input_index == 0 ? &aliased_input : nullptr,
               input_index == 1 ? &aliased_input : nullptr);
  }

  Status RunWithSeparateCapsuleWrapper() {
    OrtValue capsule_wrapper;
    const auto& backing = capsule_value_.Get<Tensor>();
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), backing.Shape(),
                         capsule_value_.GetMutable<Tensor>()->MutableDataRaw(),
                         backing.Location(), capsule_wrapper);
    return Run(true, nullptr, &capsule_wrapper);
  }

 private:
  Status Upload(std::vector<float>& data, OrtValue& gpu_value) {
    Tensor cpu_tensor(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(data.size())},
                      data.data(), cpu_allocator_.Info());
    return provider_->GetDataTransfer()->CopyTensor(cpu_tensor, *gpu_value.GetMutable<Tensor>());
  }

  Status Download(const OrtValue& gpu_value, std::vector<float>& data) {
    Tensor cpu_tensor(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(data.size())},
                      data.data(), cpu_allocator_.Info());
    return provider_->GetDataTransfer()->CopyTensor(gpu_value.Get<Tensor>(), cpu_tensor);
  }

  std::unique_ptr<InferenceSession> session_;
  IExecutionProvider* provider_ = nullptr;
  AllocatorPtr allocator_;
  CPUAllocator cpu_allocator_;
  std::vector<float> source_, capsule_, destination_;
  size_t state_elements_ = 0;
  bool shared_source_capsule_ = false;
  std::array<int64_t, 11> desc_{};
  OrtValue source_value_, capsule_value_, destination_value_, metadata_value_;
};

class GatedDeltaNetStateReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!DefaultWebGpuExecutionProvider()) {
      GTEST_SKIP() << "WebGPU EP is unavailable.";
    }
  }
};

TEST_F(GatedDeltaNetStateReplayTest, EveryPrefixWithNonDivisibleHeadsAndPoisonedTails) {
  for (const int64_t capacity : {1, 4, 8}) {
    SCOPED_TRACE(capacity);
    ReplayHarness harness;
    ASSERT_STATUS_OK(harness.Initialize({5, 7, 11, 3, capacity}));
    for (int64_t kept = 1; kept <= capacity; ++kept) {
      SCOPED_TRACE(kept);
      ASSERT_STATUS_OK(harness.Prepare(kept));
      ASSERT_STATUS_OK(harness.Run());
      ASSERT_NO_FATAL_FAILURE(harness.Verify());
    }
  }
}

TEST_F(GatedDeltaNetStateReplayTest, QwenGeometryAndUnalignedRealCapsuleRow) {
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({48, 128, 128, 16, 7}));
  EXPECT_EQ(harness.Descriptor()[2], 57680);
  EXPECT_EQ(harness.Descriptor()[2] * static_cast<int64_t>(sizeof(float)) % 256, 64);
  for (int64_t kept = 1; kept <= 7; ++kept) {
    SCOPED_TRACE(kept);
    ASSERT_STATUS_OK(harness.Prepare(kept));
    ASSERT_STATUS_OK(harness.Run());
    ASSERT_NO_FATAL_FAILURE(harness.Verify());
  }
}

#if !defined(ORT_USE_EP_API_ADAPTERS) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
TEST_F(GatedDeltaNetStateReplayTest, RunCompletesBeforeReadback) {
  auto provider = DefaultWebGpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  auto& ep = static_cast<WebGpuExecutionProvider&>(*provider);
  auto& context = webgpu::WebGpuContextFactory::GetContext(ep.GetDeviceId());
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({48, 128, 128, 16, 7}, std::move(provider)));
  for (const int64_t kept : {1, 4, 7}) {
    SCOPED_TRACE(kept);
    ASSERT_STATUS_OK(harness.Prepare(kept));
    ASSERT_STATUS_OK(harness.Run());

    // No readback, Flush or blocking Wait may mask the operator's completion boundary.
    const auto& recording = ep.Recording();
    ASSERT_TRUE(recording.deferred_dispatches.empty());
    ASSERT_EQ(recording.command_encoder, nullptr);
    ASSERT_FALSE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
    auto result = std::make_shared<wgpu::QueueWorkDoneStatus>(wgpu::QueueWorkDoneStatus::CallbackCancelled);
    const auto future = context.Device().GetQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [result](wgpu::QueueWorkDoneStatus status, wgpu::StringView /*message*/) noexcept {
          *result = status;
        });
    ASSERT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
    ASSERT_EQ(*result, wgpu::QueueWorkDoneStatus::Success);
    ASSERT_NO_FATAL_FAILURE(harness.Verify());
  }
}
#endif

TEST_F(GatedDeltaNetStateReplayTest, SegmentedBackingViews) {
  auto provider = WebGpuExecutionProviderWithTestStorageBufferBindingSize(65536);
  if (!provider) {
    GTEST_SKIP() << "Artificially small storage binding limits require the native test factory.";
  }
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({3, 64, 64, 16, 8}, std::move(provider)));
  // Source/capsule each have two segments; destination has three. Logical views cross segment boundaries.
  ASSERT_STATUS_OK(harness.Prepare(8));
  ASSERT_STATUS_OK(harness.Run());
  harness.Verify();
}

TEST_F(GatedDeltaNetStateReplayTest, SharedSourceCapsuleBindingOwner) {
  constexpr uint64_t kSegmentBytes = 65536;
  ConfigOptions options;
  ASSERT_STATUS_OK(options.AddConfigEntry(webgpu::options::kMaxStorageBuffersPerShaderStage, "8"));
  auto provider = WebGpuExecutionProviderWithTestStorageBufferBindingSize(options, kSegmentBytes);
  if (!provider) {
    GTEST_SKIP() << "Artificially small storage binding limits require the native test factory.";
  }
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({3, 64, 64, 16, 8}, std::move(provider), true));
  // Shared read-only backing: 44,050 floats (3 segments); destination: 36,881 floats (3 segments).
  // Distinct Tensor wrappers do not share Program binding ownership, even with the same raw buffer.
  ASSERT_STATUS_OK(harness.Prepare(3));
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.RunWithSeparateCapsuleWrapper(), "exceed WebGPU binding limits");
  harness.VerifyDestinationUnchanged();
  for (int64_t kept = 1; kept <= 8; ++kept) {
    SCOPED_TRACE(kept);
    ASSERT_STATUS_OK(harness.Prepare(kept));
    ASSERT_STATUS_OK(harness.Run());
    ASSERT_NO_FATAL_FAILURE(harness.Verify());
  }
}

TEST_F(GatedDeltaNetStateReplayTest, RejectsInvalidMetadataBeforeWriting) {
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({5, 7, 11, 3, 4}));
  ASSERT_STATUS_OK(harness.Prepare(2));
  const auto valid = harness.Descriptor();
  for (size_t field = 0; field < valid.size(); ++field) {
    SCOPED_TRACE(field);
    for (const int64_t invalid :
         std::array<int64_t, 2>{-1, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1}) {
      harness.Descriptor() = valid;
      harness.Descriptor()[field] = invalid;
      ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "outside the uint32 indexing range");
    }
  }
  for (size_t field = 5; field < 10; ++field) {
    harness.Descriptor() = valid;
    harness.Descriptor()[field] = 0;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "dimensions and capacity must be positive");
  }
  for (const int64_t kept : {0, 5}) {
    harness.Descriptor() = valid;
    harness.Descriptor()[10] = kept;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "kept_count must be in [1, capacity]");
  }
  for (size_t field = 0; field < 5; ++field) {
    harness.Descriptor() = valid;
    harness.Descriptor()[field] = std::numeric_limits<uint32_t>::max();
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "view exceeds its backing tensor");
  }
  for (size_t field = 5; field < 10; ++field) {
    harness.Descriptor() = valid;
    harness.Descriptor()[field] = std::numeric_limits<uint32_t>::max();
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "geometry exceeds the uint32 indexing range");
  }
  harness.VerifyDestinationUnchanged();
  harness.Descriptor() = valid;
  ASSERT_STATUS_OK(harness.Run());
  harness.Verify();
}

TEST_F(GatedDeltaNetStateReplayTest, RequiresPreboundDestinationAlias) {
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({5, 7, 11, 3, 4}));
  ASSERT_STATUS_OK(harness.Prepare(2));
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(false), "output must be prebound to destination_backing");
  harness.VerifyDestinationUnchanged();
  ASSERT_STATUS_OK(harness.Run());
  harness.Verify();
}

TEST_F(GatedDeltaNetStateReplayTest, RejectsWritableInputAliasing) {
  ReplayHarness harness;
  ASSERT_STATUS_OK(harness.Initialize({5, 7, 11, 3, 4}));
  ASSERT_STATUS_OK(harness.Prepare(2));
  for (size_t input_index = 0; input_index < 2; ++input_index) {
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.RunWithAliasedDestination(input_index),
                                        "destination backing must be distinct");
  }
  harness.VerifyDestinationUnchanged();
}

#endif  // USE_WEBGPU

}  // namespace
}  // namespace onnxruntime::test
