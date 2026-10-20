// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

#ifdef USE_WEBGPU
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#endif

namespace onnxruntime::test {
namespace {

using Shapes = std::array<std::vector<int64_t>, 4>;

std::unique_ptr<Model> MakeConvReplayModel(const Shapes& shapes, bool half) {
  auto model = std::make_unique<Model>(
      "conv_state_replay", false, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), std::unordered_map<std::string, int>{{kMSDomain, 1}},
      std::vector<ONNX_NAMESPACE::FunctionProto>{}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();
  constexpr std::array<const char*, 4> names{
      "source_backing", "capture_backing", "destination_backing", "metadata"};
  std::vector<NodeArg*> inputs;
  for (size_t index = 0; index < names.size(); ++index) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(index == 3 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                          : half     ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                     : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (const auto dim : shapes[index]) {
      auto* dimension = tensor->mutable_shape()->add_dim();
      if (dim >= 0) {
        dimension->set_dim_value(dim);
      } else {
        dimension->set_dim_param("backing_extent");
      }
    }
    inputs.push_back(&graph.GetOrCreateNodeArg(names[index], &type));
  }
  auto& output = graph.GetOrCreateNodeArg("destination_out", nullptr);
  auto& node = graph.AddNode("replay", "ConvStateReplay", "", inputs, {&output}, nullptr, kMSDomain);
  node.SetExecutionProviderType(kWebGpuExecutionProvider);
  graph.SetOutputs({&output});
  return model;
}

TEST(ConvStateReplayShapeInferenceTest, DestinationShapeAndType) {
  for (const bool half : {false, true}) {
    auto model = MakeConvReplayModel(Shapes{{{20}, {30}, {40}, {8}}}, half);
    ASSERT_STATUS_OK(model->MainGraph().Resolve());
    const auto* output = model->MainGraph().GetOutputs()[0];
    EXPECT_EQ(output->TypeAsProto()->tensor_type().elem_type(),
              half ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    ASSERT_EQ(output->Shape()->dim_size(), 1);
    EXPECT_EQ(output->Shape()->dim(0).dim_value(), 40);
    model = MakeConvReplayModel(Shapes{{{-1}, {-1}, {-1}, {8}}}, half);
    ASSERT_STATUS_OK(model->MainGraph().Resolve());
    EXPECT_EQ(model->MainGraph().GetOutputs()[0]->Shape()->dim(0).dim_param(), "backing_extent");
  }
}

TEST(ConvStateReplayShapeInferenceTest, RejectsRanksAndMetadataLength) {
  for (size_t index = 0; index < 4; ++index) {
    auto shapes = Shapes{{{20}, {30}, {40}, {8}}};
    shapes[index] = {1, shapes[index][0]};
    auto model = MakeConvReplayModel(shapes, false);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(model->MainGraph().Resolve(), "inputs must have rank 1");
  }
  for (const int64_t size : {0, 7, 9}) {
    auto model = MakeConvReplayModel(Shapes{{{20}, {30}, {40}, {size}}}, true);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(model->MainGraph().Resolve(), "metadata must have shape [8]");
  }
}

#ifdef USE_WEBGPU

struct Geometry {
  int64_t batch, channels, width, capacity;
};

// Store the CPU oracle and transfers as integers so neither can canonicalize NaNs.
template <typename Bits>
class ConvReplayHarness {
 public:
  Status Initialize(const Geometry& geometry, const std::array<int64_t, 3>& offsets = {5, 7, 9},
                    const std::array<int64_t, 3>& tails = {13, 11, 15},
                    std::unique_ptr<IExecutionProvider> provider = DefaultWebGpuExecutionProvider()) {
    ORT_RETURN_IF_NOT(provider, "WebGPU EP is unavailable.");
    provider_ = provider.get();
    const auto allocators = provider_->CreatePreferredAllocators();
    ORT_RETURN_IF_NOT(!allocators.empty(), "WebGPU EP has no preferred allocator.");
    const OrtMemoryInfo memory_info = allocators[0]->Info();
    const auto [batch, channels, width, capacity] = geometry;
    desc_ = {offsets[0] * static_cast<int64_t>(sizeof(Bits)), offsets[1] * static_cast<int64_t>(sizeof(Bits)),
             offsets[2] * static_cast<int64_t>(sizeof(Bits)),
             batch, channels, width, capacity, 0};
    const int64_t state_elements = batch * channels * width;
    source_.resize(static_cast<size_t>(offsets[0] + state_elements + tails[0]));
    destination_.resize(static_cast<size_t>(offsets[1] + state_elements + tails[1]));
    capture_.resize(static_cast<size_t>(offsets[2] + batch * channels * capacity + tails[2]));
    Fill(source_, 17);
    Fill(capture_, 113);
    Fill(destination_, 971);
    auto model = MakeConvReplayModel(Shapes{{{static_cast<int64_t>(source_.size())},
                                             {static_cast<int64_t>(capture_.size())},
                                             {static_cast<int64_t>(destination_.size())},
                                             {8}}},
                                     sizeof(Bits) == 2);
    ORT_RETURN_IF_ERROR(model->MainGraph().Resolve());
    std::stringstream stream;
    model->ToProto().SerializeToOstream(&stream);
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    ORT_RETURN_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    session_ = std::make_unique<InferenceSession>(options, GetEnvironment());
    ORT_RETURN_IF_ERROR(session_->RegisterExecutionProvider(std::move(provider)));
    ORT_RETURN_IF_ERROR(session_->Load(stream));
    ORT_RETURN_IF_ERROR(session_->Initialize());
    allocator_ = session_->GetAllocator(memory_info);
    ORT_RETURN_IF_NOT(allocator_, "WebGPU allocator is unavailable.");
    const std::array backing_values{std::make_pair(&source_, &source_value_), std::make_pair(&capture_, &capture_value_),
                                   std::make_pair(&destination_, &destination_value_)};
    for (const auto& item : backing_values) {
      Tensor::InitOrtValue(ElementType(), TensorShape{static_cast<int64_t>(item.first->size())},
                           allocator_, *item.second);
    }
    Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape{8}, desc_.data(),
                         cpu_allocator_.Info(), metadata_value_);
    return Status::OK();
  }

  Status Prepare(int64_t kept) {
    desc_[7] = kept;
    ORT_RETURN_IF_ERROR(Transfer(source_, source_value_, true));
    ORT_RETURN_IF_ERROR(Transfer(capture_, capture_value_, true));
    return Transfer(destination_, destination_value_, true);
  }

  Status Run(bool alias_output = true, int aliased_input = -1) {
    std::unique_ptr<IOBinding> binding;
    ORT_RETURN_IF_ERROR(session_->NewIOBinding(&binding));
    OrtValue input_alias;
    if (aliased_input >= 0) {
      const auto& input = (aliased_input == 0 ? source_value_ : capture_value_).template Get<Tensor>();
      Tensor::InitOrtValue(ElementType(), input.Shape(), destination_value_.GetMutable<Tensor>()->MutableDataRaw(),
                           input.Location(), input_alias);
    }
    ORT_RETURN_IF_ERROR(binding->BindInput("source_backing", aliased_input == 0 ? input_alias : source_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("capture_backing", aliased_input == 1 ? input_alias : capture_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("destination_backing", destination_value_));
    ORT_RETURN_IF_ERROR(binding->BindInput("metadata", metadata_value_));
    OrtValue separate_output;
    if (!alias_output) {
      Tensor::InitOrtValue(ElementType(), destination_value_.Get<Tensor>().Shape(), allocator_, separate_output);
    }
    ORT_RETURN_IF_ERROR(binding->BindOutput("destination_out", alias_output ? destination_value_ : separate_output));
    return session_->Run(RunOptions{}, *binding);
  }

  void Verify() {
    auto expected = destination_;
    const auto batch = desc_[3], channels = desc_[4], width = desc_[5], capacity = desc_[6];
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t c = 0; c < channels; ++c) {
        const auto row = static_cast<size_t>((b * channels + c) * width);
        const auto source_start = source_.begin() + desc_[0] / sizeof(Bits) + row;
        std::deque<Bits> history(source_start, source_start + width);
        // Independent token-by-token rolling queue, not the shader's direct selection formula.
        for (int64_t t = 0; t < desc_[7] && width > 0; ++t) {
          history.pop_front();
          history.push_back(capture_[static_cast<size_t>(desc_[2] / sizeof(Bits) + (b * capacity + t) * channels + c)]);
        }
        std::copy(history.begin(), history.end(), expected.begin() + desc_[1] / sizeof(Bits) + row);
      }
    }
    VerifyBacking(expected, destination_value_);
    VerifyBacking(source_, source_value_);
    VerifyBacking(capture_, capture_value_);
  }

  void VerifyDestinationUnchanged() { VerifyBacking(destination_, destination_value_); }
  std::array<int64_t, 8>& Descriptor() { return desc_; }
  size_t SourceBytes() const { return source_.size() * sizeof(Bits); }
  size_t CaptureBytes() const { return capture_.size() * sizeof(Bits); }
  size_t DestinationBytes() const { return destination_.size() * sizeof(Bits); }

 private:
  static MLDataType ElementType() {
    return sizeof(Bits) == 2 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  }

  static void Fill(std::vector<Bits>& values, uint32_t seed) {
    constexpr std::array<uint32_t, 16> special{
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234, 0xffc54321, 0xff800001,
        0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x3f800000, 0xbf800000, 0x7fffffff, 0xffffffff};
    for (size_t index = 0; index < values.size(); ++index) {
      if constexpr (sizeof(Bits) == 2) {
        // Odd multiplier visits all 65,536 half payloads in a sufficiently large backing.
        values[index] = static_cast<Bits>(index * 40503 + seed);
      } else {
        values[index] = static_cast<Bits>(index % 2 == 0 ? special[(index / 2 + seed) % special.size()]
                                                         : static_cast<uint32_t>(index * 2654435761u + seed));
      }
    }
  }

  Status Transfer(std::vector<Bits>& data, OrtValue& value, bool upload) {
    if (data.empty()) {
      return Status::OK();
    }
    Tensor cpu_tensor(ElementType(), TensorShape{static_cast<int64_t>(data.size())},
                      data.data(), cpu_allocator_.Info());
    return upload ? provider_->GetDataTransfer()->CopyTensor(cpu_tensor, *value.GetMutable<Tensor>())
                  : provider_->GetDataTransfer()->CopyTensor(value.Get<Tensor>(), cpu_tensor);
  }

  void VerifyBacking(const std::vector<Bits>& expected, OrtValue& value) {
    std::vector<Bits> actual(expected.size());
    ASSERT_STATUS_OK(Transfer(actual, value, false));
    ASSERT_EQ(actual, expected);
  }

  std::unique_ptr<InferenceSession> session_;
  IExecutionProvider* provider_ = nullptr;
  AllocatorPtr allocator_;
  CPUAllocator cpu_allocator_;
  std::vector<Bits> source_, capture_, destination_;
  std::array<int64_t, 8> desc_{};
  OrtValue source_value_, capture_value_, destination_value_, metadata_value_;
};

template <typename Bits>
class ConvStateReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!DefaultWebGpuExecutionProvider()) {
      GTEST_SKIP() << "WebGPU EP is unavailable.";
    }
  }
};

using ReplayPayloadTypes = ::testing::Types<uint16_t, uint32_t>;
TYPED_TEST_SUITE(ConvStateReplayTest, ReplayPayloadTypes);

TYPED_TEST(ConvStateReplayTest, EveryPrefixBatchChannelsAndHistoryWidths) {
  for (const int64_t capacity : {1, 4, 8}) {
    for (const int64_t width : {1, 3, 11}) {
      SCOPED_TRACE(::testing::Message() << "capacity " << capacity << " width " << width);
      ConvReplayHarness<TypeParam> harness;
      ASSERT_STATUS_OK(harness.Initialize({2, 5, width, capacity}));
      for (int64_t kept = 0; kept <= capacity; ++kept) {
        SCOPED_TRACE(kept);
        ASSERT_STATUS_OK(harness.Prepare(kept));
        ASSERT_STATUS_OK(harness.Run());
        ASSERT_NO_FATAL_FAILURE(harness.Verify());
      }
    }
  }
}

TYPED_TEST(ConvStateReplayTest, ArbitraryOffsetsPreserveAdjacentElements) {
  for (const auto offsets : {std::array<int64_t, 3>{0, 0, 0}, {1, 1, 1}, {2, 3, 4}, {31, 128, 257}}) {
    SCOPED_TRACE(::testing::Message() << offsets[0] << "," << offsets[1] << "," << offsets[2]);
    ConvReplayHarness<TypeParam> harness;
    ASSERT_STATUS_OK(harness.Initialize({3, 3, 5, 7}, offsets));
    for (int64_t kept = 0; kept <= 7; ++kept) {
      ASSERT_STATUS_OK(harness.Prepare(kept));
      ASSERT_STATUS_OK(harness.Run());
      ASSERT_NO_FATAL_FAILURE(harness.Verify());
    }
  }
}

TYPED_TEST(ConvStateReplayTest, ExactEndAndSingleElementBoundaries) {
  for (const int64_t width : {1, 2, 5}) {
    for (const int64_t offset : {0, 1}) {
      ConvReplayHarness<TypeParam> harness;
      ASSERT_STATUS_OK(harness.Initialize({1, 1, width, 7}, {offset, offset, offset}, {0, 0, 0}));
      for (int64_t kept = 0; kept <= 7; ++kept) {
        ASSERT_STATUS_OK(harness.Prepare(kept));
        ASSERT_STATUS_OK(harness.Run());
        ASSERT_NO_FATAL_FAILURE(harness.Verify());
      }
    }
  }
}

TYPED_TEST(ConvStateReplayTest, ZeroCapacityHistoryAndBatch) {
  for (const auto geometry : {Geometry{2, 3, 5, 0}, {2, 3, 0, 4}, {0, 3, 5, 4}, {0, 3, 0, 0}}) {
    for (const int64_t padding : {0, 1}) {
      ConvReplayHarness<TypeParam> harness;
      ASSERT_STATUS_OK(harness.Initialize(geometry, {padding, padding, padding}, {0, 0, 0}));
      for (int64_t kept = 0; kept <= geometry.capacity; ++kept) {
        ASSERT_STATUS_OK(harness.Prepare(kept));
        ASSERT_STATUS_OK(harness.Run());
        ASSERT_NO_FATAL_FAILURE(harness.Verify());
      }
    }
  }
}

TYPED_TEST(ConvStateReplayTest, ExactSpecialPayloadBits) {
  ConvReplayHarness<TypeParam> harness;
  ASSERT_STATUS_OK(harness.Initialize({1, 1, 65536, 65537}, {1, 3, 5}));
  // Source-only and capture-only selections cover every FP16 bit pattern, including
  // signaling NaNs, both zero signs, infinities and subnormals, at odd half offsets.
  for (const int64_t kept : {0, 1, 65536, 65537}) {
    ASSERT_STATUS_OK(harness.Prepare(kept));
    ASSERT_STATUS_OK(harness.Run());
    ASSERT_NO_FATAL_FAILURE(harness.Verify());
  }
}

TYPED_TEST(ConvStateReplayTest, SegmentedBackingViews) {
  WebGpuExecutionProviderTestOptions options{};
  options.max_storage_buffer_binding_size = 16384 * sizeof(TypeParam);
  auto provider = WebGpuExecutionProviderWithTestOptions(options);
  if (!provider) {
    GTEST_SKIP() << "Artificial storage limits require the native test factory.";
  }
  ConvReplayHarness<TypeParam> harness;
  ASSERT_STATUS_OK(harness.Initialize({2, 1000, 7, 8}, {11001, 13001, 15001}, {1, 1, 1}, std::move(provider)));
  for (const int64_t kept : {0, 3, 7, 8}) {
    ASSERT_STATUS_OK(harness.Prepare(kept));
    ASSERT_STATUS_OK(harness.Run());
    ASSERT_NO_FATAL_FAILURE(harness.Verify());
  }
}

TYPED_TEST(ConvStateReplayTest, RejectsInvalidMetadataBeforeWriting) {
  ConvReplayHarness<TypeParam> harness;
  ASSERT_STATUS_OK(harness.Initialize({2, 3, 5, 7}));
  ASSERT_STATUS_OK(harness.Prepare(3));
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
  harness.Descriptor() = valid;
  harness.Descriptor()[4] = 0;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "channels must be positive");
  harness.Descriptor() = valid;
  harness.Descriptor()[7] = 8;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "kept_count must be in [0, capacity]");
  for (size_t field = 0; field < 3; ++field) {
    harness.Descriptor() = valid;
    harness.Descriptor()[field] += 1;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "byte offsets must be element-aligned");
    harness.Descriptor() = valid;
    const auto bytes = field == 0   ? harness.SourceBytes()
                       : field == 1 ? harness.DestinationBytes()
                                    : harness.CaptureBytes();
    harness.Descriptor()[field] = static_cast<int64_t>(bytes);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "view exceeds its backing tensor");
  }
  for (size_t field = 3; field < 7; ++field) {
    harness.Descriptor() = valid;
    harness.Descriptor()[field] = std::numeric_limits<uint32_t>::max();
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(), "geometry exceeds the uint32 indexing range");
  }
  harness.VerifyDestinationUnchanged();
  harness.Descriptor() = valid;
  ASSERT_STATUS_OK(harness.Run());
  harness.Verify();
}

TYPED_TEST(ConvStateReplayTest, RequiresPreboundDestinationAlias) {
  ConvReplayHarness<TypeParam> harness;
  ASSERT_STATUS_OK(harness.Initialize({2, 3, 5, 7}));
  ASSERT_STATUS_OK(harness.Prepare(4));
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(false), "output must be prebound to destination_backing");
  harness.VerifyDestinationUnchanged();
  ASSERT_STATUS_OK(harness.Run());
  harness.Verify();
}

TYPED_TEST(ConvStateReplayTest, RejectsWritableInputAliasing) {
  ConvReplayHarness<TypeParam> harness;
  ASSERT_STATUS_OK(harness.Initialize({2, 3, 5, 7}));
  ASSERT_STATUS_OK(harness.Prepare(4));
  for (int index = 0; index < 2; ++index) {
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(harness.Run(true, index), "destination backing must be distinct");
  }
  harness.VerifyDestinationUnchanged();
}

#endif  // USE_WEBGPU

}  // namespace
}  // namespace onnxruntime::test
