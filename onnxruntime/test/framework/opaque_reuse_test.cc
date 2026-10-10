// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <sstream>

#include "core/framework/allocation_planner.h"
#include "core/framework/customregistry.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

class TestPrng {
 public:
  TestPrng() { ++created; }
  ~TestPrng() { ++destroyed; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestPrng);

  void Seed(uint64_t seed) { state_ = seed; }
  uint64_t State() const { return state_; }
  void CopyFrom(const TestPrng& other) {
    state_ = other.state_;
    ++copied;
  }
  float Next() {
    uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<float>(value >> 40) * (1.0f / 16777216.0f);
  }

  static inline std::atomic<int> created{0}, destroyed{0}, copied{0}, reused{0};

 private:
  uint64_t state_{0};
};

extern const char kPrngDomain[] = "ort.test.prng";
extern const char kPrngType[] = "State";
extern const char kOtherPrngDomain[] = "ort.test.other";
extern const char kOtherPrngType[] = "OtherState";
class OtherDomainPrng : public TestPrng {};
class OtherNamePrng : public TestPrng {};
}  // namespace test

ORT_REGISTER_OPAQUE_TYPE(test::TestPrng, test::kPrngDomain, test::kPrngType);
ORT_REGISTER_OPAQUE_TYPE(test::OtherDomainPrng, test::kOtherPrngDomain, test::kPrngType);
ORT_REGISTER_OPAQUE_TYPE(test::OtherNamePrng, test::kPrngDomain, test::kOtherPrngType);

namespace test {
namespace {
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::TypeProto;

class SeedPrng final : public OpKernel {
 public:
  explicit SeedPrng(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override {
    ctx->Output<TestPrng>(0)->Seed(static_cast<uint64_t>(*ctx->Input<Tensor>(0)->Data<int64_t>()));
    return Status::OK();
  }
};

template <typename InputState, typename OutputState>
class GeneratePrng final : public OpKernel {
 public:
  explicit GeneratePrng(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override {
    const TestPrng* input = ctx->Input<InputState>(0);
    TestPrng* output = ctx->Output<OutputState>(1);
    if (input != output) {
      output->CopyFrom(*input);
    } else {
      ++TestPrng::reused;
    }
    *ctx->Output(0, TensorShape{})->MutableData<float>() = output->Next();
    return Status::OK();
  }
};

class OpaqueReuseTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // The datatype singleton outlives every registry, session and OrtValue in these tests.
    [[maybe_unused]] static const bool registered = [] {
      DataTypeImpl::RegisterDataType(DataTypeImpl::GetType<TestPrng>());
      DataTypeImpl::RegisterDataType(DataTypeImpl::GetType<OtherDomainPrng>());
      DataTypeImpl::RegisterDataType(DataTypeImpl::GetType<OtherNamePrng>());
      return true;
    }();
  }

  void SetUp() override {
    TestPrng::created = TestPrng::destroyed = TestPrng::copied = TestPrng::reused = 0;
    registry_ = std::make_shared<CustomRegistry>();
    std::vector<OpSchema> schemas;
    OpSchema seed("Seed", __FILE__, __LINE__);
    seed.SetDomain(kPrngDomain).SinceVersion(1);
    seed.Input(0, "seed", "Scalar seed", "tensor(int64)");
    seed.Output(0, "state", "PRNG state", "T");
    seed.TypeConstraint("T", {"opaque(ort.test.prng,State)"}, "Test-only PRNG state");
    schemas.push_back(seed);
    RegisterGenerate<TestPrng, TestPrng>(schemas, "Generate");
    RegisterGenerate<TestPrng, TestPrng>(schemas, "GenerateCopy", false);
    RegisterGenerate<TestPrng, OtherNamePrng>(schemas, "ToOtherName");
    RegisterGenerate<OtherNamePrng, TestPrng>(schemas, "FromOtherName");
    RegisterGenerate<TestPrng, OtherDomainPrng>(schemas, "ToOtherDomain");
    RegisterGenerate<OtherDomainPrng, TestPrng>(schemas, "FromOtherDomain");
    ASSERT_STATUS_OK(registry_->RegisterOpSet(schemas, kPrngDomain, 1, 1));
    KernelDefBuilder seed_def;
    seed_def.SetName("Seed").SetDomain(kPrngDomain).SinceVersion(1).Provider(kCpuExecutionProvider);
    seed_def.TypeConstraint("T", DataTypeImpl::GetType<TestPrng>());
    ASSERT_STATUS_OK(registry_->RegisterCustomKernel(
        seed_def, [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) {
          kernel = std::make_unique<SeedPrng>(info);
          return Status::OK();
        }));
    model_ = std::make_unique<Model>(
        "OpaqueReuse", false, ModelMetaData(), PathString(),
        IOnnxRuntimeOpSchemaRegistryList{registry_->GetOpschemaRegistry()},
        std::unordered_map<std::string, int>{{kPrngDomain, 1}}, std::vector<ONNX_NAMESPACE::FunctionProto>{},
        DefaultLoggingManager().DefaultLogger());
  }

  void TearDown() override {
    EXPECT_EQ(TestPrng::created.load(), TestPrng::destroyed.load());
  }

  template <typename InputState, typename OutputState>
  void RegisterGenerate(std::vector<OpSchema>& schemas, const char* name, bool may_inplace = true) {
    const auto* input_type = DataTypeImpl::GetType<InputState>();
    const auto* output_type = DataTypeImpl::GetType<OutputState>();
    OpSchema generate(name, __FILE__, __LINE__);
    generate.SetDomain(kPrngDomain).SinceVersion(1);
    generate.Input(0, "state", "PRNG state", "TInput");
    generate.Output(0, "value", "Scalar sample", "tensor(float)");
    generate.Output(1, "next_state", "Updated PRNG state", "TOutput");
    generate.TypeConstraint("TInput", {DataTypeImpl::ToString(input_type)}, "Test-only PRNG state");
    generate.TypeConstraint("TOutput", {DataTypeImpl::ToString(output_type)}, "Test-only PRNG state");
    schemas.push_back(generate);
    KernelDefBuilder def;
    def.SetName(name).SetDomain(kPrngDomain).SinceVersion(1).Provider(kCpuExecutionProvider);
    def.TypeConstraint("TInput", input_type).TypeConstraint("TOutput", output_type);
    if (may_inplace) def.MayInplace(0, 1);
    ASSERT_STATUS_OK(registry_->RegisterCustomKernel(
        def, [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) {
          kernel = std::make_unique<GeneratePrng<InputState, OutputState>>(info);
          return Status::OK();
        }));
  }

  NodeArg& State(const std::string& name, MLDataType type = DataTypeImpl::GetType<TestPrng>()) {
    return model_->MainGraph().GetOrCreateNodeArg(name, type->GetTypeProto());
  }

  NodeArg& Scalar(const std::string& name, int element_type) {
    TypeProto type;
    type.mutable_tensor_type()->set_elem_type(element_type);
    type.mutable_tensor_type()->mutable_shape();
    return model_->MainGraph().GetOrCreateNodeArg(name, &type);
  }

  Node& AddSeed() {
    return model_->MainGraph().AddNode(
        "seed", "Seed", "", {&Scalar("seed", ONNX_NAMESPACE::TensorProto_DataType_INT64)},
        {&State("s0")}, nullptr, kPrngDomain);
  }

  template <typename InputState = TestPrng, typename OutputState = TestPrng>
  Node& AddGenerate(const std::string& input, const std::string& value, const std::string& state,
                    const char* op = "Generate") {
    return model_->MainGraph().AddNode(
        value, op, "", {&State(input, DataTypeImpl::GetType<InputState>())},
        {&Scalar(value, ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
         &State(state, DataTypeImpl::GetType<OutputState>())},
        nullptr, kPrngDomain);
  }

  std::unique_ptr<InferenceSession> Initialize(const std::vector<std::string>& outputs,
                                               ExecutionMode mode = ExecutionMode::ORT_SEQUENTIAL) {
    auto& graph = model_->MainGraph();
    std::vector<const NodeArg*> output_args;
    for (const auto& name : outputs) output_args.push_back(graph.GetNodeArg(name));
    graph.SetOutputs(output_args);
    ORT_THROW_IF_ERROR(graph.Resolve());
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.execution_mode = mode;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 2;
    auto session = std::make_unique<InferenceSession>(options, GetEnvironment());
    ORT_THROW_IF_ERROR(session->RegisterCustomRegistry(registry_));
    std::istringstream stream(model_->ToProto().SerializeAsString());
    ORT_THROW_IF_ERROR(session->Load(stream));
    ORT_THROW_IF_ERROR(session->Initialize());
    return session;
  }

  static NameMLValMap SeedFeed() {
    OrtValue seed;
    CreateMLValue<int64_t>(CPUAllocator::DefaultInstance(), {}, {42}, &seed);
    return {{"seed", seed}};
  }

  static const AllocPlanPerValue& Plan(const InferenceSession& session, const std::string& name) {
    const auto& state = session.GetSessionState();
    int index;
    ORT_THROW_IF_ERROR(state.GetOrtValueNameIdxMap().GetIdx(name, index));
    return state.GetPerValueAllocPlan().at(index);
  }

  static std::vector<OrtValue> Run(InferenceSession& session, const NameMLValMap& feeds,
                                   const std::vector<std::string>& outputs, bool only_fetches = false) {
    RunOptions options;
    options.only_execute_path_to_fetches = only_fetches;
    std::vector<OrtValue> values;
    ORT_THROW_IF_ERROR(session.Run(options, feeds, outputs, &values));
    return values;
  }

  static OrtValue PrngValue(uint64_t seed) {
    const auto* type = DataTypeImpl::GetType<TestPrng>();
    OrtValue value;
    value.Init(type->AsNonTensorType()->GetCreateFunc()(), type, type->GetDeleteFunc());
    value.GetMutable<TestPrng>()->Seed(seed);
    return value;
  }

  static void ExpectSamples(const std::vector<OrtValue>& values) {
    // SplitMix64 seed 42: the upper 24 bits of its first two outputs.
    EXPECT_EQ(values[0].Get<Tensor>().Shape().NumDimensions(), 0u);
    EXPECT_EQ(values[1].Get<Tensor>().Shape().NumDimensions(), 0u);
    EXPECT_EQ(*values[0].Get<Tensor>().Data<float>(), 12441394.0f / 16777216.0f);
    EXPECT_EQ(*values[1].Get<Tensor>().Data<float>(), 2682851.0f / 16777216.0f);
  }

  std::shared_ptr<CustomRegistry> registry_;
  std::unique_ptr<Model> model_;
};

TEST_F(OpaqueReuseTest, LinearReuseAndLifetime) {
  AddSeed();
  AddGenerate("s0", "v1", "s1");
  AddGenerate("s1", "v2", "s2");
  AddGenerate("s2", "v3", "s3");
  auto session = Initialize({"v1", "v2", "s3"});
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kReuse);
  EXPECT_EQ(Plan(*session, "s2").alloc_kind, AllocKind::kReuse);
  int initial_index;
  ASSERT_STATUS_OK(session->GetSessionState().GetOrtValueNameIdxMap().GetIdx("s0", initial_index));
  EXPECT_EQ(Plan(*session, "s1").reused_buffer, initial_index);
  EXPECT_EQ(Plan(*session, "s2").reused_buffer, initial_index);
  EXPECT_EQ(Plan(*session, "s3").alloc_kind, AllocKind::kAllocateOutput);
  for (int run = 0; run < 3; ++run) {
    SCOPED_TRACE(run);
    auto values = Run(*session, SeedFeed(), {"v1", "v2", "s3"}, run == 1);
    ExpectSamples(values);
    EXPECT_EQ(TestPrng::reused.load(), 2 * (run + 1));
    EXPECT_EQ(TestPrng::copied.load(), run + 1);
    EXPECT_EQ(TestPrng::created.load(), 2 * (run + 1));
    EXPECT_EQ(TestPrng::destroyed.load(), 2 * run + 1);
    OrtValue retained = values[2];
    const auto* object = &retained.Get<TestPrng>();
    values.clear();
    EXPECT_EQ(&retained.Get<TestPrng>(), object);
    EXPECT_EQ(TestPrng::destroyed.load(), 2 * run + 1);
    retained = OrtValue{};
    EXPECT_EQ(TestPrng::destroyed.load(), 2 * (run + 1));
  }
}

TEST_F(OpaqueReuseTest, ForkPreservesBothSequences) {
  AddSeed();
  AddGenerate("s0", "a1", "sa1");
  AddGenerate("s0", "b1", "sb1");
  AddGenerate("sa1", "a2", "sa2");
  AddGenerate("sb1", "b2", "sb2");
  auto session = Initialize({"a1", "a2", "b1", "b2", "sa2", "sb2"});
  // One fork may reuse the last reference; the other must preserve s0 for its sibling.
  const bool a_reuses = Plan(*session, "sa1").alloc_kind == AllocKind::kReuse;
  const bool b_reuses = Plan(*session, "sb1").alloc_kind == AllocKind::kReuse;
  EXPECT_NE(a_reuses, b_reuses);
  for (int run = 0; run < 3; ++run) {
    auto values = Run(*session, SeedFeed(), {"a1", "a2", "b1", "b2", "sa2", "sb2"});
    ExpectSamples(values);
    EXPECT_EQ(*values[0].Get<Tensor>().Data<float>(), *values[2].Get<Tensor>().Data<float>());
    EXPECT_EQ(*values[1].Get<Tensor>().Data<float>(), *values[3].Get<Tensor>().Data<float>());
    EXPECT_NE(&values[4].Get<TestPrng>(), &values[5].Get<TestPrng>());
    EXPECT_EQ(values[4].Get<TestPrng>().State(), values[5].Get<TestPrng>().State());
    EXPECT_EQ(TestPrng::reused.load(), run + 1);
    EXPECT_EQ(TestPrng::copied.load(), 3 * (run + 1));
  }
}

TEST_F(OpaqueReuseTest, GraphOutputRemainsObservable) {
  AddSeed();
  AddGenerate("s0", "v1", "s1");
  AddGenerate("s1", "v2", "s2");
  auto session = Initialize({"v1", "v2", "s0", "s2"});
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  auto values = Run(*session, SeedFeed(), {"v1", "v2", "s0", "s2"});
  ExpectSamples(values);
  EXPECT_EQ(values[2].Get<TestPrng>().State(), 42u);
  EXPECT_NE(&values[2].Get<TestPrng>(), &values[3].Get<TestPrng>());
  EXPECT_EQ(TestPrng::reused.load(), 0);
}

TEST_F(OpaqueReuseTest, GraphInputRemainsObservableAcrossRuns) {
  AddGenerate("s0", "v1", "s1");
  AddGenerate("s1", "v2", "s2");
  auto session = Initialize({"v1", "v2", "s2"});
  EXPECT_EQ(Plan(*session, "s0").alloc_kind, AllocKind::kPreExisting);
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  const auto* type = DataTypeImpl::GetType<TestPrng>();
  OrtValue input;
  input.Init(type->AsNonTensorType()->GetCreateFunc()(), type, type->GetDeleteFunc());
  input.GetMutable<TestPrng>()->Seed(42);
  for (int run = 0; run < 3; ++run) {
    auto values = Run(*session, {{"s0", input}}, {"v1", "v2", "s2"});
    ExpectSamples(values);
    EXPECT_EQ(input.Get<TestPrng>().State(), 42u);
    EXPECT_NE(&input.Get<TestPrng>(), &values[2].Get<TestPrng>());
  }
  EXPECT_EQ(TestPrng::reused.load(), 0);
}

TEST_F(OpaqueReuseTest, GraphInputOutputAliasMutatesCallerObject) {
  AddGenerate("s0", "v1", "s1");
  auto session = Initialize({"v1", "s1"});
  OrtValue state = PrngValue(42);
  const auto* object = &state.Get<TestPrng>();
  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session->NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("s0", state));
  ASSERT_STATUS_OK(binding->BindOutput("v1"));
  ASSERT_STATUS_OK(binding->BindOutput("s1", state));

  uint64_t expected_state = 42;
  for (int run = 0; run < 3; ++run) {
    ASSERT_STATUS_OK(session->Run(RunOptions{}, *binding));
    expected_state += 0x9e3779b97f4a7c15ULL;

    ASSERT_EQ(binding->GetOutputs().size(), 2u);
    EXPECT_EQ(&binding->GetOutputs()[1].Get<TestPrng>(), object);
    EXPECT_EQ(&state.Get<TestPrng>(), object);
    EXPECT_EQ(state.Get<TestPrng>().State(), expected_state);
    EXPECT_EQ(TestPrng::reused.load(), run + 1);
  }

  EXPECT_EQ(TestPrng::copied.load(), 0);
  EXPECT_EQ(TestPrng::created.load(), 1);
}

TEST_F(OpaqueReuseTest, DistinctGraphInputOutputPreservesCallerObject) {
  AddGenerate("s0", "v1", "s1");
  auto session = Initialize({"v1", "s1"});
  OrtValue state = PrngValue(42);
  std::vector<OrtValue> fetches;
  const NameMLValMap feeds{{"s0", state}};
  const std::vector<std::string> outputs{"v1", "s1"};

  ASSERT_STATUS_OK(session->Run(RunOptions{}, feeds, outputs, &fetches));

  EXPECT_EQ(state.Get<TestPrng>().State(), 42u);
  EXPECT_NE(&state.Get<TestPrng>(), &fetches[1].Get<TestPrng>());
  EXPECT_EQ(fetches[1].Get<TestPrng>().State(), 42u + 0x9e3779b97f4a7c15ULL);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 1);
}

TEST_F(OpaqueReuseTest, GraphInputOutputAliasRequiresMayInplace) {
  AddGenerate("s0", "v1", "s1", "GenerateCopy");
  auto session = Initialize({"v1", "s1"});
  OrtValue state = PrngValue(42);
  std::vector<OrtValue> fetches(2);
  fetches[1] = state;
  const NameMLValMap feeds{{"s0", state}};
  const std::vector<std::string> outputs{"v1", "s1"};

  const auto status = session->Run(RunOptions{}, feeds, outputs, &fetches);

  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("does not declare MayInplace"));
  EXPECT_EQ(state.Get<TestPrng>().State(), 42u);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 0);
}

TEST_F(OpaqueReuseTest, GraphInputOutputAliasRequiresLinearUse) {
  AddGenerate("s0", "v1", "s1");
  AddGenerate("s0", "v2", "s2");
  auto session = Initialize({"v1", "v2", "s1"});
  OrtValue state = PrngValue(42);
  std::vector<OrtValue> fetches(3);
  fetches[2] = state;
  const NameMLValMap feeds{{"s0", state}};
  const std::vector<std::string> outputs{"v1", "v2", "s1"};

  const auto status = session->Run(RunOptions{}, feeds, outputs, &fetches);

  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("direct linear input-to-output kernel path"));
  EXPECT_EQ(state.Get<TestPrng>().State(), 42u);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 0);
}

TEST_F(OpaqueReuseTest, MatchingTypeWithoutMayInplaceDoesNotReuse) {
  AddSeed();
  AddGenerate("s0", "v1", "s1", "GenerateCopy");
  AddGenerate("s1", "v2", "s2", "GenerateCopy");
  auto session = Initialize({"v1", "v2", "s2"});
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  auto values = Run(*session, SeedFeed(), {"v1", "v2", "s2"});
  ExpectSamples(values);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 2);
}

TEST_F(OpaqueReuseTest, ParallelExecutionDoesNotReuse) {
  AddSeed();
  AddGenerate("s0", "v1", "s1");
  AddGenerate("s1", "v2", "s2");
  auto session = Initialize({"v1", "v2", "s2"}, ExecutionMode::ORT_PARALLEL);
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  auto values = Run(*session, SeedFeed(), {"v1", "v2", "s2"});
  ExpectSamples(values);
  EXPECT_EQ(TestPrng::reused.load(), 0);
}

TEST_F(OpaqueReuseTest, DifferentOpaqueNamesDoNotReuse) {
  AddSeed();
  AddGenerate<TestPrng, OtherNamePrng>("s0", "v1", "s1", "ToOtherName");
  AddGenerate<OtherNamePrng, TestPrng>("s1", "v2", "s2", "FromOtherName");
  auto session = Initialize({"v1", "v2", "s2"});
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  auto values = Run(*session, SeedFeed(), {"v1", "v2", "s2"});
  ExpectSamples(values);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 2);
}

TEST_F(OpaqueReuseTest, DifferentOpaqueDomainsDoNotReuse) {
  AddSeed();
  AddGenerate<TestPrng, OtherDomainPrng>("s0", "v1", "s1", "ToOtherDomain");
  AddGenerate<OtherDomainPrng, TestPrng>("s1", "v2", "s2", "FromOtherDomain");
  auto session = Initialize({"v1", "v2", "s2"});
  EXPECT_EQ(Plan(*session, "s1").alloc_kind, AllocKind::kAllocate);
  auto values = Run(*session, SeedFeed(), {"v1", "v2", "s2"});
  ExpectSamples(values);
  EXPECT_EQ(TestPrng::reused.load(), 0);
  EXPECT_EQ(TestPrng::copied.load(), 2);
}

TEST_F(OpaqueReuseTest, OptimizerAllocatesDistinctOwnedObjects) {
  const auto& seed = AddSeed();
  const auto& generate = AddGenerate("s0", "v1", "s1");
  ASSERT_STATUS_OK(model_->MainGraph().Resolve());
  CPUExecutionProvider provider{CPUExecutionProviderInfo{}};
  const std::function<bool(const std::string&)> is_sparse = [](const std::string&) { return false; };
  OptimizerExecutionFrame::Info info(
      {&seed, &generate}, InitializedTensorSet{}, model_->MainGraph().ModelPath(), provider, is_sparse,
      DefaultLoggingManager().DefaultLogger());
  OrtValue retained;
  {
    OptimizerExecutionFrame frame(info, {info.GetMLValueIndex("s1")});
    OrtValue* initial = nullptr;
    OrtValue* next = nullptr;
    ASSERT_STATUS_OK(frame.GetOrCreateNodeOutputMLValue(
        0, frame.GetNodeOffset(seed.Index()) + 1, nullptr, initial, seed));
    ASSERT_STATUS_OK(frame.GetOrCreateNodeOutputMLValue(
        1, frame.GetNodeOffset(generate.Index()) + 2, nullptr, next, generate));
    ASSERT_NE(initial, nullptr);
    ASSERT_NE(next, nullptr);
    EXPECT_NE(&initial->Get<TestPrng>(), &next->Get<TestPrng>());
    EXPECT_EQ(TestPrng::created.load(), 2);
    retained = *next;
  }
  EXPECT_EQ(TestPrng::destroyed.load(), 1);
  retained = OrtValue{};
  EXPECT_EQ(TestPrng::destroyed.load(), 2);
}
}  // namespace
}  // namespace test
}  // namespace onnxruntime
