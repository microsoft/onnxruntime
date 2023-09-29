// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <chrono>

#include "core/common/common.h"
#include "core/framework/tunable.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL

using namespace std::chrono_literals;

namespace onnxruntime {
namespace test {
namespace {

// test on CPU and it does not use stream
using StreamT = void*;

constexpr static const char* kTestKey = "THE_TEST_KEY";
constexpr static const char* kValidTestValue = "THE_VALID_TEST_VALUE";

static std::string GetTestValue() {
  return kValidTestValue;
}

static Status ValidateTestValue(const std::string& value) {
  auto current = GetTestValue();
  ORT_RETURN_IF(current != value, "Only ", kValidTestValue, " is valid for key ", kTestKey);
  return Status::OK();
}

class TestTuningResultsValidator : public TuningResultsValidator {
 public:
  TestTuningResultsValidator() {
    RegisterValidator(kTestKey, GetTestValue, ValidateTestValue);
  };

 protected:
  std::string GetOrtBuildConfig() const override {
    return "TEST_BUILD";
  }
};

class TestTuningContext : public ITuningContext {
 public:
  using ITuningContext::ITuningContext;

  void EnableTunableOp() override { op_enabled_ = true; }
  void DisableTunableOp() override { op_enabled_ = false; }
  bool IsTunableOpEnabled() const override { return op_enabled_; }

  void EnableTuning() override { tuning_enabled_ = true; }
  void DisableTuning() override { tuning_enabled_ = false; }
  bool IsTuningEnabled() const override { return tuning_enabled_; }

  void SetMaxTuningDurationMs(int max_duration_ms) override { max_tuning_duration_ms_ = max_duration_ms; }
  int GetMaxTuningDurationMs() const override {
    return max_tuning_duration_ms_ > 0 ? max_tuning_duration_ms_ : std::numeric_limits<int>::max();
  }

  TuningResultsManager& GetTuningResultsManager() override { return manager_; }
  const TuningResultsManager& GetTuningResultsManager() const override { return manager_; }

  const TuningResultsValidator& GetTuningResultsValidator() const override { return validator_; }

  void ClearCache() { manager_.Clear(); }

 private:
  bool op_enabled_{false};
  bool tuning_enabled_{false};
  int max_tuning_duration_ms_{};
  TuningResultsManager manager_{};
  TestTuningResultsValidator validator_{};
};

class TestEP : public IExecutionProvider {
  static constexpr const char* kEPType = "TestEP";
  TestTuningContext tuning_ctx_{this};

 public:
  TestEP() : IExecutionProvider{kEPType, true} {}

  ITuningContext* GetTuningContext() const override {
    return const_cast<TestTuningContext*>(&tuning_ctx_);
  }

  void ClearCache() { tuning_ctx_.ClearCache(); }
};

class TestTimer : public ITimer<StreamT> {
 public:
  using TimerBase = ITimer<StreamT>;

  explicit TestTimer(StreamT stream) : TimerBase{stream} {}
  ~TestTimer() = default;

  void Start() override {
    start_ = std::chrono::steady_clock::now();
  }

  void End() override {
    end_ = std::chrono::steady_clock::now();
  }

  float Duration() override {
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end_ - start_).count();
  }

 private:
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  TimePoint start_;
  TimePoint end_;
};

using OpParams = OpParams<TestTuningContext, void*>;

template <typename ParamsT>
using Op = Op<ParamsT>;

template <typename ParamsT>
using TunableOp = TunableOp<ParamsT, TestTimer>;

}  // namespace

struct VecAddParams : OpParams {
  VecAddParams(const int* a_buf, const int* b_buf, int* c_buf, int num_elem, int beta)
      : OpParams(nullptr, nullptr),
        a(a_buf),
        b(b_buf),
        c(c_buf),
        num_elem(num_elem),
        beta(beta) {
    ep = std::make_shared<TestEP>();
    tuning_ctx = static_cast<TestTuningContext*>(ep->GetTuningContext());
  }

  std::string Signature() const {
    return std::to_string(num_elem) + "_" + std::to_string(beta);
  }

  const int* a;
  const int* b;
  int* c;
  int num_elem;
  int beta;

  // Create a temporary tuning context for the test case.
  std::shared_ptr<TestEP> ep;
};

void LaunchVecAddKernel(const int* a, const int* b, int* c, int num_elem, int beta) {
  for (int i = 0; i < num_elem; i++) {
    c[i] = a[i] + b[i] + beta * c[i];
  }
}

Status VecAddFunc(const VecAddParams* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->c == nullptr, "output buffer cannot be nullptr");
  LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
  return Status::OK();
}

namespace wrapper {

TEST(TunableOp, OpWrapsFunction) {
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  Op<VecAddParams> vec_add(VecAddFunc);

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);

  params.c = nullptr;
  status = vec_add(&params);
  ASSERT_EQ(status.Category(), common::StatusCategory::NONE);
  ASSERT_EQ(status.Code(), common::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr("output buffer cannot be nullptr"));
}

TEST(TunableOp, OpWrapsLambda) {
  constexpr int a = 7500000;
  constexpr int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  Op<VecAddParams> vec_add([](const VecAddParams* params) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  });

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

TEST(TunableOp, OpWrapsMoveOnlyLambda) {
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  auto non_copyable = std::make_unique<int>(0);
  Op<VecAddParams> vec_add([non_copyable = std::move(non_copyable)](const VecAddParams* params) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  });

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddConstFunctor {
 public:
  Status operator()(const VecAddParams* params) const {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  }
};

TEST(TunableOp, OpWrapsConstFunctor) {
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  Op<VecAddParams> vec_add(VecAddConstFunctor{});

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddMutableFunctor {
 public:
  Status operator()(const VecAddParams* params) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  }
};

TEST(TunableOp, OpWrapsMutableFunctor) {
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  Op<VecAddParams> vec_add(VecAddMutableFunctor{});

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddMoveOnlyFunctor {
 public:
  VecAddMoveOnlyFunctor() = default;
  VecAddMoveOnlyFunctor(VecAddMoveOnlyFunctor&&) = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(VecAddMoveOnlyFunctor);

  Status operator()(const VecAddParams* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->c == nullptr, "output buffer cannot be nullptr");
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  }
};

TEST(TunableOp, OpWrapsMoveOnlyFunctor) {
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParams params(&a, &b, &c, 1, 0);

  Op<VecAddParams> vec_add(VecAddMoveOnlyFunctor{});

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddWithIsSupportedMethod {
 public:
  VecAddWithIsSupportedMethod() = default;
  VecAddWithIsSupportedMethod(VecAddWithIsSupportedMethod&&) = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(VecAddWithIsSupportedMethod);

  Status operator()(const VecAddParams* params) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
    return Status::OK();
  }

  Status IsSupported(const VecAddParams* params) {
    // Purely for testing purpose. In real world, this methods must be crafted with excessive carefulness.
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->num_elem != 4, "only support num_elem == 4");
    return Status::OK();
  }
};

TEST(TunableOp, OpWrapsFunctorWithExtendedIsSupported) {
  constexpr const int a[] = {0, 1, 2, 3};
  constexpr const int b[] = {42, 42, 42, 42};
  int c[4] = {};

  Status status;

  // Test Op::IsSupported will have correct fallback if user does not implement it in its functor.
  {
    Op<VecAddParams> vec_add(VecAddMoveOnlyFunctor{});
    VecAddParams params(a, b, nullptr, 1, 0);
    status = vec_add.IsSupported(&params);
    ASSERT_EQ(status.Category(), common::StatusCategory::NONE);
    ASSERT_EQ(status.Code(), common::StatusCode::INVALID_ARGUMENT);
    ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr("output buffer cannot be nullptr"));

    params.c = c;
    status = vec_add.IsSupported(&params);
    ASSERT_TRUE(status.IsOK());
  }

  // Test Op::IsSupported will use user provided one if they implemented it.
  {
    Op<VecAddParams> vec_add(VecAddWithIsSupportedMethod{});

    VecAddParams params(a, b, c, 4, 0);
    status = vec_add.IsSupported(&params);
    ASSERT_TRUE(status.IsOK());

    params.num_elem = 1;
    status = vec_add.IsSupported(&params);
    ASSERT_EQ(status.Category(), common::StatusCategory::NONE);
    ASSERT_EQ(status.Code(), common::StatusCode::INVALID_ARGUMENT);
    ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr("only support num_elem == 4"));
  }
}

}  // namespace wrapper

namespace tuning {

struct VecAddParamsRecordLastRun : public VecAddParams {
  using VecAddParams::VecAddParams;

  std::string* last_run{nullptr};
};

Status SlowFull(const VecAddParamsRecordLastRun* params) {
  *(params->last_run) = "SlowFull";
  for (int i = 0; i < 1000000; i++) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
  }
  std::this_thread::sleep_for(5ms);
  return Status::OK();
}

Status FastFull(const VecAddParamsRecordLastRun* params) {
  *(params->last_run) = "FastFull";
  for (int i = 0; i < 3000; i++) {
    LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
  }
  return Status::OK();
}

Status FastestNarrow(const VecAddParamsRecordLastRun* params) {
  *(params->last_run) = "FastestNarrow";
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(params->num_elem != 4, "FastestNarrow only supports VecAdd 4 elements");
  LaunchVecAddKernel(params->a, params->b, params->c, params->num_elem, params->beta);
  return Status::OK();
}

class TunableVecAddSelectFast : public TunableOp<VecAddParamsRecordLastRun> {
 public:
  TunableVecAddSelectFast() {
    this->RegisterOp(SlowFull);
    this->RegisterOp(FastFull);
  }

  constexpr static int kSlowFullId = 0;
  constexpr static int kFastFullId = 1;
};

TEST(TunableOp, SelectFastIfTuning) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  TunableVecAddSelectFast op{};
  // Only enable op usage, slow (default) should be selected
  params.TuningContext()->EnableTunableOp();
  auto status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "SlowFull");

  // Also enable tuning, fast should be selected
  params.TuningContext()->EnableTuning();
  status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "FastFull");

  // Also set max_tuning_duration_ms, fast should be selected
  params.TuningContext()->SetMaxTuningDurationMs(10);
  status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "FastFull");

#endif
}

class TunableVecAddSelectSupported : public TunableOp<VecAddParamsRecordLastRun> {
 public:
  TunableVecAddSelectSupported() {
    this->RegisterOp(SlowFull);
    this->RegisterOp(FastestNarrow);
  }
};

TEST(TunableOp, SelectSupported) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  TunableVecAddSelectSupported op{};
  params.TuningContext()->EnableTunableOpAndTuning();

  auto status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "SlowFull");
#endif
}

class TunableVecAddSelectFastestIfSupported : public TunableOp<VecAddParamsRecordLastRun> {
 public:
  TunableVecAddSelectFastestIfSupported() {
    this->RegisterOp(SlowFull);
    this->RegisterOp(FastFull);
    this->RegisterOp(FastestNarrow);

    this->SetDefaultId(2);
  }
};

TEST(TunableOp, SelectFastestIfSupported) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a[] = {0, 1, 2, 3};
  constexpr const int b[] = {42, 42, 42, 42};
  int c[4] = {};
  VecAddParamsRecordLastRun params(a, b, c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  TunableVecAddSelectFastestIfSupported op{};
  params.TuningContext()->EnableTunableOpAndTuning();

  auto status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "FastFull");

  params.num_elem = 4;
  status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "FastestNarrow");
#endif
}

TEST(TunableOp, DisabledWithManualSelection) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  TunableVecAddSelectFastestIfSupported op{};
  params.TuningContext()->DisableTunableOp();

  auto status = op(&params);
  ASSERT_EQ(last_run, "FastestNarrow");
  ASSERT_EQ(status.Category(), common::StatusCategory::NONE);
  ASSERT_EQ(status.Code(), common::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr("FastestNarrow only supports VecAdd 4 elements"));
#endif
}

class TunableVecAddNotHandleInplaceUpdate : public TunableOp<VecAddParams> {
 public:
  TunableVecAddNotHandleInplaceUpdate() {
    this->RegisterOp(VecAddFunc);
  }
};

#ifndef ORT_NO_RTTI
class TunableVecAddHandleInplaceUpdate : public TunableOp<VecAddParams> {
 public:
  TunableVecAddHandleInplaceUpdate() {
    this->RegisterOp(VecAddFunc);
  }

  const VecAddParams* PreTuning(const VecAddParams* params) override {
    if (params->beta != 0) {
      is_proxy_params_used = true;
      std::unique_ptr<VecAddParams> proxy = std::make_unique<VecAddParams>(*params);
      proxy->c = new int[params->num_elem];
      return proxy.release();
    }
    is_proxy_params_used = false;
    return params;
  }

  void PostTuning(const VecAddParams* params) override {
    if (params->beta != 0) {
      GSL_SUPPRESS(i.11)
      delete[] params->c;
      delete params;
    }
  }

  bool is_proxy_params_used{false};
};
#endif

TEST(TunableOp, HandleInplaceUpdate) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};

  {
    // NO INPLACE UPDATE is carried out during tuning. We automatically get correct result.
    c = 4200;
    VecAddParamsRecordLastRun params(&a, &b, &c, 1, /*beta=*/0);
    TunableVecAddNotHandleInplaceUpdate op_not_handle_inplace_update{};
    params.TuningContext()->EnableTunableOpAndTuning();
    auto status = op_not_handle_inplace_update(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7500042);
  }

  {
    // inplace update in this case, If we don't process the params, we will get incorrect result (during tuning)
    c = 4200;
    VecAddParamsRecordLastRun params(&a, &b, &c, 1, /*beta=*/1);
    TunableVecAddNotHandleInplaceUpdate op_not_handle_inplace_update{};
    params.TuningContext()->EnableTunableOpAndTuning();
    auto status = op_not_handle_inplace_update(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_NE(c, 4200);     // value should be changed
    ASSERT_NE(c, 7504242);  // NOT EQUAL to the expected result
  }

  {
    // NO INPLACE UPDATE is carried out during tuning. We skip params processing. And we get correct result.
    c = 4200;
    VecAddParamsRecordLastRun params(&a, &b, &c, 1, /*beta=*/0);
    TunableVecAddHandleInplaceUpdate op{};
    params.TuningContext()->EnableTunableOpAndTuning();
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7500042);
    ASSERT_EQ(op.is_proxy_params_used, false);
  }

  {
    // inplace update in this case, We will handle the buffer and we will get correct result
    c = 4200;
    VecAddParamsRecordLastRun params(&a, &b, &c, 1, /*beta=*/1);
    TunableVecAddHandleInplaceUpdate op{};
    params.TuningContext()->EnableTunableOpAndTuning();
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7504242);
    ASSERT_EQ(op.is_proxy_params_used, true);
  }
#endif
}

TEST(TunableOp, OpSignatureMustNotChange) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  std::vector<std::string> signatures1;
  std::vector<std::string> signatures2;
  signatures1.emplace_back(TunableVecAddSelectFast{}.Signature());
  signatures1.emplace_back(TunableVecAddSelectSupported{}.Signature());
  signatures1.emplace_back(TunableVecAddSelectFastestIfSupported{}.Signature());
  signatures1.emplace_back(TunableVecAddNotHandleInplaceUpdate{}.Signature());
  signatures1.emplace_back(TunableVecAddHandleInplaceUpdate{}.Signature());

  signatures2.emplace_back(TunableVecAddSelectFast{}.Signature());
  signatures2.emplace_back(TunableVecAddSelectSupported{}.Signature());
  signatures2.emplace_back(TunableVecAddSelectFastestIfSupported{}.Signature());
  signatures2.emplace_back(TunableVecAddNotHandleInplaceUpdate{}.Signature());
  signatures2.emplace_back(TunableVecAddHandleInplaceUpdate{}.Signature());

  ASSERT_EQ(signatures1, signatures2);
#endif
}

TEST(TunableOp, OpSignatureMustNotCollide) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  std::unordered_set<std::string> signatures;
  signatures.insert(TunableVecAddSelectFast{}.Signature());
  signatures.insert(TunableVecAddSelectSupported{}.Signature());
  signatures.insert(TunableVecAddSelectFastestIfSupported{}.Signature());
  signatures.insert(TunableVecAddNotHandleInplaceUpdate{}.Signature());
  signatures.insert(TunableVecAddHandleInplaceUpdate{}.Signature());

  ASSERT_THAT(signatures, ::testing::SizeIs(5));
#endif
}

}  // namespace tuning

namespace tuning_context {

TEST(TuningContext, TunableOpRespectTuningContext) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  tuning::VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  tuning::TunableVecAddSelectFast op{};
  auto* ctx = params.TuningContext();
  auto& mgr = ctx->GetTuningResultsManager();
  ctx->EnableTunableOpAndTuning();

  {
    // Before TunableOp(...), there is no entry in it.
    ASSERT_EQ(mgr.Lookup(op.Signature()).size(), 0u);

    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(last_run, "FastFull");

    // After TunableOp(...), the result entry is corretly written.
    ASSERT_EQ(mgr.Lookup(op.Signature()).size(), 1u);
    ASSERT_EQ(mgr.Lookup(op.Signature(), params.Signature()), tuning::TunableVecAddSelectFast::kFastFullId);
  }

  last_run.clear();
  mgr.Clear();
  {
    ASSERT_EQ(mgr.Lookup(op.Signature()).size(), 0u);

    // TunableOp(...), respect the existing entry (manually loaded) if id in bound
    mgr.Add(op.Signature(), params.Signature(), tuning::TunableVecAddSelectFast::kSlowFullId);
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(last_run, "SlowFull");
  }

  last_run.clear();
  mgr.Clear();
  {
    // TunableOp(...), must not respect the existing entry if id not in bound
    // manually create an out of bound id
    mgr.Add(op.Signature(), params.Signature(), 1000000);
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK()) << "TunableOp should recover from an out of bound id";
    ASSERT_EQ(last_run, "FastFull");
    ASSERT_EQ(mgr.Lookup(op.Signature(), params.Signature()), tuning::TunableVecAddSelectFast::kFastFullId);
  }
#endif
}

TEST(TuningContext, GetAndLoadTuningResults) {
#ifdef ORT_NO_RTTI
  GTEST_SKIP() << "TunableOp needs RTTI to work correctly";
#else
  constexpr const int a = 7500000;
  constexpr const int b = 42;
  int c{};
  tuning::VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);
  std::string last_run;
  params.last_run = &last_run;

  tuning::TunableVecAddSelectFast op{};
  auto* ctx = params.TuningContext();
  ctx->EnableTunableOpAndTuning();

  auto status = op(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(last_run, "FastFull");

  auto trs = ctx->GetTuningResults();
  ASSERT_EQ(trs.ep, "TestEP");

  ASSERT_EQ(trs.validators.size(), TestTuningResultsValidator::mandatory_keys.size() + 1);
  for (const auto& key : TestTuningResultsValidator::mandatory_keys) {
    ASSERT_THAT(trs.validators, ::testing::Contains(::testing::Key(key)));
  }
  ASSERT_THAT(trs.validators, ::testing::Contains(::testing::Key(kTestKey)));

  ASSERT_EQ(trs.results.size(), 1u);
  ASSERT_THAT(trs.results, ::testing::Contains(::testing::Key(op.Signature())));
  ASSERT_THAT(trs.results[op.Signature()], ::testing::Contains(::testing::Key(params.Signature())));
  ASSERT_EQ(trs.results[op.Signature()][params.Signature()], tuning::TunableVecAddSelectFast::kFastFullId);
#endif
}

}  // namespace tuning_context

}  // namespace test
}  // namespace onnxruntime
