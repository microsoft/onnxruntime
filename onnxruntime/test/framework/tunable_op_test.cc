// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <chrono>

#include "core/common/common.h"
#include "core/framework/tunable.h"

using namespace std::chrono_literals;

namespace onnxruntime {
namespace test {
namespace {

// test on CPU and it does not use stream
using StreamT = void*;

class Timer : public ::onnxruntime::tunable::Timer<StreamT> {
 public:
  using TimerBase = ::onnxruntime::tunable::Timer<StreamT>;

  explicit Timer(StreamT stream) : TimerBase{stream} {}
  ~Timer() = default;

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

using OpParams = ::onnxruntime::tunable::OpParams<StreamT>;

template <typename ParamsT>
using Op = ::onnxruntime::tunable::Op<ParamsT>;

template <typename ParamsT>
using TunableOp = ::onnxruntime::tunable::TunableOp<ParamsT, Timer>;

}  // namespace

struct VecAddParams : ::onnxruntime::tunable::OpParams<StreamT> {
  VecAddParams(const int* a_buf, const int* b_buf, int* c_buf, int num_elem, int beta)
      : ::onnxruntime::tunable::OpParams<StreamT>(nullptr),
        a(a_buf),
        b(b_buf),
        c(c_buf),
        num_elem(num_elem),
        beta(beta) {}

  std::string Signature() const {
    return std::to_string(num_elem) + "_" + std::to_string(beta);
  }

  const int* a;
  const int* b;
  int* c;
  int num_elem;
  int beta;
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

  tunable::Op<VecAddParams> vec_add(VecAddFunc);

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

  tunable::Op<VecAddParams> vec_add([](const VecAddParams* params) {
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
  tunable::Op<VecAddParams> vec_add([non_copyable = std::move(non_copyable)](const VecAddParams* params) {
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

  tunable::Op<VecAddParams> vec_add(VecAddConstFunctor{});

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

  tunable::Op<VecAddParams> vec_add(VecAddMutableFunctor{});

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddMoveOnlyFunctor {
 public:
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

  tunable::Op<VecAddParams> vec_add(VecAddMoveOnlyFunctor{});

  auto status = vec_add(&params);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(c, 7500042);
}

class VecAddWithIsSupportedMethod {
 public:
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
    tunable::Op<VecAddParams> vec_add(VecAddMoveOnlyFunctor{});
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
    tunable::Op<VecAddParams> vec_add(VecAddWithIsSupportedMethod{});

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
};

TEST(TunableOp, SelectFast) {
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
  op.EnableTuning();

  auto status = op(&params);
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
  op.EnableTuning();

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
  op.EnableTuning();

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
  op.DisableTuning();

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
      GSL_SUPPRESS(i .11)
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
  VecAddParamsRecordLastRun params(&a, &b, &c, 1, 0);

  {
    // NO INPLACE UPDATE is carried out during tuning. We automatically get correct result.
    c = 4200;
    params.beta = 0;
    TunableVecAddNotHandleInplaceUpdate op_not_handle_inplace_update{};
    op_not_handle_inplace_update.EnableTuning();
    auto status = op_not_handle_inplace_update(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7500042);
  }

  {
    // inplace update in this case, If we don't process the params, we will get incorrect result (during tuning)
    c = 4200;
    params.beta = 1;
    TunableVecAddNotHandleInplaceUpdate op_not_handle_inplace_update{};
    op_not_handle_inplace_update.EnableTuning();
    auto status = op_not_handle_inplace_update(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_NE(c, 4200);     // value should be changed
    ASSERT_NE(c, 7504242);  // NOT EQUAL to the expected result
  }

  {
    // NO INPLACE UPDATE is carried out during tuning. We skip params processing. And we get correct result.
    c = 4200;
    params.beta = 0;
    TunableVecAddHandleInplaceUpdate op{};
    op.EnableTuning();
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7500042);
    ASSERT_EQ(op.is_proxy_params_used, false);
  }

  {
    // inplace update in this case, We will handle the buffer and we will get correct result
    c = 4200;
    params.beta = 1;
    TunableVecAddHandleInplaceUpdate op{};
    op.EnableTuning();
    auto status = op(&params);
    ASSERT_TRUE(status.IsOK());
    ASSERT_EQ(c, 7504242);
    ASSERT_EQ(op.is_proxy_params_used, true);
  }
#endif
}

}  // namespace tuning
}  // namespace test
}  // namespace onnxruntime
