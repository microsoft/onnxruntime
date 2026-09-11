// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#include "gtest/gtest.h"

#include <wrl/implements.h>
#include <wrl/client.h>
#include "core/providers/dml/DmlExecutionProvider/src/SafeMakeOrThrow.h"

#include <stdexcept>

namespace onnxruntime {
namespace test {

// A trivial COM interface for testing.
MIDL_INTERFACE("A1B2C3D4-E5F6-7890-ABCD-EF1234567890")
ITestInterface : public IUnknown {
  virtual int STDMETHODCALLTYPE GetValue() = 0;
};

// A RuntimeClass whose constructor succeeds and stores a value.
class SucceedingClass : public Microsoft::WRL::RuntimeClass<
                            Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, ITestInterface> {
 public:
  int value;

  SucceedingClass(int v) : value(v) {}

  int STDMETHODCALLTYPE GetValue() override { return value; }
};

// A RuntimeClass that tracks whether its destructor ran.
class TrackedClass : public Microsoft::WRL::RuntimeClass<
                         Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, ITestInterface> {
 public:
  bool& destroyed;

  TrackedClass(bool& flag) : destroyed(flag) { destroyed = false; }
  ~TrackedClass() { destroyed = true; }

  int STDMETHODCALLTYPE GetValue() override { return 42; }
};

// A RuntimeClass whose constructor always throws.
// Uses a ref-counted witness to verify cleanup: the witness is destroyed
// (via Release) during stack unwinding if memory is freed correctly.
class ThrowingClass : public Microsoft::WRL::RuntimeClass<
                          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, ITestInterface> {
 public:
  Microsoft::WRL::ComPtr<TrackedClass> witness;

  ThrowingClass(bool& witness_destroyed) {
    // Create a witness that will be destroyed when this object's members
    // are cleaned up during stack unwinding.
    witness = Dml::SafeMakeOrThrow<TrackedClass>(witness_destroyed);
    throw std::runtime_error("intentional throw");
  }

  int STDMETHODCALLTYPE GetValue() override { return -1; }
};

// Verify that SafeMakeOrThrow creates an object with ref count 1,
// and that the object is properly released when the ComPtr goes out of scope.
TEST(SafeMakeOrThrowTest, SuccessPath_RefCountIsOne) {
  Microsoft::WRL::ComPtr<SucceedingClass> obj = Dml::SafeMakeOrThrow<SucceedingClass>(123);

  ASSERT_NE(obj.Get(), nullptr);
  EXPECT_EQ(obj->GetValue(), 123);

  // AddRef/Release to observe ref count: AddRef returns new count.
  unsigned long refAfterAdd = obj->AddRef();
  EXPECT_EQ(refAfterAdd, 2u);

  unsigned long refAfterRelease = obj->Release();
  EXPECT_EQ(refAfterRelease, 1u);
}

// Verify that the object is destroyed when the last ComPtr releases it.
TEST(SafeMakeOrThrowTest, SuccessPath_DestructorRunsOnRelease) {
  bool destroyed = false;
  {
    auto obj = Dml::SafeMakeOrThrow<TrackedClass>(destroyed);
    EXPECT_FALSE(destroyed);
  }
  // ComPtr went out of scope — destructor should have run.
  EXPECT_TRUE(destroyed);
}

// Verify that copying the ComPtr increments the ref count and
// the object survives until the last reference is released.
TEST(SafeMakeOrThrowTest, SuccessPath_MultipleReferences) {
  bool destroyed = false;
  Microsoft::WRL::ComPtr<TrackedClass> copy;
  {
    auto obj = Dml::SafeMakeOrThrow<TrackedClass>(destroyed);
    copy = obj;
    EXPECT_FALSE(destroyed);
  }
  // Original ComPtr gone, but copy still holds a reference.
  EXPECT_FALSE(destroyed);

  copy.Reset();
  EXPECT_TRUE(destroyed);
}

// Verify that when the constructor throws, the exception propagates
// and sub-objects are properly cleaned up (no leak).
TEST(SafeMakeOrThrowTest, FailurePath_ConstructorThrows) {
  bool witness_destroyed = false;
  EXPECT_THROW(
      Dml::SafeMakeOrThrow<ThrowingClass>(witness_destroyed),
      std::runtime_error);
  // The witness ComPtr member was constructed before the throw.
  // If cleanup worked correctly, the witness should have been destroyed
  // when the ThrowingClass sub-objects were unwound.
  EXPECT_TRUE(witness_destroyed);
}

// Verify that QI works correctly on a SafeMakeOrThrow-created object.
TEST(SafeMakeOrThrowTest, SuccessPath_QueryInterface) {
  auto obj = Dml::SafeMakeOrThrow<SucceedingClass>(42);

  Microsoft::WRL::ComPtr<IUnknown> unk;
  HRESULT hr = obj.As(&unk);
  EXPECT_EQ(hr, S_OK);
  EXPECT_NE(unk.Get(), nullptr);

  Microsoft::WRL::ComPtr<ITestInterface> iface;
  hr = unk.As(&iface);
  EXPECT_EQ(hr, S_OK);
  EXPECT_EQ(iface->GetValue(), 42);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_DML
