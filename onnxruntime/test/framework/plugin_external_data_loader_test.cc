// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "core/framework/allocator.h"
#include "core/framework/external_data_loader_manager.h"
#include "core/framework/plugin_external_data_loader.h"
#include "core/session/onnxruntime_cxx_api.h"

#if defined(__wasm__) && !defined(ORT_NO_EXCEPTIONS)
#include <emscripten.h>
#pragma push_macro("ORT_PLUGIN_EP_SKIP_API_MANUAL_INIT")
#undef ORT_PLUGIN_EP_SKIP_API_MANUAL_INIT
#define ORT_PLUGIN_EP_SKIP_API_MANUAL_INIT
#include "ep/common.h"
#pragma pop_macro("ORT_PLUGIN_EP_SKIP_API_MANUAL_INIT")
#endif

// Exercise the header-only adapters without changing the host test's EP build mode.
#pragma push_macro("EP_ENFORCE")
#undef EP_ENFORCE
#define EP_ENFORCE ORT_ENFORCE
#pragma push_macro("ORT_EP_API_ADAPTER_HEADER_INCLUDED")
#undef ORT_EP_API_ADAPTER_HEADER_INCLUDED
#define ORT_EP_API_ADAPTER_HEADER_INCLUDED
#include "ep/adapter/tensor_helper.h"
#if defined(__wasm__) && !defined(ORT_NO_EXCEPTIONS)
#include "ep/adapter/external_data_loader.h"
#endif
#pragma pop_macro("ORT_EP_API_ADAPTER_HEADER_INCLUDED")
#pragma pop_macro("EP_ENFORCE")

namespace onnxruntime {
namespace test {
namespace {

struct LoaderState {
  int created{0};
  int released{0};
  int loaded{0};
  bool decline{false};
  bool fail_create{false};
  bool fail_query{false};
  bool fail_type_query{false};
  bool fail_load{false};
  bool invalid_read{false};
  bool return_null{false};
  bool missing_callback{false};
};

struct TestLoader : OrtExternalDataLoaderImpl {
  explicit TestLoader(LoaderState& state) : OrtExternalDataLoaderImpl{}, state_(state) {
    ++state_.created;
    ort_version_supported = ORT_API_VERSION;
    Release = [](OrtExternalDataLoaderImpl* self) noexcept { delete static_cast<TestLoader*>(self); };
    CanLoad = [](const OrtExternalDataLoaderImpl* self, const OrtMemoryInfo*, bool* result) noexcept -> OrtStatus* {
      const auto& state = static_cast<const TestLoader*>(self)->state_;
      if (state.fail_query) {
        return OrtApis::CreateStatus(ORT_FAIL, "loader capability failure");
      }
      *result = !state.decline;
      return nullptr;
    };
    SupportsDataType = [](const OrtExternalDataLoaderImpl* self, ONNXTensorElementDataType type,
                          bool* result) noexcept -> OrtStatus* {
      if (static_cast<const TestLoader*>(self)->state_.fail_type_query) {
        return OrtApis::CreateStatus(ORT_FAIL, "loader type capability failure");
      }
      *result = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      return nullptr;
    };
    LoadTensor = [](OrtExternalDataLoaderImpl* self, const OrtExternalDataSource* source,
                    OrtValue* value) noexcept -> OrtStatus* {
      auto& state = static_cast<TestLoader*>(self)->state_;
      ++state.loaded;
      if (state.fail_load) {
        return OrtApis::CreateStatus(ORT_FAIL, "loader load failure");
      }
      void* destination = nullptr;
      ORT_API_RETURN_IF_ERROR(OrtApis::GetTensorMutableData(value, &destination));
      if (source->Read != nullptr) {
        return state.invalid_read
                   ? source->Read(source, source->byte_size, destination, 1)
                   : source->Read(source, 0, destination, source->byte_size);
      }
      return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Test loader requires a native reader.");
    };
    if (state.missing_callback) {
      CanLoad = nullptr;
    }
  }
  ~TestLoader() { ++state_.released; }
  LoaderState& state_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestLoader);
};

struct TestEp : OrtEp {
  explicit TestEp(LoaderState& state) : OrtEp{}, state_(state) {
    ort_version_supported = ORT_API_VERSION;
    CreateExternalDataLoader = [](const OrtEp* self, OrtExternalDataLoaderImpl** result) noexcept -> OrtStatus* {
      auto& state = static_cast<const TestEp*>(self)->state_;
      if (state.fail_create) {
        return OrtApis::CreateStatus(ORT_FAIL, "loader creation failure");
      }
      *result = state.return_null ? nullptr : new TestLoader(state);
      return nullptr;
    };
  }
  LoaderState& state_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestEp);
};

TEST(PluginExternalDataLoader, OptionalAndOlderEpFallback) {
  LoaderState state;
  TestEp ep{state};
  ep.ort_version_supported = 31;
  EXPECT_EQ(plugin_ep::CreateExternalDataLoader(ep), nullptr);
  EXPECT_EQ(state.created, 0);
  ep.ort_version_supported = ORT_API_VERSION;
  const auto create = ep.CreateExternalDataLoader;
  ep.CreateExternalDataLoader = nullptr;
  EXPECT_EQ(plugin_ep::CreateExternalDataLoader(ep), nullptr);
  ep.CreateExternalDataLoader = create;
  state.return_null = true;
  EXPECT_EQ(plugin_ep::CreateExternalDataLoader(ep), nullptr);
  EXPECT_EQ(state.created, 0);
}

TEST(PluginExternalDataLoader, FiltersAndReleases) {
  LoaderState state;
  TestEp ep{state};
  ExternalDataLoaderManager manager;
  ASSERT_TRUE(manager.RegisterExternalDataLoader(plugin_ep::CreateExternalDataLoader(ep)).IsOK());
  const auto& memory_info = CPUAllocator::DefaultInstance()->Info();
  EXPECT_NE(manager.GetExternalDataLoader(memory_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), nullptr);
  EXPECT_EQ(manager.GetExternalDataLoader(memory_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), nullptr);
  state.decline = true;
  EXPECT_EQ(manager.GetExternalDataLoader(memory_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), nullptr);
  manager.Clear();
  EXPECT_EQ(state.created, 1);
  EXPECT_EQ(state.released, 1);
}

TEST(PluginExternalDataLoader, OptionalDataTypeFilter) {
  LoaderState state;
  auto impl = std::make_unique<TestLoader>(state);
  impl->SupportsDataType = nullptr;
  plugin_ep::ExternalDataLoader loader{plugin_ep::OrtExternalDataLoaderImplUniquePtr{impl.release()}};
  EXPECT_TRUE(loader.SupportsDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
}

#ifndef ORT_NO_EXCEPTIONS
TEST(PluginExternalDataLoader, CreationAndCapabilityErrorsPropagate) {
  LoaderState state;
  TestEp ep{state};
  state.fail_create = true;
  EXPECT_THROW(plugin_ep::CreateExternalDataLoader(ep), OnnxRuntimeException);
  state.fail_create = false;
  auto loader = plugin_ep::CreateExternalDataLoader(ep);
  state.fail_query = true;
  EXPECT_THROW(loader->CanLoad(CPUAllocator::DefaultInstance()->Info()), OnnxRuntimeException);
  state.fail_query = false;
  state.fail_type_query = true;
  EXPECT_THROW(loader->SupportsDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), OnnxRuntimeException);
  loader.reset();
  EXPECT_EQ(state.created, state.released);
  ep.CreateExternalDataLoader = [](const OrtEp* self, OrtExternalDataLoaderImpl** result) noexcept -> OrtStatus* {
    *result = new TestLoader(static_cast<const TestEp*>(self)->state_);
    return OrtApis::CreateStatus(ORT_FAIL, "loader creation failure after allocation");
  };
  EXPECT_THROW(plugin_ep::CreateExternalDataLoader(ep), OnnxRuntimeException);
  EXPECT_EQ(state.created, state.released);
}

TEST(PluginExternalDataLoader, InvalidLoaderIsReleased) {
  LoaderState state;
  state.missing_callback = true;
  TestEp ep{state};
  EXPECT_THROW(plugin_ep::CreateExternalDataLoader(ep), OnnxRuntimeException);
  EXPECT_EQ(state.created, 1);
  EXPECT_EQ(state.released, 1);
}
#endif

TEST(PluginExternalDataLoader, TensorAdapterPreservesMemoryTypes) {
  const Ort::MemoryInfo infos[] = {
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput),
      Ort::MemoryInfo{"Pinned", OrtMemoryInfoDeviceType_GPU, 0x10de, 3,
                      OrtDeviceMemoryType_HOST_ACCESSIBLE, 0, OrtDeviceAllocator}};
  std::array<float, 2> data{};
  const int64_t shape[] = {2};
  for (const auto& info : infos) {
    SCOPED_TRACE(info.GetAllocatorName());
    auto value = Ort::Value::CreateTensor<float>(info, data.data(), data.size(), shape, 1);
    auto tensor = ep::adapter::CreateTensorFromApiValue(value);
    EXPECT_EQ(tensor.MutableDataRaw(), data.data());
    EXPECT_EQ(tensor.Location().mem_type, info.GetMemoryType());
    EXPECT_EQ(tensor.Location().device.Type(), static_cast<OrtDevice::DeviceType>(info.GetDeviceType()));
    EXPECT_EQ(tensor.Location().device.MemType(),
              static_cast<OrtDevice::MemoryType>(info.GetDeviceMemoryType()));
    EXPECT_EQ(tensor.Location().device.Vendor(), info.GetVendorId());
    EXPECT_EQ(tensor.Location().device.Id(), info.GetDeviceId());
  }
}

#if defined(__wasm__) && !defined(ORT_NO_EXCEPTIONS)
struct AdapterLoaderState {
  mutable int loads{0};
  mutable OrtMemType memory_type{OrtMemTypeDefault};
  mutable OrtDevice::MemoryType device_memory_type{OrtDevice::MemType::DEFAULT};
};

class WasmTestLoader final : public IExternalDataLoader {
 public:
  explicit WasmTestLoader(AdapterLoaderState& state) : state_(state) {}
  bool CanLoad(const OrtMemoryInfo& info) const override {
    state_.memory_type = info.mem_type;
    state_.device_memory_type = info.device.MemType();
    return true;
  }
  Status LoadTensor(const Env& env, const std::filesystem::path& path, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
    ++state_.loads;
    return LoadWebAssemblyExternalData(env, path, offset, length, ExternalDataLoadType::CPU,
                                       tensor.MutableDataRaw());
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WasmTestLoader);
  AdapterLoaderState& state_;
};

TEST(PluginExternalDataLoader, WasmAdapterPreservesCapabilityMemoryTypes) {
  AdapterLoaderState state;
  ep::adapter::ExternalDataLoader loader{std::make_unique<WasmTestLoader>(state)};
  const Ort::MemoryInfo infos[] = {
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput),
      Ort::MemoryInfo{"Pinned", OrtMemoryInfoDeviceType_GPU, 0x10de, 3,
                      OrtDeviceMemoryType_HOST_ACCESSIBLE, 0, OrtDeviceAllocator}};
  for (const auto& info : infos) {
    bool can_load = false;
    Ort::Status status{loader.CanLoad(&loader, info, &can_load)};
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_TRUE(can_load);
    EXPECT_EQ(state.memory_type, info.GetMemoryType());
    EXPECT_EQ(state.device_memory_type, static_cast<OrtDevice::MemoryType>(info.GetDeviceMemoryType()));
  }
}

TEST(PluginExternalDataLoader, WasmAdapterRejectsOffsetTruncationBeforeLoading) {
  // clang-format off
  EM_ASM(({
    if (!Module.MountedFiles) {
      Module.MountedFiles = new Map();
    }
    Module.MountedFiles.set('loader-test.data', new Uint8Array([0, 0, 0, 0, 0, 160, 63, 0, 0, 32, 192]));
  }));
  // clang-format on
  AdapterLoaderState state;
  ep::adapter::ExternalDataLoader loader{std::make_unique<WasmTestLoader>(state)};
  const auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<float, 2> data{7.0f, 9.0f};
  const int64_t shape[] = {2};
  auto value = Ort::Value::CreateTensor<float>(info, data.data(), data.size(), shape, 1);
  OrtExternalDataSource source{};
  source.file_path = "loader-test.data";
  source.byte_size = sizeof(data);
  const int64_t offsets[] = {-1, (int64_t{1} << 32) + 3, (int64_t{1} << 32) - 8,
                             int64_t{1} << 32, std::numeric_limits<int64_t>::max()};
  for (const auto offset : offsets) {
    SCOPED_TRACE(offset);
    source.file_offset = offset;
    Ort::Status status{loader.LoadTensor(&loader, &source, value)};
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_EQ(state.loads, 0);
    EXPECT_FLOAT_EQ(data[0], 7.0f);
    EXPECT_FLOAT_EQ(data[1], 9.0f);
  }
  source.file_offset = (int64_t{1} << 32) - 9;
  Ort::Status boundary_status{loader.LoadTensor(&loader, &source, value)};
  ASSERT_FALSE(boundary_status.IsOK());
  EXPECT_EQ(boundary_status.GetErrorCode(), ORT_FAIL);
  EXPECT_NE(std::string{boundary_status.GetErrorMessage()}.find("Out of bounds."), std::string::npos);
  EXPECT_EQ(state.loads, 1);
  source.file_offset = 3;
  Ort::Status status{loader.LoadTensor(&loader, &source, value)};
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(state.loads, 2);
  EXPECT_FLOAT_EQ(data[0], 1.25f);
  EXPECT_FLOAT_EQ(data[1], -2.5f);
  EM_ASM({ Module.MountedFiles.delete('loader-test.data'); });
}
#endif

#if !defined(__wasm__)
class MemoryFile final : public RandomAccessFile {
 public:
  MemoryFile() : bytes_("abc") {
    const std::array<float, 2> weights{1.25f, -2.5f};
    bytes_.append(reinterpret_cast<const char*>(weights.data()), sizeof(weights));
  }
  Status GetLength(uint64_t& length) const override {
    length = bytes_.size();
    return Status::OK();
  }
  Status GetCanonicalPath(PathString& path) const override {
    path = ORT_TSTR("already_open.data");
    return Status::OK();
  }
  Status Map(FileOffsetType, size_t, MappedMemoryPtr&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Mapping is not used by this test.");
  }
  Status Read(FileOffsetType offset, gsl::span<char> buffer) const override {
    ++reads;
    ORT_RETURN_IF(fail_read, "source read failure");
    ORT_RETURN_IF(offset < 0 || static_cast<uint64_t>(offset) > bytes_.size() ||
                      buffer.size() > bytes_.size() - static_cast<size_t>(offset),
                  "Invalid test source range.");
    if (!buffer.empty()) {
      std::memcpy(buffer.data(), bytes_.data() + offset, buffer.size());
    }
    return Status::OK();
  }
  mutable int reads{0};
  bool fail_read{false};

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MemoryFile);
  std::string bytes_;
};

TEST(PluginExternalDataLoader, ReadsBorrowedOpenFileIntoBorrowedTensor) {
  LoaderState state;
  TestEp ep{state};
  auto loader = plugin_ep::CreateExternalDataLoader(ep);
  Tensor tensor{DataTypeImpl::GetType<float>(), TensorShape{2}, CPUAllocator::DefaultInstance()};
  MemoryFile file;
  ASSERT_TRUE(loader->LoadTensor(file, 3, SafeInt<size_t>{8}, tensor).IsOK());
  EXPECT_EQ(file.reads, 1);
  EXPECT_EQ(state.loaded, 1);
  loader.reset();
  EXPECT_EQ(state.released, 1);
  EXPECT_FLOAT_EQ(tensor.Data<float>()[0], 1.25f);
  EXPECT_FLOAT_EQ(tensor.Data<float>()[1], -2.5f);
}

TEST(PluginExternalDataLoader, ReaderBoundsAndLoadErrorsPropagate) {
  LoaderState state;
  TestEp ep{state};
  auto loader = plugin_ep::CreateExternalDataLoader(ep);
  Tensor tensor{DataTypeImpl::GetType<float>(), TensorShape{2}, CPUAllocator::DefaultInstance()};
  MemoryFile file;
  EXPECT_FALSE(loader->LoadTensor(file, -1, SafeInt<size_t>{8}, tensor).IsOK());
  EXPECT_FALSE(loader->LoadTensor(file, 3, SafeInt<size_t>{4}, tensor).IsOK());
  EXPECT_EQ(state.loaded, 0);
  EXPECT_EQ(file.reads, 0);
  state.invalid_read = true;
  auto status = loader->LoadTensor(file, 3, SafeInt<size_t>{8}, tensor);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), ORT_INVALID_ARGUMENT);
  EXPECT_EQ(file.reads, 0);
  state.invalid_read = false;
  state.fail_load = true;
  status = loader->LoadTensor(file, 3, SafeInt<size_t>{8}, tensor);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage(), "loader load failure");
  EXPECT_EQ(file.reads, 0);
  state.fail_load = false;
  file.fail_read = true;
  status = loader->LoadTensor(file, 3, SafeInt<size_t>{8}, tensor);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("source read failure"), std::string::npos);
  EXPECT_EQ(file.reads, 1);
  loader.reset();
  EXPECT_EQ(state.released, 1);
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
