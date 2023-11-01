// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "lib/Api.Ort/pch.h"

#include "OnnxruntimeEngine.h"

#include "PheonixSingleton.h"
#include "OnnxruntimeEnvironment.h"
#include "OnnxruntimeEngineBuilder.h"
#include "OnnxruntimeModel.h"
#include "OnnxruntimeSessionBuilder.h"
#include "OnnxruntimeErrors.h"

#include "core/providers/dml/dml_provider_factory.h"

using namespace _winml;

static ONNXTensorElementDataType ONNXTensorElementDataTypeFromTensorKind(winml::TensorKind kind) {
  switch (kind) {
    case winml::TensorKind::Boolean: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    case winml::TensorKind::String: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    case winml::TensorKind::Float16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    case winml::TensorKind::Float: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    case winml::TensorKind::Double: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    case winml::TensorKind::Int8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }
    case winml::TensorKind::Int16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    }
    case winml::TensorKind::Int32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    case winml::TensorKind::Int64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    case winml::TensorKind::UInt8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    case winml::TensorKind::UInt16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }
    case winml::TensorKind::UInt32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    }
    case winml::TensorKind::UInt64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
    case winml::TensorKind::Complex64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    }
    case winml::TensorKind::Complex128: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    }
    default: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
}

OnnxruntimeValue::OnnxruntimeValue() : value_(nullptr, nullptr), allocator_(nullptr, nullptr) {
}

OnnxruntimeValue::~OnnxruntimeValue() {
  value_.reset(nullptr);
  allocator_.reset(nullptr);
}

HRESULT OnnxruntimeValue::RuntimeClassInitialize(
  OnnxruntimeEngine* engine, UniqueOrtValue&& ort_value, UniqueOrtAllocator&& allocator
) {
  engine_ = engine;
  value_ = std::move(ort_value);
  allocator_ = std::move(allocator);

  return S_OK;
}

HRESULT OnnxruntimeValue::IsEmpty(bool* out) {
  *out = UseOrtValue() == nullptr;
  return S_OK;
}

HRESULT OnnxruntimeValue::IsCpu(bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  const OrtMemoryInfo* ort_memory_info;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorMemoryInfo(value_.get(), &ort_memory_info), ort_api);

  const char* name;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->MemoryInfoGetName(ort_memory_info, &name), ort_api);

  OrtMemType type;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->MemoryInfoGetMemType(ort_memory_info, &type), ort_api);

  *out = !strcmp(name, "Cpu") || type == OrtMemType::OrtMemTypeCPUOutput || type == OrtMemType::OrtMemTypeCPUInput;
  return S_OK;
}

static uint64_t ShapeSize(const int64_t* shape, size_t count) {
  // for each dim
  int64_t size = 1;
  for (size_t i = 0; i < count; i++) {
    // find out it's total size
    size *= shape[i];
    // make sure there are no invalid dimensions (-1 or any invalid shape)
    THROW_HR_IF(E_INVALIDARG, shape[i] <= 0);
  }
  return size;
}

static auto GetStrings(
  const OrtApi* ort_api, const OrtValue* ort_value, OrtTensorTypeAndShapeInfo* type_and_shape_info
) {
  std::vector<std::string> out;

  size_t size;
  THROW_IF_NOT_OK_MSG(ort_api->GetDimensionsCount(type_and_shape_info, &size), ort_api);

  std::vector<int64_t> shape(size);

  if (size > 0) {
    THROW_IF_NOT_OK_MSG(ort_api->GetDimensions(type_and_shape_info, &shape[0], size), ort_api);
  }
  auto length = ShapeSize(shape.data(), shape.size());

  // make a big buffer to hold all the string data
  size_t buffer_length;
  THROW_IF_NOT_OK_MSG(ort_api->GetStringTensorDataLength(ort_value, &buffer_length), ort_api);

  std::vector<std::string_view> strings;
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_length]);
  std::vector<size_t> offsets(static_cast<size_t>(length));

  THROW_IF_NOT_OK_MSG(
    ort_api->GetStringTensorContent(ort_value, buffer.get(), buffer_length, offsets.data(), offsets.size()), ort_api
  );

  // now go build all the strings
  for (size_t i = 0; i < length; ++i) {
    size_t str_len = 0;
    // are we on the last one?
    if (i == (length - 1)) {
      str_len = buffer_length - offsets[i];
    } else {
      str_len = offsets[i + 1] - offsets[i];
    }
    strings.push_back(std::string_view(reinterpret_cast<const char*>(buffer.get() + offsets[i]), str_len));
  }

  return std::make_shared<std::pair<decltype(strings), decltype(buffer)>>(std::move(strings), std::move(buffer));
}

HRESULT OnnxruntimeValue::GetResource(_winml::Resource& out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  void* mutable_data = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorMutableData(value_.get(), &mutable_data), ort_api);

  const OrtMemoryInfo* ort_memory_info;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorMemoryInfo(value_.get(), &ort_memory_info), ort_api);

  bool is_cpu = false;
  if (SUCCEEDED(IsCpu(&is_cpu)) && !is_cpu) {
    const OrtDmlApi* ort_dml_api;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)), ort_api
    );

    OrtAllocator* ort_allocator;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->CreateAllocator(engine_->UseOrtSession(), ort_memory_info, &ort_allocator), ort_api
    );
    auto allocator = UniqueOrtAllocator(ort_allocator, ort_api->ReleaseAllocator);

    winrt::com_ptr<ID3D12Resource> resource;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_dml_api->GetD3D12ResourceFromAllocation(allocator.get(), mutable_data, resource.put()), ort_api
    );
    out = _winml::Resource(resource.get(), [](void*) { /*do nothing, as this pointer is actually a com pointer! */ });
  } else {
    int is_tensor;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->IsTensor(value_.get(), &is_tensor), ort_api);
    if (is_tensor == 0) {
      out = _winml::Resource(
        mutable_data, [](void*) { /*do nothing, as this pointer is actually owned elsewhere in ORT! */ }
      );
      return S_OK;
    }

    OrtTensorTypeAndShapeInfo* info = nullptr;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorTypeAndShape(value_.get(), &info), ort_api);
    auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

    ONNXTensorElementDataType data_type;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorElementType(type_and_shape_info.get(), &data_type), ort_api);

    if (data_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      auto strings = GetStrings(ort_api, value_.get(), info);
      auto string_data = strings->first.data();
      out = _winml::Resource(string_data, [capture_strings = strings](void*) {
        /*This deleter does nothing but capture the strings, which extends the lifetime of the returned strings.*/
      });
    } else {
      out = _winml::Resource(
        mutable_data, [](void*) { /*do nothing, as this pointer is actually owned elsewhere in ORT! */ }
      );
    }
  }
  return S_OK;
}

HRESULT OnnxruntimeValue::IsTensor(bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  ONNXType type = ONNXType::ONNX_TYPE_UNKNOWN;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetValueType(value_.get(), &type), ort_api);
  *out = type == ONNXType::ONNX_TYPE_TENSOR;
  return S_OK;
}

HRESULT OnnxruntimeValue::IsOfTensorType(winml::TensorKind kind, bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  OrtTensorTypeAndShapeInfo* info = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorTypeAndShape(value_.get(), &info), ort_api);
  auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

  ONNXTensorElementDataType data_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorElementType(type_and_shape_info.get(), &data_type), ort_api);

  *out = data_type == ONNXTensorElementDataTypeFromTensorKind(kind);
  return S_OK;
}

HRESULT OnnxruntimeValue::GetTensorShape(std::vector<int64_t>& shape_vector) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();
  OrtTensorTypeAndShapeInfo* info = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorTypeAndShape(value_.get(), &info), ort_api);
  auto type_and_shape_info = UniqueOrtTensorTypeAndShapeInfo(info, ort_api->ReleaseTensorTypeAndShapeInfo);

  size_t size;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetDimensionsCount(type_and_shape_info.get(), &size), ort_api);

  std::vector<int64_t> shape(size);
  if (size > 0) {
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetDimensions(type_and_shape_info.get(), &shape[0], size), ort_api);
  }

  shape_vector = std::move(shape);
  return S_OK;
}

static bool EnsureMapTypeInfo(
  OnnxruntimeEngine* engine, OrtTypeInfo* type_info, winml::TensorKind key_kind, winml::TensorKind value_kind
) {
  auto ort_api = engine->GetEngineFactory()->UseOrtApi();

  const OrtMapTypeInfo* map_info;
  THROW_IF_NOT_OK_MSG(ort_api->CastTypeInfoToMapTypeInfo(type_info, &map_info), ort_api);

  if (map_info == nullptr) {
    // It must be a seq<tensor<*>> type
    return false;
  }

  ONNXTensorElementDataType map_key_type;
  THROW_IF_NOT_OK_MSG(ort_api->GetMapKeyType(map_info, &map_key_type), ort_api);

  if (map_key_type == ONNXTensorElementDataTypeFromTensorKind(key_kind)) {
    OrtTypeInfo* value_info;
    THROW_IF_NOT_OK_MSG(ort_api->GetMapValueType(map_info, &value_info), ort_api);
    auto map_value_info = UniqueOrtTypeInfo(value_info, ort_api->ReleaseTypeInfo);

    const OrtTensorTypeAndShapeInfo* value_tensor_info = nullptr;
    THROW_IF_NOT_OK_MSG(ort_api->CastTypeInfoToTensorInfo(map_value_info.get(), &value_tensor_info), ort_api);

    if (value_tensor_info) {
      ONNXTensorElementDataType map_value_tensor_type;
      THROW_IF_NOT_OK_MSG(ort_api->GetTensorElementType(value_tensor_info, &map_value_tensor_type), ort_api);

      if (map_value_tensor_type == ONNXTensorElementDataTypeFromTensorKind(value_kind)) {
        size_t num_dims;
        THROW_IF_NOT_OK_MSG(ort_api->GetDimensionsCount(value_tensor_info, &num_dims), ort_api);

        return num_dims == 0;
      }
    }
  }
  return false;
}

HRESULT OnnxruntimeValue::IsOfMapType(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  OrtTypeInfo* info = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTypeInfo(value_.get(), &info), ort_api);
  auto unique_type_info = UniqueOrtTypeInfo(info, ort_api->ReleaseTypeInfo);

  ONNXType type;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetOnnxTypeFromTypeInfo(unique_type_info.get(), &type), ort_api);

  if (type == ONNXType::ONNX_TYPE_MAP) {
    *out = EnsureMapTypeInfo(engine_.Get(), unique_type_info.get(), key_kind, value_kind);
  }

  *out = false;

  return S_OK;
}

HRESULT OnnxruntimeValue::IsOfVectorMapType(winml::TensorKind key_kind, winml::TensorKind value_kind, bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  OrtTypeInfo* info = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTypeInfo(value_.get(), &info), ort_api);
  auto unique_type_info = UniqueOrtTypeInfo(info, ort_api->ReleaseTypeInfo);

  ONNXType type;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetOnnxTypeFromTypeInfo(unique_type_info.get(), &type), ort_api);

  if (type == ONNXType::ONNX_TYPE_SEQUENCE) {
    const OrtSequenceTypeInfo* sequence_info;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->CastTypeInfoToSequenceTypeInfo(unique_type_info.get(), &sequence_info), ort_api);

    OrtTypeInfo* element_info;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetSequenceElementType(sequence_info, &element_info), ort_api);
    auto unique_element_info = UniqueOrtTypeInfo(element_info, ort_api->ReleaseTypeInfo);

    *out = EnsureMapTypeInfo(engine_.Get(), unique_element_info.get(), key_kind, value_kind);
  }
  return S_OK;
}

HRESULT OnnxruntimeValue::IsOfVectorTensorType(winml::TensorKind kind, bool* out) {
  auto ort_api = engine_->GetEngineFactory()->UseOrtApi();

  *out = false;

  OrtTypeInfo* info = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTypeInfo(value_.get(), &info), ort_api);
  auto unique_type_info = UniqueOrtTypeInfo(info, ort_api->ReleaseTypeInfo);

  ONNXType type;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetOnnxTypeFromTypeInfo(unique_type_info.get(), &type), ort_api);

  if (type == ONNXType::ONNX_TYPE_SEQUENCE) {
    const OrtSequenceTypeInfo* sequence_info;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->CastTypeInfoToSequenceTypeInfo(unique_type_info.get(), &sequence_info), ort_api);

    OrtTypeInfo* element_info;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetSequenceElementType(sequence_info, &element_info), ort_api);
    auto unique_element_info = UniqueOrtTypeInfo(element_info, ort_api->ReleaseTypeInfo);

    ONNXType element_type;
    RETURN_HR_IF_NOT_OK_MSG(ort_api->GetOnnxTypeFromTypeInfo(unique_element_info.get(), &element_type), ort_api);

    if (element_type == ONNXType::ONNX_TYPE_TENSOR) {
      const OrtTensorTypeAndShapeInfo* element_tensor_info = nullptr;
      RETURN_HR_IF_NOT_OK_MSG(
        ort_api->CastTypeInfoToTensorInfo(unique_element_info.get(), &element_tensor_info), ort_api
      );

      if (element_tensor_info) {
        ONNXTensorElementDataType element_tensor_type;
        RETURN_HR_IF_NOT_OK_MSG(ort_api->GetTensorElementType(element_tensor_info, &element_tensor_type), ort_api);
        *out = element_tensor_type == ONNXTensorElementDataTypeFromTensorKind(kind);
      }
    }
  }
  return S_OK;
}

HRESULT OnnxruntimeValue::SetParameter(IUnknown* param) {
  param_ = param;
  return S_OK;
}

OrtValue* OnnxruntimeValue::UseOrtValue() {
  return value_.get();
}

HRESULT OnnxruntimeValue::AssignOrtValue(OrtValue* in) {
  value_.reset(in);
  return S_OK;
}

OnnxruntimeEngine::OnnxruntimeEngine() : session_(nullptr, nullptr) {
}

HRESULT OnnxruntimeEngine::RuntimeClassInitialize(
  OnnxruntimeEngineFactory* engine_factory, UniqueOrtSession&& session, IOrtSessionBuilder* session_builder
) {
  engine_factory_ = engine_factory;
  session_ = std::move(session);
  session_builder_ = session_builder;
  return S_OK;
}

OnnxruntimeEngine::~OnnxruntimeEngine() {
  for (auto& handle : custom_op_library_handles_) {
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
  }
}

HRESULT OnnxruntimeEngine::RegisterCustomOpLibraryHandles(const gsl::span<void*> handles) {
  custom_op_library_handles_.insert(custom_op_library_handles_.end(), handles.begin(), handles.end());
  return S_OK;
}

HRESULT OnnxruntimeEngine::LoadModel(_In_ IModel* model) {
  Microsoft::WRL::ComPtr<IOnnxruntimeModel> onnxruntime_model;
  RETURN_IF_FAILED(model->QueryInterface(IID_PPV_ARGS(&onnxruntime_model)));

  OrtModel* ort_model;
  RETURN_IF_FAILED(onnxruntime_model->DetachOrtModel(&ort_model));

  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionLoadAndPurloinModel(session_.get(), ort_model), engine_factory_->UseOrtApi()
  );
  return S_OK;
}

HRESULT OnnxruntimeEngine::Initialize() {
  RETURN_IF_FAILED(session_builder_->Initialize(session_.get()));
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterGraphTransformers() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionRegisterGraphTransformers(session_.get()), engine_factory_->UseOrtApi()
  );
  return S_OK;
}

HRESULT OnnxruntimeEngine::RegisterCustomRegistry(IMLOperatorRegistry* registry) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionRegisterCustomRegistry(session_.get(), registry), engine_factory_->UseOrtApi()
  );
  return S_OK;
}

HRESULT OnnxruntimeEngine::EndProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SessionEndProfiling(session_.get()), engine_factory_->UseOrtApi());
  return S_OK;
}

HRESULT OnnxruntimeEngine::StartProfiling() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  engine_factory_->GetOrtEnvironment(&ort_env);

  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionStartProfiling(ort_env, session_.get()), engine_factory_->UseOrtApi()
  );
  return S_OK;
}

HRESULT OnnxruntimeEngine::FlushContext() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider), engine_factory_->UseOrtApi()
  );

  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider), engine_factory_->UseOrtApi()
  );
  return S_OK;
}

HRESULT OnnxruntimeEngine::ReleaseCompletedReferences() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider), engine_factory_->UseOrtApi()
  );

  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->DmlExecutionProviderReleaseCompletedReferences(ort_provider), engine_factory_->UseOrtApi()
  );

  return S_OK;
}

HRESULT OnnxruntimeEngine::CopyValueAcrossDevices(IValue* src, IValue* dest) {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider), engine_factory_->UseOrtApi()
  );

  auto src_value = static_cast<OnnxruntimeValue*>(src);
  auto dest_value = static_cast<OnnxruntimeValue*>(dest);

  bool is_empty;
  auto has_null_source = (SUCCEEDED(src_value->IsEmpty(&is_empty)) && is_empty);
  RETURN_HR_IF(E_FAIL, has_null_source);

  auto has_null_dest = (SUCCEEDED(dest_value->IsEmpty(&is_empty)) && is_empty);
  RETURN_HR_IF(E_FAIL, has_null_dest);

  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->DmlCopyTensor(ort_provider, src_value->UseOrtValue(), dest_value->UseOrtValue()),
    engine_factory_->UseOrtApi()
  );

  return S_OK;
}

HRESULT OnnxruntimeEngine::Sync() {
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider), engine_factory_->UseOrtApi()
  );

  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ExecutionProviderSync(ort_provider), engine_factory_->UseOrtApi());

  return S_OK;
}

OrtSession* OnnxruntimeEngine::UseOrtSession() {
  return session_.get();
}

const OrtApi* OnnxruntimeEngine::UseOrtApi() {
  return engine_factory_->UseOrtApi();
}

OnnxruntimeEngineFactory* OnnxruntimeEngine::GetEngineFactory() {
  return engine_factory_.Get();
}

HRESULT OnnxruntimeEngine::CreateTensorValueFromDefaultAllocator(
  const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out
) {
  *out = nullptr;
  auto ort_api = engine_factory_->UseOrtApi();

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->GetAllocatorWithDefaultOptions(&ort_allocator),
    ort_api
  );  // This should not be freed as this owned by ort

  OrtValue* ort_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateTensorAsOrtValue(
      ort_allocator, shape, count, ONNXTensorElementDataTypeFromTensorKind(kind), &ort_value
    ),
    ort_api
  );
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

/*
* OnnxruntimeEngine::CreateTensorValue
*
* Used by callers like ImageFeatureValue to allocate a cpu or gpu OrtValue with ORT owned memory.
* In the image feature value case, tensorization creates temporary buffers, and will need to copy the value from
* its source location to the ort value. Since a copy is required, there is need to preserve the caller's memory locations.
* We simply allocate memory with ORT and copy the tensorized values into it.
*/
HRESULT
OnnxruntimeEngine::CreateTensorValue(const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetExecutionProvider(session_.get(), 0, &ort_provider), engine_factory_->UseOrtApi()
  );

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api->GetProviderAllocator(session_.get(), ort_provider, &ort_allocator), engine_factory_->UseOrtApi()
  );

  auto unique_allocator = UniqueOrtAllocator(ort_allocator, [](OrtAllocator* allocator) {
    GetVersionedWinmlAdapterApi()->FreeProviderAllocator(allocator);
  });  // the release here should probably not return anything

  OrtValue* ort_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateTensorAsOrtValue(
      unique_allocator.get(), shape, count, ONNXTensorElementDataTypeFromTensorKind(kind), &ort_value
    ),
    ort_api
  );
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);
  RETURN_IF_FAILED(
    Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(out, this, std::move(unique_value), std::move(unique_allocator))
  );
  return S_OK;
}

using DmlAllocatorResource = std::unique_ptr<void, void (*)(void*)>;
class DmlAllocatorWrapper
  : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown> {
 public:
  DmlAllocatorWrapper() : dml_resource_(nullptr, nullptr) {}

  HRESULT RuntimeClassInitialize(DmlAllocatorResource&& dml_resource) {
    dml_resource_ = std::move(dml_resource);
    return S_OK;
  }

 private:
  DmlAllocatorResource dml_resource_;
};

/*
* OnnxruntimeEngine::CreateTensorValueFromExternalD3DResource
*
* Used by callers like TensorBase to allocate a gpu OrtValue based on a called owned ID3D12Resource.
* WinML cannot use ORT allocators here since they will allocate the ID3D12Resource and force a copy from the user provided value.
*/
HRESULT OnnxruntimeEngine::CreateTensorValueFromExternalD3DResource(
  ID3D12Resource* d3d_resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out
) {
  auto ort_api = engine_factory_->UseOrtApi();
  const OrtDmlApi* ort_dml_api;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)), ort_api
  );

  OrtMemoryInfo* ort_memory_info;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateMemoryInfo(
      "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &ort_memory_info
    ),
    ort_api
  );

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->CreateAllocator(session_.get(), ort_memory_info, &ort_allocator), ort_api);
  auto allocator = UniqueOrtAllocator(ort_allocator, ort_api->ReleaseAllocator);

  void* dml_allocator_resource;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_dml_api->CreateGPUAllocationFromD3DResource(d3d_resource, &dml_allocator_resource), engine_factory_->UseOrtApi()
  );

  auto unique_dml_allocator_resource = DmlAllocatorResource(dml_allocator_resource, [](void* ptr) {
    const OrtDmlApi* ort_dml_api;
    GetVersionedOrtApi()->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api));
    ort_dml_api->FreeGPUAllocation(ptr);
  });

  // create the OrtValue as a tensor letting ort know that we own the data buffer
  OrtValue* ort_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateTensorWithDataAsOrtValue(
      ort_memory_info,
      unique_dml_allocator_resource.get(),
      static_cast<size_t>(d3d_resource->GetDesc().Width),
      shape,
      count,
      ONNXTensorElementDataTypeFromTensorKind(kind),
      &ort_value
    ),
    ort_api
  );
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);

  Microsoft::WRL::ComPtr<OnnxruntimeValue> out_value;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    &out_value, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)
  ));

  // Cache the allocator on the value so it destructs appropriately when the value is dropped
  Microsoft::WRL::ComPtr<DmlAllocatorWrapper> dml_allocator_resource_wrapper;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<DmlAllocatorWrapper>(
    &dml_allocator_resource_wrapper, std::move(unique_dml_allocator_resource)
  ));

  RETURN_IF_FAILED(out_value->SetParameter(dml_allocator_resource_wrapper.Get()));

  *out = out_value.Detach();

  return S_OK;
}

/*
* OnnxruntimeEngine::CreateStringTensorValueFromDataWithCopy
*
* Used by callers like TensorString to allocate a cpu OrtValue and populate the contents with use specified data.
* WinML cannot use CreateTensorWithDataAsOrtValue since externally allocated strings are not supported on the c-abi.
* The c-abi string implementation requires a copy the external buffer into its own internal std::string copy.
* In addition, strings have different APIs on the c-abi like FillStringTensor to populate the buffer, and so strings
* have a different calling pattern than other Tensor<T> types of simple data types.
*/
HRESULT OnnxruntimeEngine::CreateStringTensorValueFromDataWithCopy(
  const char* const* data, size_t num_elements, const int64_t* shape, size_t count, _Out_ IValue** out
) {
  auto ort_api = engine_factory_->UseOrtApi();

  RETURN_IF_FAILED(CreateTensorValueFromDefaultAllocator(shape, count, winml::TensorKind::String, out));

  auto ort_value = reinterpret_cast<_winml::OnnxruntimeValue*>(*out)->UseOrtValue();
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->FillStringTensor(ort_value, reinterpret_cast<const char* const*>(data), num_elements), ort_api
  );
  return S_OK;
}

/*
* OnnxruntimeEngine::CreateTensorValueFromExternalBuffer
*
* Used by callers like TensorBase<T> to allocate a cpu OrtValue that is backed by caller owned memory.
*/
HRESULT OnnxruntimeEngine::CreateTensorValueFromExternalBuffer(
  void* data, size_t size_in_bytes, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out
) {
  auto ort_api = engine_factory_->UseOrtApi();

  if (kind == winml::TensorKind::String) {
    // String buffers cannot be passed into the ort api directly because ort c-api tensor strings cannot be backed by external memory
    return E_NOTIMPL;
  }

  // TODO:  what is the difference between the device allocator and the arena allocator?
  OrtMemoryInfo* cpu_memory;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory), ort_api);

  OrtValue* ort_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateTensorWithDataAsOrtValue(
      cpu_memory, data, size_in_bytes, shape, count, ONNXTensorElementDataTypeFromTensorKind(kind), &ort_value
    ),
    ort_api
  );
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);

  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

HRESULT OnnxruntimeEngine::CreateSequenceOfValuesValue(IValue** values, size_t size, IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();

  std::vector<OrtValue*> sequence(size);
  std::transform(values, values + size, std::begin(sequence), [](auto value) {
    return static_cast<OnnxruntimeValue*>(value)->UseOrtValue();
  });

  OrtValue* ort_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateValue(sequence.data(), size, ONNXType::ONNX_TYPE_SEQUENCE, &ort_value), ort_api
  );

  UniqueOrtValue unique_value(ort_value, ort_api->ReleaseValue);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

/*
* OnnxruntimeEngine::CreateNullValue
*
* Used by callers like TensorBase<T> and the binding object to allocate a cpu OrtValue that is empty.
* This is used for WinML unbound outputs.
*/
HRESULT OnnxruntimeEngine::CreateNullValue(_Out_ IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto unique_value = UniqueOrtValue(nullptr, ort_api->ReleaseValue);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

template <typename TAbiType>
struct AbiTypeInfo {
  using CppWinRTType = TAbiType;
  using OrtType = TAbiType;
  using ResourceType = TAbiType;
};

template <>
struct AbiTypeInfo<HSTRING> {
  using CppWinRTType = winrt::hstring;
  using OrtType = const char*;
  using ResourceType = std::string_view;
};

template <typename TCppwinrtType>
typename auto CppwinrtTypeToOrtType(TCppwinrtType raw) {
  return raw;
}

template <>
typename auto CppwinrtTypeToOrtType<winrt::hstring>(winrt::hstring raw) {
  return _winml::Strings::UTF8FromHString(raw);
}

template <typename TAbiType>
typename auto ResourceTypeToCppwinrtType(typename AbiTypeInfo<TAbiType>::ResourceType value) {
  return value;
}

template <>
typename auto ResourceTypeToCppwinrtType<HSTRING>(typename AbiTypeInfo<HSTRING>::ResourceType value) {
  return _winml::Strings::HStringFromUTF8(value.data(), value.size());
}

template <typename TAbiKey, typename TAbiValue>
auto CastToWinrtMap(IInspectable* map_insp) {
  using cppwinrt_key_type = typename AbiTypeInfo<TAbiKey>::CppWinRTType;
  using cppwinrt_value_type = typename AbiTypeInfo<TAbiValue>::CppWinRTType;

  wf::IInspectable map_inspectable;
  wfc::IMap<cppwinrt_key_type, cppwinrt_value_type> map;
  winrt::copy_from_abi(map_inspectable, map_insp);
  map_inspectable.as(map);
  return map;
}

template <typename TAbiKey, typename TAbiValue>
auto CastToWinrtSequenceOfMaps(IInspectable* sequence_insp) {
  using cppwinrt_key_type = typename AbiTypeInfo<TAbiKey>::CppWinRTType;
  using cppwinrt_value_type = typename AbiTypeInfo<TAbiValue>::CppWinRTType;

  using cppwinrt_element_map_type = wfc::IMap<cppwinrt_key_type, cppwinrt_value_type>;
  using cppwinrt_sequence_type = wfc::IVector<cppwinrt_element_map_type>;
  cppwinrt_sequence_type sequence;
  wf::IInspectable sequence_inspectable;
  winrt::copy_from_abi(sequence_inspectable, sequence_insp);
  sequence_inspectable.as(sequence);
  return sequence;
}

template <typename TAbiKey, typename TAbiValue>
struct FillMapTensors {
  static HRESULT Run(
    const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value
  ) {
    typename AbiTypeInfo<TAbiKey>::OrtType* keys_mutable_data;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetTensorMutableData(keys_ort_value, reinterpret_cast<void**>(&keys_mutable_data)), ort_api
    );

    typename AbiTypeInfo<TAbiValue>::OrtType* values_mutable_data;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetTensorMutableData(values_ort_value, reinterpret_cast<void**>(&values_mutable_data)), ort_api
    );

    auto map = CastToWinrtMap<TAbiKey, TAbiValue>(map_insp);
    size_t index = 0;
    for (const auto& pair : map) {
      keys_mutable_data[index] = CppwinrtTypeToOrtType(pair.Key());
      values_mutable_data[index] = CppwinrtTypeToOrtType(pair.Value());
      index++;
    }
    return S_OK;
  }
};

template <typename TAbiValue>
struct FillMapTensors<HSTRING, TAbiValue> {
  static HRESULT Run(
    const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value
  ) {
    typename AbiTypeInfo<TAbiValue>::OrtType* values_mutable_data;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetTensorMutableData(values_ort_value, reinterpret_cast<void**>(&values_mutable_data)), ort_api
    );

    auto map = CastToWinrtMap<HSTRING, TAbiValue>(map_insp);
    size_t index = 0;
    std::vector<std::string> keys;
    for (const auto& pair : map) {
      keys.push_back(CppwinrtTypeToOrtType(pair.Key()));
      values_mutable_data[index] = CppwinrtTypeToOrtType(pair.Value());
      index++;
    }

    std::vector<const char*> raw_values;
    std::transform(keys.begin(), keys.end(), std::back_inserter(raw_values), [&](auto& str) { return str.c_str(); });

    RETURN_HR_IF_NOT_OK_MSG(ort_api->FillStringTensor(keys_ort_value, raw_values.data(), raw_values.size()), ort_api);

    return S_OK;
  }
};

template <typename TAbiKey>
struct FillMapTensors<TAbiKey, HSTRING> {
  static HRESULT Run(
    const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value
  ) {
    typename AbiTypeInfo<TAbiKey>::OrtType* keys_mutable_data;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetTensorMutableData(keys_ort_value, reinterpret_cast<void**>(&keys_mutable_data)), ort_api
    );

    auto map = CastToWinrtMap<TAbiKey, HSTRING>(map_insp);
    size_t index = 0;
    std::vector<std::string> values;
    for (const auto& pair : map) {
      keys_mutable_data[index] = CppwinrtTypeToOrtType(pair.Key());
      values.push_back(CppwinrtTypeToOrtType(pair.Value()));
      index++;
    }

    std::vector<const char*> raw_values;
    std::transform(values.begin(), values.end(), std::back_inserter(raw_values), [&](auto& str) {
      return str.c_str();
    });

    RETURN_HR_IF_NOT_OK_MSG(ort_api->FillStringTensor(keys_ort_value, raw_values.data(), raw_values.size()), ort_api);
    return S_OK;
  }
};

template <>
struct FillMapTensors<HSTRING, HSTRING> {
  static HRESULT Run(
    const OrtApi* ort_api, IInspectable* map_insp, OrtValue* keys_ort_value, OrtValue* values_ort_value
  ) {
    auto map = CastToWinrtMap<HSTRING, HSTRING>(map_insp);
    std::vector<std::string> keys;
    std::vector<std::string> values;
    for (const auto& pair : map) {
      keys.push_back(CppwinrtTypeToOrtType(pair.Key()));
      values.push_back(CppwinrtTypeToOrtType(pair.Value()));
    }

    std::vector<const char*> raw_keys;
    std::transform(keys.begin(), keys.end(), std::back_inserter(raw_keys), [&](auto& str) { return str.c_str(); });

    std::vector<const char*> raw_values;
    std::transform(values.begin(), values.end(), std::back_inserter(raw_values), [&](auto& str) {
      return str.c_str();
    });

    RETURN_HR_IF_NOT_OK_MSG(ort_api->FillStringTensor(keys_ort_value, raw_keys.data(), raw_keys.size()), ort_api);
    RETURN_HR_IF_NOT_OK_MSG(ort_api->FillStringTensor(values_ort_value, raw_values.data(), raw_values.size()), ort_api);
    return S_OK;
  }
};

template <typename TAbiKey, typename TAbiValue>
HRESULT CreateMapValue(
  OnnxruntimeEngine* engine,
  IInspectable* map_insp,
  winml::TensorKind key_kind,
  winml::TensorKind value_kind,
  _Out_ IValue** out
) {
  auto ort_api = engine->UseOrtApi();
  auto map = CastToWinrtMap<TAbiKey, TAbiValue>(map_insp);
  std::vector<int64_t> shape = {static_cast<int64_t>(map.Size())};

  winrt::com_ptr<_winml::IValue> key_value;
  RETURN_IF_FAILED(engine->CreateTensorValueFromDefaultAllocator(shape.data(), shape.size(), key_kind, key_value.put())
  );
  auto keys_ort_value = static_cast<OnnxruntimeValue*>(key_value.get())->UseOrtValue();

  winrt::com_ptr<_winml::IValue> value_value;
  RETURN_IF_FAILED(
    engine->CreateTensorValueFromDefaultAllocator(shape.data(), shape.size(), value_kind, value_value.put())
  );
  auto values_ort_value = static_cast<OnnxruntimeValue*>(value_value.get())->UseOrtValue();

  auto hr = FillMapTensors<TAbiKey, TAbiValue>::Run(ort_api, map_insp, keys_ort_value, values_ort_value);
  RETURN_IF_FAILED(hr);

  OrtValue* inputs[2] = {keys_ort_value, values_ort_value};

  OrtValue* map_value;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->CreateValue(inputs, 2, ONNXType::ONNX_TYPE_MAP, &map_value), ort_api);
  auto unique_map_ort_value = UniqueOrtValue(map_value, ort_api->ReleaseValue);

  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, engine, std::move(unique_map_ort_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

static auto GetMapValueCreator(OnnxruntimeEngine* engine, winml::TensorKind key_kind, winml::TensorKind value_kind) {
  using namespace std::placeholders;
  if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Int64) {
    return std::bind(
      &CreateMapValue<int64_t, int64_t>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Int64, _2
    );
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    return std::bind(
      &CreateMapValue<int64_t, float>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Float, _2
    );
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Double) {
    return std::bind(
      &CreateMapValue<int64_t, double>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Double, _2
    );
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::String) {
    return std::bind(
      &CreateMapValue<int64_t, HSTRING>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::String, _2
    );
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Int64) {
    return std::bind(
      &CreateMapValue<HSTRING, int64_t>, engine, _1, winml::TensorKind::String, winml::TensorKind::Int64, _2
    );
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    return std::bind(
      &CreateMapValue<HSTRING, float>, engine, _1, winml::TensorKind::String, winml::TensorKind::Float, _2
    );
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Double) {
    return std::bind(
      &CreateMapValue<HSTRING, double>, engine, _1, winml::TensorKind::String, winml::TensorKind::Double, _2
    );
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::String) {
    return std::bind(
      &CreateMapValue<HSTRING, HSTRING>, engine, _1, winml::TensorKind::String, winml::TensorKind::String, _2
    );
  }

  THROW_HR(E_NOTIMPL);
}

HRESULT OnnxruntimeEngine::CreateMapValue(
  IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out
) {
  return GetMapValueCreator(this, key_kind, value_kind)(map, out);
}

template <typename TAbiKey, typename TAbiValue>
HRESULT CreateSequenceOfMapsValue(
  OnnxruntimeEngine* engine,
  IInspectable* sequence_insp,
  winml::TensorKind key_kind,
  winml::TensorKind value_kind,
  _Out_ IValue** out
) {
  auto ort_api = engine->UseOrtApi();
  auto sequence = CastToWinrtSequenceOfMaps<TAbiKey, TAbiValue>(sequence_insp);

  std::vector<winrt::com_ptr<_winml::IValue>> element_values;
  for (auto element : sequence) {
    winrt::com_ptr<_winml::IValue> element_value;
    engine->CreateMapValue(
      reinterpret_cast<IInspectable*>(winrt::get_abi(element)), key_kind, value_kind, element_value.put()
    );
    element_values.push_back(element_value);
  }

  std::vector<OrtValue*> element_ort_values;
  std::transform(element_values.begin(), element_values.end(), std::back_inserter(element_ort_values), [](auto value) {
    return static_cast<OnnxruntimeValue*>(value.get())->UseOrtValue();
  });

  OrtValue* sequence_value;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->CreateValue(
      element_ort_values.data(), element_ort_values.size(), ONNXType::ONNX_TYPE_SEQUENCE, &sequence_value
    ),
    ort_api
  );
  auto unique_sequence_ort_value = UniqueOrtValue(sequence_value, ort_api->ReleaseValue);

  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    out, engine, std::move(unique_sequence_ort_value), UniqueOrtAllocator(nullptr, nullptr)
  ));
  return S_OK;
}

static auto GetSequenceOfMapsValueCreator(
  OnnxruntimeEngine* engine, winml::TensorKind key_kind, winml::TensorKind value_kind
) {
  using namespace std::placeholders;
  if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    return std::bind(
      &CreateSequenceOfMapsValue<HSTRING, float>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Int64, _2
    );
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    return std::bind(
      &CreateSequenceOfMapsValue<int64_t, float>, engine, _1, winml::TensorKind::Int64, winml::TensorKind::Float, _2
    );
  }

  THROW_HR(E_NOTIMPL);
}

HRESULT OnnxruntimeEngine::CreateSequenceOfMapsValue(
  IInspectable* sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, _Out_ IValue** out
) {
  RETURN_IF_FAILED(GetSequenceOfMapsValueCreator(this, key_kind, value_kind)(sequence, out));
  return S_OK;
}

template <typename TAbiKey, typename TAbiValue>
static HRESULT FillAbiSequence(IInspectable* sequence_insp, std::vector<wf::IInspectable>& elements) {
  using cppwinrt_key_type = typename AbiTypeInfo<TAbiKey>::CppWinRTType;
  using cppwinrt_value_type = typename AbiTypeInfo<TAbiValue>::CppWinRTType;
  auto sequence = CastToWinrtSequenceOfMaps<TAbiKey, TAbiValue>(sequence_insp);
  for (auto element : elements) {
    wfc::IMap<cppwinrt_key_type, cppwinrt_value_type> map_element;
    element.as(map_element);
    sequence.Append(map_element);
  }
  return S_OK;
}

static auto GetAbiSequenceFiller(winml::TensorKind key_kind, winml::TensorKind value_kind) {
  using namespace std::placeholders;
  if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    return &FillAbiSequence<winrt::hstring, float>;
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    return &FillAbiSequence<int64_t, float>;
  }
  THROW_HR(E_NOTIMPL);
}

static wf::IInspectable CreateMap(winml::TensorKind key_kind, winml::TensorKind value_kind) {
  wf::IInspectable map_insp;
  if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    auto map = winrt::single_threaded_map<winrt::hstring, float>();
    map.as(map_insp);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    auto map = winrt::single_threaded_map<int64_t, float>();
    map.as(map_insp);
  }

  return map_insp;
}

HRESULT OnnxruntimeEngine::FillSequenceOfMapsValue(
  IInspectable* sequence, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* sequence_value
) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto onnxruntime_squence_value = static_cast<OnnxruntimeValue*>(sequence_value);
  auto ort_sequence_value = onnxruntime_squence_value->UseOrtValue();

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->GetAllocatorWithDefaultOptions(&ort_allocator),
    ort_api
  );  // This should not be freed as this owned by ort

  size_t num_elements;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetValueCount(ort_sequence_value, &num_elements), ort_api);

  // get the elements
  std::vector<wf::IInspectable> element_map_inspectables;
  for (size_t index = 0; index < num_elements; index++) {
    OrtValue* elements_ort_value = nullptr;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetValue(ort_sequence_value, static_cast<int>(index), ort_allocator, &elements_ort_value), ort_api
    );
    auto unique_element_value = UniqueOrtValue(elements_ort_value, ort_api->ReleaseValue);

    winrt::com_ptr<IValue> element_value;
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
      element_value.put(), this, std::move(unique_element_value), UniqueOrtAllocator(nullptr, nullptr)
    ));

    wf::IInspectable map_inspectable = CreateMap(key_kind, value_kind);
    RETURN_IF_FAILED(FillFromMapValue(
      reinterpret_cast<IInspectable*>(winrt::get_abi(map_inspectable)), key_kind, value_kind, element_value.get()
    ));
    element_map_inspectables.push_back(map_inspectable);
  }

  GetAbiSequenceFiller(key_kind, value_kind)(sequence, element_map_inspectables);
  return S_OK;
}

HRESULT OnnxruntimeEngine::GetSequenceOfTensorValues(
  _In_ _winml::IValue* sequence_value, _Out_ std::vector<winrt::com_ptr<_winml::IValue>>& out_values
) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto onnxruntime_squence_value = static_cast<OnnxruntimeValue*>(sequence_value);
  auto ort_sequence_value = onnxruntime_squence_value->UseOrtValue();

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->GetAllocatorWithDefaultOptions(&ort_allocator),
    ort_api
  );  // This should not be freed as this owned by ort

  size_t num_elements;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetValueCount(ort_sequence_value, &num_elements), ort_api);

  // get the elements
  out_values.clear();
  for (size_t index = 0; index < num_elements; index++) {
    OrtValue* elements_ort_value = nullptr;
    RETURN_HR_IF_NOT_OK_MSG(
      ort_api->GetValue(ort_sequence_value, static_cast<int>(index), ort_allocator, &elements_ort_value), ort_api
    );
    auto unique_element_value = UniqueOrtValue(elements_ort_value, ort_api->ReleaseValue);

    winrt::com_ptr<IValue> element_value;
    RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
      element_value.put(), this, std::move(unique_element_value), UniqueOrtAllocator(nullptr, nullptr)
    ));
    out_values.push_back(element_value);
  }

  return S_OK;
}

HRESULT OnnxruntimeEngine::GetNumberOfIntraOpThreads(uint32_t* num_threads) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SessionGetNumberOfIntraOpThreads(session_.get(), num_threads), ort_api);
  return S_OK;
}

HRESULT OnnxruntimeEngine::GetIntraOpThreadSpinning(bool* allow_spinning) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SessionGetIntraOpThreadSpinning(session_.get(), allow_spinning), ort_api);
  return S_OK;
}

HRESULT OnnxruntimeEngine::GetNamedDimensionOverrides(wfc::IMapView<winrt::hstring, uint32_t>& overrides) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SessionGetNamedDimensionsOverrides(session_.get(), overrides), ort_api);
  return S_OK;
}

HRESULT OnnxruntimeEngine::CreateOneInputAcrossDevices(const char* name, IValue* src, IValue** out) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  auto src_value = static_cast<OnnxruntimeValue*>(src);

  bool is_set;
  auto is_empty = SUCCEEDED(src_value->IsEmpty(&is_set)) && is_set;
  auto is_tensor = SUCCEEDED(src_value->IsTensor(&is_set)) && is_set;

  if (is_tensor && !is_empty) {
    int16_t source_location;
    int16_t input_required_location;
    RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(src_value->UseOrtValue(), &source_location), ort_api);
    RETURN_HR_IF_NOT_OK_MSG(
      winml_adapter_api->SessionGetInputRequiredDeviceId(session_.get(), name, &input_required_location), ort_api
    );

    if (source_location != input_required_location) {
      OrtValue* dest_ort_value = nullptr;
      RETURN_HR_IF_NOT_OK_MSG(
        winml_adapter_api->SessionCopyOneInputAcrossDevices(
          session_.get(), name, src_value->UseOrtValue(), &dest_ort_value
        ),
        ort_api
      );
      auto unique_dest_ort_value = UniqueOrtValue(dest_ort_value, ort_api->ReleaseValue);

      RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
        out, this, std::move(unique_dest_ort_value), UniqueOrtAllocator(nullptr, nullptr)
      ));
      return S_OK;
    }
  }

  *out = src;
  (*out)->AddRef();
  return S_OK;
}

HRESULT OnnxruntimeEngine::Run(
  const char** input_names,
  IValue** inputs,
  size_t num_inputs,
  const char** output_names,
  IValue** outputs,
  size_t num_outputs
) {
  auto ort_api = engine_factory_->UseOrtApi();

  OrtRunOptions* run_options;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->CreateRunOptions(&run_options), ort_api);
  auto unique_run_options = UniqueOrtRunOptions(run_options, ort_api->ReleaseRunOptions);

  std::vector<OrtValue*> input_ort_values;
  std::transform(inputs, inputs + num_inputs, std::back_inserter(input_ort_values), [&](auto& input) {
    auto input_value = static_cast<OnnxruntimeValue*>(input);
    return input_value->UseOrtValue();
  });

  std::vector<OrtValue*> output_ort_values;
  std::transform(outputs, outputs + num_outputs, std::back_inserter(output_ort_values), [&](auto& output) {
    auto output_value = static_cast<OnnxruntimeValue*>(output);
    return output_value->UseOrtValue();
  });

  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->Run(
      session_.get(),
      unique_run_options.get(),
      input_names,
      input_ort_values.data(),
      num_inputs,
      output_names,
      num_outputs,
      output_ort_values.data()
    ),
    ort_api
  );

  for (size_t index = 0; index < num_outputs; index++) {
    auto output_value = static_cast<OnnxruntimeValue*>(outputs[index]);
    if (output_value->UseOrtValue() != output_ort_values[index]) {
      RETURN_IF_FAILED(output_value->AssignOrtValue(output_ort_values[index]));
    }
  }

  return S_OK;
}

template <typename TAbiKey, typename TAbiValue>
HRESULT FillAbiMap(IInspectable* map_insp, size_t num_elements, void* keys_data, void* values_data) {
  auto map = CastToWinrtMap<TAbiKey, TAbiValue>(map_insp);

  auto keys = reinterpret_cast<typename AbiTypeInfo<TAbiKey>::ResourceType*>(keys_data);
  auto values = reinterpret_cast<typename AbiTypeInfo<TAbiValue>::ResourceType*>(values_data);

  for (size_t i = 0; i < num_elements; ++i) {
    map.Insert(ResourceTypeToCppwinrtType<TAbiKey>(keys[i]), ResourceTypeToCppwinrtType<TAbiValue>(values[i]));
  }
  return S_OK;
}

static auto GetAbiMapFiller(winml::TensorKind key_kind, winml::TensorKind value_kind) {
  using namespace std::placeholders;
  if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Int64) {
    return std::bind(&FillAbiMap<int64_t, int64_t>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Float) {
    return std::bind(&FillAbiMap<int64_t, float>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::Double) {
    return std::bind(&FillAbiMap<int64_t, double>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::Int64 && value_kind == winml::TensorKind::String) {
    return std::bind(&FillAbiMap<int64_t, HSTRING>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Int64) {
    return std::bind(&FillAbiMap<HSTRING, int64_t>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Float) {
    return std::bind(&FillAbiMap<HSTRING, float>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::Double) {
    return std::bind(&FillAbiMap<HSTRING, double>, _1, _2, _3, _4);
  } else if (key_kind == winml::TensorKind::String && value_kind == winml::TensorKind::String) {
    return std::bind(&FillAbiMap<HSTRING, HSTRING>, _1, _2, _3, _4);
  }

  THROW_HR(E_NOTIMPL);
}

HRESULT OnnxruntimeEngine::FillFromMapValue(
  IInspectable* map, winml::TensorKind key_kind, winml::TensorKind value_kind, IValue* map_value
) {
  auto ort_api = engine_factory_->UseOrtApi();
  auto onnxruntime_map_value = static_cast<OnnxruntimeValue*>(map_value);
  auto ort_map_value = onnxruntime_map_value->UseOrtValue();

  OrtAllocator* ort_allocator;
  RETURN_HR_IF_NOT_OK_MSG(
    ort_api->GetAllocatorWithDefaultOptions(&ort_allocator),
    ort_api
  );  // This should not be freed as this owned by ort

  // get the keys
  OrtValue* keys_ort_value = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetValue(ort_map_value, 0, ort_allocator, &keys_ort_value), ort_api);
  auto unique_keys_value = UniqueOrtValue(keys_ort_value, ort_api->ReleaseValue);
  winrt::com_ptr<IValue> keys_value;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    keys_value.put(), this, std::move(unique_keys_value), UniqueOrtAllocator(nullptr, nullptr)
  ));

  // get the keys
  OrtValue* values_ort_value = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->GetValue(ort_map_value, 1, ort_allocator, &values_ort_value), ort_api);
  auto unique_values_value = UniqueOrtValue(values_ort_value, ort_api->ReleaseValue);
  winrt::com_ptr<IValue> values_value;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(
    values_value.put(), this, std::move(unique_values_value), UniqueOrtAllocator(nullptr, nullptr)
  ));

  std::vector<int64_t> keys_shape;
  keys_value->GetTensorShape(keys_shape);

  _winml::Resource keys_data;
  RETURN_IF_FAILED(keys_value->GetResource(keys_data));
  _winml::Resource values_data;
  RETURN_IF_FAILED(values_value->GetResource(values_data));

  auto num_elements = static_cast<size_t>(ShapeSize(keys_shape.data(), keys_shape.size()));
  GetAbiMapFiller(key_kind, value_kind)(map, num_elements, keys_data.get(), values_data.get());

  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::RuntimeClassInitialize() {
  ort_api_ = GetVersionedOrtApi();
  winml_adapter_api_ = GetVersionedWinmlAdapterApi();
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::EnsureEnvironment() {
  if (environment_ == nullptr) {
    std::lock_guard lock(mutex_);
    if (environment_ == nullptr) {
      environment_ = PheonixSingleton<OnnxruntimeEnvironment>(ort_api_);
    }
  }
  return S_OK;
}

STDMETHODIMP
OnnxruntimeEngineFactory::CreateModel(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out) {
  RETURN_IF_FAILED(EnsureEnvironment());

  OrtModel* ort_model = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api_->CreateModelFromPath(model_path, len, &ort_model), ort_api_);

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateModel(_In_opt_ void* data, _In_ size_t size, _Outptr_ IModel** out) {
  RETURN_IF_FAILED(EnsureEnvironment());
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModelFromData(data, size, &ort_model)) {
    return __HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateEmptyModel(_In_ int64_t opset, _Outptr_ _winml::IModel** out) {
  RETURN_IF_FAILED(EnsureEnvironment());
  OrtModel* ort_model = nullptr;
  if (auto status = winml_adapter_api_->CreateModel(opset, &ort_model)) {
    return E_INVALIDARG;
  }

  auto model = UniqueOrtModel(ort_model, winml_adapter_api_->ReleaseModel);
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnruntimeModel>(out, this, std::move(model)));
  return S_OK;
}

STDMETHODIMP OnnxruntimeEngineFactory::CreateEngineBuilder(_Outptr_ _winml::IEngineBuilder** out) {
  RETURN_IF_FAILED(EnsureEnvironment());
  Microsoft::WRL::ComPtr<OnnxruntimeEngineBuilder> onnxruntime_engine_builder;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineBuilder>(&onnxruntime_engine_builder, this));
  RETURN_IF_FAILED(onnxruntime_engine_builder.CopyTo(out));
  return S_OK;
}

const OrtApi* OnnxruntimeEngineFactory::UseOrtApi() {
  return ort_api_;
}

const WinmlAdapterApi* OnnxruntimeEngineFactory::UseWinmlAdapterApi() {
  return winml_adapter_api_;
}

HRESULT OnnxruntimeEngineFactory::GetOrtEnvironment(_Out_ OrtEnv** ort_env) {
  RETURN_IF_FAILED(EnsureEnvironment());
  RETURN_IF_FAILED(environment_->GetOrtEnvironment(ort_env));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::EnableDebugOutput(bool is_enabled) {
  RETURN_IF_FAILED(EnsureEnvironment());
  RETURN_IF_FAILED(environment_->EnableDebugOutput(is_enabled));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::CreateCustomRegistry(_Out_ IMLOperatorRegistry** registry) {
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api_->CreateCustomRegistry(registry), ort_api_);
  return S_OK;
}

STDAPI CreateOnnxruntimeEngineFactory(_Out_ _winml::IEngineFactory** engine_factory) {
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> onnxruntime_engine_factory;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeEngineFactory>(&onnxruntime_engine_factory));
  RETURN_IF_FAILED(onnxruntime_engine_factory.CopyTo(engine_factory));
  return S_OK;
}
struct OrtDescriptorInfo : public Microsoft::WRL::RuntimeClass<
                             Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                             IDescriptorInfo,
                             IOrtTypeInfoProvider> {
  OrtDescriptorInfo() : info_(nullptr, nullptr) {}

  HRESULT RuntimeClassInitialize(UniqueOrtTypeInfo info) {
    info_ = std::move(info);
    return S_OK;
  }

  STDMETHOD(GetTypeInfo)(OrtTypeInfo** info) override {
    *info = info_.get();
    return S_OK;
  }

  OrtTypeInfo* UseOrtTypeInfo() { return info_.get(); }

 private:
  UniqueOrtTypeInfo info_;
};

HRESULT OnnxruntimeEngineFactory::CreateTensorDescriptorInfo(
  _In_ winml::TensorKind kind, _In_ int64_t* dims, _In_ size_t num_dims, _Out_ IDescriptorInfo** tensor_info
) {
  OrtTypeInfo* tensor_type_info = nullptr;
  winml_adapter_api_->CreateTensorTypeInfo(
    dims, num_dims, ONNXTensorElementDataTypeFromTensorKind(kind), &tensor_type_info
  );
  UniqueOrtTypeInfo info(tensor_type_info, ort_api_->ReleaseTypeInfo);

  Microsoft::WRL::ComPtr<OrtDescriptorInfo> descriptor_info;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OrtDescriptorInfo>(&descriptor_info, std::move(info)));
  RETURN_IF_FAILED(descriptor_info.CopyTo(tensor_info));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::CreateSequenceDescriptorInfo(_Out_ IDescriptorInfo** seq_info) {
  OrtTypeInfo* sequence_type_info = nullptr;
  winml_adapter_api_->CreateSequenceTypeInfo(&sequence_type_info);
  UniqueOrtTypeInfo info(sequence_type_info, ort_api_->ReleaseTypeInfo);

  Microsoft::WRL::ComPtr<OrtDescriptorInfo> descriptor_info;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OrtDescriptorInfo>(&descriptor_info, std::move(info)));
  RETURN_IF_FAILED(descriptor_info.CopyTo(seq_info));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::CreateMapDescriptorInfo(_Out_ IDescriptorInfo** desc_info) {
  OrtTypeInfo* map_type_info = nullptr;
  winml_adapter_api_->CreateMapTypeInfo(&map_type_info);
  UniqueOrtTypeInfo info(map_type_info, ort_api_->ReleaseTypeInfo);

  Microsoft::WRL::ComPtr<OrtDescriptorInfo> descriptor_info;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OrtDescriptorInfo>(&descriptor_info, std::move(info)));
  RETURN_IF_FAILED(descriptor_info.CopyTo(desc_info));
  return S_OK;
}

HRESULT OnnxruntimeEngineFactory::CreateThreadPool(
  _In_ bool allow_spinning, _In_ uint32_t num_intra_op_threads, _Out_ IThreading** thread_pool
) {
  Microsoft::WRL::ComPtr<OnnxruntimeThreading> threading;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeThreading>(&threading, this));

  OrtThreadPoolOptions intra_op_params = {};
  intra_op_params.name = L"WinML Worker Thread";
  intra_op_params.thread_pool_size = num_intra_op_threads;
  intra_op_params.set_denormal_as_zero = false;
  intra_op_params.allow_spinning = allow_spinning;

  OrtThreadPool* ort_intra_op_thread_pool = nullptr;
  RETURN_HR_IF_NOT_OK_MSG(
    winml_adapter_api_->CreateThreadPool(ThreadPoolType::INTRA_OP, &intra_op_params, &ort_intra_op_thread_pool),
    ort_api_
  );
  UniqueOrtThreadPool intra_op_pool(ort_intra_op_thread_pool, winml_adapter_api_->ReleaseThreadPool);
  threading->SetIntraOpThreadPool(std::move(intra_op_pool));

  // There is no need to set the inter_op thread pool, as WinML does not use parallel execution...

  RETURN_IF_FAILED(threading.CopyTo(thread_pool));
  return S_OK;
}

OnnxruntimeThreading::OnnxruntimeThreading()
  : inter_op_ort_pool_(nullptr, nullptr),
    intra_op_ort_pool_(nullptr, nullptr) {
}
OnnxruntimeThreading::~OnnxruntimeThreading() = default;

HRESULT OnnxruntimeThreading::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory) {
  RETURN_HR_IF_NULL(E_INVALIDARG, engine_factory);
  engine_factory_ = engine_factory;
  return S_OK;
}

HRESULT OnnxruntimeThreading::SetIntraOpThreadPool(UniqueOrtThreadPool&& intra_op_ort_pool) {
  RETURN_HR_IF_NULL(E_INVALIDARG, intra_op_ort_pool);
  intra_op_ort_pool_ = std::move(intra_op_ort_pool);
  return S_OK;
}

HRESULT OnnxruntimeThreading::SetInterOpThreadPool(UniqueOrtThreadPool&& inter_op_ort_pool) {
  RETURN_HR_IF_NULL(E_INVALIDARG, inter_op_ort_pool);
  inter_op_ort_pool_ = std::move(inter_op_ort_pool);
  return S_OK;
}

OrtThreadPool* OnnxruntimeThreading::UseIntraOpThreadPool() {
  return intra_op_ort_pool_.get();
}

OrtThreadPool* OnnxruntimeThreading::UseInterOpThreadPool() {
  return inter_op_ort_pool_.get();
}
