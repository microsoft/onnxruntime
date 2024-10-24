#pragma once

#ifndef WINML_H_
#define WINML_H_

#include "weak_buffer.h"
#include "buffer_backed_random_access_stream_reference.h"
#include "weak_single_threaded_iterable.h"

#define RETURN_HR_IF_FAILED(expression) \
  do {                                  \
    auto _hr = expression;              \
    if (FAILED(_hr)) {                  \
      return static_cast<int32_t>(_hr); \
    }                                   \
  } while (0)

#define FAIL_FAST_IF_HR_FAILED(expression)   \
  do {                                       \
    auto _hr = expression;                   \
    if (FAILED(_hr)) {                       \
      __fastfail(static_cast<int32_t>(_hr)); \
    }                                        \
  } while (0)

struct float16 {
  uint16_t value;
};

namespace Microsoft {
namespace AI {
namespace MachineLearning {
namespace Details {

class WinMLLearningModel;
class WinMLLearningModelBinding;
class WinMLLearningModelSession;
class WinMLLearningModelResults;

extern const __declspec(selectany) _Null_terminated_ wchar_t MachineLearningDll[] = L"windows.ai.machinelearning.dll";

template <typename T>
struct Tensor {};
template <>
struct Tensor<float> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorFloat;
};
template <>
struct Tensor<float16> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorFloat16Bit;
};
template <>
struct Tensor<int8_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorInt8Bit;
};
template <>
struct Tensor<uint8_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorUInt8Bit;
};
template <>
struct Tensor<uint16_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorUInt16Bit;
};
template <>
struct Tensor<int16_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorInt16Bit;
};
template <>
struct Tensor<uint32_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorUInt32Bit;
};
template <>
struct Tensor<int32_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorInt32Bit;
};
template <>
struct Tensor<uint64_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorUInt64Bit;
};
template <>
struct Tensor<int64_t> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorInt64Bit;
};
template <>
struct Tensor<bool> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorBoolean;
};
template <>
struct Tensor<double> {
  using Type = ABI::Windows::AI::MachineLearning::ITensorDouble;
};

template <typename T>
struct TensorRuntimeClassID {};
template <>
struct TensorRuntimeClassID<float> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<float16> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<int8_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<uint8_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<uint16_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<int16_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<uint32_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<int32_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<uint64_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<int64_t> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<bool> {
  static const wchar_t* RuntimeClass_ID;
};
template <>
struct TensorRuntimeClassID<double> {
  static const wchar_t* RuntimeClass_ID;
};

__declspec(selectany
) const wchar_t* TensorRuntimeClassID<float>::RuntimeClass_ID = RuntimeClass_Windows_AI_MachineLearning_TensorFloat;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<float16>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorFloat16Bit;
__declspec(selectany
) const wchar_t* TensorRuntimeClassID<int8_t>::RuntimeClass_ID = RuntimeClass_Windows_AI_MachineLearning_TensorInt8Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<uint8_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorUInt8Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<uint16_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorUInt16Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<int16_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorInt16Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<uint32_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorUInt32Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<int32_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorInt32Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<uint64_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorUInt64Bit;
__declspec(selectany) const wchar_t* TensorRuntimeClassID<int64_t>::RuntimeClass_ID =
  RuntimeClass_Windows_AI_MachineLearning_TensorInt64Bit;
__declspec(selectany
) const wchar_t* TensorRuntimeClassID<bool>::RuntimeClass_ID = RuntimeClass_Windows_AI_MachineLearning_TensorBoolean;
__declspec(selectany
) const wchar_t* TensorRuntimeClassID<double>::RuntimeClass_ID = RuntimeClass_Windows_AI_MachineLearning_TensorDouble;

template <typename T>
struct TensorFactory {};
template <>
struct TensorFactory<float> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorFloatStatics;
};
template <>
struct TensorFactory<float16> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorFloat16BitStatics;
};
template <>
struct TensorFactory<int8_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt8BitStatics;
};
template <>
struct TensorFactory<uint8_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt8BitStatics;
};
template <>
struct TensorFactory<uint16_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt16BitStatics;
};
template <>
struct TensorFactory<int16_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt16BitStatics;
};
template <>
struct TensorFactory<uint32_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt32BitStatics;
};
template <>
struct TensorFactory<int32_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt32BitStatics;
};
template <>
struct TensorFactory<uint64_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt64BitStatics;
};
template <>
struct TensorFactory<int64_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt64BitStatics;
};
template <>
struct TensorFactory<bool> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorBooleanStatics;
};
template <>
struct TensorFactory<double> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorDoubleStatics;
};

template <typename T>
struct TensorFactory2 {};
template <>
struct TensorFactory2<float> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorFloatStatics2;
};
template <>
struct TensorFactory2<float16> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorFloat16BitStatics2;
};
template <>
struct TensorFactory2<int8_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt8BitStatics2;
};
template <>
struct TensorFactory2<uint8_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt8BitStatics2;
};
template <>
struct TensorFactory2<uint16_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt16BitStatics2;
};
template <>
struct TensorFactory2<int16_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt16BitStatics2;
};
template <>
struct TensorFactory2<uint32_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt32BitStatics2;
};
template <>
struct TensorFactory2<int32_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt32BitStatics2;
};
template <>
struct TensorFactory2<uint64_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorUInt64BitStatics2;
};
template <>
struct TensorFactory2<int64_t> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorInt64BitStatics2;
};
template <>
struct TensorFactory2<bool> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorBooleanStatics2;
};
template <>
struct TensorFactory2<double> {
  using Factory = ABI::Windows::AI::MachineLearning::ITensorDoubleStatics2;
};

template <typename T>
struct TensorFactoryIID {};
template <>
struct TensorFactoryIID<float> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<float16> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<int8_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<uint8_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<uint16_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<int16_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<uint32_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<int32_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<uint64_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<int64_t> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<bool> {
  static const GUID IID;
};
template <>
struct TensorFactoryIID<double> {
  static const GUID IID;
};

__declspec(selectany
) const GUID TensorFactoryIID<float>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorFloatStatics;
__declspec(selectany
) const GUID TensorFactoryIID<float16>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorFloat16BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<int8_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt8BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<uint8_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt8BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<uint16_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt16BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<int16_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt16BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<uint32_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt32BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<int32_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt32BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<uint64_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt64BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<int64_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt64BitStatics;
__declspec(selectany
) const GUID TensorFactoryIID<bool>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorBooleanStatics;
__declspec(selectany
) const GUID TensorFactoryIID<double>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorDoubleStatics;

template <typename T>
struct TensorFactory2IID {};
template <>
struct TensorFactory2IID<float> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<float16> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<int8_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<uint8_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<uint16_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<int16_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<uint32_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<int32_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<uint64_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<int64_t> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<bool> {
  static const GUID IID;
};
template <>
struct TensorFactory2IID<double> {
  static const GUID IID;
};

__declspec(selectany
) const GUID TensorFactory2IID<float>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorFloatStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<float16>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorFloat16BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<int8_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt8BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<uint8_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt8BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<uint16_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt16BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<int16_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt16BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<uint32_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt32BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<int32_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt32BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<uint64_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorUInt64BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<int64_t>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorInt64BitStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<bool>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorBooleanStatics2;
__declspec(selectany
) const GUID TensorFactory2IID<double>::IID = ABI::Windows::AI::MachineLearning::IID_ITensorDoubleStatics2;

inline HRESULT GetActivationFactory(const wchar_t* p_class_id, const IID& iid, void** factory) noexcept {
  // Fallback to OS binary if the redistributable is not present!
  auto library = LoadLibraryExW(MachineLearningDll, nullptr, 0);

  using DllGetActivationFactory = HRESULT __stdcall(HSTRING, void** factory);
  auto call = reinterpret_cast<DllGetActivationFactory*>(GetProcAddress(library, "DllGetActivationFactory"));
  if (!call) {
    auto hr = HRESULT_FROM_WIN32(GetLastError());
    FreeLibrary(library);
    return hr;
  }

  Microsoft::WRL::ComPtr<IActivationFactory> activation_factory;
  auto hr = call(
    Microsoft::WRL::Wrappers::HStringReference(p_class_id, static_cast<unsigned int>(wcslen(p_class_id))).Get(),
    reinterpret_cast<void**>(activation_factory.GetAddressOf())
  );

  if (FAILED(hr)) {
    FreeLibrary(library);
    return hr;
  }

  return activation_factory->QueryInterface(iid, factory);
}

class WinMLLearningModel {
  friend class WinMLLearningModelSession;

 public:
  WinMLLearningModel(const wchar_t* model_path, size_t size) { ML_FAIL_FAST_IF(0 != Initialize(model_path, size)); }

  WinMLLearningModel(const char* bytes, size_t size) {
    ML_FAIL_FAST_IF(0 != Initialize(bytes, size, false /*with_copy*/));
  }

 private:
  int32_t Initialize(const wchar_t* model_path, size_t size) {
    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelStatics> learningModel;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModel,
      ABI::Windows::AI::MachineLearning::IID_ILearningModelStatics,
      &learningModel
    ));

    RETURN_HR_IF_FAILED(learningModel->LoadFromFilePath(
      Microsoft::WRL::Wrappers::HStringReference(model_path, static_cast<unsigned int>(size)).Get(),
      m_learning_model.GetAddressOf()
    ));
    return 0;
  }

  struct StoreCompleted : Microsoft::WRL::RuntimeClass<
                            Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                            ABI::Windows::Foundation::IAsyncOperationCompletedHandler<uint32_t>> {
    HANDLE completed_event_;

    StoreCompleted() : completed_event_(CreateEvent(nullptr, true, false, nullptr)) {}

    ~StoreCompleted() { CloseHandle(completed_event_); }

    HRESULT STDMETHODCALLTYPE Invoke(
      ABI::Windows::Foundation::IAsyncOperation<uint32_t>* asyncInfo, ABI::Windows::Foundation::AsyncStatus status
    ) {
      SetEvent(completed_event_);
      return S_OK;
    }

    HRESULT Wait() {
      WaitForSingleObject(completed_event_, INFINITE);
      return S_OK;
    }
  };

  int32_t Initialize(const char* bytes, size_t size, bool with_copy = false) {
    RoInitialize(RO_INIT_TYPE::RO_INIT_SINGLETHREADED);

    Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IRandomAccessStreamReference> random_access_stream_ref;
    if (with_copy) {
      // Create in memory stream
      Microsoft::WRL::ComPtr<IInspectable> in_memory_random_access_stream_insp;
      RETURN_HR_IF_FAILED(RoActivateInstance(
        Microsoft::WRL::Wrappers::HStringReference(RuntimeClass_Windows_Storage_Streams_InMemoryRandomAccessStream)
          .Get(),
        in_memory_random_access_stream_insp.GetAddressOf()
      ));

      // QI memory stream to output stream
      Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IOutputStream> output_stream;
      RETURN_HR_IF_FAILED(in_memory_random_access_stream_insp.As(&output_stream));

      // Create data writer factory
      Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IDataWriterFactory> activation_factory;
      RETURN_HR_IF_FAILED(RoGetActivationFactory(
        Microsoft::WRL::Wrappers::HStringReference(RuntimeClass_Windows_Storage_Streams_DataWriter).Get(),
        IID_PPV_ARGS(activation_factory.GetAddressOf())
      ));

      // Create data writer object based on the in memory stream
      Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IDataWriter> data_writer;
      RETURN_HR_IF_FAILED(activation_factory->CreateDataWriter(output_stream.Get(), data_writer.GetAddressOf()));

      // Write the model to the data writer and thus to the stream
      RETURN_HR_IF_FAILED(
        data_writer->WriteBytes(static_cast<uint32_t>(size), reinterpret_cast<BYTE*>(const_cast<char*>(bytes)))
      );

      // QI the in memory stream to a random access stream
      Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IRandomAccessStream> random_access_stream;
      RETURN_HR_IF_FAILED(in_memory_random_access_stream_insp.As(&random_access_stream));

      // Create a random access stream reference factory
      Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IRandomAccessStreamReferenceStatics>
        random_access_stream_ref_statics;
      RETURN_HR_IF_FAILED(RoGetActivationFactory(
        Microsoft::WRL::Wrappers::HStringReference(RuntimeClass_Windows_Storage_Streams_RandomAccessStreamReference)
          .Get(),
        IID_PPV_ARGS(random_access_stream_ref_statics.GetAddressOf())
      ));

      // Create a random access stream reference from the random access stream view on top of
      // the in memory stream
      RETURN_HR_IF_FAILED(random_access_stream_ref_statics->CreateFromStream(
        random_access_stream.Get(), random_access_stream_ref.GetAddressOf()
      ));
    } else {
      Microsoft::WRL::ComPtr<WinMLTest::WeakBuffer<BYTE>> buffer;
      RETURN_HR_IF_FAILED(Microsoft::WRL::MakeAndInitialize<WinMLTest::WeakBuffer<BYTE>>(&buffer, bytes, bytes + size));

      RETURN_HR_IF_FAILED(Microsoft::WRL::MakeAndInitialize<WinMLTest::BufferBackedRandomAccessStreamReference>(
        &random_access_stream_ref, buffer.Get()
      ));
    }

    // Create a learning model factory
    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelStatics> learning_model;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModel,
      ABI::Windows::AI::MachineLearning::IID_ILearningModelStatics,
      &learning_model
    ));

    Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncOperation<uint32_t>> async_operation;
    RETURN_HR_IF_FAILED(data_writer->StoreAsync(&async_operation));
    auto store_completed_handler = Microsoft::WRL::Make<StoreCompleted>();
    RETURN_HR_IF_FAILED(async_operation->put_Completed(store_completed_handler.Get()));
    RETURN_HR_IF_FAILED(store_completed_handler->Wait());

    // Create a learning model from the factory with the random access stream reference that points
    // to the random access stream view on top of the in memory stream copy of the model
    RETURN_HR_IF_FAILED(learning_model->LoadFromStream(random_access_stream_ref.Get(), m_learning_model.GetAddressOf())
    );

    return 0;
  }

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModel> m_learning_model;
};

class WinMLLearningModelResults {
  friend class WinMLLearningModelSession;

 public:
  int32_t get_output(const wchar_t* feature_name, size_t feature_name_size, void** pp_buffer, size_t* p_capacity) {
    Microsoft::WRL::ComPtr<ABI::Windows::Foundation::Collections::IMapView<HSTRING, IInspectable*>> output_map;
    RETURN_HR_IF_FAILED(m_result->get_Outputs(&output_map));

    Microsoft::WRL::ComPtr<IInspectable> inspectable;
    RETURN_HR_IF_FAILED(output_map->Lookup(
      Microsoft::WRL::Wrappers::HStringReference(feature_name, static_cast<unsigned int>(feature_name_size)).Get(),
      inspectable.GetAddressOf()
    ));

    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelFeatureValue> output_feature_value;
    RETURN_HR_IF_FAILED(inspectable.As(&output_feature_value));

    Microsoft::WRL::ComPtr<ITensorNative> native_tensor_float_feature_value;
    RETURN_HR_IF_FAILED(output_feature_value.As(&native_tensor_float_feature_value));

    uint32_t size;
    RETURN_HR_IF_FAILED(native_tensor_float_feature_value->GetBuffer(reinterpret_cast<BYTE**>(pp_buffer), &size));
    *p_capacity = size;

    return 0;
  }

 private:
  WinMLLearningModelResults(ABI::Windows::AI::MachineLearning::ILearningModelEvaluationResult* p_result) {
    m_result = p_result;
  }

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelEvaluationResult> m_result;
};

class WinMLLearningModelBinding {
  friend class WinMLLearningModelSession;

 public:
  WinMLLearningModelBinding(const WinMLLearningModelSession& session) { ML_FAIL_FAST_IF(0 != Initialize(session)); }

  template <typename T = float>
  int32_t bind(
    const wchar_t* feature_name,
    size_t feature_name_size,
    tensor_shape_type* p_shape,
    size_t shape_size,
    T* p_data,
    size_t data_size
  ) {
    using ITensor = typename Tensor<T>::Type;
    using ITensorFactory = typename TensorFactory<T>::Factory;

    Microsoft::WRL::ComPtr<ITensorFactory> tensor_factory;
    RETURN_HR_IF_FAILED(
      GetActivationFactory(TensorRuntimeClassID<T>::RuntimeClass_ID, TensorFactoryIID<T>::IID, &tensor_factory)
    );

    Microsoft::WRL::ComPtr<weak_single_threaded_iterable<int64_t>> input_shape_iterable;
    RETURN_HR_IF_FAILED(Microsoft::WRL::MakeAndInitialize<weak_single_threaded_iterable<int64_t>>(
      &input_shape_iterable, p_shape, p_shape + shape_size
    ));

    Microsoft::WRL::ComPtr<ITensor> tensor;
    RETURN_HR_IF_FAILED(tensor_factory->CreateFromArray(
      input_shape_iterable.Get(), static_cast<uint32_t>(data_size), p_data, tensor.GetAddressOf()
    ));

    Microsoft::WRL::ComPtr<IInspectable> inspectable_tensor;
    RETURN_HR_IF_FAILED(tensor.As(&inspectable_tensor));

    RETURN_HR_IF_FAILED(m_learning_model_binding->Bind(
      Microsoft::WRL::Wrappers::HStringReference(feature_name, static_cast<unsigned int>(feature_name_size)).Get(),
      inspectable_tensor.Get()
    ));
    return 0;
  }

  template <typename T = float>
  int32_t bind(
    const wchar_t* /*feature_name*/, size_t /*feature_name_size*/, tensor_shape_type* /*p_shape*/, size_t /*shape_size*/
  ) {
    return 0;
  }

  template <typename T = float>
  int32_t bind_as_reference(
    const wchar_t* feature_name,
    size_t feature_name_size,
    tensor_shape_type* p_shape,
    size_t shape_size,
    T* p_data,
    size_t data_size
  ) {
    using ITensor = typename Tensor<T>::Type;
    using ITensorFactory = typename TensorFactory2<T>::Factory;

    Microsoft::WRL::ComPtr<ITensorFactory> tensor_factory;
    RETURN_HR_IF_FAILED(
      GetActivationFactory(TensorRuntimeClassID<T>::RuntimeClass_ID, TensorFactory2IID<T>::IID, &tensor_factory)
    );

    Microsoft::WRL::ComPtr<WinMLTest::WeakBuffer<T>> buffer;
    RETURN_HR_IF_FAILED(Microsoft::WRL::MakeAndInitialize<WinMLTest::WeakBuffer<T>>(&buffer, p_data, p_data + data_size)
    );

    Microsoft::WRL::ComPtr<ITensor> tensor;
    RETURN_HR_IF_FAILED(
      tensor_factory->CreateFromBuffer(static_cast<uint32_t>(shape_size), p_shape, buffer.Get(), tensor.GetAddressOf())
    );

    Microsoft::WRL::ComPtr<IInspectable> inspectable_tensor;
    RETURN_HR_IF_FAILED(tensor.As(&inspectable_tensor));

    RETURN_HR_IF_FAILED(m_learning_model_binding->Bind(
      Microsoft::WRL::Wrappers::HStringReference(feature_name, static_cast<unsigned int>(feature_name_size)).Get(),
      inspectable_tensor.Get()
    ));
    return 0;
  }

  template <typename T = float>
  int32_t bind_as_references(
    const wchar_t* feature_name, size_t feature_name_size, T** p_data, size_t* data_sizes, size_t num_buffers
  ) {
    using ITensor = typename Tensor<T>::Type;
    using ITensorFactory = typename TensorFactory2<T>::Factory;

    std::vector<Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IBuffer>> vec_buffers(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      RETURN_HR_IF_FAILED(Microsoft::WRL::MakeAndInitialize<WinMLTest::WeakBuffer<T>>(
        &vec_buffers.at(i), p_data[i], p_data[i] + data_sizes[i]
      ));
    }

    std::vector<ABI::Windows::Storage::Streams::IBuffer*> raw_buffers(num_buffers);
    std::transform(std::begin(vec_buffers), std::end(vec_buffers), std::begin(raw_buffers), [](auto buffer) {
      return buffer.Get();
    });

    Microsoft::WRL::ComPtr<weak_single_threaded_iterable<ABI::Windows::Storage::Streams::IBuffer*>> buffers;
    RETURN_HR_IF_FAILED(
      Microsoft::WRL::MakeAndInitialize<weak_single_threaded_iterable<ABI::Windows::Storage::Streams::IBuffer*>>(
        &buffers, raw_buffers, raw_buffers + num_buffers
      )
    );

    Microsoft::WRL::ComPtr<IInspectable> inspectable_tensor;
    RETURN_HR_IF_FAILED(buffers.As(&inspectable_tensor));

    RETURN_HR_IF_FAILED(m_learning_model_binding->Bind(
      Microsoft::WRL::Wrappers::HStringReference(feature_name, static_cast<unsigned int>(feature_name_size)).Get(),
      inspectable_tensor.Get()
    ));
    return 0;
  }

 private:
  inline int32_t Initialize(const WinMLLearningModelSession& session);

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelBinding> m_learning_model_binding;
};

class WinMLLearningModelDevice {
  friend class WinMLLearningModelSession;

 public:
  WinMLLearningModelDevice()
    : WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind_Default) {}

  WinMLLearningModelDevice(WinMLLearningModelDevice&& device)
    : m_learning_model_device(std::move(device.m_learning_model_device)) {}

  WinMLLearningModelDevice(const WinMLLearningModelDevice& device)
    : m_learning_model_device(device.m_learning_model_device) {}

  void operator=(const WinMLLearningModelDevice& device) { m_learning_model_device = device.m_learning_model_device; }

  WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind kind) {
    ML_FAIL_FAST_IF(0 != Initialize(kind));
  }

  WinMLLearningModelDevice(ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice* d3dDevice) {
    ML_FAIL_FAST_IF(0 != Initialize(d3dDevice));
  }

  WinMLLearningModelDevice(ID3D12CommandQueue* queue) { ML_FAIL_FAST_IF(0 != Initialize(queue)); }

  static WinMLLearningModelDevice create_cpu_device() {
    return WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind_Cpu);
  }

  static WinMLLearningModelDevice create_directx_device() {
    return WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind_DirectX);
  }

  static WinMLLearningModelDevice create_directx_high_power_device() {
    return WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind_DirectXHighPerformance);
  }

  static WinMLLearningModelDevice create_directx_min_power_device() {
    return WinMLLearningModelDevice(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind_DirectXMinPower);
  }

  static WinMLLearningModelDevice create_directx_device(
    ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice* d3dDevice
  ) {
    return WinMLLearningModelDevice(d3dDevice);
  }

  static WinMLLearningModelDevice create_directx_device(ID3D12CommandQueue* queue) {
    return WinMLLearningModelDevice(queue);
  }

 private:
  int32_t Initialize(ABI::Windows::AI::MachineLearning::LearningModelDeviceKind kind) {
    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelDeviceFactory>
      learning_model_device_factory;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModelDevice,
      ABI::Windows::AI::MachineLearning::IID_ILearningModelDeviceFactory,
      &learning_model_device_factory
    ));

    RETURN_HR_IF_FAILED(learning_model_device_factory->Create(kind, &m_learning_model_device));

    return 0;
  }

  int32_t Initialize(ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice* d3dDevice) {
    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelDeviceStatics>
      learning_model_device_factory;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModelDevice,
      ABI::Windows::AI::MachineLearning::IID_ILearningModelDeviceStatics,
      &learning_model_device_factory
    ));

    RETURN_HR_IF_FAILED(learning_model_device_factory->CreateFromDirect3D11Device(d3dDevice, &m_learning_model_device));

    return 0;
  }

  int32_t Initialize(ID3D12CommandQueue* queue) {
    Microsoft::WRL::ComPtr<ILearningModelDeviceFactoryNative> learning_model_device_factory;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModelDevice,
      __uuidof(ILearningModelDeviceFactoryNative),
      &learning_model_device_factory
    ));

    RETURN_HR_IF_FAILED(learning_model_device_factory->CreateFromD3D12CommandQueue(queue, &m_learning_model_device));

    return 0;
  }

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelDevice> m_learning_model_device;
};

class WinMLLearningModelSession {
  friend class WinMLLearningModelBinding;

 public:
  using Model = WinMLLearningModel;
  using Device = WinMLLearningModelDevice;

 public:
  WinMLLearningModelSession(const Model& model) { ML_FAIL_FAST_IF(0 != Initialize(model, Device())); }

  WinMLLearningModelSession(const Model& model, const Device& device) {
    ML_FAIL_FAST_IF(0 != Initialize(model, device));
  }

  WinMLLearningModelResults evaluate(WinMLLearningModelBinding& binding) {
    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelEvaluationResult>
      m_learning_model_evaluation_result;

    FAIL_FAST_IF_HR_FAILED(m_learning_model_session->Evaluate(
      binding.m_learning_model_binding.Get(), nullptr, m_learning_model_evaluation_result.GetAddressOf()
    ));

    return WinMLLearningModelResults(m_learning_model_evaluation_result.Get());
  }

 private:
  int32_t Initialize(const Model& model, const Device& device) {
    // {0f6b881d-1c9b-47b6-bfe0-f1cf62a67579}
    static const GUID IID_ILearningModelSessionFactory = {
      0x0f6b881d, 0x1c9b, 0x47b6, {0xbf, 0xe0, 0xf1, 0xcf, 0x62, 0xa6, 0x75, 0x79}
    };

    Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelSessionFactory>
      m_learning_model_session_factory;
    RETURN_HR_IF_FAILED(GetActivationFactory(
      RuntimeClass_Windows_AI_MachineLearning_LearningModelSession,
      IID_ILearningModelSessionFactory,
      &m_learning_model_session_factory
    ));

    RETURN_HR_IF_FAILED(m_learning_model_session_factory->CreateFromModelOnDevice(
      model.m_learning_model.Get(), device.m_learning_model_device.Get(), m_learning_model_session.GetAddressOf()
    ));

    return 0;
  }

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelSession> m_learning_model_session;
};

inline int32_t WinMLLearningModelBinding::Initialize(const WinMLLearningModelSession& session) {
  // {c95f7a7a-e788-475e-8917-23aa381faf0b}
  static const GUID IID_ILearningModelBindingFactory = {
    0xc95f7a7a, 0xe788, 0x475e, {0x89, 0x17, 0x23, 0xaa, 0x38, 0x1f, 0xaf, 0x0b}
  };

  Microsoft::WRL::ComPtr<ABI::Windows::AI::MachineLearning::ILearningModelBindingFactory>
    learning_model_binding_factory;

  RETURN_HR_IF_FAILED(GetActivationFactory(
    RuntimeClass_Windows_AI_MachineLearning_LearningModelBinding,
    IID_ILearningModelBindingFactory,
    &learning_model_binding_factory
  ));

  RETURN_HR_IF_FAILED(learning_model_binding_factory->CreateFromSession(
    session.m_learning_model_session.Get(), m_learning_model_binding.GetAddressOf()
  ));

  return 0;
}

}  // namespace Details
}  // namespace MachineLearning
}  // namespace AI
}  // namespace Microsoft

#endif  // WINML_H_
