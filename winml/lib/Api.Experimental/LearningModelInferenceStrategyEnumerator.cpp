#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelEnumerateInferenceStrategiesOptions.h"
#include "LearningModelInferenceStrategyEnumerator.h"
#include "LearningModelBindingStrategyFilter.h"
#include "LearningModelBindingStrategyFilter.h"
#include "LearningModelBatchingStrategyFilter.h"
#include "LearningModelInferenceStrategy.h"
#include "TypeConversionHelpers.h"

#include "LearningModel.h"
#include "LearningModelDevice.h"
#include "LearningModelSession.h"

#include <algorithm>
#include <chrono>

#include <robuffer.h>
#include <Memorybuffer.h>

namespace WINML_EXPERIMENTALP
{
    //
    // CreateFromShapeArrayAndDataBuffer<T, DataType>
    // Call the CreateFromBuffer API
    //
    template <TensorKind T, typename DataType>
    winrt::Windows::Foundation::IInspectable CreateFromShapeArrayAndDataBuffer(
        const std::vector<int64_t>& shape,
        std::vector<DataType>& data) {
        using TensorValue = typename TensorKindToValue<T>::Type;

        winrt::Windows::Storage::Streams::DataWriter writer;

        const uint8_t* pData = reinterpret_cast<const uint8_t*>(&data[0]);
        winrt::array_view<const uint8_t> dataArray(pData, pData + (data.size() * sizeof(DataType)));
        writer.WriteBytes(dataArray);

        return TensorValue::CreateFromBuffer(shape, writer.DetachBuffer());
    }

    //
    // CreateFromShapeArrayAndDataBuffer<TensorKind::String, winrt::hstring>
    // Specialization for strings that should never be called. Need to specialize since TensorString doesnt have CreateFromBuffer
    // and the compiler will attempt to find the symbol if not specialized.
    //
    template <>
    winrt::Windows::Foundation::IInspectable CreateFromShapeArrayAndDataBuffer<TensorKind::String, winrt::hstring>(
        const std::vector<int64_t>& /*shape*/,
        std::vector<winrt::hstring>& /*data*/) {
      return nullptr;
    }

    //
    // CreateFromArray<T>
    // Calls the CreateFromArray API
    //
    template <TensorKind T>
    auto CreateFromArray(
        const std::vector<int64_t>& shape,
        std::vector<typename TensorKindToPointerType<T>::Type>& data) {
      using TensorValue = typename TensorKindToValue<T>::Type;
      using DataType = typename TensorKindToPointerType<T>::Type;

      return TensorValue::CreateFromArray(shape, winrt::array_view<const DataType>(data));
    }

    //
    // CreateFromArray<T>
    // Specialization for boolean to convert to array_view<bool>, but test data is in std::vector<boolean>
    //
    template <>
    auto CreateFromArray<TensorKind::Boolean>(
        const std::vector<int64_t>& shape,
        std::vector<boolean>& data) {
      std::unique_ptr<bool> out(new bool[data.size()]);
      std::transform(data.begin(), data.end(), out.get(), [](boolean in) { return static_cast<bool>(in); });
      return winml::TensorBoolean::CreateFromArray(
          shape,
          winrt::array_view<const bool>(out.get(), out.get() + data.size()));
    }

    //
    // CreateFromShapeArrayAndDataArray<T>
    // Calls the CreateFromShapeArrayAndDataArray API
    //
    template <TensorKind T>
    auto CreateFromShapeArrayAndDataArray(
        const std::vector<int64_t>& shape,
        std::vector<typename TensorKindToPointerType<T>::Type>& data) {
      using TensorValue = typename TensorKindToValue<T>::Type;
      using DataType = typename TensorKindToPointerType<T>::Type;

      return TensorValue::CreateFromShapeArrayAndDataArray(shape, winrt::array_view<const DataType>(data));
    }

    //
    // CreateFromShapeArrayAndDataArray<T>
    // Specialization for boolean to convert to array_view<bool>, but test data is in std::vector<boolean>
    //
    template <>
    auto CreateFromShapeArrayAndDataArray<TensorKind::Boolean>(
        const std::vector<int64_t>& shape,
        std::vector<boolean>& data) {
        std::unique_ptr<bool> out(new bool[data.size()]);
        std::transform(data.begin(), data.end(), out.get(), [](boolean in) { return static_cast<bool>(in); });

        return winml::TensorBoolean::CreateFromShapeArrayAndDataArray(
            shape,
            winrt::array_view<const bool>(out.get(), out.get() + data.size()));
    }

    template <TensorKind T>
    auto CreateVectorOverData(
        std::vector<typename TensorKindToPointerType<T>::Type>& data)
    {
        using DataType = typename TensorKindToPointerType<T>::Type;
        return winrt::single_threaded_vector<DataType>(std::move(data));
    }

    //
    // CreateVectorOverData<T>
    // Specialization for boolean to convert to single_threaded_vector<bool>, but test data is in std::vector<boolean>
    //
    template <>
    auto CreateVectorOverData<TensorKind::Boolean>(std::vector<boolean>& data) {
        auto vector = winrt::single_threaded_vector<bool>();
        for (auto value : data) {
          vector.Append(static_cast<bool>(value));
        }
        return vector;
    }

    template <typename T>
    winrt::com_ptr<ID3D12Resource> CreateD3D12Resource(
        const std::vector<int64_t>& shape,
        winml::LearningModelDevice device) {
      // Try to allocate the backing memory for the caller
      auto bufferSize = std::accumulate(std::begin(shape), std::end(shape), static_cast<int64_t>(1), std::multiplies<int64_t>());
      auto bufferByteSize = sizeof(T) * bufferSize;

      // DML needs the resources' sizes to be a multiple of 4 bytes
      if (bufferByteSize % 4 != 0) {
        bufferByteSize += 4 - (bufferByteSize % 4);
      }

      D3D12_HEAP_PROPERTIES heapProperties = {
          D3D12_HEAP_TYPE_DEFAULT,
          D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
          D3D12_MEMORY_POOL_UNKNOWN,
          0,
          0};
      D3D12_RESOURCE_DESC resourceDesc = {
          D3D12_RESOURCE_DIMENSION_BUFFER,
          0,
          static_cast<uint64_t>(bufferByteSize),
          1,
          1,
          1,
          DXGI_FORMAT_UNKNOWN,
          {1, 0},
          D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

      winrt::com_ptr<ID3D12Resource> gpu_resource = nullptr;
      device.as<winmlp::LearningModelDevice>()->GetD3DDevice()->CreateCommittedResource(
          &heapProperties,
          D3D12_HEAP_FLAG_NONE,
          &resourceDesc,
          D3D12_RESOURCE_STATE_COMMON,
          nullptr,
          __uuidof(ID3D12Resource),
          gpu_resource.put_void());

      return gpu_resource;
    }

    template <TensorKind T>
    static wf::IInspectable CreateTensor(
        const winml::LearningModelSession& session,
        const winml::TensorFeatureDescriptor& descriptor,
        const LearningModelBindingStrategy& strategy,
        float* metric) {
      using TensorValue = typename TensorKindToValue<T>::Type;
      using DataType = typename TensorKindToPointerType<T>::Type;
      
      // Get the true shape that the session expects...
      auto session_impl = session.as<winmlp::LearningModelSession>();
      auto shape = session_impl->GetShapeOfInputOutput(descriptor.Name());

      // TODO: Temporary workaround to get shape from model, but adapter api should be added here...
      auto descriptor_shape = descriptor.Shape();
      shape = std::vector<int64_t>(begin(descriptor_shape), end(descriptor_shape));

      // There should never be free dimensions. If there are they should be overridden with NamedDimensionOverrides
      auto found_freedim_it = std::find(shape.begin(), shape.end(), -1);
      if (found_freedim_it != shape.end()) {
        throw; // Throw error about how all dims should be name overridden or batch size overridden.
      }

      // Calulate the size of the input tensor
      auto size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

      // Get tensor factory
      auto tensor_factory = winrt::get_activation_factory<TensorValue, ITensorStaticsNative>();

      // Create the device specific resources
      std::vector<DataType> data;
      winrt::com_ptr<ID3D12Resource> resource = nullptr;
      if (strategy == LearningModelBindingStrategy::CreateFromD3D12Resource) {
        resource = CreateD3D12Resource<DataType>(shape, session.Device());
      } else {
        data = std::vector<DataType>(size);
      }

      auto start = std::chrono::high_resolution_clock::now();
      wf::IInspectable feature_value = nullptr;
      switch (strategy)
      {
          case LearningModelBindingStrategy::CreateWithZeroCopyITensorNative:
          {
              auto tensorValue = TensorValue::Create(shape);

              winrt::com_ptr<ITensorNative> spTensorValueNative;
              tensorValue.as(spTensorValueNative);

              BYTE* actualData;
              uint32_t actualSizeInBytes;
              spTensorValueNative->GetBuffer(&actualData, &actualSizeInBytes);

              // Copy into the memory buffer
              memcpy(actualData, reinterpret_cast<BYTE*>(data.data()), data.size() * sizeof(DataType));

              feature_value = tensorValue;
              break;
          }

          case LearningModelBindingStrategy::CreateWithZeroCopyIMemoryBuffer:
          {
              auto tensorValue = TensorValue::Create(shape);

              if (auto memoryBuffer = tensorValue.try_as<wf::IMemoryBuffer>())
              {
                  auto memoryBufferReference = memoryBuffer.CreateReference();
                  auto memoryBufferByteAccess = memoryBufferReference.as<::Windows::Foundation::IMemoryBufferByteAccess>();

                  BYTE* actualData;
                  uint32_t actualSizeInBytes;
                  memoryBufferByteAccess->GetBuffer(&actualData, &actualSizeInBytes);

                  // Copy into the memory buffer
                  memcpy(actualData, reinterpret_cast<BYTE*>(data.data()), data.size() * sizeof(DataType));
              }

              feature_value = tensorValue;
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray:
          {
              feature_value = CreateFromShapeArrayAndDataArray<T>(shape, data);
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeIterableAndDataArray:
          {
              feature_value = CreateFromArray<T>(shape, data);
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeIterableAndDataIterable:
          {
              feature_value = TensorValue::CreateFromIterable(shape, CreateVectorOverData<T>(data).GetView());
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeIterableAndDataIterableRaw:
          {
              feature_value = CreateVectorOverData<T>(data);
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeIterableAndDataIterableRawView:
          {
              feature_value = CreateVectorOverData<T>(data).GetView();
              break;
          }

          case LearningModelBindingStrategy::CreateFromShapeArrayAndDataBuffer:
          {
              feature_value = CreateFromShapeArrayAndDataBuffer<T>(shape, data);
              break;
          }

          case LearningModelBindingStrategy::CreateFromD3D12Resource:
          {
            winrt::com_ptr<IUnknown> unknown_tensor;
            tensor_factory->CreateFromD3D12Resource(resource.get(),
                                                    shape.data(),
                                                    static_cast<int>(shape.size()),
                                                    unknown_tensor.put());
            feature_value = unknown_tensor.as<wf::IInspectable>();
            break;
          }
      }

      // calculate the duration as the metric
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_ms = end - start;
      *metric = duration_ms.count();

      return feature_value;
    }

    static wf::IInspectable CreateTensorInput(
        const winml::LearningModelSession& session,
        const winml::TensorFeatureDescriptor& descriptor,
        const LearningModelBindingStrategy& strategy,
        float* metric)
    {
        // Create input
        switch (descriptor.TensorKind())
        {
            case winml::TensorKind::Float:   return CreateTensor<winml::TensorKind::Float>(session, descriptor, strategy, metric);
            case winml::TensorKind::UInt8:   return CreateTensor<winml::TensorKind::UInt8>(session, descriptor, strategy, metric);
            case winml::TensorKind::Int8:    return CreateTensor<winml::TensorKind::Int8>(session, descriptor, strategy, metric);
            case winml::TensorKind::UInt16:  return CreateTensor<winml::TensorKind::UInt16>(session, descriptor, strategy, metric);
            case winml::TensorKind::Int16:   return CreateTensor<winml::TensorKind::Int16>(session, descriptor, strategy, metric);
            case winml::TensorKind::Int32:   return CreateTensor<winml::TensorKind::Int32>(session, descriptor, strategy, metric);
            case winml::TensorKind::Int64:   return CreateTensor<winml::TensorKind::Int64>(session, descriptor, strategy, metric);
            case winml::TensorKind::String:  return CreateTensor<winml::TensorKind::String>(session, descriptor, strategy, metric);
            case winml::TensorKind::Boolean: return CreateTensor<winml::TensorKind::Boolean>(session, descriptor, strategy, metric);
            case winml::TensorKind::Float16: return CreateTensor<winml::TensorKind::Float16>(session, descriptor, strategy, metric);
            case winml::TensorKind::Double:  return CreateTensor<winml::TensorKind::Double>(session, descriptor, strategy, metric);
            case winml::TensorKind::UInt32:  return CreateTensor<winml::TensorKind::UInt32>(session, descriptor, strategy, metric);
            case winml::TensorKind::UInt64:  return CreateTensor<winml::TensorKind::UInt64>(session, descriptor, strategy, metric);
            default:
                return nullptr;
        }
    }
    
    static void GetAsVectorView(wf::IInspectable inspectable_value)
    {
        auto tensor_value = inspectable_value.try_as<winml::ITensor>();

        if (tensor_value == nullptr) {
            // it must be an vector view directly bound, no need to convert to a vector view...
            return;
        }

        // Create input
        switch (tensor_value.TensorKind())
        {
            case winml::TensorKind::Float:   { tensor_value.as<winml::TensorFloat>().GetAsVectorView();      break; }
            case winml::TensorKind::UInt8:   { tensor_value.as<winml::TensorUInt8Bit>().GetAsVectorView();   break; }
            case winml::TensorKind::Int8:    { tensor_value.as<winml::TensorInt8Bit>().GetAsVectorView();    break; }
            case winml::TensorKind::UInt16:  { tensor_value.as<winml::TensorUInt16Bit>().GetAsVectorView();  break; }
            case winml::TensorKind::Int16:   { tensor_value.as<winml::TensorInt16Bit>().GetAsVectorView();   break; }
            case winml::TensorKind::Int32:   { tensor_value.as<winml::TensorInt32Bit>().GetAsVectorView();   break; }
            case winml::TensorKind::Int64:   { tensor_value.as<winml::TensorInt64Bit>().GetAsVectorView();   break; }
            case winml::TensorKind::String:  { tensor_value.as<winml::TensorString>().GetAsVectorView();     break; }
            case winml::TensorKind::Boolean: { tensor_value.as<winml::TensorBoolean>().GetAsVectorView();    break; }
            case winml::TensorKind::Float16: { tensor_value.as<winml::TensorFloat16Bit>().GetAsVectorView(); break; }
            case winml::TensorKind::Double:  { tensor_value.as<winml::TensorDouble>().GetAsVectorView();     break; }
            case winml::TensorKind::UInt32:  { tensor_value.as<winml::TensorUInt32Bit>().GetAsVectorView();  break; }
            case winml::TensorKind::UInt64:  { tensor_value.as<winml::TensorUInt64Bit>().GetAsVectorView();  break; }
        }
        return;
    }

    static wf::IInspectable CreateFeatureValue(
        const winml::LearningModelSession& sesison,
        winml::ILearningModelFeatureDescriptor input,
        winml_experimental::LearningModelBindingStrategy input_strategy,
        float* metric) {

        switch (input.Kind()) {
            case winml::LearningModelFeatureKind::Tensor:
            return CreateTensorInput(sesison, input.as<winml::TensorFeatureDescriptor>(), input_strategy, metric);
              break;
            case winml::LearningModelFeatureKind::Image:
              break;
            case winml::LearningModelFeatureKind::Map:
              break;
            case winml::LearningModelFeatureKind::Sequence:
              break;
        }

        return nullptr;
    }

    static float MeasureBind(const winml::LearningModelBinding& binding, const winrt::hstring& name, wf::IInspectable value) {
      auto start = std::chrono::high_resolution_clock::now();
      binding.Bind(name, value);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_ms = end - start;
      return duration_ms.count();
    }

    static winml::LearningModel MeasureModelLoad(
        const winrt::hstring& path,
        float* metric) {
      *metric = 0;
      auto start = std::chrono::high_resolution_clock::now();
      auto model = winml::LearningModel::LoadFromFilePath(path);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_ms = end - start;
      *metric = duration_ms.count();
      return model;
    }

    static winml::LearningModelEvaluationResult MeasureEvaluate(
        winml::LearningModelSession& session,
        const winml::LearningModelBinding& binding,
        float* metric) {
      *metric = 0;
      auto start = std::chrono::high_resolution_clock::now();
      auto result = session.Evaluate(binding, L"");
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_ms = end - start;
      *metric = duration_ms.count();
      return result;
    }

    static winml::LearningModelSession MeasureSessionCreate(
        const winml::LearningModel& model,
        const winml::LearningModelDevice& learning_model_device,
        winml_experimental::LearningModelEnumerateInferenceStrategiesOptions const& options,
        uint32_t batch_size,
        float* metric) {
      *metric = 0;
      auto start = std::chrono::high_resolution_clock::now();
      auto options_impl = options.as<winml_experimentalp::LearningModelEnumerateInferenceStrategiesOptions>();

      auto learning_model_session_options = winml::LearningModelSessionOptions();

      for (auto override_pair : options_impl->NamedDimensionOverrides()) {
        learning_model_session_options.OverrideNamedDimension(winrt::to_hstring(override_pair.first), override_pair.second);
      }

      learning_model_session_options.BatchSizeOverride(batch_size);
      auto learning_model_session = winml::LearningModelSession(model, learning_model_device, learning_model_session_options);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> duration_ms = end - start;
      *metric = duration_ms.count();
      return learning_model_session;
    }

    static bool MeasureEvaluateOne(
        winml::LearningModelSession session,
        winml_experimental::LearningModelBindingStrategy input_strategy,
        winml_experimental::LearningModelBindingStrategy output_strategy,
        winml_experimental::LearningModelReadMode read_mode,
        winml_experimental::LearningModelBindMode bind_mode,
        float* bind_inputs_metric,
        float* bind_outputs_metric,
        float* evaluate_metric,
        float* read_outputs_metric)
    {
        *bind_inputs_metric = 0;
        *bind_outputs_metric = 0;
        *evaluate_metric = 0;
        *read_outputs_metric = 0;

        // Create binding
        winml::LearningModelBinding binding(session);

        if (session.Device().Direct3D11Device() == nullptr &&
            (input_strategy == winml_experimental::LearningModelBindingStrategy::CreateFromD3D12Resource ||
             output_strategy == winml_experimental::LearningModelBindingStrategy::CreateFromD3D12Resource)) {
            // Cannot bind D3D12Resource, when the session does not have a device...
            return nullptr;
        }

        // Create inputs
        for (const auto& input : session.Model().InputFeatures()) {
            float bind_metric;
            auto value = CreateFeatureValue(session, input, input_strategy, &bind_metric);
            if (value == nullptr) {
                // skip...
                return false;
            }
            *bind_inputs_metric += bind_metric;
            *bind_inputs_metric += MeasureBind(binding, input.Name(), value);
        }

        if (bind_mode == winml_experimental::LearningModelBindMode::Bound) {
            for (const auto& output : session.Model().OutputFeatures()) {
              float bind_metric;
              auto value = CreateFeatureValue(session, output, output_strategy, &bind_metric);
              if (value == nullptr) {
                // skip...
                return false;
              }
              *bind_outputs_metric += bind_metric;
              *bind_outputs_metric += MeasureBind(binding, output.Name(), value);
            }
        }

        // Evaluate
        auto learning_model_evaluation_result = MeasureEvaluate(session, binding, evaluate_metric);

        // Read
        for (const auto& pair : learning_model_evaluation_result.Outputs()) {
            float read_metric = 0;
            switch (read_mode)
            {
                case winml_experimental::LearningModelReadMode::GetAsVectorView:
                {
                    GetAsVectorView(pair.Value());
                    break;
                }

                case winml_experimental::LearningModelReadMode::GetFromNativeBufferAccess:
                {
                    if (auto tensor_value = pair.Value().try_as<winml::ITensor>()) {
                        // Get as ITensorNative
                        auto spTensorValueNative = tensor_value.as<ITensorNative>();

                        // Get buffer
                        uint32_t actual_size_in_bytes;
                        BYTE* actual_data;
                        spTensorValueNative->GetBuffer(&actual_data, &actual_size_in_bytes);
                    }
                    break;
                }

                case winml_experimental::LearningModelReadMode::GetFromMemoryBufferReferenceAccess:
                {
                    if (auto tensor_value = pair.Value().try_as<winml::ITensor>()) {
                        // Get as ITensorNative
                        if (auto memoryBuffer = tensor_value.as<winrt::Windows::Foundation::IMemoryBuffer>())
                        {
                            auto memoryBufferReference = memoryBuffer.CreateReference();
                            auto memoryBufferByteAccess = memoryBufferReference.as<::Windows::Foundation::IMemoryBufferByteAccess>();

                            uint32_t actual_size_in_bytes;
                            BYTE* actual_data;
                            memoryBufferByteAccess->GetBuffer(&actual_data, &actual_size_in_bytes);
                        }
                    }
                    break;
                }
            }
            *read_outputs_metric += read_metric;
        }
        return true;
    }

    static wfc::IVectorView<winml_experimental::LearningModelInferenceStrategy> EnumerateStrategies(
        winrt::hstring const& path,
        winml_experimental::LearningModelEnumerateInferenceStrategiesOptions const& options,
        std::function<void(winml_experimental::EnumerateInferenceStrategiesProgress)> update_progress,
        std::function<bool()> is_cancelled)
    {
        uint32_t unused;
        const auto include_model_load = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::LoadModel, unused);
        const auto include_session_create = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::CreateSession, unused);
        const auto include_evaluate = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::Evaluate, unused);
        const auto include_bind_inputs = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::BindInputs, unused);
        const auto include_bind_outputs = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::BindOutputs, unused);
        const auto include_fetch_results = options.PhaseFilter().IndexOf(winml_experimental::LearningModelPhase::FetchResults, unused);

        // Create the output strategies
        auto strategies = std::vector<winml_experimental::LearningModelInferenceStrategy>();

        // load the model
        float model_load_metric;
        auto learning_model = MeasureModelLoad(path, &model_load_metric);
        auto learning_model_impl = learning_model.as<winmlp::LearningModel>();

        // Define a status struct
        winml_experimental::EnumerateInferenceStrategiesProgress strategies_progress = {};

        // Check if batching is supported by the model...
        auto batch_size_filter = options.BatchingStrategyFilter();
        if (learning_model_impl->IsBatchingSupported() == false) {
          batch_size_filter = winrt::make<winml_experimentalp::LearningModelBatchingStrategyFilter>();
        }

        // Calculate total number of strategies to evaluate
        auto total_devices = options.DeviceFilter().Size() *
                             batch_size_filter.Size() *
                             options.InputStrategyFilter().Size() *
                             options.OutputStrategyFilter().Size() *
                             options.OutputReadModeFilter().Size() *
                             options.BindModeFilter().Size();

        // If async, notify initial progress
        if (update_progress != nullptr) {

          strategies_progress.StrategiesEvaluated = 0;
          strategies_progress.TotalNumberOfStrategies = total_devices;
          update_progress(strategies_progress);
        }

        for (const auto& device : options.DeviceFilter()) {
            auto learning_model_device = winml::LearningModelDevice(device);

            for (const auto& batch_size : batch_size_filter) {
                float session_create_metric;
                auto learning_model_session = MeasureSessionCreate(learning_model,
                                                                   learning_model_device,
                                                                   options,
                                                                   batch_size,
                                                                   &session_create_metric);

                // Throw away first run, dont measure
                float bind_inputs_metric;
                float bind_outputs_metric;
                float evaluate_metric;
                float fetch_results_metric;
                MeasureEvaluateOne(
                    learning_model_session,
                    winml_experimental::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray,
                    winml_experimental::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray,
                    winml_experimental::LearningModelReadMode::GetAsVectorView,
                    winml_experimental::LearningModelBindMode::Bound,
                    &bind_inputs_metric,
                    &bind_outputs_metric, 
                    &evaluate_metric, 
                    &fetch_results_metric);

                for (const auto& input_strategy : options.InputStrategyFilter()) {
                    for (const auto& output_strategy : options.OutputStrategyFilter()) {
                        for (const auto& output_read_mode : options.OutputReadModeFilter()) {
                            for (const auto& bind_mode : options.BindModeFilter()) {
                                // Measure evaluation
                                auto succeeded = MeasureEvaluateOne(learning_model_session,
                                                                    input_strategy,
                                                                    output_strategy,
                                                                    output_read_mode,
                                                                    bind_mode,
                                                                    &bind_inputs_metric,
                                                                    &bind_outputs_metric,
                                                                    &evaluate_metric,
                                                                    &fetch_results_metric);
                                if (succeeded) {
                                    float total_metric = 0;
                                    float batch_size_float = (batch_size == 0) ? 1.f : static_cast<float>(batch_size);
                                    if (include_model_load) {
                                      total_metric += model_load_metric;
                                    }
                                    if (include_evaluate) {
                                      total_metric += (evaluate_metric / batch_size_float);
                                    }
                                    if (include_session_create) {
                                      total_metric += session_create_metric;
                                    }
                                    if (include_bind_inputs) {
                                      total_metric += (bind_inputs_metric / batch_size_float);
                                    }
                                    if (include_bind_outputs) {
                                      total_metric += (bind_outputs_metric / batch_size_float);
                                    }
                                    if (include_fetch_results) {
                                      total_metric += (fetch_results_metric / batch_size_float);
                                    }

                                    auto strategy = winrt::make<winml_experimentalp::LearningModelInferenceStrategy>(
                                        device, input_strategy, output_strategy, output_read_mode, bind_mode, batch_size, total_metric);
                                    strategies.push_back(std::move(strategy));
                                }
                                ++strategies_progress.StrategiesEvaluated;
                                if (update_progress != nullptr) {
                                    update_progress(strategies_progress);
                                }

                                if (is_cancelled && is_cancelled()) {
                                  // return an empty array if cancelled
                                  return winrt::single_threaded_vector<winml_experimental::LearningModelInferenceStrategy>().GetView();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort strategies based on result characteristics
        std::sort(strategies.begin(), strategies.end(),
                  [](auto a, auto b) {
                    return a.Metric() < b.Metric();
                  });

        auto out_strategies = winrt::single_threaded_vector<winml_experimental::LearningModelInferenceStrategy>(std::move(strategies));
        return out_strategies.GetView();
    }

    
    wfc::IVectorView<winml_experimental::LearningModelInferenceStrategy> LearningModelInferenceStrategyEnumerator::EnumerateInferenceStrategies(
        winrt::hstring const& path,
        winml_experimental::LearningModelEnumerateInferenceStrategiesOptions const& options) {
        return EnumerateStrategies(path, options, nullptr, nullptr);
    }

    wf::IAsyncOperationWithProgress<wfc::IVectorView<winml_experimental::LearningModelInferenceStrategy>, winml_experimental::EnumerateInferenceStrategiesProgress>
    LearningModelInferenceStrategyEnumerator::EnumerateInferenceStrategiesAsync(
        winrt::hstring path,
        winml_experimental::LearningModelEnumerateInferenceStrategiesOptions options) {
      auto progress{co_await winrt::get_progress_token()};
      auto cancellation_token{co_await winrt::get_cancellation_token()};

      co_await winrt::resume_background();

      auto update_progress = std::function<void(winml_experimental::EnumerateInferenceStrategiesProgress)>(
          [&](winml_experimental::EnumerateInferenceStrategiesProgress strategies_progress) {
            progress(strategies_progress);
          });

      
      auto is_cancelled = std::function<bool()>(
          [&]() {
            return cancellation_token();
          });

      co_return EnumerateStrategies(path, options, update_progress, is_cancelled);
    }
}
