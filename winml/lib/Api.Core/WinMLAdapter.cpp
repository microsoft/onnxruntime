// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "inc/WinMLAdapter.h"
#include "inc/CustomRegistryHelper.h"
#include "PheonixSingleton.h"
#include "inc/LotusEnvironment.h"
#include "inc/AbiCustomRegistryImpl.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"

#include "LearningModelDevice.h"
#include "TensorFeatureDescriptor.h"
#include "ImageFeatureDescriptor.h"
#include "api.image/inc/D3DDeviceCache.h"

#include "PheonixSingleton.h"

#include "DmlOrtSessionBuilder.h"
#include "CpuOrtSessionBuilder.h"

#include <io.h>
#include <fcntl.h>

#include "ZeroCopyInputStreamWrapper.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "FeatureDescriptorFactory.h"


using namespace winrt::Windows::AI::MachineLearning;

namespace Windows::AI::MachineLearning::Adapter {

// ORT intentionally requires callers derive from their session class to access
// the protected Load method used below.
class InferenceSessionProtectedLoadAccessor : public onnxruntime::InferenceSession {
 public:
  onnxruntime::common::Status
  Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) {
    return onnxruntime::InferenceSession::Load(std::move(p_model_proto));
  }
};



// class AbiSafeTensor
//
class AbiSafeTensor : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    ITensor> {
private:
    onnxruntime::Tensor& tensor_;   // weak ref
    ComPtr<IOrtValue> value_;       // strong ref

public:

    AbiSafeTensor(onnxruntime::Tensor* tensor,
        IOrtValue * value_in) : tensor_(*tensor), value_(value_in) {
    }
    const onnxruntime::Tensor& STDMETHODCALLTYPE get() override {
        return tensor_;
    }
    onnxruntime::Tensor* STDMETHODCALLTYPE getMutable() override {
        return &tensor_;
    }
    onnxruntime::MLDataType STDMETHODCALLTYPE DataType() override {
        return tensor_.DataType();
    }
    const void* STDMETHODCALLTYPE DataRaw() override {
        return tensor_.DataRaw();
    }
    const std::vector<int64_t>& STDMETHODCALLTYPE ShapeGetDims() override {
        return tensor_.Shape().GetDims();
    }
    int64_t STDMETHODCALLTYPE ShapeSize() override {
        return tensor_.Shape().Size();
    }
    const char * STDMETHODCALLTYPE LocationName() override {
        return tensor_.Location().name;
    }
    OrtMemType STDMETHODCALLTYPE LocationMemType() override {
        return tensor_.Location().mem_type;
    }
};

// class OrtValue
//
class AbiSafeOrtValue : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IOrtValue> {
private:
    OrtValue ort_value_;

public:
    AbiSafeOrtValue() {}
    AbiSafeOrtValue(OrtValue value_in) : ort_value_(value_in) {}

    OrtValue& STDMETHODCALLTYPE get() override {
        return ort_value_;
    }

    onnxruntime::MLDataType STDMETHODCALLTYPE Type() override {
        return ort_value_.Type();
    }
    bool STDMETHODCALLTYPE IsTensor() override {
        return ort_value_.IsTensor();
    }
    // end
    HRESULT STDMETHODCALLTYPE GetTensor(ITensor ** tensor) override {
        auto tensor_inner = ort_value_.GetMutable<onnxruntime::Tensor>();
        auto tensor_outer = wil::MakeOrThrow<AbiSafeTensor>(tensor_inner, this);
        *tensor = tensor_outer.Detach();
        return S_OK;
    }
};  // class AbiSafeOrtValue

class ModelProto : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IModelProto> {
public:

    ModelProto::ModelProto(onnx::ModelProto* model_proto) : model_proto_(model_proto) {

    }

    onnx::ModelProto* STDMETHODCALLTYPE get() override {
        return model_proto_.get();
    }

private:
    std::shared_ptr<onnx::ModelProto> model_proto_;
}; // class ModelProto


class ModelInfo : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IModelInfo> {

private:
    std::string author_;
    std::string name_;
    std::string domain_;
    std::string description_;
    int64_t version_;
    std::unordered_map<std::string, std::string> model_metadata_;
    wfc::IVector<winml::ILearningModelFeatureDescriptor> input_features_;
    wfc::IVector<winml::ILearningModelFeatureDescriptor> output_features_;

public:

    ModelInfo(const onnx::ModelProto* model_proto) {
        Initialize(model_proto);
    }

    std::string STDMETHODCALLTYPE author() override {
        return author_;
    }
    std::string STDMETHODCALLTYPE name() override {
        return name_;
    }
    std::string STDMETHODCALLTYPE domain() override {
        return domain_;
    }
    std::string STDMETHODCALLTYPE description() override {
        return description_;
    }
    int64_t STDMETHODCALLTYPE version()  override {
        return version_;
    }
    std::unordered_map<std::string, std::string> STDMETHODCALLTYPE model_metadata()  override {
        return model_metadata_;
    }
    wfc::IVector<winml::ILearningModelFeatureDescriptor> STDMETHODCALLTYPE input_features()  override {
        return input_features_;
    }
    wfc::IVector<winml::ILearningModelFeatureDescriptor> STDMETHODCALLTYPE output_features()  override {
        return output_features_;
    }

    static std::vector<const char*>
        GetAllNodeOutputs(const onnx::ModelProto& model_proto) {
        std::vector<const char*> nodes_outputs;
        auto& graph = model_proto.graph();
        auto& nodes = graph.node();
        for (auto& node : nodes) {
            for (auto& node_output : node.output()) {
                nodes_outputs.push_back(node_output.c_str());
            }
        }
        return nodes_outputs;
    }

    static std::vector<const char*>
        GetInitializers(const onnx::ModelProto& model_proto) {
        std::vector<const char*> initializers;
        auto& graph = model_proto.graph();
        auto& graph_initializers = graph.initializer();
        for (auto& initializer : graph_initializers) {
            initializers.push_back(initializer.name().c_str());
        }
        return initializers;
    }

    static std::vector<const onnx::ValueInfoProto*>
        GetInputsWithoutInitializers(const onnx::ModelProto& model_proto) {
        auto initializers = GetInitializers(model_proto);

        std::vector<const onnx::ValueInfoProto*> inputs_without_initializers;
        auto& graph = model_proto.graph();
        auto& inputs = graph.input();
        for (auto& input : inputs) {
            if (input.has_name() && input.has_type()) {
                auto found_it = std::find_if(
                    std::begin(initializers),
                    std::end(initializers),
                    [&](auto& initializer) {
                    return std::strcmp(initializer, input.name().c_str()) == 0;
                });

                auto is_initializer = found_it != std::end(initializers);
                if (!is_initializer) {
                    inputs_without_initializers.push_back(&input);
                }
            }
        }
        return inputs_without_initializers;
    }

    static 
    std::vector<const onnx::ValueInfoProto*> GetOutputs(const onnx::ModelProto& model_proto) {
        std::vector<const onnx::ValueInfoProto*> outputs_with_name;
        auto& graph = model_proto.graph();
        auto& outputs = graph.output();
        for (auto& output : outputs) {
            if (output.has_name() && output.has_type()) {
                outputs_with_name.push_back(&output);
            }
        }
        return outputs_with_name;
    }

private:
    void Initialize(const onnx::ModelProto* model_proto) {
        // metadata
        for (auto& prop : model_proto->metadata_props()) {
            model_metadata_[prop.key()] = prop.value();
        }

        WinML::FeatureDescriptorFactory builder(model_metadata_);

        // Create inputs
        auto inputs = GetInputsWithoutInitializers(*model_proto);
        input_features_ = builder.CreateDescriptorsFromValueInfoProtos(inputs);

        // Create outputs
        auto outputs = GetOutputs(*model_proto);
        output_features_ = builder.CreateDescriptorsFromValueInfoProtos(outputs);

        // author
        auto has_producer_name = model_proto->has_producer_name();
        author_ = has_producer_name
            ? model_proto->producer_name()
            : "";

        // domain
        auto has_domain = model_proto->has_domain();
        domain_ = has_domain
            ? model_proto->domain()
            : "";

        // name
        auto has_graph = model_proto->has_graph();
        auto graph_has_name = model_proto->graph().has_name();
        auto is_name_available = has_graph && graph_has_name;
        name_ = is_name_available
            ? model_proto->graph().name()
            : "";

        // description
        auto has_description = model_proto->has_doc_string();
        description_ = has_description
            ? model_proto->doc_string()
            : "";

        // version
        auto has_version = model_proto->has_model_version();
        version_ = has_version
            ? model_proto->model_version()
            : 0;
    }
};  // class ModelInfo 

class WinMLAdapter : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IWinMLAdapter> {
private:
    std::shared_ptr<WinML::LotusEnvironment> lotus_environment_;

public:
    WinMLAdapter() : lotus_environment_(PheonixSingleton<WinML::LotusEnvironment>()) {

    }

    // factory methods for creating an ort model from a path
    HRESULT STDMETHODCALLTYPE CreateModelProto(
            const char* path,
            IModelProto** model_proto) override {
        int file_descriptor;
        _sopen_s(
            &file_descriptor,
            path,
            O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
            _SH_DENYWR,
            _S_IREAD | _S_IWRITE);

        THROW_HR_IF_MSG(
            E_FAIL,
            0 > file_descriptor,
            "Failed");  //errno

        auto stream = google::protobuf::io::FileInputStream(file_descriptor);
        stream.SetCloseOnDelete(true);

        auto model_proto_inner = new onnx::ModelProto();
        THROW_HR_IF_MSG(
            E_INVALIDARG,
            model_proto_inner->ParseFromZeroCopyStream(&stream) == false,
            "The stream failed to parse.");

        auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner);
        return model_proto_outer.CopyTo(__uuidof(IModelProto), (void**)model_proto);
    }

    // factory methods for creating an ort model from a stream
    HRESULT STDMETHODCALLTYPE CreateModelProto(        
        ABI::Windows::Storage::Streams::IRandomAccessStreamReference* stream_reference,
        IModelProto** model_proto) override {

        ZeroCopyInputStreamWrapper wrapper(stream_reference);

        auto model_proto_inner = new onnx::ModelProto();
        THROW_HR_IF_MSG(
            E_INVALIDARG,
            model_proto_inner->ParseFromZeroCopyStream(&wrapper) == false,
            "The stream failed to parse.");

        auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner);
        return model_proto_outer.CopyTo(__uuidof(IModelProto), (void**)model_proto);
    }

    // factory methods for creating an ort model from a model_proto
    HRESULT STDMETHODCALLTYPE CreateModelProto(IModelProto * model_proto_in, IModelProto** model_proto) override {
        auto model_proto_inner = new onnx::ModelProto(*model_proto_in->get());
        auto model_proto_outer = wil::MakeOrThrow<ModelProto>(model_proto_inner);
        return model_proto_outer.CopyTo(__uuidof(IModelProto), (void**)model_proto);
    }

    HRESULT STDMETHODCALLTYPE CreateModelInfo(IModelProto * model_proto, IModelInfo ** model_info) override {
        auto model_info_outer = wil::MakeOrThrow<ModelInfo>(model_proto->get());
        return model_info_outer.CopyTo(__uuidof(IModelInfo), (void**)model_info);
    }


    void STDMETHODCALLTYPE EnableDebugOutput() override {
        WinML::CWinMLLogSink::EnableDebugOutput();
    }

    static bool IsFeatureDescriptorFp16(
            winml::ILearningModelFeatureDescriptor descriptor) {
        if (auto imageFeatureDescriptor = descriptor.try_as<winml::IImageFeatureDescriptor2>()) {
            return TensorKind::Float16 == imageFeatureDescriptor.TensorKind();
        }

        if (auto tensorFeatureDescriptor = descriptor.try_as<winml::ITensorFeatureDescriptor>()) {
            return TensorKind::Float16 == tensorFeatureDescriptor.TensorKind();
        }

        return false;
    }

    HRESULT STDMETHODCALLTYPE EnsureModelDeviceCompatibility(
        winml::LearningModel const& model,
        IModelProto* p_model_proto,
        bool is_float16_supported) override {
        if (!is_float16_supported) {
            auto& graph = p_model_proto->get()->graph();

            // The model will not contain fp16 operations if:
            // 1. The model has no fp16 inputs
            // 2. The model has no fp16 initializers
            // 3. The model does not create any fp16 intermediary tensors via the Cast (to float16) operator
            // 4. The model does not have any fp16 outputs

            // 1. Ensure that The model has no fp16 inputs
            for (auto descriptor : model.InputFeatures()) {
                THROW_HR_IF_MSG(
                    DXGI_ERROR_UNSUPPORTED,
                    IsFeatureDescriptorFp16(descriptor),
                    "The model contains a 16-bit input (%ls), but the current device does not support 16-bit float.",
                    descriptor.Name().c_str());
            }

            // 2. Ensure that the model has no fp16 initializers
            for (int i = 0; i < graph.node_size(); i++) {
                auto node = graph.node(i);
                if (node.op_type() == "Cast" && node.domain().empty()) {
                    for (int attribIndex = 0; attribIndex < node.attribute_size(); attribIndex++) {
                        auto attribute = node.attribute(attribIndex);
                        if (attribute.name() == "to") {
                            THROW_HR_IF_MSG(
                                DXGI_ERROR_UNSUPPORTED,
                                attribute.i() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
                                "The model contains a 16-bit float Cast Op (%s), but the current device does not support 16-bit float.",
                                node.name().c_str());
                        }
                    }
                }
            }

            // 3. Ensure that the model does not create any fp16 intermediary
            //    tensors via the Cast (to float16) operator
            for (int i = 0; i < graph.initializer_size(); i++) {
                auto initializer = graph.initializer(i);

                THROW_HR_IF_MSG(
                    DXGI_ERROR_UNSUPPORTED,
                    initializer.data_type() == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16,
                    "The model contains a 16-bit float initializer (%s), but the current device does not support 16-bit float.",
                    initializer.name().c_str());
            }

            // 4. Ensure that the model does not have any fp16 outputs
            for (auto descriptor : model.OutputFeatures()) {
                THROW_HR_IF_MSG(
                    DXGI_ERROR_UNSUPPORTED,
                    IsFeatureDescriptorFp16(descriptor),
                    "The model contains a 16-bit output (%ls), but the current device does not support 16-bit float.",
                    descriptor.Name().c_str());
            }
        }
        return S_OK;
    }

    ID3D12Resource* STDMETHODCALLTYPE GetD3D12ResourceFromAllocation(onnxruntime::IExecutionProvider* provider, void* allocation) override {
        auto d3dResource =
            Dml::GetD3D12ResourceFromAllocation(
                provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault).get(),
                allocation);
        return d3dResource;
    }

    static onnxruntime::MLDataType GetType(winml::TensorKind kind) {
        switch (kind) {
        case winml::TensorKind::Float:
            return onnxruntime::DataTypeImpl::GetType<float>();
        case winml::TensorKind::Float16:
            return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
        };
        return nullptr;
    }

    // factory method for creating an ortsessionbuilder from a device
    HRESULT STDMETHODCALLTYPE CreateOrtSessionBuilder(
            ID3D12Device* device,
            ID3D12CommandQueue* queue,
            IOrtSessionBuilder** session_builder) override {

        if (device == nullptr) {
            auto builder = wil::MakeOrThrow<CpuOrtSessionBuilder>();
            return builder.CopyTo(__uuidof(IOrtSessionBuilder), (void**)session_builder);
        } else {
            auto builder = wil::MakeOrThrow<DmlOrtSessionBuilder>(device, queue);
            return builder.CopyTo(__uuidof(IOrtSessionBuilder), (void**)session_builder);
        }
    }

    onnxruntime::MLDataType STDMETHODCALLTYPE GetTensorType() override {
        return onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>();
    }

    onnxruntime::MLDataType STDMETHODCALLTYPE GetTensorType(winml::TensorKind kind) override {
        if (kind == TensorKind::Float) {
            return onnxruntime::DataTypeImpl::GetType<float>();
        }
        else if (kind == TensorKind::Double) {
            return onnxruntime::DataTypeImpl::GetType<double>();
        }
        else if (kind == TensorKind::String) {
            return onnxruntime::DataTypeImpl::GetType<std::string>();
        }
        else if (kind == TensorKind::UInt8) {
            return onnxruntime::DataTypeImpl::GetType<uint8_t>();
        }
        else if (kind == TensorKind::Int8) {
            return onnxruntime::DataTypeImpl::GetType<int8_t>();
        }
        else if (kind == TensorKind::UInt16) {
            return onnxruntime::DataTypeImpl::GetType<uint16_t>();
        }
        else if (kind == TensorKind::Int16) {
            return onnxruntime::DataTypeImpl::GetType<int16_t>();
        }
        else if (kind == TensorKind::UInt32) {
            return onnxruntime::DataTypeImpl::GetType<uint32_t>();
        }
        else if (kind == TensorKind::Int32) {
            return onnxruntime::DataTypeImpl::GetType<int32_t>();
        }
        else if (kind == TensorKind::UInt64) {
            return onnxruntime::DataTypeImpl::GetType<uint64_t>();
        }
        else if (kind == TensorKind::Int64) {
            return onnxruntime::DataTypeImpl::GetType<int64_t>();
        }
        else if (kind == TensorKind::Boolean) {
            return onnxruntime::DataTypeImpl::GetType<bool>();
        }
        else if (kind == TensorKind::Float16) {
            return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
        }
        return nullptr;
    }

    onnxruntime::MLDataType STDMETHODCALLTYPE GetMapType(winml::TensorKind key_kind, winml::TensorKind value_kind) override {
        if (key_kind == TensorKind::String) {
            if (value_kind == TensorKind::String) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToString>();
            }
            else if (value_kind == TensorKind::Int64) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToInt64>();
            }
            else if (value_kind == TensorKind::Float) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToFloat>();
            }
            else if (value_kind == TensorKind::Double) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapStringToDouble>();
            }
        }
        else if (key_kind == TensorKind::Int64) {
            if (value_kind == TensorKind::String) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToString>();
            }
            else if (value_kind == TensorKind::Int64) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToInt64>();
            }
            else if (value_kind == TensorKind::Float) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToFloat>();
            }
            else if (value_kind == TensorKind::Double) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::MapInt64ToDouble>();
            }
        }
        return nullptr;
    }

    onnxruntime::MLDataType STDMETHODCALLTYPE GetVectorMapType(winml::TensorKind key_kind, winml::TensorKind value_kind) override {
        if (key_kind == TensorKind::String) {
            if (value_kind == TensorKind::Float) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::VectorMapStringToFloat>();
            }
        }
        else if (key_kind == TensorKind::Int64) {
            if (value_kind == TensorKind::Float) {
                return onnxruntime::DataTypeImpl::GetType<onnxruntime::VectorMapInt64ToFloat>();
            }
        }
        return nullptr;
    }

    // returns the raw mutable data.
    void * STDMETHODCALLTYPE GetTensorData(IOrtValue * ort_Value) override {
        return nullptr;
    }
    void * STDMETHODCALLTYPE GetMapData(IOrtValue * ort_Value, winml::TensorKind key_kind, winml::TensorKind value_kind) override {
        return nullptr;
    }
    void * STDMETHODCALLTYPE GetVectorData(IOrtValue * ort_Value, winml::TensorKind key_kind, winml::TensorKind value_kind) override {
        return nullptr;
    }

    HRESULT STDMETHODCALLTYPE GetCustomRegistry(IMLOperatorRegistry** registry) override {
        auto impl = wil::MakeOrThrow<AbiCustomRegistryImpl>();
        *registry = impl.Detach();
        return S_OK;
    }

    void* STDMETHODCALLTYPE CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource) override {
        return Dml::CreateGPUAllocationFromD3DResource(pResource);
    }

    void STDMETHODCALLTYPE FreeGPUAllocation(void* ptr) override {
        Dml::FreeGPUAllocation(ptr);
    }
    HRESULT STDMETHODCALLTYPE CopyTensor(
        onnxruntime::IExecutionProvider* provider, 
        ITensor* src, 
        ITensor* dst) override {
        ORT_THROW_IF_ERROR(Dml::CopyTensor(provider, src->get(), *(dst->getMutable())));
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateGPUMLValue(
        void * execution_provider_allocated_resource,
        onnxruntime::IExecutionProvider* provider,
        std::vector<int64_t>* shape,
        onnxruntime::MLDataType data_type,
        IOrtValue ** gpu_value) override {

        THROW_HR_IF_MSG(WINML_ERR_INVALID_BINDING,
            "DmlExecutionProvider" != provider->Type(),
            "Cannot creat GPU tensor on CPU device");

        onnxruntime::TensorShape tensor_shape(*shape);

        auto tensor = new onnxruntime::Tensor(
            data_type,
            tensor_shape,
            execution_provider_allocated_resource,
            provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault)->Info());

        auto ort_value = wil::MakeOrThrow<AbiSafeOrtValue>();
        ort_value->get().Init(tensor,
            onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
            onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

        *gpu_value = ort_value.Detach();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateCPUMLValue(
        std::vector<int64_t>* shape,
        onnxruntime::MLDataType data_type,
        onnxruntime::BufferNakedPtr buffer,
        IOrtValue ** cpu_value) override {
        auto registrations = onnxruntime::DeviceAllocatorRegistry::Instance().AllRegistrations();
        auto alloc = registrations[onnxruntime::CPU].factory(0);
       
       onnxruntime::TensorShape tensor_shape(*shape);

        // Unowned raw tensor pointer passed to engine
        auto tensor = new onnxruntime::Tensor(
            data_type,
            tensor_shape,
            buffer,
            alloc->Info());

        auto ort_value = wil::MakeOrThrow<AbiSafeOrtValue>();
        ort_value->get().Init(tensor,
            onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
            onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

        *cpu_value = ort_value.Detach();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateMLValue(
        winml::TensorKind kind,
        onnxruntime::MLDataType data_type,
        const int64_t * shape,
        uint32_t shape_count,
        onnxruntime::IExecutionProvider* provider,
        IOrtValue ** ort_value) override {
        onnxruntime::TensorShape tensor_shape(shape, shape_count);
        auto tensor = new onnxruntime::Tensor(
            GetType(kind),
            tensor_shape,
            provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault));
        auto ort_value_out = wil::MakeOrThrow<AbiSafeOrtValue>();
        ort_value_out->get().Init(tensor,
            data_type,
            data_type->GetDeleteFunc());

        *ort_value = ort_value_out.Detach();;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateOrtValue(
        void * data,
        onnxruntime::MLDataType data_type,
        IOrtValue ** ort_value) override {

        auto ort_value_out = wil::MakeOrThrow<AbiSafeOrtValue>();
        ort_value_out->get().Init(
            data,
            data_type,
            data_type->GetDeleteFunc());

        *ort_value = ort_value_out.Detach();
        return S_OK;
    }

    // Override select shape inference functions which are incomplete in ONNX with versions that are complete,
    // and are also used in DML kernel registrations.  Doing this avoids kernel and shader creation being
    // deferred until first evaluation.  It also prevents a situation where inference functions in externally
    // registered schema are reachable only after upstream schema have been revised in a later OS release,
    // which would be a compatibility risk.
    HRESULT STDMETHODCALLTYPE OverrideSchemaInferenceFunctions() override {
        static std::once_flag schema_override_once_flag;
        std::call_once(schema_override_once_flag, []() {
            SchemaInferenceOverrider::OverrideSchemaInferenceFunctions();
        });
        return S_OK;
    }


};

extern "C"
HRESULT STDMETHODCALLTYPE OrtGetWinMLAdapter(IWinMLAdapter** adapter) {
    // make an adapter instance
    Microsoft::WRL::ComPtr<WinMLAdapter> adapterptr = wil::MakeOrThrow<WinMLAdapter>();
    return adapterptr.CopyTo(__uuidof(IWinMLAdapter), (void **)adapter);
}


// class IOBinding
// ===============
class IOBinding : public Microsoft::WRL::RuntimeClass <
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IIOBinding> {
private:
    std::shared_ptr<onnxruntime::IOBinding> binding_;
    std::vector<IOrtValue*> outputs_weak;
    std::vector<ComPtr<IOrtValue>> outputs_;

public:

    IOBinding(onnxruntime::IOBinding * binding) : binding_(binding) {
    }

    onnxruntime::IOBinding* STDMETHODCALLTYPE get() override {
        return binding_.get();
    }

    HRESULT STDMETHODCALLTYPE BindInput(const std::string& name, IOrtValue* ml_value) override {
        ORT_THROW_IF_ERROR(binding_->BindInput(name, ml_value->get()));
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE BindOutput(const std::string& name, IOrtValue* ml_value) override {
        // this can be null for unbound outputs
        if (ml_value == nullptr) {
            OrtValue empty_value = {};
            ORT_THROW_IF_ERROR(binding_->BindOutput(name, empty_value));
        } else {
            ORT_THROW_IF_ERROR(binding_->BindOutput(name, ml_value->get()));
        }
        return S_OK;
    }

    const std::vector<std::string>& STDMETHODCALLTYPE GetOutputNames() override {
        return binding_->GetOutputNames();
    }
    std::vector<IOrtValue*>& STDMETHODCALLTYPE GetOutputs() override {
        auto output_inner = binding_->GetOutputs();
        outputs_.clear();
        for (unsigned i = 0; i < output_inner.size(); i++) {
            auto ort_value = wil::MakeOrThrow<AbiSafeOrtValue>(output_inner[i]);
            outputs_.push_back(ort_value);
            outputs_weak.push_back(ort_value.Get());
        }
        return outputs_weak;
    }
};

// InferenceSession
// ================

InferenceSession::InferenceSession(onnxruntime::InferenceSession * session) : session_(session) {

}

void STDMETHODCALLTYPE InferenceSession::RegisterGraphTransformers(bool registerLotusTransforms) {
    GraphTransformerHelpers::RegisterGraphTransformers(session_.get(), registerLotusTransforms);
}

HRESULT STDMETHODCALLTYPE InferenceSession::NewIOBinding(IIOBinding** io_binding) {
    std::unique_ptr<onnxruntime::IOBinding> binding;
    ORT_THROW_IF_ERROR(this->session_->NewIOBinding(&binding));
    auto io_binding_outer = wil::MakeOrThrow<IOBinding>(binding.release());
    return io_binding_outer.CopyTo(__uuidof(IIOBinding), (void**)io_binding);
}

HRESULT STDMETHODCALLTYPE InferenceSession::Run(const onnxruntime::RunOptions* run_options, IIOBinding* io_binding) {
    ORT_THROW_IF_ERROR(this->session_->Run(*run_options, *(io_binding->get())));
    return S_OK;
}
HRESULT STDMETHODCALLTYPE InferenceSession::StartProfiling() {
    this->session_->StartProfiling(PheonixSingleton<WinML::LotusEnvironment>()->GetDefaultLogger());
    return S_OK;

}
HRESULT STDMETHODCALLTYPE InferenceSession::EndProfiling() {
    this->session_->EndProfiling();
    return S_OK;

}

HRESULT STDMETHODCALLTYPE
InferenceSession::LoadModel(
        IModelProto* model_proto)  {
    auto session_protected_load_accessor =
        static_cast<InferenceSessionProtectedLoadAccessor*>(session_.get());
    std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_ptr(model_proto->get());
    ORT_THROW_IF_ERROR(session_protected_load_accessor->Load(std::move(model_proto_ptr)));
    return S_OK;
}

HRESULT STDMETHODCALLTYPE
InferenceSession::RegisterCustomRegistry(
        IMLOperatorRegistry* registry) {
    RETURN_HR_IF(S_OK, registry == nullptr);

    auto custom_registries = GetLotusCustomRegistries(registry);

    // Register
    for (auto& custom_registry : custom_registries) {
        ORT_THROW_IF_ERROR(session_->RegisterCustomRegistry(custom_registry));
    }

    return S_OK;
}

void STDMETHODCALLTYPE InferenceSession::FlushContext(onnxruntime::IExecutionProvider* dml_provider) {
    Dml::FlushContext(dml_provider);
}


void STDMETHODCALLTYPE InferenceSession::TrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider) {
    Dml::TrimUploadHeap(dml_provider);
}

void STDMETHODCALLTYPE InferenceSession::ReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider) {
    Dml::ReleaseCompletedReferences(dml_provider);
}

}  // namespace Windows::AI::MachineLearning::Adapter