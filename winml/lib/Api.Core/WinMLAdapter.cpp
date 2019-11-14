// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "inc/WinMLAdapter.h"
#include "inc/CustomRegistryHelper.h"
#include "inc/LotusEnvironment.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"

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
};

class WinMLAdapter : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, 
    IWinMLAdapter> {
public:
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
            !model_proto_inner->ParseFromZeroCopyStream(&stream) == false,
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

    onnxruntime::Tensor* STDMETHODCALLTYPE CreateTensor(
        winml::TensorKind kind,
        const int64_t * shape,
        uint32_t shape_count,
        onnxruntime::IExecutionProvider* provider) override {
        onnxruntime::TensorShape tensor_shape(shape, shape_count);
        auto pTensor = new onnxruntime::Tensor(
            GetType(kind),
            tensor_shape,
            provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault));
        return pTensor;
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

public:

    IOBinding(onnxruntime::IOBinding * binding) : binding_(binding) {
    }

    onnxruntime::IOBinding* STDMETHODCALLTYPE get() override {
        return binding_.get();
    }

    HRESULT STDMETHODCALLTYPE BindInput(const std::string& name, const OrtValue& ml_value) override {
        ORT_THROW_IF_ERROR(binding_->BindInput(name, ml_value));
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE BindOutput(const std::string& name, const OrtValue& ml_value) override {
        ORT_THROW_IF_ERROR(binding_->BindOutput(name, ml_value));
        return S_OK;
    }

    const std::vector<std::string>& STDMETHODCALLTYPE GetOutputNames() override {
        return binding_->GetOutputNames();
    }
    std::vector<OrtValue>& STDMETHODCALLTYPE GetOutputs() override {
        return binding_->GetOutputs();
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

    auto custom_registries = WinML::GetLotusCustomRegistries(registry);

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