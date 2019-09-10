#pragma once

struct OrtValue;

namespace Windows::AI::MachineLearning
{

class PoolObjectWrapper;

enum class BindingType { Input, Output };

struct BindingContext
{
    BindingType Type = BindingType::Input;
    winrt::Windows::AI::MachineLearning::LearningModelSession Session = nullptr;
    winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor Descriptor = nullptr;
    winrt::Windows::Foundation::Collections::IPropertySet Properties = nullptr;
    std::shared_ptr<PoolObjectWrapper> Converter;
};

// ILotusValueProviderPrivate exposes a private Lotus interface to the engine so that it can retrieve tensor
// resources stored in winrt structures. 
struct __declspec(uuid("27e2f437-0112-4693-849e-e04323a620fb")) __declspec(novtable) ILotusValueProviderPrivate : IUnknown
{
    virtual  HRESULT __stdcall GetOrtValue(BindingContext& bindingContext, OrtValue* mlValue) = 0;
    virtual  HRESULT __stdcall IsPlaceholder(bool* pIsPlaceholder) = 0;
    virtual  HRESULT __stdcall UpdateSourceResourceData(BindingContext& bindingContext, OrtValue& mlValue) = 0;
    virtual  HRESULT __stdcall AbiRepresentation(winrt::Windows::Foundation::IInspectable& abiRepresentation) = 0;
};

}