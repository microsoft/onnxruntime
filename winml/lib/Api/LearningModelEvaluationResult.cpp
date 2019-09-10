#include "pch.h"
#include "LearningModelEvaluationResult.h"

namespace winrt::Windows::AI::MachineLearning::implementation
{
    hstring LearningModelEvaluationResult::CorrelationId() try
    {
        return m_correlationId;
    }
    WINML_CATCH_ALL

    void LearningModelEvaluationResult::CorrelationId(const hstring& correlationId)
    {
        m_correlationId = correlationId;
    }

    int32_t LearningModelEvaluationResult::ErrorStatus() try
    {
        return m_errorStatus;
    }
    WINML_CATCH_ALL

    void LearningModelEvaluationResult::ErrorStatus(int32_t errorStatus)
    {
        m_errorStatus = errorStatus;
    }

    bool LearningModelEvaluationResult::Succeeded() try
    {
        return m_succeeded;
    }
    WINML_CATCH_ALL

    void LearningModelEvaluationResult::Succeeded(bool succeeded)
    {
        m_succeeded = succeeded;
    }

    Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable> LearningModelEvaluationResult::Outputs() try
    {
        std::unordered_map<hstring, Windows::Foundation::IInspectable> outputs;

        for (auto& output : m_outputs)
        {
            auto key = WinML::Strings::hstring_from_utf8(output.first);
            auto value = output.second;
            outputs.emplace(key, value);
        }

        return winrt::single_threaded_map(std::move(outputs)).GetView();
    }
    WINML_CATCH_ALL

    void LearningModelEvaluationResult::Outputs(Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable> outputs)
    {
        m_outputs.clear();

        for (auto pair : outputs)
        {
            auto key = WinML::Strings::utf8_from_hstring(pair.Key());
            auto value = pair.Value();
            m_outputs.emplace(key, value);
        }
    }

    HRESULT LearningModelEvaluationResult::GetOutput(
        const wchar_t *name,
        UINT32 cchName,
        IUnknown ** result)
    {
        *result = nullptr;

        auto outputName = WinML::Strings::utf8_from_unicode(name, cchName);
        auto foundIt = m_outputs.find(outputName);

        if (foundIt == std::end(m_outputs))
        {
            return E_FAIL;
        }

        auto inspectable = foundIt->second;
        *result = inspectable.as<::IUnknown>().detach();

        return S_OK;
    }

    HRESULT LearningModelEvaluationResult::SetOutputs(
        std::unordered_map<std::string, Windows::Foundation::IInspectable>&& outputs)
    {
        m_outputs = std::move(outputs);
        return S_OK;
    }

}
