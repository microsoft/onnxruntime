#pragma once
#include "core/common/logging/isink.h"
#include "WinMLProfiler.h"
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Core.h>

#pragma warning(push)
#pragma warning(disable : 4505)

namespace Windows
{
namespace AI
{
namespace MachineLearning
{
    class CWinMLLogSink : public onnxruntime::logging::ISink {
    public:
        CWinMLLogSink() 
        {
        }
        static void EnableDebugOutput()
        {
            DebugOutput = true;
            OutputDebugStringW(L"Windows.AI.MachineLearning: Debug Output Enabled \r\n");
        }
        void SendProfileEvent(onnxruntime::profiling::EventRecord& eventRecord) const;
        void SendImpl(const onnxruntime::logging::Timestamp &timestamp, const std::string &logger_id, const onnxruntime::logging::Capture &message);
    private:
        static bool DebugOutput;
    };
// TODO: a bug in lotus requires a logging manager.  This function registers a static singleton logger as "default"
inline onnxruntime::logging::LoggingManager& DefaultLoggingManager()
{
    // create a CLog based default logging manager
    static std::string default_logger_id{ "Default" };
    static onnxruntime::logging::LoggingManager default_logging_manager
    {
        std::unique_ptr<onnxruntime::logging::ISink> { new CWinMLLogSink() },
        onnxruntime::logging::Severity::kVERBOSE,
        false,
        onnxruntime::logging::LoggingManager::InstanceType::Default,
        &default_logger_id,
        MAXINT32
    };

    return default_logging_manager;
}

static void OnSuspending(winrt::Windows::Foundation::IInspectable const& sender, winrt::Windows::ApplicationModel::SuspendingEventArgs const& args)
{
    if (!g_Profiler.IsStillReset()) //If profiler is still reset, then don't log RuntimePerf
    {
        g_Telemetry.LogRuntimePerf(g_Profiler, true);
    }
}

class LotusEnvironment
{
private:
    std::unique_ptr<onnxruntime::Environment> m_lotusEnvironment;
    winrt::event_token m_suspendToken;
    onnxruntime::logging::LoggingManager* m_defaultLoggingManager;
    void RegisterSuspendHandler()
    {
		// Bug 23401273: Layering: onnxruntime\winml\lib\Api.Core\inc\LotusEnvironment.h has RegisterSuspendHandler disabled. Need to reenable.
		return;
		/*
        try
        {
            m_suspendToken = winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(
                winrt::Windows::Foundation::EventHandler<winrt::Windows::ApplicationModel::SuspendingEventArgs>(&OnSuspending));
        }
        catch (...) {}//Catch in case CoreApplication cannot be found for non-UWP executions
		*/
    }
public:
    LotusEnvironment()
    {
        // TODO: Do we need to call this or just define the method?
        m_defaultLoggingManager = &DefaultLoggingManager();

        if (!onnxruntime::Environment::Create(m_lotusEnvironment).IsOK())
        {
            throw winrt::hresult_error(E_FAIL);
        }

        auto allocatorMap = onnxruntime::DeviceAllocatorRegistry::Instance().AllRegistrations();
        if (allocatorMap.find("Cpu") == allocatorMap.end())
        {
            onnxruntime::DeviceAllocatorRegistry::Instance().RegisterDeviceAllocator(
                "Cpu",
                [](int) { return std::make_unique<onnxruntime::CPUAllocator>(); },
                std::numeric_limits<size_t>::max());
        }
        RegisterSuspendHandler();
    }

    ~LotusEnvironment()
    {
        if (m_suspendToken)
        {
            winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(m_suspendToken);
        }
    }

    const onnxruntime::logging::Logger* GetDefaultLogger()
    {
        return &m_defaultLoggingManager->DefaultLogger();
    }
};

namespace ExecutionProviders
{
    __declspec(selectany) const char* CPUExecutionProvider = "CPUExecutionProvider";
}

} // namespace MachineLearning
} // namespace AI
} // namespace Windows

#pragma warning(pop)