// Implements PixWrapper by calling into the redistributable library WinPixEventRuntime.dll.

#include "precomp.h"
#include "PixWrapper.h"
#include <appmodel.h>
#include <mutex>
#include <filesystem>
#include <WinPixEventRuntime/pix3.h>
#include <DXProgrammableCapture.h>
#include <dxgi1_5.h>
#include <dxgidebug.h>

typedef HRESULT(WINAPI* BeginEventOnCommandList)(ID3D12GraphicsCommandList* commandList, UINT64 color, _In_ PCSTR formatString);
typedef HRESULT(WINAPI* EndEventOnCommandList)(ID3D12GraphicsCommandList* commandList);
typedef HRESULT(WINAPI* SetMarkerOnCommandList)(ID3D12GraphicsCommandList* commandList, UINT64 color, _In_ PCSTR formatString);
typedef HRESULT(WINAPI* BeginCapture2)(DWORD captureFlags, _In_opt_ const PPIXCaptureParameters captureParameters);
typedef HRESULT(WINAPI* EndCapture)(BOOL discard);
typedef HRESULT(WINAPI* GetDebugInterface1)(UINT Flags, REFIID riid, _COM_Outptr_ void **pDebug);

static BeginEventOnCommandList g_beginEventFn = nullptr;
static EndEventOnCommandList g_endEventFn = nullptr;
static SetMarkerOnCommandList g_setMarkerFn = nullptr;
static BeginCapture2 g_beginCaptureFn = nullptr;
static EndCapture g_endCaptureFn = nullptr;
static bool g_pixLoadAttempted = false;
static std::mutex g_mutex;
static bool g_saveProgrammaticCapturesToDisk = false;
static std::wstring g_diskCaptureFilename = L"directml.wpix";

// Legacy interface to record a GPU capture when the process is launched through
// the PIX UI or pixtool. WinPixGpuCapturer.dll is automatically loaded by PIX.
// This will be nullptr if running outside of PIX.
// https://devblogs.microsoft.com/pix/programmatic-capture/
static ComPtr<IDXGraphicsAnalysis> g_pixAttachedCaptureInterface;

void Pix::Initialize(bool saveProgrammaticCapturesToDisk, std::optional<std::wstring> diskCaptureFilename)
{
    std::lock_guard lock(g_mutex);
    if (!g_pixLoadAttempted)
    {
        g_pixLoadAttempted = true;
        std::vector<wchar_t> modulePath(MAX_PATH);
        while (true)
        {
            DWORD moduleLength = GetModuleFileNameW(nullptr, modulePath.data(), static_cast<DWORD>(modulePath.size()));
            if (moduleLength == 0)
            {
                return;
            }
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            {
                modulePath.resize(modulePath.size() * 2);
            }
            else
            {
                break;
            }
        }

        std::filesystem::path dllPath(modulePath.data());
        uint32_t length = 0;
        long rc = GetCurrentPackageFullName(&length, nullptr);
        // APPMODEL_ERROR_NO_PACKAGE will be returned if running in desktop context
        // see: https://docs.microsoft.com/en-us/archive/blogs/appconsult/desktop-bridge-identify-the-applications-context
        if (rc == APPMODEL_ERROR_NO_PACKAGE)
        {
            dllPath.replace_filename(L"WinPixEventRuntime.dll");
        }
        else
        {
            dllPath.replace_filename(L"WinPixEventRuntime_UAP.dll");
        }

        // Intentionally leaked
        HMODULE pixRuntime = LoadLibrary(dllPath.c_str());
        if (pixRuntime != nullptr)
        {
            g_beginEventFn = (BeginEventOnCommandList)GetProcAddress(pixRuntime, "PIXBeginEventOnCommandList");
            g_endEventFn = (EndEventOnCommandList)GetProcAddress(pixRuntime, "PIXEndEventOnCommandList");
            g_setMarkerFn = (SetMarkerOnCommandList)GetProcAddress(pixRuntime, "PIXSetMarkerOnCommandList");
            g_beginCaptureFn = (BeginCapture2)GetProcAddress(pixRuntime, "PIXBeginCapture2");
            g_endCaptureFn = (EndCapture)GetProcAddress(pixRuntime, "PIXEndCapture");
        }

        // Try to create IDXGraphicsAnalysis. This will only succeed if the process is launched through PIX.
        HMODULE dxgiRuntime = LoadLibrary(L"dxgi.dll");
        if (dxgiRuntime)
        {
            auto getDebugInterface1 = (GetDebugInterface1)GetProcAddress(dxgiRuntime, "DXGIGetDebugInterface1");

            if (getDebugInterface1)
            {
                getDebugInterface1(0, IID_PPV_ARGS(&g_pixAttachedCaptureInterface));
            }
        }


        g_saveProgrammaticCapturesToDisk = saveProgrammaticCapturesToDisk;
        if (g_saveProgrammaticCapturesToDisk && !g_pixAttachedCaptureInterface)
        {
            // Intentionally leaked
            PIXLoadLatestWinPixGpuCapturerLibrary();

            if (diskCaptureFilename)
            {
                g_diskCaptureFilename = *diskCaptureFilename;
            }
        }
    }
}

void Pix::BeginEvent(ID3D12GraphicsCommandList* commandList, UINT64 color, PCSTR formatString)
{
    if (g_beginEventFn)
    {
        g_beginEventFn(commandList, color, formatString);
    }
}

void Pix::BeginEvent(ID3D12CommandQueue* commandQueue, UINT64 color, PCSTR formatString)
{
    // Intentionally not implemented since this API isn't part of PIX's stable ABI.
    // There is no guarantee on the version of WinPixEventRuntime.dll that is used with DirectML.dll.
}

void Pix::BeginEvent(ID3D12CommandQueue* commandQueue, UINT64 color, PCWSTR formatString)
{
    // Intentionally not implemented since this API isn't part of PIX's stable ABI.
    // There is no guarantee on the version of WinPixEventRuntime.dll that is used with DirectML.dll.
}

void Pix::EndEvent(ID3D12GraphicsCommandList* commandList)
{
    if (g_endEventFn)
    {
        g_endEventFn(commandList);
    }
}

void Pix::EndEvent(ID3D12CommandQueue* commandQueue)
{
    // Intentionally not implemented since this API isn't part of PIX's stable ABI.
    // There is no guarantee on the version of WinPixEventRuntime.dll that is used with DirectML.dll.
}

void Pix::SetMarker(ID3D12GraphicsCommandList* commandList, UINT64 color, _In_ PCSTR string)
{
    if (g_setMarkerFn)
    {
        g_setMarkerFn(commandList, color, string);
    }
}

void Pix::BeginGpuCapture(ID3D12CommandQueue* commandQueue)
{
    if (g_pixAttachedCaptureInterface)
    {
        g_pixAttachedCaptureInterface->BeginCapture();
    }
    else if (g_saveProgrammaticCapturesToDisk)
    {
        // NOTE: the API for this is technically not part of the stable WinPixEventRuntime.dll ABI.
        // However, we only use this functionality in test code where we have control over the version
        // of WinPixEventRuntime.dll. The core DirectML library should never initiate programmatic GPU
        // captures!
        PIXCaptureParameters captureParams = {};
        captureParams.TimingCaptureParameters.FileName = g_diskCaptureFilename.data();
        PIXBeginCapture(PIX_CAPTURE_GPU, &captureParams);
    }
}

void Pix::EndGpuCapture(ID3D12CommandQueue* commandQueue)
{
    if (g_pixAttachedCaptureInterface)
    {
        g_pixAttachedCaptureInterface->EndCapture();
    }
    else if (g_saveProgrammaticCapturesToDisk)
    {
        // NOTE: the API for this is technically not part of the stable WinPixEventRuntime.dll ABI.
        // However, we only use this functionality in test code where we have control over the version
        // of WinPixEventRuntime.dll. The core DirectML library should never initiate programmatic GPU
        // captures!
        PIXEndCapture(/*discard*/FALSE);
    }
}
