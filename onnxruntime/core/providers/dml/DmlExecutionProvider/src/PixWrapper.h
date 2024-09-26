#pragma once

// This namespace abstracts PIX APIs to provide a consistent interface that works for DirectML's inbox, redist,
// Xbox, and WSL variants.
//
// | DirectML Target Platform    | Dependency               | Implementation              | Supports Disk Capture |
// | ----------------------------|--------------------------|-----------------------------|-----------------------|
// | Windows Inbox (x64, ARM64)  | d3d12.dll                | PixWrapperWindowsInbox.cpp  | No                    |
// | Windows Inbox (x86, ARM)    | -                        | PixWrapperNoop.cpp          | No                    |
// | Windows Redist (x64, ARM64) | WinPixEventRuntime.dll   | PixWrapperWindowsRedist.cpp | Yes                   |
// | Windows Redist (x86, ARM)   | -                        | PixWrapperNoop.cpp          | No                    |
// | Xbox                        | d3d12_xs.dll             | PixWrapperXbox.cpp          | Yes                   |
// | Linux                       | -                        | PixWrapperNoop.cpp          | No                    |
//
// The "Supports Disk Capture" column indicates whether or not programmatic GPU captures can be taken when running
// DirectML outside of PIX/pixtool. A programmatic capture is a recording of GPU work that is initiated in code
// instead of a user-initiated action (e.g. pressing button in PIX UI). Programmatic captures are only recorded
// under the following conditions (in other cases the call is effectively a no-op):
//
//     1. (Windows) The process is launched from within PIX or by pixtool.
//     2. (Windows & Xbox) Pix::Initialize was called with saveProgrammaticCapturesToDisk == TRUE.
//
// NOTE: all functions exposed in the Pix namespace below can be safely called regardless of target platform. If
// the wrapped PIX API does not exist or is not stable on a target platform then the function call will be a no-op.
namespace Pix
{
    // This function should be called at least once to ensure PIX libraries are loaded and the APIs
    // in this namespace are usable. You must call this before D3D device creation when enabling the
    // saveProgrammaticCapturesToDisk option. Subsequent calls to this function have no effect.
    //
    // saveProgrammaticCapturesToDisk:
    //     This boolean affects the behavior of programmatic GPU captures taken with BeginGpuCapture and
    //     EndGpuCapture. By default, on Windows, D3D work between begin/end capture calls is only recorded
    //     when the process is launched from the PIX UI or pixtool. Setting this value to TRUE will record
    //     and save the GPU capture to disk if the process is launched outside of PIX (no effect when launching
    //     within PIX for Windows). On Xbox, you MUST use saveProgrammaticCapturesToDisk to programmatically
    //     record a GPU capture. The boolean does not affect user-initiated captures from the PIX UI.
    //
    // diskCaptureFilename:
    //     Sets the name of any PIX GPU captures recorded to disk. This has no effect unless
    //     saveProgrammaticCapturesToDisk is TRUE.
    //
    void Initialize(
        bool saveProgrammaticCapturesToDisk = false,
        std::optional<std::wstring> diskCaptureFilename = std::nullopt
    );

    // Wraps PIX(Begin|End)Event with command lists.
    // This API is functional on the following platforms: Windows inbox, Windows redist, and Xbox.
    void BeginEvent(ID3D12GraphicsCommandList* commandList, UINT64 color, PCSTR formatString);
    void EndEvent(ID3D12GraphicsCommandList* commandList);

    // Wraps PIX(Begin|End)Event with command queues.
    // This API is functional on the following platforms: Windows inbox and Xbox.
    void BeginEvent(ID3D12CommandQueue* commandQueue, UINT64 color, PCSTR formatString);
    void BeginEvent(ID3D12CommandQueue* commandQueue, UINT64 color, PCWSTR formatString);
    void EndEvent(ID3D12CommandQueue* commandQueue);

    // Wraps PIXSetMarker with command lists.
    // This API is functional on the following platforms: Windows inbox, Windows redist, and Xbox.
    void SetMarker(ID3D12GraphicsCommandList* commandList, UINT64 color, _In_ PCSTR string);

    // Wraps PIX APIs to record a programmatic GPU capture. The commandQueue parameter must be non-null
    // for captures to be recorded on Xbox; setting the commandQueue parameter has no effect on Windows.
    // This API is functional on the following platforms: Windows redist and Xbox.
    void BeginGpuCapture(ID3D12CommandQueue* commandQueue = nullptr);
    void EndGpuCapture(ID3D12CommandQueue* commandQueue = nullptr);

    // RAII helper to start and end a PIX event using object lifetime.
    template <typename T>
    struct ScopedEvent
    {
        T* m_commandListOrQueue;

        ScopedEvent(T* commandListOrQueue, UINT64 color, PCSTR formatString) : m_commandListOrQueue(commandListOrQueue)
        {
            BeginEvent(commandListOrQueue, color, formatString);
        }

        ~ScopedEvent()
        {
            EndEvent(m_commandListOrQueue);
        }
    };
}
