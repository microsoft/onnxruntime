#pragma once

#include "pch.h"

//
// Exception information
//
#ifndef FACILITY_VISUALCPP
#define FACILITY_VISUALCPP  ((LONG)0x6d)
#endif

#define VcppException(sev,err)  ((sev) | (FACILITY_VISUALCPP<<16) | err)

namespace winrt::Windows::AI::MachineLearning::implementation
{
    enum class PipelineStateCacheType : unsigned char {
        Float32 = 0,
        Float16 = 1,
        Count = 2
    };
    
    enum class PipelineStateCacheFormat : unsigned char {
        RGB8 = 0,
        BGR8 = 1,
        GRAY8 = 2,
        Count = 3
    };

    enum class PipelineStateCacheOperation : unsigned char {
        Tensorize = 0,
        Detensorize = 1,
        Count = 2
    };

    class D3DDeviceCache 
    {
    public:
        ~D3DDeviceCache();
        D3DDeviceCache(Windows::AI::MachineLearning::LearningModelDeviceKind const& deviceKind);
        D3DDeviceCache(Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device);
        D3DDeviceCache(ID3D12CommandQueue* queue);

        ID3D11Device* GetD3D11Device();
        ID3D11DeviceContext4* GetD3D11DeviceContext();

        ID3D12Device1* GetD3D12Device() { return m_device.get(); }
        ID3D12CommandQueue* GetCommandQueue() { return m_commandQueue.get(); }

        Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice GetWinrtDevice();

        ID3D12RootSignature* GetTensorizeRootSignature();
        ID3D12RootSignature* GetDetensorizeRootSignature();
        ID3D12PipelineState* GetCachedPipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo, PipelineStateCacheOperation operation);

        ID3D12Resource* GetDetensorizeVertexBuffer(_Out_ UINT *vertexBufferSize);

        HANDLE GetConverterFenceHandle();

        const GUID& GetFenceGuid() const;

        void GPUSyncD3D11ToD3D12();
        void GPUSyncD3D12ToD3D11();
        void SyncD3D12ToCPU();

        void SyncConverterToD3D11Device(_In_ ID3D11Fence* pD3D11Fence);
        void SyncD3D11DeviceToConverter(_In_ ID3D11Fence* pD3D11Fence);
        
        UINT64 QueueFenceToD3D12();
        void WaitForFenceValue(UINT64 fenceValue);

        const LUID& GetDeviceLuid() { return m_deviceLuid; };

        bool IsFloat16Supported();
        bool SharedHandleInitialized();

    private:
        void EnsureD3D11FromD3D12();
        void EnsureD3D12Fence();
        void EnsureSharedFences();
        void InitializeCommandQueue(ID3D12Device1* device);

        ID3D12PipelineState* CreateTensorizePipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo);
        ID3D12PipelineState* CreateDetensorizePipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo);
        
        com_ptr<ID3D12Device1> m_device;
        com_ptr<ID3D12CommandQueue> m_commandQueue;
        com_ptr<ID3D12SharingContract> m_sharingContract;

        com_ptr<ID3D11Device> m_device11;
        Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice;
        com_ptr<ID3D11DeviceContext4> m_deviceContext11;

        com_ptr<ID3D12RootSignature> m_spTensorizeRootSignature;
        com_ptr<ID3D12RootSignature> m_spDetensorizeRootSignature;

        com_ptr<ID3D12PipelineState> m_spCachedPipelineState[PipelineStateCacheType::Count][PipelineStateCacheFormat::Count][PipelineStateCacheFormat::Count][PipelineStateCacheOperation::Count];

        com_ptr<ID3D12Resource> m_detensorizeVertexBuffer;
        
        com_ptr<ID3D11Fence> m_d3d11Fence;
        com_ptr<ID3D12Fence> m_d3d12Fence;
        std::atomic<UINT64> m_fenceValue = 1;

        GUID m_fenceGuid;

        com_ptr<ID3D12Fence> m_converterFence;
        wil::unique_handle m_converterFenceHandle;
        std::atomic<UINT64> m_converterFenceValue = 1;

        LUID m_deviceLuid;
        static const UINT sc_vertexBufferSize = sizeof(DirectX::XMFLOAT3) * 4;

        // added a lock when we added delay loading to the device cache.   Since parts of 
        // initialization happen later, we need make it thread safe.
        CWinML_Lock m_lock;
    };
}