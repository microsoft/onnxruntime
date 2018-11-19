// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    internal class NativeCpuAllocatorInfo : IDisposable
    {
        // static singleton
        private static readonly Lazy<NativeCpuAllocatorInfo> _instance = new Lazy<NativeCpuAllocatorInfo>(() => new NativeCpuAllocatorInfo());

        // member variables
        private IntPtr _nativeHandle;

        internal static IntPtr Handle  // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _instance.Value._nativeHandle;
            }
        }

        private NativeCpuAllocatorInfo()
        {
            _nativeHandle = CreateCPUAllocatorInfo();
        }

        private static IntPtr CreateCPUAllocatorInfo()
        {
            IntPtr allocInfo = IntPtr.Zero;
            try
            {
                IntPtr status = NativeMethods.ONNXRuntimeCreateCpuAllocatorInfo(NativeMethods.AllocatorType.DeviceAllocator, NativeMethods.MemoryType.Cpu, out allocInfo);
                NativeApiStatus.VerifySuccess(status);
                return allocInfo;
            }
            catch (Exception e)
            {
                if (allocInfo != IntPtr.Zero)
                {
                    NativeMethods.ReleaseONNXRuntimeAllocatorInfo(allocInfo);
                }
                throw e;
            }
        }

        ~NativeCpuAllocatorInfo()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                //release managed resource
            }

            //release unmanaged resource
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeMethods.ReleaseONNXRuntimeAllocatorInfo(_nativeHandle);
            }
        }
    }
}
