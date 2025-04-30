// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime;

public class OrtEpDevice : SafeHandle
{
    public OrtEpDevice(IntPtr epDeviceHandle)
        : base(epDeviceHandle, ownsHandle: false)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    public string EpName
    {
        get
        {
            IntPtr namePtr = NativeMethods.OrtEpDevice_EpName(handle);
            return NativeOnnxValueHelper.StringFromNativeUtf8(namePtr);
        }
    }

    public string EpVendor
    {
        get
        {
            IntPtr vendorPtr = NativeMethods.OrtEpDevice_EpVendor(handle);
            return NativeOnnxValueHelper.StringFromNativeUtf8(vendorPtr);
        }
    }

    public OrtKeyValuePairs EpMetadata
    {
        get
        {
            return new OrtKeyValuePairs(NativeMethods.OrtEpDevice_EpMetadata(handle));
        }
    }

    public OrtKeyValuePairs EpOptions
    {
        get
        {
            return new OrtKeyValuePairs(NativeMethods.OrtEpDevice_EpOptions(handle)); 
        }
    }

    public OrtHardwareDevice HardwareDevice
    {
        get
        {
            IntPtr devicePtr = NativeMethods.OrtEpDevice_Device(handle);
            return new OrtHardwareDevice(devicePtr);
        }
    }

    // This wrapper doesn't own the handle, so no cleanup needed
    protected override bool ReleaseHandle()
    {
        return true;
    }
}

