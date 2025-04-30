// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime;
using System;
using System.Runtime.InteropServices;

public enum OrtHardwareDeviceType
{
    CPU = 0,
    GPU = 1,
    NPU = 2,
}

public class OrtHardwareDevice : SafeHandle
{

    public OrtHardwareDevice(IntPtr deviceHandle)
        : base(deviceHandle, ownsHandle: false)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    public OrtHardwareDeviceType Type
    {
        get
        {
            return (OrtHardwareDeviceType)NativeMethods.OrtHardwareDevice_Type(handle);
        }
    }

    public uint VendorId
    {
        get
        {
            return NativeMethods.OrtHardwareDevice_VendorId(handle);
        }
    }

    public string Vendor
    {
        get
        {
            IntPtr vendorPtr = NativeMethods.OrtHardwareDevice_Vendor(handle);
            return NativeOnnxValueHelper.StringFromNativeUtf8(vendorPtr);
        }
    }

    public uint DeviceId
    {
        get
        {
            return NativeMethods.OrtHardwareDevice_DeviceId(handle);
        }
    }

    public OrtKeyValuePairs Metadata
    {
        get
        {
            return new OrtKeyValuePairs(NativeMethods.OrtHardwareDevice_Metadata(handle));
        }
    }

    // No need to release handle because we don't own it.
    protected override bool ReleaseHandle()
    {
        return true;
    }
}
