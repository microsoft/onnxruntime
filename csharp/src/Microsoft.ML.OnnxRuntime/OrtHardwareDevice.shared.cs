// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Represents the type of hardware device.
    /// Matches OrtHardwareDeviceType in the ORT C API.
    /// </summary>
    public enum OrtHardwareDeviceType
    {
        CPU = 0,
        GPU = 1,
        NPU = 2,
    }

    /// <summary>
    /// Represents a hardware device that is available on the current system.
    /// </summary>
    public class OrtHardwareDevice : SafeHandle
    {

        /// <summary>
        /// Construct an OrtHardwareDevice for a native OrtHardwareDevice instance.
        /// </summary>
        /// <param name="deviceHandle">Native OrtHardwareDevice handle.</param>
        internal OrtHardwareDevice(IntPtr deviceHandle)
            : base(deviceHandle, ownsHandle: false)
        {
        }

        /// <summary>
        /// Get the type of hardware device.
        /// </summary>
        public OrtHardwareDeviceType Type
        {
            get
            {
                return (OrtHardwareDeviceType)NativeMethods.OrtHardwareDevice_Type(handle);
            }
        }

        /// <summary>
        /// Get the vendor ID of the hardware device if known.
        /// </summary>
        /// <remarks>
        /// For PCIe devices the vendor ID is the PCIe vendor ID. See https://pcisig.com/membership/member-companies.
        /// </remarks>
        public uint VendorId
        {
            get
            {
                return NativeMethods.OrtHardwareDevice_VendorId(handle);
            }
        }

        /// <summary>
        /// The vendor (manufacturer) of the hardware device.
        /// </summary>
        public string Vendor
        {
            get
            {
                IntPtr vendorPtr = NativeMethods.OrtHardwareDevice_Vendor(handle);
                return NativeOnnxValueHelper.StringFromNativeUtf8(vendorPtr);
            }
        }

        /// <summary>
        /// Get the device ID of the hardware device if known.
        /// </summary>
        /// <remarks>
        /// This is the identifier of the device model. 
        /// PCIe device IDs can be looked up at https://www.pcilookup.com/ when combined with the VendorId.
        /// It is NOT a unique identifier for the device in the current system.
        /// </remarks>
        public uint DeviceId
        {
            get
            {
                return NativeMethods.OrtHardwareDevice_DeviceId(handle);
            }
        }

        /// <summary>
        /// Get device metadata.
        /// This may include information such as whether a GPU is discrete or integrated.
        /// The available metadata will differ by platform and device type.
        /// </summary>
        public OrtKeyValuePairs Metadata
        {
            get
            {
                return new OrtKeyValuePairs(NativeMethods.OrtHardwareDevice_Metadata(handle));
            }
        }

        /// <summary>
        /// Indicates whether the native handle is invalid.
        /// </summary>
        public override bool IsInvalid => handle == IntPtr.Zero;

        /// <summary>
        /// No-op. OrtHardwareDevice is always read-only as the instance is owned by native ORT.
        /// </summary>
        /// <returns>True</returns>
        protected override bool ReleaseHandle()
        {
            return true;
        }
    }
}