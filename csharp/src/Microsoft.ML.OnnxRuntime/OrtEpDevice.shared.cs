// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents the combination of an execution provider and a hardware device 
    /// that the execution provider can utilize.
    /// </summary>
    public class OrtEpDevice
    {
        /// <summary>
        /// Construct an OrtEpDevice from an existing native OrtEpDevice instance.
        /// </summary>
        /// <param name="epDeviceHandle">Native OrtEpDevice handle.</param>
        internal OrtEpDevice(IntPtr epDeviceHandle)
        {
            _handle = epDeviceHandle;
        }

        internal IntPtr Handle => _handle;

        /// <summary>
        /// The name of the execution provider.
        /// </summary>
        public string EpName
        {
            get
            {
                IntPtr namePtr = NativeMethods.OrtEpDevice_EpName(_handle);
                return NativeOnnxValueHelper.StringFromNativeUtf8(namePtr);
            }
        }

        /// <summary>
        /// The vendor who owns the execution provider.
        /// </summary>
        public string EpVendor
        {
            get
            {
                IntPtr vendorPtr = NativeMethods.OrtEpDevice_EpVendor(_handle);
                return NativeOnnxValueHelper.StringFromNativeUtf8(vendorPtr);
            }
        }

        /// <summary>
        /// Execution provider metadata.
        /// </summary>
        public OrtKeyValuePairs EpMetadata
        {
            get
            {
                return new OrtKeyValuePairs(NativeMethods.OrtEpDevice_EpMetadata(_handle));
            }
        }

        /// <summary>
        /// Execution provider options.
        /// </summary>
        public OrtKeyValuePairs EpOptions
        {
            get
            {
                return new OrtKeyValuePairs(NativeMethods.OrtEpDevice_EpOptions(_handle));
            }
        }

        /// <summary>
        /// The hardware device that the execution provider can utilize.
        /// </summary>
        public OrtHardwareDevice HardwareDevice
        {
            get
            {
                IntPtr devicePtr = NativeMethods.OrtEpDevice_Device(_handle);
                return new OrtHardwareDevice(devicePtr);
            }
        }

        private readonly IntPtr _handle;
    }
}