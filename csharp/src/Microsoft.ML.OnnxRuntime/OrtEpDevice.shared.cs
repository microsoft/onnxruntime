// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents a synchronization primitive for stream operations.
    /// </summary>
    public class OrtSyncStream : SafeHandle
    {
        internal OrtSyncStream(IntPtr streamHandle)
            : base(IntPtr.Zero, true) // Provide required arguments to SafeHandle constructor
        {
            handle = streamHandle;
        }

        /// <summary>
        /// Fetch sync stream handle for possible use
        /// in session options.
        /// </summary>
        /// <returns>Opaque stream handle</returns>
        public IntPtr GetHandle()
        {
            return NativeMethods.OrtSyncStream_GetHandle(handle);
        }

        internal IntPtr Handle => handle;

        /// <summary>
        /// Implements SafeHandle interface
        /// </summary>
        public override bool IsInvalid => handle == IntPtr.Zero;

        /// <summary>
        /// Implements SafeHandle interface to release native handle
        /// </summary>
        /// <returns>always true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseSyncStream(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }

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

        /// <summary>
        /// The OrtMemoryInfo instance describing the memory characteristics of the device.
        /// </summary>
        /// <param name="deviceMemoryType">memory type requested</param>
        /// <returns></returns>
        public OrtMemoryInfo GetMemoryInfo(OrtDeviceMemoryType deviceMemoryType)
        {
            IntPtr memoryInfoPtr = NativeMethods.OrtEpDevice_MemoryInfo(_handle, deviceMemoryType);
            return new OrtMemoryInfo(memoryInfoPtr, /* owned= */ false);
        }

        /// <summary>
        /// Creates a synchronization stream for operations on this device.
        /// Can be used to implement async operations on the device such as
        /// CopyTensors.
        /// </summary>
        /// <param name="streamOptions">stream options can be null</param>
        /// <returns></returns>
        public OrtSyncStream CreateSyncStream(IReadOnlyDictionary<string, string> streamOptions)
        {
            OrtKeyValuePairs options = null;
            IntPtr optionsHandle = IntPtr.Zero;
            try
            {
                if (streamOptions != null)
                {
                    options = new OrtKeyValuePairs(streamOptions);
                    optionsHandle = options.Handle;
                }

                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSyncStreamForEpDevice(_handle,
                    optionsHandle, out IntPtr syncStream));
                return new OrtSyncStream(syncStream);
            }
            finally
            {
                options?.Dispose();
            }
        }

        private readonly IntPtr _handle;
    }
}