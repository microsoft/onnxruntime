// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Diagnostics;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Class to that stores information about the file location where an "external" initializer is stored.
    /// </summary>
    /// <see cref="OrtModelCompilationOptions.HandleInitializerDelegate"/>
    public class OrtExternalInitializerInfo : SafeHandle, IReadOnlyExternalInitializerInfo
    {
        // Set to false when constructed with an externally managed constant handle owned by ORT.
        private readonly bool _ownsHandle = true;

        /// <summary>
        /// Create a new OrtExternalInitializerInfo instance. 
        /// </summary>
        /// <param name="filePath">The path to the file that stores the initializer data.</param>
        /// <param name="fileOffset">The byte offset in the file where the data is stored.</param>
        /// <param name="byteSize">The size of the data (in bytes) within the file.</param>
        public OrtExternalInitializerInfo(string filePath, long fileOffset, long byteSize)
            : base(IntPtr.Zero, ownsHandle: true)
        {
            var platformFilePath = NativeOnnxValueHelper.GetPlatformSerializedString(filePath);
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtCreateExternalInitializerInfo(platformFilePath, fileOffset, (UIntPtr)byteSize, out handle));
            _ownsHandle = true;
        }

        /// <summary>
        /// Create a new OrtExternalInitializerInfo instance from an existing native OrtExternalInitializerInfo handle.
        /// </summary>
        /// <param name="constHandle">Native OrtExternalInitializerInfo handle.</param>
        /// <param name="ownsHandle">True if the OrtExternalInitializerInfo instance owns the native handle.
        /// Defaults to false.</param>
        internal OrtExternalInitializerInfo(IntPtr constHandle, bool ownsHandle = false)
            : base(IntPtr.Zero, ownsHandle)
        {
            Debug.Assert(constHandle != IntPtr.Zero);
            SetHandle(constHandle);
            _ownsHandle = ownsHandle;
        }

        /// <summary>
        /// Get the file path to the file that store's the initializer's data.
        /// </summary>
        /// <remarks>
        /// The path is relative to the filesystem directory where the ONNX model was stored.
        /// </remarks>
        /// <returns>The file path.</returns>
        public string GetFilePath()
        {
            IntPtr filePathPtr = NativeMethods.OrtExternalInitializerInfo_GetFilePath(handle);
            if (filePathPtr == IntPtr.Zero)
            {
                return string.Empty;
            }

            return NativeOnnxValueHelper.StringFromNativePathString(filePathPtr);
        }

        /// <summary>
        /// Get the byte offset within the file where the initializer's data is stored.
        /// </summary>
        /// <returns>The file offset location.</returns>
        public long GetFileOffset()
        {
            return NativeMethods.OrtExternalInitializerInfo_GetFileOffset(handle);
        }

        /// <summary>
        /// Get the size in bytes of the initializer's data within the file.
        /// </summary>
        /// <returns>The size in bytes of the initializer data.</returns>
        public long GetByteSize()
        {
            UIntPtr byteSize = NativeMethods.OrtExternalInitializerInfo_GetByteSize(handle);
            return checked((long)byteSize);
        }

        /// <summary>
        /// Indicates whether the native handle is invalid.
        /// </summary>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Release the native instance of OrtExternalInitializerInfo if we own it.
        /// </summary>
        /// <returns>true on success and false on error.</returns>
        protected override bool ReleaseHandle()
        {
            if (!_ownsHandle)
            {
                // Return false to indicate an error.
                // ReleaseHandle() should not be called on a const handle that this class does not own.
                return false;
            }

            NativeMethods.OrtReleaseExternalInitializerInfo(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }

    /// <summary>
    /// Interface for all readonly methods implemented by OrtExternalInitializerInfo.
    /// </summary>
    public interface IReadOnlyExternalInitializerInfo
    {
        /// <summary>
        /// Get the file path to the file that store's the initializer's data.
        /// </summary>
        /// <remarks>
        /// The path is relative to the filesystem directory where the ONNX model was stored.
        /// </remarks>
        /// <returns>The file path.</returns>
        string GetFilePath();

        /// <summary>
        /// Get the byte offset within the file where the initializer's data is stored.
        /// </summary>
        /// <returns>The file offset location.</returns>
        long GetFileOffset();

        /// <summary>
        /// Get the size in bytes of the initializer's data within the file.
        /// </summary>
        /// <returns>The size in bytes of the initializer data.</returns>
        long GetByteSize();
    }
}