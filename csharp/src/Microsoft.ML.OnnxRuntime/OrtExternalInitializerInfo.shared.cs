// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Class to that stores information about the file location where an "external" initializer is stored.
    /// </summary>
    /// <see cref="OrtModelCompilationOptions.HandleInitializerDelegate"/>
    public class OrtExternalInitializerInfo : SafeHandle
    {
        // Set to true when constructed with an externally managed constant handle owned by ORT.
        private readonly bool _isConst;

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
            _isConst = false;
        }

        /// <summary>
        /// Create a new OrtExternalInitializerInfo instance from an existing native OrtExternalInitializerInfo handle.
        /// </summary>
        /// <param name="constHandle">Native OrtExternalInitializerInfo handle.</param>
        /// <remarks>
        /// The instance is read-only.
        /// </remarks>
        /// <exception cref="InvalidOperationException"></exception>
        internal OrtExternalInitializerInfo(IntPtr constHandle)
            : base(IntPtr.Zero, ownsHandle: false)
        {
            if (constHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException($"{nameof(OrtExternalInitializerInfo)}: Invalid instance.");
            }
            SetHandle(constHandle);
            _isConst = true;
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

            return NativeOnnxValueHelper.StringFromNativeUtf8(filePathPtr);
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
            if (_isConst)
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
}