// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Flags representing options to enable when compiling a model.
    /// Matches OrtCompileApiFlags in the ORT C API.
    /// </summary>
    [Flags]
    public enum OrtCompileApiFlags : uint
    {
        NONE = 0,
        ERROR_IF_NO_NODES_COMPILED = 1 << 0,
        ERROR_IF_OUTPUT_FILE_EXISTS = 1 << 1,
    }

    /// <summary>
    /// This class is used to set options for model compilation, and to produce a compiled model using those options.
    /// See https://onnxruntime.ai/docs/api/c/ for further details of various options.
    /// </summary>
    public class OrtModelCompilationOptions : SafeHandle
    {
        /// <summary>
        /// Create a new OrtModelCompilationOptions object from SessionOptions.
        /// </summary>
        /// <param name="sessionOptions">SessionOptions instance to read settings from.</param>
        public OrtModelCompilationOptions(SessionOptions sessionOptions)
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtCreateModelCompilationOptionsFromSessionOptions(
                    OrtEnv.Instance().Handle, sessionOptions.Handle, out handle));
        }

        /// <summary>
        /// Compile the model using the options set in this object.
        /// </summary>
        public void CompileModel()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.CompileApi.OrtCompileModel(OrtEnv.Instance().Handle, handle));
        }


        /// <summary>
        /// Set the input model to compile.
        /// </summary>
        /// <param name="path">Path to ONNX model to compile.</param>
        public void SetInputModelPath(string path)
        {
            var platformPath = NativeOnnxValueHelper.GetPlatformSerializedString(path);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetInputModelPath(handle, platformPath));
        }

        /// <summary>
        /// Set the input model to compile to be a byte array.
        /// The input bytes are NOT copied and must remain valid while in use by ORT.
        /// </summary>
        /// <param name="buffer">Input model bytes.</param>
        public void SetInputModelFromBuffer(byte[] buffer)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetInputModelFromBuffer(
                    handle, buffer, (UIntPtr)buffer.Length));
        }

        /// <summary>
        /// Set the path to write the compiled ONNX model to.
        /// </summary>
        /// <param name="path">Path to write compiled model to.</param>
        public void SetOutputModelPath(string path)
        {
            var platformPath = NativeOnnxValueHelper.GetPlatformSerializedString(path);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelPath(handle, platformPath));

        }

        /// <summary>
        /// Set the path to a file to write initializers as external data to,
        /// and the threshold that determines when to write an initializer to the external data file.
        /// </summary>
        /// <param name="filePath">Path to file to write external data to.</param>
        /// <param name="threshold">Size at which an initializer will be written to external data.</param>
        public void SetOutputModelExternalInitializersFile(string filePath, ulong threshold)
        {
            var platformPath = NativeOnnxValueHelper.GetPlatformSerializedString(filePath);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelExternalInitializersFile(
                    handle, platformPath, new UIntPtr(threshold)));
        }

        // TODO: In order to use this to create an InferenceSession without copying bytes we need more infrastructure.
        // - Need something that wraps the allocator, pointer and size and is SafeHandle based.
        //   - When it is disposed we need to use the allocator to release the native buffer.
        // - Need the 4 InferenceSession ctors that take byte[] for the model to be duplicated to handle this new
        //   wrapper type.
        // Due to that making this API internal so we can test it. We can make it public when the other infrastructure
        // is in place as it will change the signature of the API.
        internal void SetOutputModelBuffer(OrtAllocator allocator,
                                           ref IntPtr outputModelBufferPtr, ref UIntPtr outputModelBufferSizePtr)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelBuffer(
                    handle, allocator.Pointer, ref outputModelBufferPtr, ref outputModelBufferSizePtr));
        }

        /// <summary>
        /// Enables or disables the embedding of EPContext binary data into the `ep_cache_context` attribute
        /// of EPContext nodes.
        /// </summary>
        /// <param name="embed">Enable if true. Default is false.</param>
        public void SetEpContextEmbedMode(bool embed)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetEpContextEmbedMode(handle, embed));
        }

        /// <summary>
        /// Sets flags from OrtCompileApiFlags that represent one or more boolean options to enable.
        /// </summary>
        /// <param name="flags">bitwise OR of flags in OrtCompileApiFlags to enable.</param>
        public void SetFlags(OrtCompileApiFlags flags)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetFlags(handle, (uint)flags));
        }

        internal IntPtr Handle => handle;


        /// <summary>
        /// Indicates whether the native handle is invalid.
        /// </summary>
        public override bool IsInvalid => handle == IntPtr.Zero;

        /// <summary>
        /// Release the native instance of OrtModelCompilationOptions.
        /// </summary>
        /// <returns>true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.CompileApi.OrtReleaseModelCompilationOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }
}