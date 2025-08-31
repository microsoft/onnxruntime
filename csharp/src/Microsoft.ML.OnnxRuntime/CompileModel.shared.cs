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

        /// <summary>
        /// Sets information related to EP context binary file. The Ep uses this information to decide the
        /// location and context binary file name when compiling with both the input and output models
        /// stored in buffers.
        /// </summary>
        /// <param name="outputDirectory">Path to the model directory.</param>
        /// <param name="modelName">The name of the model.</param>
        public void SetEpContextBinaryInformation(string outputDirectory, string modelName)
        {
            var platformOutputDirectory = NativeOnnxValueHelper.GetPlatformSerializedString(outputDirectory);
            var platformModelName = NativeOnnxValueHelper.GetPlatformSerializedString(modelName);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetEpContextBinaryInformation(
                    handle, platformOutputDirectory, platformModelName));
        }

        /// <summary>
        /// Delegate to write/save a buffer containing ONNX model bytes to a custom destination.
        /// </summary>
        /// <param name="buffer">The buffer to write out.</param>
        public delegate void WriteBufferDelegate(ReadOnlySpan<byte> buffer);

        /// <summary>
        /// Sets a delegate that is called by ORT to write out the output model's serialized ONNX bytes.
        /// The provided delegate may be called repeatedly until the entire output model has been written out.
        /// Each call to the delegate is expected to consume/handle the entire input buffer.
        /// </summary>
        /// <param name="writeBufferDelegate">The delegate called by ORT to write out the model.</param>
        public void SetOutputModelWriteBufferDelegate(WriteBufferDelegate writeBufferDelegate)
        {
            _writeBufferDelegateState?.Dispose();
            _writeBufferDelegateState =
                new DelegateResources<WriteBufferConnector, NativeMethods.DOrtWriteBufferDelegate>(
                    new WriteBufferConnector(writeBufferDelegate),
                    new NativeMethods.DOrtWriteBufferDelegate(WriteBufferConnector.WriteBufferDelegateWrapper));

            IntPtr funcPtr = _writeBufferDelegateState.GetFunctionPointerForDelegate();

            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelWriteFunc(
                    handle,
                    funcPtr,
                    _writeBufferDelegateState.GetConnectorHandleAsPointer()));
        }

        // TODO: Add return value and other parameters
        /// <summary>
        /// Delegate called by ORT for every initializer when generating the compiled model.
        /// The delegate allows the user to determine whether the initializer should be stored within the compiled
        /// model or externally in a file. If the delegate chooses to store an initializer externally, the delegate
        /// implementation is responsible for writing the initializer data to a file.
        /// </summary>
        /// <param name="initializerName">The initializer's name.</param>
        public delegate void HandleInitializerDelegate(string initializerName);

        /// <summary>
        /// Sets a delegate that is called by ORT for every initializer when generating the compiled model.
        /// The delegate allows the user to determine whether the initializer should be stored within the compiled
        /// model or externally in a file. If the delegate chooses to store an initializer externally, the delegate
        /// implementation is responsible for writing the initializer data to a file.
        /// </summary>
        /// <param name="handleInitializerDelegate">The delegate called by ORT for every initializer.</param>
        public void SetOutputModelHandleInitializerDelegate(HandleInitializerDelegate handleInitializerDelegate)
        {
            _handleInitializerDelegateState?.Dispose();
            _handleInitializerDelegateState =
                new DelegateResources<HandleInitializerConnector, NativeMethods.DOrtHandleInitializerDataDelegate>(
                    new HandleInitializerConnector(handleInitializerDelegate),
                    new NativeMethods.DOrtHandleInitializerDataDelegate(
                        HandleInitializerConnector.HandleInitializerDelegateWrapper));

            IntPtr funcPtr = _handleInitializerDelegateState.GetFunctionPointerForDelegate();

            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelHandleInitializerFunc(
                    handle,
                    funcPtr,
                    _handleInitializerDelegateState.GetConnectorHandleAsPointer()));
        }

        #region Delegate helpers
        /// <summary>
        /// Class to bridge the C# and native worlds for the "write buffer" delegate
        /// </summary>
        private class WriteBufferConnector
        {
            private readonly WriteBufferDelegate _csharpDelegate;

            internal WriteBufferConnector(WriteBufferDelegate writeBufferDelegate)
            {
                _csharpDelegate = writeBufferDelegate;
            }

            public static IntPtr WriteBufferDelegateWrapper(IntPtr /* void* */ state,
                                                            IntPtr /* const void* */ buffer,
                                                            UIntPtr /* size_t */ bufferNumBytes)
            {
                try
                {

                    WriteBufferConnector connector = (WriteBufferConnector)GCHandle.FromIntPtr(state).Target;
                    ReadOnlySpan<byte> bufferSpan;

                    unsafe
                    {
                        // NOTE: A Span<byte> can only view 2GB of data. This is fine because ORT does not write out
                        // chunks that large. However, if we ever need to, the solution is to just write a loop here
                        // that repeatedly calls the delegate with smaller chunks of data.
                        bufferSpan = new ReadOnlySpan<byte>(buffer.ToPointer(), checked((int)bufferNumBytes));
                    }

                    connector._csharpDelegate(bufferSpan);
                }
                catch (Exception ex)
                {
                    var error = $"The C# WriteBuffer delegate threw an exception: {ex.Message}";
                    IntPtr status = NativeMethods.OrtCreateStatus((uint)ErrorCode.Fail,
                                            NativeOnnxValueHelper.StringToZeroTerminatedUtf8(error));
                    return status;
                }

                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Class to bridge the C# and native worlds for the "write buffer" delegate
        /// </summary>
        private class HandleInitializerConnector
        {
            private readonly HandleInitializerDelegate _csharpDelegate;

            internal HandleInitializerConnector(HandleInitializerDelegate handleInitializerDelegate)
            {
                _csharpDelegate = handleInitializerDelegate;
            }

            public static IntPtr HandleInitializerDelegateWrapper(
                IntPtr /* void* */ state,
                IntPtr /* const char* */ initializerName,
                IntPtr /* const OrtValue* */ initializerValue,
                IntPtr /* const OrtExternalInitializerInfo* */ externalInfo,
                out IntPtr /* OrtExternalInitializerInfo** */ newExternalInfo)
            {
                newExternalInfo = IntPtr.Zero;

                try
                {

                    HandleInitializerConnector connector = (HandleInitializerConnector)GCHandle.FromIntPtr(state).Target;
                    var utf8InitializerName = NativeOnnxValueHelper.StringFromNativeUtf8(initializerName);
                    connector._csharpDelegate(utf8InitializerName);
                }
                catch (Exception ex)
                {
                    var error = $"The C# HandleInitializer delegate threw an exception: {ex.Message}";
                    IntPtr status = NativeMethods.OrtCreateStatus((uint)ErrorCode.Fail,
                                            NativeOnnxValueHelper.StringToZeroTerminatedUtf8(error));
                    return status;
                }

                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Disposable class that stores resources for a delegate provided by the user.
        /// </summary>
        /// <typeparam name="Connector">The type of the connector class (e.g., WriteBufferConnector)</typeparam>
        /// <typeparam name="Delegate">The type of the native delegate.</typeparam>
        private class DelegateResources<Connector, Delegate> : IDisposable
            where Connector : class
            where Delegate : class
        {
            public DelegateResources(Connector connector, Delegate @delegate)
            {
                _connector = connector;
                _delegate = @delegate;
                _connectorHandle = GCHandle.Alloc(_connector);
                _delegateHandle = GCHandle.Alloc(_delegate);
            }

            public IntPtr GetFunctionPointerForDelegate()
            {
                return Marshal.GetFunctionPointerForDelegate(_delegate);
            }

            public IntPtr GetConnectorHandleAsPointer()
            {
                return GCHandle.ToIntPtr(_connectorHandle);
            }

            #region IDispose implementation
            public void Dispose()
            {
                DisposeImpl();
                GC.SuppressFinalize(this);
            }

            protected virtual void DisposeImpl()
            {
                if (_disposed)
                {
                    return;
                }

                if (_connectorHandle.IsAllocated)
                {
                    _connectorHandle.Free();
                    _connector = null;
                }

                if (_delegateHandle.IsAllocated)
                {
                    _delegateHandle.Free();
                    _delegate = null;
                }

                _disposed = true;
            }

            ~DelegateResources()
            {
                DisposeImpl();
            }
            #endregion

            private Connector _connector = null;
            private Delegate _delegate = null;
            private GCHandle _connectorHandle = default;
            private GCHandle _delegateHandle = default;
            private bool _disposed = false;
        }
        #endregion

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

            _writeBufferDelegateState?.Dispose();
            _handleInitializerDelegateState?.Dispose();
            return true;
        }

        /// <summary>
        /// Stores delegate state for the "write buffer" delegate.
        /// </summary>
        private DelegateResources<WriteBufferConnector, NativeMethods.DOrtWriteBufferDelegate>
            _writeBufferDelegateState = null;

        /// <summary>
        /// Stores delegate state for the "handle initializer" delegate.
        /// </summary>
        private DelegateResources<HandleInitializerConnector, NativeMethods.DOrtHandleInitializerDataDelegate>
            _handleInitializerDelegateState = null;
    }
}