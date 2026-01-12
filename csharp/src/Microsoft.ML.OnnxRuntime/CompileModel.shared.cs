// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Diagnostics;
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
    public class OrtModelCompilationOptions : IDisposable
    {
        /// <summary>
        /// Create a new OrtModelCompilationOptions object from SessionOptions.
        /// </summary>
        /// <remarks>By default, the GraphOptimizationLevel is set to ORT_DISABLE_ALL. Use SetGraphOptimizationLevel()
        /// to enable graph optimizations.</remarks>
        /// <param name="sessionOptions">SessionOptions instance to read settings from.</param>
        public OrtModelCompilationOptions(SessionOptions sessionOptions)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtCreateModelCompilationOptionsFromSessionOptions(
                    OrtEnv.Instance().Handle, sessionOptions.Handle, out _handle));
        }

        /// <summary>
        /// Compile the model using the options set in this object.
        /// </summary>
        public void CompileModel()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.CompileApi.OrtCompileModel(OrtEnv.Instance().Handle, _handle));
        }


        /// <summary>
        /// Set the input model to compile.
        /// </summary>
        /// <param name="path">Path to ONNX model to compile.</param>
        public void SetInputModelPath(string path)
        {
            var platformPath = NativeOnnxValueHelper.GetPlatformSerializedString(path);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetInputModelPath(_handle, platformPath));
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
                    _handle, buffer, (UIntPtr)buffer.Length));
        }

        /// <summary>
        /// Set the path to write the compiled ONNX model to.
        /// </summary>
        /// <param name="path">Path to write compiled model to.</param>
        public void SetOutputModelPath(string path)
        {
            var platformPath = NativeOnnxValueHelper.GetPlatformSerializedString(path);
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelPath(_handle, platformPath));

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
                    _handle, platformPath, new UIntPtr(threshold)));
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
                    _handle, allocator.Pointer, ref outputModelBufferPtr, ref outputModelBufferSizePtr));
        }

        /// <summary>
        /// Enables or disables the embedding of EPContext binary data into the `ep_cache_context` attribute
        /// of EPContext nodes.
        /// </summary>
        /// <param name="embed">Enable if true. Default is false.</param>
        public void SetEpContextEmbedMode(bool embed)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetEpContextEmbedMode(_handle, embed));
        }

        /// <summary>
        /// Sets flags from OrtCompileApiFlags that represent one or more boolean options to enable.
        /// </summary>
        /// <param name="flags">bitwise OR of flags in OrtCompileApiFlags to enable.</param>
        public void SetFlags(OrtCompileApiFlags flags)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetFlags(_handle, (uint)flags));
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
                    _handle, platformOutputDirectory, platformModelName));
        }

        /// <summary>
        /// Sets the graph optimization level. Defaults to ORT_DISABLE_ALL if not specified.
        /// </summary>
        /// <param name="graphOptimizationLevel">The graph optimization level to set.</param>
        public void SetGraphOptimizationLevel(GraphOptimizationLevel graphOptimizationLevel)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetGraphOptimizationLevel(
                    _handle, graphOptimizationLevel));
        }

        /// <summary>
        /// Delegate to write/save a buffer containing ONNX model bytes to a custom destination. The delegate
        /// may be called repeatedly until the entire output model has been written out. Each call to the delegate
        /// is expected to consume the entire buffer.
        /// </summary>
        /// <param name="buffer">The buffer to write out.</param>
        /// <see cref="OrtModelCompilationOptions.SetOutputModelWriteDelegate"/>
        public delegate void WriteBufferToDestinationDelegate(ReadOnlySpan<byte> buffer);

        /// <summary>
        /// Sets a delegate that is called by ORT to write out the output model's serialized ONNX bytes.
        /// The provided delegate may be called repeatedly until the entire output model has been written out.
        /// Each call to the delegate is expected to consume/handle the entire input buffer.
        /// </summary>
        /// <param name="writeBufferDelegate">The delegate called by ORT to write out the model.</param>
        public void SetOutputModelWriteDelegate(WriteBufferToDestinationDelegate writeBufferDelegate)
        {
            _writeBufferToDestinationDelegateState?.Dispose();
            _writeBufferToDestinationDelegateState =
                new DelegateResources<WriteBufferToDestinationConnector,
                                      NativeMethods.DOrtWriteBufferToDestinationDelegate>(
                    new WriteBufferToDestinationConnector(writeBufferDelegate),
                    new NativeMethods.DOrtWriteBufferToDestinationDelegate(
                        WriteBufferToDestinationConnector.WriteBufferToDestinationDelegateWrapper));

            IntPtr funcPtr = _writeBufferToDestinationDelegateState.GetFunctionPointerForDelegate();

            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelWriteFunc(
                    _handle,
                    funcPtr,
                    _writeBufferToDestinationDelegateState.GetConnectorHandleAsPointer()));
        }

        /// <summary>
        /// Delegate called by ORT for every initializer when generating the compiled model.
        /// The delegate allows the user to determine whether the initializer should be stored within the compiled
        /// model or externally in a file. If the delegate chooses to store an initializer externally, the delegate
        /// implementation is responsible for writing the initializer data to a file.
        /// </summary>
        /// <param name="initializerName">The initializer's name.</param>
        /// <param name="initializerValue">The readonly OrtValue instance containing the data, type, and
        /// shape of the initializer.</param>
        /// <param name="originalInitializerLocation">May be null. If the initializer is originally stored externally,
        /// this contains the file path, file offset, and data size. Otherwise, this is null.</param>
        /// <returns>A new OrtExternalInitializerInfo indicating the new location of the initializer.
        /// Returns null if the initializer should be stored within the generated compiled model.</returns>
        /// <remarks>The return value may be null.</remarks>
        /// <see cref="OrtModelCompilationOptions.SetOutputModelGetInitializerLocationDelegate"/>
        public delegate OrtExternalInitializerInfo GetInitializerLocationDelegate(
            string initializerName,
            IReadOnlyOrtValue initializerValue,
            IReadOnlyExternalInitializerInfo originalInitializerLocation);

        /// <summary>
        /// Sets a delegate that is called by ORT for every initializer when generating the compiled model.
        /// The delegate allows the user to determine whether the initializer should be stored within the compiled
        /// model or externally in a file. If the delegate chooses to store an initializer externally, the delegate
        /// implementation is responsible for writing the initializer data to a file.
        /// </summary>
        /// <param name="getInitializerLocationDelegate">The delegate called by ORT for every initializer.</param>
        public void SetOutputModelGetInitializerLocationDelegate(
            GetInitializerLocationDelegate getInitializerLocationDelegate)
        {
            _getInitializerLocationDelegateState?.Dispose();
            _getInitializerLocationDelegateState =
                new DelegateResources<GetInitializerLocationConnector,
                                      NativeMethods.DOrtGetInitializerLocationDelegate>(
                    new GetInitializerLocationConnector(getInitializerLocationDelegate),
                    new NativeMethods.DOrtGetInitializerLocationDelegate(
                        GetInitializerLocationConnector.GetInitializerLocationDelegateWrapper));

            IntPtr funcPtr = _getInitializerLocationDelegateState.GetFunctionPointerForDelegate();

            NativeApiStatus.VerifySuccess(
                NativeMethods.CompileApi.OrtModelCompilationOptions_SetOutputModelGetInitializerLocationFunc(
                    _handle,
                    funcPtr,
                    _getInitializerLocationDelegateState.GetConnectorHandleAsPointer()));
        }

        #region Delegate helpers
        /// <summary>
        /// Class to bridge the C# and native worlds for the "write buffer to destination" delegate
        /// </summary>
        private class WriteBufferToDestinationConnector
        {
            private readonly WriteBufferToDestinationDelegate _userDelegate;

            internal WriteBufferToDestinationConnector(WriteBufferToDestinationDelegate writeBufferDelegate)
            {
                _userDelegate = writeBufferDelegate;
            }

            public static IntPtr WriteBufferToDestinationDelegateWrapper(IntPtr /* void* */ state,
                                                                         IntPtr /* const void* */ buffer,
                                                                         UIntPtr /* size_t */ bufferNumBytes)
            {
                try
                {

                    WriteBufferToDestinationConnector connector = (WriteBufferToDestinationConnector)
                        GCHandle.FromIntPtr(state).Target;
                    ReadOnlySpan<byte> bufferSpan;

                    unsafe
                    {
                        // NOTE: A Span<byte> can only view 2GB of data. This is fine because ORT does not write out
                        // chunks that large. However, if we ever need to, the solution is to just write a loop here
                        // that repeatedly calls the delegate with smaller chunks of data.
                        bufferSpan = new ReadOnlySpan<byte>(buffer.ToPointer(), checked((int)bufferNumBytes));
                    }

                    connector._userDelegate(bufferSpan);
                }
                catch (Exception ex)
                {
                    var error = $"The C# WriteBufferToDestination delegate threw an exception: {ex.Message}";
                    IntPtr status = NativeMethods.OrtCreateStatus((uint)ErrorCode.Fail,
                                            NativeOnnxValueHelper.StringToZeroTerminatedUtf8(error));
                    return status;
                }

                return IntPtr.Zero;
            }
        }

        /// <summary>
        /// Class to bridge the C# and native worlds for the "get initializer location" delegate
        /// </summary>
        private class GetInitializerLocationConnector
        {
            private readonly GetInitializerLocationDelegate _userDelegate;

            internal GetInitializerLocationConnector(GetInitializerLocationDelegate getInitializerLocationDelegate)
            {
                _userDelegate = getInitializerLocationDelegate;
            }

            public static IntPtr GetInitializerLocationDelegateWrapper(
                IntPtr /* void* */ state,
                IntPtr /* const char* */ initializerName,
                IntPtr /* const OrtValue* */ initializerValue,
                IntPtr /* const OrtExternalInitializerInfo* */ originalInitializerLocation,
                out IntPtr /* OrtExternalInitializerInfo** */ newInitializerLocationOutput)
            {
                newInitializerLocationOutput = IntPtr.Zero;

                try
                {

                    GetInitializerLocationConnector connector = (GetInitializerLocationConnector)GCHandle.
                        FromIntPtr(state).Target;
                    string utf8InitializerName = NativeOnnxValueHelper.StringFromNativeUtf8(initializerName);
                    IReadOnlyOrtValue readOnlyInitializerValue = new OrtValue(initializerValue, owned: false);
                    IReadOnlyExternalInitializerInfo readOnlyOriginalInitializerLocation = null;

                    if (originalInitializerLocation != IntPtr.Zero)
                    {
                        readOnlyOriginalInitializerLocation = new OrtExternalInitializerInfo(
                            originalInitializerLocation, ownsHandle: false);
                    }

                    // Call user's delegate, which may return the new location of the initializer.
                    OrtExternalInitializerInfo newInitializerLocation = connector._userDelegate(
                        utf8InitializerName, readOnlyInitializerValue, readOnlyOriginalInitializerLocation);

                    if (newInitializerLocation != null)
                    {
                        // Delegate returned info about a new location for the initializer.
                        // Can't guarantee that the new external info returned by user's delegate is not referenced
                        // by other C# code. ORT expects to own the new external info, so create a copy here and
                        // give it to ORT.
                        string newFilePath = newInitializerLocation.GetFilePath();
                        byte[] newFilePathBytes = NativeOnnxValueHelper.GetPlatformSerializedString(newFilePath);

                        IntPtr status = NativeMethods.OrtCreateExternalInitializerInfo(
                            newFilePathBytes,
                            newInitializerLocation.GetFileOffset(),
                            (UIntPtr)newInitializerLocation.GetByteSize(),
                            out newInitializerLocationOutput);

                        if (status != IntPtr.Zero)
                        {
                            return status;
                        }
                    }
                    else
                    {
                        // User's delegate did not return a new location for the initializer. ORT will store initializer
                        // within the generated compiled model.
                        newInitializerLocationOutput = IntPtr.Zero;
                    }
                }
                catch (Exception ex)
                {
                    var error = $"The C# GetInitializerLocation delegate threw an exception: {ex.Message}";
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
        /// <typeparam name="Connector">The type of the connector class
        /// (e.g., WriteBufferToDestinationConnector)</typeparam>
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

            internal IntPtr GetFunctionPointerForDelegate()
            {
                return Marshal.GetFunctionPointerForDelegate(_delegate);
            }

            internal IntPtr GetConnectorHandleAsPointer()
            {
                return GCHandle.ToIntPtr(_connectorHandle);
            }

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (_disposed)
                {
                    return;
                }

                if (disposing)
                {
                    // Dispose other children disposables. We have none.
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
                Dispose(false);
            }

            private Connector _connector = null;
            private Delegate _delegate = null;
            private GCHandle _connectorHandle = default;
            private GCHandle _delegateHandle = default;
            private bool _disposed = false;
        }
        #endregion

        #region IDispose implementation
        /// <summary>
        /// IDispose implementation.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDispose implementation
        /// </summary>
        /// <param name="disposing">True if Dispose() has been called by the user-side code. False if
        /// called by the runtime from inside the finalizer.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                _writeBufferToDestinationDelegateState?.Dispose();
                _getInitializerLocationDelegateState?.Dispose();
            }

            Debug.Assert(_handle != IntPtr.Zero);
            NativeMethods.CompileApi.OrtReleaseModelCompilationOptions(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }

        /// <summary>
        /// Finalizer that releases the native handle if not already released by Dispose().
        /// </summary>
        ~OrtModelCompilationOptions()
        {
            Dispose(false);
        }
        #endregion

        /// <summary>
        /// Handle to the native OrtModelCompilationOptions object.
        /// </summary>
        private IntPtr _handle;

        /// <summary>
        /// True if this OrtModelCompilationOptions instance has already been disposed.
        /// </summary>
        private bool _disposed = false;

        /// <summary>
        /// Stores delegate state for the "write buffer to destination" delegate.
        /// </summary>
        private DelegateResources<WriteBufferToDestinationConnector, NativeMethods.DOrtWriteBufferToDestinationDelegate>
            _writeBufferToDestinationDelegateState = null;

        /// <summary>
        /// Stores delegate state for the "get initializer location" delegate.
        /// </summary>
        private DelegateResources<GetInitializerLocationConnector, NativeMethods.DOrtGetInitializerLocationDelegate>
            _getInitializerLocationDelegateState = null;
    }
}
