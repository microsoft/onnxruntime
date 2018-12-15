// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;


namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Various providers of ONNX operators
    /// </summary>
    public enum ExecutionProvider
    {
        Cpu,
        MklDnn
        //TODO: add more providers gradually
    };

    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions:IDisposable
    {
        protected SafeHandle _nativeOption;
        protected static readonly Lazy<SessionOptions> _default = new Lazy<SessionOptions>(MakeSessionOptionWithMklDnnProvider);

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
        {
            _nativeOption = new NativeOnnxObjectHandle(NativeMethods.OrtCreateSessionOptions());
        }

        /// <summary>
        /// Default instance
        /// </summary>
        public static SessionOptions Default
        {
            get
            {
                return _default.Value;
            }
        }

        /// <summary>
        /// Append an execution propvider. When any operator is evaluated, it is executed on the first execution provider that provides it
        /// </summary>
        /// <param name="provider"></param>
        public void AppendExecutionProvider(ExecutionProvider provider)
        {
            switch (provider)
            {
                case ExecutionProvider.Cpu:
                    AppendExecutionProvider(CpuExecutionProviderFactory.Default);
                    break;
                case ExecutionProvider.MklDnn:
                    AppendExecutionProvider(MklDnnExecutionProviderFactory.Default);
                    break;
                default:
                    break;
            }
        }


        private static SessionOptions MakeSessionOptionWithMklDnnProvider()
        {
            SessionOptions options = new SessionOptions();
            options.AppendExecutionProvider(MklDnnExecutionProviderFactory.Default);
            options.AppendExecutionProvider(CpuExecutionProviderFactory.Default);

            return options;
        }


        internal IntPtr NativeHandle
        {
            get
            {
                return _nativeOption.DangerousGetHandle(); //Note: this is unsafe, and not ref counted, use with caution
            }
        }

        private void AppendExecutionProvider(NativeOnnxObjectHandle providerFactory)
        {
            unsafe
            {
                bool success = false;
                providerFactory.DangerousAddRef(ref success);
                if (success)
                {
                    NativeMethods.OrtSessionOptionsAppendExecutionProvider(_nativeOption.DangerousGetHandle(), providerFactory.DangerousGetHandle());
                    providerFactory.DangerousRelease();
                }

            }
        }

        #region destructors disposers


        ~SessionOptions()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // cleanup managed resources
            }
            _nativeOption.Dispose();
        }

        #endregion
    }
}
