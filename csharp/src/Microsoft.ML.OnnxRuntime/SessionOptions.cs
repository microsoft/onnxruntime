// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;


namespace Microsoft.ML.OnnxRuntime
{
    public enum ExecutionProvider
    {
        Cpu,
        MklDnn
        //TODO: add more providers gradually
    };

    public class SessionOptions
    {
        protected SafeHandle _nativeOption;
        protected static readonly Lazy<SessionOptions> _default = new Lazy<SessionOptions>(MakeSessionOptionWithMklDnnProvider);

        public SessionOptions()
        {
            _nativeOption = new NativeOnnxObjectHandle(NativeMethods.ONNXRuntimeCreateSessionOptions());
        }

        public static SessionOptions Default
        {
            get
            {
                return _default.Value;
            }
        }

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
                    NativeMethods.ONNXRuntimeSessionOptionsAppendExecutionProvider(_nativeOption.DangerousGetHandle(), providerFactory.DangerousGetHandle());
                    providerFactory.DangerousRelease();
                }

            }
        }
    }
}
