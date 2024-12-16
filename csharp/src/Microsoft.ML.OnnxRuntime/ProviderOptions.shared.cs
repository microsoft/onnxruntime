// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for configuring a TensorRT Execution Provider instance
    /// </summary>
    public class OrtTensorRTProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        private int _deviceId = 0;
        private string _deviceIdStr = "device_id";

        #region Constructor

        /// <summary>
        /// Constructs an empty OrtTensorRTProviderOptions instance
        /// </summary>
        public OrtTensorRTProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorRTProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get TensorRT EP provider options
        /// </summary>
        /// <returns> return C# UTF-16 encoded string </returns>
        public string GetOptions()
        {
            var allocator = OrtAllocator.DefaultInstance;
            // Process provider options string
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorRTProviderOptionsAsString(handle,
                allocator.Pointer, out IntPtr providerOptions));
            return NativeOnnxValueHelper.StringFromNativeUtf8(providerOptions, allocator);
        }

        private static IntPtr UpdateTRTOptions(IntPtr handle, IntPtr[] keys, IntPtr[] values, UIntPtr count)
        {
            return NativeMethods.OrtUpdateTensorRTProviderOptions(handle, keys, values, count);
        }

        /// <summary>
        /// Updates  the configuration knobs of OrtTensorRTProviderOptions that will eventually be used to configure a TensorRT EP
        /// Please refer to the following on different key/value pairs to configure a TensorRT EP and their meaning:
        /// https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
        /// </summary>
        /// <param name="providerOptions">key/value pairs used to configure a TensorRT Execution Provider</param>
        public void UpdateOptions(Dictionary<string, string> providerOptions)
        {
            ProviderOptionsUpdater.Update(providerOptions, handle, UpdateTRTOptions);

            if (providerOptions.ContainsKey(_deviceIdStr))
            {
                _deviceId = Int32.Parse(providerOptions[_deviceIdStr]);
            }
        }

        /// <summary>
        /// Get device id of TensorRT EP.
        /// </summary>
        /// <returns> device id </returns>
        public int GetDeviceId()
        {
            return _deviceId;
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region Private Methods


        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtTensorRTProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseTensorRTProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }


    /// <summary>
    /// Holds the options for configuring a CUDA Execution Provider instance
    /// </summary>
    public class OrtCUDAProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }


        #region Constructor

        /// <summary>
        /// Constructs an empty OrtCUDAroviderOptions instance
        /// </summary>
        public OrtCUDAProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCUDAProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get CUDA EP provider options
        /// </summary>
        /// <returns> return C# UTF-16 encoded string </returns>
        public string GetOptions()
        {
            var allocator = OrtAllocator.DefaultInstance;
            // Process provider options string
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetCUDAProviderOptionsAsString(handle,
                allocator.Pointer, out IntPtr providerOptions));
            return NativeOnnxValueHelper.StringFromNativeUtf8(providerOptions, allocator);
        }

        private static IntPtr UpdateCUDAProviderOptions(IntPtr handle, IntPtr[] keys, IntPtr[] values, UIntPtr count)
        {
            return NativeMethods.OrtUpdateCUDAProviderOptions(handle, keys, values, count);
        }

        /// <summary>
        /// Updates  the configuration knobs of OrtCUDAProviderOptions that will eventually be used to configure a CUDA EP
        /// Please refer to the following on different key/value pairs to configure a CUDA EP and their meaning:
        /// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        /// </summary>
        /// <param name="providerOptions">key/value pairs used to configure a CUDA Execution Provider</param>
        public void UpdateOptions(Dictionary<string, string> providerOptions)
        {
            ProviderOptionsUpdater.Update(providerOptions, handle, UpdateCUDAProviderOptions);
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region Private Methods


        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtCUDAProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseCUDAProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }


    /// <summary>
    /// Holds the options for configuring a ROCm Execution Provider instance
    /// </summary>
    public class OrtROCMProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }


        #region Constructor

        /// <summary>
        /// Constructs an empty OrtROCMroviderOptions instance
        /// </summary>
        public OrtROCMProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateROCMProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get ROCm EP provider options
        /// </summary>
        /// <returns> return C# UTF-16 encoded string </returns>
        public string GetOptions()
        {
            var allocator = OrtAllocator.DefaultInstance;
            // Process provider options string
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetROCMProviderOptionsAsString(handle,
                allocator.Pointer, out IntPtr providerOptions));
            return NativeOnnxValueHelper.StringFromNativeUtf8(providerOptions, allocator);
        }

        private static IntPtr UpdateROCMProviderOptions(IntPtr handle, IntPtr[] keys, IntPtr[] values, UIntPtr count)
        {
            return NativeMethods.OrtUpdateROCMProviderOptions(handle, keys, values, count);
        }

        /// <summary>
        /// Updates  the configuration knobs of OrtROCMProviderOptions that will eventually be used to configure a ROCm EP
        /// Please refer to the following on different key/value pairs to configure a ROCm EP and their meaning:
        /// https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
        /// </summary>
        /// <param name="providerOptions">key/value pairs used to configure a ROCm Execution Provider</param>
        public void UpdateOptions(Dictionary<string, string> providerOptions)
        {
            ProviderOptionsUpdater.Update(providerOptions, handle, UpdateROCMProviderOptions);
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtROCMProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseROCMProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }


    /// <summary>
    /// This helper class contains methods to handle values of provider options
    /// </summary>
    public class ProviderOptionsValueHelper
    {
        /// <summary>
        /// Parse from string and save to dictionary
        /// </summary>
        /// <param name="s">C# string</param>
        /// <param name="dict">Dictionary instance to store the parsing result of s</param>
        public static void StringToDict(string s, Dictionary<string, string> dict)
        {
            string[] paris = s.Split(';');

            foreach (var p in paris)
            {
                string[] keyValue = p.Split('=');
                if (keyValue.Length != 2)
                {
                    throw new ArgumentException("Make sure input string contains key-value paris, e.g. key1=value1;key2=value2...", "s");
                }
                dict.Add(keyValue[0], keyValue[1]);
            }
        }
    }

    /// <summary>
    /// CoreML flags for use with SessionOptions.
    /// See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
    /// </summary>
    [Flags]
    public enum CoreMLFlags : uint
    {
        COREML_FLAG_USE_NONE = 0x000,
        COREML_FLAG_USE_CPU_ONLY = 0x001,
        COREML_FLAG_ENABLE_ON_SUBGRAPH = 0x002,
        COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004,
        COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008,
        COREML_FLAG_CREATE_MLPROGRAM = 0x010,
        COREML_FLAG_USE_CPU_AND_GPU = 0x020,
        COREML_FLAG_LAST = COREML_FLAG_USE_CPU_AND_GPU,
    }

    /// <summary>
    /// NNAPI flags for use with SessionOptions.
    /// See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h
    /// </summary>
    [Flags]
    public enum NnapiFlags
    {
        NNAPI_FLAG_USE_NONE = 0x000,
        NNAPI_FLAG_USE_FP16 = 0x001,
        NNAPI_FLAG_USE_NCHW = 0x002,
        NNAPI_FLAG_CPU_DISABLED = 0x004,
        NNAPI_FLAG_CPU_ONLY = 0x008,
        NNAPI_FLAG_LAST = NNAPI_FLAG_CPU_ONLY
    }


}
