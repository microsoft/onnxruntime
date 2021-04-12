// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Search Algorithm for Cudnn Conv.
    /// </summary>
    public enum OrtCudnnConvAlgoSearch
    {
        EXHAUSTIVE,  //!< expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
        HEURISTIC,   //!< lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
        DEFAULT,     //!< default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    }

    /// <summary>
    /// Provider options for CUDA EP.
    /// </summary>
    public struct OrtCUDAProviderOptions
    {
        public int device_id;                                   //!< cuda device with id=0 as default device.
        public OrtCudnnConvAlgoSearch cudnn_conv_algo_search;   //!< cudnn conv algo search option
        public UIntPtr gpu_mem_limit;                           //!< default cuda memory limitation to maximum finite value of size_t.
        public int arena_extend_strategy;                       //!< default area extend strategy to KNextPowerOfTwo.
        public int do_copy_in_default_stream;                   //!< Whether to do copies in the default stream or use separate streams.
    }

    /// <summary>
    /// Holds provider options configuration for creating an InferenceSession.
    /// </summary>
    public class ProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty ProviderOptions
        /// </summary>
        public ProviderOptions()
            : base(IntPtr.Zero, true)
        {
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get CUDA provider options with default setting.
        /// </summary>
        /// <returns> CUDA provider options instance.  </returns>
        public static OrtCUDAProviderOptions GetDefaultCUDAProviderOptions()
        {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch.EXHAUSTIVE;
            if (IntPtr.Size == 8)
            {
                cuda_options.gpu_mem_limit = (UIntPtr)UInt64.MaxValue;
            }
            else
            {
                cuda_options.gpu_mem_limit = (UIntPtr)UInt32.MaxValue;
            }
            cuda_options.arena_extend_strategy = 0;
            cuda_options.do_copy_in_default_stream = 1;

            return cuda_options;
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
        /// the native instance of SessionOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}