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
    /// Holds provider options configuration for creating an InferenceSession.
    /// </summary>
    public static class ProviderOptions
    {
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
            cuda_options.has_user_compute_stream = 0;
            cuda_options.user_compute_stream = IntPtr.Zero;

            return cuda_options;
        }

        #endregion
    }
}