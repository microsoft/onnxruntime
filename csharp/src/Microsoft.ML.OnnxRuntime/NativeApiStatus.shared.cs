// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.OnnxRuntime
{
    class NativeApiStatus
    {
        private static string GetErrorMessage(IntPtr /*(ONNXStatus*)*/status)
        {
            // nativeString belongs to status, no need for separate release
            IntPtr nativeString = NativeMethods.OrtGetErrorMessage(status);
            return NativeOnnxValueHelper.StringFromNativeUtf8(nativeString);
        }

        /// <summary>
        /// Checks the native Status if the errocode is OK/Success. Otherwise constructs an appropriate exception and throws.
        /// Releases the native status object, as needed.
        /// </summary>
        /// <param name="nativeStatus"></param>
        /// <throws></throws>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VerifySuccess(IntPtr nativeStatus)
        {
            if (nativeStatus != IntPtr.Zero)
            {
                try
                {
                    ErrorCode statusCode = NativeMethods.OrtGetErrorCode(nativeStatus);
                    string errorMessage = GetErrorMessage(nativeStatus);
                    throw new OnnxRuntimeException(statusCode, errorMessage);
                }
                finally
                {
                    NativeMethods.OrtReleaseStatus(nativeStatus);
                }
            }
        }
    }
}
