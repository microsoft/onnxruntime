// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    class NativeApiStatus
    {
        private static string GetErrorMessage(IntPtr /*(ONNXStatus*)*/status)
        {
            IntPtr nativeString = NativeMethods.OrtGetErrorMessage(status);
            string str = Marshal.PtrToStringAnsi(nativeString); //assumes charset = ANSI
            return str;
        }

        /// <summary>
        /// Checks the native Status if the errocode is OK/Success. Otherwise constructs an appropriate exception and throws.
        /// Releases the native status object, as needed.
        /// </summary>
        /// <param name="nativeStatus"></param>
        /// <throws></throws>
        public static void VerifySuccess(IntPtr nativeStatus)
        {
            if (nativeStatus != IntPtr.Zero)
            {
                ErrorCode statusCode = NativeMethods.OrtGetErrorCode(nativeStatus);
                string errorMessage = GetErrorMessage(nativeStatus);
                NativeMethods.OrtReleaseStatus(nativeStatus);
                throw new OnnxRuntimeException(statusCode, errorMessage);
            }
        }
    }
}
