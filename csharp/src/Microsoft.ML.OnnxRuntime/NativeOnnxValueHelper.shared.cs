// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This helper class contains methods to create native OrtValue from a managed value object
    /// </summary>
    internal static class NativeOnnxValueHelper
    {
        /// <summary>
        /// Converts C# UTF-16 string to UTF-8 zero terminated
        /// byte[] instance
        /// </summary>
        /// <param name="s">string to be converted</param>
        /// <returns>UTF-8 encoded equivalent</returns>
        internal static byte[] StringToZeroTerminatedUtf8(string s)
        {
            int arraySize = UTF8Encoding.UTF8.GetByteCount(s);
            byte[] utf8Bytes = new byte[arraySize + 1];
            var bytesWritten = UTF8Encoding.UTF8.GetBytes(s, 0, s.Length, utf8Bytes, 0);
            Debug.Assert(arraySize == bytesWritten);
            utf8Bytes[utf8Bytes.Length - 1] = 0;
            return utf8Bytes;
        }

        /// <summary>
        /// This function converts the input string into UTF-8 encoding string (no zero termination)
        /// straight into the pre-allocated native buffer. The buffer size
        /// must match the required size and can be obtained in advance with
        /// System.Text.Encoding.UTF8.GetByteCount(s).
        /// 
        /// The function is helpful when we populate native string tensor buffers directly where
        /// the elements stored do not have zero terminator.
        /// </summary>
        /// <param name="s">managed string</param>
        /// <param name="ptr">natively allocated buffer</param>
        /// <param name="totalBytesToWrite">pre-allocated buffer size</param>
        internal static void StringToUtf8NativeMemory(string s, IntPtr ptr, int totalBytesToWrite)
        {
            // We throw here, because we are expecting the caller to properly calculate and allocate the right size buffer
            unsafe
            {
                fixed (char* c = s) // create char pointer to start of managed string.
                {
                    var nativeBytes = (byte*)ptr; // get managed byte* from native intptr
                    var bytesWritten = Encoding.UTF8.GetBytes(c, s.Length, nativeBytes, totalBytesToWrite); // total bytes to write is size of native memory buffer
                    if (bytesWritten != totalBytesToWrite)
                    {
                        throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                            $"Failed to convert to UTF8. Expected bytes: {totalBytesToWrite}, written: {bytesWritten}");
                    }
                }
            }
        }

        /// <summary>
        /// Reads UTF-8 encode string from a C zero terminated string
        /// and converts it into a C# UTF-16 encoded string
        /// </summary>
        /// <param name="nativeUtf8">pointer to native or pinned memory where Utf-8 resides</param>
        /// <param name="allocator">optional allocator to free nativeUtf8 if it was allocated by OrtAllocator</param>
        /// <returns></returns>
        internal static string StringFromNativeUtf8(IntPtr nativeUtf8, OrtAllocator allocator = null)
        {
            try
            {
                unsafe
                {
                    int len = 0;
                    while (*(byte*)(nativeUtf8 + len) != 0) ++len;

                    if (len == 0)
                    {
                        return string.Empty;
                    }
                    var nativeBytes = (byte*)nativeUtf8;
                    return Encoding.UTF8.GetString(nativeBytes, len);
                }
            }
            finally
            {
                allocator?.FreeMemory(nativeUtf8);
            }
        }

        /// <summary>
        /// Reads UTF-8 string from native C zero terminated string,
        /// makes a copy of it on unmanaged heap and converts it to C# UTF-16 string,
        /// then returns both C# string and the unmanaged copy of the UTF-8 string.
        /// 
        /// On return it deallocates the nativeUtf8 string using the specified allocator
        /// </summary>
        /// <param name="allocator">allocator to use to free nativeUtf8</param>
        /// <param name="nativeUtf8">input</param>
        /// <param name="str">C# UTF-16 string</param>
        /// <param name="utf8">UTF-8 bytes in a unmanaged allocation, zero terminated</param>
        internal static void StringAndUtf8FromNative(OrtAllocator allocator, IntPtr nativeUtf8, out string str, out IntPtr utf8)
        {
            try
            {
                unsafe
                {
                    int len = 0;
                    while (*(byte*)(nativeUtf8 + len) != 0) ++len;

                    if (len == 0)
                    {
                        str = string.Empty;
                        utf8 = IntPtr.Zero;
                        return;
                    }

                    var src = new Span<byte>((nativeUtf8).ToPointer(), len);
                    utf8 = Marshal.AllocHGlobal(len + 1);
                    try
                    {
                        // Make a copy of the UTF-8 bytes and add a zero terminator
                        // on unmanaged heap
                        var dest = new Span<byte>((utf8).ToPointer(), len + 1);
                        src.CopyTo(dest);
                        dest[len] = 0;
                        var nativeBytes = (byte*)nativeUtf8;
                        str = Encoding.UTF8.GetString(nativeBytes, len);
                    }
                    catch (Exception)
                    {
                        Marshal.FreeHGlobal(utf8);
                        throw;
                    }
                }
            }
            finally
            {
                allocator.FreeMemory(nativeUtf8);
            }
        }

        /// <summary>
        /// Converts C# UTF-16 string to UTF-8 zero terminated
        /// byte[] instance
        /// </summary>
        /// <param name="str">string to be converted</param>
        /// <returns>UTF-8 encoded equivalent</returns>
        internal static byte[] GetPlatformSerializedString(string str)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return System.Text.Encoding.Unicode.GetBytes(str + Char.MinValue);
            else
                return StringToZeroTerminatedUtf8(str);
        }
    }

    internal static class TensorElementTypeConverter
    {
        public static bool GetTypeAndWidth(TensorElementType elemType, out Type type, out int width)
        {
            bool result = true;
            TensorElementTypeInfo typeInfo = TensorBase.GetElementTypeInfo(elemType);
            if (typeInfo != null)
            {
                type = typeInfo.TensorType;
                width = typeInfo.TypeSize;
            }
            else
            {
                type = null;
                width = 0;
                result = false;
            }
            return result;
        }
    }

    /// <summary>
    /// This class converts a string to a UTF8 encoded byte array and then copies it to an unmanaged buffer.
    /// This is done, so we can pass it to the native code and avoid pinning.
    /// </summary>
    public unsafe struct MarshaledString : IDisposable
    {
        internal MarshaledString(string input)
        {
            int length;
            IntPtr value;

            if (input is null)
            {
                length = 0;
                value = IntPtr.Zero;
            }
            else
            {
                var valueBytes = (input.Length != 0) ? Encoding.UTF8.GetBytes(input) :
                    ArrayUtilities.GetEmpty<byte>();
                length = valueBytes.Length;
                value = Marshal.AllocHGlobal(length + 1);

                Span<byte> destination = new Span<byte>(value.ToPointer(), length + 1);
                valueBytes.AsSpan(0, length).CopyTo(destination);
                destination[length] = 0;
            }

            Length = length;
            Value = value;
        }

        /// <summary>
        // Native allocation (UTF8-8 string length with terminating zero)
        /// </summary>
        internal int Length { get; private set; }

        /// <summary>
        /// Actual native buffer
        /// </summary>
        internal IntPtr Value { get; private set; }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            // No managed resources to dispose
            if (Value != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Value);
                Value = IntPtr.Zero;
                Length = 0;
            }
        }
    }

    /// <summary>
    /// Keeps a list of MarshaledString instances and provides a way to dispose them all at once.
    /// It is a ref struct, so it can not be IDisposable.
    /// </summary>
    public unsafe ref struct MarshaledStringArray
    {
        private MarshaledString[] _values;

        internal MarshaledStringArray(Tensor<string> inputs)
        {
            if (inputs.Length == 0)
            {
                _values = null;
            }
            else
            {
                _values = new MarshaledString[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    _values[i] = new MarshaledString(inputs.GetValue(i));
                }
            }
        }

        internal MarshaledStringArray(IEnumerable<string> inputs)
        {
            if (inputs is null)
            {
                _values = null;
            }
            else
            {
                _values = new MarshaledString[inputs.Count()];
                int i = 0;
                foreach (var input in inputs)
                {
                    _values[i++] = new MarshaledString(input);
                }
            }
        }

        internal ReadOnlySpan<MarshaledString> Values => _values;

        internal void Fill(IntPtr[] pDestination)
        {
            if (_values != null)
            {
                for (var i = 0; i < _values.Length; i++)
                {
                    pDestination[i] = Values[i].Value;
                }
            }
        }
        public void Dispose()
        {
            if (_values != null)
            {
                for (var i = 0; i < _values.Length; i++)
                {
                    _values[i].Dispose();
                }
                _values = null;
            }
        }

    }

    /// <summary>
    /// Utility class used in SessioniOptions and ProviderOptions
    /// </summary>
    internal class ProviderOptionsUpdater
    {
        /// <summary>
        /// A utility method to update the provider options, provides common functionality.
        /// 
        /// </summary>
        /// <param name="providerOptions">The actual key/value option pairs</param>
        /// <param name="handle">to the object</param>
        /// <param name="updateFunc">encapsulates a native method that returns 
        /// Arg1=handle, Arg2=array of keys, Arg3=array of values, Arg4 - count, Arg5 - return ORT status</param>
        internal static void Update(Dictionary<string, string> providerOptions,
                                    IntPtr handle,
                                    Func<IntPtr, IntPtr[], IntPtr[], UIntPtr, IntPtr> updateFunc)
        {
            var keyStrings = providerOptions.Keys.ToArray();
            var valStrings = providerOptions.Values.ToArray();

            MarshaledStringArray keys = default;
            MarshaledStringArray values = default;
            try
            {
                keys = new MarshaledStringArray(keyStrings);
                values = new MarshaledStringArray(valStrings);

                var nativeKeys = new IntPtr[keyStrings.Length];
                keys.Fill(nativeKeys);

                var nativeVals = new IntPtr[valStrings.Length];
                values.Fill(nativeVals);

                NativeApiStatus.VerifySuccess(updateFunc(handle, nativeKeys, nativeVals, (UIntPtr)providerOptions.Count));
            }
            finally
            {
                keys.Dispose();
                values.Dispose();
            }
        }
    }
}
