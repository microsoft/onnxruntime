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
            if (arraySize != UTF8Encoding.UTF8.GetBytes(s, 0, s.Length, utf8Bytes, 0))
            {
                throw new OnnxRuntimeException(ErrorCode.RuntimeException, "Failed to convert to UTF8");
            }
            utf8Bytes[utf8Bytes.Length - 1] = 0;
            return utf8Bytes;
        }

        /// <summary>
        /// This string converts the input string into UTF-8 encoding string (no zero termination)
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
            unsafe
            {
#if NETSTANDARD1_1
                var utf8Bytes = (s.Length != 0) ? Encoding.UTF8.GetBytes(s) : ArrayUtilities.GetEmpty<byte>();
                Span<byte> destination = new Span<byte>((ptr).ToPointer(), utf8Bytes.Length);
                utf8Bytes.AsSpan(0, utf8Bytes.Length).CopyTo(destination);
#else
                fixed (char* c = s) // create char pointer to start of managed string.
                {
                    var nativeBytes = (byte*)ptr; // get managed byte* from native intptr
                    var bytesWritten = Encoding.UTF8.GetBytes(c, s.Length, nativeBytes, totalBytesToWrite); // total bytes to write is size of native memory buffer
                    Debug.Assert(bytesWritten == totalBytesToWrite);
                }
#endif
            }
        }

        /// <summary>
        /// Reads UTF-8 encode string from a C zero terminated string
        /// and converts it into a C# UTF-16 encoded string
        /// </summary>
        /// <param name="nativeUtf8">pointer to native or pinned memory where Utf-8 resides</param>
        /// <returns></returns>
        internal static string StringFromNativeUtf8(IntPtr nativeUtf8)
        {
            unsafe
            {
                int len = 0;
                while(*(byte*)(nativeUtf8 + len) != 0) ++len;

                if(len == 0)
                {
                    return string.Empty;
                }
#if NETSTANDARD1_1
                var src = new Span<byte>((nativeUtf8).ToPointer(), len);
                byte[] buffer = new byte[len];
                src.CopyTo(buffer);
                return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
#else
                var nativeBytes = (byte*)nativeUtf8;
                return Encoding.UTF8.GetString(nativeBytes, len);
#endif
            }
        }

        /// <summary>
        /// Reads UTF-8 string from native C zero terminated string,
        /// converts it to C# UTF-16 string and returns both C# string and utf-8
        /// bytes as a zero terminated array, suitable for use as a C-string
        /// </summary>
        /// <param name="nativeUtf8">input</param>
        /// <param name="str">C# UTF-16 string</param>
        /// <param name="utf8">UTF-8 bytes in a managed buffer, zero terminated</param>
        internal static void StringAndUtf8FromNative(IntPtr nativeUtf8, out string str, out byte[] utf8)
        {
            unsafe
            {
                int len = 0;
                while (*(byte*)(nativeUtf8 + len) != 0) ++len;

                if (len == 0)
                {
                    str = string.Empty;
                    utf8 = ArrayUtilities.GetEmpty<byte>();
                    return;
                }

                var src = new Span<byte>((nativeUtf8).ToPointer(), len);
                utf8 = new byte[len + 1];
                src.CopyTo(utf8);
                utf8[len] = 0;
#if NETSTANDARD1_1
                str = Encoding.UTF8.GetString(utf8, 0, len);
#else
                var nativeBytes = (byte*)nativeUtf8;
                str = Encoding.UTF8.GetString(nativeBytes, len);
#endif
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
}
