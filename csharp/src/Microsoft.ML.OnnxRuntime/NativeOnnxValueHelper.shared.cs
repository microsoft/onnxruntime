// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Linq;

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
        /// Reads UTF-8 encode string from a C zero terminated string
        /// and converts it into a C# UTF-16 encoded string
        /// </summary>
        /// <param name="nativeUtf8">pointer to native or pinned memory where Utf-8 resides</param>
        /// <returns></returns>
        internal static string StringFromNativeUtf8(IntPtr nativeUtf8)
        {
            // .NET 8.0 has Marshal.PtrToStringUTF8 that does the below
            int len = 0;
            while (Marshal.ReadByte(nativeUtf8, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(nativeUtf8, buffer, 0, len);
            return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
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
            // .NET 8.0 has Marshal.PtrToStringUTF8 that does the below
            int len = 0;
            while (Marshal.ReadByte(nativeUtf8, len) != 0) ++len;
            utf8 = new byte[len + 1];
            Marshal.Copy(nativeUtf8, utf8, 0, len);
            utf8[len] = 0;
            str = Encoding.UTF8.GetString(utf8, 0, len);
        }

        internal static string StringFromUtf8Span(ReadOnlySpan<byte> utf8Span)
        {
            // XXX: For now we have to copy into byte[], this produces a copy
            // Converting from span is available in later versions
            var utf8Bytes = utf8Span.ToArray();
            return Encoding.UTF8.GetString(utf8Bytes, 0, utf8Bytes.Length);
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
