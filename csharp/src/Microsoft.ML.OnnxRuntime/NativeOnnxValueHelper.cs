// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    internal class PinnedGCHandle : IDisposable
    {
        private GCHandle _handle;

        public PinnedGCHandle(GCHandle handle)
        {
            _handle = handle;
        }

        public IntPtr Pointer
        {
            get
            {
                return _handle.AddrOfPinnedObject();
            }
        }

        #region Disposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _handle.Free();
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // No need for the finalizer
        // If this is not disposed timely GC can't help us
        #endregion
   }

/// <summary>
/// This helper class contains methods to create native OrtValue from a managed value object
/// </summary>
internal static class NativeOnnxValueHelper
    {
        /// <summary>
        /// Attempts to Pin the buffer, and create a native OnnxValue out of it. the pinned MemoryHandle is passed to output.
        /// In this case, the pinnedHandle should be kept alive till the native OnnxValue is used, then dispose it.
        /// If it is not possible to Pin the buffer, then creates OnnxValue from the copy of the data. The output pinnedMemoryHandle
        /// contains a default value in that case.
        /// Attempts to infer the type of the value while creating the OnnxValue
        /// </summary>
        /// <param name="value"></param>
        /// <param name="onnxValue"></param>
        /// <param name="pinnedMemoryHandle"></param>
        /// <param name="elementType"></param>
        internal static void CreateNativeOnnxValue(Object value, out IntPtr onnxValue, out MemoryHandle pinnedMemoryHandle, out OnnxValueType onnxValueType, out TensorElementType elementType)
        {
            //try to cast _value to Tensor<T>
            elementType = TensorElementType.DataTypeMax; //invalid
            IntPtr dataBufferPointer = IntPtr.Zero;
            int dataBufferLength = 0;
            ReadOnlySpan<int> shape = null;
            int rank = 0;
            onnxValue = IntPtr.Zero;

            if (!(value is Tensor<string>))
            {
                if (TryPinAsTensor<float>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<double>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<int>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<uint>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<long>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<ulong>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<short>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<ushort>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<byte>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<sbyte>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                else if (TryPinAsTensor<bool>(value, out pinnedMemoryHandle,
                                          out dataBufferPointer,
                                          out dataBufferLength,
                                          out shape,
                                          out rank,
                                          out elementType))
                {
                }
                //TODO: add other types
                else
                {
                    // nothing to cleanup here, since no memory has been pinned
                    throw new NotSupportedException("The inference value " + nameof(value) + " is not of a supported type");
                }

                try
                {
                    Debug.Assert(dataBufferPointer != IntPtr.Zero, "dataBufferPointer must be non-null after obtaining the pinned buffer");

                    onnxValueType = OnnxValueType.ONNX_TYPE_TENSOR; // set onnx value type to tensor

                    // copy to an ulong[] shape to match int64_t[]
                    long[] longShape = new long[rank];
                    for (int i = 0; i < rank; i++)
                    {
                        longShape[i] = shape[i];
                    }

                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                        MemoryInfo.DefaultInstance.Pointer,
                        dataBufferPointer,
                        (UIntPtr)(dataBufferLength),
                        longShape,
                        (UIntPtr)rank,
                        elementType,
                        out onnxValue));
                }
                catch (OnnxRuntimeException e)
                {
                    pinnedMemoryHandle.Dispose();
                    throw e;
                }
            }
            // special case for string Tensor, data needs to be copied to the native buffer
            else
            {
                // calculate native tensor length (sum of string lengths in utf-8)
                var tensorValue = value as Tensor<string>;
                int totalLength = 0;
                for (int i = 0; i < tensorValue.Length; i++)
                {
                    totalLength += Encoding.UTF8.GetByteCount(tensorValue.GetValue(i));
                }

                long[] longShape = new long[tensorValue.Dimensions.Length];
                for (int i = 0; i < tensorValue.Dimensions.Length; i++)
                {
                    longShape[i] = tensorValue.Dimensions[i];
                }

                // allocate the native tensor
                IntPtr nativeTensor = IntPtr.Zero;
                try
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorAsOrtValue(
                                                    MemoryAllocator.DefaultInstance.Pointer,
                                                    longShape,
                                                    (UIntPtr)(longShape.Length),
                                                    TensorElementType.String,
                                                    out nativeTensor
                                                    ));

                    // fill the native tensor, using GetValue(index) from the Tensor<string>
                    var len = tensorValue.Length;
                    var stringsInTensor = new IntPtr[len];
                    using(var pinnedHandles = new DisposableList<PinnedGCHandle>((int)len))
                    {
                        for (int i = 0; i < len; i++)
                        {
                            var utf8str = StringToZeroTerminatedUtf8(tensorValue.GetValue(i));
                            var gcHandle = GCHandle.Alloc(utf8str, GCHandleType.Pinned);
                            stringsInTensor[i] = gcHandle.AddrOfPinnedObject();
                            pinnedHandles.Add(new PinnedGCHandle(gcHandle));
                        }

                        using (var pinnedStrings = new PinnedGCHandle(GCHandle.Alloc(stringsInTensor, GCHandleType.Pinned)))
                            NativeApiStatus.VerifySuccess(NativeMethods.OrtFillStringTensor(nativeTensor, stringsInTensor, (UIntPtr)len));
                    }
                }
                catch (OnnxRuntimeException e)
                {
                    if (nativeTensor != IntPtr.Zero)
                    {
                        NativeMethods.OrtReleaseValue(nativeTensor);
                        throw e;
                    }
                }

                onnxValue = nativeTensor; // set the output
                pinnedMemoryHandle = default; // dummy value for the output

                onnxValueType = OnnxValueType.ONNX_TYPE_TENSOR; // set onnx value type to tensor
                elementType = TensorElementType.String; // set tensor element type to string
            }
        }

        private static bool TryPinAsTensor<T>(
            Object value,
            out MemoryHandle pinnedMemoryHandle,
            out IntPtr dataBufferPointer,
            out int dataBufferLength,
            out ReadOnlySpan<int> shape,
            out int rank,
            out TensorElementType nativeElementType)
        {
            nativeElementType = TensorElementType.DataTypeMax; //invalid
            dataBufferPointer = IntPtr.Zero;
            dataBufferLength = 0;
            shape = null;
            rank = 0;
            pinnedMemoryHandle = default;

            Debug.Assert(typeof(T) != typeof(string), "NativeOnnxValueHelper.TryPinAsTensor() must not be called with a string Tensor value");

            if (value is Tensor<T>)
            {
                Tensor<T> t = value as Tensor<T>;
                if (t.IsReversedStride)
                {
                    //TODO: not sure how to support reverse stride. may be able to calculate the shape differently
                    throw new NotSupportedException(nameof(Tensor<T>) + " of reverseStride is not supported");
                }

                DenseTensor<T> dt = null;
                if (value is DenseTensor<T>)
                {
                    dt = value as DenseTensor<T>;
                }
                else
                {
                    dt = t.ToDenseTensor();
                }

                shape = dt.Dimensions;  // does not work for reverse stride
                rank = dt.Rank;
                pinnedMemoryHandle = dt.Buffer.Pin();
                unsafe
                {
                    dataBufferPointer = (IntPtr)pinnedMemoryHandle.Pointer;
                }

                // find the native type
                if (typeof(T) == typeof(float))
                {
                    nativeElementType = TensorElementType.Float;
                    dataBufferLength = dt.Buffer.Length * sizeof(float);
                }
                else if (typeof(T) == typeof(double))
                {
                    nativeElementType = TensorElementType.Double;
                    dataBufferLength = dt.Buffer.Length * sizeof(double);
                }
                else if (typeof(T) == typeof(int))
                {
                    nativeElementType = TensorElementType.Int32;
                    dataBufferLength = dt.Buffer.Length * sizeof(int);
                }
                else if (typeof(T) == typeof(uint))
                {
                    nativeElementType = TensorElementType.UInt32;
                    dataBufferLength = dt.Buffer.Length * sizeof(uint);
                }
                else if (typeof(T) == typeof(long))
                {
                    nativeElementType = TensorElementType.Int64;
                    dataBufferLength = dt.Buffer.Length * sizeof(long);
                }
                else if (typeof(T) == typeof(ulong))
                {
                    nativeElementType = TensorElementType.UInt64;
                    dataBufferLength = dt.Buffer.Length * sizeof(ulong);
                }
                else if (typeof(T) == typeof(short))
                {
                    nativeElementType = TensorElementType.Int16;
                    dataBufferLength = dt.Buffer.Length * sizeof(short);
                }
                else if (typeof(T) == typeof(ushort))
                {
                    nativeElementType = TensorElementType.UInt16;
                    dataBufferLength = dt.Buffer.Length * sizeof(ushort);
                }
                else if (typeof(T) == typeof(byte))
                {
                    nativeElementType = TensorElementType.UInt8;
                    dataBufferLength = dt.Buffer.Length * sizeof(byte);
                }
                else if (typeof(T) == typeof(sbyte))
                {
                    nativeElementType = TensorElementType.Int8;
                    dataBufferLength = dt.Buffer.Length * sizeof(sbyte);
                }
                else if (typeof(T) == typeof(string))
                {
                    nativeElementType = TensorElementType.String;
                    dataBufferLength = dt.Buffer.Length * IntPtr.Size;
                }
                else if (typeof(T) == typeof(bool))
                {
                    nativeElementType = TensorElementType.Bool;
                    dataBufferLength = dt.Buffer.Length * sizeof(bool); // Assumes sizeof(BOOL) is always 1 byte in native
                }
                else
                {
                    //TODO: may extend the supported types
                    // do not throw exception, rather assign the sentinel value
                    nativeElementType = TensorElementType.DataTypeMax;
                }
                return true;
            }

            return false;
        }

        /// <summary>
        /// Converts C# UTF-16 string to UTF-8 zero terminated
        /// byte[] instance
        /// </summary>
        /// <param name="s">string to be converted</param>
        /// <returns>UTF-8 encoded equivalent</returns>
        internal static byte[] StringToZeroTerminatedUtf8(string s)
        {
            return UTF8Encoding.UTF8.GetBytes(s + '\0');
        }

        /// <summary>
        /// Reads UTF-8 encode string from a C zero terminated string
        /// and converts it into a C# UTF-16 encoded string
        /// </summary>
        /// <param name="nativeUtf8">pointer to native or pinned memory where Utf-8 resides</param>
        /// <returns></returns>
        internal static string StringFromNativeUtf8(IntPtr nativeUtf8) 
        {
            // .NET 5.0 has Marshal.PtrToStringUTF8 that does the below
            int len = 0;
            while (Marshal.ReadByte(nativeUtf8, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(nativeUtf8, buffer, 0, buffer.Length);
            return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
        }
    }

    public enum OnnxValueType
    {
        ONNX_TYPE_UNKNOWN = 0,
        ONNX_TYPE_TENSOR = 1,
        ONNX_TYPE_SEQUENCE = 2,
        ONNX_TYPE_MAP = 3,
        ONNX_TYPE_OPAQUE = 4,
        ONNX_TYPE_SPARSETENSOR = 5,
    }

    internal static class TensorElementTypeConverter
    {
        public static void GetTypeAndWidth(TensorElementType elemType, out Type type, out int width)
        {
            switch (elemType)
            {
                case TensorElementType.Float:
                    type = typeof(float);
                    width = sizeof(float);
                    break;
                case TensorElementType.Double:
                    type = typeof(double);
                    width = sizeof(double);
                    break;
                case TensorElementType.Int16:
                    type = typeof(short);
                    width = sizeof(short);
                    break;
                case TensorElementType.UInt16:
                    type = typeof(ushort);
                    width = sizeof(ushort);
                    break;
                case TensorElementType.Int32:
                    type = typeof(int);
                    width = sizeof(int);
                    break;
                case TensorElementType.UInt32:
                    type = typeof(uint);
                    width = sizeof(uint);
                    break;
                case TensorElementType.Int64:
                    type = typeof(long);
                    width = sizeof(long);
                    break;
                case TensorElementType.UInt64:
                    type = typeof(ulong);
                    width = sizeof(ulong);
                    break;
                case TensorElementType.UInt8:
                    type = typeof(byte);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Int8:
                    type = typeof(sbyte);
                    width = sizeof(sbyte);
                    break;
                case TensorElementType.String:
                    type = typeof(string);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Bool:
                    type = typeof(bool);
                    width = sizeof(bool);
                    break;
                default:
                    type = null;
                    width = 0;
                    break;
            }
        }
    }
}
