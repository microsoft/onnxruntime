// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    public class NamedOnnxValue
    {
        protected Object _value;
        protected string _name;

        protected NamedOnnxValue(string name, Object value)
        {
            _name = name;
            _value = value;
        }

        public static NamedOnnxValue CreateFromTensor<T>(string name, Tensor<T> value)
        {
            return new NamedOnnxValue(name, value);
        }

        public string Name { get { return _name; } }

        /// <summary>
        /// Try-get value as a Tensor&lt;T&gt;.
        /// </summary>
        /// <typeparam name="T">Type</typeparam>
        /// <returns>Tensor object if contained value is a Tensor. Null otherwise</returns>
        public Tensor<T> AsTensor<T>()
        {
            return _value as Tensor<T>;  // will return null if not castable
        }

        /// <summary>
        /// Try-get value as an Enumerable&lt;T&gt;.
        /// </summary>
        /// <typeparam name="T">Type</typeparam>
        /// <returns>Enumerable object if contained value is a Enumerable. Null otherwise</returns>
        public IEnumerable<T> AsEnumerable<T>()
        {
            var x = _value as IEnumerable<T>;
            return x;
        }

        /// <summary>
        /// Try-get value as an Dictionary&lt;K,V&gt;.
        /// </summary>
        /// <typeparam name="K">Key type</typeparam>
        /// <typeparam name="V">Value type</typeparam>
        /// <returns>Dictionary object if contained value is a Dictionary. Null otherwise</returns>
        public IDictionary<K, V> AsDictionary<K, V>()
        {
            return _value as IDictionary<K, V>;
        }

        /// <summary>
        /// Attempts to Pin the buffer, and create a native OnnxValue out of it. the pinned MemoryHandle is passed to output.
        /// In this case, the pinnedHandle should be kept alive till the native OnnxValue is used, then dispose it.
        /// If it is not possible to Pin the buffer, then creates OnnxValue from the copy of the data. The output pinnedMemoryHandle
        /// contains a default value in that case.
        /// Attempts to infer the type of the value while creating the OnnxValue
        /// </summary>
        /// <param name="onnxValue"></param>
        /// <param name="pinnedMemoryHandle"></param>
        internal void ToNativeOnnxValue(out IntPtr onnxValue, out MemoryHandle pinnedMemoryHandle)
        {
            //try to cast _value to Tensor<T>
            TensorElementType nativeElementType = TensorElementType.DataTypeMax; //invalid
            IntPtr dataBufferPointer = IntPtr.Zero;
            int dataBufferLength = 0;
            ReadOnlySpan<int> shape = null;
            int rank = 0;
            onnxValue = IntPtr.Zero;

            if (TryPinAsTensor<float>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<double>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<int>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<uint>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<long>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<ulong>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<short>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<ushort>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<byte>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<sbyte>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            else if (TryPinAsTensor<bool>(out pinnedMemoryHandle,
                                      out dataBufferPointer,
                                      out dataBufferLength,
                                      out shape,
                                      out rank,
                                      out nativeElementType
                                    ))
            {
            }
            //TODO: add other types
            // special case for string Tensor, data needs to be copied to the native buffer
            else if (!(_value is Tensor<string>))
            {
                // nothing to cleanup here, since no memory has been pinned
                throw new NotSupportedException("The inference value " + nameof(_value) + " is not of a supported type");
            }


            if (_value is Tensor<string>)
            {
                // calculate native tensor length (sum of string lengths in utf-8)
                var tensorValue = _value as Tensor<string>;
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
                                                    NativeMemoryAllocator.DefaultInstance.Handle,
                                                    longShape,
                                                    (UIntPtr)(longShape.Length),
                                                    TensorElementType.String,
                                                    out nativeTensor
                                                    ));

                    // fill the native tensor, using GetValue(index) from the Tensor<string>
                    string[] stringsInTensor = new string[tensorValue.Length];
                    for (int i = 0; i < tensorValue.Length; i++)
                    {
                        stringsInTensor[i] = tensorValue.GetValue(i);
                    }
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtFillStringTensor(nativeTensor, stringsInTensor, (UIntPtr)tensorValue.Length));
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
            }
            else
            {
                Debug.Assert(dataBufferPointer != IntPtr.Zero, "dataBufferPointer must be non-null after obtaining the pinned buffer");

                // copy to an ulong[] shape to match size_t[]
                long[] longShape = new long[rank];
                for (int i = 0; i < rank; i++)
                {
                    longShape[i] = shape[i];
                }

                IntPtr status = NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                        NativeMemoryInfo.DefaultInstance.Handle,
                        dataBufferPointer,
                        (UIntPtr)(dataBufferLength),
                        longShape,
                        (UIntPtr)rank,
                        nativeElementType,
                        out onnxValue
                    );
                try
                {
                    NativeApiStatus.VerifySuccess(status);
                }
                catch (OnnxRuntimeException e)
                {
                    pinnedMemoryHandle.Dispose();
                    throw e;
                }

            }

        }

        private bool TryPinAsTensor<T>(
                out MemoryHandle pinnedMemoryHandle,
                out IntPtr dataBufferPointer,
                out int dataBufferLength,
                out ReadOnlySpan<int> shape,
                out int rank,
                out TensorElementType nativeElementType
            )
        {
            nativeElementType = TensorElementType.DataTypeMax; //invalid
            dataBufferPointer = IntPtr.Zero;
            dataBufferLength = 0;
            shape = null;
            rank = 0;
            pinnedMemoryHandle = default;

            Debug.Assert(typeof(T) != typeof(string), "NamedOnnxValue.TryPinAsTensor() must not be called with a string Tensor value");

            if (_value is Tensor<T>)
            {
                Tensor<T> t = _value as Tensor<T>;
                if (t.IsReversedStride)
                {
                    //TODO: not sure how to support reverse stride. may be able to calculate the shape differently
                    throw new NotSupportedException(nameof(Tensor<T>) + " of reverseStride is not supported");
                }

                DenseTensor<T> dt = null;
                if (_value is DenseTensor<T>)
                {
                    dt = _value as DenseTensor<T>;
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

        // may expose different types of getters in future

    }

    internal enum TensorElementType
    {
        Float = 1,
        UInt8 = 2,
        Int8 = 3,
        UInt16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        UInt32 = 12,
        UInt64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        BFloat16 = 16,
        DataTypeMax = 17
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
