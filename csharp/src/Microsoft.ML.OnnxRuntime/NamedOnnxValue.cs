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

        public string Name { get { return _name; } set { _name = value; } }

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
        /// Pin the underlying memory and create native onnx value
        /// </summary>
        /// <param name="onnxValue"></param>
        /// <param name="pinnedMemoryHandle"></param>
        /// <param name="disposeOnnxValueAfterUse"></param>
        internal virtual void ToNativeOnnxValue(out IntPtr onnxValue, out MemoryHandle pinnedMemoryHandle, out bool disposeOnnxValueAfterUse)
        {
            NativeOnnxValueHelper.CreateNativeOnnxValue(_value, out onnxValue, out pinnedMemoryHandle);
            disposeOnnxValueAfterUse = true;
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
