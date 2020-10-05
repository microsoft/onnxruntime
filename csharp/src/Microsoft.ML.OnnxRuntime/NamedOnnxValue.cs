// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices.ComTypes;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The name of the class is a misnomer, it does not hold any
    /// Onnx values
    /// </summary>
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
        public Object Value { get { return _value; } set { _value = value; } }

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
        internal virtual OrtValue ToOrtValue(out MemoryHandle? pinnedMemoryHandle)
        {
            return OrtValue.CreateFromTensorObject(_value, out pinnedMemoryHandle, out TensorElementType elementType);
        }

        // may expose different types of getters in future

    }
}
