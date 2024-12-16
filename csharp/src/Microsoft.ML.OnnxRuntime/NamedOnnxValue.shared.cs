// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class holds keys and values for the dictionary
    /// in a for of two DenseTensors. The class is used to avoid
    /// data copy and make these available to the native code.
    /// Strings require special handling.
    /// </summary>
    internal class MapHelper
    {
        internal MapHelper(TensorBase keys, TensorBase values)
        {
            Keys = keys;
            Values = values;
        }
        internal TensorBase Keys { get; }   // DenseTensor<K>
        internal TensorBase Values { get; } // DenseTensor<V>
    }

    /// <summary>
    /// This is a legacy class that is kept for backward compatibility.
    /// Use OrtValue based API.
    /// 
    /// The class associates a name with an Object. 
    /// The name of the class is a misnomer, it does not hold any Onnx values,
    /// just managed representation of them.
    /// 
    /// The class is currently used as both inputs and outputs. Because it is non-
    /// disposable, it can not hold on to any native objects.
    /// 
    /// When used as input, we temporarily create OrtValues that map managed inputs
    /// directly. Thus we are able to avoid copying of contiguous data.
    /// 
    /// For outputs, tensor buffers works the same as input, providing it matches
    /// the expected output shape. For other types (maps and sequences) we create a copy of the data.
    /// This is because, the class is not Disposable and it is a public interface, thus it can not own
    /// the underlying OrtValues that must be destroyed before Run() returns.
    /// 
    /// To avoid data copying on output, use DisposableNamedOnnxValue class that is returned from Run() methods.
    /// This provides access to the native memory tensors and avoids copying.
    /// 
    /// It is a recursive structure that may contain Tensors (base case)
    /// Other sequences and maps. Although the OnnxValueType is exposed,
    /// the caller is supposed to know the actual data type contained.
    /// 
    /// The convention is that for tensors, it would contain a DenseTensor{T} instance or
    /// anything derived from Tensor{T}.
    /// 
    /// For sequences, it would contain a IList{T} where T is an instance of NamedOnnxValue that
    /// would contain a tensor or another type.
    /// 
    /// For Maps, it would contain a IDictionary{K, V} where K,V are primitive types or strings.
    /// 
    /// </summary>
    public class NamedOnnxValue
    {
        /// <summary>
        /// Managed Tensor, Dictionary or IList
        /// </summary>
        private Object _value;
        /// <summary>
        /// Name of the instance, model input/output
        /// </summary>
        private string _name;

        private MapHelper _mapHelper; // used for maps, otherwise null

        /// <summary>
        /// Constructs an instance of NamedOnnxValue and represents
        /// a model input to an inference session.
        /// </summary>
        /// <param name="name">input/output name</param>
        /// <param name="value">Object that may be a tensor, Dictionary, IList</param>
        [Obsolete("Use constructors with valueType or static factory methods")]
        protected NamedOnnxValue(string name, Object value)
        {
            _name = name;
            _value = value;
            ValueType = OnnxValueType.ONNX_TYPE_UNKNOWN;
        }

        /// <summary>
        /// Constructs an instance that contains a tensor, sequence or optional type.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <param name="valueType"></param>
        internal NamedOnnxValue(string name, Object value, OnnxValueType valueType)
        {
            _name = name;
            _value = value;
            ValueType = valueType;

            if (valueType == OnnxValueType.ONNX_TYPE_MAP)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Use another __ctor for maps");
            }
        }

        /// <summary>
        /// Use this to construct maps
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <param name="helper"></param>
        internal NamedOnnxValue(string name, Object value, MapHelper helper)
        {
            _name = name;
            _value = value;
            ValueType = OnnxValueType.ONNX_TYPE_MAP;
            _mapHelper = helper;
        }

        /// <summary>
        /// Onnx Value Type if known. In general, NamedOnnxValue is able to contain
        /// arbitrary objects. Please, follow the convention described in the class doc.
        /// </summary>
        public OnnxValueType ValueType { get; internal set; }

        /// <summary>
        /// This is a factory method that instantiates NamedOnnxValue
        /// and associated name with an instance of a Tensor<typeparamref name="T"/>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="name">name</param>
        /// <param name="value">Tensor<typeparamref name="T"/></param>
        /// <returns></returns>
        public static NamedOnnxValue CreateFromTensor<T>(string name, Tensor<T> value)
        {
            return new NamedOnnxValue(name, value, OnnxValueType.ONNX_TYPE_TENSOR);
        }

        /// <summary>
        /// This is a factory method that instantiates NamedOnnxValue.
        /// It would contain a sequence of elements
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static NamedOnnxValue CreateFromSequence<T>(string name, IEnumerable<T> value)
        {
            return new NamedOnnxValue(name, value, OnnxValueType.ONNX_TYPE_SEQUENCE);
        }

        /// <summary>
        /// Instantiates NamedOnnxValue that contains IDictionary{K, V}
        /// </summary>
        /// <typeparam name="K">Keys type</typeparam>
        /// <typeparam name="V">Values type</typeparam>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <returns>new instance of NamedOnnxValue</returns>
        public static NamedOnnxValue CreateFromMap<K, V>(string name, IDictionary<K, V> value)
        {
            // The order in which Keys and Values are unspecified,
            // but it is guaranteed to be the same order
            // These tensors are 1-D
            return CreateFromMap<K, V>(name, value.Keys, value.Values);
        }

        internal static NamedOnnxValue CreateFromMap<K, V>(string name, ICollection<K> keys, ICollection<V> values)
        {
            var keysTensor = new DenseTensor<K>(keys.ToArray(), new int[1] { keys.Count });
            var valuesTensor = new DenseTensor<V>(values.ToArray(), new int[1] { values.Count });
            return new NamedOnnxValue(name, values, new MapHelper(keysTensor, valuesTensor));
        }

        /// <summary>
        /// Exposes the name of the of the model input/output
        /// </summary>
        /// <value>name string</value>
        public string Name { get { return _name; } set { _name = value; } }
        /// <summary>
        /// Exposes the underlying managed object
        /// </summary>
        /// <value>object</value>
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
        /// T is usually a NamedOnnxValue instance that may contain
        /// Tensors, Sequences, Maps or optional types
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
        /// <typeparam name="K">Key type currently primitive type only</typeparam>
        /// <typeparam name="V">Value type, currently primitive type only</typeparam>
        /// <returns>Dictionary object if contained value is a Dictionary. Null otherwise</returns>
        public IDictionary<K, V> AsDictionary<K, V>()
        {
            return _value as IDictionary<K, V>;
        }

        /// <summary>
        /// Pin the underlying memory and create an instance of OrtValue containing a tensor
        /// based on the pinned managed memory. The caller is responsible for Disposing
        /// both OrtValue and pinnedMemoryHandle
        /// </summary>
        /// <param name="memoryOwner">dispose after returned OrtValue is disposed</param>
        /// <returns>The native OrtValue handle</returns>
        internal virtual IntPtr InputToOrtValueHandle(NodeMetadata metadata, out IDisposable memoryOwner)
        {
            var projection = ManagedTypeProjection.CreateProjection(this, metadata);
            memoryOwner = projection;
            return projection.Handle;
        }

        /// <summary>
        /// Produces an output value for outputs. This produces an output value
        /// only for tensors or optional types that can contain a tensor.
        /// For all other Onnx value types, this method throws. Use Run() overloads
        /// that return DisposableNamedOnnxValue to get access to all Onnx value types
        /// that may be returned as output.
        /// </summary>
        /// <param name="metadata"></param>
        /// <param name="memoryOwner"></param>
        /// <returns></returns>
        internal virtual IntPtr OutputToOrtValueHandle(NodeMetadata metadata, out IDisposable memoryOwner)
        {
            // For NamedOnnxValue for output we only allow to produce OrtValue for tensors
            // or optional type that may contain a tensor
            if (metadata.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR)
            {
                var projection = ManagedTypeProjection.CreateProjection(this, metadata);
                memoryOwner = projection;
                return projection.Handle;
            }

            if (metadata.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
            {
                var meta = metadata.AsOptionalMetadata().ElementMeta;
                if (meta.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                {
                    var projection = ManagedTypeProjection.CreateProjection(this, metadata);
                    memoryOwner = projection;
                    return projection.Handle;
                }
            }

            throw new OnnxRuntimeException(ErrorCode.NotImplemented, 
                $"Can not create output OrtValue for NamedOnnxValue '{metadata.OnnxValueType}' type." +
                $" Only tensors can be pre-allocated for outputs " +
                $" Use Run() overloads that return DisposableNamedOnnxValue to get access to all Onnx value types that may be returned as output.");
        }

        internal TensorBase GetDictionaryKeys()
        {
            if (ValueType != OnnxValueType.ONNX_TYPE_MAP)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "This NamedOnnxValue instance does not contain a dictionary");
            }

            Debug.Assert(_mapHelper != null);
            return _mapHelper.Keys;
        }

        internal TensorBase GetDictionaryValues()
        {
            if (ValueType != OnnxValueType.ONNX_TYPE_MAP)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "This NamedOnnxValue instance does not contain a dictionary");
            }

            Debug.Assert(_mapHelper != null);
            return _mapHelper.Values;
        }
    }
}
