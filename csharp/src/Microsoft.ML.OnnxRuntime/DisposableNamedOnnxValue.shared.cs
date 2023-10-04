// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Return immutable collection of results
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IDisposableReadOnlyCollection<T> : IReadOnlyCollection<T>, IReadOnlyList<T>, IDisposable
    {

    }

    internal class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T : IDisposable
    {
        private bool _disposed;
        public DisposableList() { }
        public DisposableList(int count) : base(count) { }

        public DisposableList(IEnumerable<T> collection) : base(collection) { }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                // Dispose in the reverse order.
                // Objects should typically be destroyed/disposed
                // in the reverse order of its creation
                // especially if the objects created later refer to the
                // objects created earlier. For homogeneous collections of objects
                // it would not matter.
                for (int i = this.Count - 1; i >= 0; --i)
                {
                    this[i]?.Dispose();
                }
                this.Clear();
                _disposed = true;
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }

    /// <summary>
    /// This is a legacy class that is kept for backward compatibility.
    /// Use OrtValue based API.
    /// 
    /// This class serves as a container for model run output values including
    /// tensors, sequences of tensors, sequences and maps.
    /// The class must be disposed of.
    /// It disposes of _ortValueHolder that owns the underlying Ort output value and
    /// anything else that would need to be disposed by the instance of the class.
    /// Use factory method CreateFromOrtValue to obtain an instance of the class.
    /// </summary>
    public class DisposableNamedOnnxValue : NamedOnnxValue, IDisposable
    {
        private IOrtValueOwner _ortValueHolder;
        private bool _disposed;

        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="name">Name of the output value</param>
        /// <param name="value">Managed object created to represent output value, such as DenseTensor<T>
        /// List or Dictionary
        /// </param>
        /// <param name="elementType">Tensor element type if value type is a Tensor</param>
        /// <param name="ortValueHolder">Object that holds native resources. 
        /// Typically, this is an output OrtValue that holds native memory where Tensor is mapped but may also be
        /// other things that would need to be disposed by this instance depending on how IOrtValueOwner is implemented.</param>
        private DisposableNamedOnnxValue(string name, Object value, TensorElementType elementType, IOrtValueOwner ortValueHolder)
            : base(name, value, OnnxValueType.ONNX_TYPE_TENSOR)
        {
            _ortValueHolder = ortValueHolder;
            ElementType = elementType;
        }

        /// <summary>
        /// Ctor for non-tensor values
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <param name="onnxValueType"></param>
        /// <param name="ortValueHolder"></param>
        private DisposableNamedOnnxValue(string name, Object value, OnnxValueType onnxValueType, IOrtValueOwner ortValueHolder)
            : base(name, value, onnxValueType)
        {
            _ortValueHolder = ortValueHolder;
            ElementType = TensorElementType.DataTypeMax;
        }

        /// <summary>
        /// Construct an instance that would contain a map in a form of a Dictionary
        /// Currently a limited number of primitive types are supported as map keys and values.
        /// So this is not a full implementation of the map type.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <param name="mapHelper"></param>
        /// <param name="ortValueHolder"></param>
        private DisposableNamedOnnxValue(string name, Object value, MapHelper mapHelper, IOrtValueOwner ortValueHolder)
            : base(name, value, mapHelper)
        {
            _ortValueHolder = ortValueHolder;
            ElementType = TensorElementType.DataTypeMax;
        }

        /// <summary>
        /// Only valid if ValueType is Tensor
        /// </summary>
        public TensorElementType ElementType { get; }

        /// <summary>
        /// Overrides the base class method. With respect to pinnedMemoryHandle, it has no operation
        /// to do, as this class maintains a native buffer via _ortValueHolder and the memory will be
        /// disposed by it. This is the case when we are dealing with an OrtValue that is backed by native memory
        /// and not by pinned managed memory.
        /// 
        /// This class is generally used for outputs to be created on top of the output OrtValue,
        /// but the interface (derived from NamedOnnxValue) allows it to be passed as output and one of the test
        /// cases does it. Unless we deprecate and re-do the interface, we must support it.
        /// </summary>
        /// <param name="pinnedMemoryHandle">always set to null</param>
        /// <returns>Native OrtValue handle</returns>
        internal override IntPtr InputToOrtValueHandle(NodeMetadata metadata, out IDisposable memoryHolder)
        {
            if (_ortValueHolder == null)
            {
                throw new InvalidOperationException("The instance of this class does not own an OrtValue");
            }
            // PinnedMemoryHandle holds the default value as DisposableNamedOnnxValue
            // doesn't hold any managed buffer (that needs to be pinned)
            memoryHolder = null;
            // Return non-owning instance of OrtValue
            return _ortValueHolder.Value.Handle;
        }

        /// <summary>
        /// Generally, this class is created on top of the values that are returned by the model run.
        /// However, there is a test case that uses this value for output
        /// It will return the OrtValue that was previously created, since the caller must understand what they are doing.
        /// </summary>
        /// <param name="metadata"></param>
        /// <param name="memoryOwner"></param>
        /// <returns></returns>
        internal override IntPtr OutputToOrtValueHandle(NodeMetadata metadata, out IDisposable memoryOwner)
        {
            return InputToOrtValueHandle(metadata, out memoryOwner);
        }

        /// <summary>
        /// This function takes ortValue and constructs an instance of DisposableNamedOnnxValue.
        /// The new instance takes ownership of the OrtValue and will dispose of it when it is disposed of.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="ortValue">becomes null on success.</param>
        /// <returns>an instance of DisposableNamedOnnxValue</returns>
        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, ref OrtValue ortValue)
        {
            return CreateFromOrtValue(name, ref ortValue, OrtAllocator.DefaultInstance);
        }

        /// <summary>
        /// This function takes ortValue and constructs an instance of DisposableNamedOnnxValue.
        /// The new instance takes ownership of the OrtValue and will dispose of it when it is disposed of.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="ortValue">becomes null on success.</param>
        /// <param name="allocator"></param>
        /// <returns>an instance of DisposableNamedOnnxValue</returns>
        /// <exception cref="NotSupportedException"></exception>
        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, ref OrtValue ortValue, OrtAllocator allocator)
        {
            DisposableNamedOnnxValue result;

            var onnxValueType = ortValue.OnnxType;
            switch (onnxValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    result = FromNativeTensor(name, ref ortValue);
                    break;

                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    result = FromNativeSequence(name, ref ortValue, allocator);
                    break;

                case OnnxValueType.ONNX_TYPE_MAP:
                    result = FromNativeMap(name, ref ortValue, allocator);
                    break;
                default:
                    throw new NotSupportedException($"OnnxValueType : {onnxValueType} is not supported");
            }
            return result;
        }

        /// <summary>
        /// Creates an instance of DisposableNamedOnnxValue and takes ownership of ortValue.
        /// on success.
        /// </summary>
        /// <param name="name">name of the value</param>
        /// <param name="ortValue">Underlying OrtValue. This becomes null on successful return.</param>
        /// <returns></returns>
        private static DisposableNamedOnnxValue FromNativeTensor(string name, ref OrtValue ortValue)
        {
            DisposableNamedOnnxValue result;

            var typeShape = ortValue.GetTensorTypeAndShape();
            switch (typeShape.ElementDataType)
            {
                case TensorElementType.Float:
                    result = FromNativeTensor<float>(name, ref ortValue);
                    break;
                case TensorElementType.Double:
                    result = FromNativeTensor<double>(name, ref ortValue);
                    break;
                case TensorElementType.Int16:
                    result = FromNativeTensor<short>(name, ref ortValue);
                    break;
                case TensorElementType.UInt16:
                    result = FromNativeTensor<ushort>(name, ref ortValue);
                    break;
                case TensorElementType.Int32:
                    result = FromNativeTensor<int>(name, ref ortValue);
                    break;
                case TensorElementType.UInt32:
                    result = FromNativeTensor<uint>(name, ref ortValue);
                    break;
                case TensorElementType.Int64:
                    result = FromNativeTensor<long>(name, ref ortValue);
                    break;
                case TensorElementType.UInt64:
                    result = FromNativeTensor<ulong>(name, ref ortValue);
                    break;
                case TensorElementType.UInt8:
                    result = FromNativeTensor<byte>(name, ref ortValue);
                    break;
                case TensorElementType.Int8:
                    result = FromNativeTensor<sbyte>(name, ref ortValue);
                    break;
                case TensorElementType.String:
                    {
                        var shape = Array.ConvertAll<long, int>(typeShape.Shape, Convert.ToInt32);
                        result = FromNativeStringTensor(name, shape, ref ortValue);
                    }
                    break;
                case TensorElementType.Bool:
                    result = FromNativeTensor<bool>(name, ref ortValue);
                    break;
                case TensorElementType.Float16:
                    result = FromNativeTensor<Float16>(name, ref ortValue);
                    break;
                case TensorElementType.BFloat16:
                    result = FromNativeTensor<BFloat16>(name, ref ortValue);
                    break;
                default:
                    throw new NotSupportedException($"Tensor of element type: {typeShape.ElementDataType} is not supported");

            }

            return result;
        }

        private static DisposableNamedOnnxValue FromNativeStringTensor(string name, int[] shape, ref OrtValue ortValue)
        {
            var dt = new DenseTensor<string>(ortValue.GetStringTensorAsArray(), shape);
            // still need to hold on to ortValue in case we need this for input handles
            var result = new DisposableNamedOnnxValue(name, dt, TensorElementType.String, ortValue);
            ortValue = null;
            return result;
        }


        /// <summary>
        /// This method creates an instance of DisposableNamedOnnxValue that has possession of ortValueElement
        /// native memory Tensor and returns it to the caller.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="name">name of the output</param>
        /// <param name="ortValue">native tensor. Becomes null on successful return.</param>
        /// <returns>DisposableNamedOnnxValue instance</returns>
        private static DisposableNamedOnnxValue FromNativeTensor<T>(string name, ref OrtValue ortValue)
        {
            Debug.Assert(typeof(T) != typeof(string), "Use FromNativeStringTensor for strings");
            var ortValueTensor = new OrtValueTensor<T>(ref ortValue);
            try
            {
                var dt = new DenseTensor<T>(ortValueTensor.Memory, ortValueTensor.Dimensions);
                return new DisposableNamedOnnxValue(name, dt, ortValueTensor.ElementType, ortValueTensor);
            }
            catch (Exception)
            {
                ortValueTensor.Dispose();
                throw;
            }
        }

        /// <summary>
        /// This method will create an instance of DisposableNamedOnnxValue that will own ortSequenceValue
        /// an all disposable native objects that are elements of the sequence
        /// </summary>
        /// <param name="name"></param>
        /// <param name="ortValueSequence">ortValueElement that has native sequence</param>
        /// <param name="allocator"> used allocator</param>
        /// <returns>DisposableNamedOnnxValue</returns>
        private static DisposableNamedOnnxValue FromNativeSequence(string name, ref OrtValue ortValueSequence, OrtAllocator allocator)
        {
            var valueCount = ortValueSequence.GetValueCount();
            var sequence = new DisposableList<DisposableNamedOnnxValue>(valueCount);
            try
            {
                for (int i = 0; i < valueCount; i++)
                {
                    var ortValueElement = ortValueSequence.GetValue(i, allocator);
                    try
                    {
                        // Will take ownership or throw
                        sequence.Add(CreateFromOrtValue(string.Empty, ref ortValueElement, allocator));
                    }
                    finally
                    {
                        ortValueElement?.Dispose();
                    }
                }
                // NativeOrtValueCollectionOwner will take ownership of ortValueSequence and will make sure sequence
                // is also disposed.
                var nativeCollectionManager = new NativeOrtValueCollectionOwner(ref ortValueSequence, sequence);
                return new DisposableNamedOnnxValue(name, sequence, OnnxValueType.ONNX_TYPE_SEQUENCE, nativeCollectionManager);
            }
            catch (Exception)
            {
                sequence.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Will extract keys and values from the map and create a DisposableNamedOnnxValue from it
        /// </summary>
        /// <param name="name">name of the output</param>
        /// <param name="ortValueMap">ortValue that represents a map. Becomes null on success</param>
        /// <param name="allocator"></param>
        /// <returns>DisposableNamedOnnxValue</returns>
        private static DisposableNamedOnnxValue FromNativeMap(string name, ref OrtValue ortValueMap, OrtAllocator allocator)
        {
            DisposableNamedOnnxValue result = null;
            // Map processing is not recursive. It is assumed to contain
            // only primitive types and strings tensors. No sequences or maps.
            // The data is being copied to a dictionary and all ortValues are being disposed.
            // not mapped for client consumption.

            // Keys in element 0, values in element 1
            Span<OrtValue> valSpan = new OrtValue[2];
            var disposer = new DisposableArray<OrtValue>(valSpan);
            try
            {
                valSpan[0] = ortValueMap.GetValue(0, allocator);
                valSpan[1] = ortValueMap.GetValue(1, allocator);

                var keysTypeShape = valSpan[0].GetTensorTypeAndShape();
                var valsTypeInfo = valSpan[1].GetTensorTypeAndShape();

                int[] intKeyShape = Array.ConvertAll<long, int>(keysTypeShape.Shape, Convert.ToInt32);
                int[] intValsShape = Array.ConvertAll<long, int>(valsTypeInfo.Shape, Convert.ToInt32);

                // The supported combinations of key and value types are taken from the ORT C API.
                switch (keysTypeShape.ElementDataType)
                {
                    case TensorElementType.Int64:
                        switch (valsTypeInfo.ElementDataType)
                        {
                            case TensorElementType.Float:
                                result = FromNativeMapElements<Int64, float>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.Double:
                                result = FromNativeMapElements<Int64, double>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.Int64:
                                result = FromNativeMapElements<Int64, Int64>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.String:
                                result = FromNativeMapElements<Int64, string>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            default:
                                throw new NotSupportedException($"Map value type: {valsTypeInfo.ElementDataType} is not supported");
                        }
                        break;
                    case TensorElementType.String:
                        switch (valsTypeInfo.ElementDataType)
                        {
                            case TensorElementType.Float:
                                result = FromNativeMapElements<string, float>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.Double:
                                result = FromNativeMapElements<string, double>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.Int64:
                                result = FromNativeMapElements<string, Int64>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            case TensorElementType.String:
                                result = FromNativeMapElements<string, string>(name, ref ortValueMap,
                                    intKeyShape, ref valSpan[0], intValsShape, ref valSpan[1]);
                                break;
                            default:
                                throw new NotSupportedException($"Map value type: {valsTypeInfo.ElementDataType} is not supported");
                        }
                        break;
                    default:
                        throw new NotSupportedException($"Map key type: {keysTypeShape.ElementDataType} is not supported");
                }
            }
            finally
            {
                // Any values that are taken possession of
                // will be null, others, like string tensors, will be disposed
                disposer.Dispose();
            }

            return result;
        }


        /// <summary>
        /// This method maps keys and values of the map and copies them into a managed Dictionary
        /// and returns as an instance of DisposableNamedOnnxValue. The method takes possession of ortValueMap,
        /// ortValueTensorKeys and ortValueTensorValues and disposes of them.
        /// </summary>
        /// <typeparam name="K"></typeparam>
        /// <typeparam name="V"></typeparam>
        /// <param name="name"></param>
        /// <param name="ortValueMap">becomes null on success return</param>
        /// <param name="keysShape">keys shape in ints</param>
        /// <param name="ortValueTensorKeys">becomes null on success</param>
        /// <param name="valsShape">values shape in ints</param>
        /// <param name="ortValueTensorValues">becomes null on success</param>
        /// <returns></returns>
        private static DisposableNamedOnnxValue FromNativeMapElements<K, V>(string name, ref OrtValue ortValueMap,
            int[] keysShape, ref OrtValue ortValueTensorKeys,
            int[] valsShape, ref OrtValue ortValueTensorValues)
        {
            if (typeof(K) == typeof(string))
            {
                var denseTensorKeys = new DenseTensor<string>(ortValueTensorKeys.GetStringTensorAsArray(), keysShape);

                if (typeof(V) == typeof(string))
                {
                    var denseTensorValues = new DenseTensor<string>(ortValueTensorValues.GetStringTensorAsArray(), valsShape);
                    var map = Enumerable.Range(0, (int)denseTensorKeys.Length).ToDictionary(i => denseTensorKeys[i], i => denseTensorValues[i]);
                    var mapHelper = new MapHelper(denseTensorKeys, denseTensorValues);
                    var result = new DisposableNamedOnnxValue(name, map, mapHelper, ortValueMap);
                    ortValueMap = null;
                    return result;
                }
                else
                {
                    var tensorValues = new OrtValueTensor<V>(ref ortValueTensorValues);
                    try
                    {
                        var denseTensorValues = new DenseTensor<V>(tensorValues.Memory, tensorValues.Dimensions);
                        return FromMapDenseTensors(name, ref ortValueMap, denseTensorKeys, denseTensorValues, tensorValues);
                    }
                    catch (Exception)
                    {
                        tensorValues.Dispose();
                        throw;
                    }
                }
            }
            else
            {
                var disposer = new DisposableList<IDisposable>(2);
                try
                {
                    var tensorKeys = new OrtValueTensor<K>(ref ortValueTensorKeys);
                    disposer.Add(tensorKeys);
                    var denseTensorKeys = new DenseTensor<K>(tensorKeys.Memory, tensorKeys.Dimensions);
                    if (typeof(V) == typeof(string))
                    {
                        var denseTensorValues = new DenseTensor<string>(ortValueTensorValues.GetStringTensorAsArray(), valsShape);
                        return FromMapDenseTensors(name, ref ortValueMap, denseTensorKeys, denseTensorValues, disposer);
                    }
                    else
                    {
                        var tensorValues = new OrtValueTensor<V>(ref ortValueTensorValues);
                        disposer.Add(tensorValues);
                        var denseTensorValues = new DenseTensor<V>(tensorValues.Memory, tensorValues.Dimensions);
                        return FromMapDenseTensors(name, ref ortValueMap, denseTensorKeys, denseTensorValues, disposer);
                    }
                }
                catch (Exception)
                {
                    disposer.Dispose();
                    throw;
                }
            }
        }

        #region IDisposable Support

        private static DisposableNamedOnnxValue FromMapDenseTensors<K, V>(string name, ref OrtValue ortValueMap,
            DenseTensor<K> keys, DenseTensor<V> values, IDisposable disposables)
        {
            var map = Enumerable.Range(0, (int)keys.Length).ToDictionary(i => keys[i], i => values[i]);
            var mapHelper = new MapHelper(keys, values);
            var collOwner = new NativeOrtValueCollectionOwner(ref ortValueMap, disposables);
            return new DisposableNamedOnnxValue(name, map, mapHelper, collOwner);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked by Dispose()</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            // dispose managed state (managed objects).
            if (disposing)
            {
                // _ortValueHolder can be null when no native memory is involved
                if (_ortValueHolder != null)
                {
                    _ortValueHolder.Dispose();
                    _ortValueHolder = null;
                }
            }
            _disposed = true;
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }
        #endregion

    }
}
