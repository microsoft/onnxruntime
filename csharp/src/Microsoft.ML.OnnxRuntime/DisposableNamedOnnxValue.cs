// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{
    public interface IDisposableReadOnlyCollection<T> : IReadOnlyCollection<T>, IDisposable
    {

    }

    internal class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T : IDisposable
    {
        public DisposableList() { }
        public DisposableList(int count) : base(count) { }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
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
    /// This class serves as a container for model run output values including
    /// tensors, sequences of tensors, sequences and maps.
    /// It extends NamedOnnxValue, exposes the OnnxValueType and Tensor type
    /// The class must be disposed of.
    /// It disposes of _ortValueHolder that owns the underlying Ort output value and
    /// anything else that would need to be disposed by the instance of the class.
    /// Use factory method CreateFromOrtValue to obtain an instance of the class.
    /// </summary>
    public class DisposableNamedOnnxValue : NamedOnnxValue, IDisposable
    {
        private IOrtValueOwner _ortValueHolder;
        private bool _disposed = false;

        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="name">Name of the output value</param>
        /// <param name="value">Managed object created to represent output value, such as DenseTensor<T>
        /// List or Dictionary
        /// </param>
        /// <param name="onnxValueType">Use this to decide what you want to call to fetch data, AsTensor(), AsDictionary()
        /// or AsEnumerable()</param>
        /// <param name="elementType">Tensor element type if value type is a Tensor</param>
        /// <param name="ortValueHolder">Object that holds native resources. 
        /// Typically, this is an output OrtValue that holds native memory where Tensor is mapped but may also be
        /// other things that would need to be disposed by this instance depending on how IOrtValueOwner is implemented.</param>
        private DisposableNamedOnnxValue(string name, Object value, OnnxValueType onnxValueType, TensorElementType elementType, IOrtValueOwner ortValueHolder)
            : base(name, value)
        {
            _ortValueHolder = ortValueHolder;
            ValueType = onnxValueType;
            ElementType = elementType;
        }

        /// <summary>
        /// Returns OnnxValueType
        /// </summary>
        public OnnxValueType ValueType { get; }

        /// <summary>
        /// Only valid if ValueType is Tensor
        /// </summary>
        public TensorElementType ElementType { get; }

        /// <summary>
        /// Overrides the base class method. Since the instance already owns underlying OrtValue handle,
        /// it returns an instance of OrtValue that does not own the raw handle
        /// that to the output onnxValue. With respect to pinnedMemoryHandle, it has no operation
        /// to do, as this class maintains a native buffer via _ortValueHolder and the memory will be
        /// disposed by it. This is the case when we are dealing with an OrtValue that is backed by native memory
        /// and not by pinned managed memory.
        /// </summary>
        /// <param name="pinnedMemoryHandle">always set to null</param>
        /// <returns>An instance of OrtValue that does not own underlying memory</returns>
        internal override OrtValue ToOrtValue(out MemoryHandle? pinnedMemoryHandle)
        {
            if(_ortValueHolder == null)
            {
                throw new InvalidOperationException("The instance of this class does not own any OrtValues");
            }
            // PinnedMemoryHandle holds the default value as DisposableNamedOnnxValue
            // doesn't hold any managed buffer (that needs to be pinned)
            pinnedMemoryHandle = null;
            // Return non-owning instance of OrtValue
            return _ortValueHolder.Value;
        }

        /// <summary>
        /// Creates an instance of DisposableNamedOnnxValue and takes ownership of ortValueElement
        /// on success.
        /// </summary>
        /// <param name="name">name of the value</param>
        /// <param name="ortValue">underlying OrtValue</param>
        /// <returns></returns>
        internal static DisposableNamedOnnxValue CreateTensorFromOnnxValue(string name, OrtValue ortValue)
        {
            DisposableNamedOnnxValue result = null;

            /* Get Tensor element type */  //TODO: Assumed value is Tensor, need to support non-tensor types in future
            IntPtr typeAndShape = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(ortValue.Handle, out typeAndShape));
            TensorElementType elemType = TensorElementType.DataTypeMax;
            try
            {
                IntPtr el_type;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                elemType = (TensorElementType)el_type;
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
            }

            switch (elemType)
            {
                case TensorElementType.Float:
                    result = DisposableNamedOnnxValueFromNativeTensor<float>(name, ortValue);
                    break;
                case TensorElementType.Double:
                    result = DisposableNamedOnnxValueFromNativeTensor<double>(name, ortValue);
                    break;
                case TensorElementType.Int16:
                    result = DisposableNamedOnnxValueFromNativeTensor<short>(name, ortValue);
                    break;
                case TensorElementType.UInt16:
                    result = DisposableNamedOnnxValueFromNativeTensor<ushort>(name, ortValue);
                    break;
                case TensorElementType.Int32:
                    result = DisposableNamedOnnxValueFromNativeTensor<int>(name, ortValue);
                    break;
                case TensorElementType.UInt32:
                    result = DisposableNamedOnnxValueFromNativeTensor<uint>(name, ortValue);
                    break;
                case TensorElementType.Int64:
                    result = DisposableNamedOnnxValueFromNativeTensor<long>(name, ortValue);
                    break;
                case TensorElementType.UInt64:
                    result = DisposableNamedOnnxValueFromNativeTensor<ulong>(name, ortValue);
                    break;
                case TensorElementType.UInt8:
                    result = DisposableNamedOnnxValueFromNativeTensor<byte>(name, ortValue);
                    break;
                case TensorElementType.Int8:
                    result = DisposableNamedOnnxValueFromNativeTensor<sbyte>(name, ortValue);
                    break;
                case TensorElementType.String:
                    result = DisposableNamedOnnxValueFromNativeTensor<string>(name, ortValue);
                    break;
                case TensorElementType.Bool:
                    result = DisposableNamedOnnxValueFromNativeTensor<bool>(name, ortValue);
                    break;
                case TensorElementType.Float16:
                    result = DisposableNamedOnnxValueFromNativeTensor<Float16>(name, ortValue);
                    break;
                case TensorElementType.BFloat16:
                    result = DisposableNamedOnnxValueFromNativeTensor<BFloat16>(name, ortValue);
                    break;
                default:
                    throw new NotSupportedException("Tensor of element type: " + elemType + " is not supported");

            }

            return result;
        }

        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, OrtValue ortValue)
        {
            return CreateFromOrtValue(name, ortValue, OrtAllocator.DefaultInstance);
        }

        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, OrtValue ortValue, OrtAllocator allocator)
        {
            DisposableNamedOnnxValue result = null;

            IntPtr valueType;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueType(ortValue.Handle, out valueType));
            OnnxValueType onnxValueType = (OnnxValueType)valueType;
            switch (onnxValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    result = CreateTensorFromOnnxValue(name, ortValue);
                    break;

                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    result = DisposableNamedOnnxValueFromSequence(name, ortValue, allocator);
                    break;

                case OnnxValueType.ONNX_TYPE_MAP:
                    result = DisposableNamedOnnxValueFromNativeMap(name, ortValue, allocator);
                    break;
                default:
                    throw new NotSupportedException("OnnxValueType : " + onnxValueType + " is not supported");
            }
            return result;
        }

        /// <summary>
        /// This method creates an instance of DisposableNamedOnnxValue that has possession of ortValueElement
        /// native memory Tensor and returns it to the caller. The original ortValueElement argument looses
        /// ownership of the native ortValueElement handle, however, the caller is still responsible for disposing them
        /// on exception. Disposing of OrtValue that has no ownership is a no-op and fine.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="name">name of the output</param>
        /// <param name="ortValue">native tensor</param>
        /// <returns>DisposableNamedOnnxValue instance</returns>
        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeTensor<T>(string name, OrtValue ortValue)
        {
            if (typeof(T) == typeof(string))
            {
                var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(ortValue);
                try
                {
                    var dt = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                    return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
                } catch(Exception e)
                {
                    nativeTensorWrapper.Dispose();
                    throw e;
                }
            }
            else
            {
                NativeOnnxTensorMemory<T> nativeTensorWrapper = new NativeOnnxTensorMemory<T>(ortValue);
                try
                {
                    DenseTensor<T> dt = new DenseTensor<T>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                    return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
                }
                catch (Exception e)
                {
                    nativeTensorWrapper.Dispose();
                    throw e;
                }
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
        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromSequence(string name, OrtValue ortValueSequence, OrtAllocator allocator)
        {
            DisposableNamedOnnxValue result = null;
            IntPtr count;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueCount(ortValueSequence.Handle, out count));
            var sequence = new DisposableList<DisposableNamedOnnxValue>(count.ToInt32());
            try
            {
                for (int i = 0; i < count.ToInt32(); i++)
                {
                    IntPtr nativeOnnxValueSeq;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValueSequence.Handle, i, allocator.Pointer, out nativeOnnxValueSeq));
                    using (var ortValueElement = new OrtValue(nativeOnnxValueSeq))
                    {
                        // Will take ownership or throw
                        sequence.Add(CreateFromOrtValue(string.Empty, ortValueElement, allocator));
                    }
                }
                // NativeOrtValueCollectionOwner will take ownership of ortValueSequence and will make sure sequence
                // is also disposed.
                var nativeCollectionManager = new NativeOrtValueCollectionOwner(ortValueSequence, sequence);
                result = new DisposableNamedOnnxValue(name, sequence, OnnxValueType.ONNX_TYPE_SEQUENCE, TensorElementType.DataTypeMax, nativeCollectionManager);
            }
            catch (Exception e)
            {
                sequence.Dispose();
                throw e;
            }
            return result;
        }

        /// <summary>
        /// Will extract keys and values from the map and create a DisposableNamedOnnxValue from it
        /// </summary>
        /// <param name="name">name of the output</param>
        /// <param name="ortValueMap">ortValue that represents a map. 
        /// This function does not take ownership of the map as it we copy all keys an values into a dictionary. We let the caller dispose of it</param>
        /// <param name="allocator"></param>
        /// <returns>DisposableNamedOnnxValue</returns>
        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeMap(string name, OrtValue ortValueMap, OrtAllocator allocator)
        {
            DisposableNamedOnnxValue result = null;
            // Map processing is currently not recursing. It is assumed to contain
            // only primitive types and strings tensors. No sequences or maps.
            // The data is being copied to a dictionary and all ortValues are being disposed.
            // not mapped for client consumption.
            using (var cleanUpList = new DisposableList<IDisposable>())
            {
                // Take possession of the map ortValueElement
                IntPtr nativeOnnxValueMapKeys = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValueMap.Handle, 0, allocator.Pointer, out nativeOnnxValueMapKeys));
                var ortValueKeys = new OrtValue(nativeOnnxValueMapKeys);
                cleanUpList.Add(ortValueKeys);

                IntPtr nativeOnnxValueMapValues = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValueMap.Handle, 1, allocator.Pointer, out nativeOnnxValueMapValues));
                var ortValueValues = new OrtValue(nativeOnnxValueMapValues);
                cleanUpList.Add(ortValueValues);

                IntPtr typeAndShape = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(nativeOnnxValueMapKeys, out typeAndShape));
                TensorElementType elemType = TensorElementType.DataTypeMax;
                try
                {
                    IntPtr el_type;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                    elemType = (TensorElementType)el_type;
                }
                finally
                {
                    NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
                }

                /// XXX: This code always assumes that the value type is float and makes no checks
                /// similar to that of the key. Also Map type in general can also be another sequence or map,
                /// not just a tensor
                switch (elemType)
                {
                    case TensorElementType.Int64:
                        result = DisposableNamedOnnxValueFromNativeMapElements<Int64, float>(string.Empty, ortValueKeys, ortValueValues);
                        break;
                    case TensorElementType.String:
                        result = DisposableNamedOnnxValueFromNativeMapElements<string, float>(string.Empty, ortValueKeys, ortValueValues);
                        break;
                    default:
                        throw new NotSupportedException("Map of element type: " + elemType + " is not supported");
                }
            }
            return result;
        }


        /// <summary>
        /// This method maps keys and values of the map and copies them into a Dictionary
        /// and returns as an instance of DisposableNamedOnnxValue that does not own or dispose
        /// any onnx/ortValueElement. The method takes possession of ortValueTensorKeys and ortValueTensorValues
        /// and disposes of them. The original ortValueElement looses ownership of the Tensor. The caller is still responsible
        /// for disposing these arguments. Disposing ortValueElement that does not have ownership is a no-op, however, either
        /// of the arguments may still need to be disposed on exception.
        /// </summary>
        /// <typeparam name="K">key type</typeparam>
        /// <typeparam name="V">value type</typeparam>
        /// <param name="name">name of the output parameter</param>
        /// <param name="ortValueTensorKeys">tensor with map keys.</param>
        /// <param name="nativeOnnxValueValues">tensor with map values</param>
        /// <returns>instance of DisposableNamedOnnxValue with Dictionary</returns>
        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeMapElements<K, V>(string name,
            OrtValue ortValueTensorKeys, OrtValue ortValueTensorValues)
        {
            using (var nativeTensorWrapperValues = new NativeOnnxTensorMemory<V>(ortValueTensorValues))
            {
                var denseTensorValues = new DenseTensor<V>(nativeTensorWrapperValues.Memory, nativeTensorWrapperValues.Dimensions);

                if (typeof(K) == typeof(string))
                {
                    var map = new Dictionary<string, V>();
                    using (var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(ortValueTensorKeys))
                    {
                        var denseTensorKeys = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                        for (var i = 0; i < denseTensorKeys.Length; i++)
                        {
                            map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                        }
                        return new DisposableNamedOnnxValue(name, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
                    }
                }
                else
                {
                    var map = new Dictionary<K, V>();
                    using (var nativeTensorWrapper = new NativeOnnxTensorMemory<K>(ortValueTensorKeys))
                    {
                        var denseTensorKeys = new DenseTensor<K>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                        for (var i = 0; i < denseTensorKeys.Length; i++)
                        {
                            map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                        }
                        return new DisposableNamedOnnxValue(name, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
                    }
                }
            }
        }

        #region IDisposable Support

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked by Dispose()</param>
        protected virtual void Dispose(bool disposing)
        {
            if(_disposed)
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
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion

    }
}
