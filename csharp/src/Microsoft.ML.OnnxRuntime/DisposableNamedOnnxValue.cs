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

    public class DisposableNamedOnnxValue : NamedOnnxValue, IDisposable
    {
        private NativeMemoryHandler _nativeMemoryManager;
        private TensorElementType _elementType;
        private OnnxValueType _onnxValueType;

        private DisposableNamedOnnxValue(string name, Object value, OnnxValueType onnxValueType, TensorElementType elementType, NativeMemoryHandler nativeMemoryManager)
            : base(name, value)
        {
            _onnxValueType = onnxValueType;
            _elementType = elementType;
            _nativeMemoryManager = nativeMemoryManager;
        }

        /// <summary>
        /// Overrides the base class method. Since the instance already has access to the 
        /// underlying OrtValue handle, it returns an instance of OrtValue that does not own the raw handle
        /// that to the output onnxValue. With respect to pinnedMemoryHandle, it has no operation
        /// to do, as this class doesn't maintain a managed buffer. It doesn't have to maintain it
        /// as it already is associated with the object of interest (native OrtValue)
        /// </summary>
        /// <param name="pinnedMemoryHandle"></param>
        internal override OrtValue ToOrtValue(out MemoryHandle? pinnedMemoryHandle)
        {
            // PinnedMemoryHandle holds the default value as DisposableNamedOnnxValue
            // doesn't hold any managed buffer (that needs to be pinned)
            pinnedMemoryHandle = null;
            // Assign the onnxValue by querying this instance's NativeOnnxTensorMemory instance
            return new OrtValue(_nativeMemoryManager.Handle, false);
        }

        internal static DisposableNamedOnnxValue CreateTensorFromOnnxValue(string name, IntPtr nativeOnnxValue)
        {
            DisposableNamedOnnxValue result = null;

            /* Get Tensor element type */  //TODO: Assumed value is Tensor, need to support non-tensor types in future
            IntPtr typeAndShape = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(nativeOnnxValue, out typeAndShape));
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
                    result = DisposableNamedOnnxValueFromNativeTensor<float>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Double:
                    result = DisposableNamedOnnxValueFromNativeTensor<double>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int16:
                    result = DisposableNamedOnnxValueFromNativeTensor<short>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt16:
                    result = DisposableNamedOnnxValueFromNativeTensor<ushort>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int32:
                    result = DisposableNamedOnnxValueFromNativeTensor<int>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt32:
                    result = DisposableNamedOnnxValueFromNativeTensor<uint>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int64:
                    result = DisposableNamedOnnxValueFromNativeTensor<long>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt64:
                    result = DisposableNamedOnnxValueFromNativeTensor<ulong>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt8:
                    result = DisposableNamedOnnxValueFromNativeTensor<byte>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int8:
                    result = DisposableNamedOnnxValueFromNativeTensor<sbyte>(name, nativeOnnxValue);
                    break;
                case TensorElementType.String:
                    result = DisposableNamedOnnxValueFromNativeTensor<string>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Bool:
                    result = DisposableNamedOnnxValueFromNativeTensor<bool>(name, nativeOnnxValue);
                    break;
                default:
                    throw new NotSupportedException("Tensor of element type: " + elemType + " is not supported");

            }

            return result;
        }

        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, OrtValue ortValue)
        {
            var result = CreateFromOnnxValue(name, ortValue.Handle, OrtAllocator.DefaultInstance);
            ortValue.Disown();
            return result;
        }

        internal static DisposableNamedOnnxValue CreateFromOnnxValue(string name, IntPtr nativeOnnxValue, OrtAllocator allocator)
        {
            IntPtr valueType;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueType(nativeOnnxValue, out valueType));
            OnnxValueType onnxValueType = (OnnxValueType)valueType;
            switch (onnxValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    return CreateTensorFromOnnxValue(name, nativeOnnxValue);

                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    IntPtr count = IntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueCount(nativeOnnxValue, out count));
                    var sequence = new DisposableList<DisposableNamedOnnxValue>(count.ToInt32());
                    for (int i = 0; i < count.ToInt32(); i++)
                    {
                        IntPtr nativeOnnxValueSeq;
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(nativeOnnxValue, i, allocator.Pointer, out nativeOnnxValueSeq));
                        sequence.Add(CreateFromOnnxValue(string.Empty, nativeOnnxValueSeq, allocator));
                    }
                    return new DisposableNamedOnnxValue(name, sequence, OnnxValueType.ONNX_TYPE_SEQUENCE, TensorElementType.DataTypeMax, null);

                case OnnxValueType.ONNX_TYPE_MAP:
                    IntPtr nativeOnnxValueMapKeys = IntPtr.Zero;
                    IntPtr nativeOnnxValueMapValues = IntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(nativeOnnxValue, 0, allocator.Pointer, out nativeOnnxValueMapKeys));
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(nativeOnnxValue, 1, allocator.Pointer, out nativeOnnxValueMapValues));

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

                    switch (elemType)
                    {
                        case TensorElementType.Int64:
                            return DisposableNamedOnnxValueFromNativeMap<Int64, float>(string.Empty, nativeOnnxValueMapKeys, nativeOnnxValueMapValues);
                        case TensorElementType.String:
                            return DisposableNamedOnnxValueFromNativeMap<string, float>(string.Empty, nativeOnnxValueMapKeys, nativeOnnxValueMapValues);
                        default:
                            throw new NotSupportedException("Map of element type: " + elemType + " is not supported");
                    }
                default:
                    throw new NotSupportedException("OnnxValueType : " + onnxValueType + " is not supported");
            }
        }

        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeTensor<T>(string name, IntPtr nativeOnnxValue)
        {
            if (typeof(T) == typeof(string))
            {
                var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(nativeOnnxValue);
                var dt = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
            }
            else
            {
                NativeOnnxTensorMemory<T> nativeTensorWrapper = new NativeOnnxTensorMemory<T>(nativeOnnxValue);
                DenseTensor<T> dt = new DenseTensor<T>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
            }
        }

        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeMap<K, V>(string name, IntPtr nativeOnnxValueKeys, IntPtr nativeOnnxValueValues)
        {
            var nativeTensorWrapperValues = new NativeOnnxTensorMemory<V>(nativeOnnxValueValues);
            var denseTensorValues = new DenseTensor<V>(nativeTensorWrapperValues.Memory, nativeTensorWrapperValues.Dimensions);

            if (typeof(K) == typeof(string))
            {
                var map = new Dictionary<string, V>();
                var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(nativeOnnxValueKeys);
                var denseTensorKeys = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                for (var i = 0; i < denseTensorKeys.Length; i++)
                {
                    map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                }
                // release native memory
                nativeTensorWrapperValues.Dispose();
                nativeTensorWrapper.Dispose();
                return new DisposableNamedOnnxValue(string.Empty, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
            }
            else
            {
                var map = new Dictionary<K, V>();
                var nativeTensorWrapper = new NativeOnnxTensorMemory<K>(nativeOnnxValueKeys);
                var denseTensorKeys = new DenseTensor<K>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                for (var i = 0; i < denseTensorKeys.Length; i++)
                {
                    map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                }
                // release native memory
                nativeTensorWrapperValues.Dispose();
                nativeTensorWrapper.Dispose();
                return new DisposableNamedOnnxValue(string.Empty, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
            }
        }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // dispose managed state (managed objects).
                if (_nativeMemoryManager != null)
                {
                    _nativeMemoryManager.Dispose();
                    _nativeMemoryManager = null;
                }
            }
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion

    }
}
