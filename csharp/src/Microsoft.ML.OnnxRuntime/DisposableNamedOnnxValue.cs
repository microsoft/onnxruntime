// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Numerics.Tensors;


namespace Microsoft.ML.OnnxRuntime
{
    public interface IDisposableReadOnlyCollection<T>: IReadOnlyCollection<T>, IDisposable
    {

    }

    internal class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T: IDisposable
    {

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                    for (int i = 0; i < this.Count; i++)
                    {
                        this[i].Dispose();
                    }
                    this.Clear();
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
            }
        }

        ~DisposableList()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
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


    public class DisposableNamedOnnxValue: NamedOnnxValue, IDisposable
    {
        protected IDisposable _nativeMemoryManager;
        protected DisposableNamedOnnxValue(string name, Object value, IDisposable nativeMemoryManager)
            :base(name, value)
        {
            _nativeMemoryManager = nativeMemoryManager;
        }


        internal static DisposableNamedOnnxValue CreateFromOnnxValue(string name, IntPtr nativeOnnxValue)
        {
            DisposableNamedOnnxValue result = null;

            /* Get Tensor element type */  //TODO: Assumed value is Tensor, need to support non-tensor types in future
            IntPtr typeAndShape = IntPtr.Zero;
            TensorElementType elemType = TensorElementType.DataTypeMax;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeAndType(nativeOnnxValue, out typeAndShape));
                elemType = NativeMethods.OrtGetTensorElementType(typeAndShape);
            }
            finally
            {
                if (typeAndShape != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
                }
            }

            switch (elemType)
            {
                case TensorElementType.Float:
                    result = NameOnnxValueFromNativeTensor<float>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Double:
                    result = NameOnnxValueFromNativeTensor<double>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int16:
                    result = NameOnnxValueFromNativeTensor<short>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt16:
                    result = NameOnnxValueFromNativeTensor<ushort>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int32:
                    result = NameOnnxValueFromNativeTensor<int>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt32:
                    result = NameOnnxValueFromNativeTensor<uint>(name, nativeOnnxValue);
                    break;
                case TensorElementType.Int64:
                    result = NameOnnxValueFromNativeTensor<long>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt64:
                    result = NameOnnxValueFromNativeTensor<ulong>(name, nativeOnnxValue);
                    break;
                case TensorElementType.UInt8:
                    result = NameOnnxValueFromNativeTensor<byte>(name, nativeOnnxValue);
                    break;
                default:
                    throw new NotSupportedException("Tensor of element type: " + elemType + " is not supported");

            }

            return result;
        }


        private static DisposableNamedOnnxValue NameOnnxValueFromNativeTensor<T>(string name, IntPtr nativeOnnxValue)
        {
            NativeOnnxTensorMemory<T> nativeTensorWrapper = new NativeOnnxTensorMemory<T>(nativeOnnxValue);
            DenseTensor<T> dt = new DenseTensor<T>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
            return new DisposableNamedOnnxValue(name, dt, nativeTensorWrapper);
        }



        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
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

                // free unmanaged resources (unmanaged objects) and override a finalizer below.
                // set large fields to null.
                disposedValue = true;
            }
        }

        ~DisposableNamedOnnxValue()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
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
}
