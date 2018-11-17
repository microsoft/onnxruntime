// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;


namespace Microsoft.ML.OnnxRuntime
{
    internal class NativeOnnxTensorMemory<T> : MemoryManager<T>
    {
        private bool _disposed;
        private int _referenceCount;
        private IntPtr _onnxValueHandle;
        private IntPtr _dataBufferHandle;
        private int _elementCount;
        private int _elementWidth;
        private int[] _dimensions;

        public NativeOnnxTensorMemory(IntPtr onnxValueHandle)
        {
            IntPtr typeAndShape = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.ONNXRuntimeGetTensorShapeAndType(onnxValueHandle, out typeAndShape));

                TensorElementType elemType = NativeMethods.ONNXRuntimeGetTensorElementType(typeAndShape);

                Type type = null;
                int width = 0;
                GetTypeAndWidth(elemType, out type, out width);
                if (typeof(T) != type)
                    throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>)+" does not support T = "+nameof(T));
                _elementWidth = width;

                _onnxValueHandle = onnxValueHandle;
                // derive the databuffer pointer, element_count, element_width, and shape
                NativeApiStatus.VerifySuccess(NativeMethods.ONNXRuntimeGetTensorMutableData(_onnxValueHandle, out _dataBufferHandle));
                // throws OnnxRuntimeException if native call failed

                ulong dimension = NativeMethods.ONNXRuntimeGetNumOfDimensions(typeAndShape);
                long count = NativeMethods.ONNXRuntimeGetTensorShapeElementCount(typeAndShape);  // count can be negative. 
                if (count < 0)
                {
                    throw new NotSupportedException("Symbolic dimensions in the tensor is not supported");
                }

                long[] shape = new long[dimension];
                NativeMethods.ONNXRuntimeGetDimensions(typeAndShape, shape, dimension); //Note: shape must be alive during the call

                _elementCount = (int)count;
                _dimensions = new int[dimension];
                for (ulong i = 0; i < dimension; i++)
                {
                    _dimensions[i] = (int)shape[i];
                }
            }
            catch (Exception e)
            {
                //TODO: cleanup any partially created state
                //Do not call ReleaseTensor here. If the constructor has thrown exception, then this NativeOnnxTensorWrapper is not created, so caller should take appropriate action to dispose
                throw e;
            }
            finally
            {
                if (typeAndShape != IntPtr.Zero)
                {
                    NativeMethods.ONNXRuntimeReleaseObject(typeAndShape);
                }
            }
        }
        ~NativeOnnxTensorMemory()
        {
            Dispose(false);
        }

        public bool IsDisposed => _disposed;

        protected bool IsRetained => _referenceCount > 0;

        public int[] Dimensions
        {
            get
            {
                return _dimensions;
            }
        }

        public int Rank
        {
            get
            {
                return _dimensions.Length;
            }
        }

        public override Span<T> GetSpan()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(NativeOnnxTensorMemory<T>));
            Span<T> span = null;
            unsafe
            {
                span = new Span<T>((void*)_dataBufferHandle, _elementCount);
            }

            return span;
        }


        public override MemoryHandle Pin(int elementIndex = 0)
        {
            //Note: always pin the full buffer and return
            unsafe
            {
                if (elementIndex >= _elementCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(elementIndex));
                }
                Retain();

                return new MemoryHandle((void*)((int)_dataBufferHandle + elementIndex*_elementWidth)); //could not use Unsafe.Add
            }
        }


        public override void Unpin()
        {
            Release();
        }


        private bool Release()
        {
            int newRefCount = Interlocked.Decrement(ref _referenceCount);

            if (newRefCount < 0)
            {
                throw new InvalidOperationException("Unmatched Release/Retain");
            }

            return newRefCount != 0;
        }

        private void Retain()
        {
            if (IsDisposed)
            {
                throw new ObjectDisposedException(nameof(NativeOnnxTensorMemory<T>));
            }

            Interlocked.Increment(ref _referenceCount);
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                // do managed objects cleanup
            }

            NativeMethods.ReleaseONNXValue(_onnxValueHandle);

            _disposed = true;
        }

        protected override bool TryGetArray(out ArraySegment<T> arraySegment)
        {
            // cannot expose managed array
            arraySegment = default(ArraySegment<T>);
            return false;
        }


        internal static void GetTypeAndWidth(TensorElementType elemType, out Type type, out int width)
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
                default:
                    type = null;
                    width = 0;
                    break;
            }
        }
        
    }
}
