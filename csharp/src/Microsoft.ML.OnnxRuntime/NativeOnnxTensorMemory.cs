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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeAndType(onnxValueHandle, out typeAndShape));

                TensorElementType elemType = NativeMethods.OrtGetTensorElementType(typeAndShape);

                Type type = null;
                int width = 0;
                TensorElementTypeConverter.GetTypeAndWidth(elemType, out type, out width);
                if (typeof(T) != type)
                    throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>)+" does not support T = "+nameof(T));
                _elementWidth = width;

                _onnxValueHandle = onnxValueHandle;
                // derive the databuffer pointer, element_count, element_width, and shape
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(_onnxValueHandle, out _dataBufferHandle));
                // throws OnnxRuntimeException if native call failed

                ulong dimension = NativeMethods.OrtGetNumOfDimensions(typeAndShape);
                long count = NativeMethods.OrtGetTensorShapeElementCount(typeAndShape);  // count can be negative. 
                if (count < 0)
                {
                    throw new NotSupportedException("Symbolic dimensions in the tensor is not supported");
                }

                long[] shape = new long[dimension];
                NativeMethods.OrtGetDimensions(typeAndShape, shape, dimension); //Note: shape must be alive during the call

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
                    NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
                }
            }
        }
        ~NativeOnnxTensorMemory()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
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

        public int Count
        {
            get
            {
                return _elementCount;
            }
        }

        public int ElementWidth
        {
            get
            {
                return _elementWidth;
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

            NativeMethods.OrtReleaseValue(_onnxValueHandle);

            _disposed = true;
        }

        protected override bool TryGetArray(out ArraySegment<T> arraySegment)
        {
            // cannot expose managed array
            arraySegment = default(ArraySegment<T>);
            return false;
        }



        
    }
}
