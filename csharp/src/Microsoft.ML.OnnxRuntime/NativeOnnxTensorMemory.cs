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
        private IntPtr _onnxValueHandle;      // pointer to onnxvalue object in native
        private IntPtr _dataBufferPointer;    // pointer to mutable tensor data in native memory
        private string[] _dataBufferAsString; // string tensor values copied into managed memory
        private int _elementCount;
        private int _elementWidth;
        private int[] _dimensions;

        public NativeOnnxTensorMemory(IntPtr onnxValueHandle)
        {
            IntPtr typeAndShape = IntPtr.Zero;
            try
            {
                Type type = null;
                int width = 0;
                _onnxValueHandle = onnxValueHandle;

                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(onnxValueHandle, out typeAndShape));
                TensorElementType elemType;
                unsafe
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, new IntPtr(&elemType)));
                }
                TensorElementTypeConverter.GetTypeAndWidth(elemType, out type, out width);

                if (typeof(T) != type)
                    throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>) + " does not support T = " + nameof(T));

                _elementWidth = width;
                UIntPtr dimension;
                long count;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(typeAndShape, out dimension));
                unsafe
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShape, new IntPtr(&count)));  // count can be negative. 
                }
                if (count < 0)
                {
                    throw new NotSupportedException("Symbolic dimensions in the tensor is not supported");
                }

                long[] shape = new long[dimension.ToUInt64()];
                unsafe
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(typeAndShape, shape, new UIntPtr(&dimension))); //Note: shape must be alive during the call
                }

                _elementCount = (int)count;
                _dimensions = new int[dimension.ToUInt64()];
                for (ulong i = 0; i < dimension.ToUInt64(); i++)
                {
                    _dimensions[i] = (int)shape[i];
                }

                if (typeof(T) != typeof(string))
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(_onnxValueHandle, out _dataBufferPointer));
                }
                else
                {
                    UIntPtr strLen;
                    var offsets = new UIntPtr[_elementCount];
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetStringTensorDataLength(_onnxValueHandle, out strLen));
                    var dataBuffer = new byte[strLen.ToUInt64()];
                    var dataBufferMemory = new Memory<byte>(dataBuffer);
                    var dataBufferHandle = dataBufferMemory.Pin();
                    IntPtr dataBufferPointer = IntPtr.Zero;

                    var offsetMemory = new Memory<UIntPtr>(offsets);
                    var offsetMemoryHandle = offsetMemory.Pin();
                    IntPtr offsetBufferPointer = IntPtr.Zero;
                    unsafe
                    {
                        dataBufferPointer = (IntPtr)dataBufferHandle.Pointer;
                        offsetBufferPointer = (IntPtr)offsetMemoryHandle.Pointer;
                    }
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetStringTensorContent(_onnxValueHandle, dataBufferPointer, strLen, offsetBufferPointer, (UIntPtr)_elementCount));
                    _dataBufferPointer = dataBufferPointer;
                    _dataBufferAsString = new string[_elementCount];

                    for (var i = 0; i < offsets.Length; i++)
                    {
                        var length = (i == offsets.Length - 1)
                            ? strLen.ToUInt64() - offsets[i].ToUInt64()
                            : offsets[i + 1].ToUInt64() - offsets[i].ToUInt64();
                        // Onnx specifies strings always in UTF-8, no trailing null, no leading BOM
                        _dataBufferAsString[i] = Encoding.UTF8.GetString(dataBuffer, (int)offsets[i], (int)length);
                    }

                    // unpin memory
                    offsetMemoryHandle.Dispose();
                    dataBufferHandle.Dispose();
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
                span = new Span<T>((void*)_dataBufferPointer, _elementCount);
            }

            return span;
        }

        public Memory<String> GetBytesAsStringMemory()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(NativeOnnxTensorMemory<T>));

            if (typeof(T) != typeof(string))
                throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>.GetBytesAsStringMemory) + ": T must be byte");

            return (_dataBufferAsString == null) ? new Memory<string>() : new Memory<string>(_dataBufferAsString);
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

                return new MemoryHandle((void*)((int)_dataBufferPointer + elementIndex * _elementWidth)); //could not use Unsafe.Add
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
