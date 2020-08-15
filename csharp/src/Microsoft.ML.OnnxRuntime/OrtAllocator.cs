// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// See documentation for OrtAllocatorType in C API
    /// </summary>
    public enum OrtAllocatorType
    {
        DeviceAllocator = 0,
        ArenaAllocator = 1
    }

    /// <summary>
    /// See documentation for OrtMemType in C API
    /// </summary>
    public enum OrtMemType
    {
        CpuInput = -2,                      // Any CPU memory used by non-CPU execution provider
        CpuOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
        Cpu = CpuOutput,                    // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
        Default = 0,                        // the default allocator for execution provider
    }

    /// <summary>
    /// This class encapsulates and most of the time owns the underlying native OrtMemoryInfo instance.
    /// Instance returned from OrtAllocator will not own OrtMemoryInfo, the class must be disposed
    /// regardless.
    /// 
    /// Use this class to query and create OrtAllocator instances so you can pre-allocate memory for model
    /// inputs/outputs and use it for binding. Instances of the class can also used to created OrtValues bound
    /// to pre-allocated memory. In that case, the instance of OrtMemoryInfo contains the information about the allocator
    /// used to allocate the underlying memory.
    /// </summary>
    public class OrtMemoryInfo : IDisposable
    {
        private static readonly Lazy<OrtMemoryInfo> _defaultCpuAllocInfo = new Lazy<OrtMemoryInfo>(CreateCpuMemoryInfo);
        private IntPtr _pointer;
        private readonly bool _owned; // false if we are exposing OrtMemoryInfo from an allocator which owns it

        private static OrtMemoryInfo CreateCpuMemoryInfo()
        {
            IntPtr memoryInfo = IntPtr.Zero;
            // Returns OrtMemoryInfo instance that needs to be disposed
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCpuMemoryInfo(OrtAllocatorType.DeviceAllocator, OrtMemType.Cpu, out memoryInfo));
            return new OrtMemoryInfo(memoryInfo, true);
        }

        /// <summary>
        /// Default CPU based instance
        /// </summary>
        public static OrtMemoryInfo DefaultInstance
        {
            get
            {
                return _defaultCpuAllocInfo.Value;
            }
        }

        internal IntPtr Pointer
        {
            get
            {
                return _pointer;
            }
        }

        /// <summary>
        /// This allocator takes an native pointer to already existing
        /// instance of OrtMemoryInfo. That instance may either be owned or not
        /// owned. In the latter case, this class serves to expose native properties
        /// of the instance.
        /// </summary>
        /// <param name="allocInfo"></param>
        internal OrtMemoryInfo(IntPtr allocInfo, bool owned)
        {
            _pointer = allocInfo;
            _owned = owned;
        }

        // Predefined utf8 encoded allocator names. Use them to construct an instance of
        // OrtMemoryInfo to avoid UTF-16 to UTF-8 conversion
        public static readonly byte[] allocatorCPU = Encoding.UTF8.GetBytes("Cpu" + Char.MinValue);
        public static readonly byte[] allocatorCUDA = Encoding.UTF8.GetBytes("Cuda" + Char.MinValue);
        public static readonly byte[] allocatorCUDA_PINNED = Encoding.UTF8.GetBytes("CudaPinned" + Char.MinValue);
        /// <summary>
        /// Create an instance of OrtMemoryInfo according to the specification
        /// Memory info instances are usually used to get a handle of a native allocator
        /// that is present within the current inference session object. That, in turn, depends
        /// of what execution providers are available within the binary that you are using and are
        /// registered with Add methods.
        /// </summary>
        /// <param name="utf8AllocatorName">Allocator name. Use of the predefined above.</param>
        /// <param name="allocatorType">Allocator type</param>
        /// <param name="deviceId">Device id</param>
        /// <param name="memoryType">Memory type</param>
        public OrtMemoryInfo(byte[] utf8AllocatorName, OrtAllocatorType allocatorType, int deviceId, OrtMemType memoryType)
        {
            using (var pinnedName = new PinnedGCHandle(GCHandle.Alloc(utf8AllocatorName, GCHandleType.Pinned)))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateMemoryInfo(pinnedName.Pointer,
                                                                                allocatorType,
                                                                                deviceId,
                                                                                memoryType,
                                                                                out _pointer));
            }
            _owned = true;
        }

        /// <summary>
        /// Create an instance of OrtMemoryInfo according to the specification.
        /// </summary>
        /// <param name="allocatorName">Allocator name</param>
        /// <param name="allocatorType">Allocator type</param>
        /// <param name="deviceId">Device id</param>
        /// <param name="memoryType">Memory type</param>
        public OrtMemoryInfo(string allocatorName, OrtAllocatorType allocatorType, int deviceId, OrtMemType memoryType)
            : this(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(allocatorName), allocatorType, deviceId, memoryType)
        {
        }

        /// <summary>
        /// Name of the allocator associated with the OrtMemoryInfo instance
        /// </summary>
        public string Name
        {
            get
            {
                IntPtr utf8Name = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetName(_pointer, out utf8Name));
                return NativeOnnxValueHelper.StringFromNativeUtf8(utf8Name);
            }
        }

        /// <summary>
        /// Returns device ID
        /// </summary>
        public int Id
        {
            get
            {
                int id = 0;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetId(_pointer, out id));
                return id;
            }
        }

        /// <summary>
        ///  The below 2 are really properties but naming them is a challenge
        ///  as names would conflict with the returned type. Also, there are native
        ///  calls behind them so exposing them as Get() would be appropriate.
        /// </summary>
        /// <returns></returns>
        public OrtMemType GetMemoryType()
        {
            OrtMemType memoryType = OrtMemType.Default;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetMemType(_pointer, out memoryType));
            return memoryType;
        }

        /// <summary>
        /// Returns alloctor type
        /// </summary>
        /// <returns></returns>
        public OrtAllocatorType GetAllocatorType()
        {
            OrtAllocatorType allocatorType = OrtAllocatorType.ArenaAllocator;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetType(_pointer, out allocatorType));
            return allocatorType;
        }

        public override bool Equals(object obj)
        {
            var other = obj as OrtMemoryInfo;
            if(other == null)
            {
                return false;
            }
            return Equals(other);
        }

        public bool Equals(OrtMemoryInfo other)
        {
            if(this == other)
            {
                return true;
            }
            int result = -1;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCompareMemoryInfo(_pointer, other._pointer, out result));
            return (result == 0);
        }

        public override int GetHashCode()
        {
            return Pointer.ToInt32();
        }

        #region IDisposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_owned)
                {
                    NativeMethods.OrtReleaseMemoryInfo(_pointer);
                }
                _pointer = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // We intentionally do not provider an finalizer for the class
        #endregion
    }

    /// <summary>
    /// This class represents memory allocation made by a specific onnxruntime
    /// allocator. Use OrtAllocator.Allocate() to obtain an instance of this class.
    /// It implements IDisposable and makes use of the original allocator
    /// used to allocate the memory. The lifespan of the allocator instance must eclipse the
    /// lifespan of the allocation. Or, if you prefer, all OrtMemoryAllocation instances must be
    /// disposed of before the corresponding allocator instances are disposed of.
    /// </summary>
    public class OrtMemoryAllocation : IDisposable
    {
        private OrtAllocator _allocator;

        /// <summary>
        /// This constructs an instance representing an native memory allocation.
        /// Typically returned by OrtAllocator.Allocate(). However, some APIs return
        /// natively allocated IntPtr using a specific allocator. It is a good practice
        /// to wrap such a memory into OrtAllocation for proper disposal. You can set
        /// size to zero if not known, which is not important for disposing.
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="pointer"></param>
        /// <param name="size"></param>
        internal OrtMemoryAllocation(OrtAllocator allocator, IntPtr pointer, uint size)
        {
            _allocator = allocator;
            Pointer = pointer;
            Size = size;
        }

        /// <summary>
        /// Internal accessor to call native methods
        /// </summary>
        internal IntPtr Pointer { get; private set; }

        /// <summary>
        /// Returns the size of the allocation
        /// </summary>
        public uint Size { get; private set; }

        public OrtMemoryInfo Info
        {
            get
            {
                return _allocator.Info;
            }
        }
        #region IDisposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_allocator != null)
                {
                    _allocator.FreeMemory(Pointer);
                }
                Pointer = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }

    /// <summary>
    /// The class exposes native internal allocator for Onnxruntime.
    /// This allocator enables you to allocate memory from the internal
    /// memory pools including device allocations. Useful for binding.
    /// </summary>
    public class OrtAllocator : IDisposable
    {
        private static readonly Lazy<OrtAllocator> _defaultInstance = new Lazy<OrtAllocator>(GetDefaultCpuAllocator);
        private IntPtr _pointer;
        private readonly bool _owned;

        private static OrtAllocator GetDefaultCpuAllocator()
        {
            IntPtr allocator = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetAllocatorWithDefaultOptions(out allocator));
            // Instance of default cpu allocator is a native singleton
            // Do not dispose of
            return new OrtAllocator(allocator, false);
        }

        /// <summary>
        /// Default CPU allocator instance
        /// </summary>
        public static OrtAllocator DefaultInstance // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _defaultInstance.Value;
            }
        }

        internal IntPtr Pointer
        {
            get
            {
                return _pointer;
            }
        }

        /// <summary>
        /// Internal constructor wraps existing native allocators
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="owned"></param>
        internal OrtAllocator(IntPtr allocator, bool owned)
        {
            this._pointer = allocator;
            this._owned = owned;
        }

        /// <summary>
        /// Creates an instance of OrtAllocator according to the specifications in OrtMemorInfo.
        /// The requested allocator should be available within the given session instance. This means
        /// both, the native library was build with specific allocators (for instance CUDA) and the corresponding
        /// provider was added to SessionsOptions before instantiating the session object.
        /// </summary>
        /// <param name="session"></param>
        /// <param name="memInfo"></param>
        public OrtAllocator(InferenceSession session, OrtMemoryInfo memInfo)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateAllocator(session.Handle, memInfo.Pointer, out _pointer));
            _owned = true;
        }

        /// <summary>
        /// OrtMemoryInfo instance owned by the allocator
        /// </summary>
        public OrtMemoryInfo Info
        {
            get
            {
                IntPtr memoryInfo = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorGetInfo(_pointer, out memoryInfo));
                // This serves as an exposure of memory_info owned by the allocator
                return new OrtMemoryInfo(memoryInfo, false);
            }
        }

        /// <summary>
        /// Allocate native memory. Returns a disposable instance of OrtMemoryAllocation
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public OrtMemoryAllocation Allocate(uint size)
        {
            IntPtr allocation = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorAlloc(_pointer, (UIntPtr)size, out allocation));
            return new OrtMemoryAllocation(this, allocation, size);
        }

        /// <summary>
        /// This internal interface is used for freeing memory
        /// </summary>
        /// <param name="allocation"></param>
        internal void FreeMemory(IntPtr allocation)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorFree(_pointer, allocation));
        }

        #region IDisposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_owned)
                {
                    NativeMethods.OrtReleaseAllocator(_pointer);
                }
                _pointer = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // We intentionally do not provider an finalizer for the class
        #endregion
    }
}
