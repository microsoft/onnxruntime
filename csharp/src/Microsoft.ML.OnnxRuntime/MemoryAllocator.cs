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
    public enum AllocatorType
    {
        DeviceAllocator = 0,
        ArenaAllocator = 1
    }

    /// <summary>
    /// See documentation for OrtMemType in C API
    /// </summary>
    public enum MemoryType
    {
        CpuInput = -2,                      // Any CPU memory used by non-CPU execution provider
        CpuOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
        Cpu = CpuOutput,                    // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
        Default = 0,                        // the default allocator for execution provider
    }

    /// <summary>
    /// This class encapsulates and most of the time owns the underlying native OrtMemoryInfo instance.
    /// Instance returned from MemoryAllocator will not own OrtMemoryInfo, the class must be disposed
    /// regardless.
    /// 
    /// Use this class to query and create MemoryAllocator instances so you can pre-allocate memory for model
    /// inputs/outputs and use it for binding. Instances of the class can also used to created OrtValues bound
    /// to pre-allocated memory. In that case, the instance of MemoryInfo contains the information about the allocator
    /// used to allocate the underlying memory.
    /// </summary>
    public class MemoryInfo : IDisposable
    {
        private static readonly Lazy<MemoryInfo> _defaultCpuAllocInfo = new Lazy<MemoryInfo>(CreateCpuMemoryInfo);
        private IntPtr _pointer;
        private readonly bool _owned; // false if we are exposing OrtMemoryInfo from an allocator which owns it

        private static MemoryInfo CreateCpuMemoryInfo()
        {
            IntPtr allocInfo = IntPtr.Zero;
            // Returns OrtMemoryInfo instance that needs to be disposed
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCpuMemoryInfo(AllocatorType.DeviceAllocator, MemoryType.Cpu, out allocInfo));
            return new MemoryInfo(allocInfo, true);
        }

        public static MemoryInfo DefaultInstance
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
        /// instance of MemoryInfo. That instance may either be owned or not
        /// owned. In the latter case, this class serves to expose native properties
        /// of the instance.
        /// </summary>
        /// <param name="allocInfo"></param>
        internal MemoryInfo(IntPtr allocInfo, bool owned)
        {
            _pointer = allocInfo;
            _owned = owned;
        }

        // Predefined utf8 encoded allocator names. Use them to construct an instance of
        // MemoryInfo
        public static readonly byte[] CPU_allocator = Encoding.UTF8.GetBytes("Cpu" + '\0');
        public static readonly byte[] CUDA_allocator = Encoding.UTF8.GetBytes("Cuda" + '\0');
        public static readonly byte[] CUDA_PINNED_allocator = Encoding.UTF8.GetBytes("CudaPinned" + '\0');
        /// <summary>
        /// Create an instance of MemoryInfo according to the specification
        /// Memory info instances are usually used to get a handle of a native allocator
        /// that is present within the current inference session object. That, in turn, depends
        /// of what execution providers are available within the binary that you are using and are
        /// registered with Add methods.
        /// </summary>
        /// <param name="utf8_allocator_name">Allocator name. Use of the predefined above.</param>
        /// <param name="alloc_type">Allocator type</param>
        /// <param name="device_id">Device id</param>
        /// <param name="mem_type">Memory type</param>
        public MemoryInfo(byte[] utf8_allocator_name, AllocatorType alloc_type, int device_id, MemoryType mem_type)
        {
            using (var pinned_handle = new PinnedGCHandle(GCHandle.Alloc(utf8_allocator_name, GCHandleType.Pinned)))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateMemoryInfo(pinned_handle.Pointer,
                                                                                alloc_type,
                                                                                device_id,
                                                                                mem_type,
                                                                                out _pointer));
            }
            _owned = true;
        }

        /// <summary>
        /// Create an instance of MemoryInfo according to the specification.
        /// </summary>
        /// <param name="allocator_name">Allocator name</param>
        /// <param name="alloc_type">Allocator type</param>
        /// <param name="device_id">Device id</param>
        /// <param name="mem_type">Memory type</param>
        public MemoryInfo(string allocator_name, AllocatorType alloc_type, int device_id, MemoryType mem_type)
            : this(Encoding.UTF8.GetBytes(allocator_name + '\0'), alloc_type, device_id, mem_type)
        {
        }

        public string Name
        {
            get
            {
                IntPtr utf8_name = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetName(_pointer, out utf8_name));
                return NativeOnnxValueHelper.StringFromNativeUtf8(utf8_name);
            }
        }

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
        public MemoryType GetMemoryType()
        {
            MemoryType mem_type = MemoryType.Default;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetMemType(_pointer, out mem_type));
            return mem_type;
        }

        public AllocatorType GetAllocatorType()
        {
            AllocatorType alloc_type = AllocatorType.ArenaAllocator;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetType(_pointer, out alloc_type));
            return alloc_type;
        }

        public bool CompareMemoryInfo(MemoryInfo other)
        {
            int result = -1;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCompareMemoryInfo(_pointer, other._pointer, out result));
            return (result == 0);
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
    /// allocator. Use MemoryAllocator.Allocate() to obtain an instance of this class.
    /// It implements IDisposable and makes use of the original allocator
    /// used to allocate the memory. The lifespan of the allocator instance must eclipse the
    /// lifespan of the allocation. Or, if you prefer, all MemoryAllocation instances must be
    /// disposed of before the corresponding allocator instances are disposed of.
    /// </summary>
    public class MemoryAllocation : IDisposable
    {
        private MemoryAllocator _allocator;

        /// <summary>
        /// Bind an arbitrary piece of native memory to the instance
        /// The instance will not have the ownership of this memory.
        /// </summary>
        /// <param name="pointer"></param>
        /// <param name="size"></param>
        public MemoryAllocation(IntPtr pointer, uint size)
        {
            _allocator = null;
            Pointer = pointer;
            Size = size;
        }

        /// <summary>
        /// This an instance with a piece of memory allocated
        /// by onnxruntime MemoryAllocator. The same allocator will be used for
        /// for memory disposal. For memory allocated elsewhere, the instance will not own the memory
        /// and will not dispose of it.
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="pointer"></param>
        /// <param name="size"></param>
        internal MemoryAllocation(MemoryAllocator allocator, IntPtr pointer, uint size)
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

        public MemoryInfo Info
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

    public class MemoryAllocator : IDisposable
    {
        private static readonly Lazy<MemoryAllocator> _defaultInstance = new Lazy<MemoryAllocator>(GetDefaultCpuAllocator);
        private IntPtr _pointer;
        private readonly bool _owned;

        private static MemoryAllocator GetDefaultCpuAllocator()
        {
            IntPtr allocator = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetAllocatorWithDefaultOptions(out allocator));
            // Instance of default cpu allocator is a native singleton
            // Do not dispose of
            return new MemoryAllocator(allocator, false);
        }

        public static MemoryAllocator DefaultInstance // May throw exception in every access, if the constructor have thrown an exception
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
        internal MemoryAllocator(IntPtr allocator, bool owned)
        {
            this._pointer = allocator;
            this._owned = owned;
        }

        public MemoryAllocator(InferenceSession session, MemoryInfo memInfo)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateAllocator(session.Handle, memInfo.Pointer, out _pointer));
            _owned = true;
        }

        public MemoryInfo Info
        {
            get
            {
                IntPtr mem_info = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorGetInfo(_pointer, out mem_info));
                // This serves as an exposure of memory_info owned by the allocator
                return new MemoryInfo(mem_info, false);
            }
        }

        /// <summary>
        /// Allocate native memory
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public MemoryAllocation Allocate(uint size)
        {
            IntPtr allocation = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorAlloc(_pointer, (UIntPtr)size, out allocation));
            return new MemoryAllocation(this, allocation, size);
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
