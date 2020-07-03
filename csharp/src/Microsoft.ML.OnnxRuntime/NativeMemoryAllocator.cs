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
    /// The only exception is when it is returned from the allocator, then the allocator owns the actual
    /// native instance.
    /// 
    /// Use this class to query and create MemoryAllocator instances so you can pre-allocate memory for model
    /// inputs/outputs and use it for binding
    /// </summary>
    public class MemoryInfo : SafeHandle
    {
        private static readonly Lazy<MemoryInfo> _defaultCpuAllocInfo = new Lazy<MemoryInfo>(CreateCpuMemoryInfo);
        private readonly bool _owned; // false if we are exposing OrtMemoryInfo from an allocator which owns it

        private static MemoryInfo CreateCpuMemoryInfo()
        {
            IntPtr allocInfo = IntPtr.Zero;
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

        internal IntPtr Handle  // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return handle;
            }
        }

        public override bool IsInvalid
        {
            get
            {
                return (handle == IntPtr.Zero);
            }
        }

        /// <summary>
        /// Default instance construction
        /// </summary>
        /// <param name="allocInfo"></param>
        internal MemoryInfo(IntPtr allocInfo, bool p_owned)
        : base(IntPtr.Zero, true)   //set 0 as invalid pointer
        {
            handle = allocInfo;
            _owned = p_owned;
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
            : base(IntPtr.Zero, true)    //set 0 as invalid pointer
        {
            var pinned_bytes = GCHandle.Alloc(utf8_allocator_name, GCHandleType.Pinned);
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateMemoryInfo(pinned_bytes.AddrOfPinnedObject(),
                                                                                alloc_type,
                                                                                device_id,
                                                                                mem_type,
                                                                                out handle));
            }
            finally
            {
                pinned_bytes.Free();
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtMemoryInfoGetName(handle, out utf8_name));
                // Encoding.UTF8.GetString()
                // Marshal.PtrTo
            }
        }

        protected override bool ReleaseHandle()
        {
            if (_owned)
            {
                NativeMethods.OrtReleaseMemoryInfo(handle);
            }
            return true;
        }
    }

    internal class NativeMemoryAllocator : SafeHandle
    {
        protected static readonly Lazy<NativeMemoryAllocator> _defaultInstance = new Lazy<NativeMemoryAllocator>(GetDefaultCpuAllocator);

        private static NativeMemoryAllocator GetDefaultCpuAllocator()
        {
            IntPtr allocator = IntPtr.Zero;
            try
            {
                IntPtr status = NativeMethods.OrtGetAllocatorWithDefaultOptions(out allocator);
                NativeApiStatus.VerifySuccess(status);
            }
            catch (Exception e)
            {
                throw e;
            }

            return new NativeMemoryAllocator(allocator);
        }

        static internal NativeMemoryAllocator DefaultInstance // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _defaultInstance.Value;
            }
        }

        /// <summary>
        /// Releases native memory previously allocated by the allocator
        /// </summary>
        /// <param name="memory"></param>
        internal void FreeMemory(IntPtr memory)
        {
            NativeMethods.OrtAllocatorFree(handle, memory);
        }

        public override bool IsInvalid
        {
            get
            {
                return (this.handle == IntPtr.Zero);
            }
        }

        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        private NativeMemoryAllocator(IntPtr allocator)
            : base(IntPtr.Zero, true)
        {
            this.handle = allocator;
        }

        protected override bool ReleaseHandle()
        {
            return true;
        }
    }

}
