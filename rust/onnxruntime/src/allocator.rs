//! Module abstracting OrtAllocator.

use crate::{
    ort_api,
    error::{assert_not_null_pointer, status_to_result, OrtResult, OrtError}
};
use onnxruntime_sys as sys;
use std::fmt::Debug;

#[derive(Debug)]
/// A session Allocator
pub struct Allocator {
    pub(crate) ptr: *mut sys::OrtAllocator,
}

impl Allocator {
    /// try to create new default Allocator
    pub fn try_new() -> OrtResult<Self> {
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe { ort_api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;
        Ok(Self::from(allocator_ptr))
    }
}

impl From<*mut sys::OrtAllocator> for Allocator {
    fn from(ptr: *mut sys::OrtAllocator) -> Self {
        Self { ptr }
    }
}

impl Drop for Allocator {
    #[tracing::instrument]
    fn drop(&mut self) {
        // docs state 'Returned value should NOT be freed'
    }
}

/// Allocator type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
    /// Invalid allocator
    Invalid,
    /// Device allocator
    Device,
    /// Arena allocator
    Arena,
}

impl From<AllocatorType> for sys::OrtAllocatorType {
    fn from(val: AllocatorType) -> Self {
        match val {
            AllocatorType::Invalid => sys::OrtAllocatorType::OrtInvalidAllocator,
            AllocatorType::Device => sys::OrtAllocatorType::OrtDeviceAllocator,
            AllocatorType::Arena => sys::OrtAllocatorType::OrtArenaAllocator,
        }
    }
}

impl From<sys::OrtAllocatorType> for AllocatorType {
    fn from(val: sys::OrtAllocatorType) -> Self {
        match val {
            sys::OrtAllocatorType::OrtInvalidAllocator => AllocatorType::Invalid,
            sys::OrtAllocatorType::OrtDeviceAllocator => AllocatorType::Device,
            sys::OrtAllocatorType::OrtArenaAllocator => AllocatorType::Arena,
        }
    }
}
