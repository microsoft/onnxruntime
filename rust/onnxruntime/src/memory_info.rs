//! Module abstracting OrtMemoryInfo.

use crate::{
    ort_api,
    error::{assert_not_null_pointer, status_to_result, OrtResult, OrtError},
    allocator::AllocatorType, DeviceName, MemType,
};
use onnxruntime_sys as sys;
use std::{ffi::{CStr, CString}, convert::TryFrom};
use std::fmt::Debug;
use std::os::raw::c_char;
use tracing::{error, trace};

#[derive(Debug)]
/// MemoryInfo
pub struct MemoryInfo {
    pub(crate) ptr: *mut sys::OrtMemoryInfo,
    device_id: i32,
    name: DeviceName,
    allocator_type: AllocatorType,
    mem_type: MemType,
    drop: bool,
}

unsafe impl Send for MemoryInfo {}
unsafe impl Sync for MemoryInfo {}

impl MemoryInfo {
    #[tracing::instrument]
    /// Create new MemoryInfo
    pub fn new(
        name: DeviceName,
        device_id: i32,
        allocator_type: AllocatorType,
        mem_type: MemType,
    ) -> OrtResult<Self> {
        let mut ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();

        let status = unsafe {
            ort_api().CreateMemoryInfo.unwrap()(
                CString::from(name.clone()).as_ptr(),
                allocator_type.clone().into(),
                device_id,
                mem_type.clone().into(),
                &mut ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateMemoryInfo)?;
        assert_not_null_pointer(ptr, "MemoryInfo")?;

        trace!("Created MemoryInfo: {ptr:?}.");
        Ok(Self {
            ptr,
            device_id,
            name,
            allocator_type,
            mem_type,
            drop: true,
        })
    }

    /// gets the device_id from MemoryInfo
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// gets the name from MemoryInfo
    pub fn name(&self) -> &DeviceName {
        &self.name
    }

    /// gets the AllocatorType from MemoryInfo
    pub fn allocator_type(&self) -> &AllocatorType {
        &self.allocator_type
    }

    /// gets the MemType from MemoryInfo
    pub fn mem_type(&self) -> &MemType {
        &self.mem_type
    }

    #[allow(dead_code)]
    fn get_id(memory_info_ptr: *const sys::OrtMemoryInfo) -> OrtResult<i32> {
        let mut id = 0;
        let status = unsafe { ort_api().MemoryInfoGetId.unwrap()(memory_info_ptr, &mut id) };
        status_to_result(status).map_err(OrtError::MemoryInfoGetId)?;
        Ok(id)
    }

    #[allow(dead_code)]
    fn get_name(memory_info_ptr: *const sys::OrtMemoryInfo) -> OrtResult<DeviceName> {
        let mut name_ptr: *const c_char = std::ptr::null_mut();
        let status = unsafe { ort_api().MemoryInfoGetName.unwrap()(memory_info_ptr, &mut name_ptr) };
        status_to_result(status).map_err(OrtError::MemoryInfoGetName)?;
        assert_not_null_pointer(name_ptr, "Name")?;
        let name: String = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();
        Ok(DeviceName::from(name.as_str()))
    }

    #[allow(dead_code)]
    fn get_allocator_type(memory_info_ptr: *const sys::OrtMemoryInfo) -> OrtResult<AllocatorType> {
        let mut allocator_type = sys::OrtAllocatorType::OrtInvalidAllocator;
        let status =
            unsafe { ort_api().MemoryInfoGetType.unwrap()(memory_info_ptr, &mut allocator_type) };
        status_to_result(status).map_err(OrtError::MemoryInfoGetType)?;
        Ok(allocator_type.into())
    }

    #[allow(dead_code)]
    fn get_mem_type(memory_info_ptr: *const sys::OrtMemoryInfo) -> OrtResult<MemType> {
        let mut mem_type = sys::OrtMemType::OrtMemTypeDefault;
        let status =
            unsafe { ort_api().MemoryInfoGetMemType.unwrap()(memory_info_ptr, &mut mem_type) };
        status_to_result(status).map_err(OrtError::MemoryInfoGetMemType)?;
        Ok(mem_type.into())
    }
}

impl TryFrom<*const sys::OrtMemoryInfo> for MemoryInfo {
    type Error = OrtError;

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn try_from(ptr: *const sys::OrtMemoryInfo) -> OrtResult<Self> {
        trace!("Created MemoryInfo: {ptr:?}.");
        Ok(Self {
            ptr: ptr as *mut sys::OrtMemoryInfo,
            device_id: Self::get_id(ptr)?,
            name: Self::get_name(ptr)?,
            allocator_type: Self::get_allocator_type(ptr)?,
            mem_type: Self::get_mem_type(ptr)?,
            drop: false,
        })
    }
}

impl Drop for MemoryInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.drop {
            if self.ptr.is_null() {
                error!("MemoryInfo pointer is null, not dropping.");
            } else {
                trace!("Dropping MemoryInfo: {:?}.", self.ptr);
                unsafe { ort_api().ReleaseMemoryInfo.unwrap()(self.ptr) };
            }
        } else {
            trace!("MemoryInfo not dropped: {:?}.", self.ptr);
        }
        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_info_constructor_destructor() {
        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default).unwrap();
        std::mem::drop(memory_info);
    }
}
