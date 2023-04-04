use tracing::debug;

use onnxruntime_sys as sys;

use crate::{
    error::{status_to_result, OrtError, Result},
    g_ort, AllocatorType, MemType,
};

#[derive(Debug)]
pub(crate) struct MemoryInfo {
    pub ptr: *mut sys::OrtMemoryInfo,
}

impl MemoryInfo {
    #[tracing::instrument]
    pub fn new(allocator: AllocatorType, memory_type: MemType) -> Result<Self> {
        debug!("Creating new memory info.");
        let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort().CreateCpuMemoryInfo.unwrap()(
                allocator.into(),
                memory_type.into(),
                &mut memory_info_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateCpuMemoryInfo)?;
        assert_ne!(memory_info_ptr, std::ptr::null_mut());

        Ok(Self {
            ptr: memory_info_ptr,
        })
    }
}

impl Drop for MemoryInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping the memory information.");
        assert_ne!(self.ptr, std::ptr::null_mut());

        unsafe { g_ort().ReleaseMemoryInfo.unwrap()(self.ptr) };

        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_env_log::test;

    #[test]
    fn memory_info_constructor_destructor() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        std::mem::drop(memory_info);
    }
}
