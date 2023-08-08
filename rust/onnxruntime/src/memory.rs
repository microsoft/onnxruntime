use tracing::debug;

use onnxruntime_sys as sys;

use crate::{
    environment::{Environment, _Environment},
    error::{assert_not_null_pointer, status_to_result, OrtError, Result},
    AllocatorType, MemType,
};

use tracing::error;

#[derive(Debug)]
pub struct MemoryInfo {
    pub ptr: *mut sys::OrtMemoryInfo,
    env: _Environment,
}

impl MemoryInfo {
    #[tracing::instrument]
    pub fn new(allocator: AllocatorType, memory_type: MemType, env: &Environment) -> Result<Self> {
        debug!("Creating new memory info.");
        let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
        let status = unsafe {
            env.env().api().CreateCpuMemoryInfo.unwrap()(
                allocator.into(),
                memory_type.into(),
                &mut memory_info_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateCpuMemoryInfo)?;
        assert_not_null_pointer(memory_info_ptr, "MemoryInfo")?;

        Ok(Self {
            ptr: memory_info_ptr,
            env: env.env.clone(),
        })
    }
}

impl Drop for MemoryInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("MemoryInfo pointer is null, not dropping.");
        } else {
            debug!("Dropping the memory information.");
            unsafe { self.env.env().api().ReleaseMemoryInfo.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use std::env::var;

    use super::*;
    use crate::{environment::tests::ONNX_RUNTIME_LIBRARY_PATH, LoggingLevel};
    use test_log::test;

    #[test]
    fn memory_info_constructor_destructor() {
        let path = var(ONNX_RUNTIME_LIBRARY_PATH).ok();

        let builder = Environment::builder()
            .with_name("test")
            .with_log_level(LoggingLevel::Warning);

        let builder = if let Some(path) = path {
            builder.with_library_path(path)
        } else {
            builder
        };
        let env = builder.build().unwrap();

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, &env).unwrap();
        std::mem::drop(memory_info);
    }
}
