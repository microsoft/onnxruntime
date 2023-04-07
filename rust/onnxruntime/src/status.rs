//! Module abstracting OrtStatus.

use crate::{char_ptr_to_string, g_ort, OrtApiError, OrtError};
use onnxruntime_sys as sys;
use std::os::raw::c_char;
use std::result::Result;
use tracing::trace;

#[derive(Debug)]
/// OrtStatus used to unwrap errors to the C API
pub struct Status {
    ptr: *const sys::OrtStatus,
}

impl From<*const sys::OrtStatus> for Status {
    fn from(ptr: *const sys::OrtStatus) -> Self {
        Status {
            ptr: ptr.cast_mut(),
        }
    }
}

impl From<Status> for Result<(), OrtApiError> {
    fn from(status: Status) -> Self {
        if status.ptr.is_null() {
            Ok(())
        } else {
            let raw: *const c_char = unsafe { g_ort().GetErrorMessage.unwrap()(status.ptr) };
            match char_ptr_to_string(raw) {
                Ok(msg) => Err(OrtApiError::Msg(msg)),
                Err(err) => match err {
                    OrtError::StringConversion(OrtApiError::IntoStringError(e)) => {
                        Err(OrtApiError::IntoStringError(e))
                    }
                    _ => unreachable!(),
                },
            }
        }
    }
}

impl Drop for Status {
    #[tracing::instrument]
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            trace!("Dropping Status: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseStatus.unwrap()(self.ptr as *mut sys::OrtStatus) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
