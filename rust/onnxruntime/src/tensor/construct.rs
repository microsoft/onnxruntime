//! convert module has the trait for conversion of Inputs ConstructTensor.

use crate::{memory::MemoryInfo, OrtError};
use onnxruntime_sys::OrtAllocator;
use onnxruntime_sys::OrtValue;
use std::fmt::Debug;

/// The Input type for Rust onnxruntime Session::run
pub trait ConstructTensor: Debug {
    /// Constuct an OrtTensor Input using the `MemoryInfo` and a raw pointer to the `OrtAllocator`.
    fn construct<'a>(
        &'a mut self,
        memory_info: &MemoryInfo,
        allocator: *mut OrtAllocator,
    ) -> Result<Box<dyn InputTensor + 'a>, OrtError>;
}

/// Allows the return value of ConstructTensor::construct
/// to be generic.
pub trait InputTensor {
    /// The input tensor's shape
    fn shape(&self) -> &[usize];

    /// The input tensor's ptr
    fn ptr(&self) -> *mut OrtValue;
}

impl<'a, T> From<T> for Box<dyn ConstructTensor + 'a>
where
    T: ConstructTensor + 'a,
{
    fn from(other: T) -> Self {
        Box::new(other)
    }
}
