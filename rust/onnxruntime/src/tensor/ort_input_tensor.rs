//! Module containing tensor with memory owned by Rust

use super::construct::{ConstructTensor, InputTensor};
use crate::{
    environment::ENV,
    error::{assert_not_null_pointer, call_ort, status_to_result},
    memory::MemoryInfo,
    OrtError, Result, TensorElementDataType, TypeToTensorElementDataType,
};
use ndarray::{Array, Dimension};
use onnxruntime_sys as sys;
use std::{ffi, fmt::Debug};
use sys::OrtAllocator;
use tracing::{debug, error};

/// An Input tensor.
///
/// This ties the lifetime of T to the OrtValue; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
#[derive(Debug)]
pub struct OrtInputTensor<T>
where
    T: Debug,
{
    pub(crate) c_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
    #[allow(dead_code)]
    item: T,
}

impl<T> OrtInputTensor<T>
where
    T: Debug,
{
    /// The shape of the OrtTensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, D> ConstructTensor for Array<T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: Dimension,
{
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn construct<'a>(
        &'a mut self,
        memory_info: &MemoryInfo,
        allocator_ptr: *mut OrtAllocator,
    ) -> Result<Box<dyn InputTensor + 'a>> {
        // where onnxruntime will write the tensor data to
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

        let sh = self.shape().to_vec();

        let shape: Vec<i64> = self.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = self.shape().len();

        match T::tensor_element_data_type() {
            TensorElementDataType::Float
            | TensorElementDataType::Uint8
            | TensorElementDataType::Int8
            | TensorElementDataType::Uint16
            | TensorElementDataType::Int16
            | TensorElementDataType::Int32
            | TensorElementDataType::Int64
            | TensorElementDataType::Double
            | TensorElementDataType::Uint32
            | TensorElementDataType::Uint64 => {
                let buffer_size = self.len() * std::mem::size_of::<T>();

                // primitive data is already suitably laid out in memory; provide it to
                // onnxruntime as is
                let tensor_values_ptr: *mut std::ffi::c_void =
                    self.as_mut_ptr().cast::<std::ffi::c_void>();

                assert_not_null_pointer(tensor_values_ptr, "TensorValues")?;

                unsafe {
                    call_ort(|ort| {
                        ort.CreateTensorWithDataAsOrtValue.unwrap()(
                            memory_info.ptr,
                            tensor_values_ptr,
                            buffer_size,
                            shape_ptr,
                            shape_len,
                            T::tensor_element_data_type().into(),
                            tensor_ptr_ptr,
                        )
                    })
                }
                .map_err(OrtError::CreateTensorWithData)?;
                assert_not_null_pointer(tensor_ptr, "Tensor")?;

                let mut is_tensor = 0;
                let status = unsafe {
                    ENV.get().unwrap().lock().unwrap().api().IsTensor.unwrap()(
                        tensor_ptr,
                        &mut is_tensor,
                    )
                };
                status_to_result(status).map_err(OrtError::IsTensor)?;
            }
            TensorElementDataType::String => {
                // create tensor without data -- data is filled in later
                unsafe {
                    call_ort(|ort| {
                        ort.CreateTensorAsOrtValue.unwrap()(
                            allocator_ptr,
                            shape_ptr,
                            shape_len,
                            T::tensor_element_data_type().into(),
                            tensor_ptr_ptr,
                        )
                    })
                }
                .map_err(OrtError::CreateTensor)?;

                // create null-terminated copies of each string, as per `FillStringTensor` docs
                let null_terminated_copies: Vec<ffi::CString> = self
                    .iter()
                    .map(|elt| {
                        let slice = elt
                            .try_utf8_bytes()
                            .expect("String data type must provide utf8 bytes");
                        ffi::CString::new(slice)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(OrtError::CStringNulError)?;

                let string_pointers = null_terminated_copies
                    .iter()
                    .map(|cstring| cstring.as_ptr())
                    .collect::<Vec<_>>();

                unsafe {
                    call_ort(|ort| {
                        ort.FillStringTensor.unwrap()(
                            tensor_ptr,
                            string_pointers.as_ptr(),
                            string_pointers.len(),
                        )
                    })
                }
                .map_err(OrtError::FillStringTensor)?;
            }
        }

        assert_not_null_pointer(tensor_ptr, "Tensor")?;

        Ok(Box::new(OrtInputTensor {
            c_ptr: tensor_ptr,
            shape: sh,
            item: self,
        }))
    }
}

impl<T> Drop for OrtInputTensor<T>
where
    T: Debug,
{
    #[tracing::instrument]
    fn drop(&mut self) {
        // We need to let the C part free
        debug!("Dropping Tensor.");
        if self.c_ptr.is_null() {
            error!("Null pointer, not calling free.");
        } else {
            unsafe {
                ENV.get()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .api()
                    .ReleaseValue
                    .unwrap()(self.c_ptr)
            }
        }

        self.c_ptr = std::ptr::null_mut();
    }
}

impl<T, D> InputTensor for OrtInputTensor<&mut Array<T, D>>
where
    T: TypeToTensorElementDataType + Debug,
    D: Dimension,
{
    fn ptr(&self) -> *mut sys::OrtValue {
        self.c_ptr
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        environment::{tests::ONNX_RUNTIME_LIBRARY_PATH, Environment},
        AllocatorType, LoggingLevel, MemType,
    };
    use ndarray::{arr0, arr1, arr2, arr3};
    use once_cell::sync::Lazy;
    use std::env::var;
    use test_log::test;

    static ENV: Lazy<Environment> = Lazy::new(|| {
        let path = var(ONNX_RUNTIME_LIBRARY_PATH).ok();

        let builder = Environment::builder()
            .with_name("test")
            .with_log_level(LoggingLevel::Warning);
        let builder = if let Some(path) = path {
            builder.with_library_path(path)
        } else {
            builder
        };

        builder.build().unwrap()
    });

    #[test]
    fn orttensor_from_array_0d_i32() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr0::<i32>(123);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        let expected_shape: &[usize] = &[];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_1d_i32() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        let expected_shape: &[usize] = &[6];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_2d_i32() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        assert_eq!(tensor.shape(), &[2, 6]);
    }

    #[test]
    fn orttensor_from_array_3d_i32() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 6]);
    }

    #[test]
    fn orttensor_from_array_1d_string() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr1(&[
            String::from("foo"),
            String::from("bar"),
            String::from("baz"),
        ]);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn orttensor_from_array_3d_str() {
        let env = &*ENV;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, env).unwrap();
        let mut array = arr3(&[
            [["1", "2", "3"], ["4", "5", "6"]],
            [["7", "8", "9"], ["10", "11", "12"]],
        ]);
        let tensor = array
            .construct(&memory_info, ort_default_allocator())
            .unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 3]);
    }

    fn ort_default_allocator() -> *mut sys::OrtAllocator {
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        unsafe {
            // this default non-arena allocator doesn't need to be deallocated
            call_ort(|ort| ort.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr))
        }
        .unwrap();
        allocator_ptr
    }
}
