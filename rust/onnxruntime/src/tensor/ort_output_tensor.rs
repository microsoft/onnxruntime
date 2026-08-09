//! Module containing tensor with memory owned by the ONNX Runtime

use crate::{
    environment::{_Environment, ENV},
    error::status_to_result,
    OrtError, Result, TypeToTensorElementDataType,
};
use ndarray::ArrayView;
use onnxruntime_sys as sys;

use std::{convert::TryFrom, fmt::Debug};
use tracing::debug;

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
#[derive(Debug)]
pub struct OrtOutputTensor {
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
    env: _Environment,
}

#[derive(Debug)]
pub(crate) struct OrtOwnedTensorExtractor {
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
    env: _Environment,
}

impl OrtOwnedTensorExtractor {
    pub(crate) fn new(shape: Vec<usize>, env: _Environment) -> OrtOwnedTensorExtractor {
        OrtOwnedTensorExtractor {
            tensor_ptr: std::ptr::null_mut(),
            shape,
            env,
        }
    }

    pub(crate) fn extract(self) -> Result<OrtOutputTensor> {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.

        assert_ne!(self.tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status =
            unsafe { self.env.env().api().IsTensor.unwrap()(self.tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        (is_tensor == 1)
            .then_some(())
            .ok_or(OrtError::IsTensorCheck)?;

        Ok(OrtOutputTensor {
            tensor_ptr: self.tensor_ptr,
            shape: self.shape,
            env: self.env,
        })
    }
}

impl Drop for OrtOutputTensor {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping OrtOwnedTensor.");
        unsafe { self.env.env().api().ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = std::ptr::null_mut();
    }
}

/// An Output tensor with the ptr and the item that will copy from the ptr.
///
/// The view is materialized on each access via [`view()`](Self::view) to ensure the
/// borrowed lifetime is tied to `&self`, preventing the view from outliving the
/// underlying buffer owned by the `OrtOutputTensor`.
pub struct WithOutputTensor<T> {
    pub(crate) tensor: OrtOutputTensor,
    data_ptr: *const T,
    shape: Vec<usize>,
}

impl<T> Debug for WithOutputTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithOutputTensor")
            .field("tensor", &self.tensor)
            .field("data_ptr", &self.data_ptr)
            .field("shape", &self.shape)
            .finish()
    }
}

// SAFETY: The data pointer is derived from OrtOutputTensor which owns the allocation.
// Access is only possible through &self (via view()), so Send/Sync follow from T: Send/Sync.
unsafe impl<T: Send> Send for WithOutputTensor<T> {}
unsafe impl<T: Sync> Sync for WithOutputTensor<T> {}

impl<T> WithOutputTensor<T> {
    /// Returns an [`ArrayView`] over the output tensor data.
    ///
    /// The returned view borrows `self`, so it cannot outlive the tensor owner.
    pub fn view(&self) -> ArrayView<'_, T, ndarray::IxDyn> {
        unsafe { ArrayView::from_shape_ptr(ndarray::IxDyn(&self.shape), self.data_ptr) }
    }
}

impl<T> TryFrom<OrtOutputTensor> for WithOutputTensor<T>
where
    T: TypeToTensorElementDataType,
{
    type Error = OrtError;

    fn try_from(value: OrtOutputTensor) -> Result<Self> {
        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr.cast::<*mut std::ffi::c_void>();
        let status = unsafe {
            ENV.get()
                .unwrap()
                .lock()
                .unwrap()
                .api()
                .GetTensorMutableData
                .unwrap()(value.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let shape = value.shape.clone();

        Ok(WithOutputTensor {
            tensor: value,
            data_ptr: output_array_ptr,
            shape,
        })
    }
}

/// The onnxruntime Run output type.
pub enum OrtOutput {
    /// Tensor of f32s
    Float(WithOutputTensor<f32>),
    /// Tensor of f64s
    Double(WithOutputTensor<f64>),
    /// Tensor of u8s
    UInt8(WithOutputTensor<u8>),
    /// Tensor of u16s
    UInt16(WithOutputTensor<u16>),
    /// Tensor of u32s
    UInt32(WithOutputTensor<u32>),
    /// Tensor of u64s
    UInt64(WithOutputTensor<u64>),
    /// Tensor of i8s
    Int8(WithOutputTensor<i8>),
    /// Tensor of i16s
    Int16(WithOutputTensor<i16>),
    /// Tensor of i32s
    Int32(WithOutputTensor<i32>),
    /// Tensor of i64s
    Int64(WithOutputTensor<i64>),
    /// Tensor of Strings
    String(WithOutputTensor<String>),
}

impl OrtOutput {
    /// Return `WithOutputTensor<f32>` which provides a `view()` method for an `ArrayView`.
    pub fn float_array(&self) -> Option<&WithOutputTensor<f32>> {
        if let Self::Float(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<f64>` which provides a `view()` method for an `ArrayView`.
    pub fn double_array(&self) -> Option<&WithOutputTensor<f64>> {
        if let Self::Double(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<u8>` which provides a `view()` method for an `ArrayView`.
    pub fn uint8_array(&self) -> Option<&WithOutputTensor<u8>> {
        if let Self::UInt8(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<u16>` which provides a `view()` method for an `ArrayView`.
    pub fn uint16_array(&self) -> Option<&WithOutputTensor<u16>> {
        if let Self::UInt16(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<u32>` which provides a `view()` method for an `ArrayView`.
    pub fn uint32_array(&self) -> Option<&WithOutputTensor<u32>> {
        if let Self::UInt32(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<u64>` which provides a `view()` method for an `ArrayView`.
    pub fn uint64_array(&self) -> Option<&WithOutputTensor<u64>> {
        if let Self::UInt64(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<i8>` which provides a `view()` method for an `ArrayView`.
    pub fn int8_array(&self) -> Option<&WithOutputTensor<i8>> {
        if let Self::Int8(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<i16>` which provides a `view()` method for an `ArrayView`.
    pub fn int16_array(&self) -> Option<&WithOutputTensor<i16>> {
        if let Self::Int16(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<i32>` which provides a `view()` method for an `ArrayView`.
    pub fn int32_array(&self) -> Option<&WithOutputTensor<i32>> {
        if let Self::Int32(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<i64>` which provides a `view()` method for an `ArrayView`.
    pub fn int64_array(&self) -> Option<&WithOutputTensor<i64>> {
        if let Self::Int64(item) = self {
            Some(item)
        } else {
            None
        }
    }

    /// Return `WithOutputTensor<String>` which provides a `view()` method for an `ArrayView`.
    pub fn string_array(&self) -> Option<&WithOutputTensor<String>> {
        if let Self::String(item) = self {
            Some(item)
        } else {
            None
        }
    }
}

impl TryFrom<OrtOutputTensor> for OrtOutput {
    type Error = OrtError;

    fn try_from(value: OrtOutputTensor) -> Result<OrtOutput> {
        unsafe {
            let mut shape_info = std::ptr::null_mut();

            let status = ENV
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .api()
                .GetTensorTypeAndShape
                .unwrap()(value.tensor_ptr, &mut shape_info);

            status_to_result(status).map_err(OrtError::IsTensor)?;

            assert_ne!(shape_info, std::ptr::null_mut());

            let mut element_type =
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

            let status = ENV
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .api()
                .GetTensorElementType
                .unwrap()(shape_info, &mut element_type);

            status_to_result(status).map_err(OrtError::IsTensor)?;

            ENV.get()
                .unwrap()
                .lock()
                .unwrap()
                .api()
                .ReleaseTensorTypeAndShapeInfo
                .unwrap()(shape_info);

            match element_type {
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Float)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::UInt8)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Int8)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::UInt16)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Int16)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Int32)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Int64)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => {
                    WithOutputTensor::try_from(value).map(OrtOutput::String)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
                    WithOutputTensor::try_from(value).map(OrtOutput::Double)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::UInt32)
                }
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
                    WithOutputTensor::try_from(value).map(OrtOutput::UInt64)
                }
                // Unimplemented output tensor data types
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ
                | sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => {
                    unimplemented!("{:?}", element_type)
                }
            }
        }
    }
}
