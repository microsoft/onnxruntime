//! Module containing tensor with memory owned by the ONNX Runtime

use std::{fmt::Debug, ops::Deref};

use ndarray::{Array, ArrayView};
use tracing::debug;

use onnxruntime_sys as sys;

use crate::{
    error::status_to_result, g_ort, memory::MemoryInfo, tensor::ndarray_tensor::NdArrayTensor,
    OrtError, Result, TypeToTensorElementDataType,
};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
#[allow(dead_code)] // This is to appease clippy as `memory_info` is not read.
pub struct OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    array_view: ArrayView<'t, T, D>,
    memory_info: &'m MemoryInfo,
}

impl<'t, 'm, T, D> Deref for OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array_view
    }
}

impl<'t, 'm, T, D> OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    /// Apply a softmax on the specified axis
    pub fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
    {
        self.array_view.softmax(axis)
    }
}

#[derive(Debug)]
pub(crate) struct OrtOwnedTensorExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    memory_info: &'m MemoryInfo,
    shape: D,
}

impl<'m, D> OrtOwnedTensorExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) fn new(memory_info: &'m MemoryInfo, shape: D) -> OrtOwnedTensorExtractor<'m, D> {
        OrtOwnedTensorExtractor {
            tensor_ptr: std::ptr::null_mut(),
            memory_info,
            shape,
        }
    }

    pub(crate) fn extract<'t, T>(self) -> Result<OrtOwnedTensor<'t, 'm, T, D>>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
    {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.

        assert_ne!(self.tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        (is_tensor == 1)
            .then_some(())
            .ok_or(OrtError::IsTensorCheck)?;

        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr.cast::<*mut std::ffi::c_void>();
        let status = unsafe {
            g_ort().GetTensorMutableData.unwrap()(self.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let array_view = unsafe { ArrayView::from_shape_ptr(self.shape, output_array_ptr) };

        Ok(OrtOwnedTensor {
            tensor_ptr: self.tensor_ptr,
            array_view,
            memory_info: self.memory_info,
        })
    }
}

impl<'t, 'm, T, D> Drop for OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping OrtOwnedTensor.");
        unsafe { g_ort().ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = std::ptr::null_mut();
    }
}
