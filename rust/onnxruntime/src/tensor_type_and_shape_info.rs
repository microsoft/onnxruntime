//! Module abstracting OrtTensorTypeAndShapeInfo.

use crate::{
    ort_api,
    error::{status_to_result, OrtResult, OrtError},
    TensorElementDataType, OrtValue,
};
use onnxruntime_sys as sys;
use std::{fmt::Debug, convert::{TryFrom, TryInto}};
use tracing::{error, trace};

#[derive(Debug)]
/// A tensor’s type and shape information.
pub struct TensorTypeAndShapeInfo {
    ptr: *mut sys::OrtTensorTypeAndShapeInfo,
    /// The Tensor Data Type
    pub element_data_type: TensorElementDataType,
    /// The shape of the Tensor
    pub dimensions: Vec<i64>,
}

impl TryFrom<*mut sys::OrtTensorTypeAndShapeInfo> for TensorTypeAndShapeInfo {
    type Error = OrtError;

    fn try_from(ptr: *mut sys::OrtTensorTypeAndShapeInfo) -> OrtResult<Self> {
        let element_data_type = TensorTypeAndShapeInfo::try_get_data_type(ptr)?;
        let dimensions = TensorTypeAndShapeInfo::try_get_dimensions(ptr)?;

        Ok(Self {
            ptr,
            element_data_type,
            dimensions,
        })
    }
}

impl TensorTypeAndShapeInfo {
    pub(crate) fn try_new(ort_value: &OrtValue) -> OrtResult<Self> {
        // ensure tensor
        ort_value
            .is_tensor()?
            .then_some(())
            .ok_or(OrtError::NotTensor)?;

        // get info
        let mut ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe { ort_api().GetTensorTypeAndShape.unwrap()(**ort_value, &mut ptr) };
        status_to_result(status)
            .map_err(OrtError::GetTensorTypeAndShape)
            .unwrap();

        trace!("Created TensorTypeAndShapeInfo: {ptr:?}.");
        ptr.try_into()
    }

    fn try_get_data_type(
        type_and_shape_info: *mut sys::OrtTensorTypeAndShapeInfo,
    ) -> OrtResult<TensorElementDataType> {
        let mut onnx_data_type =
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status = unsafe {
            ort_api().GetTensorElementType.unwrap()(type_and_shape_info, &mut onnx_data_type)
        };
        status_to_result(status).map_err(OrtError::GetTensorElementType)?;
        (onnx_data_type != sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
            .then_some(())
            .ok_or(OrtError::UndefinedTensorElementType)?;

        Ok(onnx_data_type.into())
    }

    fn try_get_dimensions(
        type_and_shape_info: *mut sys::OrtTensorTypeAndShapeInfo,
    ) -> OrtResult<Vec<i64>> {
        let mut num_dims = 0;
        let status =
            unsafe { ort_api().GetDimensionsCount.unwrap()(type_and_shape_info, &mut num_dims) };
        status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
        (num_dims != 0)
            .then_some(())
            .ok_or(OrtError::InvalidDimensions)?;

        let mut dimensions: Vec<i64> = vec![0; num_dims];
        let status = unsafe {
            ort_api().GetDimensions.unwrap()(type_and_shape_info, dimensions.as_mut_ptr(), num_dims)
        };
        status_to_result(status).map_err(OrtError::GetDimensions)?;

        Ok(dimensions)
    }

    /// Return the tensor dimensions as usize instead of i64.
    pub fn get_dimensions_as_usize(&self) -> Vec<usize> {
        self.dimensions.iter()
            .map(|dim| *dim as usize)
            .collect::<Vec<_>>()
    }
}

impl Drop for TensorTypeAndShapeInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("TensorTypeAndShapeInfo pointer is null, not dropping.");
        } else {
            trace!("Dropping TensorTypeAndShapeInfo: {:?}.", self.ptr);
            unsafe { ort_api().ReleaseTensorTypeAndShapeInfo.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}

