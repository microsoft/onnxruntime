//! Module abstracting OrtValue.

use std::{fmt::Debug, convert::TryFrom};
use std::ops::Deref;

use ndarray::{Array, ArrayView, Dim, IxDynImpl};
use tracing::{error, trace};

use onnxruntime_sys as sys;

use crate::session::Session;
use crate::{
    ort_api,
    error::{status_to_result, NonMatchingDataTypes, NonMatchingDeviceName, OrtResult, OrtError},
    DeviceName, MemoryInfo, TensorElementDataType,
    TensorTypeAndShapeInfo, TypeToTensorElementDataType,
};

#[derive(Debug)]
/// An ::OrtValue
pub struct OrtValue {
    pub(crate) ptr: *mut sys::OrtValue,
}

unsafe impl Send for OrtValue {}
unsafe impl Sync for OrtValue {}

impl From<*mut sys::OrtValue> for OrtValue {
    fn from(ptr: *mut sys::OrtValue) -> Self {
        trace!("Created Value: {ptr:?}.");
        Self { ptr }
    }
}

impl Default for OrtValue {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }
}

impl OrtValue {
    pub(crate) fn new(ptr: *mut sys::OrtValue) -> Self {
        trace!("Created Value {ptr:?}.");
        Self { ptr }
    }

    /// Create an OrtValue with a specified element type and shape that owns its data.
    /// A raw pointer to the data is returned, and it should be initialized directly 
    /// after this call.
    pub(crate) fn new_from_type_and_shape<T>(session: &Session, shape: &[i64]) 
    -> OrtResult<(OrtValue, *mut T)> 
    where
        T: TypeToTensorElementDataType,
    {
        let elem_type = T::tensor_element_data_type();

        let shape_ptr = shape.as_ptr();
        let shape_len = shape.len();
    
        // output is this sys::OrtValue
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
    
        let status = unsafe {
            ort_api().CreateTensorAsOrtValue.unwrap()(
                session.allocator.ptr,
                shape_ptr,
                shape_len,
                elem_type.into(),
                &mut tensor_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;

        // Allocate memory for data pointer
        let mut data_ptr: *mut T = std::ptr::null_mut();
        // This is the address of data_ptr
        let data_ptr_ptr: *mut *mut T = &mut data_ptr;
        let data_void_ptr_ptr: *mut *mut std::ffi::c_void = data_ptr_ptr as *mut *mut std::ffi::c_void;

        let status =
            unsafe { ort_api().GetTensorMutableData.unwrap()(tensor_ptr, data_void_ptr_ptr) };
        status_to_result(status).map_err(OrtError::GetTensorMutableData)?;

        Ok( (OrtValue::new(tensor_ptr), data_ptr) )
    }

    /// Return if an OrtValue is a tensor type.
    pub fn is_tensor(&self) -> OrtResult<bool> {
        let mut is_tensor = 0;
        let status = unsafe { ort_api().IsTensor.unwrap()(self.ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;

        Ok(is_tensor == 1)
    }

    /// Return OrtTensorTypeAndShapeInfo if OrtValue is a tensor type.
    pub fn type_and_shape_info(&self) -> OrtResult<TensorTypeAndShapeInfo> {
        TensorTypeAndShapeInfo::try_new(self)
    }

    /// Return MemoryInfo of OrtValue
    pub fn memory_info(&self) -> OrtResult<MemoryInfo> {
        let mut memory_info_ptr: *const sys::OrtMemoryInfo = std::ptr::null_mut();
        let status =
            unsafe { ort_api().GetTensorMemoryInfo.unwrap()(self.ptr, &mut memory_info_ptr) };
        status_to_result(status).map_err(OrtError::GetTensorMemoryInfo)?;

        MemoryInfo::try_from(memory_info_ptr)
    }

    /// Provide an array_view over the data contained in the OrtValue (CPU Only)
    pub fn array_view<T>(&self) -> OrtResult<ArrayView<T, Dim<IxDynImpl>>>
    where
        T: TypeToTensorElementDataType,
    {
        let memory_info = self.memory_info()?;
        if !matches!(memory_info.name(), DeviceName::Cpu) {
            return Err(OrtError::GetTensorMutableDataNonMatchingDeviceName(
                NonMatchingDeviceName::DeviceName {
                    tensor: memory_info.name().clone(),
                    requested: DeviceName::Cpu,
                },
            ));
        }

        let type_and_shape_info = self.type_and_shape_info()?;
        if T::tensor_element_data_type() != type_and_shape_info.element_data_type {
            return Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType {
                input: type_and_shape_info.element_data_type.clone(),
                requested: T::tensor_element_data_type(),
            }));
        };

        // return empty array if any dimension is 0
        if type_and_shape_info.dimensions.iter().any(|dim| dim == &0) {
            Ok(ArrayView::from_shape(
                type_and_shape_info
                    .dimensions
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap())
        } else {
            self.array_view_unchecked::<T>(Some(type_and_shape_info))
        }
    }

    /// # Safety
    pub fn array_view_unchecked<T>(
        &self,
        type_and_shape_info: Option<TensorTypeAndShapeInfo>,
    ) -> OrtResult<ArrayView<T, Dim<IxDynImpl>>> {
        let type_and_shape_info = type_and_shape_info
            .map(OrtResult::Ok)
            .unwrap_or_else(|| self.type_and_shape_info())?;

        // Get pointer to output tensor values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr as *mut *mut std::ffi::c_void;
        let status =
            unsafe { ort_api().GetTensorMutableData.unwrap()(self.ptr, output_array_ptr_ptr_void) };
        status_to_result(status).map_err(OrtError::GetTensorMutableData)?;

        Ok(unsafe {
            ArrayView::<T, _>::from_shape_ptr(
                type_and_shape_info
                    .dimensions
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
                output_array_ptr,
            )
        })
    }
}

impl Deref for OrtValue {
    type Target = *mut sys::OrtValue;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl Drop for OrtValue {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("Value pointer is null, not dropping.");
        } else {
            trace!("Dropping Value: {:?}.", self.ptr);
            unsafe { ort_api().ReleaseValue.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}

