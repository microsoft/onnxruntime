//! Module abstracting OrtValue.

use std::{fmt::Debug, convert::TryFrom};
use std::ops::Deref;

use ndarray::{Array, ArrayView, Dim, IxDynImpl, ArrayViewMut};
use tracing::{error, trace};

use onnxruntime_sys as sys;

use crate::{
    ort_api,
    error::{status_to_result, NonMatchingDataTypes, NonMatchingDeviceName, OrtResult, OrtError},
    DeviceName, MemoryInfo, Session, TensorElementDataType,
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
    /// Create an OrtValue struct from the bare pointer.
    pub(crate) fn new(ptr: *mut sys::OrtValue) -> Self {
        trace!("Created Value {ptr:?}.");
        Self { ptr }
    }

    /// Create an OrtValue with a specified element type and shape that owns its data.
    /// The contents should be initialized immediately after this function returns.
    fn new_from_type_and_shape<T>(session: &Session, shape: &[usize]) 
    -> OrtResult<OrtValue> 
    where
        T: TypeToTensorElementDataType,
    {
        let elem_type = T::tensor_element_data_type();

        let shape_i64 = shape.iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<i64>>();
        let shape_ptr = shape_i64.as_ptr();
        let shape_len = shape_i64.len();
    
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
        Ok( OrtValue::new(tensor_ptr) )
    }

    /// Return raw pointer to tensor data
    fn get_tensor_mutable_data<T>(&self) -> OrtResult<*mut T> {
        // Pointer to allocated memory for data
        let mut data_ptr: *mut T = std::ptr::null_mut();
        // This is the address of data_ptr
        let data_ptr_ptr: *mut *mut T = &mut data_ptr;
        let data_void_ptr_ptr: *mut *mut std::ffi::c_void = data_ptr_ptr as *mut *mut std::ffi::c_void;

        let status =
            unsafe { ort_api().GetTensorMutableData.unwrap()(self.ptr, data_void_ptr_ptr) };
        status_to_result(status).map_err(OrtError::GetTensorMutableData)?;
        Ok( data_ptr )
    }
    
    /// Create an OrtValue using data memory owned by array. Unsafe: the lifetime of this
    /// OrtValue must be guaranteed to outlive the lifetime of array.
    fn try_from_array<T, D>(session: &Session, array: &Array<T, D>) -> OrtResult<Self>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        // where onnxruntime will write the tensor data to
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();

        // shapes
        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr = shape.as_ptr();
        let shape_len = array.shape().len();

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
                // primitive data is already suitably laid out in memory; provide it to
                // onnxruntime as is
                let tensor_values_ptr = array.as_ptr() as *mut std::ffi::c_void;

                let status = unsafe {
                    ort_api().CreateTensorWithDataAsOrtValue.unwrap()(
                        session.memory_info.ptr,
                        tensor_values_ptr,
                        array.len() * std::mem::size_of::<T>(),
                        shape_ptr,
                        shape_len,
                        T::tensor_element_data_type().into(),
                        &mut tensor_ptr,
                    )
                };
                status_to_result(status).map_err(OrtError::IsTensor)?;
            }
            _ => todo!(),
        }

        Ok(Self::new(tensor_ptr))
    }

    /// Create an OrtValue that owns its data memory, copying the data values
    /// from array.
    pub fn copy_from_array<T, D>(session: &Session, array: &Array<T, D>) -> OrtResult<Self>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        let ort_value = OrtValue::new_from_type_and_shape::<T>(&session, array.shape())?;
        let data_ptr = ort_value.get_tensor_mutable_data()?;

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
                // Array must be in standard layout to be (easily) copied into OrtValue
                assert!(array.is_standard_layout());
                // Copy elements from array to ort_value's data memory
                let elem_n: usize = array.shape().iter().product();
                unsafe { std::ptr::copy_nonoverlapping(array.as_ptr(), data_ptr, elem_n); }
            },
            _ => unimplemented!(),
        }

        Ok( ort_value )
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

    /// ArrayView only possible if tensor in CPU memory.
    fn array_view_mem_is_cpu(memory_info: &MemoryInfo) -> OrtResult<()> {
        if !matches!(memory_info.name(), DeviceName::Cpu) {
            return Err(OrtError::GetTensorMutableDataNonMatchingDeviceName(
                NonMatchingDeviceName::DeviceName {
                    tensor: memory_info.name().clone(),
                    requested: DeviceName::Cpu,
                },
            ));
        }

        Ok(())
    }

    /// ArrayView only possible if tensor type matches requested type T
    fn array_view_element_type_check<T>(element_data_type: TensorElementDataType) -> OrtResult<()>
    where
        T: TypeToTensorElementDataType,
    {
        if T::tensor_element_data_type() != element_data_type {
            return Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType {
                input: element_data_type,
                requested: T::tensor_element_data_type(),
            }));
        };
        
        Ok(())
    }

    /// Provide an ArrayView over the data contained in the OrtValue (CPU Only)
    pub fn array_view<T>(&self) -> OrtResult<ArrayView<T, Dim<IxDynImpl>>>
    where
        T: TypeToTensorElementDataType,
    {
        let type_and_shape_info = self.type_and_shape_info()?;

        Self::array_view_mem_is_cpu(&self.memory_info()?)?;
        Self::array_view_element_type_check::<T>(type_and_shape_info.element_data_type)?;

        // return empty array if any dimension is 0
        if type_and_shape_info.dimensions.iter().any(|dim| dim == &0) {
            Ok(ArrayView::from_shape(
                type_and_shape_info.get_dimensions_as_usize(),
                &[],
            )
            .unwrap())
        } else {
            self.array_view_unchecked::<T>(Some(type_and_shape_info))
        }
    }

    /// Provide an ArrayViewMut over the data contained in the OrtValue (CPU Only)
    pub fn array_view_mut<T>(&self) -> OrtResult<ArrayViewMut<T, Dim<IxDynImpl>>>
    where
        T: TypeToTensorElementDataType,
    {
        let type_and_shape_info = self.type_and_shape_info()?;

        Self::array_view_mem_is_cpu(&self.memory_info()?)?;
        Self::array_view_element_type_check::<T>(type_and_shape_info.element_data_type)?;

        // return empty array if any dimension is 0
        if type_and_shape_info.dimensions.iter().any(|dim| dim == &0) {
            Ok(ArrayViewMut::from_shape(
                type_and_shape_info.get_dimensions_as_usize(),
                &mut [],
            )
            .unwrap())
        } else {
            self.array_view_mut_unchecked::<T>(Some(type_and_shape_info))
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

        let output_array_ptr = self.get_tensor_mutable_data()?;

        Ok(unsafe {
            ArrayView::<T, _>::from_shape_ptr(
                type_and_shape_info.get_dimensions_as_usize(),
                output_array_ptr,
            )
        })
    }

    /// # Safety
    pub fn array_view_mut_unchecked<T>(
        &self,
        type_and_shape_info: Option<TensorTypeAndShapeInfo>,
    ) -> OrtResult<ArrayViewMut<T, Dim<IxDynImpl>>> {
        let type_and_shape_info = type_and_shape_info
            .map(OrtResult::Ok)
            .unwrap_or_else(|| self.type_and_shape_info())?;

        let output_array_ptr = self.get_tensor_mutable_data()?;

        Ok(unsafe {
            ArrayViewMut::<T, _>::from_shape_ptr(
                type_and_shape_info.get_dimensions_as_usize(),
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

/// Trait for tensors that maintain a respresentation as an OrtValue, 
/// such as MutableOrtValue and NdArrayOrtValue.
pub trait AsOrtValue : Debug {
    /// Return OrtValue representation
    fn as_ort_value(&self) -> &OrtValue;
}

pub mod mutable_ort_value;
pub use mutable_ort_value::MutableOrtValue;

pub mod mutable_ort_value_typed;
pub use mutable_ort_value_typed::MutableOrtValueTyped;

pub mod ndarray_ort_value;
pub use ndarray_ort_value::NdArrayOrtValue;

