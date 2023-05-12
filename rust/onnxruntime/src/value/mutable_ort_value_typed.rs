//! Module for MutableOrtValueTyped
use std::fmt::Debug;

use ndarray::{Array, ArrayViewMut, IxDyn};

use crate::{
    error::{OrtResult},
    AsOrtValue, OrtValue, Session, TensorElementDataType,
    TypeToTensorElementDataType,
};

/// MutableOrtValueTyped<T> holds an OrtValue with element type T that
/// owns its data, plus an ArrayViewMut<T,IxDyn> that allows the data to 
/// be read / written in Rust. Unlike MutableOrtValue, the type of 
/// MutableOrtValueTyped<T> must be statically known.
#[derive(Debug)]
pub struct MutableOrtValueTyped<T>
where
    T: TypeToTensorElementDataType + 'static,
{
    /// An OrtValue that owns its own data memory.
    ort_value: OrtValue,
    /// A read / write view of the data within the OrtValue. 
    /// The view is valid for the life of the OrtValue.
    pub view: ArrayViewMut<'static, T, IxDyn>,
}

impl<T> MutableOrtValueTyped<T> 
where
    T: TypeToTensorElementDataType,
{
    /// Create a MutableOrtValueTyped containing all zeros. Can be updated
    /// later by writing to the view.
    pub fn zeros(session: &Session, shape: &[usize]) -> OrtResult<Self> {
        let ort_value = OrtValue::new_from_type_and_shape::<T>(&session, shape)?;
        let data_ptr = ort_value.get_tensor_mutable_data()?;

        let shape: Vec<usize> = shape.into_iter().map(|dim| *dim as usize).collect();

        // Zero out all memory, which constitutes a 0 in any of the supported base types.
        let elem_n: usize = shape.iter().product();
        unsafe { std::ptr::write_bytes::<T>(data_ptr, 0, elem_n); }

        let view = unsafe {
            ArrayViewMut::<T, _>::from_shape_ptr(shape, data_ptr)
        };

        Ok( Self { ort_value, view } )
    }

    /// Create a MutableOrtValueTyped from an OrtValue. The OrtValue should
    /// own its data memory.
    pub fn try_from(ort_value: OrtValue) -> OrtResult<Self> {
        let type_and_shape_info = ort_value.type_and_shape_info()?;
        let output_array_ptr = ort_value.get_tensor_mutable_data()?;

        let view = unsafe {
            ArrayViewMut::<T, _>::from_shape_ptr(
                type_and_shape_info
                    .dimensions
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
                output_array_ptr,
            )
        };
        Ok( Self { ort_value, view } )
    }

    /// Create a MutableOrtValueTyped from an ndarray::Array. Array's data is copied into
    /// the OrtValue, which owns its own data memory.  
    pub fn try_from_array<D>(session: &Session, array: &Array<T, D>) -> OrtResult<Self>
    where
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

        let view = unsafe {
            ArrayViewMut::<T, _>::from_shape_ptr(array.shape(), data_ptr)
        };

        Ok( Self { ort_value, view } )
    }

    /// Clone the MutableOrtValueTyped. The result contains an OrtValue 
    /// that owns its copy of the data.
    pub fn try_clone(&self, session: &Session) -> OrtResult<Self> {
        let shape = self.view.shape();
        // Create OrtValue that owns its data
        let ort_value = OrtValue::new_from_type_and_shape::<T>(session, shape)?;
        // Create ArrayMutView of OrtValue's data
        let mut view = unsafe { 
            ArrayViewMut::<T, _>::from_shape_ptr(shape, ort_value.get_tensor_mutable_data()?) 
        };
        // Copy data from view
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
            | TensorElementDataType::Uint64 => unsafe { 
                let elem_n: usize = shape.iter().product();
                // Both views should be in standard layout
                debug_assert!(self.view.is_standard_layout());
                debug_assert!(view.is_standard_layout());

                std::ptr::copy_nonoverlapping(self.view.as_ptr(), view.as_mut_ptr(), elem_n); 
            }
            _ => unimplemented!(),
        }

        Ok( Self { ort_value, view } )

    }

}

impl<T> AsOrtValue for MutableOrtValueTyped<T>
where
    T: TypeToTensorElementDataType + Debug,
{
    fn as_ort_value(&self) -> &OrtValue {
        &self.ort_value
    }
}

impl<'a, T> From<&'a MutableOrtValueTyped<T>> for &'a OrtValue
where
    T: TypeToTensorElementDataType + Debug,
{
    fn from(mut_ort_value: &'a MutableOrtValueTyped<T>) -> Self {
        &mut_ort_value.ort_value
    }
}

