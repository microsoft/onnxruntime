//! Module for MutableOrtValue
use ndarray::{Array, ArrayViewMut, IxDyn};

use crate::{
    error::{NonMatchingDataTypes, OrtResult, OrtError},
    AsOrtValue, OrtValue, Session, TensorElementDataType,
    TypeToTensorElementDataType,
};

/// MutableOrtValue's internal element-type-multiplexed
/// (T) storage of the ArrayViewMut<T, IxDyn>
#[derive(Debug)]
pub enum TypedArrayViewMut
{
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float(ArrayViewMut<'static, f32, IxDyn>),
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double(ArrayViewMut<'static, f64, IxDyn>),
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8(ArrayViewMut<'static, u8, IxDyn>),
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8(ArrayViewMut<'static, i8, IxDyn>),
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16(ArrayViewMut<'static, u16, IxDyn>),
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16(ArrayViewMut<'static, i16, IxDyn>),
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32(ArrayViewMut<'static, u32, IxDyn>),
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32(ArrayViewMut<'static, i32, IxDyn>),
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64(ArrayViewMut<'static, u64, IxDyn>),
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64(ArrayViewMut<'static, i64, IxDyn>),
    /// String, equivalent to Rust's `String`
    String(ArrayViewMut<'static, String, IxDyn>),
    /// Boolean, equivalent to Rust's `bool`
    Bool(ArrayViewMut<'static, bool, IxDyn>),
    // 16-bit floating point, equivalent to Rust's `f16`
    //Float16(ArrayViewMut<'static, f32, IxDyn>),
}

impl TypedArrayViewMut {
    /// Convert TypedArrayViewMut element type to TensorElementDataType
    fn tensor_element_data_type(&self) -> TensorElementDataType {
        match self {
            TypedArrayViewMut::Float(_) => TensorElementDataType::Float,
            TypedArrayViewMut::Double(_) => TensorElementDataType::Double,
            TypedArrayViewMut::Uint8(_) => TensorElementDataType::Uint8,
            TypedArrayViewMut::Int8(_) => TensorElementDataType::Int8,
            TypedArrayViewMut::Uint16(_) => TensorElementDataType::Uint16,
            TypedArrayViewMut::Int16(_) => TensorElementDataType::Int16,
            TypedArrayViewMut::Uint32(_) => TensorElementDataType::Uint32,
            TypedArrayViewMut::Int32(_) => TensorElementDataType::Int32,
            TypedArrayViewMut::Uint64(_) => TensorElementDataType::Uint64,
            TypedArrayViewMut::Int64(_) => TensorElementDataType::Int64,
            TypedArrayViewMut::String(_) => TensorElementDataType::String,
            TypedArrayViewMut::Bool(_) => TensorElementDataType::Bool,
        }
    }
}

/// Trait for base-type-specific (f32, f64, etc.) helper functions
/// to and from TypedArrayViewMut. Used internally by MutableOrtValue.
pub trait TypedArrayViewMutConversions<T> : Clone
where
    T: TypeToTensorElementDataType + 'static,
{
    /// Borrow &ArrayViewMut from TypedArrayViewMut based on static type T.
    fn to_view(typed_view: &TypedArrayViewMut) -> OrtResult<&ArrayViewMut<'static, T, IxDyn>>;
    /// Borrow &mut ArrayViewMut from TypedArrayViewMut based on static type T.
    fn to_view_mut(typed_view: &mut TypedArrayViewMut) -> OrtResult<&mut ArrayViewMut<'static, T, IxDyn>>;
    /// Create TypedArrayViewMut from ArrayViewMut.
    fn from_view(view: ArrayViewMut<'static, T, IxDyn>) -> TypedArrayViewMut;
    /// Create TypedArrayViewMut from OrtValue's shape and pointer to data.
    fn from_ort_value(shape: &[usize], data_ptr: *mut T) -> TypedArrayViewMut;
}

macro_rules! impl_typed_view_trait {
    ($type_:ty, $variant:ident) => {
        impl TypedArrayViewMutConversions<$type_> for $type_ {
            fn to_view(typed_view: &TypedArrayViewMut) -> OrtResult<&ArrayViewMut<'static, $type_, IxDyn>> {
                if let TypedArrayViewMut::$variant (view) = typed_view {
                    Ok(view)
                } else {
                    Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType{
                        input: typed_view.tensor_element_data_type(), 
                        requested: <$type_>::tensor_element_data_type(), 
                    }))
                }
            }        
            fn to_view_mut(typed_view: &mut TypedArrayViewMut) -> OrtResult<&mut ArrayViewMut<'static, $type_, IxDyn>> {
                if let TypedArrayViewMut::$variant (view) = typed_view {
                    Ok(view)
                } else {
                    Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType{
                        input: typed_view.tensor_element_data_type(), 
                        requested: <$type_>::tensor_element_data_type(), 
                    }))
                }
            }        
            fn from_view(view: ArrayViewMut<'static, $type_, IxDyn>) -> TypedArrayViewMut {
                TypedArrayViewMut::$variant (view)
            }
            fn from_ort_value(shape: &[usize], data_ptr: *mut $type_) -> TypedArrayViewMut {
                TypedArrayViewMut::$variant(
                    unsafe { ArrayViewMut::<$type_, _>::from_shape_ptr(shape, data_ptr) }
                ) 
            }
        }
    };
}

impl_typed_view_trait!(f32, Float);
impl_typed_view_trait!(f64, Double);
impl_typed_view_trait!(i8, Int8);
impl_typed_view_trait!(u8, Uint8);
impl_typed_view_trait!(i16, Int16);
impl_typed_view_trait!(u16, Uint16);
impl_typed_view_trait!(i32, Int32);
impl_typed_view_trait!(u32, Uint32);
impl_typed_view_trait!(i64, Int64);
impl_typed_view_trait!(u64, Uint64);

/// MutableOrtValue holds an OrtValue that owns its data, plus
/// an ArrayViewMut that allows the data to be read / written in 
/// Rust. Unlike MutableOrtValueTyped<T>, MutableOrtValue hides the 
/// element type behind an enumeration. 
/// 
/// This provides flexibility when working with tensors of multiple 
/// types, such as being able to store them in the same container.
/// Also, type-agnotic code can be written, such as when propagating 
/// autoregressive model outputs to inputs between calls to run().
/// 
/// Compared to MutableOrtValueTyped<T>, some additional functionality
/// is provided to support type-agnostic code, such as assign().
#[derive(Debug)]
pub struct MutableOrtValue
{
    /// An OrtValue that owns its own data memory.
    ort_value: OrtValue,
    /// A read / write view of the data within the OrtValue, encapsulated
    /// by type within a TypedArrayViewMut enum.
    /// The view is valid for the life of the OrtValue.
    typed_view: TypedArrayViewMut,
}

impl MutableOrtValue {
    /// Return &ArrayViewMut, or Err if type of ArrayViewMut has the wrong element type.
    pub fn view<T>(&self) -> OrtResult<&ArrayViewMut<'static, T, IxDyn>>
    where
        T: TypeToTensorElementDataType + TypedArrayViewMutConversions<T> + 'static,
    {
        T::to_view(&self.typed_view)
    }

    /// Return &mut ArrayViewMut, or Err if type of ArrayViewMut has the wrong element type.
    pub fn view_mut<T>(&mut self) -> OrtResult<&mut ArrayViewMut<'static, T, IxDyn>>
    where
        T: TypeToTensorElementDataType + TypedArrayViewMutConversions<T> + 'static,
    {
        T::to_view_mut(&mut self.typed_view)
    }

    /// Create MutableOrtValue with the statically-specified type, containing all zeros.
    /// Can be updated later by writing to the view.
    pub fn zeros<T>(session: &Session, shape: &[usize]) -> OrtResult<Self> 
    where
        T: TypeToTensorElementDataType + TypedArrayViewMutConversions<T> + 'static,
    {
        let ort_value = OrtValue::new_from_type_and_shape::<T>(&session, shape)?;
        let data_ptr = ort_value.get_tensor_mutable_data()?;

        let shape: Vec<usize> = shape.into_iter().map(|dim| *dim as usize).collect();

        // Assign 0s to every element
        let elem_n: usize = shape.iter().product();
        unsafe { std::ptr::write_bytes::<T>(data_ptr, 0, elem_n); }

        let typed_view = T::from_view(
            unsafe { ArrayViewMut::<T, _>::from_shape_ptr(shape, data_ptr) }
        );

        Ok( Self { ort_value, typed_view } )
    }

    /// Create MutableOrtValue with the dynamically-specified type, containing all zeros.
    /// Can be updated later by writing to the view.
    pub fn zeroes_dyn_type(session: &Session, element_type: TensorElementDataType, shape: &[usize]) -> OrtResult<Self> {
        match element_type {
            TensorElementDataType::Float => Self::zeros::<f32>(session, shape),
            TensorElementDataType::Double => Self::zeros::<f64>(session, shape),
            TensorElementDataType::Int8 => Self::zeros::<i8>(session, shape),
            TensorElementDataType::Uint8 => Self::zeros::<u8>(session, shape),
            TensorElementDataType::Int16 => Self::zeros::<i16>(session, shape),
            TensorElementDataType::Uint16 => Self::zeros::<u32>(session, shape),
            TensorElementDataType::Int32 => Self::zeros::<i32>(session, shape),
            TensorElementDataType::Uint32 => Self::zeros::<u32>(session, shape),
            TensorElementDataType::Int64 => Self::zeros::<i64>(session, shape),
            TensorElementDataType::Uint64 => Self::zeros::<u64>(session, shape),
            _ => unimplemented!(),
        }
    }

    /// Create a MutableOrtValue from an OrtValue. The OrtValue should
    /// own its data memory.
    pub fn try_from(ort_value: OrtValue) -> OrtResult<Self> {
        let type_and_shape_info = ort_value.type_and_shape_info()?;
        let shape = type_and_shape_info.get_dimensions_as_usize();

        let typed_view =
        match type_and_shape_info.element_data_type {
            TensorElementDataType::Float => f32::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Double => f64::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Int8 => i8::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Uint8 => u8::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Int16 => i16::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Uint16 => u16::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Int32 => i32::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Uint32 => u32::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Int64 => i64::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            TensorElementDataType::Uint64 => u64::from_ort_value(&shape, ort_value.get_tensor_mutable_data()?),
            _ => unimplemented!(),
        };
        Ok( Self { ort_value, typed_view })
    }

    /// Create an MutableOrtValue from an ndarray::Array. Array's data is copied into
    /// the OrtValue, which owns its own data memory. 
    pub fn try_from_array<T, D>(session: &Session, array: &Array<T, D>) -> OrtResult<Self>
    where
        T: TypeToTensorElementDataType + TypedArrayViewMutConversions<T> + 'static,
        D: ndarray::Dimension,
    {
        let ort_value = OrtValue::new_from_type_and_shape::<T>(&session, array.shape())?;
        let data_ptr = ort_value.get_tensor_mutable_data()?;

        match T::tensor_element_data_type() {
            TensorElementDataType::Float
            | TensorElementDataType::Double
            | TensorElementDataType::Int8
            | TensorElementDataType::Uint8
            | TensorElementDataType::Int16
            | TensorElementDataType::Uint16
            | TensorElementDataType::Int32
            | TensorElementDataType::Uint32
            | TensorElementDataType::Int64
            | TensorElementDataType::Uint64 => {
                // Array must be in standard layout to be (easily) copied into OrtValue
                assert!(array.is_standard_layout());
                // Copy elements from array to ort_value's data memory
                let elem_n: usize = array.shape().iter().product();
                unsafe { std::ptr::copy_nonoverlapping(array.as_ptr(), data_ptr, elem_n); }
            },
            _ => unimplemented!(),
        }

        let typed_view = unsafe {
            T::from_view(ArrayViewMut::<T, _>::from_shape_ptr(array.shape(), data_ptr))
        };

        Ok( Self { ort_value, typed_view } )
    }

    /// Copy data in from_value to this MutableOrtValue. Tensor types and sizes must be equal.
    /// Internally, this performs self.array_view_mut.assign(from_value.array_view_mut)
    pub fn assign(&mut self, from_value: &MutableOrtValue) -> OrtResult<()> {
        let from_type = from_value.typed_view.tensor_element_data_type();
        let to_type = self.typed_view.tensor_element_data_type();
        if from_type != to_type {
            Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType{ input: to_type, requested: from_type }))
        } else {
            match &mut self.typed_view {
                TypedArrayViewMut::Float(view) => {
                    if let TypedArrayViewMut::Float(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Double(view) => {
                    if let TypedArrayViewMut::Double(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Int8(view) => {
                    if let TypedArrayViewMut::Int8(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Uint8(view) => {
                    if let TypedArrayViewMut::Uint8(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Int16(view) => {
                    if let TypedArrayViewMut::Int16(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Uint16(view) => {
                    if let TypedArrayViewMut::Uint16(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Int32(view) => {
                    if let TypedArrayViewMut::Int32(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Uint32(view) => {
                    if let TypedArrayViewMut::Uint32(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Int64(view) => {
                    if let TypedArrayViewMut::Int64(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                TypedArrayViewMut::Uint64(view) => {
                    if let TypedArrayViewMut::Uint64(from_view) = &from_value.typed_view {
                        view.assign(from_view);
                    } 
                },
                _ => unimplemented!(),
            }
            Ok(())
        }
    }

    /// Clone the MutableOrtValue of a known underlying element type T. 
    /// The resulting MutableOrtValue contains an OrtValue that owns its 
    /// copy of the data.
    fn try_clone_typed<T>(session: &Session, view: &ArrayViewMut<'static, T, IxDyn>) -> OrtResult<MutableOrtValue>
    where
        T: TypeToTensorElementDataType + TypedArrayViewMutConversions<T> + 'static,
    {
        let shape = view.shape();
        // Create OrtValue that owns its data
        let ort_value = OrtValue::new_from_type_and_shape::<T>(session, shape)?;
        // Create ArrayMutView of OrtValue's data
        let mut new_view = unsafe { 
            ArrayViewMut::<T, _>::from_shape_ptr(shape, ort_value.get_tensor_mutable_data()?) 
        };
        // Copy data from view
        new_view.assign(view);
        
        Ok( Self { ort_value, typed_view: T::from_view(new_view) } )
    }

    /// Clone the MutableOrtValue. The resulting MutableOrtValue contains
    /// an OrtValue that owns its copy of the data.
    pub fn try_clone(&self, session: &Session) -> OrtResult<MutableOrtValue> {
        match &self.typed_view {
            TypedArrayViewMut::Float(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Double(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Uint8(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Int8(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Uint16(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Int16(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Uint32(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Int32(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Uint64(view) => Self::try_clone_typed(session, &view),
            TypedArrayViewMut::Int64(view) => Self::try_clone_typed(session, &view),
            _ => unimplemented!(),
        }
    }
}

impl AsOrtValue for MutableOrtValue
{
    fn as_ort_value(&self) -> &OrtValue {
        &self.ort_value
    }
}

impl<'a> From<&'a MutableOrtValue> for &'a OrtValue
{
    fn from(mut_ort_value: &'a MutableOrtValue) -> Self {
        &mut_ort_value.ort_value
    }
}

