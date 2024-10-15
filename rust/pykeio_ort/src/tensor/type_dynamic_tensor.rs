use std::fmt::Debug;

use ndarray::{Array, IxDyn};

use crate::{memory::MemoryInfo, sys, tensor::OrtTensor, OrtResult};

/// Trait used for constructing inputs with multiple element types from [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
pub trait FromArray<T> {
	/// Wrap [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) into enum with specific dtype variants.
	fn from_array(array: Array<T, IxDyn>) -> InputTensor;
}

macro_rules! impl_convert_trait {
	($type_:ty, $variant:expr) => {
		impl FromArray<$type_> for InputTensor {
			fn from_array(array: Array<$type_, IxDyn>) -> InputTensor {
				$variant(array)
			}
		}
	};
}

/// Input tensor enum with tensor element type as a variant.
///
/// Required for supplying inputs with different types
#[derive(Debug)]
#[allow(missing_docs)]
pub enum InputTensor {
	FloatTensor(Array<f32, IxDyn>),
	#[cfg(feature = "half")]
	Float16Tensor(Array<half::f16, IxDyn>),
	#[cfg(feature = "half")]
	Bfloat16Tensor(Array<half::bf16, IxDyn>),
	Uint8Tensor(Array<u8, IxDyn>),
	Int8Tensor(Array<i8, IxDyn>),
	Uint16Tensor(Array<u16, IxDyn>),
	Int16Tensor(Array<i16, IxDyn>),
	Int32Tensor(Array<i32, IxDyn>),
	Int64Tensor(Array<i64, IxDyn>),
	DoubleTensor(Array<f64, IxDyn>),
	Uint32Tensor(Array<u32, IxDyn>),
	Uint64Tensor(Array<u64, IxDyn>),
	StringTensor(Array<String, IxDyn>)
}

/// This tensor is used to copy an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// from InputTensor to the runtime's memory with support to multiple input tensor types.
///
/// **NOTE**: The type is not meant to be used directly, use an InputTensor constructed from
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) instead.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum InputOrtTensor<'t> {
	FloatTensor(OrtTensor<'t, f32, IxDyn>),
	#[cfg(feature = "half")]
	Float16Tensor(OrtTensor<'t, half::f16, IxDyn>),
	#[cfg(feature = "half")]
	Bfloat16Tensor(OrtTensor<'t, half::bf16, IxDyn>),
	Uint8Tensor(OrtTensor<'t, u8, IxDyn>),
	Int8Tensor(OrtTensor<'t, i8, IxDyn>),
	Uint16Tensor(OrtTensor<'t, u16, IxDyn>),
	Int16Tensor(OrtTensor<'t, i16, IxDyn>),
	Int32Tensor(OrtTensor<'t, i32, IxDyn>),
	Int64Tensor(OrtTensor<'t, i64, IxDyn>),
	DoubleTensor(OrtTensor<'t, f64, IxDyn>),
	Uint32Tensor(OrtTensor<'t, u32, IxDyn>),
	Uint64Tensor(OrtTensor<'t, u64, IxDyn>),
	StringTensor(OrtTensor<'t, String, IxDyn>)
}

impl InputTensor {
	/// Get shape of the underlying array.
	pub fn shape(&self) -> &[usize] {
		match self {
			InputTensor::FloatTensor(x) => x.shape(),
			#[cfg(feature = "half")]
			InputTensor::Float16Tensor(x) => x.shape(),
			#[cfg(feature = "half")]
			InputTensor::Bfloat16Tensor(x) => x.shape(),
			InputTensor::Uint8Tensor(x) => x.shape(),
			InputTensor::Int8Tensor(x) => x.shape(),
			InputTensor::Uint16Tensor(x) => x.shape(),
			InputTensor::Int16Tensor(x) => x.shape(),
			InputTensor::Int32Tensor(x) => x.shape(),
			InputTensor::Int64Tensor(x) => x.shape(),
			InputTensor::DoubleTensor(x) => x.shape(),
			InputTensor::Uint32Tensor(x) => x.shape(),
			InputTensor::Uint64Tensor(x) => x.shape(),
			InputTensor::StringTensor(x) => x.shape()
		}
	}
}

impl_convert_trait!(f32, InputTensor::FloatTensor);
#[cfg(feature = "half")]
impl_convert_trait!(half::f16, InputTensor::Float16Tensor);
#[cfg(feature = "half")]
impl_convert_trait!(half::bf16, InputTensor::Bfloat16Tensor);
impl_convert_trait!(u8, InputTensor::Uint8Tensor);
impl_convert_trait!(i8, InputTensor::Int8Tensor);
impl_convert_trait!(u16, InputTensor::Uint16Tensor);
impl_convert_trait!(i16, InputTensor::Int16Tensor);
impl_convert_trait!(i32, InputTensor::Int32Tensor);
impl_convert_trait!(i64, InputTensor::Int64Tensor);
impl_convert_trait!(f64, InputTensor::DoubleTensor);
impl_convert_trait!(u32, InputTensor::Uint32Tensor);
impl_convert_trait!(u64, InputTensor::Uint64Tensor);
impl_convert_trait!(String, InputTensor::StringTensor);

impl<'t> InputOrtTensor<'t> {
	pub(crate) fn from_input_tensor<'m, 'i>(
		memory_info: &'m MemoryInfo,
		allocator_ptr: *mut sys::OrtAllocator,
		input_tensor: &'i InputTensor
	) -> OrtResult<InputOrtTensor<'t>>
	where
		'm: 't
	{
		match input_tensor {
			InputTensor::FloatTensor(array) => Ok(InputOrtTensor::FloatTensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			#[cfg(feature = "half")]
			InputTensor::Float16Tensor(array) => Ok(InputOrtTensor::Float16Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			#[cfg(feature = "half")]
			InputTensor::Bfloat16Tensor(array) => Ok(InputOrtTensor::Bfloat16Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Uint8Tensor(array) => Ok(InputOrtTensor::Uint8Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Int8Tensor(array) => Ok(InputOrtTensor::Int8Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Uint16Tensor(array) => Ok(InputOrtTensor::Uint16Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Int16Tensor(array) => Ok(InputOrtTensor::Int16Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Int32Tensor(array) => Ok(InputOrtTensor::Int32Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Int64Tensor(array) => Ok(InputOrtTensor::Int64Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::DoubleTensor(array) => Ok(InputOrtTensor::DoubleTensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Uint32Tensor(array) => Ok(InputOrtTensor::Uint32Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::Uint64Tensor(array) => Ok(InputOrtTensor::Uint64Tensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?)),
			InputTensor::StringTensor(array) => Ok(InputOrtTensor::StringTensor(OrtTensor::from_array(memory_info, allocator_ptr, array)?))
		}
	}

	pub(crate) fn c_ptr(&self) -> *const sys::OrtValue {
		match self {
			InputOrtTensor::FloatTensor(x) => x.c_ptr,
			#[cfg(feature = "half")]
			InputOrtTensor::Float16Tensor(x) => x.c_ptr,
			#[cfg(feature = "half")]
			InputOrtTensor::Bfloat16Tensor(x) => x.c_ptr,
			InputOrtTensor::Uint8Tensor(x) => x.c_ptr,
			InputOrtTensor::Int8Tensor(x) => x.c_ptr,
			InputOrtTensor::Uint16Tensor(x) => x.c_ptr,
			InputOrtTensor::Int16Tensor(x) => x.c_ptr,
			InputOrtTensor::Int32Tensor(x) => x.c_ptr,
			InputOrtTensor::Int64Tensor(x) => x.c_ptr,
			InputOrtTensor::DoubleTensor(x) => x.c_ptr,
			InputOrtTensor::Uint32Tensor(x) => x.c_ptr,
			InputOrtTensor::Uint64Tensor(x) => x.c_ptr,
			InputOrtTensor::StringTensor(x) => x.c_ptr
		}
	}
}
