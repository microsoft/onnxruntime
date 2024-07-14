// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_utils.h"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace vulkan {
namespace {
template <typename TMat>
void InitMatFromTensor(const Tensor& tensor, TMat& mat, bool align_channels = false) {
  const auto& shape = tensor.Shape();
  const size_t rank = shape.NumDimensions();
  const int64_t num_elements = shape.Size();
  const size_t element_size = tensor.DataType()->Size();

  ORT_ENFORCE(rank > 0, "TODO: do scalars need to be 1D?");

  mat.elemsize = element_size;
  mat.elempack = 1;

  // NCNN doesn't support batches so the 4D dims are C, D, H, W where 'D' is depth.
  // For now map the same way (3D -> C, 1, H, W).
  mat.dims = gsl::narrow_cast<int32_t>(rank);
  mat.w = gsl::narrow_cast<int32_t>(shape[rank - 1]);
  mat.h = 1;
  mat.d = 1;
  mat.c = 1;
  mat.cstep = num_elements;

  switch (rank) {
    case 1:
      break;
    case 2:
      mat.h = gsl::narrow_cast<int32_t>(shape[0]);
      break;
    case 3:
      mat.c = gsl::narrow_cast<int32_t>(shape[0]);
      mat.h = gsl::narrow_cast<int32_t>(shape[1]);
      break;
    case 4:
      // Assume NCHW.
      // NCNN doesn't support batches so if the first dim is 1 assume it's the batch size
      if (mat.c == 1) {
        mat.c = gsl::narrow_cast<int32_t>(shape[1]);
      } else {
        ORT_THROW("Unsupported?");  // TODO: is there a scenario with 4D input where the first dim is not the batch?
        // mat.c = gsl::narrow_cast<int32_t>(shape[0]);
        // mat.d = gsl::narrow_cast<int32_t>(shape[1]);
      }
      mat.h = gsl::narrow_cast<int32_t>(shape[2]);
      break;
    default:
      ORT_THROW("Tensor shape is not supported in Vulkan EP. Must be 4D or less. shape:", shape);
  }

  // align channels data the same way NCNN does. this only applies if we're creating a VkMat for a value that was
  // uploaded using NCNN code that aligns the data.
  if (align_channels && rank > 2) {
    mat.cstep = ncnn::alignSize(SafeInt<size_t>(mat.w) * mat.h * mat.d * element_size, 16) / element_size;
  }

  auto bytes_required = mat.cstep * mat.c;

  // NCNN uses a few bytes past the end of the allocation for the VkMat reference counter.
  // we're not directly using the reference counter (we set it to nullptr) but it may happen if there are internal
  // allocations  made by NCNN (e.g. the Convolution kernel uses a Padding Layer internally). we don't control those.
  //
  // GPU memory allocated with the Vulkan EP allocator adds NCNN_MALLOC_OVERREAD to match NCNN so _should_ be safe.
  //
  // Not sure of a good way to check/ensure that is always the case.
  // Putting this here for now to see if it's hit. If it is we need to double check how much additional buffer our
  // allocations need.
  ORT_ENFORCE(bytes_required <= tensor.CalculateTensorStorageSize(tensor.DataType(), tensor.Shape()),
              "Need extra buffer in allocation for NCNN");
}
}  // namespace

const VulkanExecutionProvider& GetVulkanExecutionProvider(const onnxruntime::OpKernelInfo& info) {
  return *static_cast<const VulkanExecutionProvider*>(info.GetExecutionProvider());
}

// Get the index of the layer in the ncnn model. Throws if not found.
int GetNcnnLayerIndex(const std::string& layer_name) {
  int index = ncnn::layer_to_index(layer_name.c_str());
  if (index == -1) {
    // should never happen outside of during development
    ORT_THROW("Failed to find ", layer_name, " in the NCNN kernels.");
  }

  return index;
}

// get ncnn::Mat shape hints from the node
// TODO: figure out how to handle the inputs/outputs not being 1:1 between the NCNN Layer and the ONNX Node
//       and/or when there are missing optional inputs
std::tuple<std::vector<ncnn::Mat>, std::vector<ncnn::Mat>> GetLayerShapeHints(const Node& node) {
  const auto defs_to_hints = [](const ConstPointerContainer<std::vector<NodeArg*>>& defs) {
    std::vector<ncnn::Mat> shapes;
    shapes.reserve(defs.size());

    for (const auto* def : defs) {
      ncnn::Mat ncnn_shape;
      if (def->Exists()) {
        auto* tensorproto_shape = def->Shape();
        if (tensorproto_shape) {
          TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*tensorproto_shape);
          ncnn_shape.dims = gsl::narrow_cast<int32_t>(shape.NumDimensions());
          ncnn_shape.h = 1;
          ncnn_shape.d = 1;
          ncnn_shape.c = 1;
          ncnn_shape.elempack = 1;
          // The shader bases the datatype on whether fp16 is being used or not, which is set in ncnn::Option.
          // Assuming we don't need to set elemsize here for now.
          // ncnn_shape.elemsize = ???;

          switch (ncnn_shape.dims) {
            case 1:
              ncnn_shape.w = gsl::narrow_cast<int32_t>(shape[0]);
              break;
            case 2:
              ncnn_shape.h = gsl::narrow_cast<int32_t>(shape[0]);
              ncnn_shape.w = gsl::narrow_cast<int32_t>(shape[1]);
              break;
            case 3:
              ncnn_shape.c = gsl::narrow_cast<int32_t>(shape[0]);
              ncnn_shape.h = gsl::narrow_cast<int32_t>(shape[1]);
              ncnn_shape.w = gsl::narrow_cast<int32_t>(shape[2]);

              break;
            case 4:
              // 4D is OK if batch is 1
              if (shape[0] == 1) {
                ncnn_shape.c = gsl::narrow_cast<int32_t>(shape[1]);
                ncnn_shape.h = gsl::narrow_cast<int32_t>(shape[2]);
                ncnn_shape.w = gsl::narrow_cast<int32_t>(shape[3]);
              }

              [[fallthrough]];

            default:
              // as NCNN doesn't expect batches
              ORT_THROW("Unsupported shape:", shape);
          }

          ncnn_shape.cstep = ncnn_shape.w * ncnn_shape.h * ncnn_shape.d;
        }
      }

      shapes.emplace_back(std::move(ncnn_shape));
    }

    return shapes;
  };

  auto input_shapes = defs_to_hints(node.InputDefs());
  auto output_shapes = defs_to_hints(node.OutputDefs());

  return {input_shapes, output_shapes};
}

ncnn::Mat TensorToMat(const Tensor& tensor) {
  ncnn::Mat mat;

  InitMatFromTensor(tensor, mat);
  // we need to set the `data` member which is non-const, so the ugly const_cast is necessary if we're reading,
  // and there's no real value having a `ncnn::Mat TensorToMat(Tensor& tensor)` overload to avoid the const_cast
  // and all MutableDataRaw when writing.
  mat.data = const_cast<void*>(tensor.DataRaw());

  return mat;
}

ncnn::VkMat TensorToVkMat(const Tensor& tensor, ncnn::VkAllocator& allocator) {
  ncnn::VkMat vkmat;

  InitMatFromTensor(tensor, vkmat);
  vkmat.allocator = &allocator;
  // we need to set the `data` member which is non-const, so the ugly const_cast is necessary if we're reading,
  // and there's no real value having a `ncnn::VkMat TensorToVkMat(Tensor& tensor)` overload to avoid the const_cast
  // when writing.
  vkmat.data = static_cast<ncnn::VkBufferMemory*>(const_cast<void*>(tensor.DataRaw()));

  return vkmat;
}

ncnn::VkMat TensorToVkMatWithPacking(const Tensor& tensor, ncnn::VkAllocator& allocator,
                                     const ncnn::VulkanDevice& device,
                                     const ncnn::Option& options) {
  ncnn::VkMat vkmat;

  // get initial values with elempack = 1
  InitMatFromTensor(tensor, vkmat, /* align */ true);

  vkmat.allocator = &allocator;
  // we need to set the `data` member which is non-const, so the ugly const_cast is necessary if we're reading
  // and there's no real value having a `ncnn::VkMat TensorToVkMat(Tensor& tensor)` overload to avoid the const_cast
  // when writing.
  vkmat.data = static_cast<ncnn::VkBufferMemory*>(const_cast<void*>(tensor.DataRaw()));

  // now adjust based on VkCompute::record_upload if elempack differs
  int elempack = vkmat.elempack;

  int w = vkmat.w;
  int h = vkmat.h;
  int c = vkmat.c;

  // VkCompute::record_upload
  int dims = vkmat.dims;
  int elemcount = 0;
  if (dims == 1) elemcount = elempack * w;
  if (dims == 2) elemcount = elempack * h;
  if (dims == 3 || dims == 4) elemcount = elempack * c;

  int dst_elempack = 1;
  if (options.use_shader_pack8)
    dst_elempack = elemcount % 8 == 0 ? 8 : (elemcount % 4 == 0 ? 4 : 1);
  else
    dst_elempack = elemcount % 4 == 0 ? 4 : 1;

  if (dst_elempack == 1) {
    return vkmat;  // same as source so nothing else to do to the VkMat values in our usage
  }

  // vkdev->convert_packing(dst_staging, dst, dst_elempack, *this, opt);
  // ->
  // VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& _opt)

  // 0:auto 1:fp32 2:fp16p 3:fp16s

  int cast_type_to_index = options.use_fp16_storage ? 2 : options.use_fp16_packed ? 1
                                                                                  : 0;
  // 0:pack1 1:pack4 2:pack8. not needed - we just use dst_elempack.
  // int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1
  //                                                                      : 2;

  // see if we're going to convert types when packing. may be relevant if we enable auto conversion from fp32 to fp16
  int cast_type_from_index;
  if (vkmat.elembits() == 32) {
    cast_type_from_index = 0;
  } else {  // if (src.elembits() == 16)
    if (cast_type_to_index != 0) {
      cast_type_from_index = cast_type_to_index;
    } else if (device.info.support_fp16_storage()) {
      cast_type_from_index = 2;
    } else {  // if (info.support_fp16_packed())
      cast_type_from_index = 1;
    }
  }

  // const ncnn::Packing_vulkan* uop = d->get_utility_operator(0, 0, cast_type_from_index, cast_type_to_index, packing_type_to_index);
  // uop->forward(src, dst, cmd, opt);
  // ->
  // int Packing_vulkan::forward(const VkMat& vkmat, VkMat& top_blob, VkCompute& cmd, const Option& opt) const

  do {
    // use_padding is always false on this path
    // if (!use_padding) {
    // identity if use_padding not allowed
    if (dims == 1 && w * elempack % dst_elempack != 0) {
      break;
    }
    if (dims == 2 && h * elempack % dst_elempack != 0) {
      break;
    }
    if ((dims == 3 || dims == 4) && c * elempack % dst_elempack != 0) {
      break;
    }
    // }

    size_t out_elemsize = 0;  // always set below as dst_elempack can only be 1, 4 or 8
    if (cast_type_to_index == 0) {
      if (options.use_fp16_storage) {
        out_elemsize = dst_elempack * 2u;
      } else if (options.use_fp16_packed) {
        if (dst_elempack == 8) out_elemsize = 8 * 2u;
        if (dst_elempack == 4) out_elemsize = 4 * 2u;
        if (dst_elempack == 1) out_elemsize = 4u;
      } else {
        out_elemsize = dst_elempack * 4u;
      }
    } else if (cast_type_to_index == 1) {
      out_elemsize = dst_elempack * 4u;
    } else if (cast_type_to_index == 2) {
      if (dst_elempack == 8) out_elemsize = 8 * 2u;
      if (dst_elempack == 4) out_elemsize = 4 * 2u;
      if (dst_elempack == 1) out_elemsize = 4u;
    } else {  // if (cast_type_to == 3)
      out_elemsize = dst_elempack * 2u;
    }

    assert(vkmat.cstep % dst_elempack == 0);

    vkmat.elemsize = out_elemsize;
    vkmat.elempack = dst_elempack;
    vkmat.cstep /= dst_elempack;

    const auto round_to_dst_elempack_multiple = [elempack, dst_elempack](int value) {
      return (value * elempack + dst_elempack - 1) / dst_elempack;
    };

    // NCNN only supports 4D so we
    switch (dims) {
      case 1:
        vkmat.w = round_to_dst_elempack_multiple(w);
        break;
      case 2:
        vkmat.h = round_to_dst_elempack_multiple(h);
        break;
      case 3:
      case 4:
        vkmat.c = round_to_dst_elempack_multiple(c);
        break;
      default:
        // this should be impossible unless there's a bug in ORT code
        ORT_THROW("NCNN only supports 4D.");
    }
  } while (false);

  return vkmat;
}

}  // namespace vulkan
}  // namespace onnxruntime
