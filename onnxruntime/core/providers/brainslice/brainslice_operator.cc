#include "core/providers/brainslice/brainslice_operator.h"
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "float_conversion.h"
#include "bond_struct.h"
#include "bond_request.h"
#include "bond_response.h"

namespace onnxruntime {
namespace brainslice {

using float16type = half_float::half;

static const int kBrainSliceOperatorInit = 24;
static const int kBrainSliceOperatorExecute = 25;

typedef enum {
  BrainSliceMemInitialVRF = 0,
  BrainSliceMemMRF = 1,
  BrainSliceMemAddSubVRF = 2,
  BrainSliceMemMultiplyVRF = 3,
  BrainSliceMemVectorDRAM = 4,
  BrainSliceMemMatrixDRAM = 5,
  BrainSliceMemAddSubVRF_0 = 6,
  BrainSliceMemAddSubVRF_1 = 7,
} BrainSliceMemList;

static bool IsVectorMemType(BrainSliceMemList type) {
  switch (type) {
    case BrainSliceMemInitialVRF:
    case BrainSliceMemAddSubVRF:
    case BrainSliceMemAddSubVRF_0:
    case BrainSliceMemAddSubVRF_1:
    case BrainSliceMemMultiplyVRF:
    case BrainSliceMemVectorDRAM:
      return true;

    default:
      return false;
  }
}

static bool IsMatrixMemType(BrainSliceMemList type) {
  switch (type) {
    case BrainSliceMemMRF:
    case BrainSliceMemMatrixDRAM:
      return true;

    default:
      return false;
  }
}

static ISA_Mem GetBrainSliceMemoryLocation(BrainSliceMemList type) {
  switch (type) {
    case BrainSliceMemInitialVRF:
      return ISA_Mem_InitialVrf;
    case BrainSliceMemMRF:
      return ISA_Mem_MatrixRf;
    case BrainSliceMemAddSubVRF:
      return ISA_Mem_AddSubVrf;
    case BrainSliceMemAddSubVRF_0:
      return ISA_Mem_AddSubVrf_0;
    case BrainSliceMemAddSubVRF_1:
      return ISA_Mem_AddSubVrf_1;
    case BrainSliceMemMultiplyVRF:
      return ISA_Mem_MultiplyVrf;
    case BrainSliceMemVectorDRAM:
      return ISA_Mem_Dram;
    case BrainSliceMemMatrixDRAM:
      return ISA_Mem_Dram;
  }

  ORT_THROW("Unknown BrainSliceMemList type: %d", (int)type);
}

static size_t AddPaddingToPayload(const BrainSlice_Parameters& bsParameters, size_t payloadSize) {
  const uint32_t block_dim = bsParameters.NATIVE_DIM;
  return ((payloadSize + block_dim - 1) / block_dim) * block_dim;
}

// Load firmware for this BrainSlice node.
static common::Status LoadFirmware(const fpga::FPGAHandle& handle, const NodeAttributes& attributes) {
  const auto& instructionsAttr = attributes.find("firmware_instructions");
  const auto& dataAttr = attributes.find("firmware_data");
  const auto& schemaAttr = attributes.find("firmware_schema");
  ORT_ENFORCE(instructionsAttr != attributes.end(), "Failed to find 'firmware_instructions' attribute.");
  ORT_ENFORCE(dataAttr != attributes.end(), "Failed to find 'firmware_data' attribute.");
  ORT_ENFORCE(schemaAttr != attributes.end(), "Failed to find 'firmware_schema' attribute.");

  uint32_t* instructionsBytes = (uint32_t*)(instructionsAttr->second.s().c_str());
  uint32_t* dataBytes = (uint32_t*)(dataAttr->second.s().c_str());
  uint64_t* schemaBytes = (uint64_t*)(schemaAttr->second.s().c_str());

  return handle.LoadFirmware(
      std::vector<uint32_t>(instructionsBytes, instructionsBytes + (instructionsAttr->second.s().size() / sizeof(uint32_t))),
      std::vector<uint32_t>(dataBytes, dataBytes + (dataAttr->second.s().size() / sizeof(uint32_t))),
      std::vector<uint64_t>(schemaBytes, schemaBytes + (schemaAttr->second.s().size() / sizeof(uint64_t))));
}

static std::vector<float16type> GetFloat16FromTensor(const Tensor& input_value) {
  const int64_t input_size = input_value.Shape().Size();
  std::vector<float16type> result(input_size);

  if (input_value.DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    const MLFloat16* input_data = input_value.Data<MLFloat16>();
    memcpy(result.data(), input_data, input_size * sizeof(float16type));
  } else {
    const float* input_data = input_value.Data<float>();
    Float32ToFloat16(input_data, (uint16_t*)result.data(), result.size());
  }

  return result;
}

// Load model matrixes and vectors onto the FPGA.
static common::Status LoadWeights(const fpga::FPGAHandle& handle, const NodeAttributes& attributes, const OpKernelInfo& info) {
  const auto& inputs = info.node().InputDefs();

  const auto& addressAttr = attributes.find("input_addresses");
  const auto& memTypeAttr = attributes.find("input_memtypes");
  ORT_ENFORCE(addressAttr != attributes.end(), "Failed to find 'input_addresses' attribute.");
  ORT_ENFORCE(memTypeAttr != attributes.end(), "Failed to find 'input_memtypes' attribute.");
  ORT_ENFORCE(addressAttr->second.ints_size() == inputs.size(), "Attribute 'input_addresses' must be the same size as inputs.");
  ORT_ENFORCE(memTypeAttr->second.ints_size() == inputs.size(), "Attribute 'input_memtypes' must be the same size as inputs.");

  for (int i = 0; i < inputs.size(); i++) {
    const Tensor* input_value;
    if (info.TryGetConstantInput(i, &input_value)) {
      // Get BrainSlice memory type and address for the data.
      int addr = static_cast<int>(addressAttr->second.ints(i));
      BrainSliceMemList type = static_cast<BrainSliceMemList>(memTypeAttr->second.ints(i));

      ORT_ENFORCE(IsVectorMemType(type) || IsMatrixMemType(type), "Unknown BrainSlice memory type.");
      if (IsVectorMemType(type)) {
        const std::vector<float16type> data = GetFloat16FromTensor(*input_value);
        ORT_RETURN_IF_ERROR(handle.LoadVector(data, addr, GetBrainSliceMemoryLocation(type)));
      } else if (IsMatrixMemType(type)) {
        const TensorShape& shape = input_value->Shape();
        ORT_ENFORCE(shape.NumDimensions() == 2, "Matrix input must have 2 dimensions [rows, cols].");
        const int rows = static_cast<int>(shape[0]);
        const int cols = static_cast<int>(shape[1]);
        const std::vector<float16type> data = GetFloat16FromTensor(*input_value);
        ORT_RETURN_IF_ERROR(handle.LoadMatrix(data, rows, cols, addr, true, GetBrainSliceMemoryLocation(type)));
      }
    }
  }

  return common::Status::OK();
}

static bond_util::BondStruct CreateRuntimeArguments(uint32_t function_id, std::vector<uint32_t>&& data) {
  return bond_util::BondStruct({
      {{"functionId", 0}, {}, bond_util::Value(function_id)},
      {{"data", 0}, {}, bond_util::Value(std::move(data))},
  });
}

// Initialization of BrainSlice model. This is expected to verify the hardware SKU can execute this firmware.
static common::Status InitializeModel(const fpga::FPGAHandle& handle) {
  const BrainSlice_Parameters& bsParameters = handle.GetParameters();
  return handle.SendSync(
      [&](void* buffer, size_t* size) {
        void* payloadPtr;
        size_t payloadSize = 0;
        bond_util::BondStruct runtime_argument = CreateRuntimeArguments(0, {});
        return BrainSlice_Request(&bsParameters, &runtime_argument, kBrainSliceOperatorInit, payloadSize, &payloadPtr, buffer, size);
      },
      [&](void* buffer, size_t size) {
        const void* payloadPtr;
        size_t payloadSize = 0;
        return BrainSlice_Response(&bsParameters, buffer, size, &payloadPtr, &payloadSize);
      });
}

BrainSliceOperator::BrainSliceOperator(const OpKernelInfo& info) : BrainSliceOpKernel(info) {
  const NodeAttributes& attributes = info.node().GetAttributes();
  const auto& outputs = info.node().OutputDefs();
  ORT_ENFORCE(outputs.size() == 1, "BrainSlice operator currently only supports single output nodes.");

  // Store the output dimensions from the output definition.  We rely on the output shape
  // being represented in the model, since we cannot infer the output shape based on inputs.
  output_dims_ = TensorShape(utils::GetTensorShapeFromTensorShapeProto(*outputs[0]->Shape()));

  // Check if the output is interleaved.  BrainSlice v3 performs convolutions in a way that
  // leaves the output tensor interleaved relative to the output shape.
  const auto& outputInterleavedAttr = attributes.find("output_interleaved");
  if (outputInterleavedAttr != attributes.end())
    output_interleaved_ = outputInterleavedAttr->second.i() != 0;

  // Load firmware for this BrainSlice node.
  const fpga::FPGAHandle& handle = provider_->GetFPGAHandle();
  ORT_ENFORCE(LoadFirmware(handle, attributes).IsOK(), "Failed to load BrainSlice firmware.");

  // Initialization of BrainSlice model. This is expected to verify the hardware SKU can execute this firmware.
  ORT_ENFORCE(InitializeModel(handle).IsOK(), "Failed to initialize model firmware");

  // Load model matrixes and vectors onto the FPGA.
  ORT_ENFORCE(LoadWeights(handle, attributes, info).IsOK(), "Failed to load BrainSlice weights.");
}

Status BrainSliceOperator::Compute(OpKernelContext* context) const {
  const fpga::FPGAHandle& handle = provider_->GetFPGAHandle();
  const BrainSlice_Parameters& bsParameters = handle.GetParameters();

  auto input = context->Input<Tensor>(0);
  const TensorShape& input_dims = input->Shape();
  int64_t input_size = input_dims.Size();

  // If the first dimension (batch size) is negative, use the batch size from the input.
  TensorShape output_dims(output_dims_);
  if (input_dims.NumDimensions() == output_dims.NumDimensions() && output_dims[0] == -1)
    output_dims[0] = input_dims[0];

  auto output = context->Output(0, output_dims);
  int64_t output_size = output_dims.Size();

  return handle.SendSync(
      [&](void* buffer, size_t* size) {
        uint16_t* payloadPtr;
        size_t payloadSize = AddPaddingToPayload(bsParameters, input_size * sizeof(uint16_t));
        bond_util::BondStruct runtime_argument = CreateRuntimeArguments(0, {});
        auto status = BrainSlice_Request(&bsParameters, &runtime_argument, kBrainSliceOperatorExecute, payloadSize, (void**)&payloadPtr, buffer, size);
        if (status)
          return status;

        // Copy input tensor to FPGA input.
        if (input->DataType() == DataTypeImpl::GetType<float>())
          Float32ToFloat16(input->Data<float>(), payloadPtr, input_size);
        else
          memcpy(payloadPtr, input->Data<MLFloat16>(), input_size * sizeof(uint16_t));
        return 0;
      },
      [&](void* buffer, size_t size) {
        const uint16_t* payloadPtr;
        size_t payloadSize = output_size * sizeof(uint16_t);
        auto status = BrainSlice_Response(&bsParameters, buffer, size, (const void**)&payloadPtr, &payloadSize);
        if (status)
          return status;

        if (output_interleaved_) {
          // The FPGA output is interleaved.  We must scatter-gather the FPGA output to the tensor output.
          const size_t numVectors = std::accumulate(begin(output_dims.GetDims()), end(output_dims.GetDims()) - 1, 1LL, std::multiplies<>{});
          const size_t elementsPerVector = output_dims[output_dims.NumDimensions() - 1];
          const size_t numInterleavedBlocks = (elementsPerVector + native_dim_ - 1) / native_dim_;
          const size_t elementsPerInterleavedBlock = numVectors * native_dim_;

          size_t offset = 0;
          for (size_t batch = 0; batch < numVectors; batch++) {
            for (size_t interleavedBlock = 0; interleavedBlock < numInterleavedBlocks; interleavedBlock++) {
              const size_t elementsToCopy = std::min((size_t)native_dim_, elementsPerVector - interleavedBlock * native_dim_);
              const size_t payloadOffset = interleavedBlock * elementsPerInterleavedBlock;

              if (output->DataType() == DataTypeImpl::GetType<float>())
                Float16ToFloat32(payloadPtr + payloadOffset, output->MutableData<float>() + offset, elementsToCopy);
              else
                memcpy(output->MutableData<MLFloat16>() + offset, payloadPtr + payloadOffset, elementsToCopy * sizeof(uint16_t));

              offset += elementsToCopy;
            }

            payloadPtr += native_dim_;
          }
        } else {
          // Copy the FPGA output directly to the tensor output.
          if (output->DataType() == DataTypeImpl::GetType<float>())
            Float16ToFloat32(payloadPtr, output->MutableData<float>(), output_size);
          else
            memcpy(output->MutableData<MLFloat16>(), payloadPtr, input_size * sizeof(uint16_t));
        }

        return 0;
      });
}

ONNX_OPERATOR_KERNEL_EX(
	BrainSlice,
	kMSDomain,
	1,
	kBrainSliceExecutionProvider,
	KernelDefBuilder()
	.TypeConstraint("T", { DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>() })
	.SetDefaultInputsMemoryType(OrtMemTypeCPUInput)
	.SetDefaultOutputMemoryType(OrtMemTypeCPUOutput),
	brainslice::BrainSliceOperator);
}  // namespace brainslice
}  // namespace onnxruntime
