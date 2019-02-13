#include "core/providers/brainslice/fpga_util.h"
#include "core/providers/brainslice/fpga_message.h"
#include "BrainSlice_client.h"
#include "brainslice_parameters.h"

#include <algorithm>
#include <thread>

namespace onnxruntime {
namespace fpga {
FPGAUtil::FPGAUtil() : status_(FPGAUtil::FPGA_Status::NotInitialized), max_buffer_size_(0) {
}

FPGAUtil::~FPGAUtil() {
  if (status_ == FPGAUtil::FPGA_Status::Initialized)
    FPGA_CloseHandle(handle_);
}

std::once_flag fpga_init_flag;

#define FPGA_RETURN_IF_ERROR(expr)                                                                                          \
  do {                                                                                                                      \
    auto _status = (expr);                                                                                                  \
    if ((_status != FPGA_STATUS_SUCCESS))                                                                                   \
      return common::Status(common::ONNXRUNTIME, common::FAIL, "FPGA call failed with status: " + std::to_string(_status)); \
  } while (0)

Status FPGAUtil::InitFPGA(std::unique_ptr<FPGAUtil>* handle) {
  *handle = std::unique_ptr<FPGAUtil>(new FPGAUtil());
  constexpr int max_retry = 5;
  constexpr int fpga_sleep_time = 200;
  int count = 0;
  FPGA_STATUS status;
  while (count < max_retry) {
    status = FPGA_IsDevicePresent(NULL, NULL);
    if (status == FPGA_STATUS_SUCCESS)
      break;
    count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(fpga_sleep_time));
  }
  if (status != FPGA_STATUS_SUCCESS) {
    (*handle)->status_ = FPGAUtil::FPGA_Status::InitializeFailed;
    (*handle)->error_message_ = "FPGA device initialize failed after retry " + std::to_string(max_retry) + " times.";
    return Status(common::ONNXRUNTIME, common::FAIL, (*handle)->error_message_);
  }

  status = FPGA_CreateHandle(&(*handle)->handle_, 0, 0, NULL, NULL);
  if (status != FPGA_STATUS_SUCCESS) {
    (*handle)->status_ = FPGAUtil::FPGA_Status::InitializeFailed;
    (*handle)->error_message_ = "Get FPGA device handle failed with error code:  " + std::to_string(status);
    return Status(common::ONNXRUNTIME, common::FAIL, (*handle)->error_message_);
  } else {
    (*handle)->status_ = FPGAUtil::FPGA_Status::Initialized;
    DWORD buffer_size;
    FPGA_RETURN_IF_ERROR(FPGA_GetBufferSize((*handle)->handle_, &buffer_size));
    (*handle)->max_buffer_size_ = buffer_size;
    return Status::OK();
  }
}

static std::unique_ptr<FPGAUtil> fpga_default_handle;

FPGAUtil& FPGAUtil::Instance() {
  std::call_once(fpga_init_flag, InitFPGA, &fpga_default_handle);
  return *fpga_default_handle;
}

void release_slot(FPGA_HANDLE handle, SLOT_HANDLE slot) {
  FPGA_ReleaseSlot(handle, slot);
}

void release_buffer(FPGA_HANDLE handle, SLOT_HANDLE slot) {
  FPGA_DiscardSlotOutputBuffer(handle, slot);
}

struct NetworkMessageHeader {
  unsigned int dst_ip : 32;      // destination IP
  unsigned short req_id_1 : 16;  // Upper 16b of messageID
  unsigned short dst_ep : 16;    // destination endpoint
  unsigned short slot : 16;      // pcie slot to use
  unsigned short vc : 16;        // elastic router virtual channel to use
  unsigned short req_id_0 : 16;  // Lower 16b of messageID
  unsigned char reserved2 : 8;   // reserved, set to 0
  unsigned char ver : 8;         // header version, set to BSSHELL_HEADER_VER
};

void WriteMessage(const std::string& p_filename, const void* p_buffer, size_t p_size) {
  FILE* fp = fopen(p_filename.c_str(), "wb");
  fwrite(p_buffer, p_size, 1, fp);
  fclose(fp);
}

Status FPGAUtil::SendSync(const uint32_t p_dstIP, std::function<int32_t(void*, size_t*)> prepare_request, std::function<int32_t(void*, size_t)> process_response) {
  if (status_ != Initialized)
    return Status(common::ONNXRUNTIME, common::FAIL, "FPGA device is not ready, error: " + error_message_);

  SLOT_HANDLE slot;
  FPGA_RETURN_IF_ERROR(FPGA_AcquireSlot(handle_, &slot));
  std::shared_ptr<SLOT_HANDLE__> slot_guard(slot, [&](auto p) { if (p) release_slot(handle_, p); });
  PDWORD input_buffer;
  FPGA_RETURN_IF_ERROR(FPGA_GetSlotInputBuffer(slot, &input_buffer));
  //FPGA_RETURN_IF_ERROR(FPGA_GetInputBufferPointer(handle_, 0, &input_buffer));
  release_buffer(handle_, slot_guard.get());
  std::shared_ptr<DWORD> buffer_guard(input_buffer, [=](auto p) { if (p) release_buffer(handle_, slot_guard.get()); });
  //std::shared_ptr<DWORD> buffer_guard(input_buffer, [&](auto p) { if (p) FPGA_DiscardOutputBuffer(handle_, 0); });

  PDWORD request_butter = input_buffer + (sizeof(NetworkMessageHeader) / sizeof(DWORD));
  DWORD buffer_size;
  FPGA_RETURN_IF_ERROR(FPGA_GetBufferSize(handle_, &buffer_size));
  FPGARequest request(2, request_butter, buffer_size - (sizeof(NetworkMessageHeader) / sizeof(DWORD)));
  FPGARequestMessage fpgaRequestMessage(request);
  std::vector<uint32_t> hops = {p_dstIP, 0};
  fpgaRequestMessage.StartMessage(static_cast<uint8_t>(hops.size()));

  auto& requestStream = request.GetStream();
  size_t requestSize = requestStream.GetRemainingBufferSize();
  prepare_request(requestStream.GetCurrent(), &requestSize);
  requestStream.Reserve(requestSize);
  fpgaRequestMessage.EndMessage(hops, 1);

  NetworkMessageHeader hdr;
  memset((void*)&hdr, 0x0, sizeof(NetworkMessageHeader));
  //TODO: does message id matters?
  int message_id = 0;

  hdr.dst_ep = 1;
  hdr.dst_ip = 0;
  //According to Adrian, the value of slot field here doesn't matter.
  hdr.slot = 0;
  hdr.vc = 0;
  hdr.ver = 0;
  hdr.req_id_0 = message_id & 0xffff;
  hdr.req_id_1 = (message_id >> 16) & 0xffff;
  memcpy(input_buffer, &hdr, sizeof(NetworkMessageHeader));

  //send request
  DWORD message_size = static_cast<DWORD>(request.GetRequestSize());
  message_size += sizeof(NetworkMessageHeader);
  // FPGA requires 16 bytes alignment.
  uint32_t aligmentBytes = message_size % 16;
  if (aligmentBytes != 0) {
    aligmentBytes = 16 - aligmentBytes;
    message_size += aligmentBytes;
  }

  FPGA_RETURN_IF_ERROR(FPGA_SendSlotInputBuffer(handle_, slot, message_size));
  //FPGA_RETURN_IF_ERROR(FPGA_SendInputBuffer(handle_, 0, message_size));
  PDWORD response_buffer;
  FPGA_RETURN_IF_ERROR(FPGA_GetSlotOutputBuffer(slot, &response_buffer));
  //FPGA_RETURN_IF_ERROR(FPGA_GetOutputBufferPointer(handle_, 0, &response_buffer));
  DWORD response_buffer_size;
  FPGA_RETURN_IF_ERROR(FPGA_WaitSlotOutputBuffer(handle_, slot, &response_buffer_size));
  //FPGA_RETURN_IF_ERROR(FPGA_WaitOutputBuffer(handle_, 0, &response_buffer_size));
  NetworkMessageHeader recvHdr;
  memcpy(&recvHdr, response_buffer, sizeof(NetworkMessageHeader));
  PDWORD response_message = response_buffer + (sizeof(NetworkMessageHeader) / sizeof(DWORD));
  FPGAResponse response(response_message, response_buffer_size - sizeof(NetworkMessageHeader));

  FPGAResponseMessage responseMessage(response);
  FPGABuffer submessage;

  while (SubMessageHeader* submessageHdr = responseMessage.ReadNextSubMessage(submessage)) {
    switch (static_cast<SubMessageType>(submessageHdr->type_)) {
      case SubMessageType::SubMessageType_HEX_SUBMESSAGE:
        //TODO: decode the message
        //HexException::Throw(submessage.GetStart() - sizeof(SubMessageHeader), submessageHdr->m_length + sizeof(SubMessageHeader));
        WriteMessage("dumpHexMessage", submessage.GetStart() - sizeof(SubMessageHeader), submessageHdr->length_ + sizeof(SubMessageHeader));
        //return Status(common::ONNXRUNTIME, common::FAIL, "FPGA response failed with HEX_SUBMESSAGE.");
        return Status(common::ONNXRUNTIME, common::FAIL, "Hex happened, please check the dumped message.");
      case SubMessageType::SubMessageType_APPLICATION_SUBMESSAGE:
        if (process_response) {
          auto status = process_response(
              submessage.GetStart() - sizeof(SubMessageHeader),
              submessage.GetBufferSize() + sizeof(SubMessageHeader));

          if (status)
            //TODO: decode the status
            //throw std::runtime_error("Error processing response: " + ToString(static_cast<BrainSlice::Types::ClientStatus>(status)));
            return Status(common::ONNXRUNTIME, common::FAIL, "FPGA process response failed with status: " + std::to_string(status));
          break;
        }

      default:
        break;
    }
  }

  return Status::OK();
}

Status FPGAUtil::GetCapabilities(const uint32_t p_dstIP, BS_Capabilities* out) {
  return SendSync(p_dstIP,
    [out](void* buffer, size_t* size) {
      return BS_CommonFunctions_GetCapabilities_Request(&out->m_bsParameters, buffer, size);
    },
    [out](void* buffer, size_t size) {
      size_t result_size = sizeof(BS_Capabilities);
      return BS_CommonFunctions_GetCapabilities_Response(&out->m_bsParameters, buffer, size, out, &result_size);
    });
}

Status FPGAUtil::GetParameters(const uint32_t p_dstIP, BrainSlice_Parameters* out) {
  return SendSync(p_dstIP,
    [&](void* buffer, size_t* size) {
      return BrainSlice::GetParametersRequest(buffer, size);
    },
    [&](void* buffer, size_t size) {
      return BrainSlice::GetParametersResponse(buffer, size, out);
    });
}

}  // namespace fpga
}  // namespace onnxruntime
