#include "core/providers/brainslice/fpga_message.h"
#include "core/providers/brainslice/fpga_helper.h"
namespace onnxruntime {
namespace fpga {
static const uint16_t c_catapultHeaderMajorVersion = 0x01;

FPGARequestMessage::FPGARequestMessage(FPGARequest& request)
    : request_(request),
      message_header_(nullptr) {
}

CatapultHeader* FPGARequestMessage::StartMessage(uint8_t fpga_ip_count) {
  if (message_header_ != nullptr) {
    ORT_THROW("StartMessage() has already been called before.");
  }

  FPGABufferStream& requestStream = request_.GetStream();

  SubMessageHeader msgHeader;
  msgHeader.type_ = static_cast<uint32_t>(SubMessageType::SubMessageType_CATAPULT_HEADER);
  msgHeader.length_ = static_cast<uint32_t>(sizeof(CatapultHeader) + sizeof(uint32_t) * fpga_ip_count);
  requestStream.Serialize(msgHeader);

  CatapultHeader catalpultHeader;
  catalpultHeader.protocolMajorVersion_ = c_catapultHeaderMajorVersion;
  catalpultHeader.ltlHops_ = fpga_ip_count;
  catalpultHeader.ltlNextHop_ = 0;
  catalpultHeader.ltlLastHop_ = 0;
  message_header_ = requestStream.Serialize(catalpultHeader);
  // Reserve IP address.
  uint32_t fpgaIP = 0;
  for (uint8_t i = 0; i < catalpultHeader.ltlHops_; ++i) {
    requestStream.Serialize(fpgaIP);
  }
  return message_header_;
}

CatapultFooter* FPGARequestMessage::EndMessage(const std::vector<uint32_t>& fpga_ips, const int sub_message_count) {
  if (message_header_ == nullptr) {
    ORT_THROW("Must call StartMessage() first.");
  }

  if (fpga_ips.size() != message_header_->ltlHops_) {
    ORT_THROW("Expect %u IPs, but get %u IPs.",
                      message_header_->ltlHops_,
                      fpga_ips.size());
  }

  FPGABufferStream& requestStream = request_.GetStream();

  uint32_t* serviceIPs = reinterpret_cast<uint32_t*>(message_header_ + 1);
  for (uint8_t i = 0; i < message_header_->ltlHops_; ++i) {
    // FPGA expect ip in reverse order.
    serviceIPs[i] = FPGAUtil::FlipUint32(fpga_ips[i]);
  }

  SubMessageHeader msgFooter;
  msgFooter.type_ = static_cast<uint32_t>(SubMessageType::SubMessageType_CATAPULT_FOOTER);
  msgFooter.length_ = static_cast<uint32_t>(sizeof(CatapultFooter));
  requestStream.Serialize(msgFooter);

  CatapultFooter footer;
  footer.subMessageCount_ = sub_message_count;
  message_header_ = nullptr;
  return requestStream.Serialize(footer);
}

FPGARequest& FPGARequestMessage::GetRequest() {
  return request_;
}

FPGAResponseMessage::FPGAResponseMessage(const FPGAResponse& response)
    : response_(response),
      response_stream_(response),
      footer_found_(false) {
}

SubMessageHeader* FPGAResponseMessage::ReadNextSubMessage(FPGABuffer& buffer) {
  if (footer_found_ || response_stream_.GetRemainingBufferSize() < sizeof(SubMessageHeader)) {
    return nullptr;
  }

  SubMessageHeader& subMsgHeader = response_stream_.ReadAs<SubMessageHeader>();

  if (static_cast<SubMessageType>(subMsgHeader.type_) >= SubMessageType::SubMessageType_INVALID) {
    ORT_THROW("Invalid sub message type:%u.", subMsgHeader.type_);
  }

  if (static_cast<SubMessageType>(subMsgHeader.type_) == SubMessageType::SubMessageType_CATAPULT_FOOTER) {
    footer_found_ = true;
  }
  uint32_t msgLength = subMsgHeader.length_;
  FPGABuffer msgBuffer(response_stream_.Read(msgLength), msgLength);
  buffer = std::move(msgBuffer);
  return &subMsgHeader;
}

HexMessageHeader* FPGAResponseMessage::CheckHexMessage(const FPGAResponse& response, FPGABuffer& hex_message_buffer) {
  FPGAResponseMessage responseMessage(response);

  HexMessageHeader* hexMessageHeader = nullptr;
  SubMessageHeader* msgHeader = nullptr;
  FPGABuffer msgBuffer;
  while ((msgHeader = responseMessage.ReadNextSubMessage(msgBuffer)) != nullptr) {
    if (static_cast<SubMessageType>(msgHeader->type_) == SubMessageType::SubMessageType_HEX_SUBMESSAGE) {
      FPGABufferStream hexMessageStream(msgBuffer);
      HexMessageHeader& hexHeader = hexMessageStream.ReadAs<HexMessageHeader>();
      hexMessageHeader = &hexHeader;
      hex_message_buffer = std::move(msgBuffer);
      break;
    }
  }
  return hexMessageHeader;
}
}  // namespace fpga
}  // namespace onnxruntime
