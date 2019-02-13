// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_request.h"
#include "core/providers/brainslice/fpga_response.h"
namespace onnxruntime {
namespace fpga {
#pragma pack(push, 1)
// 24 bit uint
struct uint24 {
  uint16_t low16_;

  uint8_t high8_;

  explicit uint24()
      : low16_(0),
        high8_(0) {
  }

  uint24& operator=(const uint32_t integer) {
    low16_ = integer & 0xFFFF;
    high8_ = (integer >> 16) & 0xFF;
    return *this;
  }

  bool operator==(const uint32_t integer) const {
    return (high8_ == (integer >> 16)) && (low16_ == (integer & 0xFFFF));
  }
};
#pragma pack(pop)
static_assert(sizeof(uint24) * CHAR_BIT == 24, "unint24 should have 24 bits.");

// Sub message types.
enum class SubMessageType : uint8_t {
  SubMessageType_CATAPULT_HEADER = 0,
  SubMessageType_CATAPULT_FOOTER = 1,
  SubMessageType_APPLICATION_SUBMESSAGE = 2,
  SubMessageType_HEALTH_SUBMESSAGE = 3,
  SubMessageType_PADDING_SUBMESSAGE = 4,
  SubMessageType_HEX_SUBMESSAGE = 5,
  SubMessageType_INVALID = 6
};

// FPGA sub message header struct.
#pragma pack(push, 1)
struct SubMessageHeader {
  uint32_t type_ : 8; // SubMessageType enum - Windows build fails if this type is uint8_t, Linux build fails if this is SubMessageType

  uint32_t length_ : 24;
};
#pragma pack(pop)
static_assert(sizeof(SubMessageHeader) * CHAR_BIT == 32, "SubMessageHeader should have 32 bits.");

// FPGA message header struct.
struct CatapultHeader {
  uint16_t protocolMajorVersion_;

  uint16_t requestId_;

  uint8_t ltlHops_;

  uint8_t ltlNextHop_;

  uint8_t ltlLastHop_;

  uint8_t padding_;
};
static_assert(sizeof(CatapultHeader) * CHAR_BIT == 64, "CatapultHeader should have 64 bits.");

// FPGA message footer struct.
struct CatapultFooter {
  uint32_t subMessageCount_;

  uint32_t globalTime_;
};
static_assert(sizeof(CatapultFooter) * CHAR_BIT == 64, "CatapultFooter should have 64 bits.");

#pragma pack(push, 1)
// FPGA Hex message.
struct HexMessageHeader {
  uint8_t hexCode_;

  uint16_t hexMessageCount_;

  uint8_t reserved_;
};
#pragma pack(pop)
static_assert(sizeof(HexMessageHeader) * CHAR_BIT == 32, "HexMessageHeader should have 32 bits.");

// A wrapper class of FPGA request message.
class FPGARequestMessage {
 public:
  // Construct from a fpga request.
  FPGARequestMessage(FPGARequest& request);

  // 1. Start a message.
  CatapultHeader* StartMessage(uint8_t fpga_ip_count);

  // 2. End current message.
  CatapultFooter* EndMessage(const std::vector<uint32_t>& fpga_ips, const int sub_message_count);

  // Get request.
  FPGARequest& GetRequest();

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FPGARequestMessage);
  // Request.
  FPGARequest& request_;

  // Message header.
  CatapultHeader* message_header_;
};

// A wrapper class of FPGA response message.
class FPGAResponseMessage {
 public:
  // Construct from a fpga response.
  FPGAResponseMessage(const FPGAResponse& response);

  // Read next sub message, and will set the data buffer.
  // Returns nullptr if there's no message left.
  SubMessageHeader* ReadNextSubMessage(FPGABuffer& buffer);

  // Check if there's any hex error.
  static HexMessageHeader* CheckHexMessage(const FPGAResponse& response, FPGABuffer& hex_message_buffer);

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FPGAResponseMessage);
  // Response.
  const FPGAResponse& response_;

  // Response stream.
  FPGABufferStream response_stream_;

  // If the catapult footer message has been read,
  // which indicates the end of the message.
  bool footer_found_;
};
}  // namespace fpga
}  // namespace onnxruntime
