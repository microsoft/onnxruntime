#include "client/api.h"
#include "bond_parser_util.h"
#include "client/request.h"
#include "bond_request.h"

namespace onnxruntime {
int32_t BrainSlice_Request(const BrainSlice_Parameters* sku,
                           const bond_util::BondStruct* args,
                           uint32_t function_id,
                           size_t payloadSize,
                           void** payload,
                           void* message,
                           size_t* messageSize){
  assert(sku);
  assert(messageSize);

  BrainSlice::FirmwareRequest<bond_util::BondStruct> request(*sku, function_id, *args, message, *messageSize);
  assert(payload);
  request.reservePayload(*payload, payloadSize);
  *messageSize = request.size();
  return request.status();
}
}  // namespace BrainSlice
