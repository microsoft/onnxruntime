#include "client/api.h"
#include "bond_response.h"
#include "client/response.h"

namespace onnxruntime {
int32_t BrainSlice_Response(const BrainSlice_Parameters* sku,
                           const void* message,
                           const size_t messageSize,
                           const void** payload,
                           size_t* payloadSize){
  assert(sku);
  // use -1 to skip functionID check
  BrainSlice::Response<-1> response(*sku, message, messageSize);
  assert(payload);
  assert(payloadSize);
  response.getPayload(*payload, *payloadSize);
  return response.status();
}
}  // namespace BrainSlice
