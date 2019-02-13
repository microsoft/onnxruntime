#include "client/api.h"
#include "brainslice_parameters.h"
#include "client/request.h"
#include "client/response.h"
#include "Parameters_types.h"

namespace BrainSlice {

int32_t GetParametersRequest(void* message, size_t* messageSize) {
  return ParametersRequest(message, messageSize);
}

int32_t GetParametersResponse(const void* message, const size_t messageSize, BrainSlice_Parameters* parameters) {
  assert(parameters);

  Parameters out;
  auto status = ParametersResponse(message, messageSize, out);
  if (status)
    return status;

  *parameters = *reinterpret_cast<BrainSlice_Parameters*>(&out);
  return 0;
}

}  // namespace BrainSlice
