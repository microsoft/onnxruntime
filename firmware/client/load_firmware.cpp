#include <vector>
#include "load_firmware.h"
#include "client/request.h"
namespace BrainSlice {
int32_t LoadFirmwareAPI(
    uint32_t* instructions,
    size_t instruction_size,
    uint32_t* data,
    size_t data_size,
    uint64_t* schema,
    size_t schema_size,
    void* message, size_t* messageSize) {
    std::vector<uint32_t> v_instructions(instructions, instructions + instruction_size);
    std::vector<uint32_t> v_data(data, data + data_size);
    std::vector<uint64_t> v_schema(schema, schema + schema_size);
    return LoadFirmware(std::move(v_instructions),
        std::move(v_data),
        std::move(v_schema),
        message,
        messageSize);
}
}  // namespace BrainSlice
