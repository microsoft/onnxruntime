#include "Firmware/lib/BrainSliceNIOSLib.h"
#include "Firmware/lib/BrainSliceService.h"
#include "BrainSliceOperator_firmware.h"

void init(PBS_CONTEXT p_bs, const BrainSliceOperator_RuntimeArguments* args);
void execute(PBS_CONTEXT p_bs, const BrainSliceOperator_RuntimeArguments* args, int p_startLayerID, int p_endLayerID, bool p_debugMode);

// Init service function for use with ONNX Runtime.
void BrainSliceOperator_Functions_InitOperation(PBS_CONTEXT bs, const BrainSliceOperator_RuntimeArguments* args)
{
    init(bs, args);
}

// Evaluate service function for use with ONNX Runtime.
void BrainSliceOperator_Functions_ExecuteOperation(PBS_CONTEXT bs, const BrainSliceOperator_RuntimeArguments* args)
{
    execute(bs, args, /*p_startLayerID:*/ -1, /*p_endLayerID:*/ -1, /*p_debugMode:*/ false);
}
