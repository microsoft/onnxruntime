#include "Firmware/lib/BrainSliceService.h"
#include "Loopback_firmware.h"

VOID Example_Model_Loopback(PBS_CONTEXT bs, const Example_Param* args)
{
    Example_Result result;
    result.scalar = args->scalar;

    Example_Model_Loopback_PostResponse(bs, &result);

    vRead1D(bs, ISA_Mem_NetInputQ, DONTCARE, (ISA_NativeCount)args->dim);
    v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
    end_chain(bs);
}

