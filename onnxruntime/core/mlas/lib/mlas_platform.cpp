#include "mlas_api_overrides.h"
#include "kleidiai/mlasi_kleidiai.h"

void MlasPlatformInitialize() {
    MlasInitializeDefaultApiOverrides();

    if (EnableKleidiAI()) {
        MlasApiOverrides overrides = {};
        overrides.Gemm = &ArmKleidiAI::MlasGemm;
        overrides.GemmBatch = &ArmKleidiAI::MlasGemmBatch;
        overrides.GemmPackB = &ArmKleidiAI::MlasGemmPackB;
        overrides.GemmPackBSize = &ArmKleidiAI::MlasGemmPackBSize;

        MlasRegisterApiOverrides(overrides);
    }
}