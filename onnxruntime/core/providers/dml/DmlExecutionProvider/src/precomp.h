// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wunknown-warning-option"
//#pragma clang diagnostic ignored "-Werror"
//#pragma clang diagnostic ignored "-Wswitch"
//#pragma clang diagnostic ignored "-Winvalid-noreturn"
//#pragma clang diagnostic ignored "-Wmicrosoft-template-shadow"
//// 5>S:\Repo\onnxruntime1.7.1\onnxruntime\onnxruntime\core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h(468,79): 
////            error : field 'm_impl' will be initialized after base 'MLOperatorAttributes' [-Werror,-Wreorder-ctor]
//#pragma clang diagnostic ignored "-Wreorder-ctor"
////#pragma clang diagnostic ignored "--Wunknown-warning-option"



#include <memory>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <optional>
#include <list>
#include <map>
#include <deque>
#include <chrono>
#include <variant>
#include <cassert>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <wil/wrl.h>
#include <wil/result.h>

#include <gsl/gsl>

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include "External/D3DX12/d3dx12.h"

#include <DirectML.h>

// DirectML helper libraries
#include "External/DirectMLHelpers/ApiTraits.h"
#include "External/DirectMLHelpers/ApiHelpers.h"
#include "External/DirectMLHelpers/DirectMLSchema.h"
#include "External/DirectMLHelpers/AbstractOperatorDesc.h"
#include "External/DirectMLHelpers/GeneratedSchemaTypes.h"
#include "External/DirectMLHelpers/SchemaHelpers.h"
#include "External/DirectMLHelpers/GeneratedSchemaHelpers.h"

using Microsoft::WRL::ComPtr;

// Windows pollutes the macro space, causing a build break in schema.h.
#undef OPTIONAL

#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/Common.h"

#include "DmlCommon.h"
#include "ErrorHandling.h"
#include "TensorDesc.h"
#include "DescriptorPool.h"
#include "IExecutionProvider.h"

//#pragma clang diagnostic pop
