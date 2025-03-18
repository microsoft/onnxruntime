// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_session.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gets whether the CoreML execution provider is available.
 */
BOOL ORTIsCoreMLExecutionProviderAvailable(void);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_BEGIN

/**
 * Options for configuring the CoreML execution provider.
 */
@interface ORTCoreMLExecutionProviderOptions : NSObject

/**
 * Whether the CoreML execution provider should run on CPU only.
 */
@property BOOL useCPUOnly;
/**
 * exclude ANE in CoreML.
 */
@property BOOL useCPUAndGPU;
/**
 * Whether the CoreML execution provider is enabled on subgraphs.
 */
@property BOOL enableOnSubgraphs;

/**
 * Whether the CoreML execution provider is only enabled for devices with Apple
 * Neural Engine (ANE).
 */
@property BOOL onlyEnableForDevicesWithANE;

/**
 * Only allow CoreML EP to take nodes with inputs with static shapes. By default it will also allow inputs with
 * dynamic shapes. However, the performance may be negatively impacted if inputs have dynamic shapes.
 */
@property BOOL onlyAllowStaticInputShapes;

/**
 * Create an MLProgram. By default it will create a NeuralNetwork model. Requires Core ML 5 or later.
 */
@property BOOL createMLProgram;

@end

@interface ORTSessionOptions (ORTSessionOptionsCoreMLEP)

/**
 * Enables the CoreML execution provider in the session configuration options.
 * It is appended to the execution provider list which is ordered by
 * decreasing priority.
 *
 * @param options The CoreML execution provider configuration options.
 * @param error Optional error information set if an error occurs.
 * @return Whether the provider was enabled successfully.
 */
- (BOOL)appendCoreMLExecutionProviderWithOptions:(ORTCoreMLExecutionProviderOptions*)options
                                           error:(NSError**)error;
/**
 * Enables the CoreML execution provider in the session configuration options.
 * It is appended to the execution provider list which is ordered by
 * decreasing priority.
 *
 * @param provider_options The CoreML execution provider options in dict.
 *  available keys-values: more detail in core/providers/coreml/coreml_execution_provider.h
 *      kCoremlProviderOption_MLComputeUnits: one of "CPUAndNeuralEngine", "CPUAndGPU", "CPUOnly", "All"
 *      kCoremlProviderOption_ModelFormat: one of "MLProgram", "NeuralNetwork"
 *      kCoremlProviderOption_RequireStaticInputShapes: "1" or "0"
 *      kCoremlProviderOption_EnableOnSubgraphs: "1" or "0"
 * @param error Optional error information set if an error occurs.
 * @return Whether the provider was enabled successfully.
 */
- (BOOL)appendCoreMLExecutionProviderWithOptionsV2:(NSDictionary*)provider_options
                                             error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END
