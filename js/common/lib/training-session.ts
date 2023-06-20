// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OnnxValue} from './onnx-value';
import {CheckpointState as CheckpointStateImpl} from './training-session-impl';


/* eslint-disable @typescript-eslint/no-redeclare */

export declare namespace TrainingSession {

}

export interface CheckpointState {
   loadCheckpoint(checkpointPath: String): Promise<CheckpointState>;
   saveCheckpoint(checkpoint: String): Promise<CheckpointState>;
}
