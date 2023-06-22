// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OnnxValue} from './onnx-value';
import {CheckpointState as CheckpointStateImpl} from './training-session-impl';
import {Session, RunOptions} from './inference-session';

/* eslint-disable @typescript-eslint/no-redeclare */

export interface TrainingSession {
   lazyResetGrad(): void;

   trainStep(feeds: Session.FeedsType, options?: RunOptions): Promise<Session.ReturnType>;

   trainStep(feeds: Session.FeedsType, fetches: Session.FetchesType, options?: RunOptions): Promise<Session.ReturnType>;

   optimizerStep(options?: RunOptions): void;
}

export interface TrainingSessionFactory {
   create(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
      optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSession>;
}

export interface CheckpointState {
   // saveCheckpoint(checkpoint: String): Promise<CheckpointState>;
}

export interface CheckpointStateFactory {
   loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState>;
}

export const CheckpointState: CheckpointStateFactory = CheckpointStateImpl;
