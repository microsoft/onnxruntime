// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OnnxValue} from './onnx-value';
import {CheckpointState as CheckpointStateImpl, TrainingSession as TrainingSessionImpl} from './training-session-impl';
import {Session, RunOptions} from './inference-session';

/* eslint-disable @typescript-eslint/no-redeclare */

export interface TrainingSession {
   lazyResetGrad(): void;

   trainStep(feeds: Session.FeedsType, options?: RunOptions): Promise<Session.ReturnType>;

   trainStep(feeds: Session.FeedsType, fetches: Session.FetchesType, options?: RunOptions): Promise<Session.ReturnType>;

   optimizerStep(options?: RunOptions): void;

   release(): Promise<void>;
}

export interface TrainingSessionFactory {
   create(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
      optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSession>;
}

export interface CheckpointState {
   // saveCheckpoint(checkpoint: String): Promise<CheckpointState>;
   release(): Promise<void>;
}

export interface CheckpointStateFactory {
   loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState>;
}

export const CheckpointState: CheckpointStateFactory = CheckpointStateImpl;
export const TrainingSession: TrainingSessionFactory = TrainingSessionImpl;
