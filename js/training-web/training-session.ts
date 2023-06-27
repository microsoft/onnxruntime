// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {CheckpointState as CheckpointStateImpl } from './training-session-impl.js';
import {InferenceSession} from '../common/lib/inference-session.js';

/* eslint-disable @typescript-eslint/no-redeclare */

export interface TrainingSession {
   lazyResetGrad(): void;

   trainStep(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

   trainStep(feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType, options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

   optimizerStep(options?: InferenceSession.RunOptions): void;

   release(): void;
}

export interface TrainingSessionFactory {
   create(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
      optimizerModel: ArrayBufferLike|string, options?: InferenceSession.SessionOptions): Promise<TrainingSession>;
}

export interface CheckpointState {
   // saveCheckpoint(checkpoint: String): Promise<CheckpointState>;
   release(): Promise<void>;
}

export interface CheckpointStateFactory {
   loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState>;
}

export const CheckpointState: CheckpointStateFactory = CheckpointStateImpl;
// export const TrainingSession: TrainingSessionFactory = TrainingSessionImpl;
