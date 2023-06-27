// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {CheckpointState as CheckpointStateInterface } from './training-session.js'
//     TrainingSession as TrainingSessionInterface} from './training-session.js';
import {CheckpointHandler} from '../common/lib/backend.js';
import {TrainingBackend} from 'onnxruntime-common';
import { InferenceSession } from 'onnxruntime-web';
import { resolveBackend } from '../web/lib/onnxjs/backend.js';

// TODO: code would be more elegant if laodCheckpoint + saveCheckpoint were in the TrainingSession class, but
// this doesn't follow the design of ORT training in other languages

// export class TrainingSession implements TrainingSessionInterface {
//     private handler: TrainingSessionHandler;
//     private constructor(handler: TrainingSessionHandler) {
//         this.handler = handler;
//     }

//     static loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState> {
//         throw new Error("Method not implemented yet");
//     }

//     lazyResetGrad(): void {
//         throw new Error('Method not implemented.');
//     }
//     trainStep(feeds: Session.OnnxValueMapType, options?: any): Promise<Session.OnnxValueMapType>;
//     trainStep(feeds: Session.OnnxValueMapType, fetches: Session.FetchesType, options?: any): Promise<Session.OnnxValueMapType>;
//     trainStep(feeds: unknown, fetches?: unknown, options?: unknown): Promise<import("./inference-session").Session.OnnxValueMapType> {
//         throw new Error('Method not implemented.');
//     }
//     optimizerStep(options?: any): void {
//         throw new Error('Method not implemented.');
//     }

//     release(): Promise<void> {
//         throw new Error('Method not implemented.');
//     }

// }

export class CheckpointState implements CheckpointStateInterface {
    private handler: CheckpointHandler;
    private constructor(handler: CheckpointHandler) {
        this.handler = handler;
    }

    static async loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState> {
        const backendHints: string[] = [];
        const backend: TrainingBackend = await resolveBackend(backendHints);
        let fileOrArray: string|Uint8Array;
        if (checkpoint instanceof Uint8Array) {
            fileOrArray = new Uint8Array(checkpoint, 0, checkpoint.byteLength);
        }
        else if (typeof checkpoint === 'string') {
            fileOrArray = checkpoint;
        }
        else {
            throw new TypeError('unexpected argument -- loadCheckpoint must take in a file path or buffer.');
        }
        const handler = await backend.createCheckpointHandler(fileOrArray);
        return new CheckpointState(handler);
    }

    release(): Promise<void> {
        return this.handler.dispose();
    }
};
