// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TrainingBackend} from './backend.js';
import {CheckpointState as CheckpointStateInterface} from './training-session.js';
// import {CheckpointState as CheckpointStateInterface, TrainingSession as TrainingSessionInterface} from './training-session.js';
import {resolveBackend} from './backend-impl.js';
import {TrainingHandler} from './backend.js';


// export class TrainingSession implements TrainingSessionInterface {
//     private checkpointState: CheckpointState;
//     private constructor(checkpointState: CheckpointState) {
//         this.checkpointState = checkpointState;
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
    private handler: TrainingHandler;
    private constructor(handler: TrainingHandler) {
        this.handler = handler;
    }

    static async loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState> {
        console.log('loading Checkpoint');
        const backendHints: string[] = [];
        const backend = await resolveBackend(backendHints);
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

        const handler = await (backend as TrainingBackend).loadCheckpoint(fileOrArray);
        return new CheckpointState(handler);
    }

    release(): Promise<void> {
        return this.handler.disposeCheckpointState();
    }
};
