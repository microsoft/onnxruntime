// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Session, TrainingSessionHandler } from '.';
import {resolveBackend} from './backend-impl';
import {CheckpointState as CheckpointStateInterface,
    TrainingSession as TrainingSessionInterface} from './training-session';
import {CheckpointHandler} from './backend';

// TODO: code would be more elegant if laodCheckpoint + saveCheckpoint were in the TrainingSession class, but
// this doesn't follow the design of ORT training in other languages

export class TrainingSession implements TrainingSessionInterface {
    private handler: TrainingSessionHandler;
    private constructor(handler: TrainingSessionHandler) {
        this.handler = handler;
    }

    static loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState> {
        throw new Error("Method not implemented yet");
    }

    lazyResetGrad(): void {
        throw new Error('Method not implemented.');
    }
    trainStep(feeds: Session.OnnxValueMapType, options?: any): Promise<Session.OnnxValueMapType>;
    trainStep(feeds: Session.OnnxValueMapType, fetches: Session.FetchesType, options?: any): Promise<Session.OnnxValueMapType>;
    trainStep(feeds: unknown, fetches?: unknown, options?: unknown): Promise<import("./inference-session").Session.OnnxValueMapType> {
        throw new Error('Method not implemented.');
    }
    optimizerStep(options?: any): void {
        throw new Error('Method not implemented.');
    }

}

export class CheckpointState implements CheckpointStateInterface {
    private handler: CheckpointHandler;
    private constructor(handler: CheckpointHandler) {
        this.handler = handler;
    }

    static async loadCheckpoint(checkpoint: string|ArrayBufferLike): Promise<CheckpointState> {
        const eps = options.executionProviders || [];
        const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
        const backend = await resolveBackend(backendHints);
        const handler = await backend.createCheckpointState(checkpoint);
        return new CheckpointState(handler);
    }
};
