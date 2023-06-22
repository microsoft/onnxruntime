import {readFile} from 'fs';
import {CheckpointHandler, TrainingSessionHandler} from 'onnxruntime-common';
import { promisify } from 'util';
// TODO: handler usually imports methods from proxy-wrapper, and proxy-wrapper imports from wasm-core-impl
import * as core from './wasm-core-impl';

export class OnnxruntimeWebAssemblyCheckpointHandler implements CheckpointHandler {
    stateId: number;

    async loadCheckpointAllocate(path: string): Promise<SerializableModeldata> {
        const response = await fetch(path);
        const arrayBuffer = await response.arrayBuffer();
        return this.loadCheckpoint(new Uint8Array(arrayBuffer));
    }

    async loadCheckpoint(pathOrBuffer: string|Uint8Array): Promise<void> {
        if (typeof pathOrBuffer ==='string') {
            if (typeof fetch === 'undefined') {
                // node
                const checkpointData = await promisify(readFile)(pathOrBuffer);
                this.stateId = await core.loadCheckpoint(checkpointData);
            } else {

                this.loadCheckpointAllocate(pathOrBuffer);
            }
        } else {
            this.stateId = await core.loadCheckpoint(pathOrBuffer);
        }
    }
}

export class OnnxruntimeWebAssemblyTrainingSessionHandler implements TrainingSessionHandler {
    private sessionId: number;
    private checkpoint: CheckpointState;

    inputNames: string[];
    outputNames: string[];


    async createTrainingSession(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
      optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSessionHandler> {

      }
}
