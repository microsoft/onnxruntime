import {readFile} from 'fs';
import {CheckpointHandler} from 'onnxruntime-common';
import { promisify } from 'util';
import * as core from './wasm-core-impl';

export class OnnxruntimeWebAssemblyCheckpointHandler implements CheckpointHandler {
    handlerId: number;

    async loadCheckpointAllocate(path: string): Promise<number> {
        const response = await fetch(path);
        const arrayBuffer = await response.arrayBuffer();
        return this.loadCheckpoint(new Uint8Array(arrayBuffer));
    }

    async loadCheckpoint(pathOrBuffer: string|Uint8Array): Promise<number> {
        if (typeof pathOrBuffer ==='string') {
            if (typeof fetch === 'undefined') {
                // node
                const checkpointData = await promisify(readFile)(pathOrBuffer);
                this.handlerId = await core.loadCheckpoint(checkpointData);
                return this.handlerId;
            } else {
                return this.loadCheckpointAllocate(pathOrBuffer);
            }
        } else {
            this.handlerId = await core.loadCheckpoint(pathOrBuffer);
            return this.handlerId;
        }
    }

    dispose(): void {
        core.releaseCheckpoint(this.handlerId);
    }
}

// export class OnnxruntimeWebAssemblyTrainingSessionHandler implements TrainingSessionHandler {
//     private sessionId: number;
//     private checkpoint: CheckpointState;

//     inputNames: string[];
//     outputNames: string[];


//     async createTrainingSession(checkpointState: CheckpointState, trainModel: ArrayBufferLike|string, evalModel: ArrayBufferLike|string,
//       optimizerModel: ArrayBufferLike|string, options?: Session.SessionOptions): Promise<TrainingSessionHandler> {

//       }
// }
