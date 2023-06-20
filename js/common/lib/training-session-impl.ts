// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {CheckpointState as CheckpointStateInterface} from './training-session';

export class CheckpointState implements CheckpointStateInterface {
    loadCheckpoint(checkpointPath: String): Promise<CheckpointState> {
        throw new Error('Method not implemented.');
    }

};
