import {InferenceSession, SessionHandler} from 'onnxruntime-common';

export class OnnxjsSessionHandler implements SessionHandler {
  dispose(): Promise<void> {
    throw new Error('Method not implemented.');
  }
  inputNames: string[];
  outputNames: string[];
  run(_feeds: SessionHandler.FeedsType, _fetches: SessionHandler.FetchesType,
      _options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    throw new Error('Method not implemented.');
  }
}
