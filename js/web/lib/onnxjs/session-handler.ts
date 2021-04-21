import {InferenceSession, SessionHandler, Tensor} from 'onnxruntime-common';
import {Session} from './session';
import {Tensor as OnnxjsTensor} from './tensor';

export class OnnxjsSessionHandler implements SessionHandler {
  constructor(private session: Session) {
    this.inputNames = this.session.inputNames;
    this.outputNames = this.session.outputNames;
  }

  async dispose(): Promise<void> {}
  inputNames: readonly string[];
  outputNames: readonly string[];
  async run(
      feeds: SessionHandler.FeedsType, _fetches: SessionHandler.FetchesType,
      _options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType> {
    const inputMap = new Map<string, OnnxjsTensor>();
    for (const name in feeds) {
      if (Object.hasOwnProperty.call(feeds, name)) {
        const feed = feeds[name];
        inputMap.set(
            name,
            new OnnxjsTensor(
                feed.dims, feed.type as OnnxjsTensor.DataType, undefined, undefined,
                feed.data as OnnxjsTensor.NumberType));
      }
    }
    const outputMap = await this.session.run(inputMap);
    const output: SessionHandler.ReturnType = {};
    outputMap.forEach((tensor, name) => {
      output[name] = new Tensor(tensor.type, tensor.data, tensor.dims);
    });
    return output;
  }
  startProfiling(): void {
    this.session.startProfiling();
  }
  endProfiling(): void {
    this.session.endProfiling();
  }
}
