// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

 {InferenceSession, OnnxValue}    'onnxruntime-common';

     SessionOptions = InferenceSession.SessionOptions;
     FeedsType = {
  [name: string]: OnnxValue;
};
     FetchesType = {
  [name: string]: OnnxValue|null;
};
     ReturnType = {
  [name: string]: OnnxValue;
};
     RunOptions = InferenceSession.RunOptions;


/**
 * Binding exports a simple synchronized inference session object wrap.
 */
                      Binding {
                    InferenceSession {
    loadModel(modelPath: string, options: SessionOptions):    ;
    loadModel(buffer: ArrayBuffer, byteOffset: number, byteLength: number, options: SessionOptions):    ;

             inputNames: string[];
             outputNames: string[];

    run(feeds: FeedsType, fetches: FetchesType, options: RunOptions): ReturnType;
  }

                   InferenceSessionConstructor {
       (): InferenceSession;
  }
}

// export 
             binding =
    // eslint-disable-next-line typescript-eslint/no-require-imports, typescript-eslint/no-var-requires
            (`../bin/napi-v3/${process.platform}/${process.arch}/onnxruntime_binding.node`) 
    // eslint-disable-next-line typescript-eslint/naming-convention
    {InferenceSession: Binding.InferenceSessionConstructor};
