import * as tf from '@tensorflow/tfjs';
import loadImage from 'blueimp-load-image';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import * as onnx from 'onnxjs';

// TODO this benchmark is extended based on the old resnet benchmark. Merge these 2 into a generic benchmarking tool.

const IMAGE_URLS = [
    { name: 'cat', url: 'https://i.imgur.com/CzXTtJV.jpg' },
  ];
const SERVER_BASE_PATH = '/base';

const BackendMapping = {
  'ONNX.js' : {
    'webgl': 'GPU-webgl',
  },
  'TensorFlow.js': {
    'webgl': 'GPU-webgl',
  },
}

const BenchmarkImageNetData = [
    {
        model: 'superResolution',
        imageSize: 224,
        testCases: [
            {
                impl: 'TensorFlow.js',
                modelPath: ``,
                backends: [ 'webgl'],
                inputs: IMAGE_URLS,
                webglLevels: [2]
            },
            {
                impl: 'ONNX.js',
                modelPath: `${SERVER_BASE_PATH}/data/model-onnx/super_resolution.onnx`,
                backends: [  'webgl'],
                inputs: IMAGE_URLS,
                webglLevels: [2]
            },
        ]
    },
];
class ImageLoader {
    constructor(imageWidth, imageHeight) {
        this.canvas = document.createElement('canvas');
        this.canvas.width = imageWidth;
        this.canvas.height = imageHeight;
        this.ctx = this.canvas.getContext('2d');
    }
    async getImageData(url) {
        await this.loadImageAsync(url);
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return imageData;
    }
    loadImageAsync(url) {
        return new Promise((resolve, reject)=>{
            this.loadImageCb(url, ()=>{
                resolve();
            });
        });
    }
    loadImageCb(url, cb) {
        loadImage(
            url,
            img => {
                if (img.type === 'error') {
                    throw `Could not load image: ${url}`;
                } else {
                    // load image data onto input canvas
                    this.ctx.drawImage(img, 0, 0)
                    //console.log(`image was loaded`);
                    window.setTimeout(() => {  cb();  }, 0);
                }
            },
            {
                maxWidth: this.canvas.width,
                maxHeight: this.canvas.height,
                cover: true,
                crop: true,
                canvas: true,
                crossOrigin: 'Anonymous'
            }
        );
    }
}
function createBenchmark(name) {
    switch (name) {
        case 'TensorFlow.js': return new TensorFlowResnetBenchmark();
        case 'ONNX.js': return new OnnxJsResnetBenchmark();
    }
}
async function runBenchmark(benchmarkData, backend, imageSize) {
    console.log(`runBenchmark is being called with ${benchmarkData.impl}, ${backend}, ${imageSize}`)
    const impl = createBenchmark(benchmarkData.impl);
    console.log(`impl: ${benchmarkData.impl}, modelPath: ${benchmarkData.modelPath}`)
    await impl.init(backend, benchmarkData.modelPath, imageSize);
    const imageLoader = new ImageLoader(imageSize, imageSize);
    const durations = [];
    for(const input of benchmarkData.inputs) {
        console.log(`Running ${input.name} for ${runIteration} iterations.`)
        const imageData = await imageLoader.getImageData(input.url);
        for(let i = 0 ; i < runIteration; i++) {
          const outputData = await impl.runModel(imageData.data);
          durations.push(impl.duration);
        }
    }
    if(profile) {
      impl.endProfiling();
    }
    durations.shift();
    const sum = durations.reduce((a,b)=>a+b);
    const avg = sum / durations.length;
    console.log(`avg duration: ${avg}`);
    return {
        framework: benchmarkData.impl,
        backend: BackendMapping[benchmarkData.impl][backend],
        duration: avg
    };
}

class TensorFlowResnetBenchmark {
    async init(backend, modelPath, imageSize) {
        this.imageSize = imageSize;
        tf.disposeVariables();
        tf.env().set('WEBGL_PACK', pack_texture);

        console.log(`Pack mode enabled: ${tf.env().getBool('WEBGL_PACK')}`);
        if(backend) {
            console.log(`Setting the backend to ${backend}`);
            tf.setBackend(backend);
        }
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: [224,224,1],
            kernelSize: [5,5],
            padding: 'same',
            filters: 64,
            strides: 1,
            activation: 'relu',
            useBias: true,
            biasInitializer: 'ones',
            kernelInitializer: 'varianceScaling'
          }));
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            padding: 'same',
            filters: 64,
            strides: 1,
            activation: 'relu',
            useBias: true,
            biasInitializer: 'ones',
            kernelInitializer: 'varianceScaling'
          }));
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            padding: 'same',
            filters: 32,
            strides: 1,
            activation: 'relu',
            useBias: true,
            biasInitializer: 'ones',
            kernelInitializer: 'varianceScaling'
          }));
        model.add(tf.layers.conv2d({
            kernelSize: 3,
            padding: 'same',
            filters: 9,
            strides: 1,
            activation: 'relu',
            useBias: true,
            biasInitializer: 'ones',
            kernelInitializer: 'varianceScaling'
          }));
        model.add(tf.layers.reshape({targetShape: [1,3,3,224,224]}))
        model.add(tf.layers.permute({dims: [1, 4, 2, 5, 3]}));
        model.add(tf.layers.reshape({targetShape: [1,672,672]}));

        this.model = model;
    }
    async runModel(data) {
        const inputTensor = this.preprocess(data, this.imageSize, this.imageSize);
        const start = performance.now();
        const output = await this.model.predict(inputTensor);
        const outputData = output.dataSync();
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
    preprocess(data, width, height) {
        // data processing
        const dataTensor = ndarray(new Float32Array(data), [width, height, 1])
        const dataProcessedTensor = ndarray(new Float32Array(width * height * 1), [1, width, height, 1])

        ops.subseq(dataTensor.pick(null, null, 2), 103.939)
        ops.subseq(dataTensor.pick(null, null, 1), 116.779)
        ops.subseq(dataTensor.pick(null, null, 0), 123.68)
        ops.assign(dataProcessedTensor.pick(0, null, null, 0), dataTensor.pick(null, null, 2))
        ops.assign(dataProcessedTensor.pick(0, null, null, 1), dataTensor.pick(null, null, 1))
        ops.assign(dataProcessedTensor.pick(0, null, null, 2), dataTensor.pick(null, null, 0))

        return tf.tensor(dataProcessedTensor.data, dataProcessedTensor.shape);
    }
    endProfiling() {
    }
}
class OnnxJsResnetBenchmark {
    async init(backend, modelPath, imageSize) {
        onnx.backend.webgl.pack = pack_texture;
        console.log(`Pack mode enabled: ${onnx.backend.webgl.pack}`);

        this.imageSize = imageSize;
        const hint = {backendHint: backend };
        const profilerConfig = profile ? {maxNumberEvents: 65536} : undefined;
        const loggerConfig = profile ? {logLevel: 'verbose'} : undefined;
        const sessionConfig = {hint, profiler: profilerConfig, logger: loggerConfig};

        this.model = new onnx.InferenceSession(sessionConfig);
        if (profile) {
          this.model.startProfiling();
        }

        await this.model.loadModel(modelPath);
    }
    async runModel(data) {
        const preprocessedData = this.preprocess(data, this.imageSize, this.imageSize);
        const start = performance.now();

        const outputMap = await this.model.run([preprocessedData]);
        const outputData = outputMap.values().next().value.data;
        const stop = performance.now();
        this.duration = stop - start;
        console.log(`Duration:${this.duration}ms`);
        return outputData;
    }
    delay(ms)
    {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
    preprocess(data, width, height) {
      // data processing
      const dataTensor = ndarray(new Float32Array(data), [width, height, 1]);
      const dataProcessedTensor = ndarray(new Float32Array(width * height * 1), [1, 1, width, height]);

      ops.divseq(dataTensor, 128.0);
      ops.subseq(dataTensor, 1.0);

      ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
      ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
      ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));

      const tensor = new onnx.Tensor(dataProcessedTensor.data, 'float32', [1, 1, width, height]);
      return tensor;
    }
    endProfiling() {
      this.model.endProfiling();
    }
}
const results = [];
const browser = __karma__.config.browser[0];
const profile = __karma__.config.profile;
const pack_texture = __karma__.config.usePackedGlTexture;
const runIteration = __karma__.config.runIteration;

console.log(`browser: ${browser}`)
describe('super resolution Tests', ()=> {
    for(const modelTestcase of BenchmarkImageNetData) {
        describe(`model: ${modelTestcase.model}`, ()=> {
            for(const testCase of modelTestcase.testCases) {
                for(const backend of testCase.backends) {
                    it(`testCase:${testCase.impl} ${backend}`,
                        async function() {
                            // rule 1: if only supports WebGL 2 then skip Edge
                            if(browser.startsWith('Edge') && backend === 'webgl' && !testCase.webglLevels.includes(1)) {
                                this.skip();
                                return;
                            }
                            results.push(await runBenchmark(testCase, backend, modelTestcase.imageSize));
                        }
                    );
                }
            }
        });
    }
    after('printing results', ()=> {
        console.log(JSON.stringify(results));
    });
});