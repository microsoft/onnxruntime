const Tensor = require('onnxruntime').Tensor;

//
// create a [2x3x4] float tensor
//
const buffer01 = new Float32Array(24);
buffer01[0] = 0.1; // fill buffer data
const tensor01 = new Tensor('float32', buffer01, [2, 3, 4]);
// type 'float32' can be omitted and the type is inferred from data
const tensor01_B = new Tensor(buffer01, [2, 3, 4]);

//
// create a [1x2] boolean tensor
//
const buffer02 = new Uint8Array(2);
buffer02[0] = 1;  // true
buffer02[1] = 0;  // false
const tensor02 = new Tensor('bool', buffer02, [1, 2]); // type 'bool' cannot omit as both 'bool' and 'uint8' uses Uint8Array.

//
// create a scaler float64 tensor
//
const tensor03 = new Tensor(new Float64Array(1), []);
tensor03.data[0] = 1.0; // setting data after tensor is created is allowed

//
// create a one-dimension tensor
//
const tensor04 = new Tensor(new Float32Array(100), [100]);
const tensor04_B = new Tensor(new Float32Array(100));  // dims can be omitted if it is a 1-D tensor. tensor04.dims = [100]

//
// create a [1x2] string tensor
//
const tensor05 = new Tensor('string', ['a', 'b'], [1, 2]);
const tensor05_B = new Tensor(['a', 'b'], [1, 2]); // type 'string' can be omitted

//
// !!! BAD USAGES !!!
// followings are bad usages that may cause an error to be thrown. try not to make these mistakes.
//

// create from mismatched TypedArray
try {
    const tensor = new Tensor('float64', new Float32Array(100)); // 'float64' must use with Float64Array as data.
} catch{ }

// bad dimension (negative value)
try {
    const tensor = new Tensor(new Float32Array(100), [1, 2, -1]); // negative dims is not allowed.
} catch{ }

// size mismatch (scalar size should be 1)
try {
    const tensor = new Tensor(new Float32Array(0), []);
} catch{ }

// size mismatch (5 * 6 != 40)
try {
    const tensor = new Tensor(new Float32Array(40), [5, 6]);
} catch{ }
