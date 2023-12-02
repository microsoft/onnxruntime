var ortWasmThreaded = (() => {
  var _scriptDir = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== "undefined") _scriptDir = _scriptDir || __filename;
  return function (moduleArg = {}) {
    function d() {
      l.buffer != p.buffer && q();
      return p;
    }
    function u() {
      l.buffer != p.buffer && q();
      return aa;
    }
    function z() {
      l.buffer != p.buffer && q();
      return ba;
    }
    function A() {
      l.buffer != p.buffer && q();
      return ca;
    }
    function da() {
      l.buffer != p.buffer && q();
      return ea;
    }
    var B = moduleArg,
      fa,
      C;
    B.ready = new Promise((a, b) => {
      fa = a;
      C = b;
    });
    ("use strict");
    B.jsepInit = function (a, b, c, e, f, h, k, m) {
      B.Ib = a;
      B.ob = b;
      B.qb = c;
      B.ab = e;
      B.pb = f;
      B.xa = h;
      B.rb = k;
      B.sb = m;
    };
    var ha = Object.assign({}, B),
      ia = "./this.program",
      D = (a, b) => {
        throw b;
      },
      ja = "object" == typeof window,
      E = "function" == typeof importScripts,
      F = "object" == typeof process && "object" == typeof process.versions && "string" == typeof process.versions.node,
      G = B.ENVIRONMENT_IS_PTHREAD || !1,
      H = "";
    function ka(a) {
      return B.locateFile ? B.locateFile(a, H) : H + a;
    }
    var la, I, ma;
    if (F) {
      var fs = require("fs"),
        na = require("path");
      H = E ? na.dirname(H) + "/" : __dirname + "/";
      la = (b, c) => {
        b = b.startsWith("file://") ? new URL(b) : na.normalize(b);
        return fs.readFileSync(b, c ? void 0 : "utf8");
      };
      ma = (b) => {
        b = la(b, !0);
        b.buffer || (b = new Uint8Array(b));
        return b;
      };
      I = (b, c, e, f = !0) => {
        b = b.startsWith("file://") ? new URL(b) : na.normalize(b);
        fs.readFile(b, f ? void 0 : "utf8", (h, k) => {
          h ? e(h) : c(f ? k.buffer : k);
        });
      };
      !B.thisProgram && 1 < process.argv.length && (ia = process.argv[1].replace(/\\/g, "/"));
      process.argv.slice(2);
      D = (b, c) => {
        process.exitCode = b;
        throw c;
      };
      B.inspect = () => "[Emscripten Module object]";
      let a;
      try {
        a = require("worker_threads");
      } catch (b) {
        throw (
          (console.error(
            'The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?',
          ),
          b)
        );
      }
      global.Worker = a.Worker;
    } else if (ja || E)
      E
        ? (H = self.location.href)
        : "undefined" != typeof document && document.currentScript && (H = document.currentScript.src),
        _scriptDir && (H = _scriptDir),
        0 !== H.indexOf("blob:") ? (H = H.substr(0, H.replace(/[?#].*/, "").lastIndexOf("/") + 1)) : (H = ""),
        F ||
          ((la = (a) => {
            var b = new XMLHttpRequest();
            b.open("GET", a, !1);
            b.send(null);
            return b.responseText;
          }),
          E &&
            (ma = (a) => {
              var b = new XMLHttpRequest();
              b.open("GET", a, !1);
              b.responseType = "arraybuffer";
              b.send(null);
              return new Uint8Array(b.response);
            }),
          (I = (a, b, c) => {
            var e = new XMLHttpRequest();
            e.open("GET", a, !0);
            e.responseType = "arraybuffer";
            e.onload = () => {
              200 == e.status || (0 == e.status && e.response) ? b(e.response) : c();
            };
            e.onerror = c;
            e.send(null);
          }));
    F && "undefined" == typeof performance && (global.performance = require("perf_hooks").performance);
    var oa = console.log.bind(console),
      pa = console.error.bind(console);
    F && ((oa = (...a) => fs.writeSync(1, a.join(" ") + "\n")), (pa = (...a) => fs.writeSync(2, a.join(" ") + "\n")));
    var qa = B.print || oa,
      J = B.printErr || pa;
    Object.assign(B, ha);
    ha = null;
    B.thisProgram && (ia = B.thisProgram);
    B.quit && (D = B.quit);
    var K;
    B.wasmBinary && (K = B.wasmBinary);
    var noExitRuntime = B.noExitRuntime || !0;
    "object" != typeof WebAssembly && L("no native wasm support detected");
    var l,
      M,
      ra,
      N = !1,
      P,
      p,
      aa,
      ba,
      ca,
      ea;
    function q() {
      var a = l.buffer;
      B.HEAP8 = p = new Int8Array(a);
      B.HEAP16 = new Int16Array(a);
      B.HEAP32 = ba = new Int32Array(a);
      B.HEAPU8 = aa = new Uint8Array(a);
      B.HEAPU16 = new Uint16Array(a);
      B.HEAPU32 = ca = new Uint32Array(a);
      B.HEAPF32 = new Float32Array(a);
      B.HEAPF64 = ea = new Float64Array(a);
    }
    var sa = B.INITIAL_MEMORY || 16777216;
    5242880 <= sa || L("INITIAL_MEMORY should be larger than STACK_SIZE, was " + sa + "! (STACK_SIZE=5242880)");
    if (G) l = B.wasmMemory;
    else if (B.wasmMemory) l = B.wasmMemory;
    else if (
      ((l = new WebAssembly.Memory({ initial: sa / 65536, maximum: 65536, shared: !0 })),
      !(l.buffer instanceof SharedArrayBuffer))
    )
      throw (
        (J(
          "requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag",
        ),
        F &&
          J(
            "(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)",
          ),
        Error("bad memory"))
      );
    q();
    sa = l.buffer.byteLength;
    var ta = [],
      ua = [],
      va = [],
      wa = 0;
    function xa() {
      return noExitRuntime || 0 < wa;
    }
    var Q = 0,
      ya = null,
      R = null;
    function za() {
      Q++;
      B.monitorRunDependencies && B.monitorRunDependencies(Q);
    }
    function Aa() {
      Q--;
      B.monitorRunDependencies && B.monitorRunDependencies(Q);
      if (0 == Q && (null !== ya && (clearInterval(ya), (ya = null)), R)) {
        var a = R;
        R = null;
        a();
      }
    }
    function L(a) {
      if (B.onAbort) B.onAbort(a);
      a = "Aborted(" + a + ")";
      J(a);
      N = !0;
      P = 1;
      a = new WebAssembly.RuntimeError(a + ". Build with -sASSERTIONS for more info.");
      C(a);
      throw a;
    }
    function Ba(a) {
      return a.startsWith("data:application/octet-stream;base64,");
    }
    var S;
    S = "ort-wasm-simd-threaded.wasm";
    Ba(S) || (S = ka(S));
    function Ca(a) {
      if (a == S && K) return new Uint8Array(K);
      if (ma) return ma(a);
      throw "both async and sync fetching of the wasm failed";
    }
    function Da(a) {
      if (!K && (ja || E)) {
        if ("function" == typeof fetch && !a.startsWith("file://"))
          return fetch(a, { credentials: "same-origin" })
            .then((b) => {
              if (!b.ok) throw "failed to load wasm binary file at '" + a + "'";
              return b.arrayBuffer();
            })
            .catch(() => Ca(a));
        if (I)
          return new Promise((b, c) => {
            I(a, (e) => b(new Uint8Array(e)), c);
          });
      }
      return Promise.resolve().then(() => Ca(a));
    }
    function Ea(a, b, c) {
      return Da(a)
        .then((e) => WebAssembly.instantiate(e, b))
        .then((e) => e)
        .then(c, (e) => {
          J("failed to asynchronously prepare wasm: " + e);
          L(e);
        });
    }
    function Fa(a, b) {
      var c = S;
      return K ||
        "function" != typeof WebAssembly.instantiateStreaming ||
        Ba(c) ||
        c.startsWith("file://") ||
        F ||
        "function" != typeof fetch
        ? Ea(c, a, b)
        : fetch(c, { credentials: "same-origin" }).then((e) =>
            WebAssembly.instantiateStreaming(e, a).then(b, function (f) {
              J("wasm streaming compile failed: " + f);
              J("falling back to ArrayBuffer instantiation");
              return Ea(c, a, b);
            }),
          );
    }
    var T,
      Ga = {
        898348: () => {
          B.jsepRunPromise = new Promise(function (a) {
            B.tb = a;
          });
        },
        898443: (a) => {
          B.tb(a);
        },
        898481: (a) => B.ob(a),
        898514: (a) => B.qb(a),
        898546: (a, b, c) => {
          B.ab(a, b, c, !0);
        },
        898585: (a, b, c) => {
          B.ab(a, b, c);
        },
        898618: (a) => {
          B.xa("Abs", a, void 0);
        },
        898669: (a) => {
          B.xa("Neg", a, void 0);
        },
        898720: (a) => {
          B.xa("Floor", a, void 0);
        },
        898773: (a) => {
          B.xa("Ceil", a, void 0);
        },
        898825: (a) => {
          B.xa("Reciprocal", a, void 0);
        },
        898883: (a) => {
          B.xa("Sqrt", a, void 0);
        },
        898935: (a) => {
          B.xa("Exp", a, void 0);
        },
        898986: (a) => {
          B.xa("Erf", a, void 0);
        },
        899037: (a) => {
          B.xa("Sigmoid", a, void 0);
        },
        899092: (a) => {
          B.xa("Log", a, void 0);
        },
        899143: (a) => {
          B.xa("Sin", a, void 0);
        },
        899194: (a) => {
          B.xa("Cos", a, void 0);
        },
        899245: (a) => {
          B.xa("Tan", a, void 0);
        },
        899296: (a) => {
          B.xa("Asin", a, void 0);
        },
        899348: (a) => {
          B.xa("Acos", a, void 0);
        },
        899400: (a) => {
          B.xa("Atan", a, void 0);
        },
        899452: (a) => {
          B.xa("Sinh", a, void 0);
        },
        899504: (a) => {
          B.xa("Cosh", a, void 0);
        },
        899556: (a) => {
          B.xa("Asinh", a, void 0);
        },
        899609: (a) => {
          B.xa("Acosh", a, void 0);
        },
        899662: (a) => {
          B.xa("Atanh", a, void 0);
        },
        899715: (a) => {
          B.xa("Tanh", a, void 0);
        },
        899767: (a, b, c) => {
          B.xa("ClipV10", a, { min: b, max: c });
        },
        899839: (a) => {
          B.xa("Clip", a, void 0);
        },
        899891: (a, b) => {
          B.xa("Elu", a, { alpha: b });
        },
        899949: (a) => {
          B.xa("Relu", a, void 0);
        },
        900001: (a, b) => {
          B.xa("LeakyRelu", a, { alpha: b });
        },
        900065: (a, b) => {
          B.xa("ThresholdedRelu", a, { alpha: b });
        },
        900135: (a, b) => {
          B.xa("Cast", a, { to: b });
        },
        900193: (a) => {
          B.xa("Add", a, void 0);
        },
        900244: (a) => {
          B.xa("Sub", a, void 0);
        },
        900295: (a) => {
          B.xa("Mul", a, void 0);
        },
        900346: (a) => {
          B.xa("Div", a, void 0);
        },
        900397: (a) => {
          B.xa("Pow", a, void 0);
        },
        900448: (a, b, c, e, f) => {
          B.xa("ReduceMean", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        900612: (a, b, c, e, f) => {
          B.xa("ReduceMax", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        900775: (a, b, c, e, f) => {
          B.xa("ReduceMin", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        900938: (a, b, c, e, f) => {
          B.xa("ReduceProd", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901102: (a, b, c, e, f) => {
          B.xa("ReduceSum", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901265: (a, b, c, e, f) => {
          B.xa("ReduceL1", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901427: (a, b, c, e, f) => {
          B.xa("ReduceL2", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901589: (a, b, c, e, f) => {
          B.xa("ReduceLogSum", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901755: (a, b, c, e, f) => {
          B.xa("ReduceSumSquare", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        901924: (a, b, c, e, f) => {
          B.xa("ReduceLogSumExp", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        902093: (a, b, c) => {
          B.xa("Transpose", a, { perm: b ? Array.from(z().subarray(c >>> 0, (c + b) >>> 0)) : [] });
        },
        902206: (a, b, c, e, f, h, k, m, v, n) => {
          B.xa("Conv", a, {
            format: v ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, k],
            strides: [m],
            w_is_const: () => !!d()[n >>> 0],
          });
        },
        902434: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t) => {
          B.xa("Conv", a, {
            format: g ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c, e],
            group: f,
            kernel_shape: [h, k],
            pads: [m, v, n, r],
            strides: [x, y],
            w_is_const: () => !!d()[t >>> 0],
          });
        },
        902693: (a, b, c, e, f, h, k, m, v, n) => {
          B.xa("Conv", a, {
            format: v ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, k],
            strides: [m],
            w_is_const: () => !!d()[n >>> 0],
          });
        },
        902921: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t) => {
          B.xa("Conv", a, {
            format: g ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c, e],
            group: f,
            kernel_shape: [h, k],
            pads: [m, v, n, r],
            strides: [x, y],
            w_is_const: () => !!d()[t >>> 0],
          });
        },
        903180: (a, b, c, e, f, h, k, m, v, n, r, x, y, g) => {
          B.xa("ConvTranspose", a, {
            format: v ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, k],
            strides: [m],
            wIsConst: () => !!d()[n >>> 0],
            outputPadding: r ? Array.from(z().subarray(x >>> 0, (x + r) >>> 0)) : [],
            outputShape: y ? Array.from(z().subarray(g >>> 0, (g + y) >>> 0)) : [],
          });
        },
        903560: (a, b, c, e, f, h, k, m, v, n, r, x, y) => {
          B.xa("ConvTranspose", a, {
            format: m ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: Array.from(z().subarray(c >>> 0, (c + 2) >>> 0)),
            group: e,
            kernelShape: Array.from(z().subarray(f >>> 0, (f + 2) >>> 0)),
            pads: Array.from(z().subarray(h >>> 0, (h + 4) >>> 0)),
            strides: Array.from(z().subarray(k >>> 0, (k + 2) >>> 0)),
            wIsConst: () => !!d()[v >>> 0],
            outputPadding: 0 < n ? Array.from(z().subarray(r >>> 0, (r + n) >>> 0)) : [],
            outputShape: 0 < x ? Array.from(z().subarray(y >>> 0, (y + x) >>> 0)) : [],
          });
        },
        904083: (a, b, c, e, f, h, k, m, v, n, r, x, y, g) => {
          B.xa("ConvTranspose", a, {
            format: v ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, k],
            strides: [m],
            wIsConst: () => !!d()[n >>> 0],
            outputPadding: r ? Array.from(z().subarray(x >>> 0, (x + r) >>> 0)) : [],
            outputShape: y ? Array.from(z().subarray(g >>> 0, (g + y) >>> 0)) : [],
          });
        },
        904463: (a, b, c, e, f, h, k, m, v, n, r, x, y) => {
          B.xa("ConvTranspose", a, {
            format: m ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: Array.from(z().subarray(c >>> 0, (c + 2) >>> 0)),
            group: e,
            kernelShape: Array.from(z().subarray(f >>> 0, (f + 2) >>> 0)),
            pads: Array.from(z().subarray(h >>> 0, (h + 4) >>> 0)),
            strides: Array.from(z().subarray(k >>> 0, (k + 2) >>> 0)),
            wIsConst: () => !!d()[v >>> 0],
            outputPadding: 0 < n ? Array.from(z().subarray(r >>> 0, (r + n) >>> 0)) : [],
            outputShape: 0 < x ? Array.from(z().subarray(y >>> 0, (y + x) >>> 0)) : [],
          });
        },
        904986: (a, b) => {
          B.xa("GlobalAveragePool", a, { format: b ? "NHWC" : "NCHW" });
        },
        905077: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t, w) => {
          B.xa("AveragePool", a, {
            format: w ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, k],
            kernel_shape: [m, v],
            pads: [n, r, x, y],
            strides: [g, t],
          });
        },
        905361: (a, b) => {
          B.xa("GlobalAveragePool", a, { format: b ? "NHWC" : "NCHW" });
        },
        905452: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t, w) => {
          B.xa("AveragePool", a, {
            format: w ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, k],
            kernel_shape: [m, v],
            pads: [n, r, x, y],
            strides: [g, t],
          });
        },
        905736: (a, b) => {
          B.xa("GlobalMaxPool", a, { format: b ? "NHWC" : "NCHW" });
        },
        905823: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t, w) => {
          B.xa("MaxPool", a, {
            format: w ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, k],
            kernel_shape: [m, v],
            pads: [n, r, x, y],
            strides: [g, t],
          });
        },
        906103: (a, b) => {
          B.xa("GlobalMaxPool", a, { format: b ? "NHWC" : "NCHW" });
        },
        906190: (a, b, c, e, f, h, k, m, v, n, r, x, y, g, t, w) => {
          B.xa("MaxPool", a, {
            format: w ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, k],
            kernel_shape: [m, v],
            pads: [n, r, x, y],
            strides: [g, t],
          });
        },
        906470: (a, b, c, e, f) => {
          B.xa("Gemm", a, { alpha: b, beta: c, transA: e, transB: f });
        },
        906574: (a) => {
          B.xa("MatMul", a, void 0);
        },
        906628: (a, b, c, e) => {
          B.xa("ArgMax", a, { keepDims: !!b, selectLastIndex: !!c, axis: e });
        },
        906736: (a, b, c, e) => {
          B.xa("ArgMin", a, { keepDims: !!b, selectLastIndex: !!c, axis: e });
        },
        906844: (a, b) => {
          B.xa("Softmax", a, { axis: b });
        },
        906907: (a, b) => {
          B.xa("Concat", a, { axis: b });
        },
        906967: (a, b, c, e, f) => {
          B.xa("Split", a, {
            axis: b,
            numOutputs: c,
            splitSizes: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        907112: (a) => {
          B.xa("Expand", a, void 0);
        },
        907166: (a, b) => {
          B.xa("Gather", a, { axis: Number(b) });
        },
        907237: (a, b, c, e, f, h, k, m, v, n, r) => {
          B.xa("Resize", a, {
            antialias: b,
            axes: c ? Array.from(z().subarray(e >>> 0, (e + c) >>> 0)) : [],
            coordinateTransformMode: U(f),
            cubicCoeffA: h,
            excludeOutside: k,
            extrapolationValue: m,
            keepAspectRatioPolicy: U(v),
            mode: U(n),
            nearestMode: U(r),
          });
        },
        907588: (a, b, c, e, f, h, k) => {
          B.xa("Slice", a, {
            starts: b ? Array.from(z().subarray(c >>> 0, (c + b) >>> 0)) : [],
            ends: e ? Array.from(z().subarray(f >>> 0, (f + e) >>> 0)) : [],
            axes: h ? Array.from(z().subarray(k >>> 0, (k + h) >>> 0)) : [],
          });
        },
        907819: (a) => {
          B.xa("Tile", a, void 0);
        },
        907871: (a, b, c) => {
          B.xa("LayerNormalization", a, { axis: Number(b), epsilon: Number(c) });
        },
        907978: (a, b, c) => {
          B.xa("InstanceNormalization", a, { epsilon: b, format: c ? "NHWC" : "NCHW" });
        },
        908092: (a, b, c) => {
          B.xa("InstanceNormalization", a, { epsilon: b, format: c ? "NHWC" : "NCHW" });
        },
        908206: (a) => {
          B.xa("Gelu", a, void 0);
        },
        908258: (a, b) => {
          B.xa("SkipLayerNormalization", a, { epsilon: b });
        },
        908339: (a) => {
          B.rb(a);
        },
        908373: (a, b) => B.sb(a, b),
      };
    function Ha(a) {
      this.name = "ExitStatus";
      this.message = `Program terminated with exit(${a})`;
      this.status = a;
    }
    function Ia(a) {
      a.terminate();
      a.onmessage = () => {};
    }
    function Ja(a) {
      (a = V.Ja[a]) || L();
      V.xb(a);
    }
    function Ka(a) {
      var b = V.lb();
      if (!b) return 6;
      V.Ra.push(b);
      V.Ja[a.Qa] = b;
      b.Qa = a.Qa;
      var c = { cmd: "run", start_routine: a.yb, arg: a.jb, pthread_ptr: a.Qa };
      F && b.unref();
      b.postMessage(c, a.Eb);
      return 0;
    }
    var La = "undefined" != typeof TextDecoder ? new TextDecoder("utf8") : void 0,
      Ma = (a, b, c) => {
        b >>>= 0;
        var e = b + c;
        for (c = b; a[c] && !(c >= e); ) ++c;
        if (16 < c - b && a.buffer && La)
          return La.decode(a.buffer instanceof SharedArrayBuffer ? a.slice(b, c) : a.subarray(b, c));
        for (e = ""; b < c; ) {
          var f = a[b++];
          if (f & 128) {
            var h = a[b++] & 63;
            if (192 == (f & 224)) e += String.fromCharCode(((f & 31) << 6) | h);
            else {
              var k = a[b++] & 63;
              f =
                224 == (f & 240)
                  ? ((f & 15) << 12) | (h << 6) | k
                  : ((f & 7) << 18) | (h << 12) | (k << 6) | (a[b++] & 63);
              65536 > f
                ? (e += String.fromCharCode(f))
                : ((f -= 65536), (e += String.fromCharCode(55296 | (f >> 10), 56320 | (f & 1023))));
            }
          } else e += String.fromCharCode(f);
        }
        return e;
      },
      U = (a, b) => ((a >>>= 0) ? Ma(u(), a, b) : "");
    function Na(a) {
      if (G) return W(1, 1, a);
      P = a;
      if (!xa()) {
        V.zb();
        if (B.onExit) B.onExit(a);
        N = !0;
      }
      D(a, new Ha(a));
    }
    var Pa = (a) => {
        P = a;
        if (G) throw (Oa(a), "unwind");
        Na(a);
      },
      V = {
        Ua: [],
        Ra: [],
        eb: [],
        Ja: {},
        Xa: function () {
          G ? V.nb() : V.mb();
        },
        mb: function () {
          ta.unshift(() => {
            za();
            V.ub(() => Aa());
          });
        },
        nb: function () {
          V.receiveObjectTransfer = V.wb;
          V.threadInitTLS = V.cb;
          V.setExitStatus = V.bb;
          noExitRuntime = !1;
        },
        bb: function (a) {
          P = a;
        },
        Kb: ["$terminateWorker"],
        zb: function () {
          for (var a of V.Ra) Ia(a);
          for (a of V.Ua) Ia(a);
          V.Ua = [];
          V.Ra = [];
          V.Ja = [];
        },
        xb: function (a) {
          var b = a.Qa;
          delete V.Ja[b];
          V.Ua.push(a);
          V.Ra.splice(V.Ra.indexOf(a), 1);
          a.Qa = 0;
          Qa(b);
        },
        wb: function () {},
        cb: function () {
          V.eb.forEach((a) => a());
        },
        vb: (a) =>
          new Promise((b) => {
            a.onmessage = (h) => {
              h = h.data;
              var k = h.cmd;
              if (h.targetThread && h.targetThread != Ra()) {
                var m = V.Ja[h.Jb];
                m
                  ? m.postMessage(h, h.transferList)
                  : J(
                      'Internal error! Worker sent a message "' +
                        k +
                        '" to target pthread ' +
                        h.targetThread +
                        ", but that thread no longer exists!",
                    );
              } else if ("checkMailbox" === k) Sa();
              else if ("spawnThread" === k) Ka(h);
              else if ("cleanupThread" === k) Ja(h.thread);
              else if ("killThread" === k)
                (h = h.thread),
                  (k = V.Ja[h]),
                  delete V.Ja[h],
                  Ia(k),
                  Qa(h),
                  V.Ra.splice(V.Ra.indexOf(k), 1),
                  (k.Qa = 0);
              else if ("cancelThread" === k) V.Ja[h.thread].postMessage({ cmd: "cancel" });
              else if ("loaded" === k) (a.loaded = !0), b(a);
              else if ("alert" === k) alert("Thread " + h.threadId + ": " + h.text);
              else if ("setimmediate" === h.target) a.postMessage(h);
              else if ("callHandler" === k) B[h.handler](...h.args);
              else k && J("worker sent an unknown command " + k);
            };
            a.onerror = (h) => {
              J("worker sent an error! " + h.filename + ":" + h.lineno + ": " + h.message);
              throw h;
            };
            F &&
              (a.on("message", function (h) {
                a.onmessage({ data: h });
              }),
              a.on("error", function (h) {
                a.onerror(h);
              }));
            var c = [],
              e = ["onExit", "onAbort", "print", "printErr"],
              f;
            for (f of e) B.hasOwnProperty(f) && c.push(f);
            a.postMessage({
              cmd: "load",
              handlers: c,
              urlOrBlob: B.mainScriptUrlOrBlob || _scriptDir,
              wasmMemory: l,
              wasmModule: ra,
            });
          }),
        ub: function (a) {
          a();
        },
        ib: function () {
          var a = ka("ort-wasm-simd-threaded.worker.js");
          a = new Worker(a);
          V.Ua.push(a);
        },
        lb: function () {
          0 == V.Ua.length && (V.ib(), V.vb(V.Ua[0]));
          return V.Ua.pop();
        },
      };
    B.PThread = V;
    var Ta = (a) => {
      for (; 0 < a.length; ) a.shift()(B);
    };
    B.establishStackSpace = function () {
      var a = Ra(),
        b = z()[((a + 52) >> 2) >>> 0];
      a = z()[((a + 56) >> 2) >>> 0];
      Ua(b, b - a);
      Va(b);
    };
    function Oa(a) {
      if (G) return W(2, 0, a);
      Pa(a);
    }
    B.invokeEntryPoint = function (a, b) {
      a = Wa.apply(null, [a, b]);
      xa() ? V.bb(a) : Xa(a);
    };
    function Ya(a) {
      this.$a = a - 24;
      this.hb = function (b) {
        A()[((this.$a + 4) >> 2) >>> 0] = b;
      };
      this.gb = function (b) {
        A()[((this.$a + 8) >> 2) >>> 0] = b;
      };
      this.Xa = function (b, c) {
        this.fb();
        this.hb(b);
        this.gb(c);
      };
      this.fb = function () {
        A()[((this.$a + 16) >> 2) >>> 0] = 0;
      };
    }
    var Za = 0,
      $a = 0;
    function ab(a, b, c, e) {
      return G ? W(3, 1, a, b, c, e) : bb(a, b, c, e);
    }
    function bb(a, b, c, e) {
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      if ("undefined" == typeof SharedArrayBuffer)
        return J("Current environment does not support SharedArrayBuffer, pthreads are not available!"), 6;
      var f = [];
      if (G && 0 === f.length) return ab(a, b, c, e);
      a = { yb: c, Qa: a, jb: e, Eb: f };
      return G ? ((a.Gb = "spawnThread"), postMessage(a, f), 0) : Ka(a);
    }
    function cb(a, b, c) {
      return G ? W(4, 1, a, b, c) : 0;
    }
    function db(a, b) {
      if (G) return W(5, 1, a, b);
    }
    var eb = (a) => {
        for (var b = 0, c = 0; c < a.length; ++c) {
          var e = a.charCodeAt(c);
          127 >= e ? b++ : 2047 >= e ? (b += 2) : 55296 <= e && 57343 >= e ? ((b += 4), ++c) : (b += 3);
        }
        return b;
      },
      fb = (a, b, c, e) => {
        c >>>= 0;
        if (!(0 < e)) return 0;
        var f = c;
        e = c + e - 1;
        for (var h = 0; h < a.length; ++h) {
          var k = a.charCodeAt(h);
          if (55296 <= k && 57343 >= k) {
            var m = a.charCodeAt(++h);
            k = (65536 + ((k & 1023) << 10)) | (m & 1023);
          }
          if (127 >= k) {
            if (c >= e) break;
            b[c++ >>> 0] = k;
          } else {
            if (2047 >= k) {
              if (c + 1 >= e) break;
              b[c++ >>> 0] = 192 | (k >> 6);
            } else {
              if (65535 >= k) {
                if (c + 2 >= e) break;
                b[c++ >>> 0] = 224 | (k >> 12);
              } else {
                if (c + 3 >= e) break;
                b[c++ >>> 0] = 240 | (k >> 18);
                b[c++ >>> 0] = 128 | ((k >> 12) & 63);
              }
              b[c++ >>> 0] = 128 | ((k >> 6) & 63);
            }
            b[c++ >>> 0] = 128 | (k & 63);
          }
        }
        b[c >>> 0] = 0;
        return c - f;
      },
      gb = (a, b, c) => fb(a, u(), b, c);
    function hb(a, b) {
      if (G) return W(6, 1, a, b);
    }
    function ib(a, b, c) {
      if (G) return W(7, 1, a, b, c);
    }
    function jb(a, b, c) {
      return G ? W(8, 1, a, b, c) : 0;
    }
    function kb(a, b) {
      if (G) return W(9, 1, a, b);
    }
    function lb(a, b, c) {
      if (G) return W(10, 1, a, b, c);
    }
    function mb(a, b, c, e) {
      if (G) return W(11, 1, a, b, c, e);
    }
    function nb(a, b, c, e) {
      if (G) return W(12, 1, a, b, c, e);
    }
    function ob(a, b, c, e) {
      if (G) return W(13, 1, a, b, c, e);
    }
    function pb(a) {
      if (G) return W(14, 1, a);
    }
    function qb(a, b) {
      if (G) return W(15, 1, a, b);
    }
    function rb(a, b, c) {
      if (G) return W(16, 1, a, b, c);
    }
    var sb = (a) => {
      if (!N)
        try {
          if ((a(), !xa()))
            try {
              G ? Xa(P) : Pa(P);
            } catch (b) {
              b instanceof Ha || "unwind" == b || D(1, b);
            }
        } catch (b) {
          b instanceof Ha || "unwind" == b || D(1, b);
        }
    };
    function tb(a) {
      a >>>= 0;
      "function" === typeof Atomics.Fb &&
        (Atomics.Fb(z(), a >> 2, a).value.then(Sa), (a += 128), Atomics.store(z(), a >> 2, 1));
    }
    B.__emscripten_thread_mailbox_await = tb;
    function Sa() {
      var a = Ra();
      a && (tb(a), sb(() => ub()));
    }
    B.checkMailbox = Sa;
    var X = (a) => 0 === a % 4 && (0 !== a % 100 || 0 === a % 400),
      vb = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
      wb = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    function xb(a, b, c, e, f, h, k, m) {
      return G ? W(17, 1, a, b, c, e, f, h, k, m) : -52;
    }
    function yb(a, b, c, e, f, h, k) {
      if (G) return W(18, 1, a, b, c, e, f, h, k);
    }
    var Ab = (a) => {
        var b = eb(a) + 1,
          c = zb(b);
        c && gb(a, c, b);
        return c;
      },
      Bb = [],
      Cb = (a, b) => {
        Bb.length = 0;
        var c;
        for (b >>= 2; (c = u()[a++ >>> 0]); )
          (b += (105 != c) & b), Bb.push(105 == c ? z()[b >>> 0] : da()[b++ >>> 1]), ++b;
        return Bb;
      },
      Eb = (a) => {
        var b = Db();
        a = a();
        Va(b);
        return a;
      };
    function W(a, b) {
      var c = arguments.length - 2,
        e = arguments;
      return Eb(() => {
        for (var f = Fb(8 * c), h = f >> 3, k = 0; k < c; k++) {
          var m = e[2 + k];
          da()[(h + k) >>> 0] = m;
        }
        return Gb(a, c, f, b);
      });
    }
    var Hb = [],
      Ib = {},
      Kb = () => {
        if (!Jb) {
          var a = {
              USER: "web_user",
              LOGNAME: "web_user",
              PATH: "/",
              PWD: "/",
              HOME: "/home/web_user",
              LANG:
                (("object" == typeof navigator && navigator.languages && navigator.languages[0]) || "C").replace(
                  "-",
                  "_",
                ) + ".UTF-8",
              _: ia || "./this.program",
            },
            b;
          for (b in Ib) void 0 === Ib[b] ? delete a[b] : (a[b] = Ib[b]);
          var c = [];
          for (b in a) c.push(`${b}=${a[b]}`);
          Jb = c;
        }
        return Jb;
      },
      Jb;
    function Lb(a, b) {
      if (G) return W(19, 1, a, b);
      a >>>= 0;
      b >>>= 0;
      var c = 0;
      Kb().forEach(function (e, f) {
        var h = b + c;
        f = A()[((a + 4 * f) >> 2) >>> 0] = h;
        for (h = 0; h < e.length; ++h) d()[(f++ >> 0) >>> 0] = e.charCodeAt(h);
        d()[(f >> 0) >>> 0] = 0;
        c += e.length + 1;
      });
      return 0;
    }
    function Mb(a, b) {
      if (G) return W(20, 1, a, b);
      a >>>= 0;
      b >>>= 0;
      var c = Kb();
      A()[(a >> 2) >>> 0] = c.length;
      var e = 0;
      c.forEach(function (f) {
        e += f.length + 1;
      });
      A()[(b >> 2) >>> 0] = e;
      return 0;
    }
    function Nb(a) {
      return G ? W(21, 1, a) : 52;
    }
    function Ob(a, b, c, e) {
      return G ? W(22, 1, a, b, c, e) : 52;
    }
    function Pb(a, b, c, e, f) {
      return G ? W(23, 1, a, b, c, e, f) : 70;
    }
    var Qb = [null, [], []];
    function Rb(a, b, c, e) {
      if (G) return W(24, 1, a, b, c, e);
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      for (var f = 0, h = 0; h < c; h++) {
        var k = A()[(b >> 2) >>> 0],
          m = A()[((b + 4) >> 2) >>> 0];
        b += 8;
        for (var v = 0; v < m; v++) {
          var n = u()[(k + v) >>> 0],
            r = Qb[a];
          0 === n || 10 === n ? ((1 === a ? qa : J)(Ma(r, 0)), (r.length = 0)) : r.push(n);
        }
        f += m;
      }
      A()[(e >> 2) >>> 0] = f;
      return 0;
    }
    var Tb = () => {
        if ("object" == typeof crypto && "function" == typeof crypto.getRandomValues)
          return (c) => (c.set(crypto.getRandomValues(new Uint8Array(c.byteLength))), c);
        if (F)
          try {
            var a = require("crypto");
            if (a.randomFillSync) return (c) => a.randomFillSync(c);
            var b = a.randomBytes;
            return (c) => (c.set(b(c.byteLength)), c);
          } catch (c) {}
        L("initRandomDevice");
      },
      Ub = (a) => (Ub = Tb())(a),
      Vb = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
      Wb = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    function Xb(a) {
      var b = Array(eb(a) + 1);
      fb(a, b, 0, b.length);
      return b;
    }
    var Yb = (a, b) => {
      d().set(a, b >>> 0);
    };
    function Zb(a, b, c, e) {
      function f(g, t, w) {
        for (g = "number" == typeof g ? g.toString() : g || ""; g.length < t; ) g = w[0] + g;
        return g;
      }
      function h(g, t) {
        return f(g, t, "0");
      }
      function k(g, t) {
        function w(Sb) {
          return 0 > Sb ? -1 : 0 < Sb ? 1 : 0;
        }
        var O;
        0 === (O = w(g.getFullYear() - t.getFullYear())) &&
          0 === (O = w(g.getMonth() - t.getMonth())) &&
          (O = w(g.getDate() - t.getDate()));
        return O;
      }
      function m(g) {
        switch (g.getDay()) {
          case 0:
            return new Date(g.getFullYear() - 1, 11, 29);
          case 1:
            return g;
          case 2:
            return new Date(g.getFullYear(), 0, 3);
          case 3:
            return new Date(g.getFullYear(), 0, 2);
          case 4:
            return new Date(g.getFullYear(), 0, 1);
          case 5:
            return new Date(g.getFullYear() - 1, 11, 31);
          case 6:
            return new Date(g.getFullYear() - 1, 11, 30);
        }
      }
      function v(g) {
        var t = g.Sa;
        for (g = new Date(new Date(g.Ta + 1900, 0, 1).getTime()); 0 < t; ) {
          var w = g.getMonth(),
            O = (X(g.getFullYear()) ? Vb : Wb)[w];
          if (t > O - g.getDate())
            (t -= O - g.getDate() + 1),
              g.setDate(1),
              11 > w ? g.setMonth(w + 1) : (g.setMonth(0), g.setFullYear(g.getFullYear() + 1));
          else {
            g.setDate(g.getDate() + t);
            break;
          }
        }
        w = new Date(g.getFullYear() + 1, 0, 4);
        t = m(new Date(g.getFullYear(), 0, 4));
        w = m(w);
        return 0 >= k(t, g) ? (0 >= k(w, g) ? g.getFullYear() + 1 : g.getFullYear()) : g.getFullYear() - 1;
      }
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      var n = z()[((e + 40) >> 2) >>> 0];
      e = {
        Cb: z()[(e >> 2) >>> 0],
        Bb: z()[((e + 4) >> 2) >>> 0],
        Va: z()[((e + 8) >> 2) >>> 0],
        Za: z()[((e + 12) >> 2) >>> 0],
        Wa: z()[((e + 16) >> 2) >>> 0],
        Ta: z()[((e + 20) >> 2) >>> 0],
        Pa: z()[((e + 24) >> 2) >>> 0],
        Sa: z()[((e + 28) >> 2) >>> 0],
        Lb: z()[((e + 32) >> 2) >>> 0],
        Ab: z()[((e + 36) >> 2) >>> 0],
        Db: n ? U(n) : "",
      };
      c = U(c);
      n = {
        "%c": "%a %b %d %H:%M:%S %Y",
        "%D": "%m/%d/%y",
        "%F": "%Y-%m-%d",
        "%h": "%b",
        "%r": "%I:%M:%S %p",
        "%R": "%H:%M",
        "%T": "%H:%M:%S",
        "%x": "%m/%d/%y",
        "%X": "%H:%M:%S",
        "%Ec": "%c",
        "%EC": "%C",
        "%Ex": "%m/%d/%y",
        "%EX": "%H:%M:%S",
        "%Ey": "%y",
        "%EY": "%Y",
        "%Od": "%d",
        "%Oe": "%e",
        "%OH": "%H",
        "%OI": "%I",
        "%Om": "%m",
        "%OM": "%M",
        "%OS": "%S",
        "%Ou": "%u",
        "%OU": "%U",
        "%OV": "%V",
        "%Ow": "%w",
        "%OW": "%W",
        "%Oy": "%y",
      };
      for (var r in n) c = c.replace(new RegExp(r, "g"), n[r]);
      var x = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        y = "January February March April May June July August September October November December".split(" ");
      n = {
        "%a": (g) => x[g.Pa].substring(0, 3),
        "%A": (g) => x[g.Pa],
        "%b": (g) => y[g.Wa].substring(0, 3),
        "%B": (g) => y[g.Wa],
        "%C": (g) => h(((g.Ta + 1900) / 100) | 0, 2),
        "%d": (g) => h(g.Za, 2),
        "%e": (g) => f(g.Za, 2, " "),
        "%g": (g) => v(g).toString().substring(2),
        "%G": (g) => v(g),
        "%H": (g) => h(g.Va, 2),
        "%I": (g) => {
          g = g.Va;
          0 == g ? (g = 12) : 12 < g && (g -= 12);
          return h(g, 2);
        },
        "%j": (g) => {
          for (var t = 0, w = 0; w <= g.Wa - 1; t += (X(g.Ta + 1900) ? Vb : Wb)[w++]);
          return h(g.Za + t, 3);
        },
        "%m": (g) => h(g.Wa + 1, 2),
        "%M": (g) => h(g.Bb, 2),
        "%n": () => "\n",
        "%p": (g) => (0 <= g.Va && 12 > g.Va ? "AM" : "PM"),
        "%S": (g) => h(g.Cb, 2),
        "%t": () => "\t",
        "%u": (g) => g.Pa || 7,
        "%U": (g) => h(Math.floor((g.Sa + 7 - g.Pa) / 7), 2),
        "%V": (g) => {
          var t = Math.floor((g.Sa + 7 - ((g.Pa + 6) % 7)) / 7);
          2 >= (g.Pa + 371 - g.Sa - 2) % 7 && t++;
          if (t) 53 == t && ((w = (g.Pa + 371 - g.Sa) % 7), 4 == w || (3 == w && X(g.Ta)) || (t = 1));
          else {
            t = 52;
            var w = (g.Pa + 7 - g.Sa - 1) % 7;
            (4 == w || (5 == w && X((g.Ta % 400) - 1))) && t++;
          }
          return h(t, 2);
        },
        "%w": (g) => g.Pa,
        "%W": (g) => h(Math.floor((g.Sa + 7 - ((g.Pa + 6) % 7)) / 7), 2),
        "%y": (g) => (g.Ta + 1900).toString().substring(2),
        "%Y": (g) => g.Ta + 1900,
        "%z": (g) => {
          g = g.Ab;
          var t = 0 <= g;
          g = Math.abs(g) / 60;
          return (t ? "+" : "-") + String("0000" + ((g / 60) * 100 + (g % 60))).slice(-4);
        },
        "%Z": (g) => g.Db,
        "%%": () => "%",
      };
      c = c.replace(/%%/g, "\x00\x00");
      for (r in n) c.includes(r) && (c = c.replace(new RegExp(r, "g"), n[r](e)));
      c = c.replace(/\0\0/g, "%");
      r = Xb(c);
      if (r.length > b) return 0;
      Yb(r, a);
      return r.length - 1;
    }
    function $b(a) {
      try {
        a();
      } catch (b) {
        L(b);
      }
    }
    function ac(a) {
      var b = {},
        c;
      for (c in a)
        (function (e) {
          var f = a[e];
          b[e] =
            "function" == typeof f
              ? function () {
                  bc.push(e);
                  try {
                    return f.apply(null, arguments);
                  } finally {
                    N ||
                      (bc.pop() === e || L(),
                      Y &&
                        1 === Z &&
                        0 === bc.length &&
                        ((Z = 0), (wa += 1), $b(cc), "undefined" != typeof Fibers && Fibers.Mb()));
                  }
                }
              : f;
        })(c);
      return b;
    }
    var Z = 0,
      Y = null,
      dc = 0,
      bc = [],
      ec = {},
      fc = {},
      gc = 0,
      hc = null,
      ic = [];
    function jc() {
      var a = zb(65548),
        b = a + 12;
      A()[(a >> 2) >>> 0] = b;
      A()[((a + 4) >> 2) >>> 0] = b + 65536;
      b = bc[0];
      var c = ec[b];
      void 0 === c && ((c = gc++), (ec[b] = c), (fc[c] = b));
      b = c;
      z()[((a + 8) >> 2) >>> 0] = b;
      return a;
    }
    function kc() {
      var a = z()[((Y + 8) >> 2) >>> 0];
      a = M[fc[a]];
      --wa;
      return a();
    }
    function lc(a) {
      if (!N) {
        if (0 === Z) {
          var b = !1,
            c = !1;
          a((e = 0) => {
            if (!N && ((dc = e), (b = !0), c)) {
              Z = 2;
              $b(() => mc(Y));
              "undefined" != typeof Browser && Browser.Ya.kb && Browser.Ya.resume();
              e = !1;
              try {
                var f = kc();
              } catch (m) {
                (f = m), (e = !0);
              }
              var h = !1;
              if (!Y) {
                var k = hc;
                k && ((hc = null), (e ? k.reject : k.resolve)(f), (h = !0));
              }
              if (e && !h) throw f;
            }
          });
          c = !0;
          b ||
            ((Z = 1),
            (Y = jc()),
            "undefined" != typeof Browser && Browser.Ya.kb && Browser.Ya.pause(),
            $b(() => nc(Y)));
        } else 2 === Z ? ((Z = 0), $b(oc), pc(Y), (Y = null), ic.forEach((e) => sb(e))) : L(`invalid state: ${Z}`);
        return dc;
      }
    }
    function qc(a) {
      return lc((b) => {
        a().then(b);
      });
    }
    V.Xa();
    var rc = [null, Na, Oa, ab, cb, db, hb, ib, jb, kb, lb, mb, nb, ob, pb, qb, rb, xb, yb, Lb, Mb, Nb, Ob, Pb, Rb],
      uc = {
        r: function (a, b, c) {
          return qc(async () => {
            await B.pb(a, b, c);
          });
        },
        b: function (a, b, c) {
          a >>>= 0;
          new Ya(a).Xa(b >>> 0, c >>> 0);
          Za = a;
          $a++;
          throw Za;
        },
        O: function (a) {
          sc(a >>> 0, !E, 1, !ja, 131072, !1);
          V.cb();
        },
        m: function (a) {
          a >>>= 0;
          G ? postMessage({ cmd: "cleanupThread", thread: a }) : Ja(a);
        },
        J: bb,
        i: cb,
        U: db,
        G: hb,
        I: ib,
        V: jb,
        S: kb,
        K: lb,
        R: mb,
        q: nb,
        H: ob,
        E: pb,
        T: qb,
        F: rb,
        Y: () => !0,
        C: function (a, b) {
          a >>>= 0;
          a == b >>> 0
            ? setTimeout(() => Sa())
            : G
            ? postMessage({ targetThread: a, cmd: "checkMailbox" })
            : (a = V.Ja[a]) && a.postMessage({ cmd: "checkMailbox" });
        },
        M: function () {
          return -1;
        },
        N: tb,
        X: function (a) {
          F && V.Ja[a >>> 0].ref();
        },
        u: function (a, b, c) {
          a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
          c >>>= 0;
          a = new Date(1e3 * a);
          z()[(c >> 2) >>> 0] = a.getUTCSeconds();
          z()[((c + 4) >> 2) >>> 0] = a.getUTCMinutes();
          z()[((c + 8) >> 2) >>> 0] = a.getUTCHours();
          z()[((c + 12) >> 2) >>> 0] = a.getUTCDate();
          z()[((c + 16) >> 2) >>> 0] = a.getUTCMonth();
          z()[((c + 20) >> 2) >>> 0] = a.getUTCFullYear() - 1900;
          z()[((c + 24) >> 2) >>> 0] = a.getUTCDay();
          a = ((a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864e5) | 0;
          z()[((c + 28) >> 2) >>> 0] = a;
        },
        v: function (a, b, c) {
          a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
          c >>>= 0;
          a = new Date(1e3 * a);
          z()[(c >> 2) >>> 0] = a.getSeconds();
          z()[((c + 4) >> 2) >>> 0] = a.getMinutes();
          z()[((c + 8) >> 2) >>> 0] = a.getHours();
          z()[((c + 12) >> 2) >>> 0] = a.getDate();
          z()[((c + 16) >> 2) >>> 0] = a.getMonth();
          z()[((c + 20) >> 2) >>> 0] = a.getFullYear() - 1900;
          z()[((c + 24) >> 2) >>> 0] = a.getDay();
          b = ((X(a.getFullYear()) ? vb : wb)[a.getMonth()] + a.getDate() - 1) | 0;
          z()[((c + 28) >> 2) >>> 0] = b;
          z()[((c + 36) >> 2) >>> 0] = -(60 * a.getTimezoneOffset());
          b = new Date(a.getFullYear(), 6, 1).getTimezoneOffset();
          var e = new Date(a.getFullYear(), 0, 1).getTimezoneOffset();
          a = (b != e && a.getTimezoneOffset() == Math.min(e, b)) | 0;
          z()[((c + 32) >> 2) >>> 0] = a;
        },
        w: function (a) {
          a >>>= 0;
          var b = new Date(
              z()[((a + 20) >> 2) >>> 0] + 1900,
              z()[((a + 16) >> 2) >>> 0],
              z()[((a + 12) >> 2) >>> 0],
              z()[((a + 8) >> 2) >>> 0],
              z()[((a + 4) >> 2) >>> 0],
              z()[(a >> 2) >>> 0],
              0,
            ),
            c = z()[((a + 32) >> 2) >>> 0],
            e = b.getTimezoneOffset(),
            f = new Date(b.getFullYear(), 6, 1).getTimezoneOffset(),
            h = new Date(b.getFullYear(), 0, 1).getTimezoneOffset(),
            k = Math.min(h, f);
          0 > c
            ? (z()[((a + 32) >> 2) >>> 0] = Number(f != h && k == e))
            : 0 < c != (k == e) && ((f = Math.max(h, f)), b.setTime(b.getTime() + 6e4 * ((0 < c ? k : f) - e)));
          z()[((a + 24) >> 2) >>> 0] = b.getDay();
          c = ((X(b.getFullYear()) ? vb : wb)[b.getMonth()] + b.getDate() - 1) | 0;
          z()[((a + 28) >> 2) >>> 0] = c;
          z()[(a >> 2) >>> 0] = b.getSeconds();
          z()[((a + 4) >> 2) >>> 0] = b.getMinutes();
          z()[((a + 8) >> 2) >>> 0] = b.getHours();
          z()[((a + 12) >> 2) >>> 0] = b.getDate();
          z()[((a + 16) >> 2) >>> 0] = b.getMonth();
          z()[((a + 20) >> 2) >>> 0] = b.getYear();
          a = b.getTime() / 1e3;
          return (
            tc(
              ((T = a),
              1 <= +Math.abs(T)
                ? 0 < T
                  ? +Math.floor(T / 4294967296) >>> 0
                  : ~~+Math.ceil((T - +(~~T >>> 0)) / 4294967296) >>> 0
                : 0),
            ),
            a >>> 0
          );
        },
        s: xb,
        t: yb,
        A: function (a, b, c) {
          function e(n) {
            return (n = n.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? n[1] : "GMT";
          }
          a >>>= 0;
          b >>>= 0;
          c >>>= 0;
          var f = new Date().getFullYear(),
            h = new Date(f, 0, 1),
            k = new Date(f, 6, 1);
          f = h.getTimezoneOffset();
          var m = k.getTimezoneOffset(),
            v = Math.max(f, m);
          A()[(a >> 2) >>> 0] = 60 * v;
          z()[(b >> 2) >>> 0] = Number(f != m);
          a = e(h);
          b = e(k);
          a = Ab(a);
          b = Ab(b);
          m < f
            ? ((A()[(c >> 2) >>> 0] = a), (A()[((c + 4) >> 2) >>> 0] = b))
            : ((A()[(c >> 2) >>> 0] = b), (A()[((c + 4) >> 2) >>> 0] = a));
        },
        d: () => {
          L("");
        },
        c: function (a, b, c) {
          a >>>= 0;
          b = Cb(b >>> 0, c >>> 0);
          return Ga[a].apply(null, b);
        },
        l: function (a, b, c) {
          a >>>= 0;
          b = Cb(b >>> 0, c >>> 0);
          return Ga[a].apply(null, b);
        },
        n: function () {},
        j: function () {
          return Date.now();
        },
        W: () => {
          wa += 1;
          throw "unwind";
        },
        D: function () {
          return 4294901760;
        },
        f: () => performance.timeOrigin + performance.now(),
        g: function () {
          return F ? require("os").cpus().length : navigator.hardwareConcurrency;
        },
        L: function (a, b, c, e) {
          V.Hb = b >>> 0;
          Hb.length = c;
          b = (e >>> 0) >> 3;
          for (e = 0; e < c; e++) Hb[e] = da()[(b + e) >>> 0];
          return (0 > a ? Ga[-a - 1] : rc[a]).apply(null, Hb);
        },
        z: function (a) {
          a >>>= 0;
          var b = u().length;
          if (a <= b || 4294901760 < a) return !1;
          for (var c = 1; 4 >= c; c *= 2) {
            var e = b * (1 + 0.2 / c);
            e = Math.min(e, a + 100663296);
            var f = Math;
            e = Math.max(a, e);
            a: {
              f = (f.min.call(f, 4294901760, e + ((65536 - (e % 65536)) % 65536)) - l.buffer.byteLength + 65535) >>> 16;
              try {
                l.grow(f);
                q();
                var h = 1;
                break a;
              } catch (k) {}
              h = void 0;
            }
            if (h) return !0;
          }
          return !1;
        },
        P: Lb,
        Q: Mb,
        k: Pa,
        h: Nb,
        p: Ob,
        x: Pb,
        o: Rb,
        y: function (a, b) {
          a >>>= 0;
          b >>>= 0;
          Ub(u().subarray(a >>> 0, (a + b) >>> 0));
          return 0;
        },
        a: l || B.wasmMemory,
        B: Zb,
        e: function (a, b, c, e) {
          return Zb(a >>> 0, b >>> 0, c >>> 0, e >>> 0);
        },
      };
    (function () {
      function a(c, e) {
        c = c.exports;
        c = ac(c);
        M = c = vc(c);
        V.eb.push(M.wa);
        ua.unshift(M.Z);
        ra = e;
        Aa();
        return c;
      }
      var b = { a: uc };
      za();
      if (B.instantiateWasm)
        try {
          return B.instantiateWasm(b, a);
        } catch (c) {
          J("Module.instantiateWasm callback failed with error: " + c), C(c);
        }
      Fa(b, function (c) {
        a(c.instance, c.module);
      }).catch(C);
      return {};
    })();
    B._OrtInit = (a, b) => (B._OrtInit = M._)(a, b);
    B._OrtGetLastError = (a, b) => (B._OrtGetLastError = M.$)(a, b);
    B._OrtCreateSessionOptions = (a, b, c, e, f, h, k, m, v, n) =>
      (B._OrtCreateSessionOptions = M.aa)(a, b, c, e, f, h, k, m, v, n);
    B._OrtAppendExecutionProvider = (a, b) => (B._OrtAppendExecutionProvider = M.ba)(a, b);
    B._OrtAddSessionConfigEntry = (a, b, c) => (B._OrtAddSessionConfigEntry = M.ca)(a, b, c);
    B._OrtReleaseSessionOptions = (a) => (B._OrtReleaseSessionOptions = M.da)(a);
    B._OrtCreateSession = (a, b, c) => (B._OrtCreateSession = M.ea)(a, b, c);
    B._OrtReleaseSession = (a) => (B._OrtReleaseSession = M.fa)(a);
    B._OrtGetInputOutputCount = (a, b, c) => (B._OrtGetInputOutputCount = M.ga)(a, b, c);
    B._OrtGetInputName = (a, b) => (B._OrtGetInputName = M.ha)(a, b);
    B._OrtGetOutputName = (a, b) => (B._OrtGetOutputName = M.ia)(a, b);
    B._OrtFree = (a) => (B._OrtFree = M.ja)(a);
    B._OrtCreateTensor = (a, b, c, e, f) => (B._OrtCreateTensor = M.ka)(a, b, c, e, f);
    B._OrtGetTensorData = (a, b, c, e, f) => (B._OrtGetTensorData = M.la)(a, b, c, e, f);
    B._OrtReleaseTensor = (a) => (B._OrtReleaseTensor = M.ma)(a);
    B._OrtCreateRunOptions = (a, b, c, e) => (B._OrtCreateRunOptions = M.na)(a, b, c, e);
    B._OrtAddRunConfigEntry = (a, b, c) => (B._OrtAddRunConfigEntry = M.oa)(a, b, c);
    B._OrtReleaseRunOptions = (a) => (B._OrtReleaseRunOptions = M.pa)(a);
    B._OrtRun = (a, b, c, e, f, h, k, m) => (B._OrtRun = M.qa)(a, b, c, e, f, h, k, m);
    B._OrtEndProfiling = (a) => (B._OrtEndProfiling = M.ra)(a);
    B._JsepOutput = (a, b, c) => (B._JsepOutput = M.sa)(a, b, c);
    var Ra = (B._pthread_self = () => (Ra = B._pthread_self = M.ta)()),
      zb = (B._malloc = (a) => (zb = B._malloc = M.ua)(a)),
      pc = (B._free = (a) => (pc = B._free = M.va)(a));
    B.__emscripten_tls_init = () => (B.__emscripten_tls_init = M.wa)();
    var sc = (B.__emscripten_thread_init = (a, b, c, e, f, h) =>
      (sc = B.__emscripten_thread_init = M.ya)(a, b, c, e, f, h));
    B.__emscripten_thread_crashed = () => (B.__emscripten_thread_crashed = M.za)();
    var Gb = (a, b, c, e) => (Gb = M.Aa)(a, b, c, e),
      Qa = (a) => (Qa = M.Ba)(a),
      Xa = (B.__emscripten_thread_exit = (a) => (Xa = B.__emscripten_thread_exit = M.Ca)(a)),
      ub = (B.__emscripten_check_mailbox = () => (ub = B.__emscripten_check_mailbox = M.Da)()),
      tc = (a) => (tc = M.Ea)(a),
      Ua = (a, b) => (Ua = M.Fa)(a, b),
      Db = () => (Db = M.Ga)(),
      Va = (a) => (Va = M.Ha)(a),
      Fb = (a) => (Fb = M.Ia)(a),
      Wa = (B.dynCall_ii = (a, b) => (Wa = B.dynCall_ii = M.Ka)(a, b)),
      nc = (a) => (nc = M.La)(a),
      cc = () => (cc = M.Ma)(),
      mc = (a) => (mc = M.Na)(a),
      oc = () => (oc = M.Oa)();
    B.___start_em_js = 908408;
    B.___stop_em_js = 908569;
    function vc(a) {
      a = Object.assign({}, a);
      var b = (e) => () => e() >>> 0,
        c = (e) => (f) => e(f) >>> 0;
      a.__errno_location = b(a.__errno_location);
      a.pthread_self = b(a.pthread_self);
      a.malloc = c(a.malloc);
      a.stackSave = b(a.stackSave);
      a.stackAlloc = c(a.stackAlloc);
      return a;
    }
    B.keepRuntimeAlive = xa;
    B.wasmMemory = l;
    B.stackAlloc = Fb;
    B.stackSave = Db;
    B.stackRestore = Va;
    B.UTF8ToString = U;
    B.stringToUTF8 = gb;
    B.lengthBytesUTF8 = eb;
    B.ExitStatus = Ha;
    B.PThread = V;
    var wc;
    R = function xc() {
      wc || yc();
      wc || (R = xc);
    };
    function yc() {
      function a() {
        if (!wc && ((wc = !0), (B.calledRun = !0), !N)) {
          G || Ta(ua);
          fa(B);
          if (B.onRuntimeInitialized) B.onRuntimeInitialized();
          if (!G) {
            if (B.postRun)
              for ("function" == typeof B.postRun && (B.postRun = [B.postRun]); B.postRun.length; ) {
                var b = B.postRun.shift();
                va.unshift(b);
              }
            Ta(va);
          }
        }
      }
      if (!(0 < Q))
        if (G) fa(B), G || Ta(ua), startWorker(B);
        else {
          if (B.preRun)
            for ("function" == typeof B.preRun && (B.preRun = [B.preRun]); B.preRun.length; )
              ta.unshift(B.preRun.shift());
          Ta(ta);
          0 < Q ||
            (B.setStatus
              ? (B.setStatus("Running..."),
                setTimeout(function () {
                  setTimeout(function () {
                    B.setStatus("");
                  }, 1);
                  a();
                }, 1))
              : a());
        }
    }
    if (B.preInit)
      for ("function" == typeof B.preInit && (B.preInit = [B.preInit]); 0 < B.preInit.length; ) B.preInit.pop()();
    yc();

    return moduleArg.ready;
  };
})();
if (typeof exports === "object" && typeof module === "object") module.exports = ortWasmThreaded;
else if (typeof define === "function" && define["amd"]) define([], () => ortWasmThreaded);
