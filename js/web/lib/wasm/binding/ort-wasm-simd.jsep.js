var ortWasm = (() => {
  var _scriptDir = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== "undefined") _scriptDir = _scriptDir || __filename;
  return function (moduleArg = {}) {
    var d = moduleArg,
      aa,
      k;
    d.ready = new Promise((a, b) => {
      aa = a;
      k = b;
    });
    ("use strict");
    d.jsepInit = function (a, b, c, e, f, h, l, m) {
      d.Sa = a;
      d.Ea = b;
      d.Ga = c;
      d.Ca = e;
      d.Fa = f;
      d.la = h;
      d.Ha = l;
      d.Ia = m;
    };
    var ba = Object.assign({}, d),
      t = "./this.program",
      x = (a, b) => {
        throw b;
      },
      ca = "object" == typeof window,
      y = "function" == typeof importScripts,
      z = "object" == typeof process && "object" == typeof process.versions && "string" == typeof process.versions.node,
      A = "",
      da,
      B,
      C;
    if (z) {
      var fs = require("fs"),
        ea = require("path");
      A = y ? ea.dirname(A) + "/" : __dirname + "/";
      da = (a, b) => {
        a = a.startsWith("file://") ? new URL(a) : ea.normalize(a);
        return fs.readFileSync(a, b ? void 0 : "utf8");
      };
      C = (a) => {
        a = da(a, !0);
        a.buffer || (a = new Uint8Array(a));
        return a;
      };
      B = (a, b, c, e = !0) => {
        a = a.startsWith("file://") ? new URL(a) : ea.normalize(a);
        fs.readFile(a, e ? void 0 : "utf8", (f, h) => {
          f ? c(f) : b(e ? h.buffer : h);
        });
      };
      !d.thisProgram && 1 < process.argv.length && (t = process.argv[1].replace(/\\/g, "/"));
      process.argv.slice(2);
      x = (a, b) => {
        process.exitCode = a;
        throw b;
      };
      d.inspect = () => "[Emscripten Module object]";
    } else if (ca || y)
      y
        ? (A = self.location.href)
        : "undefined" != typeof document && document.currentScript && (A = document.currentScript.src),
        _scriptDir && (A = _scriptDir),
        0 !== A.indexOf("blob:") ? (A = A.substr(0, A.replace(/[?#].*/, "").lastIndexOf("/") + 1)) : (A = ""),
        (da = (a) => {
          var b = new XMLHttpRequest();
          b.open("GET", a, !1);
          b.send(null);
          return b.responseText;
        }),
        y &&
          (C = (a) => {
            var b = new XMLHttpRequest();
            b.open("GET", a, !1);
            b.responseType = "arraybuffer";
            b.send(null);
            return new Uint8Array(b.response);
          }),
        (B = (a, b, c) => {
          var e = new XMLHttpRequest();
          e.open("GET", a, !0);
          e.responseType = "arraybuffer";
          e.onload = () => {
            200 == e.status || (0 == e.status && e.response) ? b(e.response) : c();
          };
          e.onerror = c;
          e.send(null);
        });
    var fa = d.print || console.log.bind(console),
      D = d.printErr || console.error.bind(console);
    Object.assign(d, ba);
    ba = null;
    d.thisProgram && (t = d.thisProgram);
    d.quit && (x = d.quit);
    var E;
    d.wasmBinary && (E = d.wasmBinary);
    var noExitRuntime = d.noExitRuntime || !0;
    "object" != typeof WebAssembly && F("no native wasm support detected");
    var G,
      I,
      J = !1,
      K,
      L,
      M,
      N,
      O,
      ha;
    function ia() {
      var a = G.buffer;
      d.HEAP8 = L = new Int8Array(a);
      d.HEAP16 = new Int16Array(a);
      d.HEAP32 = N = new Int32Array(a);
      d.HEAPU8 = M = new Uint8Array(a);
      d.HEAPU16 = new Uint16Array(a);
      d.HEAPU32 = O = new Uint32Array(a);
      d.HEAPF32 = new Float32Array(a);
      d.HEAPF64 = ha = new Float64Array(a);
    }
    var ja = [],
      ka = [],
      la = [];
    function ma() {
      var a = d.preRun.shift();
      ja.unshift(a);
    }
    var P = 0,
      na = null,
      Q = null;
    function F(a) {
      if (d.onAbort) d.onAbort(a);
      a = "Aborted(" + a + ")";
      D(a);
      J = !0;
      K = 1;
      a = new WebAssembly.RuntimeError(a + ". Build with -sASSERTIONS for more info.");
      k(a);
      throw a;
    }
    function oa(a) {
      return a.startsWith("data:application/octet-stream;base64,");
    }
    var R;
    R = "ort-wasm-simd.wasm";
    if (!oa(R)) {
      var pa = R;
      R = d.locateFile ? d.locateFile(pa, A) : A + pa;
    }
    function qa(a) {
      if (a == R && E) return new Uint8Array(E);
      if (C) return C(a);
      throw "both async and sync fetching of the wasm failed";
    }
    function ra(a) {
      if (!E && (ca || y)) {
        if ("function" == typeof fetch && !a.startsWith("file://"))
          return fetch(a, { credentials: "same-origin" })
            .then((b) => {
              if (!b.ok) throw "failed to load wasm binary file at '" + a + "'";
              return b.arrayBuffer();
            })
            .catch(() => qa(a));
        if (B)
          return new Promise((b, c) => {
            B(a, (e) => b(new Uint8Array(e)), c);
          });
      }
      return Promise.resolve().then(() => qa(a));
    }
    function sa(a, b, c) {
      return ra(a)
        .then((e) => WebAssembly.instantiate(e, b))
        .then((e) => e)
        .then(c, (e) => {
          D("failed to asynchronously prepare wasm: " + e);
          F(e);
        });
    }
    function ta(a, b) {
      var c = R;
      return E ||
        "function" != typeof WebAssembly.instantiateStreaming ||
        oa(c) ||
        c.startsWith("file://") ||
        z ||
        "function" != typeof fetch
        ? sa(c, a, b)
        : fetch(c, { credentials: "same-origin" }).then((e) =>
            WebAssembly.instantiateStreaming(e, a).then(b, function (f) {
              D("wasm streaming compile failed: " + f);
              D("falling back to ArrayBuffer instantiation");
              return sa(c, a, b);
            }),
          );
    }
    var S,
      ua = {
        894528: () => {
          d.jsepRunPromise = new Promise(function (a) {
            d.Ja = a;
          });
        },
        894623: (a) => {
          d.Ja(a);
        },
        894661: (a) => d.Ea(a),
        894694: (a) => d.Ga(a),
        894726: (a, b, c) => {
          d.Ca(a, b, c, !0);
        },
        894765: (a, b, c) => {
          d.Ca(a, b, c);
        },
        894798: (a) => {
          d.la("Abs", a, void 0);
        },
        894849: (a) => {
          d.la("Neg", a, void 0);
        },
        894900: (a) => {
          d.la("Floor", a, void 0);
        },
        894953: (a) => {
          d.la("Ceil", a, void 0);
        },
        895005: (a) => {
          d.la("Reciprocal", a, void 0);
        },
        895063: (a) => {
          d.la("Sqrt", a, void 0);
        },
        895115: (a) => {
          d.la("Exp", a, void 0);
        },
        895166: (a) => {
          d.la("Erf", a, void 0);
        },
        895217: (a) => {
          d.la("Sigmoid", a, void 0);
        },
        895272: (a) => {
          d.la("Log", a, void 0);
        },
        895323: (a) => {
          d.la("Sin", a, void 0);
        },
        895374: (a) => {
          d.la("Cos", a, void 0);
        },
        895425: (a) => {
          d.la("Tan", a, void 0);
        },
        895476: (a) => {
          d.la("Asin", a, void 0);
        },
        895528: (a) => {
          d.la("Acos", a, void 0);
        },
        895580: (a) => {
          d.la("Atan", a, void 0);
        },
        895632: (a) => {
          d.la("Sinh", a, void 0);
        },
        895684: (a) => {
          d.la("Cosh", a, void 0);
        },
        895736: (a) => {
          d.la("Asinh", a, void 0);
        },
        895789: (a) => {
          d.la("Acosh", a, void 0);
        },
        895842: (a) => {
          d.la("Atanh", a, void 0);
        },
        895895: (a) => {
          d.la("Tanh", a, void 0);
        },
        895947: (a, b, c) => {
          d.la("ClipV10", a, { min: b, max: c });
        },
        896019: (a) => {
          d.la("Clip", a, void 0);
        },
        896071: (a, b) => {
          d.la("Elu", a, { alpha: b });
        },
        896129: (a) => {
          d.la("Relu", a, void 0);
        },
        896181: (a, b) => {
          d.la("LeakyRelu", a, { alpha: b });
        },
        896245: (a, b) => {
          d.la("ThresholdedRelu", a, { alpha: b });
        },
        896315: (a, b) => {
          d.la("Cast", a, { to: b });
        },
        896373: (a) => {
          d.la("Add", a, void 0);
        },
        896424: (a) => {
          d.la("Sub", a, void 0);
        },
        896475: (a) => {
          d.la("Mul", a, void 0);
        },
        896526: (a) => {
          d.la("Div", a, void 0);
        },
        896577: (a) => {
          d.la("Pow", a, void 0);
        },
        896628: (a, b, c, e, f) => {
          d.la("ReduceMean", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        896792: (a, b, c, e, f) => {
          d.la("ReduceMax", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        896955: (a, b, c, e, f) => {
          d.la("ReduceMin", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897118: (a, b, c, e, f) => {
          d.la("ReduceProd", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897282: (a, b, c, e, f) => {
          d.la("ReduceSum", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897445: (a, b, c, e, f) => {
          d.la("ReduceL1", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897607: (a, b, c, e, f) => {
          d.la("ReduceL2", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897769: (a, b, c, e, f) => {
          d.la("ReduceLogSum", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        897935: (a, b, c, e, f) => {
          d.la("ReduceSumSquare", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        898104: (a, b, c, e, f) => {
          d.la("ReduceLogSumExp", a, {
            keepDims: !!b,
            noopWithEmptyAxes: !!c,
            axes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        898273: (a, b, c) => {
          d.la("Transpose", a, { perm: b ? Array.from(N.subarray(c >>> 0, (c + b) >>> 0)) : [] });
        },
        898386: (a, b, c, e, f, h, l, m, r, n) => {
          d.la("Conv", a, {
            format: r ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, l],
            strides: [m],
            w_is_const: () => !!L[n >>> 0],
          });
        },
        898614: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q) => {
          d.la("Conv", a, {
            format: g ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c, e],
            group: f,
            kernel_shape: [h, l],
            pads: [m, r, n, p],
            strides: [v, w],
            w_is_const: () => !!L[q >>> 0],
          });
        },
        898873: (a, b, c, e, f, h, l, m, r, n) => {
          d.la("Conv", a, {
            format: r ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, l],
            strides: [m],
            w_is_const: () => !!L[n >>> 0],
          });
        },
        899101: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q) => {
          d.la("Conv", a, {
            format: g ? "NHWC" : "NCHW",
            auto_pad: b,
            dilations: [c, e],
            group: f,
            kernel_shape: [h, l],
            pads: [m, r, n, p],
            strides: [v, w],
            w_is_const: () => !!L[q >>> 0],
          });
        },
        899360: (a, b, c, e, f, h, l, m, r, n, p, v, w, g) => {
          d.la("ConvTranspose", a, {
            format: r ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, l],
            strides: [m],
            wIsConst: () => !!L[n >>> 0],
            outputPadding: p ? Array.from(N.subarray(v >>> 0, (v + p) >>> 0)) : [],
            outputShape: w ? Array.from(N.subarray(g >>> 0, (g + w) >>> 0)) : [],
          });
        },
        899740: (a, b, c, e, f, h, l, m, r, n, p, v, w) => {
          d.la("ConvTranspose", a, {
            format: m ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: Array.from(N.subarray(c >>> 0, (c + 2) >>> 0)),
            group: e,
            kernelShape: Array.from(N.subarray(f >>> 0, (f + 2) >>> 0)),
            pads: Array.from(N.subarray(h >>> 0, (h + 4) >>> 0)),
            strides: Array.from(N.subarray(l >>> 0, (l + 2) >>> 0)),
            wIsConst: () => !!L[r >>> 0],
            outputPadding: 0 < n ? Array.from(N.subarray(p >>> 0, (p + n) >>> 0)) : [],
            outputShape: 0 < v ? Array.from(N.subarray(w >>> 0, (w + v) >>> 0)) : [],
          });
        },
        900263: (a, b, c, e, f, h, l, m, r, n, p, v, w, g) => {
          d.la("ConvTranspose", a, {
            format: r ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: [c],
            group: e,
            kernel_shape: [f],
            pads: [h, l],
            strides: [m],
            wIsConst: () => !!L[n >>> 0],
            outputPadding: p ? Array.from(N.subarray(v >>> 0, (v + p) >>> 0)) : [],
            outputShape: w ? Array.from(N.subarray(g >>> 0, (g + w) >>> 0)) : [],
          });
        },
        900643: (a, b, c, e, f, h, l, m, r, n, p, v, w) => {
          d.la("ConvTranspose", a, {
            format: m ? "NHWC" : "NCHW",
            autoPad: b,
            dilations: Array.from(N.subarray(c >>> 0, (c + 2) >>> 0)),
            group: e,
            kernelShape: Array.from(N.subarray(f >>> 0, (f + 2) >>> 0)),
            pads: Array.from(N.subarray(h >>> 0, (h + 4) >>> 0)),
            strides: Array.from(N.subarray(l >>> 0, (l + 2) >>> 0)),
            wIsConst: () => !!L[r >>> 0],
            outputPadding: 0 < n ? Array.from(N.subarray(p >>> 0, (p + n) >>> 0)) : [],
            outputShape: 0 < v ? Array.from(N.subarray(w >>> 0, (w + v) >>> 0)) : [],
          });
        },
        901166: (a, b) => {
          d.la("GlobalAveragePool", a, { format: b ? "NHWC" : "NCHW" });
        },
        901257: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q, u) => {
          d.la("AveragePool", a, {
            format: u ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, l],
            kernel_shape: [m, r],
            pads: [n, p, v, w],
            strides: [g, q],
          });
        },
        901541: (a, b) => {
          d.la("GlobalAveragePool", a, { format: b ? "NHWC" : "NCHW" });
        },
        901632: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q, u) => {
          d.la("AveragePool", a, {
            format: u ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, l],
            kernel_shape: [m, r],
            pads: [n, p, v, w],
            strides: [g, q],
          });
        },
        901916: (a, b) => {
          d.la("GlobalMaxPool", a, { format: b ? "NHWC" : "NCHW" });
        },
        902003: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q, u) => {
          d.la("MaxPool", a, {
            format: u ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, l],
            kernel_shape: [m, r],
            pads: [n, p, v, w],
            strides: [g, q],
          });
        },
        902283: (a, b) => {
          d.la("GlobalMaxPool", a, { format: b ? "NHWC" : "NCHW" });
        },
        902370: (a, b, c, e, f, h, l, m, r, n, p, v, w, g, q, u) => {
          d.la("MaxPool", a, {
            format: u ? "NHWC" : "NCHW",
            auto_pad: b,
            ceil_mode: c,
            count_include_pad: e,
            storage_order: f,
            dilations: [h, l],
            kernel_shape: [m, r],
            pads: [n, p, v, w],
            strides: [g, q],
          });
        },
        902650: (a, b, c, e, f) => {
          d.la("Gemm", a, { alpha: b, beta: c, transA: e, transB: f });
        },
        902754: (a) => {
          d.la("MatMul", a, void 0);
        },
        902808: (a, b, c, e) => {
          d.la("ArgMax", a, { keepDims: !!b, selectLastIndex: !!c, axis: e });
        },
        902916: (a, b, c, e) => {
          d.la("ArgMin", a, { keepDims: !!b, selectLastIndex: !!c, axis: e });
        },
        903024: (a, b) => {
          d.la("Softmax", a, { axis: b });
        },
        903087: (a, b) => {
          d.la("Concat", a, { axis: b });
        },
        903147: (a, b, c, e, f) => {
          d.la("Split", a, {
            axis: b,
            numOutputs: c,
            splitSizes: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
          });
        },
        903292: (a) => {
          d.la("Expand", a, void 0);
        },
        903346: (a, b) => {
          d.la("Gather", a, { axis: Number(b) });
        },
        903417: (a, b, c, e, f, h, l, m, r, n, p) => {
          d.la("Resize", a, {
            antialias: b,
            axes: c ? Array.from(N.subarray(e >>> 0, (e + c) >>> 0)) : [],
            coordinateTransformMode: T(f),
            cubicCoeffA: h,
            excludeOutside: l,
            extrapolationValue: m,
            keepAspectRatioPolicy: T(r),
            mode: T(n),
            nearestMode: T(p),
          });
        },
        903768: (a, b, c, e, f, h, l) => {
          d.la("Slice", a, {
            starts: b ? Array.from(N.subarray(c >>> 0, (c + b) >>> 0)) : [],
            ends: e ? Array.from(N.subarray(f >>> 0, (f + e) >>> 0)) : [],
            axes: h ? Array.from(N.subarray(l >>> 0, (l + h) >>> 0)) : [],
          });
        },
        903999: (a) => {
          d.la("Tile", a, void 0);
        },
        904051: (a, b, c) => {
          d.la("LayerNormalization", a, { axis: Number(b), epsilon: Number(c) });
        },
        904158: (a, b, c) => {
          d.la("InstanceNormalization", a, { epsilon: b, format: c ? "NHWC" : "NCHW" });
        },
        904272: (a, b, c) => {
          d.la("InstanceNormalization", a, { epsilon: b, format: c ? "NHWC" : "NCHW" });
        },
        904386: (a) => {
          d.la("Gelu", a, void 0);
        },
        904438: (a, b) => {
          d.la("SkipLayerNormalization", a, { epsilon: b });
        },
        904519: (a) => {
          d.Ha(a);
        },
        904553: (a, b) => d.Ia(a, b),
      };
    function va(a) {
      this.name = "ExitStatus";
      this.message = `Program terminated with exit(${a})`;
      this.status = a;
    }
    var wa = (a) => {
      for (; 0 < a.length; ) a.shift()(d);
    };
    function xa(a) {
      this.za = a - 24;
      this.Ra = function (b) {
        O[((this.za + 4) >> 2) >>> 0] = b;
      };
      this.Qa = function (b) {
        O[((this.za + 8) >> 2) >>> 0] = b;
      };
      this.Ka = function (b, c) {
        this.Pa();
        this.Ra(b);
        this.Qa(c);
      };
      this.Pa = function () {
        O[((this.za + 16) >> 2) >>> 0] = 0;
      };
    }
    var ya = 0,
      za = 0,
      Aa = "undefined" != typeof TextDecoder ? new TextDecoder("utf8") : void 0,
      Ba = (a, b, c) => {
        b >>>= 0;
        var e = b + c;
        for (c = b; a[c] && !(c >= e); ) ++c;
        if (16 < c - b && a.buffer && Aa) return Aa.decode(a.subarray(b, c));
        for (e = ""; b < c; ) {
          var f = a[b++];
          if (f & 128) {
            var h = a[b++] & 63;
            if (192 == (f & 224)) e += String.fromCharCode(((f & 31) << 6) | h);
            else {
              var l = a[b++] & 63;
              f =
                224 == (f & 240)
                  ? ((f & 15) << 12) | (h << 6) | l
                  : ((f & 7) << 18) | (h << 12) | (l << 6) | (a[b++] & 63);
              65536 > f
                ? (e += String.fromCharCode(f))
                : ((f -= 65536), (e += String.fromCharCode(55296 | (f >> 10), 56320 | (f & 1023))));
            }
          } else e += String.fromCharCode(f);
        }
        return e;
      },
      T = (a, b) => ((a >>>= 0) ? Ba(M, a, b) : ""),
      Ca = (a) => {
        for (var b = 0, c = 0; c < a.length; ++c) {
          var e = a.charCodeAt(c);
          127 >= e ? b++ : 2047 >= e ? (b += 2) : 55296 <= e && 57343 >= e ? ((b += 4), ++c) : (b += 3);
        }
        return b;
      },
      Da = (a, b, c, e) => {
        c >>>= 0;
        if (!(0 < e)) return 0;
        var f = c;
        e = c + e - 1;
        for (var h = 0; h < a.length; ++h) {
          var l = a.charCodeAt(h);
          if (55296 <= l && 57343 >= l) {
            var m = a.charCodeAt(++h);
            l = (65536 + ((l & 1023) << 10)) | (m & 1023);
          }
          if (127 >= l) {
            if (c >= e) break;
            b[c++ >>> 0] = l;
          } else {
            if (2047 >= l) {
              if (c + 1 >= e) break;
              b[c++ >>> 0] = 192 | (l >> 6);
            } else {
              if (65535 >= l) {
                if (c + 2 >= e) break;
                b[c++ >>> 0] = 224 | (l >> 12);
              } else {
                if (c + 3 >= e) break;
                b[c++ >>> 0] = 240 | (l >> 18);
                b[c++ >>> 0] = 128 | ((l >> 12) & 63);
              }
              b[c++ >>> 0] = 128 | ((l >> 6) & 63);
            }
            b[c++ >>> 0] = 128 | (l & 63);
          }
        }
        b[c >>> 0] = 0;
        return c - f;
      },
      U = (a) => 0 === a % 4 && (0 !== a % 100 || 0 === a % 400),
      Ea = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
      Fa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
      Ha = (a) => {
        var b = Ca(a) + 1,
          c = Ga(b);
        c && Da(a, M, c, b);
        return c;
      },
      Ia = [],
      Ja = (a, b) => {
        Ia.length = 0;
        var c;
        for (b >>= 2; (c = M[a++ >>> 0]); ) (b += (105 != c) & b), Ia.push(105 == c ? N[b >>> 0] : ha[b++ >>> 1]), ++b;
        return Ia;
      },
      Ka = {},
      Na = () => {
        if (!La) {
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
              _: t || "./this.program",
            },
            b;
          for (b in Ka) void 0 === Ka[b] ? delete a[b] : (a[b] = Ka[b]);
          var c = [];
          for (b in a) c.push(`${b}=${a[b]}`);
          La = c;
        }
        return La;
      },
      La,
      Oa = (a) => {
        K = a;
        if (!noExitRuntime) {
          if (d.onExit) d.onExit(a);
          J = !0;
        }
        x(a, new va(a));
      },
      Pa = [null, [], []],
      Qa = () => {
        if ("object" == typeof crypto && "function" == typeof crypto.getRandomValues)
          return (c) => crypto.getRandomValues(c);
        if (z)
          try {
            var a = require("crypto");
            if (a.randomFillSync) return (c) => a.randomFillSync(c);
            var b = a.randomBytes;
            return (c) => (c.set(b(c.byteLength)), c);
          } catch (c) {}
        F("initRandomDevice");
      },
      Ra = (a) => (Ra = Qa())(a),
      Sa = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
      Ta = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    function Ua(a) {
      var b = Array(Ca(a) + 1);
      Da(a, b, 0, b.length);
      return b;
    }
    function Va(a, b, c, e) {
      function f(g, q, u) {
        for (g = "number" == typeof g ? g.toString() : g || ""; g.length < q; ) g = u[0] + g;
        return g;
      }
      function h(g, q) {
        return f(g, q, "0");
      }
      function l(g, q) {
        function u(Ma) {
          return 0 > Ma ? -1 : 0 < Ma ? 1 : 0;
        }
        var H;
        0 === (H = u(g.getFullYear() - q.getFullYear())) &&
          0 === (H = u(g.getMonth() - q.getMonth())) &&
          (H = u(g.getDate() - q.getDate()));
        return H;
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
      function r(g) {
        var q = g.va;
        for (g = new Date(new Date(g.wa + 1900, 0, 1).getTime()); 0 < q; ) {
          var u = g.getMonth(),
            H = (U(g.getFullYear()) ? Sa : Ta)[u];
          if (q > H - g.getDate())
            (q -= H - g.getDate() + 1),
              g.setDate(1),
              11 > u ? g.setMonth(u + 1) : (g.setMonth(0), g.setFullYear(g.getFullYear() + 1));
          else {
            g.setDate(g.getDate() + q);
            break;
          }
        }
        u = new Date(g.getFullYear() + 1, 0, 4);
        q = m(new Date(g.getFullYear(), 0, 4));
        u = m(u);
        return 0 >= l(q, g) ? (0 >= l(u, g) ? g.getFullYear() + 1 : g.getFullYear()) : g.getFullYear() - 1;
      }
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      var n = N[((e + 40) >> 2) >>> 0];
      e = {
        Na: N[(e >> 2) >>> 0],
        Ma: N[((e + 4) >> 2) >>> 0],
        xa: N[((e + 8) >> 2) >>> 0],
        Ba: N[((e + 12) >> 2) >>> 0],
        ya: N[((e + 16) >> 2) >>> 0],
        wa: N[((e + 20) >> 2) >>> 0],
        qa: N[((e + 24) >> 2) >>> 0],
        va: N[((e + 28) >> 2) >>> 0],
        Ta: N[((e + 32) >> 2) >>> 0],
        La: N[((e + 36) >> 2) >>> 0],
        Oa: n ? T(n) : "",
      };
      c = T(c);
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
      for (var p in n) c = c.replace(new RegExp(p, "g"), n[p]);
      var v = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        w = "January February March April May June July August September October November December".split(" ");
      n = {
        "%a": (g) => v[g.qa].substring(0, 3),
        "%A": (g) => v[g.qa],
        "%b": (g) => w[g.ya].substring(0, 3),
        "%B": (g) => w[g.ya],
        "%C": (g) => h(((g.wa + 1900) / 100) | 0, 2),
        "%d": (g) => h(g.Ba, 2),
        "%e": (g) => f(g.Ba, 2, " "),
        "%g": (g) => r(g).toString().substring(2),
        "%G": (g) => r(g),
        "%H": (g) => h(g.xa, 2),
        "%I": (g) => {
          g = g.xa;
          0 == g ? (g = 12) : 12 < g && (g -= 12);
          return h(g, 2);
        },
        "%j": (g) => {
          for (var q = 0, u = 0; u <= g.ya - 1; q += (U(g.wa + 1900) ? Sa : Ta)[u++]);
          return h(g.Ba + q, 3);
        },
        "%m": (g) => h(g.ya + 1, 2),
        "%M": (g) => h(g.Ma, 2),
        "%n": () => "\n",
        "%p": (g) => (0 <= g.xa && 12 > g.xa ? "AM" : "PM"),
        "%S": (g) => h(g.Na, 2),
        "%t": () => "\t",
        "%u": (g) => g.qa || 7,
        "%U": (g) => h(Math.floor((g.va + 7 - g.qa) / 7), 2),
        "%V": (g) => {
          var q = Math.floor((g.va + 7 - ((g.qa + 6) % 7)) / 7);
          2 >= (g.qa + 371 - g.va - 2) % 7 && q++;
          if (q) 53 == q && ((u = (g.qa + 371 - g.va) % 7), 4 == u || (3 == u && U(g.wa)) || (q = 1));
          else {
            q = 52;
            var u = (g.qa + 7 - g.va - 1) % 7;
            (4 == u || (5 == u && U((g.wa % 400) - 1))) && q++;
          }
          return h(q, 2);
        },
        "%w": (g) => g.qa,
        "%W": (g) => h(Math.floor((g.va + 7 - ((g.qa + 6) % 7)) / 7), 2),
        "%y": (g) => (g.wa + 1900).toString().substring(2),
        "%Y": (g) => g.wa + 1900,
        "%z": (g) => {
          g = g.La;
          var q = 0 <= g;
          g = Math.abs(g) / 60;
          return (q ? "+" : "-") + String("0000" + ((g / 60) * 100 + (g % 60))).slice(-4);
        },
        "%Z": (g) => g.Oa,
        "%%": () => "%",
      };
      c = c.replace(/%%/g, "\x00\x00");
      for (p in n) c.includes(p) && (c = c.replace(new RegExp(p, "g"), n[p](e)));
      c = c.replace(/\0\0/g, "%");
      p = Ua(c);
      if (p.length > b) return 0;
      L.set(p, a >>> 0);
      return p.length - 1;
    }
    function V(a) {
      try {
        a();
      } catch (b) {
        F(b);
      }
    }
    function Wa(a) {
      var b = {},
        c;
      for (c in a)
        (function (e) {
          var f = a[e];
          b[e] =
            "function" == typeof f
              ? function () {
                  W.push(e);
                  try {
                    return f.apply(null, arguments);
                  } finally {
                    J ||
                      (W.pop() === e || F(),
                      X && 1 === Y && 0 === W.length && ((Y = 0), V(Xa), "undefined" != typeof Fibers && Fibers.Ua()));
                  }
                }
              : f;
        })(c);
      return b;
    }
    var Y = 0,
      X = null,
      Ya = 0,
      W = [],
      Za = {},
      $a = {},
      ab = 0,
      bb = null,
      cb = [];
    function db() {
      var a = Ga(65548),
        b = a + 12;
      O[(a >> 2) >>> 0] = b;
      O[((a + 4) >> 2) >>> 0] = b + 65536;
      b = W[0];
      var c = Za[b];
      void 0 === c && ((c = ab++), (Za[b] = c), ($a[c] = b));
      N[((a + 8) >> 2) >>> 0] = c;
      return a;
    }
    function eb(a) {
      if (!J) {
        if (0 === Y) {
          var b = !1,
            c = !1;
          a((e = 0) => {
            if (!J && ((Ya = e), (b = !0), c)) {
              Y = 2;
              V(() => fb(X));
              "undefined" != typeof Browser && Browser.Aa.Da && Browser.Aa.resume();
              e = !1;
              try {
                var f = (0, I[$a[N[((X + 8) >> 2) >>> 0]]])();
              } catch (m) {
                (f = m), (e = !0);
              }
              var h = !1;
              if (!X) {
                var l = bb;
                l && ((bb = null), (e ? l.reject : l.resolve)(f), (h = !0));
              }
              if (e && !h) throw f;
            }
          });
          c = !0;
          b ||
            ((Y = 1), (X = db()), "undefined" != typeof Browser && Browser.Aa.Da && Browser.Aa.pause(), V(() => gb(X)));
        } else
          2 === Y
            ? ((Y = 0),
              V(hb),
              ib(X),
              (X = null),
              cb.forEach((e) => {
                if (!J)
                  try {
                    if ((e(), !noExitRuntime))
                      try {
                        (K = e = K), Oa(e);
                      } catch (f) {
                        f instanceof va || "unwind" == f || x(1, f);
                      }
                  } catch (f) {
                    f instanceof va || "unwind" == f || x(1, f);
                  }
              }))
            : F(`invalid state: ${Y}`);
        return Ya;
      }
    }
    function jb(a) {
      return eb((b) => {
        a().then(b);
      });
    }
    var lb = {
      o: function (a, b, c) {
        return jb(async () => {
          await d.Fa(a, b, c);
        });
      },
      a: function (a, b, c) {
        a >>>= 0;
        new xa(a).Ka(b >>> 0, c >>> 0);
        ya = a;
        za++;
        throw ya;
      },
      g: function () {
        return 0;
      },
      L: function () {},
      C: function () {},
      E: function () {},
      N: function () {
        return 0;
      },
      J: function () {},
      F: function () {},
      I: function () {},
      l: function () {},
      D: function () {},
      A: function () {},
      K: function () {},
      B: function () {},
      m: () => !0,
      r: function (a, b, c) {
        a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
        c >>>= 0;
        a = new Date(1e3 * a);
        N[(c >> 2) >>> 0] = a.getUTCSeconds();
        N[((c + 4) >> 2) >>> 0] = a.getUTCMinutes();
        N[((c + 8) >> 2) >>> 0] = a.getUTCHours();
        N[((c + 12) >> 2) >>> 0] = a.getUTCDate();
        N[((c + 16) >> 2) >>> 0] = a.getUTCMonth();
        N[((c + 20) >> 2) >>> 0] = a.getUTCFullYear() - 1900;
        N[((c + 24) >> 2) >>> 0] = a.getUTCDay();
        N[((c + 28) >> 2) >>> 0] = ((a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864e5) | 0;
      },
      s: function (a, b, c) {
        a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
        c >>>= 0;
        a = new Date(1e3 * a);
        N[(c >> 2) >>> 0] = a.getSeconds();
        N[((c + 4) >> 2) >>> 0] = a.getMinutes();
        N[((c + 8) >> 2) >>> 0] = a.getHours();
        N[((c + 12) >> 2) >>> 0] = a.getDate();
        N[((c + 16) >> 2) >>> 0] = a.getMonth();
        N[((c + 20) >> 2) >>> 0] = a.getFullYear() - 1900;
        N[((c + 24) >> 2) >>> 0] = a.getDay();
        N[((c + 28) >> 2) >>> 0] = ((U(a.getFullYear()) ? Ea : Fa)[a.getMonth()] + a.getDate() - 1) | 0;
        N[((c + 36) >> 2) >>> 0] = -(60 * a.getTimezoneOffset());
        b = new Date(a.getFullYear(), 6, 1).getTimezoneOffset();
        var e = new Date(a.getFullYear(), 0, 1).getTimezoneOffset();
        N[((c + 32) >> 2) >>> 0] = (b != e && a.getTimezoneOffset() == Math.min(e, b)) | 0;
      },
      t: function (a) {
        a >>>= 0;
        var b = new Date(
            N[((a + 20) >> 2) >>> 0] + 1900,
            N[((a + 16) >> 2) >>> 0],
            N[((a + 12) >> 2) >>> 0],
            N[((a + 8) >> 2) >>> 0],
            N[((a + 4) >> 2) >>> 0],
            N[(a >> 2) >>> 0],
            0,
          ),
          c = N[((a + 32) >> 2) >>> 0],
          e = b.getTimezoneOffset(),
          f = new Date(b.getFullYear(), 6, 1).getTimezoneOffset(),
          h = new Date(b.getFullYear(), 0, 1).getTimezoneOffset(),
          l = Math.min(h, f);
        0 > c
          ? (N[((a + 32) >> 2) >>> 0] = Number(f != h && l == e))
          : 0 < c != (l == e) && ((f = Math.max(h, f)), b.setTime(b.getTime() + 6e4 * ((0 < c ? l : f) - e)));
        N[((a + 24) >> 2) >>> 0] = b.getDay();
        N[((a + 28) >> 2) >>> 0] = ((U(b.getFullYear()) ? Ea : Fa)[b.getMonth()] + b.getDate() - 1) | 0;
        N[(a >> 2) >>> 0] = b.getSeconds();
        N[((a + 4) >> 2) >>> 0] = b.getMinutes();
        N[((a + 8) >> 2) >>> 0] = b.getHours();
        N[((a + 12) >> 2) >>> 0] = b.getDate();
        N[((a + 16) >> 2) >>> 0] = b.getMonth();
        N[((a + 20) >> 2) >>> 0] = b.getYear();
        a = b.getTime() / 1e3;
        return (
          kb(
            ((S = a),
            1 <= +Math.abs(S)
              ? 0 < S
                ? +Math.floor(S / 4294967296) >>> 0
                : ~~+Math.ceil((S - +(~~S >>> 0)) / 4294967296) >>> 0
              : 0),
          ),
          a >>> 0
        );
      },
      p: function () {
        return -52;
      },
      q: function () {},
      x: function (a, b, c) {
        function e(r) {
          return (r = r.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? r[1] : "GMT";
        }
        c >>>= 0;
        var f = new Date().getFullYear(),
          h = new Date(f, 0, 1),
          l = new Date(f, 6, 1);
        f = h.getTimezoneOffset();
        var m = l.getTimezoneOffset();
        O[((a >>> 0) >> 2) >>> 0] = 60 * Math.max(f, m);
        N[((b >>> 0) >> 2) >>> 0] = Number(f != m);
        a = e(h);
        b = e(l);
        a = Ha(a);
        b = Ha(b);
        m < f
          ? ((O[(c >> 2) >>> 0] = a), (O[((c + 4) >> 2) >>> 0] = b))
          : ((O[(c >> 2) >>> 0] = b), (O[((c + 4) >> 2) >>> 0] = a));
      },
      e: () => {
        F("");
      },
      b: function (a, b, c) {
        a >>>= 0;
        b = Ja(b >>> 0, c >>> 0);
        return ua[a].apply(null, b);
      },
      i: function (a, b, c) {
        a >>>= 0;
        b = Ja(b >>> 0, c >>> 0);
        return ua[a].apply(null, b);
      },
      h: function () {
        return Date.now();
      },
      z: function () {
        return 4294901760;
      },
      d: () => performance.now(),
      M: function (a, b, c) {
        b >>>= 0;
        return M.copyWithin((a >>> 0) >>> 0, b >>> 0, (b + (c >>> 0)) >>> 0);
      },
      w: function (a) {
        a >>>= 0;
        var b = M.length;
        if (4294901760 < a) return !1;
        for (var c = 1; 4 >= c; c *= 2) {
          var e = b * (1 + 0.2 / c);
          e = Math.min(e, a + 100663296);
          var f = Math;
          e = Math.max(a, e);
          a: {
            f = (f.min.call(f, 4294901760, e + ((65536 - (e % 65536)) % 65536)) - G.buffer.byteLength + 65535) >>> 16;
            try {
              G.grow(f);
              ia();
              var h = 1;
              break a;
            } catch (l) {}
            h = void 0;
          }
          if (h) return !0;
        }
        return !1;
      },
      G: function (a, b) {
        a >>>= 0;
        b >>>= 0;
        var c = 0;
        Na().forEach(function (e, f) {
          var h = b + c;
          f = O[((a + 4 * f) >> 2) >>> 0] = h;
          for (h = 0; h < e.length; ++h) L[(f++ >> 0) >>> 0] = e.charCodeAt(h);
          L[(f >> 0) >>> 0] = 0;
          c += e.length + 1;
        });
        return 0;
      },
      H: function (a, b) {
        a >>>= 0;
        b >>>= 0;
        var c = Na();
        O[(a >> 2) >>> 0] = c.length;
        var e = 0;
        c.forEach(function (f) {
          e += f.length + 1;
        });
        O[(b >> 2) >>> 0] = e;
        return 0;
      },
      n: (a) => {
        K = a;
        Oa(a);
      },
      f: () => 52,
      k: function () {
        return 52;
      },
      u: function () {
        return 70;
      },
      j: function (a, b, c, e) {
        b >>>= 0;
        c >>>= 0;
        e >>>= 0;
        for (var f = 0, h = 0; h < c; h++) {
          var l = O[(b >> 2) >>> 0],
            m = O[((b + 4) >> 2) >>> 0];
          b += 8;
          for (var r = 0; r < m; r++) {
            var n = M[(l + r) >>> 0],
              p = Pa[a];
            0 === n || 10 === n ? ((1 === a ? fa : D)(Ba(p, 0)), (p.length = 0)) : p.push(n);
          }
          f += m;
        }
        O[(e >> 2) >>> 0] = f;
        return 0;
      },
      v: function (a, b) {
        a >>>= 0;
        Ra(M.subarray(a >>> 0, (a + (b >>> 0)) >>> 0));
        return 0;
      },
      y: Va,
      c: function (a, b, c, e) {
        return Va(a >>> 0, b >>> 0, c >>> 0, e >>> 0);
      },
    };
    (function () {
      function a(c) {
        c = c.exports;
        c = Wa(c);
        I = c = mb(c);
        G = I.O;
        ia();
        ka.unshift(I.P);
        P--;
        d.monitorRunDependencies && d.monitorRunDependencies(P);
        if (0 == P && (null !== na && (clearInterval(na), (na = null)), Q)) {
          var e = Q;
          Q = null;
          e();
        }
        return c;
      }
      var b = { a: lb };
      P++;
      d.monitorRunDependencies && d.monitorRunDependencies(P);
      if (d.instantiateWasm)
        try {
          return d.instantiateWasm(b, a);
        } catch (c) {
          D("Module.instantiateWasm callback failed with error: " + c), k(c);
        }
      ta(b, function (c) {
        a(c.instance);
      }).catch(k);
      return {};
    })();
    d._OrtInit = (a, b) => (d._OrtInit = I.Q)(a, b);
    d._OrtGetLastError = (a, b) => (d._OrtGetLastError = I.R)(a, b);
    d._OrtCreateSessionOptions = (a, b, c, e, f, h, l, m, r, n) =>
      (d._OrtCreateSessionOptions = I.S)(a, b, c, e, f, h, l, m, r, n);
    d._OrtAppendExecutionProvider = (a, b) => (d._OrtAppendExecutionProvider = I.T)(a, b);
    d._OrtAddSessionConfigEntry = (a, b, c) => (d._OrtAddSessionConfigEntry = I.U)(a, b, c);
    d._OrtReleaseSessionOptions = (a) => (d._OrtReleaseSessionOptions = I.V)(a);
    d._OrtCreateSession = (a, b, c) => (d._OrtCreateSession = I.W)(a, b, c);
    d._OrtReleaseSession = (a) => (d._OrtReleaseSession = I.X)(a);
    d._OrtGetInputOutputCount = (a, b, c) => (d._OrtGetInputOutputCount = I.Y)(a, b, c);
    d._OrtGetInputName = (a, b) => (d._OrtGetInputName = I.Z)(a, b);
    d._OrtGetOutputName = (a, b) => (d._OrtGetOutputName = I._)(a, b);
    d._OrtFree = (a) => (d._OrtFree = I.$)(a);
    d._OrtCreateTensor = (a, b, c, e, f) => (d._OrtCreateTensor = I.aa)(a, b, c, e, f);
    d._OrtGetTensorData = (a, b, c, e, f) => (d._OrtGetTensorData = I.ba)(a, b, c, e, f);
    d._OrtReleaseTensor = (a) => (d._OrtReleaseTensor = I.ca)(a);
    d._OrtCreateRunOptions = (a, b, c, e) => (d._OrtCreateRunOptions = I.da)(a, b, c, e);
    d._OrtAddRunConfigEntry = (a, b, c) => (d._OrtAddRunConfigEntry = I.ea)(a, b, c);
    d._OrtReleaseRunOptions = (a) => (d._OrtReleaseRunOptions = I.fa)(a);
    d._OrtRun = (a, b, c, e, f, h, l, m) => (d._OrtRun = I.ga)(a, b, c, e, f, h, l, m);
    d._OrtEndProfiling = (a) => (d._OrtEndProfiling = I.ha)(a);
    d._JsepOutput = (a, b, c) => (d._JsepOutput = I.ia)(a, b, c);
    var Ga = (d._malloc = (a) => (Ga = d._malloc = I.ja)(a)),
      ib = (d._free = (a) => (ib = d._free = I.ka)(a)),
      kb = (a) => (kb = I.ma)(a),
      nb = () => (nb = I.na)(),
      ob = (a) => (ob = I.oa)(a),
      pb = (a) => (pb = I.pa)(a),
      gb = (a) => (gb = I.ra)(a),
      Xa = () => (Xa = I.sa)(),
      fb = (a) => (fb = I.ta)(a),
      hb = () => (hb = I.ua)();
    d.___start_em_js = 904588;
    d.___stop_em_js = 904749;
    function mb(a) {
      a = Object.assign({}, a);
      var b = (e) => () => e() >>> 0,
        c = (e) => (f) => e(f) >>> 0;
      a.__errno_location = b(a.__errno_location);
      a.malloc = c(a.malloc);
      a.stackSave = b(a.stackSave);
      a.stackAlloc = c(a.stackAlloc);
      return a;
    }
    d.stackAlloc = pb;
    d.stackSave = nb;
    d.stackRestore = ob;
    d.UTF8ToString = T;
    d.stringToUTF8 = (a, b, c) => Da(a, M, b, c);
    d.lengthBytesUTF8 = Ca;
    var Z;
    Q = function qb() {
      Z || rb();
      Z || (Q = qb);
    };
    function rb() {
      function a() {
        if (!Z && ((Z = !0), (d.calledRun = !0), !J)) {
          wa(ka);
          aa(d);
          if (d.onRuntimeInitialized) d.onRuntimeInitialized();
          if (d.postRun)
            for ("function" == typeof d.postRun && (d.postRun = [d.postRun]); d.postRun.length; ) {
              var b = d.postRun.shift();
              la.unshift(b);
            }
          wa(la);
        }
      }
      if (!(0 < P)) {
        if (d.preRun) for ("function" == typeof d.preRun && (d.preRun = [d.preRun]); d.preRun.length; ) ma();
        wa(ja);
        0 < P ||
          (d.setStatus
            ? (d.setStatus("Running..."),
              setTimeout(function () {
                setTimeout(function () {
                  d.setStatus("");
                }, 1);
                a();
              }, 1))
            : a());
      }
    }
    if (d.preInit)
      for ("function" == typeof d.preInit && (d.preInit = [d.preInit]); 0 < d.preInit.length; ) d.preInit.pop()();
    rb();

    return moduleArg.ready;
  };
})();
if (typeof exports === "object" && typeof module === "object") module.exports = ortWasm;
else if (typeof define === "function" && define["amd"]) define([], () => ortWasm);
