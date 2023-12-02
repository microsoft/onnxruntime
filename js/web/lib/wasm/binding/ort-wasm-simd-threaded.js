var ortWasmThreaded = (() => {
  var _scriptDir = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== "undefined") _scriptDir = _scriptDir || __filename;
  return function (moduleArg = {}) {
    function aa() {
      d.buffer != l.buffer && m();
      return l;
    }
    function n() {
      d.buffer != l.buffer && m();
      return ba;
    }
    function p() {
      d.buffer != l.buffer && m();
      return ca;
    }
    function r() {
      d.buffer != l.buffer && m();
      return da;
    }
    function ea() {
      d.buffer != l.buffer && m();
      return fa;
    }
    var w = moduleArg,
      ha,
      x;
    w.ready = new Promise((a, b) => {
      ha = a;
      x = b;
    });
    var ia = Object.assign({}, w),
      ja = "./this.program",
      z = (a, b) => {
        throw b;
      },
      ka = "object" == typeof window,
      A = "function" == typeof importScripts,
      B = "object" == typeof process && "object" == typeof process.versions && "string" == typeof process.versions.node,
      D = w.ENVIRONMENT_IS_PTHREAD || !1,
      E = "";
    function la(a) {
      return w.locateFile ? w.locateFile(a, E) : E + a;
    }
    var ma, F, H;
    if (B) {
      var fs = require("fs"),
        na = require("path");
      E = A ? na.dirname(E) + "/" : __dirname + "/";
      ma = (b, c) => {
        b = b.startsWith("file://") ? new URL(b) : na.normalize(b);
        return fs.readFileSync(b, c ? void 0 : "utf8");
      };
      H = (b) => {
        b = ma(b, !0);
        b.buffer || (b = new Uint8Array(b));
        return b;
      };
      F = (b, c, e, h = !0) => {
        b = b.startsWith("file://") ? new URL(b) : na.normalize(b);
        fs.readFile(b, h ? void 0 : "utf8", (g, k) => {
          g ? e(g) : c(h ? k.buffer : k);
        });
      };
      !w.thisProgram && 1 < process.argv.length && (ja = process.argv[1].replace(/\\/g, "/"));
      process.argv.slice(2);
      z = (b, c) => {
        process.exitCode = b;
        throw c;
      };
      w.inspect = () => "[Emscripten Module object]";
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
    } else if (ka || A)
      A
        ? (E = self.location.href)
        : "undefined" != typeof document && document.currentScript && (E = document.currentScript.src),
        _scriptDir && (E = _scriptDir),
        0 !== E.indexOf("blob:") ? (E = E.substr(0, E.replace(/[?#].*/, "").lastIndexOf("/") + 1)) : (E = ""),
        B ||
          ((ma = (a) => {
            var b = new XMLHttpRequest();
            b.open("GET", a, !1);
            b.send(null);
            return b.responseText;
          }),
          A &&
            (H = (a) => {
              var b = new XMLHttpRequest();
              b.open("GET", a, !1);
              b.responseType = "arraybuffer";
              b.send(null);
              return new Uint8Array(b.response);
            }),
          (F = (a, b, c) => {
            var e = new XMLHttpRequest();
            e.open("GET", a, !0);
            e.responseType = "arraybuffer";
            e.onload = () => {
              200 == e.status || (0 == e.status && e.response) ? b(e.response) : c();
            };
            e.onerror = c;
            e.send(null);
          }));
    B && "undefined" == typeof performance && (global.performance = require("perf_hooks").performance);
    var oa = console.log.bind(console),
      pa = console.error.bind(console);
    B && ((oa = (...a) => fs.writeSync(1, a.join(" ") + "\n")), (pa = (...a) => fs.writeSync(2, a.join(" ") + "\n")));
    var qa = w.print || oa,
      I = w.printErr || pa;
    Object.assign(w, ia);
    ia = null;
    w.thisProgram && (ja = w.thisProgram);
    w.quit && (z = w.quit);
    var J;
    w.wasmBinary && (J = w.wasmBinary);
    var noExitRuntime = w.noExitRuntime || !0;
    "object" != typeof WebAssembly && K("no native wasm support detected");
    var d,
      L,
      ra,
      M = !1,
      N,
      l,
      ba,
      ca,
      da,
      fa;
    function m() {
      var a = d.buffer;
      w.HEAP8 = l = new Int8Array(a);
      w.HEAP16 = new Int16Array(a);
      w.HEAP32 = ca = new Int32Array(a);
      w.HEAPU8 = ba = new Uint8Array(a);
      w.HEAPU16 = new Uint16Array(a);
      w.HEAPU32 = da = new Uint32Array(a);
      w.HEAPF32 = new Float32Array(a);
      w.HEAPF64 = fa = new Float64Array(a);
    }
    var O = w.INITIAL_MEMORY || 16777216;
    5242880 <= O || K("INITIAL_MEMORY should be larger than STACK_SIZE, was " + O + "! (STACK_SIZE=5242880)");
    if (D) d = w.wasmMemory;
    else if (w.wasmMemory) d = w.wasmMemory;
    else if (
      ((d = new WebAssembly.Memory({ initial: O / 65536, maximum: 65536, shared: !0 })),
      !(d.buffer instanceof SharedArrayBuffer))
    )
      throw (
        (I(
          "requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag",
        ),
        B &&
          I(
            "(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)",
          ),
        Error("bad memory"))
      );
    m();
    O = d.buffer.byteLength;
    var sa,
      ta = [],
      ua = [],
      va = [],
      wa = 0;
    function P() {
      return noExitRuntime || 0 < wa;
    }
    var Q = 0,
      xa = null,
      R = null;
    function ya() {
      Q++;
      w.monitorRunDependencies && w.monitorRunDependencies(Q);
    }
    function za() {
      Q--;
      w.monitorRunDependencies && w.monitorRunDependencies(Q);
      if (0 == Q && (null !== xa && (clearInterval(xa), (xa = null)), R)) {
        var a = R;
        R = null;
        a();
      }
    }
    function K(a) {
      if (w.onAbort) w.onAbort(a);
      a = "Aborted(" + a + ")";
      I(a);
      M = !0;
      N = 1;
      a = new WebAssembly.RuntimeError(a + ". Build with -sASSERTIONS for more info.");
      x(a);
      throw a;
    }
    function Aa(a) {
      return a.startsWith("data:application/octet-stream;base64,");
    }
    var S;
    S = "ort-wasm-simd-threaded.wasm";
    Aa(S) || (S = la(S));
    function Ba(a) {
      if (a == S && J) return new Uint8Array(J);
      if (H) return H(a);
      throw "both async and sync fetching of the wasm failed";
    }
    function Ca(a) {
      if (!J && (ka || A)) {
        if ("function" == typeof fetch && !a.startsWith("file://"))
          return fetch(a, { credentials: "same-origin" })
            .then((b) => {
              if (!b.ok) throw "failed to load wasm binary file at '" + a + "'";
              return b.arrayBuffer();
            })
            .catch(() => Ba(a));
        if (F)
          return new Promise((b, c) => {
            F(a, (e) => b(new Uint8Array(e)), c);
          });
      }
      return Promise.resolve().then(() => Ba(a));
    }
    function Da(a, b, c) {
      return Ca(a)
        .then((e) => WebAssembly.instantiate(e, b))
        .then((e) => e)
        .then(c, (e) => {
          I("failed to asynchronously prepare wasm: " + e);
          K(e);
        });
    }
    function Ea(a, b) {
      var c = S;
      return J ||
        "function" != typeof WebAssembly.instantiateStreaming ||
        Aa(c) ||
        c.startsWith("file://") ||
        B ||
        "function" != typeof fetch
        ? Da(c, a, b)
        : fetch(c, { credentials: "same-origin" }).then((e) =>
            WebAssembly.instantiateStreaming(e, a).then(b, function (h) {
              I("wasm streaming compile failed: " + h);
              I("falling back to ArrayBuffer instantiation");
              return Da(c, a, b);
            }),
          );
    }
    var T;
    function U(a) {
      this.name = "ExitStatus";
      this.message = `Program terminated with exit(${a})`;
      this.status = a;
    }
    function Fa(a) {
      a.terminate();
      a.onmessage = () => {};
    }
    function Ga(a) {
      (a = V.Fa[a]) || K();
      V.fb(a);
    }
    function Ha(a) {
      var b = V.Za();
      if (!b) return 6;
      V.Ia.push(b);
      V.Fa[a.Ha] = b;
      b.Ha = a.Ha;
      var c = { cmd: "run", start_routine: a.gb, arg: a.Ya, pthread_ptr: a.Ha };
      B && b.unref();
      b.postMessage(c, a.mb);
      return 0;
    }
    var Ia = "undefined" != typeof TextDecoder ? new TextDecoder("utf8") : void 0,
      Ja = (a, b, c) => {
        b >>>= 0;
        var e = b + c;
        for (c = b; a[c] && !(c >= e); ) ++c;
        if (16 < c - b && a.buffer && Ia)
          return Ia.decode(a.buffer instanceof SharedArrayBuffer ? a.slice(b, c) : a.subarray(b, c));
        for (e = ""; b < c; ) {
          var h = a[b++];
          if (h & 128) {
            var g = a[b++] & 63;
            if (192 == (h & 224)) e += String.fromCharCode(((h & 31) << 6) | g);
            else {
              var k = a[b++] & 63;
              h =
                224 == (h & 240)
                  ? ((h & 15) << 12) | (g << 6) | k
                  : ((h & 7) << 18) | (g << 12) | (k << 6) | (a[b++] & 63);
              65536 > h
                ? (e += String.fromCharCode(h))
                : ((h -= 65536), (e += String.fromCharCode(55296 | (h >> 10), 56320 | (h & 1023))));
            }
          } else e += String.fromCharCode(h);
        }
        return e;
      },
      Ka = (a, b) => ((a >>>= 0) ? Ja(n(), a, b) : "");
    function La(a) {
      if (D) return W(1, 1, a);
      N = a;
      if (!P()) {
        V.hb();
        if (w.onExit) w.onExit(a);
        M = !0;
      }
      z(a, new U(a));
    }
    var Na = (a) => {
        N = a;
        if (D) throw (Ma(a), "unwind");
        La(a);
      },
      V = {
        La: [],
        Ia: [],
        Ta: [],
        Fa: {},
        Pa: function () {
          D ? V.ab() : V.$a();
        },
        $a: function () {
          ta.unshift(() => {
            ya();
            V.bb(() => za());
          });
        },
        ab: function () {
          V.receiveObjectTransfer = V.eb;
          V.threadInitTLS = V.Sa;
          V.setExitStatus = V.Ra;
          noExitRuntime = !1;
        },
        Ra: function (a) {
          N = a;
        },
        rb: ["$terminateWorker"],
        hb: function () {
          for (var a of V.Ia) Fa(a);
          for (a of V.La) Fa(a);
          V.La = [];
          V.Ia = [];
          V.Fa = [];
        },
        fb: function (a) {
          var b = a.Ha;
          delete V.Fa[b];
          V.La.push(a);
          V.Ia.splice(V.Ia.indexOf(a), 1);
          a.Ha = 0;
          Oa(b);
        },
        eb: function () {},
        Sa: function () {
          V.Ta.forEach((a) => a());
        },
        cb: (a) =>
          new Promise((b) => {
            a.onmessage = (g) => {
              g = g.data;
              var k = g.cmd;
              if (g.targetThread && g.targetThread != X()) {
                var t = V.Fa[g.qb];
                t
                  ? t.postMessage(g, g.transferList)
                  : I(
                      'Internal error! Worker sent a message "' +
                        k +
                        '" to target pthread ' +
                        g.targetThread +
                        ", but that thread no longer exists!",
                    );
              } else if ("checkMailbox" === k) Y();
              else if ("spawnThread" === k) Ha(g);
              else if ("cleanupThread" === k) Ga(g.thread);
              else if ("killThread" === k)
                (g = g.thread),
                  (k = V.Fa[g]),
                  delete V.Fa[g],
                  Fa(k),
                  Oa(g),
                  V.Ia.splice(V.Ia.indexOf(k), 1),
                  (k.Ha = 0);
              else if ("cancelThread" === k) V.Fa[g.thread].postMessage({ cmd: "cancel" });
              else if ("loaded" === k) (a.loaded = !0), b(a);
              else if ("alert" === k) alert("Thread " + g.threadId + ": " + g.text);
              else if ("setimmediate" === g.target) a.postMessage(g);
              else if ("callHandler" === k) w[g.handler](...g.args);
              else k && I("worker sent an unknown command " + k);
            };
            a.onerror = (g) => {
              I("worker sent an error! " + g.filename + ":" + g.lineno + ": " + g.message);
              throw g;
            };
            B &&
              (a.on("message", function (g) {
                a.onmessage({ data: g });
              }),
              a.on("error", function (g) {
                a.onerror(g);
              }));
            var c = [],
              e = ["onExit", "onAbort", "print", "printErr"],
              h;
            for (h of e) w.hasOwnProperty(h) && c.push(h);
            a.postMessage({
              cmd: "load",
              handlers: c,
              urlOrBlob: w.mainScriptUrlOrBlob || _scriptDir,
              wasmMemory: d,
              wasmModule: ra,
            });
          }),
        bb: function (a) {
          a();
        },
        Xa: function () {
          var a = la("ort-wasm-simd-threaded.worker.js");
          a = new Worker(a);
          V.La.push(a);
        },
        Za: function () {
          0 == V.La.length && (V.Xa(), V.cb(V.La[0]));
          return V.La.pop();
        },
      };
    w.PThread = V;
    var Pa = (a) => {
      for (; 0 < a.length; ) a.shift()(w);
    };
    w.establishStackSpace = function () {
      var a = X(),
        b = p()[((a + 52) >> 2) >>> 0];
      a = p()[((a + 56) >> 2) >>> 0];
      Qa(b, b - a);
      Ra(b);
    };
    function Ma(a) {
      if (D) return W(2, 0, a);
      Na(a);
    }
    var Sa = [];
    w.invokeEntryPoint = function (a, b) {
      var c = Sa[a];
      c || (a >= Sa.length && (Sa.length = a + 1), (Sa[a] = c = sa.get(a)));
      a = c(b);
      P() ? V.Ra(a) : Ta(a);
    };
    function Ua(a) {
      this.Oa = a - 24;
      this.Wa = function (b) {
        r()[((this.Oa + 4) >> 2) >>> 0] = b;
      };
      this.Va = function (b) {
        r()[((this.Oa + 8) >> 2) >>> 0] = b;
      };
      this.Pa = function (b, c) {
        this.Ua();
        this.Wa(b);
        this.Va(c);
      };
      this.Ua = function () {
        r()[((this.Oa + 16) >> 2) >>> 0] = 0;
      };
    }
    var Va = 0,
      Wa = 0;
    function Xa(a, b, c, e) {
      return D ? W(3, 1, a, b, c, e) : Ya(a, b, c, e);
    }
    function Ya(a, b, c, e) {
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      if ("undefined" == typeof SharedArrayBuffer)
        return I("Current environment does not support SharedArrayBuffer, pthreads are not available!"), 6;
      var h = [];
      if (D && 0 === h.length) return Xa(a, b, c, e);
      a = { gb: c, Ha: a, Ya: e, mb: h };
      return D ? ((a.ob = "spawnThread"), postMessage(a, h), 0) : Ha(a);
    }
    function Za(a, b, c) {
      return D ? W(4, 1, a, b, c) : 0;
    }
    function $a(a, b) {
      if (D) return W(5, 1, a, b);
    }
    var ab = (a) => {
        for (var b = 0, c = 0; c < a.length; ++c) {
          var e = a.charCodeAt(c);
          127 >= e ? b++ : 2047 >= e ? (b += 2) : 55296 <= e && 57343 >= e ? ((b += 4), ++c) : (b += 3);
        }
        return b;
      },
      bb = (a, b, c, e) => {
        c >>>= 0;
        if (!(0 < e)) return 0;
        var h = c;
        e = c + e - 1;
        for (var g = 0; g < a.length; ++g) {
          var k = a.charCodeAt(g);
          if (55296 <= k && 57343 >= k) {
            var t = a.charCodeAt(++g);
            k = (65536 + ((k & 1023) << 10)) | (t & 1023);
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
        return c - h;
      },
      cb = (a, b, c) => bb(a, n(), b, c);
    function db(a, b) {
      if (D) return W(6, 1, a, b);
    }
    function eb(a, b, c) {
      if (D) return W(7, 1, a, b, c);
    }
    function fb(a, b, c) {
      return D ? W(8, 1, a, b, c) : 0;
    }
    function gb(a, b) {
      if (D) return W(9, 1, a, b);
    }
    function hb(a, b, c) {
      if (D) return W(10, 1, a, b, c);
    }
    function ib(a, b, c, e) {
      if (D) return W(11, 1, a, b, c, e);
    }
    function jb(a, b, c, e) {
      if (D) return W(12, 1, a, b, c, e);
    }
    function kb(a, b, c, e) {
      if (D) return W(13, 1, a, b, c, e);
    }
    function lb(a) {
      if (D) return W(14, 1, a);
    }
    function mb(a, b) {
      if (D) return W(15, 1, a, b);
    }
    function nb(a, b, c) {
      if (D) return W(16, 1, a, b, c);
    }
    var ob = (a) => {
      if (!M)
        try {
          if ((a(), !P()))
            try {
              D ? Ta(N) : Na(N);
            } catch (b) {
              b instanceof U || "unwind" == b || z(1, b);
            }
        } catch (b) {
          b instanceof U || "unwind" == b || z(1, b);
        }
    };
    function pb(a) {
      a >>>= 0;
      "function" === typeof Atomics.nb &&
        (Atomics.nb(p(), a >> 2, a).value.then(Y), (a += 128), Atomics.store(p(), a >> 2, 1));
    }
    w.__emscripten_thread_mailbox_await = pb;
    function Y() {
      var a = X();
      a && (pb(a), ob(() => qb()));
    }
    w.checkMailbox = Y;
    var Z = (a) => 0 === a % 4 && (0 !== a % 100 || 0 === a % 400),
      rb = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
      sb = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    function tb(a, b, c, e, h, g, k, t) {
      return D ? W(17, 1, a, b, c, e, h, g, k, t) : -52;
    }
    function ub(a, b, c, e, h, g, k) {
      if (D) return W(18, 1, a, b, c, e, h, g, k);
    }
    var wb = (a) => {
        var b = ab(a) + 1,
          c = vb(b);
        c && cb(a, c, b);
        return c;
      },
      yb = (a) => {
        var b = xb();
        a = a();
        Ra(b);
        return a;
      };
    function W(a, b) {
      var c = arguments.length - 2,
        e = arguments;
      return yb(() => {
        for (var h = zb(8 * c), g = h >> 3, k = 0; k < c; k++) {
          var t = e[2 + k];
          ea()[(g + k) >>> 0] = t;
        }
        return Ab(a, c, h, b);
      });
    }
    var Bb = [],
      Cb = {},
      Eb = () => {
        if (!Db) {
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
              _: ja || "./this.program",
            },
            b;
          for (b in Cb) void 0 === Cb[b] ? delete a[b] : (a[b] = Cb[b]);
          var c = [];
          for (b in a) c.push(`${b}=${a[b]}`);
          Db = c;
        }
        return Db;
      },
      Db;
    function Fb(a, b) {
      if (D) return W(19, 1, a, b);
      a >>>= 0;
      b >>>= 0;
      var c = 0;
      Eb().forEach(function (e, h) {
        var g = b + c;
        h = r()[((a + 4 * h) >> 2) >>> 0] = g;
        for (g = 0; g < e.length; ++g) aa()[(h++ >> 0) >>> 0] = e.charCodeAt(g);
        aa()[(h >> 0) >>> 0] = 0;
        c += e.length + 1;
      });
      return 0;
    }
    function Gb(a, b) {
      if (D) return W(20, 1, a, b);
      a >>>= 0;
      b >>>= 0;
      var c = Eb();
      r()[(a >> 2) >>> 0] = c.length;
      var e = 0;
      c.forEach(function (h) {
        e += h.length + 1;
      });
      r()[(b >> 2) >>> 0] = e;
      return 0;
    }
    function Hb(a) {
      return D ? W(21, 1, a) : 52;
    }
    function Ib(a, b, c, e) {
      return D ? W(22, 1, a, b, c, e) : 52;
    }
    function Mb(a, b, c, e, h) {
      return D ? W(23, 1, a, b, c, e, h) : 70;
    }
    var Nb = [null, [], []];
    function Ob(a, b, c, e) {
      if (D) return W(24, 1, a, b, c, e);
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      for (var h = 0, g = 0; g < c; g++) {
        var k = r()[(b >> 2) >>> 0],
          t = r()[((b + 4) >> 2) >>> 0];
        b += 8;
        for (var C = 0; C < t; C++) {
          var v = n()[(k + C) >>> 0],
            y = Nb[a];
          0 === v || 10 === v ? ((1 === a ? qa : I)(Ja(y, 0)), (y.length = 0)) : y.push(v);
        }
        h += t;
      }
      r()[(e >> 2) >>> 0] = h;
      return 0;
    }
    var Pb = () => {
        if ("object" == typeof crypto && "function" == typeof crypto.getRandomValues)
          return (c) => (c.set(crypto.getRandomValues(new Uint8Array(c.byteLength))), c);
        if (B)
          try {
            var a = require("crypto");
            if (a.randomFillSync) return (c) => a.randomFillSync(c);
            var b = a.randomBytes;
            return (c) => (c.set(b(c.byteLength)), c);
          } catch (c) {}
        K("initRandomDevice");
      },
      Qb = (a) => (Qb = Pb())(a),
      Rb = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
      Sb = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    function Tb(a) {
      var b = Array(ab(a) + 1);
      bb(a, b, 0, b.length);
      return b;
    }
    var Ub = (a, b) => {
      aa().set(a, b >>> 0);
    };
    function Vb(a, b, c, e) {
      function h(f, q, u) {
        for (f = "number" == typeof f ? f.toString() : f || ""; f.length < q; ) f = u[0] + f;
        return f;
      }
      function g(f, q) {
        return h(f, q, "0");
      }
      function k(f, q) {
        function u(Jb) {
          return 0 > Jb ? -1 : 0 < Jb ? 1 : 0;
        }
        var G;
        0 === (G = u(f.getFullYear() - q.getFullYear())) &&
          0 === (G = u(f.getMonth() - q.getMonth())) &&
          (G = u(f.getDate() - q.getDate()));
        return G;
      }
      function t(f) {
        switch (f.getDay()) {
          case 0:
            return new Date(f.getFullYear() - 1, 11, 29);
          case 1:
            return f;
          case 2:
            return new Date(f.getFullYear(), 0, 3);
          case 3:
            return new Date(f.getFullYear(), 0, 2);
          case 4:
            return new Date(f.getFullYear(), 0, 1);
          case 5:
            return new Date(f.getFullYear() - 1, 11, 31);
          case 6:
            return new Date(f.getFullYear() - 1, 11, 30);
        }
      }
      function C(f) {
        var q = f.Ja;
        for (f = new Date(new Date(f.Ka + 1900, 0, 1).getTime()); 0 < q; ) {
          var u = f.getMonth(),
            G = (Z(f.getFullYear()) ? Rb : Sb)[u];
          if (q > G - f.getDate())
            (q -= G - f.getDate() + 1),
              f.setDate(1),
              11 > u ? f.setMonth(u + 1) : (f.setMonth(0), f.setFullYear(f.getFullYear() + 1));
          else {
            f.setDate(f.getDate() + q);
            break;
          }
        }
        u = new Date(f.getFullYear() + 1, 0, 4);
        q = t(new Date(f.getFullYear(), 0, 4));
        u = t(u);
        return 0 >= k(q, f) ? (0 >= k(u, f) ? f.getFullYear() + 1 : f.getFullYear()) : f.getFullYear() - 1;
      }
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      e >>>= 0;
      var v = p()[((e + 40) >> 2) >>> 0];
      e = {
        kb: p()[(e >> 2) >>> 0],
        jb: p()[((e + 4) >> 2) >>> 0],
        Ma: p()[((e + 8) >> 2) >>> 0],
        Qa: p()[((e + 12) >> 2) >>> 0],
        Na: p()[((e + 16) >> 2) >>> 0],
        Ka: p()[((e + 20) >> 2) >>> 0],
        Ga: p()[((e + 24) >> 2) >>> 0],
        Ja: p()[((e + 28) >> 2) >>> 0],
        sb: p()[((e + 32) >> 2) >>> 0],
        ib: p()[((e + 36) >> 2) >>> 0],
        lb: v ? Ka(v) : "",
      };
      c = Ka(c);
      v = {
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
      for (var y in v) c = c.replace(new RegExp(y, "g"), v[y]);
      var Kb = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        Lb = "January February March April May June July August September October November December".split(" ");
      v = {
        "%a": (f) => Kb[f.Ga].substring(0, 3),
        "%A": (f) => Kb[f.Ga],
        "%b": (f) => Lb[f.Na].substring(0, 3),
        "%B": (f) => Lb[f.Na],
        "%C": (f) => g(((f.Ka + 1900) / 100) | 0, 2),
        "%d": (f) => g(f.Qa, 2),
        "%e": (f) => h(f.Qa, 2, " "),
        "%g": (f) => C(f).toString().substring(2),
        "%G": (f) => C(f),
        "%H": (f) => g(f.Ma, 2),
        "%I": (f) => {
          f = f.Ma;
          0 == f ? (f = 12) : 12 < f && (f -= 12);
          return g(f, 2);
        },
        "%j": (f) => {
          for (var q = 0, u = 0; u <= f.Na - 1; q += (Z(f.Ka + 1900) ? Rb : Sb)[u++]);
          return g(f.Qa + q, 3);
        },
        "%m": (f) => g(f.Na + 1, 2),
        "%M": (f) => g(f.jb, 2),
        "%n": () => "\n",
        "%p": (f) => (0 <= f.Ma && 12 > f.Ma ? "AM" : "PM"),
        "%S": (f) => g(f.kb, 2),
        "%t": () => "\t",
        "%u": (f) => f.Ga || 7,
        "%U": (f) => g(Math.floor((f.Ja + 7 - f.Ga) / 7), 2),
        "%V": (f) => {
          var q = Math.floor((f.Ja + 7 - ((f.Ga + 6) % 7)) / 7);
          2 >= (f.Ga + 371 - f.Ja - 2) % 7 && q++;
          if (q) 53 == q && ((u = (f.Ga + 371 - f.Ja) % 7), 4 == u || (3 == u && Z(f.Ka)) || (q = 1));
          else {
            q = 52;
            var u = (f.Ga + 7 - f.Ja - 1) % 7;
            (4 == u || (5 == u && Z((f.Ka % 400) - 1))) && q++;
          }
          return g(q, 2);
        },
        "%w": (f) => f.Ga,
        "%W": (f) => g(Math.floor((f.Ja + 7 - ((f.Ga + 6) % 7)) / 7), 2),
        "%y": (f) => (f.Ka + 1900).toString().substring(2),
        "%Y": (f) => f.Ka + 1900,
        "%z": (f) => {
          f = f.ib;
          var q = 0 <= f;
          f = Math.abs(f) / 60;
          return (q ? "+" : "-") + String("0000" + ((f / 60) * 100 + (f % 60))).slice(-4);
        },
        "%Z": (f) => f.lb,
        "%%": () => "%",
      };
      c = c.replace(/%%/g, "\x00\x00");
      for (y in v) c.includes(y) && (c = c.replace(new RegExp(y, "g"), v[y](e)));
      c = c.replace(/\0\0/g, "%");
      y = Tb(c);
      if (y.length > b) return 0;
      Ub(y, a);
      return y.length - 1;
    }
    V.Pa();
    var Wb = [null, La, Ma, Xa, Za, $a, db, eb, fb, gb, hb, ib, jb, kb, lb, mb, nb, tb, ub, Fb, Gb, Hb, Ib, Mb, Ob],
      Zb = {
        b: function (a, b, c) {
          a >>>= 0;
          new Ua(a).Pa(b >>> 0, c >>> 0);
          Va = a;
          Wa++;
          throw Va;
        },
        N: function (a) {
          Xb(a >>> 0, !A, 1, !ka, 131072, !1);
          V.Sa();
        },
        k: function (a) {
          a >>>= 0;
          D ? postMessage({ cmd: "cleanupThread", thread: a }) : Ga(a);
        },
        I: Ya,
        h: Za,
        T: $a,
        E: db,
        G: eb,
        U: fb,
        R: gb,
        J: hb,
        Q: ib,
        o: jb,
        F: kb,
        C: lb,
        S: mb,
        D: nb,
        q: () => !0,
        A: function (a, b) {
          a >>>= 0;
          a == b >>> 0
            ? setTimeout(() => Y())
            : D
            ? postMessage({ targetThread: a, cmd: "checkMailbox" })
            : (a = V.Fa[a]) && a.postMessage({ cmd: "checkMailbox" });
        },
        L: function () {
          return -1;
        },
        M: pb,
        p: function (a) {
          B && V.Fa[a >>> 0].ref();
        },
        t: function (a, b, c) {
          a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
          c >>>= 0;
          a = new Date(1e3 * a);
          p()[(c >> 2) >>> 0] = a.getUTCSeconds();
          p()[((c + 4) >> 2) >>> 0] = a.getUTCMinutes();
          p()[((c + 8) >> 2) >>> 0] = a.getUTCHours();
          p()[((c + 12) >> 2) >>> 0] = a.getUTCDate();
          p()[((c + 16) >> 2) >>> 0] = a.getUTCMonth();
          p()[((c + 20) >> 2) >>> 0] = a.getUTCFullYear() - 1900;
          p()[((c + 24) >> 2) >>> 0] = a.getUTCDay();
          a = ((a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864e5) | 0;
          p()[((c + 28) >> 2) >>> 0] = a;
        },
        u: function (a, b, c) {
          a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
          c >>>= 0;
          a = new Date(1e3 * a);
          p()[(c >> 2) >>> 0] = a.getSeconds();
          p()[((c + 4) >> 2) >>> 0] = a.getMinutes();
          p()[((c + 8) >> 2) >>> 0] = a.getHours();
          p()[((c + 12) >> 2) >>> 0] = a.getDate();
          p()[((c + 16) >> 2) >>> 0] = a.getMonth();
          p()[((c + 20) >> 2) >>> 0] = a.getFullYear() - 1900;
          p()[((c + 24) >> 2) >>> 0] = a.getDay();
          b = ((Z(a.getFullYear()) ? rb : sb)[a.getMonth()] + a.getDate() - 1) | 0;
          p()[((c + 28) >> 2) >>> 0] = b;
          p()[((c + 36) >> 2) >>> 0] = -(60 * a.getTimezoneOffset());
          b = new Date(a.getFullYear(), 6, 1).getTimezoneOffset();
          var e = new Date(a.getFullYear(), 0, 1).getTimezoneOffset();
          a = (b != e && a.getTimezoneOffset() == Math.min(e, b)) | 0;
          p()[((c + 32) >> 2) >>> 0] = a;
        },
        v: function (a) {
          a >>>= 0;
          var b = new Date(
              p()[((a + 20) >> 2) >>> 0] + 1900,
              p()[((a + 16) >> 2) >>> 0],
              p()[((a + 12) >> 2) >>> 0],
              p()[((a + 8) >> 2) >>> 0],
              p()[((a + 4) >> 2) >>> 0],
              p()[(a >> 2) >>> 0],
              0,
            ),
            c = p()[((a + 32) >> 2) >>> 0],
            e = b.getTimezoneOffset(),
            h = new Date(b.getFullYear(), 6, 1).getTimezoneOffset(),
            g = new Date(b.getFullYear(), 0, 1).getTimezoneOffset(),
            k = Math.min(g, h);
          0 > c
            ? (p()[((a + 32) >> 2) >>> 0] = Number(h != g && k == e))
            : 0 < c != (k == e) && ((h = Math.max(g, h)), b.setTime(b.getTime() + 6e4 * ((0 < c ? k : h) - e)));
          p()[((a + 24) >> 2) >>> 0] = b.getDay();
          c = ((Z(b.getFullYear()) ? rb : sb)[b.getMonth()] + b.getDate() - 1) | 0;
          p()[((a + 28) >> 2) >>> 0] = c;
          p()[(a >> 2) >>> 0] = b.getSeconds();
          p()[((a + 4) >> 2) >>> 0] = b.getMinutes();
          p()[((a + 8) >> 2) >>> 0] = b.getHours();
          p()[((a + 12) >> 2) >>> 0] = b.getDate();
          p()[((a + 16) >> 2) >>> 0] = b.getMonth();
          p()[((a + 20) >> 2) >>> 0] = b.getYear();
          a = b.getTime() / 1e3;
          return (
            Yb(
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
        r: tb,
        s: ub,
        z: function (a, b, c) {
          function e(v) {
            return (v = v.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? v[1] : "GMT";
          }
          a >>>= 0;
          b >>>= 0;
          c >>>= 0;
          var h = new Date().getFullYear(),
            g = new Date(h, 0, 1),
            k = new Date(h, 6, 1);
          h = g.getTimezoneOffset();
          var t = k.getTimezoneOffset(),
            C = Math.max(h, t);
          r()[(a >> 2) >>> 0] = 60 * C;
          p()[(b >> 2) >>> 0] = Number(h != t);
          a = e(g);
          b = e(k);
          a = wb(a);
          b = wb(b);
          t < h
            ? ((r()[(c >> 2) >>> 0] = a), (r()[((c + 4) >> 2) >>> 0] = b))
            : ((r()[(c >> 2) >>> 0] = b), (r()[((c + 4) >> 2) >>> 0] = a));
        },
        c: () => {
          K("");
        },
        l: function () {},
        i: function () {
          return Date.now();
        },
        V: () => {
          wa += 1;
          throw "unwind";
        },
        B: function () {
          return 4294901760;
        },
        e: () => performance.timeOrigin + performance.now(),
        f: function () {
          return B ? require("os").cpus().length : navigator.hardwareConcurrency;
        },
        K: function (a, b, c, e) {
          V.pb = b >>> 0;
          Bb.length = c;
          b = (e >>> 0) >> 3;
          for (e = 0; e < c; e++) Bb[e] = ea()[(b + e) >>> 0];
          return Wb[a].apply(null, Bb);
        },
        y: function (a) {
          a >>>= 0;
          var b = n().length;
          if (a <= b || 4294901760 < a) return !1;
          for (var c = 1; 4 >= c; c *= 2) {
            var e = b * (1 + 0.2 / c);
            e = Math.min(e, a + 100663296);
            var h = Math;
            e = Math.max(a, e);
            a: {
              h = (h.min.call(h, 4294901760, e + ((65536 - (e % 65536)) % 65536)) - d.buffer.byteLength + 65535) >>> 16;
              try {
                d.grow(h);
                m();
                var g = 1;
                break a;
              } catch (k) {}
              g = void 0;
            }
            if (g) return !0;
          }
          return !1;
        },
        O: Fb,
        P: Gb,
        j: Na,
        g: Hb,
        n: Ib,
        w: Mb,
        m: Ob,
        x: function (a, b) {
          a >>>= 0;
          b >>>= 0;
          Qb(n().subarray(a >>> 0, (a + b) >>> 0));
          return 0;
        },
        a: d || w.wasmMemory,
        H: Vb,
        d: function (a, b, c, e) {
          return Vb(a >>> 0, b >>> 0, c >>> 0, e >>> 0);
        },
      };
    (function () {
      function a(c, e) {
        c = c.exports;
        L = c = $b(c);
        V.Ta.push(L.sa);
        sa = L.ta;
        ua.unshift(L.W);
        ra = e;
        za();
        return c;
      }
      var b = { a: Zb };
      ya();
      if (w.instantiateWasm)
        try {
          return w.instantiateWasm(b, a);
        } catch (c) {
          I("Module.instantiateWasm callback failed with error: " + c), x(c);
        }
      Ea(b, function (c) {
        a(c.instance, c.module);
      }).catch(x);
      return {};
    })();
    w._OrtInit = (a, b) => (w._OrtInit = L.X)(a, b);
    w._OrtGetLastError = (a, b) => (w._OrtGetLastError = L.Y)(a, b);
    w._OrtCreateSessionOptions = (a, b, c, e, h, g, k, t, C, v) =>
      (w._OrtCreateSessionOptions = L.Z)(a, b, c, e, h, g, k, t, C, v);
    w._OrtAppendExecutionProvider = (a, b) => (w._OrtAppendExecutionProvider = L._)(a, b);
    w._OrtAddSessionConfigEntry = (a, b, c) => (w._OrtAddSessionConfigEntry = L.$)(a, b, c);
    w._OrtReleaseSessionOptions = (a) => (w._OrtReleaseSessionOptions = L.aa)(a);
    w._OrtCreateSession = (a, b, c) => (w._OrtCreateSession = L.ba)(a, b, c);
    w._OrtReleaseSession = (a) => (w._OrtReleaseSession = L.ca)(a);
    w._OrtGetInputOutputCount = (a, b, c) => (w._OrtGetInputOutputCount = L.da)(a, b, c);
    w._OrtGetInputName = (a, b) => (w._OrtGetInputName = L.ea)(a, b);
    w._OrtGetOutputName = (a, b) => (w._OrtGetOutputName = L.fa)(a, b);
    w._OrtFree = (a) => (w._OrtFree = L.ga)(a);
    w._OrtCreateTensor = (a, b, c, e, h) => (w._OrtCreateTensor = L.ha)(a, b, c, e, h);
    w._OrtGetTensorData = (a, b, c, e, h) => (w._OrtGetTensorData = L.ia)(a, b, c, e, h);
    w._OrtReleaseTensor = (a) => (w._OrtReleaseTensor = L.ja)(a);
    w._OrtCreateRunOptions = (a, b, c, e) => (w._OrtCreateRunOptions = L.ka)(a, b, c, e);
    w._OrtAddRunConfigEntry = (a, b, c) => (w._OrtAddRunConfigEntry = L.la)(a, b, c);
    w._OrtReleaseRunOptions = (a) => (w._OrtReleaseRunOptions = L.ma)(a);
    w._OrtRun = (a, b, c, e, h, g, k, t) => (w._OrtRun = L.na)(a, b, c, e, h, g, k, t);
    w._OrtEndProfiling = (a) => (w._OrtEndProfiling = L.oa)(a);
    var X = (w._pthread_self = () => (X = w._pthread_self = L.pa)()),
      vb = (w._malloc = (a) => (vb = w._malloc = L.qa)(a));
    w._free = (a) => (w._free = L.ra)(a);
    w.__emscripten_tls_init = () => (w.__emscripten_tls_init = L.sa)();
    var Xb = (w.__emscripten_thread_init = (a, b, c, e, h, g) =>
      (Xb = w.__emscripten_thread_init = L.ua)(a, b, c, e, h, g));
    w.__emscripten_thread_crashed = () => (w.__emscripten_thread_crashed = L.va)();
    var Ab = (a, b, c, e) => (Ab = L.wa)(a, b, c, e),
      Oa = (a) => (Oa = L.xa)(a),
      Ta = (w.__emscripten_thread_exit = (a) => (Ta = w.__emscripten_thread_exit = L.ya)(a)),
      qb = (w.__emscripten_check_mailbox = () => (qb = w.__emscripten_check_mailbox = L.za)()),
      Yb = (a) => (Yb = L.Aa)(a),
      Qa = (a, b) => (Qa = L.Ba)(a, b),
      xb = () => (xb = L.Ca)(),
      Ra = (a) => (Ra = L.Da)(a),
      zb = (a) => (zb = L.Ea)(a);
    function $b(a) {
      a = Object.assign({}, a);
      var b = (e) => () => e() >>> 0,
        c = (e) => (h) => e(h) >>> 0;
      a.__errno_location = b(a.__errno_location);
      a.pthread_self = b(a.pthread_self);
      a.malloc = c(a.malloc);
      a.stackSave = b(a.stackSave);
      a.stackAlloc = c(a.stackAlloc);
      return a;
    }
    w.keepRuntimeAlive = P;
    w.wasmMemory = d;
    w.stackAlloc = zb;
    w.stackSave = xb;
    w.stackRestore = Ra;
    w.UTF8ToString = Ka;
    w.stringToUTF8 = cb;
    w.lengthBytesUTF8 = ab;
    w.ExitStatus = U;
    w.PThread = V;
    var ac;
    R = function bc() {
      ac || cc();
      ac || (R = bc);
    };
    function cc() {
      function a() {
        if (!ac && ((ac = !0), (w.calledRun = !0), !M)) {
          D || Pa(ua);
          ha(w);
          if (w.onRuntimeInitialized) w.onRuntimeInitialized();
          if (!D) {
            if (w.postRun)
              for ("function" == typeof w.postRun && (w.postRun = [w.postRun]); w.postRun.length; ) {
                var b = w.postRun.shift();
                va.unshift(b);
              }
            Pa(va);
          }
        }
      }
      if (!(0 < Q))
        if (D) ha(w), D || Pa(ua), startWorker(w);
        else {
          if (w.preRun)
            for ("function" == typeof w.preRun && (w.preRun = [w.preRun]); w.preRun.length; )
              ta.unshift(w.preRun.shift());
          Pa(ta);
          0 < Q ||
            (w.setStatus
              ? (w.setStatus("Running..."),
                setTimeout(function () {
                  setTimeout(function () {
                    w.setStatus("");
                  }, 1);
                  a();
                }, 1))
              : a());
        }
    }
    if (w.preInit)
      for ("function" == typeof w.preInit && (w.preInit = [w.preInit]); 0 < w.preInit.length; ) w.preInit.pop()();
    cc();

    return moduleArg.ready;
  };
})();
if (typeof exports === "object" && typeof module === "object") module.exports = ortWasmThreaded;
else if (typeof define === "function" && define["amd"]) define([], () => ortWasmThreaded);
