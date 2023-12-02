var ortWasm = (() => {
  var _scriptDir = typeof document !== "undefined" && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== "undefined") _scriptDir = _scriptDir || __filename;
  return function (moduleArg = {}) {
    var e = moduleArg,
      aa,
      h;
    e.ready = new Promise((a, b) => {
      aa = a;
      h = b;
    });
    var ba = Object.assign({}, e),
      m = "./this.program",
      q = (a, b) => {
        throw b;
      },
      ca = "object" == typeof window,
      v = "function" == typeof importScripts,
      x = "object" == typeof process && "object" == typeof process.versions && "string" == typeof process.versions.node,
      y = "",
      A,
      B,
      C;
    if (x) {
      var fs = require("fs"),
        D = require("path");
      y = v ? D.dirname(y) + "/" : __dirname + "/";
      A = (a, b) => {
        a = a.startsWith("file://") ? new URL(a) : D.normalize(a);
        return fs.readFileSync(a, b ? void 0 : "utf8");
      };
      C = (a) => {
        a = A(a, !0);
        a.buffer || (a = new Uint8Array(a));
        return a;
      };
      B = (a, b, c, f = !0) => {
        a = a.startsWith("file://") ? new URL(a) : D.normalize(a);
        fs.readFile(a, f ? void 0 : "utf8", (g, k) => {
          g ? c(g) : b(f ? k.buffer : k);
        });
      };
      !e.thisProgram && 1 < process.argv.length && (m = process.argv[1].replace(/\\/g, "/"));
      process.argv.slice(2);
      q = (a, b) => {
        process.exitCode = a;
        throw b;
      };
      e.inspect = () => "[Emscripten Module object]";
    } else if (ca || v)
      v
        ? (y = self.location.href)
        : "undefined" != typeof document && document.currentScript && (y = document.currentScript.src),
        _scriptDir && (y = _scriptDir),
        0 !== y.indexOf("blob:") ? (y = y.substr(0, y.replace(/[?#].*/, "").lastIndexOf("/") + 1)) : (y = ""),
        (A = (a) => {
          var b = new XMLHttpRequest();
          b.open("GET", a, !1);
          b.send(null);
          return b.responseText;
        }),
        v &&
          (C = (a) => {
            var b = new XMLHttpRequest();
            b.open("GET", a, !1);
            b.responseType = "arraybuffer";
            b.send(null);
            return new Uint8Array(b.response);
          }),
        (B = (a, b, c) => {
          var f = new XMLHttpRequest();
          f.open("GET", a, !0);
          f.responseType = "arraybuffer";
          f.onload = () => {
            200 == f.status || (0 == f.status && f.response) ? b(f.response) : c();
          };
          f.onerror = c;
          f.send(null);
        });
    var da = e.print || console.log.bind(console),
      E = e.printErr || console.error.bind(console);
    Object.assign(e, ba);
    ba = null;
    e.thisProgram && (m = e.thisProgram);
    e.quit && (q = e.quit);
    var F;
    e.wasmBinary && (F = e.wasmBinary);
    var noExitRuntime = e.noExitRuntime || !0;
    "object" != typeof WebAssembly && G("no native wasm support detected");
    var H,
      I,
      J = !1,
      K,
      L,
      M,
      N;
    function ea() {
      var a = H.buffer;
      e.HEAP8 = K = new Int8Array(a);
      e.HEAP16 = new Int16Array(a);
      e.HEAP32 = M = new Int32Array(a);
      e.HEAPU8 = L = new Uint8Array(a);
      e.HEAPU16 = new Uint16Array(a);
      e.HEAPU32 = N = new Uint32Array(a);
      e.HEAPF32 = new Float32Array(a);
      e.HEAPF64 = new Float64Array(a);
    }
    var fa = [],
      ha = [],
      ia = [];
    function ja() {
      var a = e.preRun.shift();
      fa.unshift(a);
    }
    var O = 0,
      P = null,
      Q = null;
    function G(a) {
      if (e.onAbort) e.onAbort(a);
      a = "Aborted(" + a + ")";
      E(a);
      J = !0;
      a = new WebAssembly.RuntimeError(a + ". Build with -sASSERTIONS for more info.");
      h(a);
      throw a;
    }
    function ka(a) {
      return a.startsWith("data:application/octet-stream;base64,");
    }
    var R;
    R = "ort-wasm-simd.wasm";
    if (!ka(R)) {
      var la = R;
      R = e.locateFile ? e.locateFile(la, y) : y + la;
    }
    function ma(a) {
      if (a == R && F) return new Uint8Array(F);
      if (C) return C(a);
      throw "both async and sync fetching of the wasm failed";
    }
    function na(a) {
      if (!F && (ca || v)) {
        if ("function" == typeof fetch && !a.startsWith("file://"))
          return fetch(a, { credentials: "same-origin" })
            .then((b) => {
              if (!b.ok) throw "failed to load wasm binary file at '" + a + "'";
              return b.arrayBuffer();
            })
            .catch(() => ma(a));
        if (B)
          return new Promise((b, c) => {
            B(a, (f) => b(new Uint8Array(f)), c);
          });
      }
      return Promise.resolve().then(() => ma(a));
    }
    function oa(a, b, c) {
      return na(a)
        .then((f) => WebAssembly.instantiate(f, b))
        .then((f) => f)
        .then(c, (f) => {
          E("failed to asynchronously prepare wasm: " + f);
          G(f);
        });
    }
    function pa(a, b) {
      var c = R;
      return F ||
        "function" != typeof WebAssembly.instantiateStreaming ||
        ka(c) ||
        c.startsWith("file://") ||
        x ||
        "function" != typeof fetch
        ? oa(c, a, b)
        : fetch(c, { credentials: "same-origin" }).then((f) =>
            WebAssembly.instantiateStreaming(f, a).then(b, function (g) {
              E("wasm streaming compile failed: " + g);
              E("falling back to ArrayBuffer instantiation");
              return oa(c, a, b);
            }),
          );
    }
    var S;
    function qa(a) {
      this.name = "ExitStatus";
      this.message = `Program terminated with exit(${a})`;
      this.status = a;
    }
    var T = (a) => {
      for (; 0 < a.length; ) a.shift()(e);
    };
    function ra(a) {
      this.qa = a - 24;
      this.va = function (b) {
        N[((this.qa + 4) >> 2) >>> 0] = b;
      };
      this.ua = function (b) {
        N[((this.qa + 8) >> 2) >>> 0] = b;
      };
      this.sa = function (b, c) {
        this.ta();
        this.va(b);
        this.ua(c);
      };
      this.ta = function () {
        N[((this.qa + 16) >> 2) >>> 0] = 0;
      };
    }
    var sa = 0,
      ta = 0,
      ua = "undefined" != typeof TextDecoder ? new TextDecoder("utf8") : void 0,
      va = (a, b, c) => {
        b >>>= 0;
        var f = b + c;
        for (c = b; a[c] && !(c >= f); ) ++c;
        if (16 < c - b && a.buffer && ua) return ua.decode(a.subarray(b, c));
        for (f = ""; b < c; ) {
          var g = a[b++];
          if (g & 128) {
            var k = a[b++] & 63;
            if (192 == (g & 224)) f += String.fromCharCode(((g & 31) << 6) | k);
            else {
              var l = a[b++] & 63;
              g =
                224 == (g & 240)
                  ? ((g & 15) << 12) | (k << 6) | l
                  : ((g & 7) << 18) | (k << 12) | (l << 6) | (a[b++] & 63);
              65536 > g
                ? (f += String.fromCharCode(g))
                : ((g -= 65536), (f += String.fromCharCode(55296 | (g >> 10), 56320 | (g & 1023))));
            }
          } else f += String.fromCharCode(g);
        }
        return f;
      },
      U = (a, b) => ((a >>>= 0) ? va(L, a, b) : ""),
      V = (a) => {
        for (var b = 0, c = 0; c < a.length; ++c) {
          var f = a.charCodeAt(c);
          127 >= f ? b++ : 2047 >= f ? (b += 2) : 55296 <= f && 57343 >= f ? ((b += 4), ++c) : (b += 3);
        }
        return b;
      },
      W = (a, b, c, f) => {
        c >>>= 0;
        if (!(0 < f)) return 0;
        var g = c;
        f = c + f - 1;
        for (var k = 0; k < a.length; ++k) {
          var l = a.charCodeAt(k);
          if (55296 <= l && 57343 >= l) {
            var r = a.charCodeAt(++k);
            l = (65536 + ((l & 1023) << 10)) | (r & 1023);
          }
          if (127 >= l) {
            if (c >= f) break;
            b[c++ >>> 0] = l;
          } else {
            if (2047 >= l) {
              if (c + 1 >= f) break;
              b[c++ >>> 0] = 192 | (l >> 6);
            } else {
              if (65535 >= l) {
                if (c + 2 >= f) break;
                b[c++ >>> 0] = 224 | (l >> 12);
              } else {
                if (c + 3 >= f) break;
                b[c++ >>> 0] = 240 | (l >> 18);
                b[c++ >>> 0] = 128 | ((l >> 12) & 63);
              }
              b[c++ >>> 0] = 128 | ((l >> 6) & 63);
            }
            b[c++ >>> 0] = 128 | (l & 63);
          }
        }
        b[c >>> 0] = 0;
        return c - g;
      },
      X = (a) => 0 === a % 4 && (0 !== a % 100 || 0 === a % 400),
      wa = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
      xa = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
      Ca = (a) => {
        var b = V(a) + 1,
          c = ya(b);
        c && W(a, L, c, b);
        return c;
      },
      Y = {},
      Ea = () => {
        if (!Da) {
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
              _: m || "./this.program",
            },
            b;
          for (b in Y) void 0 === Y[b] ? delete a[b] : (a[b] = Y[b]);
          var c = [];
          for (b in a) c.push(`${b}=${a[b]}`);
          Da = c;
        }
        return Da;
      },
      Da,
      Fa = [null, [], []],
      Ga = () => {
        if ("object" == typeof crypto && "function" == typeof crypto.getRandomValues)
          return (c) => crypto.getRandomValues(c);
        if (x)
          try {
            var a = require("crypto");
            if (a.randomFillSync) return (c) => a.randomFillSync(c);
            var b = a.randomBytes;
            return (c) => (c.set(b(c.byteLength)), c);
          } catch (c) {}
        G("initRandomDevice");
      },
      Ha = (a) => (Ha = Ga())(a),
      Ia = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
      Ja = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    function Ka(a) {
      var b = Array(V(a) + 1);
      W(a, b, 0, b.length);
      return b;
    }
    function La(a, b, c, f) {
      function g(d, n, p) {
        for (d = "number" == typeof d ? d.toString() : d || ""; d.length < n; ) d = p[0] + d;
        return d;
      }
      function k(d, n) {
        return g(d, n, "0");
      }
      function l(d, n) {
        function p(za) {
          return 0 > za ? -1 : 0 < za ? 1 : 0;
        }
        var z;
        0 === (z = p(d.getFullYear() - n.getFullYear())) &&
          0 === (z = p(d.getMonth() - n.getMonth())) &&
          (z = p(d.getDate() - n.getDate()));
        return z;
      }
      function r(d) {
        switch (d.getDay()) {
          case 0:
            return new Date(d.getFullYear() - 1, 11, 29);
          case 1:
            return d;
          case 2:
            return new Date(d.getFullYear(), 0, 3);
          case 3:
            return new Date(d.getFullYear(), 0, 2);
          case 4:
            return new Date(d.getFullYear(), 0, 1);
          case 5:
            return new Date(d.getFullYear() - 1, 11, 31);
          case 6:
            return new Date(d.getFullYear() - 1, 11, 30);
        }
      }
      function w(d) {
        var n = d.ma;
        for (d = new Date(new Date(d.na + 1900, 0, 1).getTime()); 0 < n; ) {
          var p = d.getMonth(),
            z = (X(d.getFullYear()) ? Ia : Ja)[p];
          if (n > z - d.getDate())
            (n -= z - d.getDate() + 1),
              d.setDate(1),
              11 > p ? d.setMonth(p + 1) : (d.setMonth(0), d.setFullYear(d.getFullYear() + 1));
          else {
            d.setDate(d.getDate() + n);
            break;
          }
        }
        p = new Date(d.getFullYear() + 1, 0, 4);
        n = r(new Date(d.getFullYear(), 0, 4));
        p = r(p);
        return 0 >= l(n, d) ? (0 >= l(p, d) ? d.getFullYear() + 1 : d.getFullYear()) : d.getFullYear() - 1;
      }
      a >>>= 0;
      b >>>= 0;
      c >>>= 0;
      f >>>= 0;
      var t = M[((f + 40) >> 2) >>> 0];
      f = {
        ya: M[(f >> 2) >>> 0],
        xa: M[((f + 4) >> 2) >>> 0],
        oa: M[((f + 8) >> 2) >>> 0],
        ra: M[((f + 12) >> 2) >>> 0],
        pa: M[((f + 16) >> 2) >>> 0],
        na: M[((f + 20) >> 2) >>> 0],
        ha: M[((f + 24) >> 2) >>> 0],
        ma: M[((f + 28) >> 2) >>> 0],
        Aa: M[((f + 32) >> 2) >>> 0],
        wa: M[((f + 36) >> 2) >>> 0],
        za: t ? U(t) : "",
      };
      c = U(c);
      t = {
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
      for (var u in t) c = c.replace(new RegExp(u, "g"), t[u]);
      var Aa = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        Ba = "January February March April May June July August September October November December".split(" ");
      t = {
        "%a": (d) => Aa[d.ha].substring(0, 3),
        "%A": (d) => Aa[d.ha],
        "%b": (d) => Ba[d.pa].substring(0, 3),
        "%B": (d) => Ba[d.pa],
        "%C": (d) => k(((d.na + 1900) / 100) | 0, 2),
        "%d": (d) => k(d.ra, 2),
        "%e": (d) => g(d.ra, 2, " "),
        "%g": (d) => w(d).toString().substring(2),
        "%G": (d) => w(d),
        "%H": (d) => k(d.oa, 2),
        "%I": (d) => {
          d = d.oa;
          0 == d ? (d = 12) : 12 < d && (d -= 12);
          return k(d, 2);
        },
        "%j": (d) => {
          for (var n = 0, p = 0; p <= d.pa - 1; n += (X(d.na + 1900) ? Ia : Ja)[p++]);
          return k(d.ra + n, 3);
        },
        "%m": (d) => k(d.pa + 1, 2),
        "%M": (d) => k(d.xa, 2),
        "%n": () => "\n",
        "%p": (d) => (0 <= d.oa && 12 > d.oa ? "AM" : "PM"),
        "%S": (d) => k(d.ya, 2),
        "%t": () => "\t",
        "%u": (d) => d.ha || 7,
        "%U": (d) => k(Math.floor((d.ma + 7 - d.ha) / 7), 2),
        "%V": (d) => {
          var n = Math.floor((d.ma + 7 - ((d.ha + 6) % 7)) / 7);
          2 >= (d.ha + 371 - d.ma - 2) % 7 && n++;
          if (n) 53 == n && ((p = (d.ha + 371 - d.ma) % 7), 4 == p || (3 == p && X(d.na)) || (n = 1));
          else {
            n = 52;
            var p = (d.ha + 7 - d.ma - 1) % 7;
            (4 == p || (5 == p && X((d.na % 400) - 1))) && n++;
          }
          return k(n, 2);
        },
        "%w": (d) => d.ha,
        "%W": (d) => k(Math.floor((d.ma + 7 - ((d.ha + 6) % 7)) / 7), 2),
        "%y": (d) => (d.na + 1900).toString().substring(2),
        "%Y": (d) => d.na + 1900,
        "%z": (d) => {
          d = d.wa;
          var n = 0 <= d;
          d = Math.abs(d) / 60;
          return (n ? "+" : "-") + String("0000" + ((d / 60) * 100 + (d % 60))).slice(-4);
        },
        "%Z": (d) => d.za,
        "%%": () => "%",
      };
      c = c.replace(/%%/g, "\x00\x00");
      for (u in t) c.includes(u) && (c = c.replace(new RegExp(u, "g"), t[u](f)));
      c = c.replace(/\0\0/g, "%");
      u = Ka(c);
      if (u.length > b) return 0;
      K.set(u, a >>> 0);
      return u.length - 1;
    }
    var Na = {
      a: function (a, b, c) {
        a >>>= 0;
        new ra(a).sa(b >>> 0, c >>> 0);
        sa = a;
        ta++;
        throw sa;
      },
      e: function () {
        return 0;
      },
      I: function () {},
      y: function () {},
      A: function () {},
      K: function () {
        return 0;
      },
      G: function () {},
      B: function () {},
      F: function () {},
      g: function () {},
      z: function () {},
      w: function () {},
      H: function () {},
      x: function () {},
      k: () => !0,
      n: function (a, b, c) {
        a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
        c >>>= 0;
        a = new Date(1e3 * a);
        M[(c >> 2) >>> 0] = a.getUTCSeconds();
        M[((c + 4) >> 2) >>> 0] = a.getUTCMinutes();
        M[((c + 8) >> 2) >>> 0] = a.getUTCHours();
        M[((c + 12) >> 2) >>> 0] = a.getUTCDate();
        M[((c + 16) >> 2) >>> 0] = a.getUTCMonth();
        M[((c + 20) >> 2) >>> 0] = a.getUTCFullYear() - 1900;
        M[((c + 24) >> 2) >>> 0] = a.getUTCDay();
        M[((c + 28) >> 2) >>> 0] = ((a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864e5) | 0;
      },
      o: function (a, b, c) {
        a = (b + 2097152) >>> 0 < 4194305 - !!a ? (a >>> 0) + 4294967296 * b : NaN;
        c >>>= 0;
        a = new Date(1e3 * a);
        M[(c >> 2) >>> 0] = a.getSeconds();
        M[((c + 4) >> 2) >>> 0] = a.getMinutes();
        M[((c + 8) >> 2) >>> 0] = a.getHours();
        M[((c + 12) >> 2) >>> 0] = a.getDate();
        M[((c + 16) >> 2) >>> 0] = a.getMonth();
        M[((c + 20) >> 2) >>> 0] = a.getFullYear() - 1900;
        M[((c + 24) >> 2) >>> 0] = a.getDay();
        M[((c + 28) >> 2) >>> 0] = ((X(a.getFullYear()) ? wa : xa)[a.getMonth()] + a.getDate() - 1) | 0;
        M[((c + 36) >> 2) >>> 0] = -(60 * a.getTimezoneOffset());
        b = new Date(a.getFullYear(), 6, 1).getTimezoneOffset();
        var f = new Date(a.getFullYear(), 0, 1).getTimezoneOffset();
        M[((c + 32) >> 2) >>> 0] = (b != f && a.getTimezoneOffset() == Math.min(f, b)) | 0;
      },
      p: function (a) {
        a >>>= 0;
        var b = new Date(
            M[((a + 20) >> 2) >>> 0] + 1900,
            M[((a + 16) >> 2) >>> 0],
            M[((a + 12) >> 2) >>> 0],
            M[((a + 8) >> 2) >>> 0],
            M[((a + 4) >> 2) >>> 0],
            M[(a >> 2) >>> 0],
            0,
          ),
          c = M[((a + 32) >> 2) >>> 0],
          f = b.getTimezoneOffset(),
          g = new Date(b.getFullYear(), 6, 1).getTimezoneOffset(),
          k = new Date(b.getFullYear(), 0, 1).getTimezoneOffset(),
          l = Math.min(k, g);
        0 > c
          ? (M[((a + 32) >> 2) >>> 0] = Number(g != k && l == f))
          : 0 < c != (l == f) && ((g = Math.max(k, g)), b.setTime(b.getTime() + 6e4 * ((0 < c ? l : g) - f)));
        M[((a + 24) >> 2) >>> 0] = b.getDay();
        M[((a + 28) >> 2) >>> 0] = ((X(b.getFullYear()) ? wa : xa)[b.getMonth()] + b.getDate() - 1) | 0;
        M[(a >> 2) >>> 0] = b.getSeconds();
        M[((a + 4) >> 2) >>> 0] = b.getMinutes();
        M[((a + 8) >> 2) >>> 0] = b.getHours();
        M[((a + 12) >> 2) >>> 0] = b.getDate();
        M[((a + 16) >> 2) >>> 0] = b.getMonth();
        M[((a + 20) >> 2) >>> 0] = b.getYear();
        a = b.getTime() / 1e3;
        return (
          Ma(
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
      l: function () {
        return -52;
      },
      m: function () {},
      u: function (a, b, c) {
        function f(w) {
          return (w = w.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? w[1] : "GMT";
        }
        c >>>= 0;
        var g = new Date().getFullYear(),
          k = new Date(g, 0, 1),
          l = new Date(g, 6, 1);
        g = k.getTimezoneOffset();
        var r = l.getTimezoneOffset();
        N[((a >>> 0) >> 2) >>> 0] = 60 * Math.max(g, r);
        M[((b >>> 0) >> 2) >>> 0] = Number(g != r);
        a = f(k);
        b = f(l);
        a = Ca(a);
        b = Ca(b);
        r < g
          ? ((N[(c >> 2) >>> 0] = a), (N[((c + 4) >> 2) >>> 0] = b))
          : ((N[(c >> 2) >>> 0] = b), (N[((c + 4) >> 2) >>> 0] = a));
      },
      d: () => {
        G("");
      },
      h: function () {
        return Date.now();
      },
      v: function () {
        return 4294901760;
      },
      b: () => performance.now(),
      J: function (a, b, c) {
        b >>>= 0;
        return L.copyWithin((a >>> 0) >>> 0, b >>> 0, (b + (c >>> 0)) >>> 0);
      },
      t: function (a) {
        a >>>= 0;
        var b = L.length;
        if (4294901760 < a) return !1;
        for (var c = 1; 4 >= c; c *= 2) {
          var f = b * (1 + 0.2 / c);
          f = Math.min(f, a + 100663296);
          var g = Math;
          f = Math.max(a, f);
          a: {
            g = (g.min.call(g, 4294901760, f + ((65536 - (f % 65536)) % 65536)) - H.buffer.byteLength + 65535) >>> 16;
            try {
              H.grow(g);
              ea();
              var k = 1;
              break a;
            } catch (l) {}
            k = void 0;
          }
          if (k) return !0;
        }
        return !1;
      },
      D: function (a, b) {
        a >>>= 0;
        b >>>= 0;
        var c = 0;
        Ea().forEach(function (f, g) {
          var k = b + c;
          g = N[((a + 4 * g) >> 2) >>> 0] = k;
          for (k = 0; k < f.length; ++k) K[(g++ >> 0) >>> 0] = f.charCodeAt(k);
          K[(g >> 0) >>> 0] = 0;
          c += f.length + 1;
        });
        return 0;
      },
      E: function (a, b) {
        a >>>= 0;
        b >>>= 0;
        var c = Ea();
        N[(a >> 2) >>> 0] = c.length;
        var f = 0;
        c.forEach(function (g) {
          f += g.length + 1;
        });
        N[(b >> 2) >>> 0] = f;
        return 0;
      },
      s: (a) => {
        if (!noExitRuntime) {
          if (e.onExit) e.onExit(a);
          J = !0;
        }
        q(a, new qa(a));
      },
      f: () => 52,
      j: function () {
        return 52;
      },
      q: function () {
        return 70;
      },
      i: function (a, b, c, f) {
        b >>>= 0;
        c >>>= 0;
        f >>>= 0;
        for (var g = 0, k = 0; k < c; k++) {
          var l = N[(b >> 2) >>> 0],
            r = N[((b + 4) >> 2) >>> 0];
          b += 8;
          for (var w = 0; w < r; w++) {
            var t = L[(l + w) >>> 0],
              u = Fa[a];
            0 === t || 10 === t ? ((1 === a ? da : E)(va(u, 0)), (u.length = 0)) : u.push(t);
          }
          g += r;
        }
        N[(f >> 2) >>> 0] = g;
        return 0;
      },
      r: function (a, b) {
        a >>>= 0;
        Ha(L.subarray(a >>> 0, (a + (b >>> 0)) >>> 0));
        return 0;
      },
      C: La,
      c: function (a, b, c, f) {
        return La(a >>> 0, b >>> 0, c >>> 0, f >>> 0);
      },
    };
    (function () {
      function a(c) {
        c = c.exports;
        I = c = Oa(c);
        H = I.L;
        ea();
        ha.unshift(I.M);
        O--;
        e.monitorRunDependencies && e.monitorRunDependencies(O);
        if (0 == O && (null !== P && (clearInterval(P), (P = null)), Q)) {
          var f = Q;
          Q = null;
          f();
        }
        return c;
      }
      var b = { a: Na };
      O++;
      e.monitorRunDependencies && e.monitorRunDependencies(O);
      if (e.instantiateWasm)
        try {
          return e.instantiateWasm(b, a);
        } catch (c) {
          E("Module.instantiateWasm callback failed with error: " + c), h(c);
        }
      pa(b, function (c) {
        a(c.instance);
      }).catch(h);
      return {};
    })();
    e._OrtInit = (a, b) => (e._OrtInit = I.N)(a, b);
    e._OrtGetLastError = (a, b) => (e._OrtGetLastError = I.O)(a, b);
    e._OrtCreateSessionOptions = (a, b, c, f, g, k, l, r, w, t) =>
      (e._OrtCreateSessionOptions = I.P)(a, b, c, f, g, k, l, r, w, t);
    e._OrtAppendExecutionProvider = (a, b) => (e._OrtAppendExecutionProvider = I.Q)(a, b);
    e._OrtAddSessionConfigEntry = (a, b, c) => (e._OrtAddSessionConfigEntry = I.R)(a, b, c);
    e._OrtReleaseSessionOptions = (a) => (e._OrtReleaseSessionOptions = I.S)(a);
    e._OrtCreateSession = (a, b, c) => (e._OrtCreateSession = I.T)(a, b, c);
    e._OrtReleaseSession = (a) => (e._OrtReleaseSession = I.U)(a);
    e._OrtGetInputOutputCount = (a, b, c) => (e._OrtGetInputOutputCount = I.V)(a, b, c);
    e._OrtGetInputName = (a, b) => (e._OrtGetInputName = I.W)(a, b);
    e._OrtGetOutputName = (a, b) => (e._OrtGetOutputName = I.X)(a, b);
    e._OrtFree = (a) => (e._OrtFree = I.Y)(a);
    e._OrtCreateTensor = (a, b, c, f, g) => (e._OrtCreateTensor = I.Z)(a, b, c, f, g);
    e._OrtGetTensorData = (a, b, c, f, g) => (e._OrtGetTensorData = I._)(a, b, c, f, g);
    e._OrtReleaseTensor = (a) => (e._OrtReleaseTensor = I.$)(a);
    e._OrtCreateRunOptions = (a, b, c, f) => (e._OrtCreateRunOptions = I.aa)(a, b, c, f);
    e._OrtAddRunConfigEntry = (a, b, c) => (e._OrtAddRunConfigEntry = I.ba)(a, b, c);
    e._OrtReleaseRunOptions = (a) => (e._OrtReleaseRunOptions = I.ca)(a);
    e._OrtRun = (a, b, c, f, g, k, l, r) => (e._OrtRun = I.da)(a, b, c, f, g, k, l, r);
    e._OrtEndProfiling = (a) => (e._OrtEndProfiling = I.ea)(a);
    var ya = (e._malloc = (a) => (ya = e._malloc = I.fa)(a));
    e._free = (a) => (e._free = I.ga)(a);
    var Ma = (a) => (Ma = I.ia)(a),
      Pa = () => (Pa = I.ja)(),
      Qa = (a) => (Qa = I.ka)(a),
      Ra = (a) => (Ra = I.la)(a);
    function Oa(a) {
      a = Object.assign({}, a);
      var b = (f) => () => f() >>> 0,
        c = (f) => (g) => f(g) >>> 0;
      a.__errno_location = b(a.__errno_location);
      a.malloc = c(a.malloc);
      a.stackSave = b(a.stackSave);
      a.stackAlloc = c(a.stackAlloc);
      return a;
    }
    e.stackAlloc = Ra;
    e.stackSave = Pa;
    e.stackRestore = Qa;
    e.UTF8ToString = U;
    e.stringToUTF8 = (a, b, c) => W(a, L, b, c);
    e.lengthBytesUTF8 = V;
    var Z;
    Q = function Sa() {
      Z || Ta();
      Z || (Q = Sa);
    };
    function Ta() {
      function a() {
        if (!Z && ((Z = !0), (e.calledRun = !0), !J)) {
          T(ha);
          aa(e);
          if (e.onRuntimeInitialized) e.onRuntimeInitialized();
          if (e.postRun)
            for ("function" == typeof e.postRun && (e.postRun = [e.postRun]); e.postRun.length; ) {
              var b = e.postRun.shift();
              ia.unshift(b);
            }
          T(ia);
        }
      }
      if (!(0 < O)) {
        if (e.preRun) for ("function" == typeof e.preRun && (e.preRun = [e.preRun]); e.preRun.length; ) ja();
        T(fa);
        0 < O ||
          (e.setStatus
            ? (e.setStatus("Running..."),
              setTimeout(function () {
                setTimeout(function () {
                  e.setStatus("");
                }, 1);
                a();
              }, 1))
            : a());
      }
    }
    if (e.preInit)
      for ("function" == typeof e.preInit && (e.preInit = [e.preInit]); 0 < e.preInit.length; ) e.preInit.pop()();
    Ta();

    return moduleArg.ready;
  };
})();
if (typeof exports === "object" && typeof module === "object") module.exports = ortWasm;
else if (typeof define === "function" && define["amd"]) define([], () => ortWasm);
