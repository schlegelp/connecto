/* The crystal ball.
 *
 * A single full-screen fragment shader: a glass sphere with a swirling interior,
 * refracting a field of drifting motes behind it. One draw call, no dependencies.
 *
 * Loaded on every page (so instant navigation can never swap it out) and gated on
 * the presence of #cn-orb, which only the landing page renders. A MutationObserver
 * re-initialises when instant navigation swaps the container in or out.
 */
(function () {
  "use strict";

  var VERT = [
    "attribute vec2 aPos;",
    "void main() { gl_Position = vec4(aPos, 0.0, 1.0); }"
  ].join("\n");

  var FRAG = [
    "precision highp float;",
    "",
    "uniform vec2  uRes;",
    "uniform float uTime;",
    "uniform vec2  uMouse;",
    "uniform vec2  uCenter;",
    "uniform float uRadius;",
    "",
    "float hash13(vec3 p3) {",
    "  p3 = fract(p3 * 0.1031);",
    "  p3 += dot(p3, p3.zyx + 31.32);",
    "  return fract((p3.x + p3.y) * p3.z);",
    "}",
    "vec2 hash22(vec2 p) {",
    "  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));",
    "  p3 += dot(p3, p3.yzx + 33.33);",
    "  return fract((p3.xx + p3.yz) * p3.zy);",
    "}",
    "",
    "float noise(vec3 p) {",
    "  vec3 i = floor(p);",
    "  vec3 f = fract(p);",
    "  f = f * f * (3.0 - 2.0 * f);",
    "  return mix(",
    "    mix(mix(hash13(i + vec3(0.0, 0.0, 0.0)), hash13(i + vec3(1.0, 0.0, 0.0)), f.x),",
    "        mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),",
    "    mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),",
    "        mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y),",
    "    f.z);",
    "}",
    "",
    "float fbm(vec3 p) {",
    "  float sum = 0.0;",
    "  float amp = 0.5;",
    "  for (int i = 0; i < 4; i++) {",
    "    sum += amp * noise(p);",
    "    p *= 2.03;",
    "    amp *= 0.5;",
    "  }",
    "  return sum;",
    "}",
    "",
    "float segDist(vec2 p, vec2 a, vec2 b) {",
    "  vec2 pa = p - a, ba = b - a;",
    "  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);",
    "  return length(pa - ba * h);",
    "}",
    "",
    // The thing the crystal ball actually shows you: a little connectome,
    // suspended in the glass. Eight nodes on a Fibonacci sphere, turning.
    "vec3 nodePos(int i, float t) {",
    "  float fi = float(i);",
    "  float a = fi * 2.399963;",
    "  float y = 1.0 - 2.0 * (fi + 0.5) / 8.0;",
    "  float r = sqrt(max(1.0 - y * y, 0.0));",
    "  vec3 p = vec3(cos(a) * r, y, sin(a) * r) * 0.60;",
    "  float s = sin(t * 0.22), c = cos(t * 0.22);",
    "  p.xz = mat2(c, -s, s, c) * p.xz;",
    "  p *= 0.93 + 0.07 * sin(t * 0.6 + fi * 1.7);",
    "  return p;",
    "}",
    "",
    // Drifting motes on three parallax layers. These are what the ball refracts -
    // dust bending as it passes through the glass is what sells it as a sphere
    // rather than a flat disc.
    "vec3 motes(vec2 uv, float t) {",
    "  vec3 acc = vec3(0.0);",
    "  for (int L = 0; L < 3; L++) {",
    "    float fl = float(L);",
    "    float scale = 5.0 + fl * 4.5;",
    "    vec2 p = uv * scale + vec2(t * (0.010 + fl * 0.005), t * (0.020 + fl * 0.004));",
    "    vec2 gv = fract(p) - 0.5;",
    "    vec2 id = floor(p);",
    "    for (int y = -1; y <= 1; y++) {",
    "      for (int x = -1; x <= 1; x++) {",
    "        vec2 off = vec2(float(x), float(y));",
    "        vec2 h = hash22(id + off + fl * 41.0);",
    "        float seed = hash13(vec3(id + off, fl * 7.0));",
    "        float present = step(0.50, seed);",
    "        vec2 pos = off + (h - 0.5) * 0.7;",
    "        pos += 0.13 * vec2(sin(t * 0.55 + seed * 31.0), cos(t * 0.47 + seed * 23.0));",
    "        float d = length(gv - pos);",
    "        float size = mix(0.008, 0.026, h.y);",
    "        float core = smoothstep(size, 0.0, d);",
    "        float halo = exp(-d * 22.0) * 0.30;",
    "        float twinkle = 0.6 + 0.4 * sin(t * 1.4 + seed * 40.0);",
    "        vec3 tint = mix(vec3(0.58, 0.66, 1.00), vec3(1.00, 0.80, 0.52), seed);",
    "        acc += tint * (core + halo) * present * twinkle * (0.50 - fl * 0.12);",
    "      }",
    "    }",
    "  }",
    "  return acc;",
    "}",
    "",
    "vec3 background(vec2 uv, float t) {",
    "  float v = uv.y * 0.5 + 0.5;",
    "  vec3 col = mix(vec3(0.018, 0.017, 0.032), vec3(0.048, 0.036, 0.090), v);",
    "  float d = length(uv - uCenter);",
    "  col += vec3(0.30, 0.17, 0.78) * exp(-d * 1.6) * 0.20;",
    "  col += motes(uv, t);",
    "  return col;",
    "}",
    "",
    "void main() {",
    "  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;",
    "  float t = uTime;",
    "",
    "  vec2  c = uCenter + uMouse * 0.014;",
    "  float R = uRadius;",
    "  vec2  q = uv - c;",
    "  float d = length(q);",
    "",
    "  vec3 col = background(uv - uMouse * 0.022, t);",
    "",
    "  float z = sqrt(max(R * R - d * d, 0.0));",
    // normalize(), not /R. Inside the sphere the two agree (q^2 + z^2 == R^2), but
    // *outside* it z is 0 and |q| > R, so dividing by R leaves a normal longer than
    // one - and pow(dot(n, H), 500.0) on it overflows to +Inf. mix(col, glass, 0.0)
    // then evaluates 0 * Inf = NaN and the fragment renders black. Because dot(n, H)
    // is linear in q, the locus where it crosses 1 is a straight line: you get a
    // razor-sharp black wedge across the backdrop. normalize() keeps it unit
    // everywhere and the whole failure mode goes away.
    "  vec3  n = normalize(vec3(q, z));",
    "",
    // The dust behind the glass, bent on its way through. Refraction is what
    // makes it read as a sphere rather than a flat disc, so it gets to dominate.
    "  vec3 refr = refract(vec3(0.0, 0.0, -1.0), n, 1.0 / 1.5);",
    "  vec3 behind = background(uv + refr.xy * 0.45 - uMouse * 0.022, t) * 1.35;",
    "",
    // The interior: a wispy haze, kept thin enough to see through.
    "  vec3 sp = vec3(q / R, z / R);",
    "  float a = t * 0.09;",
    "  sp.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * sp.xy;",
    "",
    "  float f1 = fbm(sp * 2.1 + vec3(0.0, 0.0, t * 0.07));",
    "  float f2 = fbm(sp * 3.8 + vec3(f1 * 1.8, -t * 0.05, t * 0.04));",
    "  float neb = smoothstep(0.42, 0.92, f2);",
    "",
    "  vec3 inner = mix(vec3(0.30, 0.14, 0.78) * 0.40, vec3(0.85, 0.32, 0.80), neb);",
    "  inner = mix(inner, vec3(1.00, 0.76, 0.32), smoothstep(0.62, 1.02, f1 * f2 * 2.1));",
    "",
    "  float core = z / R;",
    "  inner *= 0.16 + 0.80 * pow(core, 1.7);",
    "",
    "  vec3 glass = behind * 0.90 + inner * 0.62;",
    "",
    // A warm glow deep in the middle - it is a fortune-teller's ball, after all.
    "  glass += vec3(1.00, 0.72, 0.38) * pow(max(1.0 - length(q) / R, 0.0), 3.0) * 0.16;",
    "",
    // The connectome in the glass.
    "  vec3 net = vec3(0.0);",
    "  for (int i = 0; i < 8; i++) {",
    "    vec3 p3 = nodePos(i, t);",
    "    vec2 sp2 = c + p3.xy * R;",
    "    float dep = 0.5 + 0.5 * p3.z;",
    "    float dn = length(uv - sp2);",
    "    net += vec3(1.00, 0.88, 0.60) * exp(-dn * 240.0) * (0.30 + 0.70 * dep) * 1.30;",
    "    net += vec3(0.80, 0.70, 1.00) * exp(-dn * 60.0) * 0.10 * dep;",
    "    for (int k = 1; k <= 3; k += 2) {",
    "      vec3 pj = nodePos(int(mod(float(i) + float(k), 8.0)), t);",
    "      vec2 sj = c + pj.xy * R;",
    "      float dl = segDist(uv, sp2, sj);",
    "      float dd = 0.5 + 0.25 * (p3.z + pj.z);",
    "      net += vec3(0.60, 0.66, 1.00) * exp(-dl * 380.0) * 0.26 * (0.25 + 0.75 * dd);",
    "    }",
    "  }",
    "  glass += net * smoothstep(R, R * 0.94, d);",
    "",
    // Fresnel rim: the bright edge that says "this is glass".
    "  float fres = pow(1.0 - core, 2.6);",
    "  glass += vec3(0.78, 0.68, 1.00) * fres * 1.55;",
    "",
    "  vec3 L = normalize(vec3(-0.50, 0.62, 0.75));",
    "  vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));",
    "  float nh = clamp(dot(n, H), 0.0, 1.0);",
    "  glass += vec3(1.0) * pow(nh, 90.0) * 0.70;",
    "  glass += vec3(1.0) * pow(nh, 500.0) * 0.60;",
    "  glass += vec3(0.90, 0.85, 1.00) *",
    "           pow(max(dot(n, normalize(vec3(0.45, -0.50, 0.72))), 0.0), 10.0) * 0.06;",
    "",
    "  col = mix(col, glass, smoothstep(R, R - 0.0035, d));",
    "",
    // Bloom around the sphere, plus a tight ring right on the limb.
    "  float out_ = max(d - R, 0.0);",
    "  col += vec3(0.42, 0.26, 0.95) * exp(-out_ * 8.0) * 0.45;",
    "  col += vec3(0.85, 0.62, 1.00) * exp(-abs(d - R) * 65.0) * 0.32;",
    "",
    "  col *= 1.0 - 0.32 * pow(length(uv * vec2(0.55, 1.0)), 2.0);",
    "  col = 1.0 - exp(-col * 1.35);",
    // Dither: the gradients are large and dark, which is exactly where 8-bit
    // banding shows up.
    "  col += (hash13(vec3(gl_FragCoord.xy, 1.0)) - 0.5) / 255.0;",
    "",
    "  gl_FragColor = vec4(col, 1.0);",
    "}"
  ].join("\n");

  var state = null;

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("[connecto] shader:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function teardown() {
    if (!state) return;
    if (state.raf) cancelAnimationFrame(state.raf);
    window.removeEventListener("resize", state.onResize);
    window.removeEventListener("pointermove", state.onMove);
    window.removeEventListener("scroll", state.onScroll);
    document.body.classList.remove("cn-scrolled");
    state = null;
  }

  function measure() {
    var header = document.querySelector(".md-header");
    var tabs = document.querySelector(".md-tabs");
    var root = document.documentElement;
    root.style.setProperty("--cn-header-h", (header ? header.offsetHeight : 0) + "px");
    root.style.setProperty("--cn-tabs-h", (tabs ? tabs.offsetHeight : 0) + "px");
  }

  function init() {
    var canvas = document.getElementById("cn-orb");
    if (!canvas) {
      document.body.removeAttribute("data-cn-home");
      teardown();
      return;
    }
    if (state && state.canvas === canvas) return;
    teardown();

    document.body.setAttribute("data-cn-home", "");
    measure();

    var hero = canvas.closest(".cn-hero");
    var gl =
      canvas.getContext("webgl", { antialias: false, alpha: false, powerPreference: "high-performance" }) ||
      canvas.getContext("experimental-webgl");

    // No WebGL: the CSS fallback (layered radial gradients) is already in place;
    // just leave it alone rather than shipping a broken black rectangle.
    if (!gl) {
      if (hero) hero.classList.add("cn-hero--fallback");
      return;
    }

    var vs = compile(gl, gl.VERTEX_SHADER, VERT);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) {
      if (hero) hero.classList.add("cn-hero--fallback");
      return;
    }

    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[connecto] link:", gl.getProgramInfoLog(prog));
      if (hero) hero.classList.add("cn-hero--fallback");
      return;
    }
    gl.useProgram(prog);

    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var U = {
      res: gl.getUniformLocation(prog, "uRes"),
      time: gl.getUniformLocation(prog, "uTime"),
      mouse: gl.getUniformLocation(prog, "uMouse"),
      center: gl.getUniformLocation(prog, "uCenter"),
      radius: gl.getUniformLocation(prog, "uRadius")
    };

    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    state = {
      canvas: canvas,
      gl: gl,
      raf: 0,
      visible: true,
      mouse: [0, 0],
      target: [0, 0],
      t0: performance.now()
    };

    function resize() {
      // Cap the device pixel ratio: the shader is fill-rate bound, and a 3x
      // retina panel at 4K buys nothing visible for triple the fragments.
      var dpr = Math.min(window.devicePixelRatio || 1, 1.75);
      var w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      var h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.uniform2f(U.res, w, h);

      // uv is normalised by height, so x spans +/- aspect/2. Sit the ball to the
      // right of the copy on wide screens; centre it above the copy on narrow ones.
      var aspect = w / h;
      var wide = window.innerWidth >= 900;
      gl.uniform2f(U.center, wide ? aspect * 0.24 : 0.0, wide ? 0.02 : 0.17);
      gl.uniform1f(U.radius, wide ? 0.27 : 0.21);
      measure();
    }

    function frame(now) {
      state.raf = requestAnimationFrame(frame);
      if (!state.visible) return;

      state.mouse[0] += (state.target[0] - state.mouse[0]) * 0.05;
      state.mouse[1] += (state.target[1] - state.mouse[1]) * 0.05;

      gl.uniform1f(U.time, reduced ? 12.0 : (now - state.t0) / 1000);
      gl.uniform2f(U.mouse, state.mouse[0], state.mouse[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // Reduced motion: draw one frame, then stop entirely.
      if (reduced) {
        cancelAnimationFrame(state.raf);
        state.raf = 0;
      }
    }

    state.onResize = function () {
      resize();
      if (reduced) requestAnimationFrame(frame);
    };
    state.onMove = function (e) {
      state.target[0] = (e.clientX / window.innerWidth) * 2 - 1;
      state.target[1] = 1 - (e.clientY / window.innerHeight) * 2;
    };
    state.onScroll = function () {
      document.body.classList.toggle("cn-scrolled", window.scrollY > 24);
      // The hero is `position: fixed`, so it always intersects the viewport - an
      // IntersectionObserver on it would report "visible" forever and the shader
      // would keep running under the whole page. Scroll position is the honest
      // test of whether anyone can still see it.
      state.visible = window.scrollY < window.innerHeight;
    };

    window.addEventListener("resize", state.onResize);
    window.addEventListener("pointermove", state.onMove, { passive: true });
    window.addEventListener("scroll", state.onScroll, { passive: true });
    state.onScroll();

    resize();
    state.raf = requestAnimationFrame(frame);
  }

  function boot() {
    init();
    var container = document.querySelector("[data-md-component=container]");
    if (container && "MutationObserver" in window) {
      // Instant navigation swaps the container wholesale, taking the hero with
      // it. Rather than hook into the theme's internals, just watch the DOM.
      new MutationObserver(function () {
        init();
      }).observe(container.parentNode || document.body, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
