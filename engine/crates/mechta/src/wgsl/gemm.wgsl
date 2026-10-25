// M3: Y = W X with X of n columns (tokens), n large (prefill). Tile of 64 rows x 64 tokens, k in pieces of 32.
// One workgroup of 256 threads: each thread computes 4 rows x 4 tokens (16 accumulators). For each piece of k (32 values):
//   - the weights of the tile (64 rows x 32) are decoded into shared memory, k-major (Ws[k][row]): 4 threads per row, 8 values each;
//   - x of the tile (64 tokens x 32) is copied into shared memory, k-major (Xs[k][token]), read as vec4;
//   - then 32 steps: 2 vec4 reads from shared memory, 4 vec4 FMAs.
// Decoding identical to matvec_n4.wgsl (Q8_0 at 34 bytes, Q4_K at 144, Q6_K at 210, F32). Layout of x and y as matvec_n4:
//   column c of x from c * x_tok (+ z * x_step_i for the experts); y[(c * nz + z) * rows + r].
// Grid: x = ceil(rows / 64), y = ceil(n / 64), z = matrices (1, or 8 experts with with_id = 1 → ids[z]).
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32, n: u32, x_tok: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

const BM: u32 = 64u;   // rows per tile
const BN: u32 = 64u;   // tokens per tile
var<workgroup> Ws: array<vec4<f32>, 512>;   // [k 32][group of 4 rows 16]
var<workgroup> Xs: array<vec4<f32>, 512>;   // [k 32][group of 4 tokens 16]

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }
fn bw(w: u32, j: u32) -> u32 { return (w >> (8u * j)) & 0xFFu; }
fn word(w: u32, k: u32, s: u32) -> u32 { return (raw[w + k] >> s) | ((raw[w + k + 1u] << (31u - s)) << 1u); }

fn scale_q4(s0: u32, s1: u32, s2: u32, pz: u32) -> vec2<u32> {
  if (pz < 4u) { return vec2<u32>(bw(s0, pz) & 63u, bw(s1, pz) & 63u); }
  let q = pz - 4u;
  return vec2<u32>((bw(s2, q) & 0xFu) | ((bw(s0, q) >> 6u) << 4u), (bw(s2, q) >> 4u) | ((bw(s1, q) >> 6u) << 4u));
}
// 8 values (words kw and kw+1, kw even) of piece i of row rr, decoded; w0/w1 = 4 values each
fn decod8(kind: u32, rr: u32, i: u32, kw: u32) -> array<vec4<f32>, 2> {
  let u = p.k / 32u;
  var r: array<vec4<f32>, 2>;
  if (kind == 0u) {
    let base = (rr * u + i) * 34u;
    let wd = raw[base >> 2u]; let sd = 8u * (base & 3u);
    let d = unpack2x16float((wd >> sd) & 0xFFFFu).x;
    let q = base + 2u; let w = q >> 2u; let s = 8u * (q & 3u);
    r[0] = d * vec4<f32>(unpack4xI8(word(w, kw, s))); r[1] = d * vec4<f32>(unpack4xI8(word(w, kw + 1u, s)));
  } else if (kind == 1u) {
    let base = (rr * (u / 8u) + i / 8u) * 144u; let pz = i % 8u;
    let w0 = base >> 2u;
    let dd = unpack2x16float(raw[w0]);
    let a = scale_q4(raw[w0 + 1u], raw[w0 + 2u], raw[w0 + 3u], pz);
    let d1 = dd.x * f32(a.x); let m1 = dd.y * f32(a.y);
    let wq = w0 + 4u + 8u * (pz / 2u); let sh = 4u * (pz % 2u);
    r[0] = d1 * vec4<f32>(unpack4xU8((raw[wq + kw] >> sh) & 0x0F0F0F0Fu)) - vec4<f32>(m1);
    r[1] = d1 * vec4<f32>(unpack4xU8((raw[wq + kw + 1u] >> sh) & 0x0F0F0F0Fu)) - vec4<f32>(m1);
  } else if (kind == 2u) {
    let base = (rr * (u / 8u) + i / 8u) * 210u; let pz = i % 8u;
    let n = pz / 4u; let c4 = pz % 4u;
    let d = h16(base + 208u);
    let s0 = s8(by(base + 192u + 8u * n + 2u * c4)); let s1 = s8(by(base + 193u + 8u * n + 2u * c4));
    let ql = base + 64u * n + 32u * (c4 % 2u); let wl = ql >> 2u; let sl = 8u * (ql & 3u); let sh = 4u * (c4 / 2u);
    let qh = base + 128u + 32u * n; let wh = qh >> 2u; let shh = 8u * (qh & 3u); let sc2 = 2u * c4;
    for (var j: u32 = 0u; j < 2u; j++) {
      let kk = kw + j;
      let lo = (word(wl, kk, sl) >> sh) & 0x0F0F0F0Fu;
      let hi = ((word(wh, kk, shh) >> sc2) & 0x03030303u) << 4u;
      r[j] = d * select(s1, s0, kk < 4u) * (vec4<f32>(unpack4xU8(lo | hi)) - vec4<f32>(32.0));
    }
  } else if (kind == 6u) {
    // Q4_1 (Flint INT4): 5 words per block of 32; values 4 kw.. of the piece: low nibbles of word 1 + kw
    // for kw < 4, high nibbles of word 1 + kw - 4 after; value = q d + m
    let w0 = (rr * u + i) * 5u;
    let dm = unpack2x16float(raw[w0]);
    for (var j: u32 = 0u; j < 2u; j++) {
      let kk = kw + j;
      let q = (raw[w0 + 1u + kk % 4u] >> select(4u, 0u, kk < 4u)) & 0x0F0F0F0Fu;
      r[j] = dm.x * vec4<f32>(unpack4xU8(q)) + vec4<f32>(dm.y);
    }
  } else if (kind == 5u) {
    // Q3_K: decoding in q3k.wgsl, appended by engine.rs
    let base = (rr * (u / 8u) + i / 8u) * 110u; let pz = i % 8u;
    let d = q3_d(base); let s = q3_scales(base, pz);
    r[0] = d * select(s.y, s.x, kw < 4u) * q3_vals(base, pz, kw);
    r[1] = d * select(s.y, s.x, kw + 1u < 4u) * q3_vals(base, pz, kw + 1u);
  } else {
    let w0 = rr * p.k + 32u * i + 4u * kw;
    r[0] = vec4<f32>(bitcast<f32>(raw[w0]), bitcast<f32>(raw[w0 + 1u]), bitcast<f32>(raw[w0 + 2u]), bitcast<f32>(raw[w0 + 3u]));
    r[1] = vec4<f32>(bitcast<f32>(raw[w0 + 4u]), bitcast<f32>(raw[w0 + 5u]), bitcast<f32>(raw[w0 + 6u]), bitcast<f32>(raw[w0 + 7u]));
  }
  return r;
}
fn put_w(k: u32, row: u32, v: f32) {
  // Ws[k][row / 4] component row % 4 (scalar write into shared memory)
  let i = k * 16u + row / 4u; let c = row % 4u;
  if (c == 0u) { Ws[i].x = v; } else if (c == 1u) { Ws[i].y = v; } else if (c == 2u) { Ws[i].z = v; } else { Ws[i].w = v; }
}
fn put_x(k: u32, tok: u32, v: f32) {
  let i = k * 16u + tok / 4u; let c = tok % 4u;
  if (c == 0u) { Xs[i].x = v; } else if (c == 1u) { Xs[i].y = v; } else if (c == 2u) { Xs[i].z = v; } else { Xs[i].w = v; }
}

fn gemm(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * BM; let c0 = wid.y * BN; let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let nz = select(1u, 8u, p.with_id == 1u);
  // with_id = 2 (M3, experts in the prefill): z = expert, the columns are the entries of its list (ids = grp of the kernel
  // gather: [start, count] at 2z, list at 512 + start, entry = (tok << 3) | slot); tile beyond the entries: exits.
  var ncol = p.n; var cbase = 0u;
  if (p.with_id == 2u) { cbase = ids[2u * z]; ncol = ids[2u * z + 1u]; if (c0 >= ncol) { return; } }
  let u = p.k / 32u;
  // weight load: thread t → row t / 4 of the tile, words 2 * (t % 4) and +1 of the piece
  let rl = t / 4u; let kw = 2u * (t % 4u);
  let rr = mat * p.rows + min(r0 + rl, p.rows - 1u);
  // x load: thread t → token t / 4 of the tile, vec4 (t % 4) and +4 of the piece (8 vec4 = 32 values per token)
  let tl = t / 4u; let xv = t % 4u;
  let tok = min(c0 + tl, ncol - 1u);
  var xo = (tok * p.x_tok + z * p.x_step_i) >> 2u;
  if (p.with_id == 2u) { let v = ids[512u + cbase + tok]; xo = ((v >> 3u) * p.x_tok + (v & 7u) * p.x_step_i) >> 2u; }
  let rg = t / 16u; let cg = t % 16u;   // group of 4 rows and of 4 tokens computed by the thread
  var acc: array<vec4<f32>, 4>;
  for (var i: u32 = 0u; i < u; i++) {
    let w8 = decod8(kind, rr, i, kw);
    for (var j: u32 = 0u; j < 4u; j++) { put_w(4u * kw + j, rl, w8[0][j]); put_w(4u * kw + 4u + j, rl, w8[1][j]); }
    let xa = x4[xo + 8u * i + xv]; let xb = x4[xo + 8u * i + xv + 4u];
    for (var j: u32 = 0u; j < 4u; j++) { put_x(4u * xv + j, tl, xa[j]); put_x(16u + 4u * xv + j, tl, xb[j]); }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 32u; kk++) {
      let w4 = Ws[kk * 16u + rg]; let x4v = Xs[kk * 16u + cg];
      acc[0] += w4.x * x4v; acc[1] += w4.y * x4v; acc[2] += w4.z * x4v; acc[3] += w4.w * x4v;
    }
    workgroupBarrier();
  }
  for (var j: u32 = 0u; j < 4u; j++) {
    let r = r0 + 4u * rg + j;
    if (r >= p.rows) { continue; }
    for (var cc: u32 = 0u; cc < 4u; cc++) {
      let c = c0 + 4u * cg + cc;
      if (c >= ncol) { continue; }
      var yo = (c * nz + z) * p.rows;
      if (p.with_id == 2u) { let v = ids[512u + cbase + c]; yo = ((v >> 3u) * 8u + (v & 7u)) * p.rows; }
      y[yo + r] = acc[j][cc];
    }
  }
}

// ===================== NARROW variant for the experts =====================
// Block profile: the three expert GEMMs cost 500 ms per block each (80% of the prefill):
// with the 64 x 64 tile each expert, which has 4 tokens on average (128 x 8 / 256), computes 64 columns to use 4 (16x waste).
// Here: tile of 64 rows x 8 tokens, one thread per (row, pair of tokens): 2 accumulators. The weight load is the same
// (4 threads per row, 8 values each); x of the tile: 8 tokens x 32 k = 64 vec4, one thread each. Only with_id = 2.
const BN8: u32 = 8u;
var<workgroup> Ws2: array<f32, 2048>;     // [k 32][row 64]
var<workgroup> Xs2: array<f32, 256>;      // [k 32][token 8]
fn gemm8(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * BM; let c0 = wid.y * BN8; let z = wid.z;
  let cbase = ids[2u * z]; let ncol = ids[2u * z + 1u];
  if (c0 >= ncol) { return; }
  let u = p.k / 32u;
  let rl = t / 4u; let kw = 2u * (t % 4u);
  let rr = z * p.rows + min(r0 + rl, p.rows - 1u);
  // x: thread t < 64: token t / 8 of the tile, vec4 t % 8 of the piece
  let tl = t / 8u; let xv = t % 8u;
  let tok = min(c0 + tl, ncol - 1u);
  let v = ids[512u + cbase + tok];
  let xo = ((v >> 3u) * p.x_tok + (v & 7u) * p.x_step_i) >> 2u;
  let row = t / 4u; let cl = 2u * (t % 4u);   // outputs of the thread: row, tokens cl and cl + 1
  var acc0 = 0.0; var acc1 = 0.0;
  let phase = p.nx;   // measurement (MECHTA_GEMM_PHASE): 1 = only weight load and decoding, 2 = only the compute loop (weights at zero)
  for (var i: u32 = 0u; i < u; i++) {
    if (phase != 2u) {
      let w8 = decod8(kind, rr, i, kw);
      for (var j: u32 = 0u; j < 4u; j++) { Ws2[(4u * kw + j) * 64u + rl] = w8[0][j]; Ws2[(4u * kw + 4u + j) * 64u + rl] = w8[1][j]; }
    }
    if (t < 64u) {
      let xa = x4[xo + 8u * i + xv];
      for (var j: u32 = 0u; j < 4u; j++) { Xs2[(4u * xv + j) * 8u + tl] = xa[j]; }
    }
    workgroupBarrier();
    if (phase != 1u) {
      for (var kk: u32 = 0u; kk < 32u; kk++) {
        let w = Ws2[kk * 64u + row];
        acc0 += w * Xs2[kk * 8u + cl]; acc1 += w * Xs2[kk * 8u + cl + 1u];
      }
    }
    workgroupBarrier();
  }
  let r = r0 + row;
  if (r >= p.rows) { return; }
  let c = c0 + cl;
  if (c < ncol) { let e = ids[512u + cbase + c]; y[((e >> 3u) * 8u + (e & 7u)) * p.rows + r] = acc0; }
  if (c + 1u < ncol) { let e = ids[512u + cbase + c + 1u]; y[((e >> 3u) * 8u + (e & 7u)) * p.rows + r] = acc1; }
}

// ===================== 128 x 16 variant for the experts =====================
// Measurement of the phases of gemm8 at 512 tokens: weights 645 ms (28 GB/s), compute 1,400 ms (0.74 TFLOP/s). Here:
// tile of 128 rows x 16 tokens (at 512 tokens an expert has ~16 tokens), thread = 4 rows x 2 tokens: 8 FMAs every 2 reads
// (vec4 of weights + vec2 of x); weight load 16 values per thread per piece (2 rows); x 128 vec4, one thread each.
const BM16: u32 = 128u; const BN16: u32 = 16u;
var<workgroup> Ws16: array<vec4<f32>, 1024>;   // [k 32][group of 4 rows 32]
var<workgroup> Xs16: array<vec2<f32>, 256>;    // [k 32][pair of tokens 8]
fn put_w16(k: u32, row: u32, v: f32) {
  let i = k * 32u + row / 4u; let c = row % 4u;
  if (c == 0u) { Ws16[i].x = v; } else if (c == 1u) { Ws16[i].y = v; } else if (c == 2u) { Ws16[i].z = v; } else { Ws16[i].w = v; }
}
fn gemm16(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * BM16; let c0 = wid.y * BN16; let z = wid.z;
  let cbase = ids[2u * z]; let ncol = ids[2u * z + 1u];
  if (c0 >= ncol) { return; }
  let u = p.k / 32u;
  let rl = t / 4u; let kw = 2u * (t % 4u);                 // rows rl and rl + 64 of the tile, words kw, kw + 1
  let rra = z * p.rows + min(r0 + rl, p.rows - 1u);
  let rrb = z * p.rows + min(r0 + rl + 64u, p.rows - 1u);
  let tl = t / 8u; let xv = t % 8u;                        // thread < 128: token tl, vec4 xv
  let tok = min(c0 + tl, ncol - 1u);
  let v = ids[512u + cbase + tok];
  let xo = ((v >> 3u) * p.x_tok + (v & 7u) * p.x_step_i) >> 2u;
  let rg = t / 8u; let cg = t % 8u;                        // outputs: rows 4 rg .. + 4, tokens 2 cg, 2 cg + 1
  var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
  let phase = p.nx;
  for (var i: u32 = 0u; i < u; i++) {
    if (phase != 2u) {
      let wa = decod8(kind, rra, i, kw); let wb = decod8(kind, rrb, i, kw);
      for (var j: u32 = 0u; j < 4u; j++) {
        put_w16(4u * kw + j, rl, wa[0][j]); put_w16(4u * kw + 4u + j, rl, wa[1][j]);
        put_w16(4u * kw + j, rl + 64u, wb[0][j]); put_w16(4u * kw + 4u + j, rl + 64u, wb[1][j]);
      }
    }
    if (t < 128u) {
      let xa = x4[xo + 8u * i + xv];
      for (var j: u32 = 0u; j < 4u; j++) {
        let k = 4u * xv + j; let ii = k * 8u + tl / 2u;
        if ((tl & 1u) == 0u) { Xs16[ii].x = xa[j]; } else { Xs16[ii].y = xa[j]; }
      }
    }
    workgroupBarrier();
    if (phase != 1u) {
      for (var kk: u32 = 0u; kk < 32u; kk++) {
        let w4 = Ws16[kk * 32u + rg]; let x2 = Xs16[kk * 8u + cg];
        acc0 += w4 * x2.x; acc1 += w4 * x2.y;
      }
    }
    workgroupBarrier();
  }
  let ca = c0 + 2u * cg; let cb = ca + 1u;
  var ya = 0u; var yb = 0u;
  if (ca < ncol) { let e = ids[512u + cbase + ca]; ya = ((e >> 3u) * 8u + (e & 7u)) * p.rows; }
  if (cb < ncol) { let e = ids[512u + cbase + cb]; yb = ((e >> 3u) * 8u + (e & 7u)) * p.rows; }
  for (var j: u32 = 0u; j < 4u; j++) {
    let r = r0 + 4u * rg + j;
    if (r >= p.rows) { continue; }
    if (ca < ncol) { y[ya + r] = acc0[j]; }
    if (cb < ncol) { y[yb + r] = acc1[j]; }
  }
}

// ===================== R variant for the experts: weights in registers =====================
// Phases of gemm16: writing the weights into shared memory costs as much as the compute. Here the weights do NOT go
// through shared memory: thread = (row of the tile, k lane), 64 rows x 4 lanes; at each step of 128 k the lane
// decodes its piece (32 values) into registers and multiplies it by the 16 tokens of x (in shared memory, broadcast
// reads, vec4 with padding for the banks). 16 accumulators per thread; at the end the 4 lanes are summed in `red`.
// One barrier every 128 k (two: before and after the x load). Only with_id = 2 (list per expert).
const BNR: u32 = 16u;
var<workgroup> Xr: array<vec4<f32>, 576>;    // [token 16][lane 4][vec4 8] with stride 9 per lane (36 per token): different banks
var<workgroup> red: array<f32, 3072>;        // [lane - 1][row 64][token 16]
fn gemm_r(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * BM; let c0 = wid.y * BNR; let z = wid.z;
  let cbase = ids[2u * z]; let ncol = ids[2u * z + 1u];
  if (c0 >= ncol) { return; }
  let u = p.k / 32u; let steps = u / 4u;          // pieces per row; steps of 4 pieces (128 k)
  let row = t / 4u; let lane = t % 4u;
  let rr = z * p.rows + min(r0 + row, p.rows - 1u);
  // x load: 2 vec4 per thread per step: v = t, t + 256 → token v / 32, vec4 v % 32 (lane = (v % 32) / 8, j = v % 8)
  let va = t; let vb = t + 256u;
  let ea = ids[512u + cbase + min(c0 + va / 32u, ncol - 1u)]; let eb = ids[512u + cbase + min(c0 + vb / 32u, ncol - 1u)];
  let xoa = (((ea >> 3u) * p.x_tok + (ea & 7u) * p.x_step_i) >> 2u) + (va % 32u);
  let xob = (((eb >> 3u) * p.x_tok + (eb & 7u) * p.x_step_i) >> 2u) + (vb % 32u);
  let sa = (va / 32u) * 36u + ((va % 32u) / 8u) * 9u + (va % 8u);
  let sb = (vb / 32u) * 36u + ((vb % 32u) / 8u) * 9u + (vb % 8u);
  var acc: array<f32, 16>;
  let phase = p.nx;
  for (var s: u32 = 0u; s < steps; s++) {
    Xr[sa] = x4[xoa + 32u * s]; Xr[sb] = x4[xob + 32u * s];
    workgroupBarrier();
    if (phase != 2u) {
      let i = 4u * s + lane;                      // the piece of this lane
      let w0 = decod8(kind, rr, i, 0u); let w1 = decod8(kind, rr, i, 2u); let w2 = decod8(kind, rr, i, 4u); let w3 = decod8(kind, rr, i, 6u);
      if (phase != 1u) {
        let xb = lane * 9u;
        for (var c: u32 = 0u; c < 16u; c++) {
          let x0 = c * 36u + xb;
          acc[c] += dot(w0[0], Xr[x0]) + dot(w0[1], Xr[x0 + 1u]) + dot(w1[0], Xr[x0 + 2u]) + dot(w1[1], Xr[x0 + 3u])
                  + dot(w2[0], Xr[x0 + 4u]) + dot(w2[1], Xr[x0 + 5u]) + dot(w3[0], Xr[x0 + 6u]) + dot(w3[1], Xr[x0 + 7u]);
        }
      }
    }
    workgroupBarrier();
  }
  if (lane > 0u) { for (var c: u32 = 0u; c < 16u; c++) { red[((lane - 1u) * 64u + row) * 16u + c] = acc[c]; } }
  workgroupBarrier();
  if (lane != 0u) { return; }
  let r = r0 + row;
  if (r >= p.rows) { return; }
  for (var c: u32 = 0u; c < 16u; c++) {
    let col = c0 + c;
    if (col >= ncol) { break; }
    let e = ids[512u + cbase + col];
    let v = acc[c] + red[(row) * 16u + c] + red[(64u + row) * 16u + c] + red[(128u + row) * 16u + c];
    y[((e >> 3u) * 8u + (e & 7u)) * p.rows + r] = v;
  }
}

// ===================== R2 variant: two pieces per lane at each step =====================
// Bench: all variants sit at ~10 GB/s on the touched weights and ~600 GFLOP/s, neither bandwidth- nor compute-bound.
// Hypothesis to check with the bench: exposed latency, each thread has only 8 words of weights (32 B) in flight per step, with
// 2 barriers every 128 k. Here each lane does 2 pieces per step (256 k): 16 words in flight, half the barriers.

// ===================== R3/R4 variants: 2 or 4 rows per thread, weights in registers =====================
// Bench r versus r2: r2 (twice the bytes in flight, half the barriers) is NOT faster: the latency
// hypothesis is false. Count of the reads from shared memory: in `gemm_r` each dot(vec4) reads one vec4 of x
// (4 FMAs per read), and every row rereads the same x: 512 KB of LDS reads per step per workgroup, ~4 ms of floor on the
// gate at 512 tokens. Here each thread holds RH rows (2 or 4): one read of x serves RH dots → 8 or 16 FMAs per read.
// Tile of 64 RH rows x 16 tokens, 256 threads = 64 row groups x 4 lanes; x as in gemm_r (stride 9 per lane).
// measurement (p.nx): 3 = Q4_K weights without the scale (unpack only), 4 = raw words bitcast (no decoding), same bytes read
fn fake8(rr: u32, i: u32, kw: u32, mode: u32) -> array<vec4<f32>, 2> {
  let u = p.k / 32u;
  let base = (rr * (u / 8u) + i / 8u) * 144u; let pz = i % 8u;
  let w0 = base >> 2u; let wq = w0 + 4u + 8u * (pz / 2u);
  var r: array<vec4<f32>, 2>;
  if (mode == 3u) { r[0] = vec4<f32>(unpack4xU8(raw[wq + kw])); r[1] = vec4<f32>(unpack4xU8(raw[wq + kw + 1u])); }
  else { r[0] = vec4<f32>(bitcast<f32>(raw[wq + kw])); r[1] = vec4<f32>(bitcast<f32>(raw[wq + kw + 1u])); }
  return r;
}
fn gemm_rh(kind: u32, wid: vec3<u32>, t: u32, rh: u32) {
  let mode = p.nx;
  let r0 = wid.x * BM * rh; let c0 = wid.y * BNR; let z = wid.z;
  let cbase = ids[2u * z]; let ncol = ids[2u * z + 1u];
  if (c0 >= ncol) { return; }
  let u = p.k / 32u; let steps = u / 4u;
  let rg = t / 4u; let lane = t % 4u;
  let va = t; let vb = t + 256u;
  let ea = ids[512u + cbase + min(c0 + va / 32u, ncol - 1u)]; let eb = ids[512u + cbase + min(c0 + vb / 32u, ncol - 1u)];
  let xoa = (((ea >> 3u) * p.x_tok + (ea & 7u) * p.x_step_i) >> 2u) + (va % 32u);
  let xob = (((eb >> 3u) * p.x_tok + (eb & 7u) * p.x_step_i) >> 2u) + (vb % 32u);
  let sa = (va / 32u) * 36u + ((va % 32u) / 8u) * 9u + (va % 8u);
  let sb = (vb / 32u) * 36u + ((vb % 32u) / 8u) * 9u + (vb % 8u);
  var acc: array<array<f32, 16>, 4>;
  var rr: array<u32, 4>;
  for (var h: u32 = 0u; h < 4u; h++) { rr[h] = z * p.rows + min(r0 + rg * rh + min(h, rh - 1u), p.rows - 1u); }
  for (var s: u32 = 0u; s < steps; s++) {
    if (mode == 5u) { Xr[sa] = vec4<f32>(f32(s)); Xr[sb] = vec4<f32>(0.5); }   // measurement: no reads of x from memory
    else { Xr[sa] = x4[xoa + 32u * s]; Xr[sb] = x4[xob + 32u * s]; }
    workgroupBarrier();
    let i = 4u * s + lane;
    let xb = lane * 9u;
    for (var h: u32 = 0u; h < rh; h++) {
      var w0: array<vec4<f32>, 2>; var w1: array<vec4<f32>, 2>; var w2: array<vec4<f32>, 2>; var w3: array<vec4<f32>, 2>;
      if (mode == 6u) {   // measurement: no reads of weights from memory
        let c = vec4<f32>(f32(i) * 0.001); w0[0] = c; w0[1] = c; w1[0] = c; w1[1] = c; w2[0] = c; w2[1] = c; w3[0] = c; w3[1] = c;
      } else if (mode >= 3u) { w0 = fake8(rr[h], i, 0u, mode); w1 = fake8(rr[h], i, 2u, mode); w2 = fake8(rr[h], i, 4u, mode); w3 = fake8(rr[h], i, 6u, mode); }
      else { w0 = decod8(kind, rr[h], i, 0u); w1 = decod8(kind, rr[h], i, 2u); w2 = decod8(kind, rr[h], i, 4u); w3 = decod8(kind, rr[h], i, 6u); }
      for (var c: u32 = 0u; c < 16u; c++) {
        let x0 = c * 36u + xb;
        acc[h][c] += dot(w0[0], Xr[x0]) + dot(w0[1], Xr[x0 + 1u]) + dot(w1[0], Xr[x0 + 2u]) + dot(w1[1], Xr[x0 + 3u])
                   + dot(w2[0], Xr[x0 + 4u]) + dot(w2[1], Xr[x0 + 5u]) + dot(w3[0], Xr[x0 + 6u]) + dot(w3[1], Xr[x0 + 7u]);
      }
    }
    workgroupBarrier();
  }
  // reduction of the 4 lanes, one row of the group at a time (red: [lane - 1][group 64][token 16])
  for (var h: u32 = 0u; h < rh; h++) {
    if (lane > 0u) { for (var c: u32 = 0u; c < 16u; c++) { red[((lane - 1u) * 64u + rg) * 16u + c] = acc[h][c]; } }
    workgroupBarrier();
    let r = r0 + rg * rh + h;
    if (lane == 0u && r < p.rows) {
      for (var c: u32 = 0u; c < 16u; c++) {
        let col = c0 + c;
        if (col >= ncol) { break; }
        let e = ids[512u + cbase + col];
        let v = acc[h][c] + red[(rg) * 16u + c] + red[(64u + rg) * 16u + c] + red[(128u + rg) * 16u + c];
        y[((e >> 3u) * 8u + (e & 7u)) * p.rows + r] = v;
      }
    }
    workgroupBarrier();
  }
}

// ===================== R5 variant: 2 rows per thread, x read ONCE for both rows =====================
// Bench r4: removing the decoding, the reads of x or those of the weights changes little (10 → 9 ms): the cost
// is in the loop. In gemm_rh the reads of x from shared memory are INSIDE the loop over the rows (reread per row)
// and the accumulators are indexed at runtime (acc[h][c], with rh a parameter): possibly scratch memory. Here everything is
// explicit: two rows, weights of both decoded first, for each token 8 vec4 of x read once and used for 2 dots.
fn gemm_r5(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * 128u; let c0 = wid.y * BNR; let z = wid.z;
  let cbase = ids[2u * z]; let ncol = ids[2u * z + 1u];
  if (c0 >= ncol) { return; }
  let u = p.k / 32u; let steps = u / 4u;
  let rg = t / 4u; let lane = t % 4u;
  let ra = z * p.rows + min(r0 + 2u * rg, p.rows - 1u); let rb = z * p.rows + min(r0 + 2u * rg + 1u, p.rows - 1u);
  let va = t; let vb = t + 256u;
  let ea = ids[512u + cbase + min(c0 + va / 32u, ncol - 1u)]; let eb = ids[512u + cbase + min(c0 + vb / 32u, ncol - 1u)];
  let xoa = (((ea >> 3u) * p.x_tok + (ea & 7u) * p.x_step_i) >> 2u) + (va % 32u);
  let xob = (((eb >> 3u) * p.x_tok + (eb & 7u) * p.x_step_i) >> 2u) + (vb % 32u);
  let sa = (va / 32u) * 36u + ((va % 32u) / 8u) * 9u + (va % 8u);
  let sb = (vb / 32u) * 36u + ((vb % 32u) / 8u) * 9u + (vb % 8u);
  var acc0: array<f32, 16>; var acc1: array<f32, 16>;
  for (var s: u32 = 0u; s < steps; s++) {
    Xr[sa] = x4[xoa + 32u * s]; Xr[sb] = x4[xob + 32u * s];
    workgroupBarrier();
    let i = 4u * s + lane;
    let a0 = decod8(kind, ra, i, 0u); let a1 = decod8(kind, ra, i, 2u); let a2 = decod8(kind, ra, i, 4u); let a3 = decod8(kind, ra, i, 6u);
    let b0 = decod8(kind, rb, i, 0u); let b1 = decod8(kind, rb, i, 2u); let b2 = decod8(kind, rb, i, 4u); let b3 = decod8(kind, rb, i, 6u);
    let xb = lane * 9u;
    for (var c: u32 = 0u; c < 16u; c++) {
      let x0 = c * 36u + xb;
      let v0 = Xr[x0]; let v1 = Xr[x0 + 1u]; let v2 = Xr[x0 + 2u]; let v3 = Xr[x0 + 3u];
      let v4 = Xr[x0 + 4u]; let v5 = Xr[x0 + 5u]; let v6 = Xr[x0 + 6u]; let v7 = Xr[x0 + 7u];
      acc0[c] += dot(a0[0], v0) + dot(a0[1], v1) + dot(a1[0], v2) + dot(a1[1], v3) + dot(a2[0], v4) + dot(a2[1], v5) + dot(a3[0], v6) + dot(a3[1], v7);
      acc1[c] += dot(b0[0], v0) + dot(b0[1], v1) + dot(b1[0], v2) + dot(b1[1], v3) + dot(b2[0], v4) + dot(b2[1], v5) + dot(b3[0], v6) + dot(b3[1], v7);
    }
    workgroupBarrier();
  }
  for (var h: u32 = 0u; h < 2u; h++) {
    if (lane > 0u) { for (var c: u32 = 0u; c < 16u; c++) { red[((lane - 1u) * 64u + rg) * 16u + c] = select(acc0[c], acc1[c], h == 1u); } }
    workgroupBarrier();
    let r = r0 + 2u * rg + h;
    if (lane == 0u && r < p.rows) {
      for (var c: u32 = 0u; c < 16u; c++) {
        let col = c0 + c;
        if (col >= ncol) { break; }
        let e = ids[512u + cbase + col];
        let v = select(acc0[c], acc1[c], h == 1u) + red[(rg) * 16u + c] + red[(64u + rg) * 16u + c] + red[(128u + rg) * 16u + c];
        y[((e >> 3u) * 8u + (e & 7u)) * p.rows + r] = v;
      }
    }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn gemmr5_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm_r5(1u, wid, t); }
@compute @workgroup_size(256)
fn gemmr5_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm_r5(2u, wid, t); }
@compute @workgroup_size(256)
fn gemmr5_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm_r5(0u, wid, t); }

// ===================== Dense GEMM 2: tile 128 x 64, thread 8 rows x 4 tokens =====================
// The dense GEMM (gemm) writes the weights into shared memory transposed, one scalar at a time with 4 branches (put_w), and does
// 2 FMAs per read. Lesson from the experts (gemm16/r5): scalar writes with branches cost as much as the compute.
// Here: weights in shared memory by ROW (Wd[row][kq], vec4 along k, stride 9 against bank conflicts): the thread that
// decodes 8 values of a row writes 2 whole vec4. Same for x by token (Xd[token][kq]). Compute: for each kq (4 k)
// 8 weight reads + 4 of x → 32 dots = 128 FMAs (10.7 FMAs per read). Thread (rg, cg) = rows rg + 16 r, tokens cg + 16 c.
var<workgroup> Wd: array<vec4<f32>, 1152>;   // [row 128][kq 8] stride 9
var<workgroup> Xd: array<vec4<f32>, 576>;    // [token 64][kq 8] stride 9
fn gemm2(kind: u32, wid: vec3<u32>, t: u32) {
  let r0 = wid.x * 128u; let c0 = wid.y * 64u;
  let u = p.k / 32u;
  // weight load: thread t → row t / 2, words 4 (t % 2) .. + 4 of the piece (16 values = 4 vec4)
  let rl = t / 2u; let half = t % 2u;
  // `ids` is not needed here, but the bind group layout is the same as the other GEMMs (with_id is 0: the branch never fires)
  let rr = min(r0 + rl, p.rows - 1u) + select(0u, ids[0], p.with_id == 7u);
  // x load: thread t → token t / 4, vec4 2 (t % 4) and + 1
  let tl = t / 4u; let xv = 2u * (t % 4u);
  let xo = (min(c0 + tl, p.n - 1u) * p.x_tok) >> 2u;
  let rg = t / 16u; let cg = t % 16u;
  var acc: array<vec4<f32>, 8>;   // acc[r][c]: row rg + 16 r, token cg + 16 c
  for (var i: u32 = 0u; i < u; i++) {
    let wa = decod8(kind, rr, i, 4u * half); let wb = decod8(kind, rr, i, 4u * half + 2u);
    let wbase = rl * 9u + 4u * half;
    Wd[wbase] = wa[0]; Wd[wbase + 1u] = wa[1]; Wd[wbase + 2u] = wb[0]; Wd[wbase + 3u] = wb[1];
    Xd[tl * 9u + xv] = x4[xo + 8u * i + xv]; Xd[tl * 9u + xv + 1u] = x4[xo + 8u * i + xv + 1u];
    workgroupBarrier();
    for (var kq: u32 = 0u; kq < 8u; kq++) {
      let x0 = Xd[cg * 9u + kq]; let x1 = Xd[(cg + 16u) * 9u + kq]; let x2 = Xd[(cg + 32u) * 9u + kq]; let x3 = Xd[(cg + 48u) * 9u + kq];
      for (var r: u32 = 0u; r < 8u; r++) {
        let w = Wd[(rg + 16u * r) * 9u + kq];
        acc[r] += vec4<f32>(dot(w, x0), dot(w, x1), dot(w, x2), dot(w, x3));
      }
    }
    workgroupBarrier();
  }
  for (var r: u32 = 0u; r < 8u; r++) {
    let row = r0 + rg + 16u * r;
    if (row >= p.rows) { continue; }
    for (var c: u32 = 0u; c < 4u; c++) {
      let col = c0 + cg + 16u * c;
      if (col < p.n) { y[col * p.rows + row] = acc[r][c]; }
    }
  }
}
@compute @workgroup_size(256)
fn gemm2_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm2(0u, wid, t); }
@compute @workgroup_size(256)
fn gemm2_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm2(1u, wid, t); }
@compute @workgroup_size(256)
fn gemm2_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm2(2u, wid, t); }
@compute @workgroup_size(256)
fn gemm2_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm2(3u, wid, t); }
@compute @workgroup_size(256)
fn gemmr5_q3_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm_r5(5u, wid, t); }
@compute @workgroup_size(256)
fn gemmr5_q4_1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm_r5(6u, wid, t); }
@compute @workgroup_size(256)
fn gemm2_q4_1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { gemm2(6u, wid, t); }
