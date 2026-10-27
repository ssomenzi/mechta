// M4: union of the experts of the two tokens of the double step. One thread.
// a = ids of token 0 [8], b = ids of token 1 [8] → o = [n, then per expert: id, slot0, slot1] (slot 0xFFFF = unused).
// Order: first the 8 of token 0 (with their slot in token 1, if any), then those only in token 1.
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn merge() {
  var n = 0u;
  for (var k: u32 = 0u; k < 8u; k++) {
    var s1 = 0xFFFFu;
    for (var j: u32 = 0u; j < 8u; j++) { if (b[j] == a[k]) { s1 = j; } }
    o[1u + 3u * n] = a[k]; o[2u + 3u * n] = k; o[3u + 3u * n] = s1; n++;
  }
  for (var j: u32 = 0u; j < 8u; j++) {
    var c = false;
    for (var k: u32 = 0u; k < 8u; k++) { if (a[k] == b[j]) { c = true; } }
    if (!c) { o[1u + 3u * n] = b[j]; o[2u + 3u * n] = 0xFFFFu; o[3u + 3u * n] = j; n++; }
  }
  o[0] = n;
}

// k-token step: union of the experts of 4 rows (ids of row t at a[t*8 ..]). One thread.
// o = [n, then per expert: id, slot0, slot1, slot2, slot3] (slot 0xFFFF = the row does not use it), in order of first use.
@compute @workgroup_size(1)
fn merge4() {
  var n = 0u;
  for (var t: u32 = 0u; t < 4u; t++) {
    for (var k: u32 = 0u; k < 8u; k++) {
      let e = a[t * 8u + k];
      var f = n;
      for (var z: u32 = 0u; z < n; z++) { if (o[1u + 5u * z] == e) { f = z; } }
      if (f == n) {
        o[1u + 5u * n] = e;
        for (var s: u32 = 0u; s < 4u; s++) { o[2u + 5u * n + s] = 0xFFFFu; }
        n++;
      }
      o[2u + 5u * f + t] = k;
    }
  }
  o[0] = n;
}
