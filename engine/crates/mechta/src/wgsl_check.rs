//! Every WGSL file of the engine parsed and validated on the CPU (naga, the same front end wgpu uses), so a
//! broken shader fails `cargo test` instead of a GPU run (twice a kernel was inserted with a duplicate of the
//! one before it: `redefinition of ...`, seen only when the model was already loaded).
#[cfg(test)]
mod tests {
    use crate::engine::{KvFormat, ops_source};

    fn check(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
    }

    #[test]
    fn ops_for_every_kv_format() {
        for f in [KvFormat::F32, KvFormat::F16] { check(&format!("ops.wgsl ({})", f.name()), &ops_source(f)); }
    }

    #[test]
    fn attn_fa_mx() { check("ops.wgsl (f16) + fa_mx.wgsl", &crate::engine::fa_mx_source()); }

    /// Not a check: writes the exact source of the matrix-unit pipelines to MECHTA_DUMP_MX, to compile it alone.
    /// `MECHTA_DUMP_MX=<file> cargo test --release dump_fa_mx_source -- --ignored`
    #[test]
    #[ignore]
    fn dump_fa_mx_source() { std::fs::write(std::env::var("MECHTA_DUMP_MX").expect("MECHTA_DUMP_MX"), crate::engine::fa_mx_source()).unwrap(); }

    #[test]
    fn other_shaders() {
        for (n, s) in [
            ("matvec.wgsl", include_str!("wgsl/matvec.wgsl")), ("routing.wgsl", include_str!("wgsl/routing.wgsl")),
            ("matvec_n4.wgsl", include_str!("wgsl/matvec_n4.wgsl")),
            ("matvec_nh.wgsl", include_str!("wgsl/matvec_nh.wgsl")), ("merge.wgsl", include_str!("wgsl/merge.wgsl")), ("dequant.wgsl", include_str!("wgsl/dequant.wgsl")),
        ] { check(n, s); }
        // the three sources with Q3_K: exactly the text the engine compiles (engine::q3_source)
        for n in ["matvec4", "matvec_uh", "gemm"] { check(&format!("{n} (q3_source)"), &crate::engine::q3_source(n)); }
        // Flint: routing, sum, and the attention source with each K/V format (exactly what Flint compiles)
        check("flint_moe.wgsl", include_str!("wgsl/flint_moe.wgsl"));
        check("flint_sum.wgsl", include_str!("wgsl/flint_sum.wgsl"));
        for f16 in [false, true] { check(&format!("flint_attn (KV f16 = {f16})"), &crate::flint::attn_source_for(f16)); }
    }
}
