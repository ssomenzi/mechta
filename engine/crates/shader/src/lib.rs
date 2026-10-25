//! The ONLY crate of mechta with `unsafe`, by design.
//!
//! Creates the shader modules with `create_shader_module_trusted`, removing ONLY the guard against infinite
//! loops (`force_loop_bounding = false`), which as measured cost from ×1.25 to ×3 on the kernels. The bounds
//! checks on buffers stay ON (`bounds_checks = true`), like all the others.
//!
//! THE PROMISE that whoever uses this crate must keep: **no infinite loop in the shaders**. Every
//! loop must have a constant bound, or a bound read from a finite uniform/push constant, and a
//! step that always advances.
#![deny(unsafe_op_in_unsafe_fn)]

/// Creates a WGSL module with the wgpu checks except the loop guard.
pub fn modulo(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    let mut checks = wgpu::ShaderRuntimeChecks::checked();
    checks.force_loop_bounding = false;
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) };
    // SAFETY: `create_shader_module_trusted` is unsafe because, with the checks removed, a shader
    // could perform operations with undefined behaviour. Here we remove only the guard against
    // infinite loops: buffer bounds stay checked by wgpu. The condition to respect is
    // that the mechta shaders contain no infinite loops (see the promise at the top of the crate).
    unsafe { device.create_shader_module_trusted(desc, checks) }
}

/// The experimental features of wgpu, needed to ask the device for the matrix units
/// (`EXPERIMENTAL_COOPERATIVE_MATRIX`, attn_fa_mx). Asked only when attn_fa_mx is on (crates/gpu,
/// `prefill_mx_requested`: the default with the f16 KV cache).
pub fn experimental_features_on() -> wgpu::ExperimentalFeatures {
    // SAFETY: `ExperimentalFeatures::enabled` is unsafe because experimental features of wgpu may be incomplete or
    // wrong. We use one, cooperative matrices, in one kernel (fa_mx.wgsl), and that kernel is checked against the
    // CPU (error 7.6e-6) and with the KL of the logits against attn_fa before it can become a default.
    unsafe { wgpu::ExperimentalFeatures::enabled() }
}
