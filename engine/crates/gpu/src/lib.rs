//! mechta GPU: device, buffers, kernel execution. All safe.
#![forbid(unsafe_code)]
pub use wgpu;
use wgpu::util::DeviceExt;

/// A dispatch: pipeline, bound buffers (binding, buffer), number of workgroups.
/// `n` and `byte` (weights read) are used only by the measurements.
pub struct Step<'a> { pub p: &'a wgpu::ComputePipeline, pub b: Vec<(u32, &'a wgpu::Buffer)>, pub g: (u32, u32, u32), pub n: String, pub byte: u64 }

/// Times of a measured batch: preparation on the CPU (bind group + commands), wait for the submit,
/// GPU time of each dispatch (timestamp, one compute pass per dispatch).
pub struct Measure { pub prep_ms: f64, pub wait_ms: f64, pub gpu_ms: Vec<f64> }

/// A dispatch already prepared (bind group created only once): it is reused at every token.
pub struct Command { pub p: wgpu::ComputePipeline, pub bg: wgpu::BindGroup, pub g: (u32, u32, u32), pub n: String, pub byte: u64 }

pub struct Gpu { pub device: wgpu::Device, pub queue: wgpu::Queue, pub info: wgpu::AdapterInfo,
    /// After the submit it busy-waits instead of sleeping in the poll: -0.6 ms per token.
    /// It costs a core at 100% while the GPU works; the test machine is dedicated to mechta. MECHTA_POLL=sleeping to turn it off.
    pub poll_active: bool }

/// Prefill attention on the matrix units (attn_fa_mx), MECHTA_PREFILL_ATTN: `mx` or `mx2` yes; `fa` or `fd4` no;
/// unset = yes when the KV cache is f16 (MECHTA_KV unset or f16), the default
/// (measured on the real server: prefill +6% at 12.6K up to +45% at 93K over attn_fa).
pub fn prefill_mx_requested() -> bool {
    match std::env::var("MECHTA_PREFILL_ATTN").unwrap_or_default().as_str() {
        "mx" | "mx2" => true,
        "" => matches!(std::env::var("MECHTA_KV").unwrap_or_default().as_str(), "" | "f16"),
        _ => false,
    }
}

impl Gpu {
    pub fn new() -> Self { Self::with_memory(wgpu::MemoryHints::Performance) }
    /// Like `new`, with the chosen allocation strategy (wgpu documentation: `MemoryHints`).
    pub fn with_memory(memory_hints: wgpu::MemoryHints) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .expect("no GPU adapter");
        let info = adapter.get_info();
        let mut feat = (wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY) & adapter.features();
        // the matrix units for attn_fa_mx, an experimental feature of wgpu (the default)
        let mx = prefill_mx_requested();
        let feat_mx = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
        if mx { assert!(adapter.features().contains(feat_mx), "MECHTA_PREFILL_ATTN=mx: this GPU has no cooperative matrices"); feat |= feat_mx; }
        let experimental_features = if mx { mechta_shader::experimental_features_on() } else { wgpu::ExperimentalFeatures::disabled() };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("mechta"), required_features: feat, required_limits: adapter.limits(), memory_hints, experimental_features, ..Default::default() }))
            .expect("request_device");
        Gpu { device, queue, info, poll_active: std::env::var("MECHTA_POLL").map(|v| v != "sleeping").unwrap_or(true) }
    }
    /// Storage buffer with the given bytes (padded to a multiple of 4 with zeros).
    pub fn buffer_byte(&self, label: &str, byte: &[u8]) -> wgpu::Buffer {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        if byte.len() % 4 == 0 { return self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: byte, usage }); }
        let mut v = byte.to_vec(); v.resize(byte.len().div_ceil(4) * 4, 0);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &v, usage })
    }
    pub fn empty_buffer(&self, label: &str, byte: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: byte, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }
    pub fn uniform(&self, values: &[u32]) -> wgpu::Buffer {
        let b: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("uniform"), contents: &b, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST })
    }
    pub fn pipeline(&self, source: &str, entry: &str) -> wgpu::ComputePipeline {
        let m = mechta_shader::modulo(&self.device, entry, source);
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: None, module: &m, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
    }
    /// Runs a pipeline with the buffers bound in order (binding 0, 1, 2...) and waits for the end.
    pub fn run(&self, p: &wgpu::ComputePipeline, buffer: &[&wgpu::Buffer], groups: (u32, u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = buffer.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries });
        let mut enc = self.device.create_command_encoder(&Default::default());
        { let mut pass = enc.begin_compute_pass(&Default::default()); pass.set_pipeline(p); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(groups.0, groups.1, groups.2); }
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    pub fn buffer_f32(&self, label: &str, v: &[f32]) -> wgpu::Buffer {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.buffer_byte(label, &b)
    }
    pub fn buffer_u32(&self, label: &str, v: &[u32]) -> wgpu::Buffer {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.buffer_byte(label, &b)
    }
    /// Like `run`, but with the explicit binding number for each buffer.
    pub fn run_b(&self, p: &wgpu::ComputePipeline, buffer: &[(u32, &wgpu::Buffer)], groups: (u32, u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = buffer.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries });
        let mut enc = self.device.create_command_encoder(&Default::default());
        { let mut pass = enc.begin_compute_pass(&Default::default()); pass.set_pipeline(p); pass.set_bind_group(0, &bg, &[]); pass.dispatch_workgroups(groups.0, groups.1, groups.2); }
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    pub fn command(&self, p: &wgpu::ComputePipeline, b: &[(u32, &wgpu::Buffer)], g: (u32, u32, u32), n: String, byte: u64) -> Command {
        let entries: Vec<wgpu::BindGroupEntry> = b.iter().map(|(i, x)| wgpu::BindGroupEntry { binding: *i, resource: x.as_entire_binding() }).collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries });
        Command { p: p.clone(), bg, g, n, byte }
    }
    /// All the commands in one compute pass and a single submit; waits for the end. Returns (preparation, wait) in ms.
    /// Waits for the end of the submitted work: busy wait (default), or sleeping poll (MECHTA_POLL=sleeping).
    pub fn wait(&self) {
        if !self.poll_active { self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll"); return; }
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f2 = done.clone();
        self.queue.on_submitted_work_done(move || f2.store(true, std::sync::atomic::Ordering::Release));
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            self.device.poll(wgpu::PollType::Poll).expect("poll");
            std::hint::spin_loop();
        }
    }
    /// A single pass with all the dispatches, two timestamps (start and end of the pass) and, in the same submit, the copy of
    /// `byte` bytes of `src` into a readable buffer. A single submit and a single poll per token (before: two, one for the
    /// logits). Returns (preparation ms, wait ms, GPU from the first to the last dispatch ms, bytes read).
    pub fn run_commands_read(&self, c: &[Command], src: &wgpu::Buffer, byte: u64) -> (f64, f64, f64, Vec<u8>) {
        let t0 = std::time::Instant::now();
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
        let br = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let bl = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 + byte, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let tw = wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(tw) });
            for x in c { pass.set_pipeline(&x.p); pass.set_bind_group(0, &x.bg, &[]); pass.dispatch_workgroups(x.g.0, x.g.1, x.g.2); }
        }
        enc.resolve_query_set(&qs, 0..2, &br, 0);
        enc.copy_buffer_to_buffer(&br, 0, &bl, 0, 16);
        enc.copy_buffer_to_buffer(src, 0, &bl, 16, byte);
        let prep = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.wait();
        let wait = t1.elapsed().as_secs_f64() * 1e3;
        let s = bl.slice(..);
        s.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = s.get_mapped_range().expect("range");
        let t: Vec<u64> = data[..16].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let v = data[16..].to_vec();
        drop(data); bl.unmap();
        let per = self.queue.get_timestamp_period() as f64 * 1e-6;
        (prep, wait, t[1].saturating_sub(t[0]) as f64 * per, v)
    }
    pub fn run_commands(&self, c: &[Command]) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for x in c { pass.set_pipeline(&x.p); pass.set_bind_group(0, &x.bg, &[]); pass.dispatch_workgroups(x.g.0, x.g.1, x.g.2); }
        }
        let prep = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.wait();
        (prep, t1.elapsed().as_secs_f64() * 1e3)
    }
    /// Like `run_commands` (a single compute pass), with two GPU timestamps: start and end of the pass.
    /// Returns (preparation ms, wait ms, GPU time from the first to the last dispatch ms). The third number
    /// also contains the gaps between one dispatch and the next, which the per-dispatch measurement does not see.
    pub fn run_commands_span(&self, c: &[Command]) -> (f64, f64, f64) {
        let t0 = std::time::Instant::now();
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
        let br = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let bl = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let tw = wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(tw) });
            for x in c { pass.set_pipeline(&x.p); pass.set_bind_group(0, &x.bg, &[]); pass.dispatch_workgroups(x.g.0, x.g.1, x.g.2); }
        }
        enc.resolve_query_set(&qs, 0..2, &br, 0);
        enc.copy_buffer_to_buffer(&br, 0, &bl, 0, 16);
        let prep = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.wait();
        let wait = t1.elapsed().as_secs_f64() * 1e3;
        let s = bl.slice(..);
        s.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let t: Vec<u64> = s.get_mapped_range().expect("range").chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        bl.unmap();
        let per = self.queue.get_timestamp_period() as f64 * 1e-6;
        (prep, wait, t[1].saturating_sub(t[0]) as f64 * per)
    }
    /// Like `run_commands`, with the GPU timestamps of each dispatch (one compute pass per dispatch).
    pub fn run_commands_measured(&self, c: &[Command]) -> Measure {
        let t0 = std::time::Instant::now();
        let n = c.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * n });
        let size = 16 * n as u64;
        let br = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let bl = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, x) in c.iter().enumerate() {
            let tw = wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(tw) });
            pass.set_pipeline(&x.p); pass.set_bind_group(0, &x.bg, &[]); pass.dispatch_workgroups(x.g.0, x.g.1, x.g.2);
        }
        enc.resolve_query_set(&qs, 0..2 * n, &br, 0);
        enc.copy_buffer_to_buffer(&br, 0, &bl, 0, size);
        let prep_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let wait_ms = t1.elapsed().as_secs_f64() * 1e3;
        let s = bl.slice(..);
        s.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let t: Vec<u64> = s.get_mapped_range().expect("range").chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        bl.unmap();
        let per = self.queue.get_timestamp_period() as f64 * 1e-6;
        Measure { prep_ms, wait_ms, gpu_ms: (0..n as usize).map(|i| t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * per).collect() }
    }
    /// Copies between buffers on the GPU, in a submit without waiting (the queue is ordered: the next submit sees them done).
    pub fn copies(&self, c: &[(&wgpu::Buffer, &wgpu::Buffer, u64)]) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (s, d, n) in c { enc.copy_buffer_to_buffer(s, 0, d, 0, *n); }
        self.queue.submit([enc.finish()]);
    }
    /// Writes bytes into a buffer (before the next submit).
    pub fn write(&self, b: &wgpu::Buffer, offset: u64, byte: &[u8]) { self.queue.write_buffer(b, offset, byte); }
    /// Several dispatches in a single compute pass and a single submit; wgpu orders the accesses between one
    /// dispatch and the next. Waits for the end.
    pub fn batch(&self, steps: &[Step]) {
        let bgs: Vec<wgpu::BindGroup> = steps.iter().map(|s| {
            let entries: Vec<wgpu::BindGroupEntry> = s.b.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &s.p.get_bind_group_layout(0), entries: &entries })
        }).collect();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (s, bg) in steps.iter().zip(&bgs) { pass.set_pipeline(s.p); pass.set_bind_group(0, bg, &[]); pass.dispatch_workgroups(s.g.0, s.g.1, s.g.2); }
        }
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
    /// Like `batch`, but with one compute pass per dispatch and the GPU timestamps of start and end.
    pub fn measured_batch(&self, steps: &[Step]) -> Measure {
        assert!(self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY), "TIMESTAMP_QUERY missing");
        let t0 = std::time::Instant::now();
        let n = steps.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * n });
        let size = 16 * n as u64;
        let br = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let bl = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bgs: Vec<wgpu::BindGroup> = steps.iter().map(|s| {
            let entries: Vec<wgpu::BindGroupEntry> = s.b.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &s.p.get_bind_group_layout(0), entries: &entries })
        }).collect();
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, (s, bg)) in steps.iter().zip(&bgs).enumerate() {
            let tw = wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(tw) });
            pass.set_pipeline(s.p); pass.set_bind_group(0, bg, &[]); pass.dispatch_workgroups(s.g.0, s.g.1, s.g.2);
        }
        enc.resolve_query_set(&qs, 0..2 * n, &br, 0);
        enc.copy_buffer_to_buffer(&br, 0, &bl, 0, size);
        let prep_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let wait_ms = t1.elapsed().as_secs_f64() * 1e3;
        let s = bl.slice(..);
        s.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let t: Vec<u64> = s.get_mapped_range().expect("range").chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        bl.unmap();
        let per = self.queue.get_timestamp_period() as f64 * 1e-6;
        Measure { prep_ms, wait_ms, gpu_ms: (0..n as usize).map(|i| t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * per).collect() }
    }
    pub fn read_f32(&self, b: &wgpu::Buffer, n: usize) -> Vec<f32> { self.read_f32_from(b, 0, n) }
    /// `n` f32 from word `da` of the buffer (compact KV cache, whose scales are after the data).
    pub fn read_f32_from(&self, b: &wgpu::Buffer, da: usize, n: usize) -> Vec<f32> {
        let size = (n * 4) as u64;
        let st = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, (da * 4) as u64, &st, 0, size);
        self.queue.submit([enc.finish()]);
        let s = st.slice(..);
        s.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let v = s.get_mapped_range().expect("range").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        st.unmap(); v
    }
}

impl Default for Gpu { fn default() -> Self { Self::new() } }
