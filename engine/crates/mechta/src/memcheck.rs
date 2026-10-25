//! Memory check before the start: does the engine fit with the chosen context and KV format?
//! Need = fixed part (weights + buffers that do not depend on the context) + KV cache + flash-decoding partials.
//! Free = free VRAM + the GTT that RAM can still give, keeping OS_RESERVE of RAM for the rest of the desktop.
//! The fixed part comes from a measurement: at context 131072 with KV f16, the server took 24.79 GB;
//! minus KV (2.95 GB) and partials (0.30 GB) = 21.5 GB, rounded up to 22 GB.
//! The server prints the measured use after loading ("GPU memory taken"), to check BASE_BYTES again.
use crate::engine::{KvFormat, CTX_MAX, CTX_MIN, N_LAYER};

const BASE_BYTES: u64 = 22_000_000_000;
const OS_RESERVE: u64 = 8 << 30;
/// Common sizes, offered as presets and used for the "largest that fits" hint
pub const COMMON: [usize; 5] = [32_768, 65_536, 131_072, 196_608, 262_144];

/// Bytes the engine needs with `ctx` tokens of context and the KV format `kv`.
pub fn need_bytes(ctx: usize, kv: KvFormat) -> u64 {
    let kv_layers = (N_LAYER / 4 + 1) as u64;   // 10 attention layers + the MTP layer, K and V each
    let kv_bytes = 2 * kv_layers * kv.buffer_bytes_for(ctx);
    let partials = 4 * 258 * 16 * (2 * (ctx / 128) + (8 * (ctx / crate::engine::PREFILL_BLOCK)).max(64 * (ctx / crate::engine::FA_CH))) as u64;   // att_p of decode and prefill
    BASE_BYTES + kv_bytes + partials
}

/// Flint (F5): weights on the GPU (all tensors but token_embd, which stays on the CPU) + K/V in f32
/// (full layers: ctx rows; sliding layers: a ring of RING_SWA rows) + FLINT_BUFFERS for the buffers of one token, of a
/// prefill block, the ring photo and the attention scratch. FLINT_BUFFERS is an estimate: the server prints the
/// measured use after loading ("GPU memory taken").
const FLINT_BUFFERS: u64 = 800_000_000;
pub fn need_bytes_flint(ctx: usize, path: &str) -> u64 {
    let Ok(g) = mechta_gguf::Gguf::open(path) else { return u64::MAX };
    let weights: u64 = g.tensors.iter().filter(|t| t.name != "token_embd.weight").map(|t| t.nbytes).sum();
    let conf = crate::flint::Conf::da_gguf(&g);
    let row = 2 * 512 * crate::flint::kv_byte() as u64;   // K and V, 4 KV heads x 128, f32 or f16 (MECHTA_FLINT_KV)
    let kv: u64 = conf.swa.iter().map(|&s| row * if s { crate::flint::RING_SWA as u64 } else { ctx as u64 }).sum();
    weights + kv + FLINT_BUFFERS
}
/// As `check`, for Flint.
pub fn check_flint(ctx: usize, path: &str) -> Result<String, String> {
    let free = read_free()?;
    let (need, usable) = (need_bytes_flint(ctx, path), free.usable());
    let detail = format!("VRAM free {} · GTT free {} · RAM available {}", gb(free.vram), gb(free.gtt), gb(free.ram_available));
    let f = if crate::flint::kv_f16() { "f16" } else { "f32" };
    if need <= usable { return Ok(format!("memory OK (Flint): context {ctx} with KV {f} needs ~{} of {} usable ({detail})", gb(need), gb(usable))); }
    let fit = (CTX_MIN / 1024..=CTX_MAX / 1024).rev().map(|k| k * 1024).find(|&c| need_bytes_flint(c, path) <= usable);
    Err(format!("NOT ENOUGH MEMORY (Flint): context {ctx} with KV {f} needs ~{}, usable {}; {} ({detail})", gb(need), gb(usable),
        fit.map(|c| format!("largest context that fits now: {c}")).unwrap_or("not even the smallest context fits".into())))
}

#[derive(Debug, Clone, Copy)]
pub struct Free { pub vram: u64, pub gtt: u64, pub ram_available: u64 }

impl Free {
    /// Bytes the engine can take: free VRAM, plus GTT up to what RAM can give without going under OS_RESERVE.
    pub fn usable(&self) -> u64 { self.vram + self.gtt.min(self.ram_available.saturating_sub(OS_RESERVE)) }
}

/// Reads free VRAM / GTT of the first amdgpu card (sysfs) and MemAvailable (/proc/meminfo).
pub fn read_free() -> Result<Free, String> {
    let dev = (0..8).map(|i| format!("/sys/class/drm/card{i}/device")).find(|d| std::path::Path::new(&format!("{d}/mem_info_vram_total")).exists())
        .ok_or("no GPU with mem_info_vram_total in /sys/class/drm")?;
    let rd = |n: &str| -> Result<u64, String> {
        std::fs::read_to_string(format!("{dev}/{n}")).map_err(|e| format!("{dev}/{n}: {e}"))?.trim().parse().map_err(|e| format!("{dev}/{n}: {e}"))
    };
    let mem = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    let ram_available = mem.lines().find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).ok_or("MemAvailable not found")? * 1024;
    Ok(Free { vram: rd("mem_info_vram_total")?.saturating_sub(rd("mem_info_vram_used")?), gtt: rd("mem_info_gtt_total")?.saturating_sub(rd("mem_info_gtt_used")?), ram_available })
}

/// Bytes of VRAM + GTT in use now (to measure what the engine took after loading).
pub fn used_now() -> Option<u64> {
    let dev = (0..8).map(|i| format!("/sys/class/drm/card{i}/device")).find(|d| std::path::Path::new(&format!("{d}/mem_info_vram_used")).exists())?;
    let rd = |n: &str| std::fs::read_to_string(format!("{dev}/{n}")).ok()?.trim().parse::<u64>().ok();
    Some(rd("mem_info_vram_used")? + rd("mem_info_gtt_used")?)
}

/// The largest context (multiple of 1024, at most CTX_MAX) that fits in `usable` bytes, if any.
pub fn largest_fit(usable: u64, kv: KvFormat) -> Option<usize> {
    (CTX_MIN / 1024..=CTX_MAX / 1024).rev().map(|k| k * 1024).find(|&c| need_bytes(c, kv) <= usable)
}

fn gb(b: u64) -> String { format!("{:.1} GB", b as f64 / 1e9) }

/// One line of explanation, and whether the start can go on.
pub fn check(ctx: usize, kv: KvFormat) -> Result<String, String> {
    let free = read_free()?;
    let (need, usable) = (need_bytes(ctx, kv), free.usable());
    let detail = format!("VRAM free {} · GTT free {} · RAM available {}", gb(free.vram), gb(free.gtt), gb(free.ram_available));
    if need <= usable {
        return Ok(format!("memory OK: context {ctx} with KV {} needs ~{} of {} usable ({detail})", kv.name(), gb(need), gb(usable)));
    }
    let hint = match largest_fit(usable, kv) {
        Some(c) => {
            let common = COMMON.iter().rev().find(|&&x| x <= c).map(|x| format!(" (largest common size: {x})")).unwrap_or_default();
            format!("largest context that fits now: {c}{common}")
        }
        None => "not even the smallest context fits: free GPU memory first".into(),
    };
    Err(format!("NOT ENOUGH MEMORY: context {ctx} with KV {} needs ~{}, usable {}; {hint} ({detail})", kv.name(), gb(need), gb(usable)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn need_grows_with_context_and_f16_is_half_of_f32() {
        let base = need_bytes(CTX_MIN, KvFormat::F16);
        assert!(need_bytes(131_072, KvFormat::F16) > base);
        let kv32 = need_bytes(131_072, KvFormat::F32) - BASE_BYTES;
        let kv16 = need_bytes(131_072, KvFormat::F16) - BASE_BYTES;
        assert!(kv32 > kv16 && kv32 < 2 * kv16 + 1_000_000_000);
        // 65536 tokens of f32 KV: 22 buffers of 65536 * 512 * 4 bytes = 2.95 GB
        assert_eq!(2 * 11 * KvFormat::F32.buffer_bytes_for(65_536), 2_952_790_016);
    }
    #[test]
    fn largest_fit_is_the_biggest_that_fits() {
        let usable = need_bytes(131_072, KvFormat::F16) + 1;
        let c = largest_fit(usable, KvFormat::F16).unwrap();
        assert!(c >= 131_072 && need_bytes(c, KvFormat::F16) <= usable && need_bytes(c + 1024, KvFormat::F16) > usable);
        assert_eq!(largest_fit(BASE_BYTES, KvFormat::F16), None);
    }
    #[test]
    fn usable_keeps_the_os_reserve() {
        let f = Free { vram: 10, gtt: 100 << 30, ram_available: 20 << 30 };
        assert_eq!(f.usable(), 10 + (12 << 30));
    }
}
