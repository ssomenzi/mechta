//! Numeric check of an engine variant against a reference.
//!
//! `logit-dump` (profile.rs) writes, for every step, the true next token (u32 LE) and the VOCAB logits (f32 LE): the
//! same format as `quality-kv`. `mechta kl <ref.bin> <var.bin> [max_mean_kl min_top1_%]` compares two
//! such files: mean / p99 / max KL(ref || var) in nats, top-1 agreement, NLL of the true token for both. With the two
//! thresholds it also prints PASS or FAIL (an optimization stays optional until its KL against the
//! default path is under a threshold). CPU only, one record at a time (a file is ~250 MB for 256 steps).
use crate::engine::VOCAB;
use std::io::Read;

/// Bytes of one record: the true token + the logits. MECHTA_KL_VOCAB=<n> for another vocabulary (Flint: 150144).
fn rec() -> usize { 4 + 4 * std::env::var("MECHTA_KL_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(VOCAB) }

/// Log-softmax in f64.
fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let s: f64 = x.iter().map(|&v| (v as f64 - m).exp()).sum();
    let ls = s.ln();
    x.iter().map(|&v| v as f64 - m - ls).collect()
}

fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for (i, &v) in x.iter().enumerate() { if v > x[b] { b = i; } }
    b
}

/// One step of the comparison: KL(ref || var), same top-1, NLL of the true token in ref and var.
pub struct Step { pub kl: f64, pub same_top1: bool, pub nll_ref: f64, pub nll_var: f64 }

pub fn compare_step(target: u32, a: &[f32], b: &[f32]) -> Step {
    let (pa, pb) = (log_softmax(a), log_softmax(b));
    let kl = pa.iter().zip(&pb).map(|(&x, &y)| x.exp() * (x - y)).sum::<f64>().max(0.0);
    Step { kl, same_top1: argmax(a) == argmax(b), nll_ref: -pa[target as usize], nll_var: -pb[target as usize] }
}

pub struct Summary { pub steps: usize, pub kl_mean: f64, pub kl_median: f64, pub kl_p99: f64, pub kl_max: f64, pub kl_max_at: usize,
                     pub top1: f64, pub nll_ref: f64, pub nll_var: f64 }

pub fn summarize(v: &[Step]) -> Summary {
    let n = v.len().max(1) as f64;
    let mut kls: Vec<f64> = v.iter().map(|s| s.kl).collect();
    let (kl_max_at, kl_max) = kls.iter().cloned().enumerate().fold((0, 0.0), |a, (i, k)| if k > a.1 { (i, k) } else { a });
    kls.sort_by(|a, b| a.total_cmp(b));
    let p99 = kls.get(((kls.len() as f64 * 0.99).ceil() as usize).saturating_sub(1)).cloned().unwrap_or(0.0);
    let median = if kls.is_empty() { 0.0 } else { kls[kls.len() / 2] };
    Summary { steps: v.len(), kl_mean: v.iter().map(|s| s.kl).sum::<f64>() / n, kl_median: median, kl_p99: p99, kl_max, kl_max_at,
              top1: 100.0 * v.iter().filter(|s| s.same_top1).count() as f64 / n,
              nll_ref: v.iter().map(|s| s.nll_ref).sum::<f64>() / n, nll_var: v.iter().map(|s| s.nll_var).sum::<f64>() / n }
}

fn read_rec(f: &mut impl Read, buf: &mut [u8]) -> Option<(u32, Vec<f32>)> {
    f.read_exact(buf).ok()?;
    let t = u32::from_le_bytes(buf[..4].try_into().unwrap());
    Some((t, buf[4..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()))
}

/// A limit of the check (chosen form: on code median KL + top-1, on prose mean KL):
/// `mean<=X`, `median<=X`, `top1>=X` (percent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Limit { Mean(f64), Median(f64), Top1(f64) }

pub fn parse_limit(s: &str) -> Result<Limit, String> {
    let num = |v: &str| v.trim().parse::<f64>().map_err(|_| format!("{s}: not a number"));
    if let Some(v) = s.strip_prefix("mean<=") { return Ok(Limit::Mean(num(v)?)); }
    if let Some(v) = s.strip_prefix("median<=") { return Ok(Limit::Median(num(v)?)); }
    if let Some(v) = s.strip_prefix("top1>=") { return Ok(Limit::Top1(num(v)?)); }
    Err(format!("{s}: use mean<=X, median<=X or top1>=X"))
}

/// Which limits fail (empty = PASS), with the measured value.
pub fn failed(s: &Summary, limits: &[Limit]) -> Vec<String> {
    limits.iter().filter_map(|l| match *l {
        Limit::Mean(x) if s.kl_mean > x => Some(format!("KL mean {:.5} > {x}", s.kl_mean)),
        Limit::Median(x) if s.kl_median > x => Some(format!("KL median {:.5} > {x}", s.kl_median)),
        Limit::Top1(x) if s.top1 < x => Some(format!("top-1 {:.1}% < {x}%", s.top1)),
        _ => None,
    }).collect()
}

/// `mechta kl <ref.bin> <var.bin> [limit ...]` (limits: `mean<=X` `median<=X` `top1>=X`)
pub fn kl(reference: &str, variant: &str, limits: &[Limit]) {
    let open = |p: &str| std::io::BufReader::with_capacity(rec(), std::fs::File::open(p).unwrap_or_else(|e| panic!("{p}: {e}")));
    let (mut fa, mut fb) = (open(reference), open(variant));
    let (mut ba, mut bb) = (vec![0u8; rec()], vec![0u8; rec()]);
    let mut steps = Vec::new();
    loop {
        let (Some((ta, a)), Some((tb, b))) = (read_rec(&mut fa, &mut ba), read_rec(&mut fb, &mut bb)) else { break };
        assert_eq!(ta, tb, "step {}: different texts (true token {ta} vs {tb})", steps.len());
        steps.push(compare_step(ta, &a, &b));
    }
    // MECHTA_KL_TSV=<file>: one line per step (step, KL, same top-1, NLL ref, NLL var), to see where the KL is
    if let Ok(p) = std::env::var("MECHTA_KL_TSV") {
        let body: String = steps.iter().enumerate().map(|(i, s)| format!("{i}\t{:.6}\t{}\t{:.4}\t{:.4}\n", s.kl, s.same_top1 as u8, s.nll_ref, s.nll_var)).collect();
        std::fs::write(&p, format!("step\tkl\tsame_top1\tnll_ref\tnll_var\n{body}")).unwrap_or_else(|e| panic!("{p}: {e}"));
    }
    let s = summarize(&steps);
    println!("{} steps · KL mean {:.5} · median {:.5} · p99 {:.4} · max {:.4} (step {}) · top-1 same {:.1}% · NLL ref {:.4} var {:.4} ({:+.2}%)",
        s.steps, s.kl_mean, s.kl_median, s.kl_p99, s.kl_max, s.kl_max_at, s.top1, s.nll_ref, s.nll_var, 100.0 * (s.nll_var / s.nll_ref - 1.0));
    if !limits.is_empty() {
        let f = failed(&s, limits);
        if f.is_empty() { println!("PASS ({limits:?})"); } else { println!("FAIL: {}", f.join(" · ")); std::process::exit(1); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_logits_give_zero_kl() {
        let a = vec![1.0f32, 2.0, 3.0, -1.0];
        let s = compare_step(2, &a, &a);
        assert!(s.kl.abs() < 1e-12 && s.same_top1 && (s.nll_ref - s.nll_var).abs() < 1e-12);
    }
    #[test]
    fn kl_of_two_known_distributions() {
        // p = softmax([0, ln 3]) = (1/4, 3/4), q = (1/2, 1/2): KL = 1/4 ln(1/2) + 3/4 ln(3/2)
        let s = compare_step(1, &[0.0, 3f32.ln()], &[0.0, 0.0]);
        let want = 0.25 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
        assert!((s.kl - want).abs() < 1e-6, "{} vs {want}", s.kl);
        assert!((s.nll_ref - (-(0.75f64).ln())).abs() < 1e-6 && (s.nll_var - 2f64.ln()).abs() < 1e-6);
    }
    #[test]
    fn limits() {
        assert_eq!(parse_limit("median<=0.001"), Ok(Limit::Median(0.001)));
        assert_eq!(parse_limit("top1>=95"), Ok(Limit::Top1(95.0)));
        assert!(parse_limit("kl<1").is_err());
        let s = Summary { steps: 1, kl_mean: 0.3, kl_median: 0.0002, kl_p99: 0.0, kl_max: 0.0, kl_max_at: 0, top1: 91.0, nll_ref: 1.0, nll_var: 1.0 };
        assert!(failed(&s, &[Limit::Median(0.001), Limit::Top1(90.0)]).is_empty());
        assert_eq!(failed(&s, &[Limit::Mean(0.01), Limit::Top1(95.0)]).len(), 2);
    }
    #[test]
    fn summary_p99_and_max() {
        let v: Vec<Step> = (0..100).map(|i| Step { kl: i as f64, same_top1: i % 2 == 0, nll_ref: 1.0, nll_var: 2.0 }).collect();
        let s = summarize(&v);
        assert_eq!((s.kl_max, s.kl_max_at, s.kl_p99, s.top1, s.kl_median), (99.0, 99, 98.0, 50.0, 50.0));
    }
}
