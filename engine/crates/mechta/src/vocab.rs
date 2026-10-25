//! M4 (reduced head for the drafts): which tokens does the model output really use in the sessions?
//! `mechta vocab <text.txt>`: tokenizes the file (Qwen3.6 tokenizer), counts the distinct tokens and the coverage
//! of the K most frequent tokens; writes `frequencies.tsv` (id, count) next to the file, in order of frequency.
pub fn vocab(path: &str) {
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let text = std::fs::read_to_string(path).expect("text");
    // in pieces of 1 MB so memory does not blow up
    let mut count = vec![0u64; crate::engine::VOCAB];
    let mut tot = 0u64;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut j = (i + 1 << 20).min(bytes.len());
        while j < bytes.len() && !text.is_char_boundary(j) { j += 1; }
        let enc = tk.encode(&text[i..j], false).expect("encode");
        for &id in enc.get_ids() { count[id as usize] += 1; tot += 1; }
        i = j;
    }
    let distinct = count.iter().filter(|&&c| c > 0).count();
    let mut ord: Vec<(u32, u64)> = count.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, c)| (i as u32, *c)).collect();
    ord.sort_by(|a, b| b.1.cmp(&a.1));
    println!("total tokens {tot} · distinct {distinct} of {}", crate::engine::VOCAB);
    let mut acc = 0u64; let mut next = [0.9, 0.99, 0.999, 0.9999, 1.0].iter().peekable();
    for (k, (_, c)) in ord.iter().enumerate() {
        acc += c;
        while let Some(&&q) = next.peek() {
            if acc as f64 >= q * tot as f64 { println!("  coverage {:.2}%: the first {} tokens", 100.0 * q, k + 1); next.next(); } else { break; }
        }
    }
    for k in [8192usize, 16384, 32768, 49152, 65536] {
        let c: u64 = ord.iter().take(k).map(|x| x.1).sum();
        println!("  the first {k:>6} tokens cover {:.3}%", 100.0 * c as f64 / tot as f64);
    }
    let mut s = String::new();
    for (id, c) in &ord { s.push_str(&format!("{id}\t{c}\n")); }
    let out = std::path::Path::new(path).with_file_name("frequenze.tsv");
    std::fs::write(&out, s).unwrap();
    println!("written {}", out.display());
}
