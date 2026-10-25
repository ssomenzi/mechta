//! Minimal GGUF reader for mechta: header, metadata, tensor table, reading of the bytes.
//! No `unsafe`, no mmap: reading is done with `File` + `seek` + `read_exact`.
//! A malformed file must give an error, never a panic or a huge allocation: lengths and
//! offsets are added with checked arithmetic, strings have a cap, nested arrays a maximum depth.
#![forbid(unsafe_code)]
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Value { U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool), Str(String), U64(u64), I64(i64), F64(f64), Arr(Vec<Value>) }

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self { Value::U8(v) => Some(v as u64), Value::U16(v) => Some(v as u64), Value::U32(v) => Some(v as u64), Value::U64(v) => Some(v),
            Value::I32(v) if v >= 0 => Some(v as u64), Value::I64(v) if v >= 0 => Some(v as u64), _ => None }
    }
    pub fn as_f32(&self) -> Option<f32> { match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => None } }
    pub fn as_str(&self) -> Option<&str> { match self { Value::Str(s) => Some(s), _ => None } }
}

/// Tensor types used by the Qwen3.6 GGUFs (ggml numbers). Names as in ggml.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TType { F32, F16, Q4_1, Q8_0, Q3_K, Q4_K, Q6_K, BF16, Other(u32) }

impl TType {
    fn from(n: u32) -> Self { match n { 0 => TType::F32, 1 => TType::F16, 3 => TType::Q4_1, 8 => TType::Q8_0, 11 => TType::Q3_K, 12 => TType::Q4_K, 14 => TType::Q6_K, 30 => TType::BF16, x => TType::Other(x) } }
    /// (values per block, bytes per block) — from ggml-common.h
    pub fn block(&self) -> Option<(u64, u64)> {
        match self { TType::F32 => Some((1, 4)), TType::F16 | TType::BF16 => Some((1, 2)), TType::Q4_1 => Some((32, 20)), TType::Q8_0 => Some((32, 34)),
            TType::Q3_K => Some((256, 110)), TType::Q4_K => Some((256, 144)), TType::Q6_K => Some((256, 210)), TType::Other(_) => None }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor { pub name: String, pub dims: Vec<u64>, pub ttype: TType, pub offset: u64, pub nbytes: u64 }
impl Tensor { pub fn n_elements(&self) -> u64 { self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).unwrap_or(u64::MAX) } }

fn malformed(msg: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, format!("malformed GGUF: {msg}")) }
/// Maximum string in the metadata (the longest value is the chat template, ~10 KB; vocabularies are arrays of short strings).
const MAX_STRING: u64 = 64 << 20;
/// Nested arrays: one level for each `[`. Real GGUFs have at most one.
const MAX_DEPTH: u32 = 8;

pub struct Gguf { pub version: u32, pub meta: BTreeMap<String, Value>, pub tensors: Vec<Tensor>, pub data_start: u64, file: File }

fn rd<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> { let mut b = [0u8; N]; r.read_exact(&mut b)?; Ok(b) }
fn u32_(r: &mut impl Read) -> io::Result<u32> { Ok(u32::from_le_bytes(rd(r)?)) }
fn u64_(r: &mut impl Read) -> io::Result<u64> { Ok(u64::from_le_bytes(rd(r)?)) }
fn str_(r: &mut impl Read) -> io::Result<String> {
    let n = u64_(r)?;
    if n > MAX_STRING { return Err(malformed("string too long")); }
    // read with `take`, so a truncated file gives an I/O error without first allocating the whole declared length
    let mut b = Vec::new(); r.take(n).read_to_end(&mut b)?;
    if b.len() as u64 != n { return Err(malformed("truncated string")); }
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
fn val(r: &mut impl Read, t: u32, depth: u32) -> io::Result<Value> {
    Ok(match t {
        0 => Value::U8(rd::<1>(r)?[0]), 1 => Value::I8(rd::<1>(r)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(rd(r)?)), 3 => Value::I16(i16::from_le_bytes(rd(r)?)),
        4 => Value::U32(u32_(r)?), 5 => Value::I32(i32::from_le_bytes(rd(r)?)), 6 => Value::F32(f32::from_le_bytes(rd(r)?)),
        7 => Value::Bool(rd::<1>(r)?[0] != 0), 8 => Value::Str(str_(r)?),
        9 => {
            if depth >= MAX_DEPTH { return Err(malformed("arrays nested too deep")); }
            let et = u32_(r)?; let n = u64_(r)?;
            let mut v = Vec::with_capacity(n.min(1 << 20) as usize); for _ in 0..n { v.push(val(r, et, depth + 1)?); } Value::Arr(v)
        }
        10 => Value::U64(u64_(r)?), 11 => Value::I64(i64::from_le_bytes(rd(r)?)), 12 => Value::F64(f64::from_le_bytes(rd(r)?)),
        x => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown metadata type {x}"))),
    })
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file.try_clone()?);
        if &rd::<4>(&mut r)? != b"GGUF" { return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file")); }
        let version = u32_(&mut r)?;
        let nt = u64_(&mut r)?; let nkv = u64_(&mut r)?;
        let mut meta = BTreeMap::new();
        for _ in 0..nkv { let k = str_(&mut r)?; let t = u32_(&mut r)?; meta.insert(k, val(&mut r, t, 0)?); }
        let mut tensors = Vec::with_capacity(nt.min(1 << 16) as usize);
        for _ in 0..nt {
            let name = str_(&mut r)?; let nd = u32_(&mut r)?;
            if nd > 8 { return Err(malformed("tensor with more than 8 dimensions")); }
            let dims = (0..nd).map(|_| u64_(&mut r)).collect::<io::Result<Vec<_>>>()?;
            let ttype = TType::from(u32_(&mut r)?); let offset = u64_(&mut r)?;
            let n = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).ok_or_else(|| malformed("tensor dimensions out of range"))?;
            let nbytes = match ttype.block() {
                Some((per, bytes)) => (n / per).checked_mul(bytes).ok_or_else(|| malformed("tensor size out of range"))?,
                None => 0,
            };
            tensors.push(Tensor { name, dims, ttype, offset, nbytes });
        }
        let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
        if align == 0 || !align.is_power_of_two() { return Err(malformed("invalid general.alignment")); }
        let pos = r.stream_position()?;
        let data_start = pos.checked_next_multiple_of(align).ok_or_else(|| malformed("data start out of range"))?;
        Ok(Gguf { version, meta, tensors, data_start, file })
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> { self.tensors.iter().find(|t| t.name == name) }
    /// Reads the bytes of a tensor (or of a part of it: `skip` bytes from the start, `len` bytes).
    pub fn read(&self, t: &Tensor, skip: u64, len: u64, out: &mut Vec<u8>) -> io::Result<()> {
        if skip.checked_add(len).is_none_or(|end| end > t.nbytes) { return Err(io::Error::new(io::ErrorKind::InvalidInput, "read beyond the tensor")); }
        let start = self.data_start.checked_add(t.offset).and_then(|p| p.checked_add(skip)).ok_or_else(|| malformed("tensor offset out of range"))?;
        let mut f = &self.file;
        f.seek(SeekFrom::Start(start))?;
        out.resize(len as usize, 0);
        f.read_exact(out)
    }
    pub fn file_len(&self) -> io::Result<u64> { self.file.metadata().map(|m| m.len()) }
}
