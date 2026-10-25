# Third-party files

mechta is MIT (`LICENSE`). These files in the repository come from other projects and keep their own license.
None of them was modified by mechta: they were checked byte for byte against the upstream files.

| file | origin | license | copyright |
|---|---|---|---|
| `tokenizers/qwen3.6-tokenizer.json` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), `tokenizer.json` | Apache-2.0 | Copyright 2026 Alibaba Cloud |
| `engine/crates/chat/template.jinja` (built into the `mechta` binary) | the `tokenizer.chat_template` of [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF): Unsloth's modified version of Qwen's chat template (it merges up to two leading system/developer messages) | Apache-2.0 | Copyright 2026 Alibaba Cloud; modifications by Unsloth |
| `tokenizers/flint-tokenizer.json` | [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), `tokenizer.json` | Apache-2.0 | RecursiveSI |
| `engine/crates/chat/template-flint.jinja` (built into the `mechta` binary): Flint's `chat_template.jinja` | [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) | Apache-2.0 | RecursiveSI |

The text of the Apache License 2.0 is in `LICENSES/Apache-2.0.txt` (from https://www.apache.org/licenses/LICENSE-2.0.txt).
Neither Qwen nor RecursiveSI ships a NOTICE file for these files (Flint's `sglang/NOTICE` covers only its SGLang code).
The model weights are not part of this repository: download them from the pages above.

`kernel/patch-upstream/omarchy-0361.patch` is a Linux kernel patch (DRM/TTM), distributed under the Linux kernel's
license, GPL-2.0; it is a copy of the fix described in `kernel/README.md`, not part of mechta's MIT code.
