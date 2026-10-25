#!/bin/bash
# Numeric reference for M4 (MTP layer): llama.cpp 42d9581, CPU ONLY (-ngl 0), same prompt and same parameters as M1.
# Needs:
#   MODEL       path to Qwen3.6-35B-A3B-UD-Q4_K_S.gguf (the unsloth MTP GGUF)
#   DUMP_MTP    path to the llama-mechta-dump-mtp binary (a llama.cpp build with the mechta-dump-mtp example)
DIR=$(cd "$(dirname "$0")" && pwd)
[ -n "${MODEL:-}" ] || { echo "set MODEL to the path of Qwen3.6-35B-A3B-UD-Q4_K_S.gguf" >&2; exit 2; }
[ -n "${DUMP_MTP:-}" ] || { echo "set DUMP_MTP to the path of the llama-mechta-dump-mtp binary" >&2; exit 2; }
echo $$ > "$DIR/dump.pid"
cd "$DIR"
F=$MODEL
DUMP_DIR=. NGEN=32 "$DUMP_MTP" \
  -m "$F" -f ../reference/prompt.txt -ngl 0 -t 8 -c 1024 > dump.log 2>&1
echo "exit $?"; tail -5 dump.log; ls -la tensors.bin token.tsv
echo "=== done $(date +%T)"
