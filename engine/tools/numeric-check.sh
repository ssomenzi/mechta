#!/bin/bash
# Numeric check of an engine variant against the default engine, before it becomes default.
# Two texts: code (`text-code.txt`) checked with median KL + top-1, English prose (`text-en.txt`) checked with
# mean KL. Prefill 16,384 tokens, 256 steps of the MTP double step (the generation path).
# The reference files are made once with the default engine and kept in OUT_DIR (delete them when the default
# engine changes). Uses the GPU: one model load per text and per variant (~2 min each).
# Usage:
#   TEXTS_DIR=/path/to/texts OUT_DIR=/path/to/out VAR_ENV="MECHTA_X=1 MECHTA_Y=2" NAME=x ./numeric-check.sh
# TEXTS_DIR holds text-code.txt and text-en.txt; OUT_DIR receives the logit dumps.
# Default limits: code median<=0.001 top1>=85, prose mean<=0.01. CODE_LIMITS / PROSE_LIMITS override them.
# Exit code 0 = PASS on both texts, 1 = FAIL, 2 = wrong use.
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
[ -n "${TEXTS_DIR:-}" ] || { echo "set TEXTS_DIR (directory with text-code.txt and text-en.txt)"; exit 2; }
[ -n "${OUT_DIR:-}" ] || { echo "set OUT_DIR (directory for the reference and variant dumps)"; exit 2; }
M=$DIR/../target/release/mechta
Q=$TEXTS_DIR
D=$OUT_DIR; mkdir -p $D
PF=${MECHTA_PREFLIGHT:-}   # optional machine check (see engine/mechta-serve.sh)
CODE_LIMITS=${CODE_LIMITS:-"median<=0.001 top1>=85"}; PROSE_LIMITS=${PROSE_LIMITS:-"mean<=0.01"}
[ -n "${NAME:-}" ] || { echo "set NAME (and VAR_ENV)"; exit 2; }
ss -lntH | grep -q ':4131 ' && { echo "STOP: port 4131 busy (a server is on)"; exit 2; }
dump() {  # $1 = env, $2 = text, $3 = out
  if [ -n "$PF" ]; then "$PF" >/dev/null 2>&1 || { echo "STOP: the machine check failed"; exit 2; }; fi
  env MECHTA_RESERVE_GB=2 $1 $M logit-dump $Q/text-$2.txt 16384 256 $3 double 2>&1 | grep -E "prefill|steps|panic|rror"
}
rc=0
for t in code en; do
  ref=$D/rif-$t.bin
  [ -s $ref ] || { echo "== reference $t (default engine)"; dump "" $t $ref; }
  echo "== variant $NAME on $t: ${VAR_ENV:-}"
  dump "${VAR_ENV:-}" $t $D/$NAME-$t.bin
  if [ $t = code ]; then L=$CODE_LIMITS; else L=$PROSE_LIMITS; fi
  $M kl $ref $D/$NAME-$t.bin $L || rc=1
done
echo "== $NAME: $([ $rc = 0 ] && echo PASS || echo FAIL)"; exit $rc
