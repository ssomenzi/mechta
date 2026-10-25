#!/bin/bash
# Start and stop the mechta server as a systemd user unit, so it outlives the terminal that started it. Usage:
#   ./mechta-serve.sh start | stop | status | log
#
# Required for `start`:
#   MECHTA_MODEL=<file.gguf>          the model (Qwen3.6-35B-A3B and family, or Flint)
#   MECHTA_TOKENIZER=<tokenizer.json> the tokenizer of that model (tokenizers/qwen3.6-tokenizer.json or tokenizers/flint-tokenizer.json)
#   MECHTA_CACHE=<directory>          where to save the roots (prefix cache across sessions), created if missing
# Optional:
#   MECHTA_ADDRESS=host:port     listen address, default 127.0.0.1:4131 (this machine only). 0.0.0.0:4131 opens it
#                                to the whole network: only with a firewall in front and, preferably, with MECHTA_API_KEY.
#   MECHTA_API_KEY=<key>         if set, /v1/* requires `Authorization: Bearer <key>` (/health stays open)
#   MECHTA_PREFLIGHT=<command>   a machine check run before the start; the server starts only if it exits with 0
#   MECHTA_KV=f32|f16 (KV cache format, default f16) · MECHTA_MTP_ACCEPT=exact|greedy · MECHTA_THINKING=no
#   MECHTA_CTX=N (context in tokens, multiple of 1024, 4096-262144; default 65536: `mechta check-memory` says whether it fits)
#   MECHTA_THINK_BUDGET=N (limit on reasoning, 0 = none) · MECHTA_FLINT_KV=f16 (K/V cache of a Flint model in f16)
UNIT=mechta-serve
DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
BIN=$DIR/target/release/mechta
PREFLIGHT=${MECHTA_PREFLIGHT:-}
case "$1" in
  start)
    if systemctl --user is-active --quiet $UNIT; then echo "already running"; exit 0; fi
    for v in MECHTA_MODEL MECHTA_TOKENIZER MECHTA_CACHE; do
      [ -n "${!v}" ] || { echo "STOP: $v is missing (see the header of this script)"; exit 1; }
    done
    [ -r "$MECHTA_MODEL" ] || { echo "STOP: MECHTA_MODEL not readable: $MECHTA_MODEL"; exit 1; }
    [ -r "$MECHTA_TOKENIZER" ] || { echo "STOP: MECHTA_TOKENIZER not readable: $MECHTA_TOKENIZER"; exit 1; }
    [ -x "$BIN" ] || { echo "STOP: binary not found ($BIN): cargo build --release in $DIR"; exit 1; }
    if [ -n "$PREFLIGHT" ]; then
      [ -x "$PREFLIGHT" ] || { echo "STOP: MECHTA_PREFLIGHT is not executable: $PREFLIGHT"; exit 1; }
      "$PREFLIGHT" >/dev/null 2>&1 || { echo "STOP: the machine check failed ($PREFLIGHT)"; exit 1; }
    fi
    ADDR=${MECHTA_ADDRESS:-127.0.0.1:4131}
    PORT=${ADDR##*:}
    ss -lntH | grep -q ":$PORT " && { echo "STOP: port $PORT is already in use by something else"; exit 1; }
    CTX=${MECHTA_CTX:-65536}
    MEM=$(MECHTA_KV=${MECHTA_KV:-f16} $BIN check-memory "$CTX" 2>&1) || { echo "STOP: $MEM"; exit 1; }
    echo "$MEM"
    systemd-run --user --unit=$UNIT --collect -p WorkingDirectory="$DIR" \
      -E MECHTA_MODEL="$MECHTA_MODEL" -E MECHTA_TOKENIZER="$MECHTA_TOKENIZER" -E MECHTA_CACHE="$MECHTA_CACHE" \
      -E MECHTA_ADDRESS="$ADDR" -E MECHTA_API_KEY="${MECHTA_API_KEY:-}" -E MECHTA_MODEL_ID="${MECHTA_MODEL_ID:-}" \
      -E MECHTA_RESERVE_GB=2 -E MECHTA_KV=${MECHTA_KV:-f16} -E MECHTA_MTP_ACCEPT=${MECHTA_MTP_ACCEPT:-exact} -E MECHTA_THINKING=${MECHTA_THINKING:-yes} \
      -E MECHTA_CTX=$CTX -E MECHTA_THINK_BUDGET=${MECHTA_THINK_BUDGET:-0} -E MECHTA_FLINT_KV=${MECHTA_FLINT_KV:-} $BIN serve
    echo "started on $ADDR (context $CTX): the model loads in ~10 s, then the port opens ('$0 log' to follow it)" ;;
  stop)   systemctl --user stop $UNIT ;;
  status) systemctl --user status $UNIT --no-pager | head -5; ss -lntH | grep ':4131 ' || echo "port 4131 closed" ;;
  log)    journalctl --user -u $UNIT -f -o cat ;;
  *) echo "usage: $0 start|stop|status|log"; exit 1 ;;
esac
