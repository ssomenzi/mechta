#!/usr/bin/env python3
"""Converts a mechta trace (MECHTA_TRACE=1|2, JSON lines) to the Chrome trace event format, which Perfetto
(ui.perfetto.dev, it runs in the browser: the file is not uploaded) and chrome://tracing open.

Usage: trace-export.py <trace.jsonl> [out.json]   (default: same name with .json)

Tracks: "CPU" with one row per thread (engine, http, root saver), "GPU" with the submits and, nested inside them,
the attention groups of MECHTA_TRACE=2. Times in microseconds from the start of the process."""
import json, sys

THREADS = {0: "engine", 1: "http", 2: "root saver"}
CPU, GPU = 1, 2


def convert(lines):
    out, lost = [], 0
    for name, pid in (("CPU", CPU), ("GPU", GPU)):
        out.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": name}})
    for tid, name in THREADS.items():
        out.append({"ph": "M", "name": "thread_name", "pid": CPU, "tid": tid, "args": {"name": name}})
    out.append({"ph": "M", "name": "thread_name", "pid": GPU, "tid": 0, "args": {"name": "submits"}})
    for line in lines:
        e = json.loads(line)
        if "header" in e:
            continue
        if "lost" in e:
            lost += e["lost"]
            continue
        k = e["k"]
        gpu = k == "gpu" or k.startswith("attn.")
        args = {"label": e["l"], "pos": e["pos"], "tokens": e["n"]}
        if e["b"]:
            args["block"] = e["b"]
        if e["ly"] != 255:
            args["layer"] = e["ly"]
        dur_ns = max(e["t1"] - e["t0"], 1)
        if e["by"]:
            args["kv_bytes"] = e["by"]
            args["kv_GB_s"] = round(e["by"] / dur_ns, 1)
        name = e["l"] if k == "gpu" else k if gpu else f"{k} ({e['l']})" if e["l"] else k
        out.append({"ph": "X", "name": name, "cat": k, "ts": e["t0"] / 1000, "dur": dur_ns / 1000,
                    "pid": GPU if gpu else CPU, "tid": 0 if gpu else e["th"], "args": args})
    return {"traceEvents": out, "displayTimeUnit": "ms", "otherData": {"lost_events": lost}}


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".json"
    with open(src) as f:
        t = convert(f)
    with open(dst, "w") as f:
        json.dump(t, f)
    print(f"{dst}: {len(t['traceEvents'])} events, {t['otherData']['lost_events']} lost")
