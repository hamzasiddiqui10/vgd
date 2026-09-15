#!/usr/bin/env python3
"""Show / grab GPUs on the Grete cluster (NHR Göttingen).

    gpu.py                  list immediately available GPUs per type
    gpu.py A100             start an interactive session on one A100
    gpu.py 1g.10gb -t 2:00  ... on a MIG slice, for 2 hours

Resource defaults are derived from what the nodes actually provide
(see the GPUS table below): each GPU gets its proportional share of the
node's CPUs and usable memory. All of it can be overridden.

GPUs held by grete:preemptible jobs are reported in their own column but
NOT counted as available: grete:preemptible has PreemptMode=OFF and no job
on the cluster was actually preempted/requeued in the last 7 days, so those
GPUs do not free up when you submit.  --preempt counts them anyway.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# GPU catalogue.  cpus/mem are *per GPU* and reflect the node hardware:
#   A100 nodes:   128 CPUs / ~486 GiB usable / 4 GPUs  -> 32 CPUs, 120G
#   A100 (80gb):  ggpu202 256/8, ggpu211-212 128/4     -> 32 CPUs, 120G
#   H100 nodes:   192 CPUs / ~989 GiB usable / 4 GPUs  -> 48 CPUs, 240G
#   MIG nodes:    128 CPUs / ~486 GiB usable, shared between the slices
#                 proportionally to slice size (10gb slice ~ 1/32 of a node)
# ---------------------------------------------------------------------------
GPUS = [
    # name        partition             cpus  mem     constraint
    ("A100",      "grete:shared",         32, "120G", None),
    ("A100:80gb", "grete:shared",         32, "120G", "80gb"),
    ("H100",      "grete-h100:shared",    48, "240G", None),
    ("3g.40gb",   "grete:interactive",    16, "60G",  "inet"),
    ("2g.20gb",   "grete:interactive",     8, "30G",  "inet"),
    ("2g.10gb",   "grete:interactive",     8, "30G",  "inet"),
    ("1g.20gb",   "grete:interactive",     8, "30G",  "inet"),
    ("1g.10gb",   "grete:interactive",     4, "15G",  "inet"),
]
SPEC = dict((g[0], g) for g in GPUS)
# partition MaxTime: grete:shared / grete-h100:shared 2 days, interactive 12 h
DEFAULT_TIME = {"grete:shared": "12:00:00", "grete-h100:shared": "12:00:00",
                "grete:interactive": "12:00:00"}
PREEMPT_PARTITION = "grete:preemptible"
BAD_STATES = ("DOWN", "DRAIN", "DRAINING", "DRAINED", "FAIL", "FAILING",
              "MAINT", "RESERVED", "INVAL", "UNKNOWN", "NOT_RESPONDING",
              "POWER_DOWN", "POWERED_DOWN", "POWERING_DOWN", "REBOOT_ISSUED",
              "REBOOT_REQUESTED")


def mib(text):
    """'120G' / '515810M' / '64' -> MiB."""
    m = re.match(r"^\s*([0-9.]+)\s*([KMGT]?)i?B?\s*$", text, re.I)
    if not m:
        raise ValueError("cannot parse memory: %r" % text)
    val, unit = float(m.group(1)), m.group(2).upper()
    return int(val * {"": 1, "K": 1 / 1024.0, "M": 1, "G": 1024, "T": 1024 ** 2}[unit])


def run(cmd):
    return subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.PIPE)


def field(line, key, default=""):
    m = re.search(r"(?:^| )%s=(\S*)" % key, line)
    return m.group(1) if m else default


def tres(text):
    """'cpu=8,mem=64G,gres/gpu:A100=2' -> dict"""
    out = {}
    for item in text.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def preemptible_usage():
    """node -> {'cpu': n, 'mem': MiB, 'gpu': {type: n}} held by preemptible jobs."""
    usage = defaultdict(lambda: {"cpu": 0, "mem": 0, "gpu": defaultdict(int)})
    try:
        out = run(["squeue", "-h", "-t", "R", "-p", PREEMPT_PARTITION,
                   "-O", "NumNodes:8,NodeList:200,tres-alloc:400"])
    except (subprocess.CalledProcessError, OSError):
        return usage
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        nnodes, nodelist, alloc = int(parts[0]), parts[1], parts[2]
        try:
            nodes = run(["scontrol", "show", "hostnames", nodelist]).split()
        except (subprocess.CalledProcessError, OSError):
            continue
        t = tres(alloc)
        for node in nodes:
            u = usage[node]
            u["cpu"] += int(int(t.get("cpu", 0)) / max(nnodes, 1))
            u["mem"] += int(mib(t.get("mem", "0")) / max(nnodes, 1))
            for k, v in t.items():
                if k.startswith("gres/gpu:"):
                    u["gpu"][k.split(":", 1)[1]] += int(int(v) / max(nnodes, 1))
    return usage


def gres_by_node():
    """node -> {gpu type: (total, used)}; typed GRES is not in CfgTRES for A100."""
    out = {}
    parts = ",".join(sorted(set(g[1] for g in GPUS)))
    for line in run(["sinfo", "-h", "-N", "-p", parts,
                     "-O", "nodelist:40,gres:200,gresused:200"]).splitlines():
        cols = line.split()
        if len(cols) < 3:
            continue
        name, gres, used = cols[0], cols[1], cols[2]
        counts = {}
        for text, idx in ((gres, 0), (used, 1)):
            for gtype, n in re.findall(r"gpu:([^:,]+):(\d+)",
                                       re.sub(r"\([^)]*\)", "", text)):
                entry = counts.setdefault(gtype, [0, 0])
                entry[idx] = int(n)
        out[name] = dict((k, tuple(v)) for k, v in counts.items())
    return out


def node_info(count_preempt):
    """Yield per-node availability dicts for the partitions we care about."""
    partitions = set(g[1] for g in GPUS)
    usage = preemptible_usage()
    preempt = usage if count_preempt else {}
    gres = gres_by_node()
    for line in run(["scontrol", "show", "node", "--oneliner"]).splitlines():
        parts = set(field(line, "Partitions").split(","))
        if not parts & partitions:
            continue
        state = field(line, "State")
        if any(bad in state.split("+") for bad in BAD_STATES):
            continue
        name = field(line, "NodeName")
        alloc = tres(field(line, "AllocTRES"))
        free = preempt.get(name, {"cpu": 0, "mem": 0, "gpu": {}})
        real = mib(field(line, "RealMemory", "0"))
        spec = mib(field(line, "MemSpecLimit", "0"))
        gpus = {}
        for gtype, (total, used) in gres.get(name, {}).items():
            idle = min(total, total - used + free["gpu"].get(gtype, 0))
            gpus[gtype] = (total, max(0, idle))
        held = usage.get(name, {"gpu": {}})["gpu"]
        yield {
            "name": name,
            "state": state,
            "partitions": parts,
            "features": set(field(line, "ActiveFeatures").split(",")),
            "cpus": int(field(line, "CPUTot", "0")) - int(field(line, "CPUAlloc", "0"))
                    + free["cpu"],
            "mem": real - spec - mib(alloc.get("mem", "0")) + free["mem"],
            "gpus": gpus,
            "preempt_gpus": held,
        }


def availability(count_preempt):
    """gpu name -> [total, idle, ready, best_single_node, planned, preempt]"""
    stat = dict((name, [0, 0, 0, 0, 0, 0]) for name in SPEC)
    for node in node_info(count_preempt):
        for name, part, cpus, mem, constraint in GPUS:
            gtype = name.split(":")[0]
            if part not in node["partitions"] or gtype not in node["gpus"]:
                continue
            if constraint and constraint not in node["features"]:
                continue
            total, idle = node["gpus"][gtype]
            ready = max(0, min(idle, node["cpus"] // cpus, node["mem"] // mib(mem)))
            s = stat[name]
            s[0] += total
            s[1] += max(0, idle)
            s[2] += ready
            s[3] = max(s[3], ready)
            if "PLANNED" in node["state"].split("+"):
                # backfill earmarked this node for a pending higher-priority job
                s[4] += ready
            s[5] += node["preempt_gpus"].get(gtype, 0)
    return stat


def show(count_preempt):
    stat = availability(count_preempt)
    print("%-11s %-18s %9s %8s %6s %8s %6s   %s" %
          ("GPU", "PARTITION", "READY", "PLANNED", "IDLE", "PREEMPT", "TOTAL",
           "PER-GPU REQUEST"))
    for name, part, cpus, mem, constraint in GPUS:
        total, idle, ready, best, planned, preempt = stat[name]
        print("%-11s %-18s %9s %8d %6d %8d %6d   %d cpus, %s%s" % (
            name, part, "%d (%d)" % (ready, best) if ready else "0",
            planned, idle, preempt, total, cpus, mem,
            "" if not constraint else "  [-C %s]" % constraint))
    print("\nREADY   GPUs allocatable right now with the request shown; in "
          "parentheses the most you")
    print("        can get in a single job (jobs are limited to one node)."
          + (" Includes PREEMPT GPUs." if count_preempt else ""))
    print("PLANNED how many of the READY GPUs sit on nodes the backfill "
          "scheduler earmarked for a")
    print("        pending higher-priority job -- those may make you wait, or "
          "need a short job")
    print("        (-t 2:00:00 --qos 2h) to squeeze in.")
    print("IDLE    unallocated GPUs, ignoring CPU/memory availability.")
    print("PREEMPT GPUs held by grete:preemptible jobs. In theory yours would "
          "preempt them, but that")
    print("        partition has PreemptMode=OFF and nothing was preempted "
          "cluster-wide in the last")
    print("        7 days -- so they are not in READY unless you pass "
          "--preempt.")


def allocate(args):
    key = {"a100": "A100", "h100": "H100", "a100-80gb": "A100:80gb",
           "a10080gb": "A100:80gb", "80gb": "A100:80gb"}.get(
               args.gpu.lower().replace("_", "-"), args.gpu)
    if key not in SPEC:
        sys.exit("unknown GPU %r; known: %s" % (args.gpu, ", ".join(g[0] for g in GPUS)))
    name, part, cpus, mem, constraint = SPEC[key]
    gtype = name.split(":")[0]
    cmd = ["srun", "--pty",
           "-p", part,
           "-t", args.time or DEFAULT_TIME[part],
           "-G", "%s:%d" % (gtype, args.count),
           "--cpus-per-gpu=%d" % (args.cpus or cpus),
           "--mem-per-gpu=%s" % (args.mem or mem)]
    constraint = args.constraint or constraint
    if constraint:
        cmd += ["-C", constraint]
    if args.qos:
        cmd += ["--qos=" + args.qos]
    cmd += args.extra or [os.environ.get("SHELL_FOR_SRUN", "/bin/bash")]
    print("+ " + " ".join(cmd), file=sys.stderr)
    if args.dry_run:
        return
    os.execvp(cmd[0], cmd)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gpu", nargs="?", help="GPU to request, e.g. A100, H100, 1g.10gb")
    p.add_argument("-n", "--count", type=int, default=1, help="number of GPUs (default 1)")
    p.add_argument("-t", "--time", help="walltime, e.g. 2:00:00 (default 12:00:00)")
    p.add_argument("-c", "--cpus", type=int, help="CPUs per GPU (default: node fair share)")
    p.add_argument("-m", "--mem", help="memory per GPU, e.g. 60G")
    p.add_argument("-C", "--constraint", help="node constraint, e.g. 80gb,inet")
    p.add_argument("--qos", help="QOS, e.g. 2h for a short high-priority job")
    p.add_argument("--preempt", action="store_true",
                   help="count GPUs held by grete:preemptible jobs as available")
    p.add_argument("--dry-run", action="store_true",
                   help="only print the srun command")
    p.epilog = "Anything after a bare -- is run instead of a login shell."
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = p.parse_args(argv)
    args.extra = extra
    if args.gpu:
        allocate(args)
    else:
        show(args.preempt)


if __name__ == "__main__":
    main()
