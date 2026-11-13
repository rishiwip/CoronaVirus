"""Disjoint-data floors for the benchmark.

A benchmark needs to know how much two trees of the same genomes differ when both are built
from alignments but from different data. The RdRp and ORF1b alignments used so far overlap:
RdRp is the first 921 columns of the 2,655-column ORF1b alignment, so their agreement is
partly circular. This script builds floors from strictly disjoint column sets.

  family     RdRp tree  vs  a tree from the ORF1b columns that lie outside RdRp
  subgenus   first half of the ORF1b alignment  vs  its second half

The subgenus ORF1b-versus-S comparison answers a different question (do different genes share
one history) and is kept separate, because within a recombining group they do not.

Writes data/floors.json.
"""
import json
import os
import subprocess
import sys

import cga

SUB = os.path.join(cga.DATA, "subgenus")
REFTREE = os.path.join(cga.DATA, "reftree")
THREADS = "4"


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def write_fasta(path, seqs):
    with open(path, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


def run_iqtree(aln, prefix, dtype, model):
    if os.path.exists(prefix + ".treefile") and not os.environ.get("CGA_REDO"):
        return
    cmd = ["iqtree2", "-s", aln, "-st", dtype, "-m", model, "-B", "1000", "-alrt", "1000",
           "-T", THREADS, "-seed", "240", "-redo", "--prefix", prefix]
    print("+", " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def splits_of(path, taxa):
    _, sup, _ = cga.read_newick_splits(path)
    ref = sorted(taxa)[0]
    return {cga.canon(s, frozenset(taxa), ref) for s in sup}


def model_of(iqtree_file):
    for line in open(iqtree_file):
        if line.startswith("Best-fit model"):
            return line.split(":")[-1].strip()
    return "GTR+F+R3"


def family_floor():
    """RdRp against the part of ORF1b that is not RdRp."""
    rdrp = read_fasta(os.path.join(REFTREE, "rdrp.faa"))
    orf1b = read_fasta(os.path.join(REFTREE, "orf1b.trim.aln"))
    n_rdrp_cols = len(next(iter(read_fasta(os.path.join(REFTREE, "rdrp.trim.aln")).values())))
    # the ORF1b alignment starts at the same residue as RdRp, so the disjoint part is its tail
    tail = {k: v[n_rdrp_cols:] for k, v in orf1b.items()}
    path = os.path.join(REFTREE, "orf1b-tail.aln")
    write_fasta(path, tail)
    prefix = os.path.join(REFTREE, "orf1b-tail")
    run_iqtree(path, prefix, "AA", "LG+F+R5")
    taxa = set(orf1b)
    a = splits_of(os.path.join(REFTREE, "rdrp.treefile"), taxa)
    b = splits_of(prefix + ".treefile", taxa)
    return {"columns_rdrp": n_rdrp_cols, "columns_disjoint_tail": len(next(iter(tail.values()))),
            "n_taxa": len(taxa), "nRF": cga.rf_normalised(a, b),
            "note": "RdRp tree vs a tree from the ORF1b columns outside RdRp; no shared columns",
            "lengths": [len(rdrp), len(orf1b)][:0] or None}


def subgenus_floor(name):
    """First half of the ORF1b alignment against its second half."""
    folder = os.path.join(SUB, name)
    aln = read_fasta(os.path.join(folder, "orf1b.trim.aln"))
    if not aln:
        return None
    model = model_of(os.path.join(folder, "orf1b.iqtree"))
    width = len(next(iter(aln.values())))
    mid = width // 2
    out = {"columns": width, "model": model, "n_taxa": len(aln)}
    halves = {}
    for half, (a, b) in {"half1": (0, mid), "half2": (mid, width)}.items():
        path = os.path.join(folder, f"orf1b-{half}.aln")
        write_fasta(path, {k: v[a:b] for k, v in aln.items()})
        prefix = os.path.join(folder, f"orf1b-{half}")
        run_iqtree(path, prefix, "DNA", model)
        halves[half] = splits_of(prefix + ".treefile", set(aln))
    out["nRF_half1_vs_half2"] = cga.rf_normalised(halves["half1"], halves["half2"])
    return out


if __name__ == "__main__":
    floors = {"family": family_floor()}
    print("family", floors["family"], flush=True)
    for name in sorted(os.listdir(SUB)):
        if os.path.isdir(os.path.join(SUB, name)):
            res = subgenus_floor(name)
            if res:
                floors[name] = res
                print(name, res, flush=True)
    json.dump(floors, open(os.path.join(cga.DATA, "floors.json"), "w"), indent=1)
    print("written", os.path.join(cga.DATA, "floors.json"))
