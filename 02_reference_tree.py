#!/usr/bin/env python3
"""Alignment-based reference trees for the 65 genomes.

1. Take the annotated replicase polyprotein (ORF1ab) translation of every record.
2. Align the polyproteins quickly (MAFFT FFT-NS-2) only to locate, in every genome, the
   columns homologous to SARS-CoV-2 nsp12 (RdRp) and to the ORF1b-encoded region nsp12-nsp16.
3. Re-align each extracted region accurately (MAFFT L-INS-i), drop gap-rich columns, and
   infer a maximum-likelihood tree (IQ-TREE 2: ModelFinder, 1000 ultrafast bootstraps,
   1000 SH-aLRT replicates).

Outputs go to data/reftree/.
"""
import os
import subprocess
import sys

from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CGA_DATA", os.path.join(HERE, "data"))
OUT = os.path.join(DATA, "reftree")
os.makedirs(OUT, exist_ok=True)
THREADS = os.environ.get("CGA_THREADS", "4")
REF = "NC_045512"
GAP_MAX = 0.2  # drop alignment columns with more than this fraction of gaps
RDRP_MODEL = "LG+F+R5"  # best-fit model (BIC) found by ModelFinder for the RdRp alignment


def accessions():
    with open(os.path.join(DATA, "records.tsv")) as fh:
        next(fh)
        return [line.split("\t")[0] for line in fh]


def replicase(rec):
    """Return the replicase polyprotein: ORF1ab if annotated as one CDS, else ORF1a + ORF1b."""
    cds = [f for f in rec.features if f.type == "CDS" and "translation" in f.qualifiers]
    cds.sort(key=lambda f: int(f.location.start))
    first = [f for f in cds if int(f.location.start) == int(cds[0].location.start)]
    longest = max(first, key=lambda f: len(f.qualifiers["translation"][0]))
    prot = longest.qualifiers["translation"][0]
    how = "ORF1ab CDS"
    if len(prot) < 5500:  # ORF1a only: append the next CDS (ORF1b), which starts near its end
        end = int(longest.location.end)
        nxt = [f for f in cds if abs(int(f.location.start) - end) < 200 and f is not longest]
        if not nxt:
            raise ValueError(f"{rec.id}: no ORF1b found")
        b = max(nxt, key=lambda f: len(f.qualifiers["translation"][0]))
        prot += b.qualifiers["translation"][0]
        how = "ORF1a + ORF1b CDS concatenated"
    return prot, how


def ref_coords(rec):
    """Protein coordinates (0-based, half-open) of nsp12 and of the end of pp1ab in the reference."""
    cds = max((f for f in rec.features if f.type == "CDS"),
              key=lambda f: len(f.qualifiers.get("translation", [""])[0]))
    cds_start = int(cds.location.start)
    for f in rec.features:
        if f.type == "mat_peptide" and "RNA-dependent RNA polymerase" in f.qualifiers.get("product", [""])[0]:
            start = (int(f.location.start) - cds_start) // 3
            length = len(f.location) // 3
            return start, start + length, len(cds.qualifiers["translation"][0])
    raise ValueError("nsp12 not annotated in reference")


def run(cmd, stdout=None):
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=stdout, stderr=subprocess.DEVNULL)


def read_fasta(path):
    return {r.id: str(r.seq) for r in SeqIO.parse(path, "fasta")}


def write_fasta(path, seqs):
    with open(path, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


def region(aln, col_from, col_to):
    return {k: v[col_from:col_to].replace("-", "") for k, v in aln.items()}


def trim(aln):
    n = len(aln)
    length = len(next(iter(aln.values())))
    keep = [i for i in range(length) if sum(s[i] == "-" for s in aln.values()) / n <= GAP_MAX]
    return {k: "".join(v[i] for i in keep) for k, v in aln.items()}, length, len(keep)


def main():
    accs = accessions()
    prots, notes = {}, []
    for a in accs:
        rec = SeqIO.read(os.path.join(DATA, "genbank", a + ".gb"), "genbank")
        prots[a], how = replicase(rec)
        notes.append(f"{a}\t{len(prots[a])}\t{how}")
        if a == REF:
            s, e, plen = ref_coords(rec)
    write_fasta(os.path.join(OUT, "replicase.faa"), prots)
    with open(os.path.join(OUT, "replicase-sources.tsv"), "w") as fh:
        fh.write("accession\tlength_aa\tsource\n" + "\n".join(notes) + "\n")
    print(f"reference nsp12: pp1ab residues {s + 1}-{e} of {plen}", file=sys.stderr)

    rough = os.path.join(OUT, "replicase.rough.aln")
    with open(rough, "w") as fh:
        run(["mafft", "--retree", "2", "--thread", THREADS, os.path.join(OUT, "replicase.faa")], stdout=fh)
    aln = read_fasta(rough)

    # map reference residue indices to alignment columns
    cols, i = [], 0
    for c, ch in enumerate(aln[REF]):
        if ch != "-":
            cols.append(c)
            i += 1
    c_from, c_to, c_end = cols[s], cols[e - 1] + 1, cols[plen - 1] + 1

    summary = []
    for name, (a, b) in {"rdrp": (c_from, c_to), "orf1b": (c_from, c_end)}.items():
        seqs = region(aln, a, b)
        write_fasta(os.path.join(OUT, f"{name}.faa"), seqs)
        lens = sorted(len(v) for v in seqs.values())
        fine = os.path.join(OUT, f"{name}.aln")
        if not os.path.exists(fine) or os.path.getsize(fine) == 0 or os.environ.get("CGA_REDO"):
            with open(fine, "w") as fh:
                run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", THREADS,
                     os.path.join(OUT, f"{name}.faa")], stdout=fh)
        trimmed, before, after = trim(read_fasta(fine))
        write_fasta(os.path.join(OUT, f"{name}.trim.aln"), trimmed)
        summary.append(f"{name}\tsegment length {lens[0]}-{lens[-1]} aa\talignment {before} columns\t"
                       f"{after} kept (<= {int(GAP_MAX * 100)}% gaps)")
        # ModelFinder over all protein matrices for the primary (RdRp) tree. The longer ORF1b
        # sensitivity alignment is run under the model selected for RdRp to keep the run short.
        model = "MFP" if name == "rdrp" else RDRP_MODEL
        if not os.path.exists(os.path.join(OUT, f"{name}.treefile")) or os.environ.get("CGA_REDO"):
            run(["iqtree2", "-s", os.path.join(OUT, f"{name}.trim.aln"), "-st", "AA", "-m", model,
                 "-B", "1000", "-alrt", "1000", "-T", THREADS, "-seed", "240", "-redo", "--prefix",
                 os.path.join(OUT, name)])
    with open(os.path.join(OUT, "alignment-summary.tsv"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
