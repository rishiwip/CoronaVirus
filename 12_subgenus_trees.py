"""Reference trees for each within-subgenus genome set.

At this depth the RdRp protein is nearly invariant, so the reference moves to nucleotides.
Two genes are used per subgenus: ORF1b, the most conserved and least recombination-prone
region, which is the reference; and S, which both sets the floor for the benchmark and acts
as a recombination warning, since a large ORF1b/S disagreement means the genomes do not share
one history.

Coding sequences are taken from the record annotation where possible and otherwise located by
translated search against the subgenus reference, then aligned as codons (translate, align the
proteins with MAFFT L-INS-i, map back) so the nucleotide alignment stays in frame.

Writes data/subgenus/<name>/{orf1b,spike}.{fna,aln,trim.aln,treefile,...}
"""
import os
import subprocess
import sys

from Bio import SeqIO
from Bio.Seq import Seq

import cga

OUT = os.path.join(cga.DATA, "subgenus")
THREADS = "4"
GAP_MAX = 0.2
MIN_TAXA = 8

# Product names in these records are inconsistent ("polyprotein", "orf1ab orf1ab", "hypothetical
# protein"), so genes are located by coding-sequence length instead, which is unambiguous here:
# pp1ab is ~20 kb, a separately annotated ORF1b ~8 kb, and S 3.5-4.5 kb.
PP1AB_MIN = 18000
ORF1B_RANGE = (7000, 9500)
SPIKE_RANGE = (3000, 5200)
ORF1B_TAKE = 8001  # nt taken from the 3' end of pp1ab; lies wholly inside ORF1b in every genus


def run(cmd, stdout=None):
    print("+", " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, stdout=stdout, stderr=subprocess.DEVNULL)


def read_fasta(path):
    return {r.id: str(r.seq) for r in SeqIO.parse(path, "fasta")}


def write_fasta(path, seqs):
    with open(path, "w") as fh:
        for k, v in seqs.items():
            fh.write(f">{k}\n{v}\n")


def coding_sequences(rec):
    """In-frame nucleotide sequence of every annotated CDS, longest first."""
    out = []
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        try:
            seq = str(feat.extract(rec.seq)).upper().replace("U", "T")
        except Exception:  # noqa: BLE001 - malformed location in a few records
            continue
        out.append(seq[: len(seq) // 3 * 3])
    return sorted(out, key=len, reverse=True)


def translates_cleanly(nt, max_internal_stops=1):
    prot = str(Seq(nt).translate())
    return prot[:-1].count("*") <= max_internal_stops


def extract_gene(rec, gene):
    """Locate ORF1b or S by coding-sequence length, since product names are not consistent."""
    cds = coding_sequences(rec)
    if not cds:
        return None
    if gene == "orf1b":
        if len(cds[0]) >= PP1AB_MIN:
            # pp1ab is annotated whole: ORF1b is its 3' end
            nt = cds[0][-ORF1B_TAKE:]
            nt = nt[len(nt) % 3:]
            return nt if translates_cleanly(nt) else None
        # ORF1a and ORF1b annotated separately
        for seq in cds:
            if ORF1B_RANGE[0] <= len(seq) <= ORF1B_RANGE[1] and translates_cleanly(seq):
                return seq
        return None
    used = cds[0] if len(cds[0]) >= PP1AB_MIN else ""
    for seq in cds:
        if seq is used:
            continue
        if SPIKE_RANGE[0] <= len(seq) <= SPIKE_RANGE[1] and translates_cleanly(seq):
            return seq
    return None


def codon_align(seqs, path_stem):
    """Align in frame: translate, align proteins with L-INS-i, then thread the codons back."""
    prot = {k: str(Seq(v).translate()).replace("*", "X") for k, v in seqs.items()}
    write_fasta(path_stem + ".faa", prot)
    aln_faa = path_stem + ".prot.aln"
    if not os.path.exists(aln_faa) or os.environ.get("CGA_REDO"):
        with open(aln_faa, "w") as fh:
            run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", THREADS,
                 path_stem + ".faa"], stdout=fh)
    aligned_prot = read_fasta(aln_faa)
    out = {}
    for name, ap in aligned_prot.items():
        nt, i, buf = seqs[name], 0, []
        for ch in ap:
            if ch == "-":
                buf.append("---")
            else:
                buf.append(nt[i * 3:i * 3 + 3])
                i += 1
        out[name] = "".join(buf)
    return out


def trim(aln):
    n = len(aln)
    length = len(next(iter(aln.values())))
    keep = [i for i in range(length) if sum(s[i] == "-" for s in aln.values()) / n <= GAP_MAX]
    return {k: "".join(v[i] for i in keep) for k, v in aln.items()}, length, len(keep)


def build(name):
    folder = os.path.join(OUT, name)
    recs = list(SeqIO.parse(os.path.join(folder, "genomes.gb"), "genbank"))
    print(f"[{name}] {len(recs)} genomes", flush=True)
    summary = []

    for gene in ("orf1b", "spike"):
        seqs, missing = {}, []
        for rec in recs:
            nt = extract_gene(rec, gene)
            if nt is None:
                missing.append(rec.id)
                continue
            seqs[rec.id] = nt
        print(f"[{name}/{gene}] {len(seqs)} sequences, {len(missing)} without a usable annotation",
              flush=True)
        if len(seqs) < MIN_TAXA:
            summary.append(f"{gene}\tskipped: only {len(seqs)} annotated sequences")
            continue

        stem = os.path.join(folder, gene)
        aligned = codon_align(seqs, stem)
        write_fasta(stem + ".aln", aligned)
        trimmed, before, after = trim(aligned)
        write_fasta(stem + ".trim.aln", trimmed)
        summary.append(f"{gene}\t{len(seqs)} taxa\t{before} columns\t{after} kept "
                       f"(<= {int(GAP_MAX * 100)}% gaps)\tmissing: {','.join(missing) or 'none'}")

        if not os.path.exists(stem + ".treefile") or os.environ.get("CGA_REDO"):
            run(["iqtree2", "-s", stem + ".trim.aln", "-st", "DNA", "-m", "MFP",
                 "-B", "1000", "-alrt", "1000", "-T", THREADS, "-seed", "240", "-redo",
                 "--prefix", stem])

    with open(os.path.join(folder, "tree-summary.tsv"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"[{name}] " + " | ".join(summary), flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(os.listdir(OUT))
    for t in targets:
        if os.path.isdir(os.path.join(OUT, t)):
            build(t)
