#!/usr/bin/env python3
"""Compute composition features for the 65 genomes and write them to results/.

- overlapping k-mer counts, k = 1..6 (single strand, ambiguous k-mers skipped); longer words
  are recounted by the benchmark script
- counts as produced by the coursework notebook, k = 1..3
- codon frequency, RSCU and amino-acid frequency from annotated coding sequences
"""
import hashlib
import os

import numpy as np

import cga


def main():
    os.makedirs(cga.RESULTS, exist_ok=True)
    meta = cga.load_meta()
    accs = [m["accession"] for m in meta]
    recs = cga.load_records(accs)
    seqs = {a: cga.genome(recs[a]) for a in accs}

    out = {}
    for k in cga.KS_STORED:
        out[f"k{k}"] = np.array([cga.kmer_counts(seqs[a], k) for a in accs])
    for k in (1, 2, 3):
        out[f"orig{k}"] = np.array([cga.original_counts(seqs[a], k) for a in accs])
    feats = [cga.codon_features(recs[a]) for a in accs]
    out["codon"] = np.array([f[0] for f in feats])
    out["rscu"] = np.array([f[1] for f in feats])
    out["aa"] = np.array([f[2] for f in feats])
    np.savez_compressed(os.path.join(cga.RESULTS, "features.npz"), accessions=np.array(accs), **out)

    # Supplementary Table S1: genomes, taxonomy, host, base composition
    C1 = out["k1"]
    P = 100 * C1 / C1.sum(1, keepdims=True)
    with open(os.path.join(cga.RESULTS, "table-s1-genomes.tsv"), "w") as fh:
        fh.write("accession_version\tname\tshort_label\tlength_nt\tncbi_genus\tncbi_subgenus\t"
                 "host_qualifier\thost_group\tA_pct\tC_pct\tG_pct\tT_pct\tambiguous_nt\tcds_codons\n")
        for i, m in enumerate(meta):
            amb = m["length"] - int(C1[i].sum())
            ncod = sum(len(s) // 3 for s in cga.coding_sequences(recs[m["accession"]]))
            fh.write("\t".join([
                m["version"], m["organism"], m["short"], str(m["length"]), m["genus"] or "not assigned",
                m["subgenus"] or "not assigned", m["host_qualifier"] or "not given", m["host_group"],
                *(f"{x:.2f}" for x in P[i]), str(amb), str(ncod)]) + "\n")
    with open(os.path.join(cga.RESULTS, "sequence-checksums.tsv"), "w") as fh:
        fh.write("accession_version\tlength_nt\tsha256_of_sequence\n")
        for m in meta:
            s = seqs[m["accession"]]
            fh.write(f"{m['version']}\t{len(s)}\t{hashlib.sha256(s.encode()).hexdigest()}\n")
    print("genomes:", len(accs))
    print("ambiguous bases in total:", sum(m["length"] for m in meta) - int(C1.sum()))
    with open(os.path.join(cga.RESULTS, "word-occupancy.tsv"), "w") as fh:
        fh.write("k\tpossible_words\tmean_fraction_of_words_absent_per_genome\n")
        for k in cga.KS:
            Ck = out[f"k{k}"] if k in cga.KS_STORED else cga.count_matrix([seqs[a] for a in accs], k)
            fh.write(f"{k}\t{4 ** k}\t{(Ck == 0).mean():.4f}\n")
            print(f"k={k}: possible words {4 ** k}, absent per genome {(Ck == 0).mean():.1%}")
    for k in (2, 3):
        cov = out[f"orig{k}"].sum(1) / out[f"k{k}"].sum(1)
        print(f"k={k}: coursework counting captures {cov.min():.3f}-{cov.max():.3f} of k-mer positions "
              f"(mean {cov.mean():.3f})")


if __name__ == "__main__":
    main()
