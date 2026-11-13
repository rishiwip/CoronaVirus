"""Run the composition benchmark inside each subgenus, where the genomes are closely related.

Identical in method to 04_benchmark.py; only the dataset and the reference tree change. Within a
subgenus every genome carries the same taxonomy labels, so the genus/subgenus recovery columns are
dropped and the scoring is: correlation with patristic distance, tree distance to the reference,
and recovery of its well-supported clades.

Divergence is measured as the mean pairwise amino-acid p-distance over the ORF1b protein alignment,
the same quantity for every dataset including the family-level one, so the results can be placed on
a common divergence axis.

Writes results/subgenus-benchmark.tsv and results/subgenus-summary.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

import cga

SUB = os.path.join(cga.DATA, "subgenus")
N_PERM = 999
STRONG = (80.0, 95.0)


def upper(M):
    return M[np.triu_indices_from(M, 1)]


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


def mean_p_distance(aln):
    """Mean pairwise p-distance over columns where both sequences have a residue."""
    names = sorted(aln)
    arr = np.array([list(aln[n]) for n in names])
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ok = (arr[i] != "-") & (arr[j] != "-")
            if ok.sum():
                vals.append(float((arr[i][ok] != arr[j][ok]).mean()))
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def spearman_mantel(D, P, rng, n_perm=N_PERM):
    n = len(D)
    iu = np.triu_indices(n, 1)
    rd = np.zeros((n, n))
    rd[iu] = rankdata(D[iu])
    rd = rd + rd.T
    rp = rankdata(P[iu])
    rp = (rp - rp.mean()) / rp.std()

    def corr(M):
        x = M[iu]
        return float(((x - x.mean()) / x.std() * rp).mean())

    obs = corr(rd)
    if not n_perm:
        return obs, float("nan")
    hits = sum(corr(rd[np.ix_(rng.permutation(n), rng.permutation(n))]) >= obs for _ in range(n_perm))
    return obs, (hits + 1) / (n_perm + 1)


def tree_from(path, taxa, ref):
    tree, sup, _ = cga.read_newick_splits(path)
    sup = {cga.canon(s, taxa, ref): v for s, v in sup.items()}
    return tree, sup


def nn_index(D):
    return [int(np.argmin(np.where(np.arange(len(D)) == i, np.inf, D[i]))) for i in range(len(D))]


def top_relatives(P, m=3):
    out = []
    for i in range(len(P)):
        thr = np.sort(np.delete(P[i], i))[m - 1]
        out.append({j for j in range(len(P)) if j != i and P[i, j] <= thr + 1e-12})
    return out


def benchmark(name, rng):
    folder = os.path.join(SUB, name)
    ref_path = os.path.join(folder, "orf1b.treefile")
    alt_path = os.path.join(folder, "spike.treefile")
    if not os.path.exists(ref_path):
        print(f"[{name}] no ORF1b tree, skipped")
        return None

    from Bio import SeqIO
    recs = {r.id: r for r in SeqIO.parse(os.path.join(folder, "genomes.gb"), "genbank")}
    tree_taxa = sorted(read_fasta(os.path.join(folder, "orf1b.trim.aln")))
    accs = [a for a in sorted(recs) if a in tree_taxa]
    n = len(accs)
    taxa = frozenset(accs)
    seqs = [cga.genome(recs[a]) for a in accs]
    print(f"[{name}] {n} genomes", flush=True)

    ref_tree, ref_sup = tree_from(ref_path, taxa, accs[0])
    ref_splits = set(ref_sup)
    ref_strong = {s for s, (alrt, ufb) in ref_sup.items() if alrt >= STRONG[0] and ufb >= STRONG[1]}
    P = cga.patristic(ref_tree, accs)

    alt_splits, P_alt, alt_taxa = None, None, None
    if os.path.exists(alt_path):
        alt_names = sorted(read_fasta(os.path.join(folder, "spike.trim.aln")))
        alt_taxa = [a for a in accs if a in alt_names]
        if len(alt_taxa) >= 4:
            at = frozenset(alt_taxa)
            tree_s, sup_s = tree_from(alt_path, at, alt_taxa[0])
            alt_splits = set(sup_s)
            P_alt = cga.patristic(tree_s, alt_taxa)

    def restrict(splits, subset):
        subset = frozenset(subset)
        out = set()
        for s in splits:
            a = s & subset
            if 2 <= len(a) <= len(subset) - 2:
                out.add(cga.canon(a, subset, sorted(subset)[0]))
        return out

    floor = {}
    if alt_splits is not None:
        ref_on_alt = restrict(ref_splits, alt_taxa)
        strong_on_alt = restrict(ref_strong, alt_taxa)
        floor = {
            "spike_taxa": len(alt_taxa),
            "nRF_orf1b_vs_spike": cga.rf_normalised(ref_on_alt, alt_splits),
            "strong_orf1b_splits_in_spike": (len(strong_on_alt & alt_splits) / len(strong_on_alt)
                                             if strong_on_alt else float("nan")),
            "spearman_patristic_orf1b_vs_spike": spearman_mantel(
                P_alt, cga.patristic(ref_tree, alt_taxa), rng, 0)[0],
        }

    C1 = cga.count_matrix(seqs, 1)
    C2 = cga.count_matrix(seqs, 2)
    relatives = top_relatives(P)

    rows = {}
    for k in cga.KS:
        Ck = C1 if k == 1 else (C2 if k == 2 else cga.count_matrix(seqs, k))
        for m, Dm in cga.kmer_distances(k, Ck, C1, C2).items():
            nn = nn_index(Dm)
            rho, p = spearman_mantel(Dm, P, rng)
            row = {"spearman": rho, "mantel_p": p,
                   "nn_top3": float(np.mean([j in relatives[i] for i, j in enumerate(nn)]))}
            for tname, splits in (("nj", cga.nj_splits(Dm, accs)), ("upgma", cga.upgma_splits(Dm, accs))):
                row[f"{tname}_nRF"] = cga.rf_normalised(splits, ref_splits)
                row[f"{tname}_strong_recovered"] = (len(ref_strong & splits) / len(ref_strong)
                                                    if ref_strong else float("nan"))
                if alt_splits is not None:
                    row[f"{tname}_nRF_spike"] = cga.rf_normalised(restrict(splits, alt_taxa), alt_splits)
            rows[m, k] = row
        del Ck
        print(f"[{name}]   k={k} done", flush=True)

    aln = read_fasta(os.path.join(folder, "orf1b.prot.aln"))
    aln = {k: v for k, v in aln.items() if k in taxa}
    div_mean, div_min, div_max = mean_p_distance(aln)

    out = {
        "n_genomes": n,
        "reference_splits": len(ref_splits),
        "reference_strong_splits": len(ref_strong),
        "divergence_orf1b_aa_mean": div_mean,
        "divergence_orf1b_aa_min": div_min,
        "divergence_orf1b_aa_max": div_max,
        "patristic_median": float(np.median(upper(P))),
        "floor": floor,
        "benchmark": {f"{m}|{k}": r for (m, k), r in rows.items()},
    }
    return out


def family_divergence():
    """The same divergence measure for the 65-genome family-level dataset."""
    aln = read_fasta(os.path.join(cga.DATA, "reftree", "orf1b.trim.aln"))
    mean, lo, hi = mean_p_distance(aln)
    import csv
    bench = {f"{r['method']}|{r['k'] or 0}": r for r in
             csv.DictReader(open(os.path.join(cga.RESULTS, "benchmark.tsv")), delimiter="\t")}
    return {"n_genomes": len(aln), "divergence_orf1b_aa_mean": mean,
            "divergence_orf1b_aa_min": lo, "divergence_orf1b_aa_max": hi,
            "benchmark": {key: {"spearman": float(r["spearman"]), "nj_nRF": float(r["nj_nRF"]),
                                "upgma_nRF": float(r["upgma_nRF"]),
                                "nj_strong_recovered": float(r["nj_strong_recovered"])}
                          for key, r in bench.items() if not key.startswith("reference")}}


if __name__ == "__main__":
    rng = np.random.default_rng(cga.SEED)
    out = {"family (65 reference genomes)": family_divergence()}
    for name in sorted(os.listdir(SUB)):
        if os.path.isdir(os.path.join(SUB, name)):
            res = benchmark(name, rng)
            if res:
                out[name] = res

    cols = ["spearman", "mantel_p", "nn_top3", "nj_nRF", "nj_strong_recovered", "nj_nRF_spike",
            "upgma_nRF", "upgma_strong_recovered", "upgma_nRF_spike"]
    with open(os.path.join(cga.RESULTS, "subgenus-benchmark.tsv"), "w") as fh:
        fh.write("dataset\tmethod\tk\t" + "\t".join(cols) + "\n")
        for dataset, res in out.items():
            if dataset.startswith("family"):
                continue
            for key, r in res["benchmark"].items():
                m, k = key.rsplit("|", 1)
                fh.write(f"{dataset}\t{m}\t{k}\t" + "\t".join(
                    f"{r[c]:.4f}" if c in r else "" for c in cols) + "\n")
    json.dump(out, open(os.path.join(cga.RESULTS, "subgenus-summary.json"), "w"), indent=1, default=float)
    print("written", os.path.join(cga.RESULTS, "subgenus-summary.json"))
