#!/usr/bin/env python3
"""Benchmark composition-based distances and trees against the RdRp reference phylogeny.

Reads results/features.npz, the GenBank records and data/reftree/*.treefile.
Writes results/benchmark.tsv, results/summary.json and several supporting tables.
"""
import itertools
import json
import os

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import cga

N_PERM = 999
N_BOOT = 1000
N_SHUFFLE = 20
STRONG = (80.0, 95.0)  # SH-aLRT and UFBoot thresholds for a well-supported reference split


def upper(M):
    return M[np.triu_indices_from(M, 1)]


def spearman_mantel(D, P, rng, n_perm=N_PERM):
    """Spearman correlation of two distance matrices with a Mantel permutation p-value."""
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
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        hits += corr(rd[np.ix_(perm, perm)]) >= obs
    return obs, (hits + 1) / (n_perm + 1)


def nn_index(D, rows, candidates):
    """Nearest neighbour (among candidates, excluding self) for each row index."""
    out = []
    for i in rows:
        cand = [j for j in candidates if j != i]
        out.append(cand[int(np.argmin(D[i, cand]))])
    return out


def nn_accuracy(D, labels, rows, candidates):
    nn = nn_index(D, rows, candidates)
    return float(np.mean([labels[i] == labels[j] for i, j in zip(rows, nn)]))


def top_relatives(P, m=3):
    """For each genome, the set of its m closest relatives by patristic distance (ties included)."""
    out = []
    for i in range(len(P)):
        d = np.delete(P[i], i)
        thr = np.sort(d)[m - 1]
        out.append({j for j in range(len(P)) if j != i and P[i, j] <= thr + 1e-12})
    return out


def restrict(splits, subset, ref):
    subset = frozenset(subset)
    out = set()
    for s in splits:
        a = s & subset
        if 2 <= len(a) <= len(subset) - 2:
            out.add(cga.canon(a, subset, ref))
    return out


def main():
    def stream(tag):
        """Independent random stream per analysis, so that adding one analysis does not change another."""
        return np.random.default_rng([cga.SEED, sum(ord(c) * (i + 1) for i, c in enumerate(tag))])

    rng = stream("mantel")
    meta = cga.load_meta()
    accs = [m["accession"] for m in meta]
    n = len(accs)
    ix = {a: i for i, a in enumerate(accs)}
    taxa = frozenset(accs)
    genus = [m["genus"] for m in meta]
    subgenus = [m["subgenus"] for m in meta]
    host = [m["host_group"] for m in meta]
    feats = np.load(os.path.join(cga.RESULTS, "features.npz"))
    assert list(feats["accessions"]) == accs
    recs = cga.load_records(accs)
    seqs = [cga.genome(recs[a]) for a in accs]
    C = {k: feats[f"k{k}"] for k in cga.KS_STORED}

    # ------------------------------------------------------------------ reference trees
    ref_tree, ref_sup, _ = cga.read_newick_splits(os.path.join(cga.DATA, "reftree", "rdrp.treefile"))
    ref_sup = {cga.canon(s, taxa, accs[0]): v for s, v in ref_sup.items()}
    ref_splits = set(ref_sup)
    ref_strong = {s for s, (alrt, ufb) in ref_sup.items() if alrt >= STRONG[0] and ufb >= STRONG[1]}
    P = cga.patristic(ref_tree, accs)
    alt_tree, alt_sup, _ = cga.read_newick_splits(os.path.join(cga.DATA, "reftree", "orf1b.treefile"))
    alt_sup = {cga.canon(s, taxa, accs[0]): v for s, v in alt_sup.items()}
    alt_splits = set(alt_sup)
    P_alt = cga.patristic(alt_tree, accs)

    lab_g = [i for i in range(n) if genus[i]]
    sub_sizes = {s: subgenus.count(s) for s in set(subgenus) if s}
    lab_s = [i for i in range(n) if subgenus[i]]
    rows_s = [i for i in lab_s if sub_sizes[subgenus[i]] >= 2]
    genus_groups = {g: frozenset(accs[i] for i in lab_g if genus[i] == g) for g in cga.GENUS_ORDER}
    sub_groups = {s: frozenset(accs[i] for i in lab_s if subgenus[i] == s)
                  for s, c in sorted(sub_sizes.items()) if c >= 2}
    L_g = frozenset(accs[i] for i in lab_g)
    L_s = frozenset(accs[i] for i in lab_s)
    relatives = top_relatives(P)

    summary = {"n_genomes": n, "n_genus_labelled": len(lab_g), "n_subgenus_labelled": len(lab_s),
               "n_subgenus_scored_nn": len(rows_s), "subgenera_multi": {s: len(g) for s, g in sub_groups.items()},
               "genus_counts": {g: len(m) for g, m in genus_groups.items()},
               "host_counts": {h: host.count(h) for h in sorted(set(host))}}

    def tree_scores(splits):
        return {
            "nRF": cga.rf_normalised(splits, ref_splits),
            "nRF_orf1b": cga.rf_normalised(splits, alt_splits),
            "strong_recovered": len(ref_strong & splits) / len(ref_strong),
            "genus_clans": sum(cga.is_clan(g, splits, L_g) for g in genus_groups.values()),
            "subgenus_clans": sum(cga.is_clan(g, splits, L_s) for g in sub_groups.values()),
        }

    ref_valid = tree_scores(ref_splits)
    summary["reference"] = {
        "splits": len(ref_splits), "strong_splits": len(ref_strong),
        "genus_clans": ref_valid["genus_clans"], "subgenus_clans": ref_valid["subgenus_clans"],
        "n_subgenera_multi": len(sub_groups),
        "nRF_rdrp_vs_orf1b": cga.rf_normalised(ref_splits, alt_splits),
        "strong_rdrp_splits_in_orf1b": len(ref_strong & alt_splits) / len(ref_strong),
        "spearman_patristic_rdrp_vs_orf1b": spearman_mantel(P_alt, P, rng, 0)[0],
        "orf1b_genus_clans": sum(cga.is_clan(g, alt_splits, L_g) for g in genus_groups.values()),
        "orf1b_subgenus_clans": sum(cga.is_clan(g, alt_splits, L_s) for g in sub_groups.values()),
    }
    # placement of the genomes without a genus in NCBI Taxonomy
    placement = {}
    for i in range(n):
        if not genus[i]:
            j = nn_index(P, [i], lab_g)[0]
            g = genus[j]
            joined = cga.is_clan(genus_groups[g] | {accs[i]}, ref_splits, L_g | {accs[i]})
            placement[accs[i]] = {"nearest_classified": accs[j], "nearest_genus": g,
                                  "nearest_subgenus": subgenus[j], "inside_genus_clan": bool(joined)}
    summary["unclassified_placement"] = placement
    tree_genus = [genus[i] or placement[accs[i]]["nearest_genus"] for i in range(n)]

    # ------------------------------------------------------------------ distances
    D = {}
    for k in cga.KS:
        Ck = C[k] if k in C else cga.count_matrix(seqs, k)
        for m, Dm in cga.kmer_distances(k, Ck, C[1], C[2]).items():
            D[m, k] = Dm
        del Ck
    for k in (1, 2, 3):
        D["original-pipeline", k] = cga.euclid(cga.zscore(feats[f"orig{k}"]))
    D["cds-codon-freq", 0] = cga.euclid(feats["codon"])
    D["cds-rscu", 0] = cga.euclid(feats["rscu"])
    D["cds-aa-freq", 0] = cga.euclid(feats["aa"])
    lengths = np.array([m["length"] for m in meta], float)
    D["genome-length", 0] = np.abs(lengths[:, None] - lengths[None, :])
    np.savez_compressed(os.path.join(cga.RESULTS, "distances.npz"), accessions=np.array(accs),
                        patristic_rdrp=P, patristic_orf1b=P_alt,
                        **{f"{m}|{k}": v for (m, k), v in D.items()})

    # chance level of a nearest-neighbour score: probability that a randomly chosen other
    # candidate carries the same label, averaged over the scored genomes
    def chance(labels, rows, cands):
        return float(np.mean([np.mean([labels[i] == labels[j] for j in cands if j != i]) for i in rows]))

    summary["chance"] = {"nn_genus": chance(genus, lab_g, lab_g), "nn_subgenus": chance(subgenus, rows_s, lab_s),
                         "nn_host": chance(host, list(range(n)), list(range(n)))}

    def all_scores(Dm, with_trees=True, single=False):
        rho, p = spearman_mantel(Dm, P, rng)
        nn_all = nn_index(Dm, range(n), range(n))
        row = {
            "spearman": rho, "mantel_p": p, "spearman_orf1b": spearman_mantel(Dm, P_alt, rng, 0)[0],
            "nn_genus": nn_accuracy(Dm, genus, lab_g, lab_g),
            "nn_subgenus": nn_accuracy(Dm, subgenus, rows_s, lab_s),
            "nn_top3": float(np.mean([j in relatives[i] for i, j in enumerate(nn_all)])),
            "nn_host": float(np.mean([host[i] == host[j] for i, j in enumerate(nn_all)])),
        }
        if with_trees:
            for name, splits in (("upgma", cga.upgma_splits(Dm, accs)), ("nj", cga.nj_splits(Dm, accs))):
                for key, val in tree_scores(splits).items():
                    row[f"{name}_{key}"] = val
            if single:
                for key, val in tree_scores(cga.upgma_splits(Dm, accs, "single")).items():
                    row[f"single_{key}"] = val
        return row

    rows = {}
    for (m, k), Dm in D.items():
        rows[m, k] = all_scores(Dm, single=(m == "original-pipeline"))
    rows["reference: RdRp tree", 0] = {
        "spearman": 1.0, "mantel_p": float("nan"), "spearman_orf1b": summary["reference"]["spearman_patristic_rdrp_vs_orf1b"],
        "nn_genus": nn_accuracy(P, genus, lab_g, lab_g), "nn_subgenus": nn_accuracy(P, subgenus, rows_s, lab_s),
        "nn_top3": 1.0, "nn_host": float(np.mean([host[i] == host[j] for i, j in
                                                   enumerate(nn_index(P, range(n), range(n)))])),
        **{f"{t}_{k}": v for t in ("upgma", "nj") for k, v in
           {"nRF": 0.0, "nRF_orf1b": summary["reference"]["nRF_rdrp_vs_orf1b"], "strong_recovered": 1.0,
            "genus_clans": ref_valid["genus_clans"],
            "subgenus_clans": ref_valid["subgenus_clans"]}.items()},
    }
    alt = all_scores(P_alt, with_trees=False)
    alt.update({f"{t}_{k}": v for t in ("upgma", "nj") for k, v in
                {"nRF": summary["reference"]["nRF_rdrp_vs_orf1b"], "nRF_orf1b": 0.0,
                 "strong_recovered": summary["reference"]["strong_rdrp_splits_in_orf1b"],
                 "genus_clans": summary["reference"]["orf1b_genus_clans"],
                 "subgenus_clans": summary["reference"]["orf1b_subgenus_clans"]}.items()})
    rows["reference: ORF1b tree", 0] = alt

    cols = ["spearman", "mantel_p", "spearman_orf1b", "nn_genus", "nn_subgenus", "nn_top3", "nn_host",
            "nj_nRF", "nj_strong_recovered", "nj_genus_clans", "nj_subgenus_clans", "nj_nRF_orf1b",
            "upgma_nRF", "upgma_strong_recovered", "upgma_genus_clans", "upgma_subgenus_clans", "upgma_nRF_orf1b",
            "single_nRF", "single_strong_recovered", "single_genus_clans", "single_subgenus_clans"]
    with open(os.path.join(cga.RESULTS, "benchmark.tsv"), "w") as fh:
        fh.write("method\tk\t" + "\t".join(cols) + "\n")
        for (m, k), r in rows.items():
            fh.write(f"{m}\t{k or ''}\t" + "\t".join(
                "" if c not in r else f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    summary["benchmark"] = {f"{m}|{k}": r for (m, k), r in rows.items()}

    # ------------------------------------------------------------------ how much is base composition?
    rng = stream("shuffle")
    comp = {}
    d1 = D["freq-euclid", 1]
    for k in cga.KS[1:]:
        rho = spearman_mantel(D["freq-euclid", k], d1, rng, 0)[0]
        De = cga.euclid(cga.expected_m0(C[1], k))
        r_exp = float(np.corrcoef(upper(D["freq-euclid", k]), upper(De))[0, 1])
        comp[k] = {"spearman_with_k1": rho, "r2_with_M0_expected": r_exp ** 2}
        del De
    summary["base_composition_share"] = comp

    # cosine similarity scale for tri-nucleotide frequencies
    F3 = cga.freqs(C[3])
    S3 = 1 - cga.cosine_dist(F3)
    E3 = cga.expected_m0(C[1], 3)
    self_expected = np.array([float(F3[i] @ E3[i] / np.linalg.norm(F3[i]) / np.linalg.norm(E3[i])) for i in range(n)])
    shuffled = []
    for a in accs:
        s = np.array(list(cga.genome(recs[a])))
        f = cga.freqs(cga.kmer_counts("".join(s), 3)[None, :])[0]
        for _ in range(N_SHUFFLE):
            g = cga.freqs(cga.kmer_counts("".join(rng.permutation(s)), 3)[None, :])[0]
            shuffled.append(float(f @ g / np.linalg.norm(f) / np.linalg.norm(g)))
    shuffled = np.array(shuffled)
    pairs = upper(S3)
    same_genus = np.array([tree_genus[i] == tree_genus[j] for i, j in itertools.combinations(range(n), 2)])
    summary["cosine_tri"] = {
        "pairs": int(len(pairs)), "min": float(pairs.min()), "median": float(np.median(pairs)),
        "max": float(pairs.max()),
        "within_genus_median": float(np.median(pairs[same_genus])),
        "between_genus_median": float(np.median(pairs[~same_genus])),
        "within_genus_range": [float(pairs[same_genus].min()), float(pairs[same_genus].max())],
        "between_genus_range": [float(pairs[~same_genus].min()), float(pairs[~same_genus].max())],
        "shuffle_mean": float(shuffled.mean()), "shuffle_min": float(shuffled.min()),
        "shuffle_max": float(shuffled.max()), "shuffle_n": int(len(shuffled)),
        "self_vs_expected_mean": float(self_expected.mean()),
        "self_vs_expected_range": [float(self_expected.min()), float(self_expected.max())],
        "pairs_at_or_below_shuffle_max": float((pairs <= shuffled.max()).mean()),
        "pairs_at_or_below_shuffle_mean": float((pairs <= shuffled.mean()).mean()),
        "between_genus_pairs_above_within_median": float((pairs[~same_genus] > np.median(pairs[same_genus])).mean()),
    }
    np.save(os.path.join(cga.RESULTS, "cosine-shuffle-baseline.npy"), shuffled)

    # ------------------------------------------------------------------ block bootstrap of UPGMA trees
    scored = [mk for mk in D if mk[0] in ("freq-euclid", "freq-cosine", "d2star-M0", "d2star-M1", "presence-jaccard")]
    best_freq = max((mk for mk in scored if mk[0] == "freq-euclid"), key=lambda mk: rows[mk]["spearman"])
    best_d2 = max((mk for mk in scored if mk[0].startswith("d2star")), key=lambda mk: rows[mk]["spearman"])
    # lowest NJ distance to the reference tree; ties go to the shorter word
    best_tree = min(scored, key=lambda mk: (round(rows[mk]["nj_nRF"], 6), mk[1]))
    targets = list(dict.fromkeys([("freq-euclid", 3), best_tree]))
    summary["bootstrap_targets"] = [f"{m}|{k}" for m, k in targets]
    block_cache = {}
    rng = stream("bootstrap")

    def blocks(i, k):
        if (i, k) not in block_cache:
            block_cache[i, k] = cga.block_kmer_indices(seqs[i], k)
        return block_cache[i, k]

    def resampled_counts(idx, need):
        Cb = {kk: np.zeros((len(idx), 4 ** kk)) for kk in need}
        for r, i in enumerate(idx):
            nb = len(blocks(i, 1))
            pick = rng.integers(0, nb, nb)
            for kk in need:
                v = blocks(i, kk)[pick].ravel()
                Cb[kk][r] = np.bincount(v[v >= 0], minlength=4 ** kk)
        return Cb

    def bootstrap(subset, method, k, n_boot=N_BOOT):
        """Block bootstrap: genome blocks of cga.BLOCK nt are resampled with replacement within each
        genome, counts are rebuilt, and the UPGMA and NJ trees are re-inferred."""
        subset = list(subset)
        idx = [ix[a] for a in subset]
        need = sorted({1, 2, k})
        Ck = C[k][idx] if k in C else cga.count_matrix([seqs[i] for i in idx], k)
        full = cga.distance_from_counts(method, k, Ck, C[1][idx], C[2][idx])
        Z = linkage(squareform(full, checks=False), "average")
        up, nj = cga.linkage_splits(Z, subset), cga.nj_splits(full, subset)
        cu, cn = dict.fromkeys(up, 0), dict.fromkeys(nj, 0)
        for _ in range(n_boot):
            Cb = resampled_counts(idx, need)
            Db = cga.distance_from_counts(method, k, Cb[k], Cb[1], Cb[2])
            for s in up & cga.upgma_splits(Db, subset):
                cu[s] += 1
            for s in nj & cga.nj_splits(Db, subset):
                cn[s] += 1
        return Z, {s: c / n_boot for s, c in cu.items()}, {s: c / n_boot for s, c in cn.items()}

    def support_table(sup, reference):
        vals = np.array(list(sup.values()))
        ok = np.array([s in reference for s in sup])
        out = {"splits": int(len(vals)), "in_reference": int(ok.sum())}
        for thr in (70, 95):
            hi = vals >= thr / 100
            out[f"support_ge{thr}"] = int(hi.sum())
            out[f"support_ge{thr}_in_reference"] = int((hi & ok).sum())
            out[f"support_ge{thr}_not_in_reference"] = int((hi & ~ok).sum())
        return out

    boot = {}
    boot_trees = {}
    for m, k in targets:
        Z, sup_u, sup_n = bootstrap(accs, m, k)
        boot_trees[f"{m}|{k}"] = (Z, sup_u, sup_n)
        boot[f"{m}|{k}"] = {"upgma": support_table(sup_u, ref_splits), "nj": support_table(sup_n, ref_splits)}
        for name, sup in (("upgma", sup_u), ("nj", sup_n)):
            with open(os.path.join(cga.RESULTS, f"bootstrap-{m}-k{k}-{name}.tsv"), "w") as fh:
                fh.write("support\tin_rdrp_tree\tsize\tmembers\n")
                for s, v in sorted(sup.items(), key=lambda kv: -kv[1]):
                    fh.write(f"{v:.3f}\t{int(s in ref_splits)}\t{len(s)}\t"
                             f"{', '.join(sorted(cga.SHORT[a] for a in s))}\n")
        block_cache.clear()
    summary["bootstrap"] = boot
    np.savez(os.path.join(cga.RESULTS, "bootstrap-trees.npz"),
             **{f"{key}|Z": Z for key, (Z, _, _) in boot_trees.items()})
    json.dump({key: {"upgma": [[sorted(s), v] for s, v in su.items()], "nj": [[sorted(s), v] for s, v in sn.items()]}
               for key, (_, su, sn) in boot_trees.items()},
              open(os.path.join(cga.RESULTS, "bootstrap-supports.json"), "w"))

    # ------------------------------------------------------------------ named groups
    embeco4 = frozenset(["NC_006213", "NC_003045", "NC_017083", "NC_046954"])
    sarbe = frozenset(i for i in accs if subgenus[ix[i]] == "Sarbecovirus")
    groups = {}
    cov2 = ix["NC_045512"]

    def neighbours(Dm, splits_u, splits_n):
        order = [j for j in np.argsort(Dm[cov2]) if j != cov2]
        return {
            "sars2_nearest": cga.SHORT[accs[order[0]]], "sars2_nearest_genus": tree_genus[order[0]],
            "rank_of_SARS_Tor2": order.index(ix["NC_004718"]) + 1, "rank_of_BM48_31": order.index(ix["NC_014470"]) + 1,
            "sarbecovirus_clan_upgma": bool(cga.is_clan(sarbe, splits_u, taxa)),
            "sarbecovirus_clan_nj": bool(cga.is_clan(sarbe, splits_n, taxa)),
            "four_genome_group_clan_upgma": bool(cga.is_clan(embeco4, splits_u, taxa)),
            "four_genome_group_clan_nj": bool(cga.is_clan(embeco4, splits_n, taxa)),
        }

    for (m, k), Dm in D.items():
        groups[f"{m}|{k}"] = neighbours(Dm, cga.upgma_splits(Dm, accs), cga.nj_splits(Dm, accs))
    groups["reference: RdRp tree|0"] = neighbours(P, ref_splits, ref_splits)
    groups["reference: ORF1b tree|0"] = neighbours(P_alt, alt_splits, alt_splits)
    summary["named_groups"] = groups
    e4 = [ix[a] for a in sorted(embeco4)]
    key4_all = cga.canon(embeco4, taxa, accs[0])
    trio = cga.canon(embeco4 - {"NC_046954"}, taxa, accs[0])
    summary["four_genome_group"] = {
        "members": [cga.SHORT[accs[i]] for i in e4],
        "rdrp_patristic_max_within": float(P[np.ix_(e4, e4)].max()),
        "rdrp_patristic_median_all_pairs": float(np.median(upper(P))),
        "clade_in_rdrp_tree": key4_all in ref_splits, "clade_in_orf1b_tree": key4_all in alt_splits,
        "support_in_orf1b_tree_alrt_ufboot": alt_sup.get(key4_all),
        "trio_without_rodent_support_rdrp": ref_sup.get(trio), "trio_without_rodent_support_orf1b": alt_sup.get(trio),
        "bootstrap_upgma": {key: sup.get(key4_all) for key, (_, sup, _) in boot_trees.items()},
        "bootstrap_nj": {key: sup.get(key4_all) for key, (_, _, sup) in boot_trees.items()},
        "methods_recovering_clan_upgma": sum(g["four_genome_group_clan_upgma"] for key, g in groups.items()
                                             if not key.startswith(("reference", "genome-length"))),
        "methods_total": sum(1 for key in groups if not key.startswith(("reference", "genome-length"))),
    }

    # ------------------------------------------------------------------ coursework panels
    panels = {"multi-host (12)": cga.PANEL12, "bat (13)": cga.PANEL_BAT, "human (4)": cga.PANEL_HUMAN,
              "bat + human (17)": cga.PANEL_BAT + cga.PANEL_HUMAN}
    panel_out = {}
    for pname, panel in panels.items():
        idx = [ix[a] for a in panel]
        refp = restrict(ref_splits, panel, panel[0])
        res = {"reference_splits": len(refp)}
        for k in (1, 2, 3):
            # as run in the coursework: non-overlapping wrapped raw counts, z-scored within the panel
            Xo = cga.zscore(feats[f"orig{k}"][idx])
            so = cga.linkage_splits(linkage(Xo, "single"), panel)
            # corrected: overlapping counts as frequencies, Euclidean, UPGMA
            sc = cga.upgma_splits(cga.euclid(cga.freqs(C[k][idx])), panel)
            res[f"k{k}"] = {
                "original_shared_with_reference": len(so & refp), "original_splits": len(so),
                "corrected_shared_with_reference": len(sc & refp), "corrected_splits": len(sc),
                "original_vs_corrected_shared": len(so & sc),
            }
        m, k = best_tree
        Ck = C[k][idx] if k in C else cga.count_matrix([seqs[i] for i in idx], k)
        Dp = cga.distance_from_counts(m, k, Ck, C[1][idx], C[2][idx])
        for name, sp in (("upgma", cga.upgma_splits(Dp, panel)), ("nj", cga.nj_splits(Dp, panel))):
            res[f"{m}|{k}|{name}"] = {"shared_with_reference": len(sp & refp), "splits": len(sp)}
        panel_out[pname] = res
    summary["panels"] = panel_out
    p12 = {}
    p12_supports = {}
    key4 = cga.canon(embeco4, frozenset(cga.PANEL12), cga.PANEL12[0])
    refp12 = restrict(ref_splits, cga.PANEL12, cga.PANEL12[0])
    for m, k in list(dict.fromkeys([("freq-euclid", 1), ("freq-euclid", 2), ("freq-euclid", 3), best_tree])):
        Z, sup_u, sup_n = bootstrap(cga.PANEL12, m, k)
        p12[f"{m}|{k}"] = {
            "four_genome_group_support_upgma": sup_u.get(key4), "four_genome_group_support_nj": sup_n.get(key4),
            "upgma": support_table(sup_u, refp12), "nj": support_table(sup_n, refp12),
            "upgma_splits": [[sorted(cga.SHORT[a] for a in s), v, s in refp12] for s, v in
                             sorted(sup_u.items(), key=lambda kv: -kv[1])]}
        p12_supports[f"{m}|{k}"] = {"upgma": [[sorted(s), v] for s, v in sup_u.items()],
                                    "nj": [[sorted(s), v] for s, v in sup_n.items()]}
        np.save(os.path.join(cga.RESULTS, f"panel12-{m}-k{k}-linkage.npy"), Z)
        block_cache.clear()
    summary["panel12_bootstrap"] = p12
    json.dump(p12_supports, open(os.path.join(cga.RESULTS, "panel12-supports.json"), "w"))

    # cosine similarity for the six-genome coursework table, standard counts
    six = [ix[a] for a in cga.PANEL_COSINE]
    summary["cosine_six"] = {"accessions": cga.PANEL_COSINE,
                             "matrix": np.round(S3[np.ix_(six, six)], 4).tolist()}

    # ------------------------------------------------------------------ host signal beyond relatedness
    rng = stream("host")
    host_out = {}
    iu = np.triu_indices(n, 1)
    same_host = np.array([host[i] == host[j] for i, j in zip(*iu)])
    same_g = np.array([tree_genus[i] == tree_genus[j] for i, j in zip(*iu)])
    from scipy.optimize import isotonic_regression
    sel = same_g
    x = P[iu][sel]
    order = np.argsort(x, kind="stable")
    sh_obs = same_host[sel]
    harr = np.array(host, dtype=object)
    garr = np.array(tree_genus, dtype=object)
    pair_i, pair_j = iu[0][sel], iu[1][sel]
    for m, k in list(dict.fromkeys([("freq-euclid", 3), best_freq, best_d2, best_tree])):
        # within-genus pairs only; composition distance is regressed on RdRp patristic distance with a
        # monotone (isotonic) fit and the residuals of same-host and different-host pairs are compared
        y = D[m, k][iu][sel]
        fit = np.empty_like(y)
        fit[order] = isotonic_regression(y[order]).x
        resid = (y - fit) / y.std()
        obs = resid[sh_obs].mean() - resid[~sh_obs].mean()
        null = []
        for _ in range(N_PERM):
            perm = harr.copy()
            for g in set(tree_genus):
                w = np.where(garr == g)[0]
                perm[w] = rng.permutation(harr[w])
            sh = perm[pair_i] == perm[pair_j]
            null.append(resid[sh].mean() - resid[~sh].mean())
        null = np.array(null)
        host_out[f"{m}|{k}"] = {
            "residual_difference_same_minus_different_host_sd_units": float(obs),
            "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)),
            "within_genus_pairs": int(sel.sum()), "same_host_pairs": int(sh_obs.sum()),
        }
    summary["host_residual_test"] = host_out
    summary["best"] = {"freq": f"{best_freq[0]}|{best_freq[1]}", "d2": f"{best_d2[0]}|{best_d2[1]}",
                       "tree": f"{best_tree[0]}|{best_tree[1]}"}

    json.dump(summary, open(os.path.join(cga.RESULTS, "summary.json"), "w"), indent=1, default=float)
    print("written", os.path.join(cga.RESULTS, "summary.json"))


if __name__ == "__main__":
    main()
