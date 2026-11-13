#!/usr/bin/env python3
"""Write the manuscript tables as Markdown: results/tables-main.md (tables used in the text) and
results/tables-supplement.md (Supplementary Tables S1-S4, copied to docs/paper-supplementary-tables.md)."""
import csv
import json
import os

import cga

NAMES = {
    "freq-euclid": "Word frequency, Euclidean", "freq-cosine": "Word frequency, cosine",
    "zfreq-euclid": "Standardised word frequency, Euclidean", "d2star-M0": "d2\\*, base composition removed",
    "d2star-M1": "d2\\*, dinucleotide composition removed", "presence-jaccard": "Word presence, Jaccard",
    "karlin-delta": "Dinucleotide relative abundance, δ\\*", "original-pipeline": "Coursework pipeline as run",
    "cds-codon-freq": "Coding sequences: codon frequency", "cds-rscu": "Coding sequences: RSCU",
    "cds-aa-freq": "Coding sequences: amino-acid frequency", "genome-length": "Genome length only (control)",
    "reference: RdRp tree": "RdRp tree (reference)", "reference: ORF1b tree": "ORF1b tree (second alignment-based tree)",
}


FRONT = '---\ncursor:\n  subagentId: "bc-c92a5b80-1934-56da-ada2-ba3615e87d47"\n---\n\n'


def name(key):
    m, k = key.split("|")
    return NAMES[m], (k if k != "0" else "–")


def main():
    S = json.load(open(os.path.join(cga.RESULTS, "summary.json")))
    B = S["benchmark"]
    meta = cga.load_meta()
    out = []

    # ---- Table 1
    out.append("**Table 1. The 65 genomes by genus and by host group of the sequenced isolate.**\n")
    hosts = sorted(S["host_counts"], key=lambda h: -S["host_counts"][h])
    placed = S["unclassified_placement"]
    rows = {g: dict.fromkeys(hosts, 0) for g in cga.GENUS_ORDER}
    unassigned = dict.fromkeys(cga.GENUS_ORDER, 0)
    for m in meta:
        g = m["genus"] or placed[m["accession"]]["nearest_genus"]
        rows[g][m["host_group"]] += 1
        unassigned[g] += not m["genus"]
    out.append("| Genus | Genomes | " + " | ".join(f"{h} ({cga.HOST_COMMON[h]})" for h in hosts) + " |")
    out.append("|---|---|" + "---|" * len(hosts))
    for g in cga.GENUS_ORDER:
        tot = sum(rows[g].values())
        extra = f" ({unassigned[g]} placed by RdRp tree)" if unassigned[g] else ""
        out.append(f"| *{g}* | {tot}{extra} | " + " | ".join(str(rows[g][h]) for h in hosts) + " |")
    out.append(f"| Total | {len(meta)} | " + " | ".join(str(S["host_counts"][h]) for h in hosts) + " |\n")

    # ---- Table 3
    sel = ["original-pipeline|1", "original-pipeline|2", "original-pipeline|3",
           "freq-euclid|1", "freq-euclid|2", "freq-euclid|3", "karlin-delta|2", "d2star-M1|3",
           "freq-euclid|6", "d2star-M0|6", "d2star-M1|6",
           "freq-euclid|8", "d2star-M0|8", "d2star-M1|8",
           "freq-euclid|10", "freq-cosine|10", "presence-jaccard|10", "d2star-M1|10",
           "cds-codon-freq|0", "cds-rscu|0", "cds-aa-freq|0", "genome-length|0", "reference: ORF1b tree|0"]
    out.append("**Table 3. Agreement of composition-based comparison with the RdRp phylogeny (selected rows; "
               "all rows in Supplementary Table S2).**\n")
    out.append("| Measure | k | Spearman ρ | Nearest neighbour in same genus | Nearest neighbour in same subgenus | "
               "NJ tree: nRF | NJ: well-supported RdRp clades recovered | NJ: genera recovered (of 4) | "
               "NJ: subgenera recovered (of 16) | UPGMA: nRF |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sel:
        r = B[key]
        n, k = name(key)
        out.append(f"| {n} | {k} | {r['spearman']:.2f} | {r['nn_genus']:.2f} | {r['nn_subgenus']:.2f} | "
                   f"{r['nj_nRF']:.2f} | {r['nj_strong_recovered']:.2f} | {r['nj_genus_clans']} | "
                   f"{r['nj_subgenus_clans']} | {r['upgma_nRF']:.2f} |")
    out.append(f"\nChance level for the nearest-neighbour columns: {S['chance']['nn_genus']:.2f} (genus), "
               f"{S['chance']['nn_subgenus']:.2f} (subgenus).\n")

    # ---- Table 4 bootstrap calibration
    out.append("**Table 4. Block-bootstrap support of groupings in composition trees, and whether the grouping is "
               "in the RdRp tree.**\n")
    out.append("| Measure | Tree | Groupings | In RdRp tree | Support ≥ 70% | of which in RdRp tree | "
               "Support ≥ 95% | of which in RdRp tree |")
    out.append("|---|---|---|---|---|---|---|---|")
    for key, v in S["bootstrap"].items():
        n, k = name(key)
        for t in ("upgma", "nj"):
            b = v[t]
            out.append(f"| {n}, k = {k} | {t.upper()} | {b['splits']} | {b['in_reference']} | {b['support_ge70']} | "
                       f"{b['support_ge70_in_reference']} | {b['support_ge95']} | {b['support_ge95_in_reference']} |")
    out.append("")

    # ---- Table 5 panels
    out.append("**Table 5. Coursework panels: groupings shared with the RdRp tree restricted to the same genomes.**\n")
    best = S["best"]["tree"]
    bn, bk = name(best)
    out.append(f"| Panel | Groupings in RdRp tree | As run, k = 1 | As run, k = 2 | As run, k = 3 | "
               f"Frequency + UPGMA, k = 1 | k = 2 | k = 3 | {bn}, k = {bk}, UPGMA | same, NJ |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for pname, p in S["panels"].items():
        out.append(f"| {pname} | {p['reference_splits']} | " +
                   " | ".join(str(p[f'k{k}']['original_shared_with_reference']) for k in (1, 2, 3)) + " | " +
                   " | ".join(str(p[f'k{k}']['corrected_shared_with_reference']) for k in (1, 2, 3)) +
                   f" | {p[best + '|upgma']['shared_with_reference']} | {p[best + '|nj']['shared_with_reference']} |")
    out.append("")

    # ---- Table 6 cosine six
    c6 = S["cosine_six"]
    labs = [cga.SHORT[a] for a in c6["accessions"]]
    out.append("**Table 6. Cosine similarity of tri-nucleotide frequency vectors (standard overlapping counts) for "
               "the six RefSeq genomes of the coursework cosine table.**\n")
    out.append("| | " + " | ".join(labs) + " |")
    out.append("|---|" + "---|" * len(labs))
    for lab, row in zip(labs, c6["matrix"]):
        out.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    out.append("")
    open(os.path.join(cga.RESULTS, "tables-main.md"), "w").write(FRONT + "\n".join(out) + "\n")

    # ---------------------------------------------------------------- supplement
    sup = ["# Supplementary tables", "",
           "Supplement to `paper-draft.md`. Generated by `internal/analysis/06_tables.py` from the files in "
           "`internal/analysis/results/`. Do not edit by hand; rerun the script.", "",
           "## Supplementary Table S1. Genomes", "",
           f"RefSeq records retrieved {open(os.path.join(cga.DATA, 'fetch-date.txt')).read().strip()}. Genus and "
           "subgenus are from the NCBI Taxonomy lineage of each record. Host group is the order (mammals) or class "
           "(birds) of the host of the sequenced isolate, taken from the record's host qualifier or, where that is "
           "absent, from the virus name.", "",
           "| Accession | Name | Label in figures | Length (nt) | Genus | Subgenus | Host (record) | Host group | "
           "A % | C % | G % | U % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in csv.DictReader(open(os.path.join(cga.RESULTS, "table-s1-genomes.tsv")), delimiter="\t"):
        host = r["host_qualifier"].split(";")[0]
        sup.append(f"| {r['accession_version']} | {r['name']} | {r['short_label']} | {int(r['length_nt']):,} | "
                   f"{r['ncbi_genus']} | {r['ncbi_subgenus']} | {host} | {r['host_group']} | {r['A_pct']} | "
                   f"{r['C_pct']} | {r['G_pct']} | {r['T_pct']} |")
    sup += ["", "Genomes without a genus in NCBI Taxonomy, and where the RdRp tree places them:", "",
            "| Accession | Label | Closest classified genome in RdRp tree | Genus | Subgenus of that genome |",
            "|---|---|---|---|---|"]
    for a, v in placed.items():
        sup.append(f"| {a} | {cga.SHORT[a]} | {cga.SHORT[v['nearest_classified']]} | {v['nearest_genus']} | "
                   f"{v['nearest_subgenus']} |")

    sup += ["", "## Supplementary Table S2. Full benchmark", "",
            "ρ: Spearman correlation between the composition distance and the RdRp patristic distance over the "
            "2,080 genome pairs (Mantel permutation p = 0.001, the smallest attainable with 999 permutations, for "
            "every row except word presence at k = 5 and k = 6). ρ ORF1b: the same against the ORF1b tree. "
            "NN: fraction of genomes whose nearest neighbour is in the same genus, in the same subgenus, among the "
            "three closest relatives in the RdRp tree, or from the same host group. nRF: normalised "
            "Robinson-Foulds distance to the RdRp tree (and to the ORF1b tree). Clades: fraction of the "
            f"{S['reference']['strong_splits']} well-supported RdRp groupings present. G and S: genera (of 4) and "
            "subgenera (of 16) recovered as groups.", "",
            "| Measure | k | ρ | ρ ORF1b | NN genus | NN subgenus | NN top 3 | NN host | NJ nRF | NJ clades | NJ G | "
            "NJ S | NJ nRF ORF1b | UPGMA nRF | UPGMA clades | UPGMA G | UPGMA S | UPGMA nRF ORF1b |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["original-pipeline", "freq-euclid", "freq-cosine", "zfreq-euclid", "karlin-delta", "d2star-M0",
             "d2star-M1", "presence-jaccard", "cds-codon-freq", "cds-rscu", "cds-aa-freq", "genome-length",
             "reference: RdRp tree", "reference: ORF1b tree"]
    keys = sorted(B, key=lambda key: (order.index(key.split("|")[0]), int(key.split("|")[1])))
    for key in keys:
        r = B[key]
        n, k = name(key)
        sup.append(f"| {n} | {k} | {r['spearman']:.2f} | {r['spearman_orf1b']:.2f} | {r['nn_genus']:.2f} | "
                   f"{r['nn_subgenus']:.2f} | {r['nn_top3']:.2f} | {r['nn_host']:.2f} | {r['nj_nRF']:.2f} | "
                   f"{r['nj_strong_recovered']:.2f} | {r['nj_genus_clans']} | {r['nj_subgenus_clans']} | "
                   f"{r['nj_nRF_orf1b']:.2f} | {r['upgma_nRF']:.2f} | {r['upgma_strong_recovered']:.2f} | "
                   f"{r['upgma_genus_clans']} | {r['upgma_subgenus_clans']} | {r['upgma_nRF_orf1b']:.2f} |")
    sup += ["", "The coursework pipeline as run used single linkage. Its single-linkage dendrograms score nRF " +
            ", ".join(f"{B[f'original-pipeline|{k}']['single_nRF']:.2f} (k = {k})" for k in (1, 2, 3)) +
            " against the RdRp tree; the UPGMA and NJ columns above apply those tree-building rules to the same "
            "distances.", "",
            "## Supplementary Table S3. Closest genome to SARS-CoV-2 under each measure", "",
            "Rank 1 is the closest genome. SARS-CoV Tor2 and bat coronavirus BM48-31 are the two other "
            "sarbecoviruses in the data set.", "",
            "| Measure | k | Closest genome | Genus of closest genome | Rank of SARS-CoV Tor2 | Rank of BM48-31 | "
            "Sarbecovirus grouped (UPGMA) | Sarbecovirus grouped (NJ) |", "|---|---|---|---|---|---|---|---|"]
    G = S["named_groups"]
    for key in sorted(G, key=lambda key: (order.index(key.split("|")[0]), int(key.split("|")[1]))):
        g = G[key]
        n, k = name(key)
        yes = {True: "yes", False: "no"}
        sup.append(f"| {n} | {k} | {g['sars2_nearest']} | {g['sars2_nearest_genus']} | {g['rank_of_SARS_Tor2']} | "
                   f"{g['rank_of_BM48_31']} | {yes[g['sarbecovirus_clan_upgma']]} | {yes[g['sarbecovirus_clan_nj']]} |")
    occ = list(csv.DictReader(open(os.path.join(cga.RESULTS, "word-occupancy.tsv")), delimiter="\t"))
    sup += ["", "## Supplementary Table S4. Word occupancy", "",
            "| k | Possible words (4^k) | Mean fraction of possible words absent from a genome |", "|---|---|---|"]
    for r in occ:
        sup.append(f"| {r['k']} | {int(r['possible_words']):,} | {float(r['mean_fraction_of_words_absent_per_genome']):.3f} |")
    open(os.path.join(cga.RESULTS, "tables-supplement.md"), "w").write("\n".join(sup) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
