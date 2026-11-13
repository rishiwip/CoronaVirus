#!/usr/bin/env python3
"""Draw all manuscript figures from results/ and data/reftree/. Output: PNG files in CGA_FIGS."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import cga

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
INK = "#222222"


def save(fig, name):
    os.makedirs(cga.FIGS, exist_ok=True)
    fig.savefig(os.path.join(cga.FIGS, name))
    plt.close(fig)
    print("wrote", name)


def genus_legend(ax, loc="lower right", extra=()):
    handles = [Line2D([], [], marker="s", ls="", color=cga.GENUS_COLOR[g], label=g) for g in cga.GENUS_ORDER]
    ax.legend(handles=handles + list(extra), loc=loc, frameon=False, fontsize=7)


# --------------------------------------------------------------------------- tree drawing

def draw_phylogram(ax, tree, label, color, node_mark=None, scale_bar=None, unit="substitutions per site"):
    """Rectangular phylogram of a rooted dendropy tree.
    node_mark(node) -> None, or (filled, text) to draw a dot at an internal node."""
    y = {}
    for i, leaf in enumerate(tree.leaf_node_iter()):
        y[leaf] = i
    for nd in tree.postorder_node_iter():
        if not nd.is_leaf():
            y[nd] = np.mean([y[c] for c in nd.child_node_iter()])
    x = {}
    for nd in tree.preorder_node_iter():
        x[nd] = 0.0 if nd.parent_node is None else x[nd.parent_node] + (nd.edge_length or 0.0)
    width = max(x.values())
    for nd in tree.preorder_node_iter():
        if nd.parent_node is not None:
            ax.plot([x[nd.parent_node], x[nd]], [y[nd], y[nd]], color=INK, lw=0.7)
        if not nd.is_leaf():
            ys = [y[c] for c in nd.child_node_iter()]
            ax.plot([x[nd], x[nd]], [min(ys), max(ys)], color=INK, lw=0.7)
            mark = node_mark(nd) if node_mark and nd.parent_node is not None else None
            if mark:
                filled, text = mark
                ax.plot(x[nd], y[nd], "o", ms=3.0, mfc=INK if filled else "white", mec=INK, mew=0.6, zorder=3)
                if text:
                    ax.text(x[nd] - width * 0.006, y[nd] - 0.12, text, fontsize=5, ha="right", va="bottom",
                            color="#555555")
        else:
            ax.text(x[nd], y[nd], " " + label(nd.taxon.label), va="center", ha="left",
                    color=color(nd.taxon.label), fontsize=6.3)
    ax.set_ylim(max(y.values()) + 1.4, -1)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    if scale_bar:
        y0 = max(y.values()) + 1.0
        ax.plot([0, scale_bar], [y0, y0], color=INK, lw=1)
        ax.text(scale_bar * 1.08, y0, f"{scale_bar:g} {unit}", ha="left", va="center", fontsize=6)
    return width


def rooted(newick_or_tree, keep=None):
    import dendropy
    tree = (dendropy.Tree.get(data=newick_or_tree, schema="newick", preserve_underscores=True)
            if isinstance(newick_or_tree, str) else newick_or_tree.clone())
    if keep is not None:
        tree.retain_taxa_with_labels(keep)
    tree.reroot_at_midpoint(update_bipartitions=False)
    tree.ladderize(ascending=True)
    return tree


def leafset(nd):
    return frozenset(l.taxon.label for l in nd.leaf_iter())


def draw_linkage(ax, Z, names, label, color, node_mark=None, xlabel="merge distance"):
    """Dendrogram of a scipy linkage with the root at the left. node_mark(members) -> (filled, text)."""
    n = len(names)
    order = list(leaves_list(Z))
    ypos = {leaf: i for i, leaf in enumerate(order)}
    height = {i: 0.0 for i in range(n)}
    members = {i: frozenset([names[i]]) for i in range(n)}
    top = Z[-1, 2]
    for i, (a, b, h, _) in enumerate(Z):
        a, b, node = int(a), int(b), n + i
        ypos[node] = (ypos[a] + ypos[b]) / 2
        height[node] = h
        members[node] = members[a] | members[b]
        for c in (a, b):
            ax.plot([h, height[c]], [ypos[c], ypos[c]], color=INK, lw=0.7)
        ax.plot([h, h], [ypos[a], ypos[b]], color=INK, lw=0.7)
        if node_mark and 2 <= len(members[node]) <= n - 2:
            filled, text = node_mark(members[node])
            ax.plot(h, ypos[node], "o", ms=3.2, mfc=INK if filled else "white", mec=INK, mew=0.6, zorder=3)
            if text:
                ax.text(h + top * 0.012, ypos[node] - 0.12, text, fontsize=5, ha="right", va="bottom", color="#555555")
    for leaf in range(n):
        ax.text(-top * 0.01, ypos[leaf], label(names[leaf]) + " ", va="center", ha="left", fontsize=6.3,
                color=color(names[leaf]))
    ax.set_xlim(top * 1.02, -top * 0.01)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)


def main():
    meta = cga.load_meta()
    accs = [m["accession"] for m in meta]
    ix = {a: i for i, a in enumerate(accs)}
    taxa = frozenset(accs)
    summary = json.load(open(os.path.join(cga.RESULTS, "summary.json")))
    feats = np.load(os.path.join(cga.RESULTS, "features.npz"))
    dist = np.load(os.path.join(cga.RESULTS, "distances.npz"))
    C = {k: feats[f"k{k}"] for k in cga.KS_STORED}
    recs = cga.load_records(accs)
    seqs = [cga.genome(recs[a]) for a in accs]

    def counts(k, idx=None):
        idx = range(len(accs)) if idx is None else idx
        return C[k][list(idx)] if k in C else cga.count_matrix([seqs[i] for i in idx], k)
    P = dist["patristic_rdrp"]
    placed = summary["unclassified_placement"]
    genus = {m["accession"]: m["genus"] or placed[m["accession"]]["nearest_genus"] for m in meta}
    unassigned = {m["accession"] for m in meta if not m["genus"]}
    subgenus = {m["accession"]: m["subgenus"] for m in meta}

    def color(a):
        return cga.GENUS_COLOR[genus[a]]

    def label(a):
        return cga.SHORT[a] + ("*" if a in unassigned else "")

    def label_sub(a):
        return f"{cga.SHORT[a]}{'*' if a in unassigned else ''}  [{subgenus[a] or 'subgenus not assigned'}]"

    ref_tree, ref_sup, _ = cga.read_newick_splits(os.path.join(cga.DATA, "reftree", "rdrp.treefile"))
    ref_splits = {cga.canon(s, taxa, accs[0]) for s in ref_sup}

    # ---------------------------------------------------------------- Figure 1: base composition
    F1 = 100 * cga.freqs(C[1])  # columns A, C, G, T
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    lengths = np.array([m["length"] for m in meta]) / 1000
    note = {"NC_045512": "SARS-CoV-2", "NC_004718": "SARS-CoV Tor2", "NC_006577": "HCoV-HKU1",
            "NC_002645": "HCoV-229E", "NC_005831": "HCoV-NL63", "NC_019843": "MERS-CoV",
            "NC_035191": "Wencheng shrew CoV (two genomes)"}
    four = [ix[a] for a in ("NC_006213", "NC_003045", "NC_017083", "NC_046954")]
    for ax, xv, xl, txt in ((axes[0], F1[:, 3], "U (T) content, %", (30.3, 12.6)),
                            (axes[1], lengths, "genome length, kb", (27.9, 11.9))):
        ax.annotate("HCoV-OC43, bovine CoV,\nrabbit CoV HKU14, rodent CoV", (xv[four].mean(), F1[four, 1].mean()),
                    xytext=txt, fontsize=6.2, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
        for i, a in enumerate(accs):
            ax.plot(xv[i], F1[i, 1], "o", ms=4.5, mfc=color(a) if a not in unassigned else "white",
                    mec=color(a), mew=1.0, alpha=0.9)
            if a in note:
                left = a == "NC_035191" and ax is axes[0]
                ax.annotate(note[a], (xv[i], F1[i, 1]), xytext=(-5, -9) if left else (4, 3),
                            textcoords="offset points", fontsize=6.2, ha="right" if left else "left")
        ax.set_xlabel(xl)
        ax.set_ylabel("C content, %")
    genus_legend(axes[0], "upper right", [Line2D([], [], marker="o", ls="", mfc="white", mec="#666666",
                                                  label="genus not assigned in NCBI Taxonomy")])
    axes[0].set_title("a", loc="left", fontweight="bold")
    axes[1].set_title("b", loc="left", fontweight="bold")
    save(fig, "fig1-base-composition-65-genomes.png")

    # ---------------------------------------------------------------- Figure 2: reference tree
    def strong(nd):
        if not nd.label or "/" not in nd.label:
            return None
        alrt, ufb = (float(v) for v in nd.label.split("/"))
        return (True, "") if alrt >= 80 and ufb >= 95 else None

    fig, ax = plt.subplots(figsize=(7.4, 10.2))
    w = draw_phylogram(ax, rooted(ref_tree), label_sub, color, strong, scale_bar=0.2)
    handles = [Line2D([], [], marker="s", ls="", color=cga.GENUS_COLOR[g], label=g) for g in cga.GENUS_ORDER]
    handles.append(Line2D([], [], marker="o", ls="", color=INK, ms=3, label="SH-aLRT ≥ 80 and UFBoot ≥ 95"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False, fontsize=7)
    ax.set_xlim(0, w * 1.6)
    save(fig, "fig2-rdrp-reference-tree.png")

    # ---------------------------------------------------------------- Figure 3: benchmark
    bench = summary["benchmark"]
    fam = {"freq-euclid": ("word frequency, Euclidean", "#0072B2", "o", "-"),
           "freq-cosine": ("word frequency, cosine", "#56B4E9", "s", "--"),
           "d2star-M0": ("d2*, base composition removed", "#E69F00", "D", "-"),
           "d2star-M1": ("d2*, dinucleotide composition removed", "#D55E00", "v", "-"),
           "presence-jaccard": ("word presence, Jaccard", "#009E73", "^", ":")}
    floor = summary["reference"]["nRF_rdrp_vs_orf1b"]
    panels = [("spearman", "Spearman correlation with\nRdRp patristic distance", None),
              ("nn_subgenus", "nearest neighbour in the\nsame subgenus (fraction of genomes)", summary["chance"]["nn_subgenus"]),
              ("nj_nRF", "normalised Robinson-Foulds distance,\nneighbour-joining tree v. RdRp tree", floor),
              ("upgma_nRF", "normalised Robinson-Foulds distance,\nUPGMA dendrogram v. RdRp tree", floor)]
    cds_x = {"cds-codon-freq|0": 11.3, "cds-rscu|0": 11.9, "cds-aa-freq|0": 12.5}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True)
    for ax, (key, yl, ref_line), tag in zip(axes.flat, panels, "abcd"):
        ax.axvspan(0.6, 3.4, color="#f0f0f0", zorder=0)
        for m, (lab, col, mk, ls) in fam.items():
            ks = [k for k in cga.KS if f"{m}|{k}" in bench]
            ax.plot(ks, [bench[f"{m}|{k}"][key] for k in ks], marker=mk, ls=ls, color=col, ms=4, lw=1.2, label=lab)
        okey = key.replace("upgma", "single") if key.startswith("upgma") else key
        ax.plot([1, 2, 3], [bench[f"original-pipeline|{k}"][okey] for k in (1, 2, 3)], marker="x", ls="",
                color="#000000", ms=6, label="wrapped-count baseline")
        ax.plot([2], [bench["karlin-delta|2"][key]], marker="*", ls="", color="#CC79A7", ms=8,
                label="dinucleotide relative abundance, δ*")
        for (m, xx), lab, col in zip(cds_x.items(), ("codon frequency", "RSCU", "amino-acid frequency"),
                                     ("#882255", "#AA4499", "#332288")):
            ax.plot([xx], [bench[m][key]], marker="P", ls="", color=col, ms=5.5, label=f"coding sequences: {lab}")
        if ref_line is not None:
            ax.axhline(ref_line, color="#888888", lw=0.8, ls="--")
            ax.text(10.5, ref_line + 0.012, "chance" if key == "nn_subgenus" else "RdRp tree v. ORF1b tree",
                    fontsize=6, color="#666666", ha="right")
        ax.set_ylabel(yl)
        ax.set_ylim(-0.08 if key == "spearman" else 0, 1.02)
        ax.set_title(tag, loc="left", fontweight="bold")
        ax.set_xticks(list(cga.KS) + [11.9])
        ax.set_xticklabels([str(k) for k in cga.KS] + ["coding\nsequences"])
    for ax in axes[1]:
        ax.set_xlabel("word length k")
    axes[0, 0].text(2, 0.97, "short-word\nregime", ha="center", va="top", fontsize=6.5, color="#666666")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    save(fig, "fig3-benchmark-against-rdrp-tree.png")

    # ---------------------------------------------------------------- Figure 4: cosine scale
    S3 = 1 - cga.cosine_dist(cga.freqs(C[3]))
    iu = np.triu_indices(len(accs), 1)
    same = np.array([genus[accs[i]] == genus[accs[j]] for i, j in zip(*iu)])
    shuffled = np.load(os.path.join(cga.RESULTS, "cosine-shuffle-baseline.npy"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(min(S3[iu].min(), shuffled.min()) - 0.002, 1.0, 46)
    ax.hist(S3[iu][~same], bins=bins, color="#999999", alpha=0.75, density=True,
            label=f"genomes from different genera ({int((~same).sum())} pairs)")
    ax.hist(S3[iu][same], bins=bins, color="#0072B2", alpha=0.65, density=True,
            label=f"genomes from the same genus ({int(same.sum())} pairs)")
    ax.hist(shuffled, bins=bins, histtype="step", color="#D55E00", lw=1.4, density=True,
            label=f"genome v. its own shuffled sequence ({len(shuffled)} shuffles)")
    ax.set_xlabel("cosine similarity of tri-nucleotide frequency vectors")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig4-cosine-similarity-baseline.png")

    # ---------------------------------------------------------------- Figure 5: composition trees, all genomes
    boots = np.load(os.path.join(cga.RESULTS, "bootstrap-trees.npz"))
    sups = json.load(open(os.path.join(cga.RESULTS, "bootstrap-supports.json")))
    titles = {"freq-euclid": "word frequency, Euclidean", "freq-cosine": "word frequency, cosine",
              "d2star-M0": "d2*, base composition removed", "d2star-M1": "d2*, dinucleotide composition removed",
              "presence-jaccard": "word presence, Jaccard"}
    best = summary["best"]["tree"]
    bm, bk = best.split("|")
    bk = int(bk)

    def marker(sup, reference, universe, ref_taxon, show_from=0.7):
        sup = {cga.canon(frozenset(s), universe, ref_taxon): v for s, v in sup} if sup is not None else None

        def f(members):
            s = cga.canon(frozenset(members), universe, ref_taxon)
            v = sup.get(s) if sup is not None else None
            return s in reference, (f"{round(100 * v)}" if v is not None and v >= show_from else "")
        return f

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 10.6))
    draw_linkage(axes[0], boots["freq-euclid|3|Z"], accs, label, color,
                 marker(sups["freq-euclid|3"]["upgma"], ref_splits, taxa, accs[0]), xlabel="UPGMA merge distance")
    axes[0].set_title("a   Tri-nucleotide frequency, Euclidean distance, UPGMA", loc="left", fontweight="bold", fontsize=8)
    Dbest = cga.distance_from_counts(bm, bk, counts(bk), C[1], C[2])
    njtree = rooted(cga.nj(Dbest, accs)[1])
    mk = marker(sups[best]["nj"], ref_splits, taxa, accs[0])
    w = draw_phylogram(axes[1], njtree, label, color, lambda nd: mk(leafset(nd)), scale_bar=0.05, unit="d2* units")
    axes[1].set_xlim(0, w * 1.45)
    axes[1].set_title(f"b   {titles[bm]}, k = {bk}, neighbour joining", loc="left", fontweight="bold", fontsize=8)
    handles = [Line2D([], [], marker="s", ls="", color=cga.GENUS_COLOR[g], label=g) for g in cga.GENUS_ORDER]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4, label="grouping also in RdRp tree"),
                Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4, label="grouping not in RdRp tree")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, "fig5-composition-trees-65-genomes.png")

    # ---------------------------------------------------------------- Figure 6: the 12-genome coursework panel
    panel = cga.PANEL12
    pidx = [ix[a] for a in panel]
    ptaxa = frozenset(panel)
    refp = set()
    for s in ref_splits:
        a = s & ptaxa
        if 2 <= len(a) <= len(panel) - 2:
            refp.add(cga.canon(a, ptaxa, panel[0]))
    psup = json.load(open(os.path.join(cga.RESULTS, "panel12-supports.json")))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    Zo = linkage(cga.zscore(feats["orig3"][pidx]), "single")
    draw_linkage(axes[0, 0], Zo, panel, label, color, marker(None, refp, ptaxa, panel[0]),
                 xlabel="single-linkage merge distance")
    axes[0, 0].set_title("a   Wrapped-count baseline, tri-nucleotide counts", loc="left", fontweight="bold", fontsize=8)
    Z = np.load(os.path.join(cga.RESULTS, "panel12-freq-euclid-k3-linkage.npy"))
    draw_linkage(axes[0, 1], Z, panel, label, color, marker(psup["freq-euclid|3"]["upgma"], refp, ptaxa, panel[0], 0),
                 xlabel="UPGMA merge distance")
    axes[0, 1].set_title("b   Tri-nucleotide frequency, Euclidean distance, UPGMA", loc="left", fontweight="bold", fontsize=8)
    Dp = cga.distance_from_counts(bm, bk, counts(bk, pidx), C[1][pidx], C[2][pidx])
    mk = marker(psup[best]["nj"], refp, ptaxa, panel[0], 0)
    w = draw_phylogram(axes[1, 0], rooted(cga.nj(Dp, panel)[1]), label, color, lambda nd: mk(leafset(nd)),
                       scale_bar=0.05, unit="d2* units")
    axes[1, 0].set_xlim(0, w * 1.5)
    axes[1, 0].set_title(f"c   {titles[bm]}, k = {bk}, neighbour joining", loc="left", fontweight="bold", fontsize=8)
    w = draw_phylogram(axes[1, 1], rooted(ref_tree, panel), label, color, strong, scale_bar=0.2)
    axes[1, 1].set_xlim(0, w * 1.6)
    axes[1, 1].set_title("d   RdRp maximum-likelihood tree, same 12 genomes", loc="left", fontweight="bold", fontsize=8)
    genus_legend(axes[1, 1], "lower right")
    fig.tight_layout()
    save(fig, "fig6-twelve-genome-panel.png")

    # ---------------------------------------------------------------- Figure S1: distance v. patristic distance
    keys = list(dict.fromkeys(["freq-euclid|1", "freq-euclid|3", "d2star-M1|3", best]))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.2))
    for ax, key in zip(np.atleast_1d(axes), keys):
        m, k = key.split("|")
        D = dist[key]
        ax.scatter(P[iu][~same], D[iu][~same], s=3, color="#999999", alpha=0.5, lw=0, label="different genera")
        ax.scatter(P[iu][same], D[iu][same], s=3, color="#0072B2", alpha=0.6, lw=0, label="same genus")
        ax.set_xlabel("RdRp patristic distance")
        ax.set_ylabel("composition distance")
        ax.set_title(f"{titles[m]}\nk = {k}; Spearman {bench[key]['spearman']:.2f}", fontsize=7)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5, markerscale=3)
    fig.tight_layout()
    save(fig, "figS1-composition-vs-patristic-distance.png")

    # ---------------------------------------------------------------- Figure S2: host
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    items = [("RdRp tree (closest relative)", bench["reference: RdRp tree|0"]["nn_host"], "#222222")]
    for key in ["freq-euclid|1", "freq-euclid|3", "karlin-delta|2", "d2star-M1|3", "freq-cosine|10",
                "presence-jaccard|10", best, "cds-rscu|0", "cds-aa-freq|0"]:
        m, k = key.split("|")
        name = {"karlin-delta": "dinucleotide relative abundance, δ*", "cds-rscu": "coding sequences: RSCU",
                "cds-aa-freq": "coding sequences: amino-acid frequency"}.get(m, f"{titles.get(m, m)}, k = {k}")
        if (name, bench[key]["nn_host"], "#0072B2") not in items:
            items.append((name, bench[key]["nn_host"], "#0072B2"))
    ax.barh(range(len(items)), [v for _, v, _ in items], color=[c for *_, c in items], height=0.6)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, *_ in items], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(summary["chance"]["nn_host"], color="#D55E00", lw=1, ls="--")
    ax.text(summary["chance"]["nn_host"] + 0.01, len(items) - 0.45, "chance", color="#D55E00", fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of genomes whose nearest neighbour was sampled from the same host group")
    save(fig, "figS2-host-group-nearest-neighbour.png")


if __name__ == "__main__":
    main()
