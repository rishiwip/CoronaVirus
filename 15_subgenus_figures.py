"""Figure and tables for the within-subgenus (shallow divergence) analysis."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import cga

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

FAMILY = "family (65 reference genomes)"
ORDER = [FAMILY, "Merbecovirus", "Embecovirus", "Sarbecovirus", "Pedacovirus"]
LABEL = {FAMILY: "Coronaviridae (65)", "Merbecovirus": "Merbecovirus (35)",
         "Embecovirus": "Embecovirus (31)", "Sarbecovirus": "Sarbecovirus (35)",
         "Pedacovirus": "Pedacovirus (32)"}
COLOR = {FAMILY: "#222222", "Merbecovirus": "#0072B2", "Embecovirus": "#009E73",
         "Sarbecovirus": "#D55E00", "Pedacovirus": "#CC79A7"}
PRIMARY = "d2star-M1"


def load():
    S = json.load(open(os.path.join(cga.RESULTS, "subgenus-summary.json")))
    F = json.load(open(os.path.join(cga.DATA, "floors.json")))
    floors = {FAMILY: F["family"]["nRF"]}
    for name in ORDER[1:]:
        floors[name] = F[name]["nRF_half1_vs_half2"]
    return S, F, floors


def main():
    S, F, floors = load()
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ks = list(cga.KS)

    # (a) tree distance against word length, with each dataset's own floor
    for name in ORDER:
        B = S[name]["benchmark"]
        y = [B[f"{PRIMARY}|{k}"]["nj_nRF"] for k in ks if f"{PRIMARY}|{k}" in B]
        xs = [k for k in ks if f"{PRIMARY}|{k}" in B]
        ax_a.plot(xs, y, "-o", ms=3, color=COLOR[name], label=LABEL[name])
        ax_a.axhline(floors[name], color=COLOR[name], ls=":", lw=1, alpha=0.8)
    ax_a.set_xlabel("word length $k$")
    ax_a.set_ylabel("tree distance to reference (nRF)")
    ax_a.set_title("(a) agreement against word length", loc="left", fontsize=8)
    ax_a.set_xticks(ks[2:])
    ax_a.set_xlim(2.7, 10.3)
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax_a.text(10.2, 0.99, "dotted lines: each dataset's floor",
              ha="right", va="top", fontsize=6, color="#555555")

    # (b) well-supported clade recovery
    for name in ORDER:
        B = S[name]["benchmark"]
        xs = [k for k in ks if f"{PRIMARY}|{k}" in B]
        y = [B[f"{PRIMARY}|{k}"]["nj_strong_recovered"] for k in xs]
        ax_b.plot(xs, y, "-o", ms=3, color=COLOR[name])
    ax_b.set_xlabel("word length $k$")
    ax_b.set_ylabel("well-supported reference clades recovered")
    ax_b.set_title("(b) recovery of supported clades", loc="left", fontsize=8)
    ax_b.set_xticks(ks[2:])
    ax_b.set_xlim(2.7, 10.3)
    ax_b.set_ylim(0, 1.02)

    # (c) composition against the floor: on the line means as good as the data allow
    for name in ORDER:
        B = S[name]["benchmark"]
        ax_c.scatter(floors[name], B["freq-euclid|3"]["nj_nRF"], marker="^", s=38,
                     color=COLOR[name], zorder=3)
        ax_c.scatter(floors[name], B[f"{PRIMARY}|8"]["nj_nRF"], marker="o", s=38,
                     color=COLOR[name], zorder=3)
    lim = [0, 1.0]
    ax_c.plot(lim, lim, color="#888888", lw=1, ls="--", zorder=1)
    ax_c.set_xlim(0, 1.0)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_xlabel("floor: two halves of the same gene (nRF)")
    ax_c.set_ylabel("composition tree to reference (nRF)")
    ax_c.set_title("(c) composition against the floor", loc="left", fontsize=8)
    ax_c.legend(handles=[Line2D([], [], marker="^", ls="", color="#444444", label="$k$ = 3, frequency"),
                         Line2D([], [], marker="o", ls="", color="#444444", label="$k$ = 8, d2*"),
                         Line2D([], [], ls="--", color="#888888", label="equal to the floor")],
                frameon=False, fontsize=6.5, loc="lower right")

    # (d) whether a tree is determined at all, against how much the genomes differ
    for name in ORDER:
        x = S[name]["divergence_orf1b_aa_mean"] * 100
        ax_d.scatter(x, floors[name], s=46, color=COLOR[name], zorder=3)
        ax_d.annotate(LABEL[name].split(" (")[0].replace("Coronaviridae", "family (65)"),
                      (x, floors[name]), textcoords="offset points",
                      xytext=(7, {"Sarbecovirus": -12, "Embecovirus": -13,
                                   "Merbecovirus": 5, FAMILY: 6, "Pedacovirus": 4}[name]),
                      fontsize=6.5,
                      color=COLOR[name])
    ax_d.set_xlabel("mean ORF1b divergence (% amino acid)")
    ax_d.set_ylabel("floor: two halves of the same gene (nRF)")
    ax_d.set_title("(d) when is a tree determined at all?", loc="left", fontsize=8)
    ax_d.set_xlim(0, 46)
    ax_d.set_ylim(0, 1.0)
    ax_d.axhspan(0, 0.4, color="#2ca02c", alpha=0.06)
    ax_d.text(44, 0.05, "tree well determined", ha="right", fontsize=6, color="#4a7a4a")
    ax_d.text(44, 0.93, "no single tree is recoverable", ha="right", fontsize=6, color="#8a5a5a")

    os.makedirs(cga.FIGS, exist_ok=True)
    fig.savefig(os.path.join(cga.FIGS, "fig7-shallow-divergence.png"))
    plt.close(fig)
    print("wrote fig7-shallow-divergence.png")

    # ------------------------------------------------------------------ tables
    lines = ["**Table 7. Composition against an alignment-based reference inside four subgenera, "
             "where the genomes are closely related.** Tree distance is the normalised Robinson-Foulds "
             "distance of the neighbour-joining tree to the ORF1b reference tree. The floor is the "
             "distance between trees built from the two halves of that same alignment: it is how far "
             "apart two alignment-based trees of the same genomes are when neither has the whole gene.",
             "",
             "| Dataset | Genomes | Mean ORF1b divergence | Floor | k = 3 frequency | k = 8 d2\\* | "
             "Supported clades at k = 8 | ORF1b vs S |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        v = S[name]
        B = v["benchmark"]
        sp = v.get("floor", {}).get("nRF_orf1b_vs_spike")
        lines.append(
            f"| {LABEL[name]} | {v['n_genomes']} | {v['divergence_orf1b_aa_mean'] * 100:.1f}% | "
            f"{floors[name]:.2f} | {B['freq-euclid|3']['nj_nRF']:.2f} | {B[f'{PRIMARY}|8']['nj_nRF']:.2f} | "
            f"{B[f'{PRIMARY}|8']['nj_strong_recovered']:.0%} | "
            f"{'—' if sp is None else f'{sp:.2f}'} |")

    lines += ["", "**Table S5. All composition measures inside each subgenus.** "
                  "Full table in `results/subgenus-benchmark.tsv`.", ""]
    open(os.path.join(cga.RESULTS, "tables-shallow.md"), "w").write("\n".join(lines) + "\n")
    print("wrote tables-shallow.md")

    out = {"floors": floors,
           "datasets": {n: {"n_genomes": S[n]["n_genomes"],
                            "divergence": S[n]["divergence_orf1b_aa_mean"],
                            "floor": floors[n],
                            "k3_nRF": S[n]["benchmark"]["freq-euclid|3"]["nj_nRF"],
                            "k8_nRF": S[n]["benchmark"][f"{PRIMARY}|8"]["nj_nRF"],
                            "k8_strong": S[n]["benchmark"][f"{PRIMARY}|8"]["nj_strong_recovered"],
                            "orf1b_vs_spike": S[n].get("floor", {}).get("nRF_orf1b_vs_spike")}
                        for n in ORDER}}
    json.dump(out, open(os.path.join(cga.RESULTS, "shallow-summary.json"), "w"), indent=1)
    print("wrote shallow-summary.json")


if __name__ == "__main__":
    main()
