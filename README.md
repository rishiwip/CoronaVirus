# Benchmarking Alignment-Free k-mer Distances Against an RdRp Phylogeny of Coronaviridae

Reproducible analysis for the manuscript targeting MDPI *Viruses*:

**Benchmarking Alignment-Free k-mer Distances Against an RdRp Phylogeny of Coronaviridae**

This repository contains the Python pipeline that recomputes every table and figure in the paper from public NCBI sequences. **No NCBI API key is required or included.**

## Requirements

- Python 3.10+ with packages in `requirements.txt`
- [MAFFT](https://mafft.cbrc.jp/alignment/software/) and [IQ-TREE 2](http://www.iqtree.org/) on `PATH`

```bash
pip install -r requirements.txt
```

## Run order (family-level benchmark)

```bash
python3 01_fetch.py            # ~1 min   65 GenBank records → data/genbank/, data/records.tsv
python3 02_reference_tree.py   # ~25 min  MAFFT + IQ-TREE → data/reftree/
python3 03_features.py         # ~5 s     k-mer / codon / amino-acid features
python3 04_benchmark.py        # ~2.5 min scoring, bootstraps, host test → results/summary.json
python3 05_figures.py          # ~10 s    figures/
python3 06_tables.py           # ~1 s     results/tables-*.md
```

## Shallow-divergence extension (within-subgenus sets)

```bash
python3 11_subgenus_datasets.py   # build four GenBank sets (no API key; polite E-utilities)
python3 12_subgenus_trees.py      # ORF1b / S codon alignments + IQ-TREE
python3 13_subgenus_benchmark.py  # same composition scores inside each subgenus
python3 14_floors.py              # disjoint-data floors (family + each subgenus)
python3 15_subgenus_figures.py    # Figure 7 / Table 7
```

`cga.py` holds shared feature extraction, distances, neighbour joining, and metadata.

All randomness is seeded from `cga.SEED`. `04_benchmark.py` uses an independent RNG stream per analysis so adding one analysis does not change another.

## What is and is not stored here

**Included:** scripts; `data/records.tsv` and fetch date; reference and subgenus tree files (`.treefile`, `.iqtree`); selection tables and filter logs; result tables and JSON summaries.

**Not included (regenerable, large):** raw GenBank downloads (`data/genbank/`, `data/subgenus/*/downloaded.gb`), alignments, and NumPy feature matrices. Re-run `01`–`04` and `11`–`14` to recreate them. Accessions for every analysed genome are pinned in `data/records.tsv` and `data/subgenus/*/selection.tsv`.

## NCBI access

Fetches use NCBI E-utilities **without an API key**. A placeholder contact email (`noreply@example.org`) is sent only as the E-utilities `email` parameter required by NCBI’s usage guidelines. Replace it with your own contact address in `11_subgenus_datasets.py` if you re-run large downloads. Do not commit API keys.

## Citation

Manuscript (preprint / submission draft): see the project documentation. Prefer citing the published version once available, and this repository’s commit or release tag.

## License

Code is provided for reproducibility of the associated manuscript. Sequence data remain under NCBI’s terms.
