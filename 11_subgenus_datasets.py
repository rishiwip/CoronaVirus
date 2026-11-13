"""Build shallow-divergence genome sets: one per well-sampled subgenus.

For each target subgenus: list complete genomes at NCBI, download them, drop records with
too many ambiguous bases, collapse near-identical records, then subsample to a fixed size
spread across the remaining diversity.

Writes data/subgenus/<name>/genomes.gb and data/subgenus/<name>/selection.tsv.
"""
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from Bio import SeqIO

import cga

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUT = os.path.join(cga.DATA, "subgenus")
TOOL = {"tool": "coronavirus-composition-benchmark", "email": "noreply@example.org"}

# Subgenus, length window for a complete genome, extra query terms, and accessions added by hand.
#
# SARS-CoV-2 (taxid 2697049) is excluded from the Sarbecovirus search: 9.2 million of the 9.17
# million records in that window are SARS-CoV-2, so an unfiltered sample would be a study of one
# virus. It re-enters as its reference genome, one representative like every other virus here.
TARGETS = [
    ("Sarbecovirus", 29000, 30500, "NOT txid2697049[Organism:exp]", ["NC_045512.2"]),
    ("Embecovirus", 30000, 31700, "", []),
    ("Pedacovirus", 27000, 29000, "", []),
    ("Merbecovirus", 29000, 31000, "", []),
]

MAX_AMBIGUOUS = 0.01   # fraction of non-ACGT bases tolerated
DEDUP_IDENTITY = 0.995  # collapse records at or above this k-mer identity
TARGET_N = 35
MAX_DOWNLOAD = 400     # cap on records pulled per subgenus before filtering


def get(endpoint, **params):
    params.update(TOOL)
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as fh:
                return fh.read()
        except Exception as exc:  # noqa: BLE001 - network retry
            if attempt == 4:
                raise
            print(f"    retry {attempt + 1} after {exc}", flush=True)
            time.sleep(4 * 2 ** attempt)
    raise RuntimeError("unreachable")


def search(term, retmax=100000):
    root = ET.fromstring(get("esearch.fcgi", db="nuccore", term=term, retmax=retmax, sort="accession"))
    return [e.text for e in root.findall(".//Id")]


def spread_sample(items, cap):
    """Evenly spaced subset, so the download spans the whole accession range rather than its start."""
    if len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def summaries(uids):
    """Map uid -> (accession.version, title, length, taxid)."""
    out = {}
    for i in range(0, len(uids), 200):
        chunk = uids[i:i + 200]
        root = ET.fromstring(get("esummary.fcgi", db="nuccore", id=",".join(chunk)))
        for doc in root.findall(".//DocSum"):
            uid = doc.findtext("Id")
            items = {it.get("Name"): it.text for it in doc.findall("Item")}
            out[uid] = (items.get("AccessionVersion"), items.get("Title"), int(items.get("Slen") or 0))
        time.sleep(0.4)
    return out


def fetch_genbank(accs, path):
    with open(path, "w") as fh:
        for i in range(0, len(accs), 25):
            chunk = accs[i:i + 25]
            fh.write(get("efetch.fcgi", db="nuccore", id=",".join(chunk),
                         rettype="gbwithparts", retmode="text").decode())
            time.sleep(0.4)


def ambiguous_fraction(seq):
    code = cga.encode(seq)
    return float((code < 0).mean())


def identity_matrix(seqs, k=12):
    """Pairwise k-mer containment identity: shared distinct words / smaller word set.

    A fast stand-in for alignment identity, good enough to spot near-duplicates.
    """
    sets = []
    for s in seqs:
        idx = cga.kmer_index(cga.encode(s), k)
        sets.append(set(idx[idx >= 0].tolist()))
    n = len(sets)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(sets[i] & sets[j])
            M[i, j] = M[j, i] = shared / max(1, min(len(sets[i]), len(sets[j])))
    return M


def greedy_spread(D, n, first):
    """Pick n indices maximising the minimum pairwise distance, starting from `first`."""
    chosen = [first]
    while len(chosen) < n:
        remaining = [i for i in range(len(D)) if i not in chosen]
        if not remaining:
            break
        nxt = max(remaining, key=lambda i: min(D[i, c] for c in chosen))
        chosen.append(nxt)
    return sorted(chosen)


def host_of(rec):
    for feat in rec.features:
        if feat.type == "source":
            for key in ("host", "isolation_source", "lab_host"):
                if key in feat.qualifiers:
                    return feat.qualifiers[key][0]
    return ""


def build(name, lo, hi, extra_term="", extra_accs=()):
    print(f"[{name}] searching", flush=True)
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    term = (f'"{name}"[Organism] {extra_term} AND {lo}:{hi}[SLEN] '
            f'AND biomol_genomic[PROP] NOT patent[PROP]')
    all_uids = search(term)
    uids = spread_sample(all_uids, MAX_DOWNLOAD)
    print(f"[{name}] {len(all_uids)} records in the length window; "
          f"taking {len(uids)} evenly spaced by accession", flush=True)
    n_all = len(all_uids)
    info = summaries(uids)
    accs = [info[u][0] for u in uids if u in info and info[u][0]]
    accs += [a for a in extra_accs if a not in accs]

    raw = os.path.join(OUT, name, "downloaded.gb")
    if not os.path.exists(raw):
        print(f"[{name}] downloading {len(accs)} records", flush=True)
        fetch_genbank(accs, raw)
    recs = list(SeqIO.parse(raw, "genbank"))
    print(f"[{name}] parsed {len(recs)}", flush=True)

    kept, dropped_amb = [], 0
    for rec in recs:
        seq = cga.genome(rec)
        if not (lo <= len(seq) <= hi):
            continue
        if ambiguous_fraction(seq) > MAX_AMBIGUOUS:
            dropped_amb += 1
            continue
        kept.append(rec)
    print(f"[{name}] {len(kept)} pass quality ({dropped_amb} dropped for ambiguity)", flush=True)

    seqs = [cga.genome(r) for r in kept]
    ident = identity_matrix(seqs)

    # Collapse near-duplicates: keep the first member of each group of mutually similar records.
    order = sorted(range(len(kept)), key=lambda i: -len(seqs[i]))
    representatives = []
    for i in order:
        if all(ident[i, j] < DEDUP_IDENTITY for j in representatives):
            representatives.append(i)
    representatives.sort()
    print(f"[{name}] {len(representatives)} after collapsing at {DEDUP_IDENTITY:.3f} identity", flush=True)

    D = 1.0 - ident[np.ix_(representatives, representatives)]
    pick = greedy_spread(D, min(TARGET_N, len(representatives)), first=0)
    selected = [representatives[i] for i in pick]

    # Hand-added accessions are included whatever the spread picks: they are there to represent
    # a specific virus, not to add diversity.
    forced = [i for i, r in enumerate(kept) if r.id in extra_accs and i not in selected]
    selected = sorted(set(selected) | set(forced))
    print(f"[{name}] selected {len(selected)}", flush=True)

    with open(os.path.join(OUT, name, "genomes.gb"), "w") as fh:
        SeqIO.write([kept[i] for i in selected], fh, "genbank")

    with open(os.path.join(OUT, name, "selection.tsv"), "w") as fh:
        fh.write("accession\tlength\torganism\thost\tcountry\tcollection_date\tambiguous_fraction\n")
        for i in selected:
            rec = kept[i]
            src = next((f for f in rec.features if f.type == "source"), None)
            qual = src.qualifiers if src else {}
            fh.write("\t".join([
                rec.id, str(len(rec.seq)), rec.annotations.get("organism", ""),
                host_of(rec), qual.get("country", qual.get("geo_loc_name", [""]))[0],
                qual.get("collection_date", [""])[0],
                f"{ambiguous_fraction(cga.genome(rec)):.5f}",
            ]) + "\n")

    with open(os.path.join(OUT, name, "filter-log.txt"), "w") as fh:
        fh.write(f"query: {term}\n")
        fh.write(f"records matching query: {n_all}\n")
        fh.write(f"downloaded (evenly spaced by accession, cap {MAX_DOWNLOAD}): {len(uids)}\n")
        fh.write(f"records downloaded: {len(recs)}\n")
        fh.write(f"passed length+ambiguity (<= {MAX_AMBIGUOUS:.0%} non-ACGT): {len(kept)}\n")
        fh.write(f"dropped for ambiguity: {dropped_amb}\n")
        fh.write(f"representatives after {DEDUP_IDENTITY:.3f} identity collapse: {len(representatives)}\n")
        fh.write(f"selected by max-min spread: {len(selected)}\n")
    return len(selected)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or None
    for name, lo, hi, extra_term, extra_accs in TARGETS:
        if only and name not in only:
            continue
        build(name, lo, hi, extra_term, extra_accs)
