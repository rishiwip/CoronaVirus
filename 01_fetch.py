#!/usr/bin/env python3
"""Fetch the 65 Coronaviridae RefSeq records (GenBank flat file, with versions) from NCBI.

Writes data/genbank/<accession>.gb and data/records.tsv. No API key or e-mail is used;
requests are spaced to stay under the anonymous E-utilities rate limit.
"""
import datetime
import os
import sys
import time
import urllib.request

from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CGA_DATA", os.path.join(HERE, "data"))
GB = os.path.join(DATA, "genbank")
os.makedirs(GB, exist_ok=True)

ACCESSIONS = """
NC_048212 NC_034440 NC_046954 NC_046964 NC_025217 NC_010437 NC_014470 NC_022103 NC_010646
NC_038294 NC_019843 NC_039207 NC_026011 NC_003045 NC_006213 NC_028811 NC_028833 NC_028814
NC_028824 NC_011547 NC_028752 NC_002645 NC_046965 NC_016996 NC_034972 NC_048214 NC_001451
NC_048213 NC_010800 NC_002306 NC_028806 NC_038861 NC_030292 NC_006577 NC_005831 NC_032730
NC_016993 NC_010438 NC_023760 NC_011550 NC_048217 NC_001846 NC_012936 NC_048216 NC_032107
NC_016994 NC_009020 NC_039208 NC_003436 NC_017083 NC_009988 NC_030886 NC_018871 NC_009021
NC_004718 NC_045512 NC_009657 NC_046955 NC_016992 NC_011549 NC_009019 NC_035191 NC_048211
NC_016991 NC_016995
""".split()
assert len(ACCESSIONS) == 65 and len(set(ACCESSIONS)) == 65


def fetch(acc):
    path = os.path.join(GB, acc + ".gb")
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return path
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           f"?db=nuccore&id={acc}&rettype=gbwithparts&retmode=text")
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            if data.startswith(b"LOCUS"):
                with open(path, "wb") as fh:
                    fh.write(data)
                time.sleep(0.4)
                return path
        except Exception as exc:  # network hiccup: back off and retry
            print(f"  retry {acc}: {exc}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {acc}")


def main():
    rows = []
    for acc in ACCESSIONS:
        rec = SeqIO.read(fetch(acc), "genbank")
        src = [f for f in rec.features if f.type == "source"][0]
        q = src.qualifiers
        lineage = rec.annotations.get("taxonomy", [])
        rows.append([
            acc, rec.id, str(len(rec.seq)), rec.annotations.get("organism", ""),
            q.get("host", [""])[0], q.get("strain", q.get("isolate", [""]))[0],
            q.get("country", q.get("geo_loc_name", [""]))[0], q.get("collection_date", [""])[0],
            "; ".join(lineage), rec.annotations.get("date", ""),
        ])
        print(acc, rec.id, len(rec.seq), rec.annotations.get("organism", ""), sep="\t")
    with open(os.path.join(DATA, "records.tsv"), "w") as fh:
        fh.write("accession\tversion\tlength\torganism\thost_qualifier\tstrain_or_isolate\tcountry\t"
                 "collection_date\tncbi_lineage\trecord_date\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(os.path.join(DATA, "fetch-date.txt"), "w") as fh:
        fh.write(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "\n")


if __name__ == "__main__":
    main()
