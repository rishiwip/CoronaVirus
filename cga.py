"""Shared code for the Coronaviridae composition benchmark.

Feature extraction, composition distances, tree utilities and metadata used by the
numbered scripts in this directory. Requires numpy, scipy, biopython and dendropy.
"""
import csv
import itertools
import os

import numpy as np
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CGA_DATA", os.path.join(HERE, "data"))
RESULTS = os.environ.get("CGA_RESULTS", os.path.join(HERE, "results"))
FIGS = os.environ.get("CGA_FIGS", os.path.join(HERE, "figures"))
SEED = 240
BASES = "ACGT"
KS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
KS_STORED = (1, 2, 3, 4, 5, 6)  # count matrices kept in results/features.npz; longer words are recounted
BLOCK = 500  # nt, block size for the block bootstrap

# Host of the sequenced isolate, coded as a mammalian order or as birds (class Aves).
# Taken from the /host qualifier of each record where present, otherwise from the virus name.
HOST_GROUP = {
    "NC_048212": "Chiroptera", "NC_034440": "Chiroptera", "NC_046954": "Rodentia",
    "NC_046964": "Chiroptera", "NC_025217": "Chiroptera", "NC_010437": "Chiroptera",
    "NC_014470": "Chiroptera", "NC_022103": "Chiroptera", "NC_010646": "Artiodactyla",
    "NC_038294": "Primates", "NC_019843": "Primates", "NC_039207": "Eulipotyphla",
    "NC_026011": "Rodentia", "NC_003045": "Artiodactyla", "NC_006213": "Primates",
    "NC_028811": "Chiroptera", "NC_028833": "Chiroptera", "NC_028814": "Chiroptera",
    "NC_028824": "Chiroptera", "NC_011547": "Aves", "NC_028752": "Artiodactyla",
    "NC_002645": "Primates", "NC_046965": "Aves", "NC_016996": "Aves", "NC_034972": "Rodentia",
    "NC_048214": "Aves", "NC_001451": "Aves", "NC_048213": "Aves", "NC_010800": "Aves",
    "NC_002306": "Carnivora", "NC_028806": "Artiodactyla", "NC_038861": "Artiodactyla",
    "NC_030292": "Carnivora", "NC_006577": "Primates", "NC_005831": "Primates",
    "NC_032730": "Rodentia", "NC_016993": "Aves", "NC_010438": "Chiroptera",
    "NC_023760": "Carnivora", "NC_011550": "Aves", "NC_048217": "Rodentia",
    "NC_001846": "Rodentia", "NC_012936": "Rodentia", "NC_048216": "Chiroptera",
    "NC_032107": "Chiroptera", "NC_016994": "Aves", "NC_009020": "Chiroptera",
    "NC_039208": "Artiodactyla", "NC_003436": "Artiodactyla", "NC_017083": "Lagomorpha",
    "NC_009988": "Chiroptera", "NC_030886": "Chiroptera", "NC_018871": "Chiroptera",
    "NC_009021": "Chiroptera", "NC_004718": "Primates", "NC_045512": "Primates",
    "NC_009657": "Chiroptera", "NC_046955": "Eulipotyphla", "NC_016992": "Aves",
    "NC_011549": "Aves", "NC_009019": "Chiroptera", "NC_035191": "Eulipotyphla",
    "NC_048211": "Eulipotyphla", "NC_016991": "Aves", "NC_016995": "Aves",
}
HOST_COMMON = {
    "Chiroptera": "bat", "Aves": "bird", "Primates": "human", "Rodentia": "rodent",
    "Artiodactyla": "even-toed ungulate or whale", "Carnivora": "carnivore",
    "Eulipotyphla": "shrew or hedgehog", "Lagomorpha": "rabbit",
}

SHORT = {
    "NC_048212": "Bat CoV (Eidolon)", "NC_034440": "Bat CoV (Pipistrellus)",
    "NC_046954": "Rodent CoV", "NC_046964": "Bat-CoV P.kuhlii/Italy",
    "NC_025217": "Bat Hp-betaCoV Zhejiang2013", "NC_010437": "Bat CoV 1A",
    "NC_014470": "Bat CoV BM48-31", "NC_022103": "Bat CoV CDPHE15",
    "NC_010646": "Beluga whale CoV SW1", "NC_038294": "MERS-CoV England 1",
    "NC_019843": "MERS-CoV", "NC_039207": "Hedgehog betaCoV (Erinaceus)",
    "NC_026011": "BetaCoV HKU24", "NC_003045": "Bovine CoV", "NC_006213": "HCoV-OC43",
    "NC_028811": "BtMr-AlphaCoV SAX2011", "NC_028833": "BtNv-AlphaCoV SC2013",
    "NC_028814": "BtRf-AlphaCoV HuB2013", "NC_028824": "BtRf-AlphaCoV YN2012",
    "NC_011547": "Bulbul CoV HKU11", "NC_028752": "Camel alphaCoV", "NC_002645": "HCoV-229E",
    "NC_046965": "Canada goose CoV", "NC_016996": "Moorhen CoV HKU21",
    "NC_034972": "AcCoV-JC34", "NC_048214": "Duck CoV", "NC_001451": "IBV (NC_001451)",
    "NC_048213": "IBV (NC_048213)", "NC_010800": "Turkey CoV", "NC_002306": "Feline inf. peritonitis virus",
    "NC_028806": "Swine enteric CoV", "NC_038861": "TGEV", "NC_030292": "Ferret CoV",
    "NC_006577": "HCoV-HKU1", "NC_005831": "HCoV-NL63", "NC_032730": "Lucheng Rn rat CoV",
    "NC_016993": "Magpie-robin CoV HKU18", "NC_010438": "Miniopterus bat CoV HKU8",
    "NC_023760": "Mink CoV", "NC_011550": "Munia CoV HKU13", "NC_048217": "MHV (NC_048217)",
    "NC_001846": "MHV (NC_001846)", "NC_012936": "Rat CoV Parker",
    "NC_048216": "NL63-related bat CoV (NC_048216)", "NC_032107": "NL63-related bat CoV (NC_032107)",
    "NC_016994": "Night heron CoV HKU19", "NC_009020": "Pipistrellus bat CoV HKU5",
    "NC_039208": "Porcine CoV HKU15", "NC_003436": "PEDV", "NC_017083": "Rabbit CoV HKU14",
    "NC_009988": "Rhinolophus bat CoV HKU2", "NC_030886": "Rousettus bat CoV GCCDC1",
    "NC_018871": "Rousettus bat CoV HKU10", "NC_009021": "Rousettus bat CoV HKU9",
    "NC_004718": "SARS-CoV Tor2", "NC_045512": "SARS-CoV-2", "NC_009657": "Scotophilus bat CoV 512",
    "NC_046955": "Shrew CoV (Sorex)", "NC_016992": "Sparrow CoV HKU17",
    "NC_011549": "Thrush CoV HKU12", "NC_009019": "Tylonycteris bat CoV HKU4",
    "NC_035191": "Wencheng shrew CoV (NC_035191)", "NC_048211": "Wencheng shrew CoV (NC_048211)",
    "NC_016991": "White-eye CoV HKU16", "NC_016995": "Wigeon CoV HKU20",
}

PANEL12 = ["NC_046954", "NC_010437", "NC_006213", "NC_003045", "NC_011547", "NC_046965",
           "NC_048214", "NC_045512", "NC_002306", "NC_016994", "NC_017083", "NC_004718"]
PANEL_BAT = ["NC_046964", "NC_025217", "NC_010437", "NC_014470", "NC_022103", "NC_010438",
             "NC_009020", "NC_009988", "NC_030886", "NC_018871", "NC_009021", "NC_009657", "NC_009019"]
PANEL_HUMAN = ["NC_002645", "NC_006577", "NC_005831", "NC_006213"]
PANEL_COSINE = ["NC_014470", "NC_022103", "NC_006213", "NC_002645", "NC_048214", "NC_017083"]

GENUS_ORDER = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
GENUS_COLOR = {"Alphacoronavirus": "#0072B2", "Betacoronavirus": "#D55E00",
               "Gammacoronavirus": "#009E73", "Deltacoronavirus": "#CC79A7", "": "#666666"}


# --------------------------------------------------------------------------- records

def load_meta():
    """Return list of dicts (one per accession, file order) with taxonomy parsed from the lineage."""
    rows = list(csv.DictReader(open(os.path.join(DATA, "records.tsv")), delimiter="\t"))
    for r in rows:
        lin = r["ncbi_lineage"].split("; ")
        r["genus"] = next((x for x in lin if x.endswith("coronavirus") and x in GENUS_ORDER), "")
        r["subgenus"] = next((x for x in lin if x.endswith("covirus")), "")
        r["host_group"] = HOST_GROUP[r["accession"]]
        r["short"] = SHORT[r["accession"]]
        r["length"] = int(r["length"])
    return rows


def load_records(accs):
    return {a: SeqIO.read(os.path.join(DATA, "genbank", a + ".gb"), "genbank") for a in accs}


def genome(rec):
    return str(rec.seq).upper().replace("U", "T")


# --------------------------------------------------------------------------- k-mer counts

def encode(seq):
    """Map A,C,G,T to 0..3 and anything else to -1."""
    lut = np.full(256, -1, dtype=np.int64)
    for i, b in enumerate(BASES):
        lut[ord(b)] = i
    return lut[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def kmer_index(code, k):
    """Index (0..4^k-1) of the k-mer starting at each position, -1 if it contains an ambiguous base."""
    n = len(code) - k + 1
    idx = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for j in range(k):
        c = code[j:j + n]
        bad |= c < 0
        idx = idx * 4 + np.where(c < 0, 0, c)
    idx[bad] = -1
    return idx


def kmer_counts(seq, k):
    """Standard overlapping k-mer counts on one strand; k-mers with ambiguous bases are skipped."""
    idx = kmer_index(encode(seq), k)
    return np.bincount(idx[idx >= 0], minlength=4 ** k).astype(float)


def count_matrix(seqs, k):
    return np.array([kmer_counts(s, k) for s in seqs])


def block_kmer_indices(seq, k, block=BLOCK):
    """k-mer indices per consecutive block (k-mers lying wholly inside a block). Shape: blocks x (block-k+1)."""
    code = encode(seq)
    nb = len(code) // block
    return np.array([kmer_index(code[b * block:(b + 1) * block], k) for b in range(nb)], dtype=np.int32)


def words(k, alphabet=BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


def original_counts(seq, k):
    """Counts as produced by the coursework notebook: str.count (non-overlapping) on FASTA text
    wrapped at 70 characters, word order A,T,G,C."""
    txt = "\n".join(seq[i:i + 70] for i in range(0, len(seq), 70)) + "\n"
    return np.array([txt.count(w) for w in words(k, "ATGC")], float)


# --------------------------------------------------------------------------- coding-sequence features

SENSE = [c for c in words(3, "TCAG") if c not in ("TAA", "TAG", "TGA")]
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
_CODE = dict(zip(words(3, "TCAG"), "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))


def coding_sequences(rec):
    """Annotated CDS nucleotide sequences, with CDS nested in a longer CDS of the same start dropped
    (ORF1a inside ORF1ab), so that no codon of the replicase is counted twice."""
    cds = [f for f in rec.features if f.type == "CDS"]
    keep = []
    for f in cds:
        longer = [g for g in cds if g is not f and int(g.location.start) == int(f.location.start)
                  and len(g.location) > len(f.location)]
        if not longer:
            keep.append(f)
    seqs = []
    for f in keep:
        s = str(f.extract(rec.seq)).upper().replace("U", "T")
        s = s[:len(s) - len(s) % 3]
        seqs.append(s)
    return seqs


def codon_features(rec):
    """Return (codon frequency over 61 sense codons, RSCU over 61 sense codons, amino-acid frequency)."""
    counts = dict.fromkeys(SENSE, 0)
    for s in coding_sequences(rec):
        for i in range(0, len(s), 3):
            c = s[i:i + 3]
            if c in counts:
                counts[c] += 1
    cod = np.array([counts[c] for c in SENSE], float)
    aa_of = np.array([_CODE[c] for c in SENSE])
    rscu = np.zeros(len(SENSE))
    aa = np.zeros(len(AA20))
    for j, a in enumerate(AA20):
        m = aa_of == a
        tot = cod[m].sum()
        aa[j] = tot
        rscu[m] = cod[m] * m.sum() / tot if tot else 1.0
    return cod / cod.sum(), rscu, aa / aa.sum()


# --------------------------------------------------------------------------- distances

def euclid(X):
    return squareform(pdist(X, "euclidean"))


def cosine_dist(X):
    return squareform(pdist(X, "cosine"))


def zscore(X):
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


def freqs(C):
    return C / C.sum(1, keepdims=True)


def expected_m0(C1, k):
    """Expected k-mer frequencies under independence of bases (zero-order Markov), per genome."""
    p = freqs(C1)
    E = p
    for _ in range(k - 1):
        E = (E[:, :, None] * p[:, None, :]).reshape(len(p), -1)
    return E


def expected_m1(C1, C2, k):
    """Expected k-mer frequencies under a first-order Markov chain fitted to each genome."""
    p = freqs(C1)
    T = C2.reshape(-1, 4, 4)
    T = T / T.sum(2, keepdims=True)  # T[a, b] = P(next base b | base a)
    E = (p[:, :, None] * T).reshape(len(p), 16)
    for _ in range(k - 2):
        m = E.shape[1]
        E = (E.reshape(len(p), m // 4, 4)[:, :, :, None] * T[:, None, :, :]).reshape(len(p), m * 4)
    return E


def _from_gram(G):
    D = np.clip(G, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2


def d2star(Ck, E):
    """d2* dissimilarity (Reinert et al. 2009; Wan et al. 2010): counts are centred and scaled by
    their expectation under the background model E (frequencies, one row per genome), and the
    dissimilarity is 0.5 * (1 - cosine of the standardised vectors)."""
    exp = np.maximum(Ck.sum(1, keepdims=True) * E, 1e-12)
    Z = (Ck - exp) / np.sqrt(exp)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return _from_gram(0.5 * (1.0 - Z @ Z.T))


def jaccard_presence(Ck):
    """Jaccard distance between the sets of k-mers present in each genome."""
    B = (Ck > 0).astype(np.float32)
    inter = (B @ B.T).astype(float)
    size = np.diag(inter)
    return _from_gram(1.0 - inter / (size[:, None] + size[None, :] - inter))


def karlin_delta(C1, C2):
    """Karlin & Burge dinucleotide relative-abundance distance, single-strand form:
    rho_xy = f_xy / (f_x f_y); delta* = mean |rho_xy(a) - rho_xy(b)| over the 16 dinucleotides."""
    rho = freqs(C2) / expected_m0(C1, 2)
    return squareform(pdist(rho, "cityblock")) / 16.0


def kmer_distances(k, Ck, C1, C2):
    """Distance matrices for one word length. Returns {method: matrix}."""
    D = {}
    F = freqs(Ck)
    D["freq-euclid"] = euclid(F)
    D["freq-cosine"] = cosine_dist(F)
    Zs = zscore(F)
    del F
    D["zfreq-euclid"] = euclid(Zs)
    del Zs
    if k >= 2:
        D["d2star-M0"] = d2star(Ck, expected_m0(C1, k))
    if k >= 3:
        D["d2star-M1"] = d2star(Ck, expected_m1(C1, C2, k))
    if k >= 5:
        D["presence-jaccard"] = jaccard_presence(Ck)
    if k == 2:
        D["karlin-delta"] = karlin_delta(C1, C2)
    return D


def distance_from_counts(method, k, Ck, C1, C2):
    """Single distance matrix from (possibly resampled) counts; used by the block bootstrap."""
    if method == "freq-euclid":
        return euclid(freqs(Ck))
    if method == "freq-cosine":
        return cosine_dist(freqs(Ck))
    if method == "d2star-M0":
        return d2star(Ck, expected_m0(C1, k))
    if method == "d2star-M1":
        return d2star(Ck, expected_m1(C1, C2, k))
    if method == "presence-jaccard":
        return jaccard_presence(Ck)
    raise ValueError(method)


# --------------------------------------------------------------------------- trees and splits

def canon(side, taxa, ref):
    """Canonical form of a split: the side that does not contain the reference taxon."""
    side = frozenset(side)
    return frozenset(taxa - side) if ref in side else side


def linkage_splits(Z, names):
    """Non-trivial splits of a scipy linkage tree, read as an unrooted tree."""
    n = len(names)
    taxa = frozenset(names)
    members = {i: frozenset([names[i]]) for i in range(n)}
    out = set()
    for i, row in enumerate(Z):
        m = members[int(row[0])] | members[int(row[1])]
        members[n + i] = m
        if 2 <= len(m) <= n - 2:
            out.add(canon(m, taxa, names[0]))
    return out


def upgma_splits(D, names, method="average"):
    return linkage_splits(linkage(squareform(D, checks=False), method), names)


def nj(D, names):
    """Neighbour joining (Saitou & Nei 1987). Returns (non-trivial splits of the unrooted tree,
    Newick string with branch lengths; negative lengths are set to zero)."""
    n = len(names)
    taxa = frozenset(names)
    D = D.astype(float).copy()
    members = [frozenset([x]) for x in names]
    text = list(names)
    active = list(range(n))
    out = set()
    while len(active) > 3:
        sub = D[np.ix_(active, active)]
        m = len(active)
        r = sub.sum(1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = active[i], active[j]
        la = 0.5 * D[a, b] + (r[i] - r[j]) / (2 * (m - 2))
        lb = D[a, b] - la
        new = 0.5 * (D[a] + D[b] - D[a, b])
        D = np.vstack([D, new])
        D = np.hstack([D, np.append(new, 0.0)[:, None]])
        members.append(members[a] | members[b])
        text.append(f"({text[a]}:{max(la, 0):.6g},{text[b]}:{max(lb, 0):.6g})")
        clade = members[-1]
        if 2 <= len(clade) <= n - 2:
            out.add(canon(clade, taxa, names[0]))
        active = [x for x in active if x not in (a, b)] + [D.shape[0] - 1]
    x, y, z = active
    lx = 0.5 * (D[x, y] + D[x, z] - D[y, z])
    ly = 0.5 * (D[x, y] + D[y, z] - D[x, z])
    lz = 0.5 * (D[x, z] + D[y, z] - D[x, y])
    newick = f"({text[x]}:{max(lx, 0):.6g},{text[y]}:{max(ly, 0):.6g},{text[z]}:{max(lz, 0):.6g});"
    return out, newick


def nj_splits(D, names):
    return nj(D, names)[0]


def read_newick_splits(path):
    """Read an IQ-TREE tree. Returns (dendropy tree, {split: (SH-aLRT, UFBoot)}, taxa)."""
    import dendropy
    tree = dendropy.Tree.get(path=path, schema="newick", preserve_underscores=True)
    taxa = frozenset(t.label for t in tree.taxon_namespace)
    ref = sorted(taxa)[0]
    splits = {}
    for nd in tree.postorder_internal_node_iter():
        leaves = frozenset(l.taxon.label for l in nd.leaf_iter())
        if 2 <= len(leaves) <= len(taxa) - 2:
            sup = (float("nan"), float("nan"))
            if nd.label:
                parts = nd.label.split("/")
                if len(parts) == 2:
                    sup = (float(parts[0]), float(parts[1]))
            splits[canon(leaves, taxa, ref)] = sup
    return tree, splits, taxa


def recanon(splits, taxa, ref):
    return {canon(s, taxa, ref) for s in splits}


def patristic(tree, names):
    pdm = tree.phylogenetic_distance_matrix()
    tx = {t.label: t for t in tree.taxon_namespace}
    n = len(names)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = P[j, i] = pdm.patristic_distance(tx[names[i]], tx[names[j]])
    return P


def rf_normalised(a, b):
    return len(a ^ b) / (len(a) + len(b))


def is_clan(group, splits, labelled):
    """True if `group` is one side of some split once the tree is restricted to `labelled` taxa."""
    group = frozenset(group)
    other = frozenset(labelled) - group
    for s in splits:
        inside = s & labelled
        if inside == group or inside == other:
            return True
    return False
