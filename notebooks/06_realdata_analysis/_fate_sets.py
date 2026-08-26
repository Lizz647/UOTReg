"""The fate calls and their supporting genes -- the settled marker analysis.

Shared by `fates_and_markers` and `figures` (embryoid seed 5 / statefate seed 2) so every notebook
makes the same claims from the same lists. Edit here, not in a notebook.

**These are the cluster ids the paper uses**, not raw K-means ids -- the raw ids are arbitrary and
are mapped onto these before anything here is used (`cluster_id_map` in `fates_and_markers`).

Each fate carries the genes the paper would quote for it. The point of the check the notebooks run is
not to re-derive them but to CONFIRM that every gene we intend to print actually appears in that
cluster's Wilcoxon ranking, and at what rank -- a gene that is absent must not be quoted, however
canonical it is.

`merge` groups clusters into the simplified fate the main text claims (embryoid's germ
layers). statefate deliberately has none: its four clusters are terminal fates, not sublineages of a
smaller set, so merging them would invent a claim the paper does not make.
"""

# --------------------------------------------------------------------------------- embryoid, seed 5
# Simplified (main-text) fates: mesoderm = pub 1 u 5, neuroectoderm = pub 2 u 4, endoderm = pub 0.
# Cluster 3 is proliferative and is NOT a fate -- it is listed so the table is complete and so the
# text can say explicitly that it was not called a lineage.
EMBRYOID = {
    "name": "embryoid",
    "fates": [
        dict(fate="ME-like (mesoderm)", pub=[1, 5], is_fate=True,
             sub={1: "cardiac", 5: "ECM-rich"},
             final={1: ["BMP4", "MEIS2", "CCBE1"],
                    5: ["POSTN", "COL1A1", "COL3A1", "COL5A1", "DCN"]},
             initial={1: ["TERF1", "TDGF1", "POU5F1", "MGST1", "PMAIP1", "LECT1", "CD9", "EPCAM"],
                      5: ["ANXA2", "S100A10", "APOE"]},
             note="POSTN, BMP4, MEIS2 are GOLD (DE + signature table + canonical). Day-0 support is "
                  "carried by the CARDIAC cluster; the ECM cluster is the weak one (5/10)."),
        dict(fate="NE-like (neuroectoderm)", pub=[2, 4], is_fate=True,
             sub={2: "neural crest", 4: "central"},
             final={2: ["TFAP2B", "SIX1", "DLX1", "DLX2", "PLP1"],
                    4: ["SOX2", "SOX3", "GPM6A"]},
             initial={2: ["TMSB4X", "POU5F1", "HIST1H4C", "L1TD1", "TUBB2A", "PTN", "CRABP1"],
                      4: ["VIM", "TPBG", "DLK1", "TUBA1B", "TUBA1A", "GPC3", "NNAT", "CRABP2",
                          "PTN"]},
             note="TFAP2B is the strongest crest call (pioneer factor for crest specification). "
                  "The crest cluster also carries POU5F1, so the clean OCT4/SOX2 contrast is "
                  "cardiac-ME vs CENTRAL NE only."),
        dict(fate="EN-like (endoderm, extraembryonic)", pub=[0], is_fate=True,
             sub={0: "visceral/extraembryonic"},
             final={0: ["AFP", "TTR", "APOA1", "APOA2", "AHSG", "KRT8", "KRT18", "EPCAM"]},
             initial={0: ["KRT18", "ANXA2", "KRT19", "S100A11", "KRT8"]},
             note="EXTRAEMBRYONIC (visceral), NOT definitive endoderm: AFP present, SOX17 only deep, "
                  "no FOXA2/GATA4/GATA6/MIXL1/GSC. Write 'extraembryonic (visceral) endoderm-like'."),
        dict(fate="proliferative (not a fate)", pub=[3], is_fate=False,
             sub={3: "proliferative/transitional"},
             final={3: ["AURKA", "TOP2A", "NUSAP1", "CENPF", "UBE2C", "PLK1"]},
             initial={3: ["UBE2C", "HMGB2", "ARL6IP1", "KPNA2", "CCNB1", "TOP2A"]},
             note="Cell-cycle programme, not a lineage. Report it as such rather than omitting it."),
    ],
    "day0_story": ("Day 0 separates by SIGNALLING STATE, not lineage markers: the mesoderm-fated "
                   "cells carry the Nodal/OCT4 arm (TDGF1, POU5F1, CD9) and no SOX2/PAX6, while the "
                   "central-neuroectoderm cells carry SOX2/PAX6 and the RA-binding proteins "
                   "(CRABP2, PTN) and none of OCT4/TDGF1/CD9."),
}

# ------------------------------------------------------------------------------- statefate, seed 2
STATEFATE = {
    "name": "statefate",
    "fates": [
        dict(fate="Neutrophil", pub=[0], is_fate=True, sub={0: "neutrophil maturation"},
             final={0: ["CEBPE", "G0S2", "LY6C2", "MGST2"]},
             initial={0: ["ELANE", "PRTN3", "MPO", "CTSG", "GFI1"]},
             note="CEBPE drives the promyelocyte->myelocyte transition; the day-2 primary-granule "
                  "genes (ELANE/PRTN3/MPO/CTSG) mark an already-committed granulocyte progenitor."),
        dict(fate="Monocyte", pub=[3], is_fate=True, sub={3: "monocyte"},
             final={3: ["MAFB", "CTSL", "MPEG1", "LGALS3"]},
             initial={3: ["CSF1R", "SIRPA", "PSAP", "FCER1G", "CTSC", "LGALS3", "MS4A6C"]},
             note="MAFB/CTSL/MPEG1 GOLD; MAFB is the original text's gene."),
        dict(fate="Baso/mast arm", pub=[2], is_fate=True, sub={2: "baso/mast"},
             final={2: ["CPA3", "HDC", "CD200R3", "IL13"]},
             initial={2: ["GATA2", "ITGA2B", "CSF2RB", "ALOX5", "CPA3", "MS4A2"]},
             note="The full label is 'erythro-megakaryocytic (baso/eo/mast)' and the "
                  "top-10 lists the erythro-mega side; we recover the BASO/MAST arm of the same "
                  "fate. A 0/10 table overlap here is not a failure -- state the arm explicitly."),
        dict(fate="Early progenitor (MPP/lymphoid)", pub=[1], is_fate=True,
             sub={1: "MPP/lymphoid"},
             final={1: ["SOX4", "CD34"]},
             initial={1: ["CD34", "FLT3", "LY6A", "CD27", "BCL11A", "IGHM", "PTPRCAP"]},
             note="WEAKEST on the literature check (+2 final, +1 initial) and much the largest "
                  "cluster. Describe as 'an early progenitor compartment'; do not claim the same "
                  "separation strength as the three committed fates."),
    ],
    "day0_story": ("Unlike embryoid, statefate's day-2 cells are already committed progenitors and "
                   "carry the lineage genes themselves, which is why the day-0 (predictability) "
                   "result is the stronger of the two datasets."),
}

SETS = {"embryoid": EMBRYOID, "statefate": STATEFATE}


def fate_sets(dataset):
    if dataset not in SETS:
        raise KeyError(f"no fate set for {dataset!r} -- add it in _fate_sets.py")
    return SETS[dataset]


def gene_rank(de_lists, pub_cluster, gene, mapping=None):
    """1-based rank of `gene` in a cluster's DE ranking, or None if absent.

    Case-insensitive: the statefate caches upper-case their gene names while the literature (and
    `FATE_GENES_FINAL.md`) writes them capitalised, so a case-sensitive lookup silently reports every
    mouse gene as missing.

    `mapping` maps paper id -> raw K-means id; pass it when `de_lists` is keyed by raw ids.
    """
    key = mapping[pub_cluster] if mapping is not None else pub_cluster
    lst = de_lists.get(key)
    if lst is None:
        return None
    up = [str(g).upper() for g in lst]
    g = str(gene).upper()
    return (up.index(g) + 1) if g in up else None
