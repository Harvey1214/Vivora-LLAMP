"""TEMPORARY audit: verify tokenization, MIC conversion, aggregation, and ranking.

Uses results/current/ when present; otherwise the archived strain-panel CSVs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results" / "current"
ARCHIVE = ROOT / "results" / "archive"
pd.set_option("display.width", 200)

if (CURRENT / "predictions_by_strain.csv").exists():
    RES = CURRENT
    strain_name = "predictions_by_strain.csv"
    species_name = "predictions_by_species.csv"
    rank_name = "candidate_ranking.csv"
    scramble_name = "scramble_control.csv"
    desc_name = "descriptor_correlations.csv"
    cal_name = "calibration.csv"
else:
    RES = ARCHIVE
    strain_name = "mic_predictions_genome_strains.csv"
    species_name = "mic_predictions_genome_species.csv"
    rank_name = "selectivity_ranking_genome_strain_panel.csv"
    scramble_name = None
    desc_name = None
    cal_name = "genome_route_calibration.csv"
    print(f"[info] results/current empty — auditing archive strain-panel CSVs in {RES}")

print("=" * 100)
print("B1. Tokenization: original LLAMP algorithm vs refactored ' '.join()")
print("=" * 100)


def original_spacing(seq):
    i = seq
    for j in range(len(i) - 1):
        i = i[: j + j + 1] + " " + i[j + j + 1:]
    return i


cands = pd.read_csv(ROOT / "data" / "candidates" / "current.csv")
seqs = cands["Sequence"].dropna().astype(str).str.strip().str.upper().tolist()
mismatch = [s for s in seqs if original_spacing(s) != " ".join(s)]
print(f"sequences checked: {len(seqs)}, spacing mismatches: {len(mismatch)}")
for s in mismatch[:5]:
    print("  MISMATCH:", s)

print()
print("=" * 100)
print("B2 / C1. Recompute MIC conversion + ranking independently")
print("=" * 100)

strain = pd.read_csv(RES / strain_name)
species = pd.read_csv(RES / species_name)
rank = pd.read_csv(RES / rank_name)
panel = pd.read_csv(ROOT / "config" / "species_panel.csv")

print(f"source: {RES}")
print(f"strain rows: {len(strain):,}  species rows: {len(species):,}  ranking rows: {len(rank):,}")
print(f"unique peptides: {strain['Sequence'].nunique()}  unique genomes: {strain['llamp_id'].nunique()}")
print(f"expected grid: {strain['Sequence'].nunique() * strain['llamp_id'].nunique():,}")

# B2: mic_uM == 10 ** mic_log_uM
conv_err = np.abs(strain["mic_uM"] - 10 ** strain["mic_log_uM"]).max()
print(f"\nB2 max |mic_uM - 10**mic_log_uM| = {conv_err:.3e}")

# C2: role integrity
print("\nC2. Role assignment from panel:")
roles = panel.groupby("role")["species"].apply(lambda s: sorted(set(s)))
for role, sp in roles.items():
    print(f"  {role:<12} ({len(sp)} species): {', '.join(sp)}")

spare_species = set(panel[panel.role == "spare"]["species"])
core_species = set(panel[panel.role == "core"]["species"])
bad_spare = [s for s in spare_species if "Lactobacillus" not in s]
print(f"\n  non-Lactobacillus in spare: {bad_spare or 'none'}")
print(f"  overlap spare/core: {spare_species & core_species or 'none'}")

# Recompute species aggregation
recomputed_species = (
    strain.groupby(["Sequence", "species", "role"], as_index=False)
    .agg(
        strain_count=("accession", "nunique"),
        strain_min_mic_uM=("mic_uM", "min"),
        strain_max_mic_uM=("mic_uM", "max"),
        species_typical_mic_log_uM=("mic_log_uM", "mean"),
    )
)
recomputed_species["species_typical_mic_uM"] = 10 ** recomputed_species["species_typical_mic_log_uM"]
merged = species.merge(
    recomputed_species, on=["Sequence", "species", "role"], suffixes=("_file", "_calc")
)
print(f"\nspecies-level rows matched: {len(merged)} / {len(species)}")
for col in ["strain_count", "strain_min_mic_uM", "strain_max_mic_uM",
            "species_typical_mic_log_uM", "species_typical_mic_uM"]:
    d = np.abs(merged[f"{col}_file"] - merged[f"{col}_calc"]).max()
    print(f"  max |delta {col}| = {d:.3e}")

# Recompute ranking
sp = recomputed_species[recomputed_species.role == "spare"]
co = recomputed_species[recomputed_species.role == "core"]
calc = pd.DataFrame({
    "lactobacillus_worst_case_uM": sp.groupby("Sequence")["strain_min_mic_uM"].min(),
    "bv_worst_case_uM": co.groupby("Sequence")["strain_max_mic_uM"].max(),
    "lactobacillus_typical_case_uM": 10 ** sp.groupby("Sequence")["species_typical_mic_log_uM"].mean(),
    "bv_typical_case_uM": 10 ** co.groupby("Sequence")["species_typical_mic_log_uM"].mean(),
})
calc["selectivity_worst_case"] = calc["lactobacillus_worst_case_uM"] / calc["bv_worst_case_uM"]
calc["selectivity_typical_case"] = calc["lactobacillus_typical_case_uM"] / calc["bv_typical_case_uM"]
calc["min_spare_over_typical_kill"] = (
    calc["lactobacillus_worst_case_uM"] / calc["bv_typical_case_uM"]
)
calc = calc.reset_index()

cmp = rank.merge(calc, on="Sequence", suffixes=("_file", "_calc"))
print(f"\nranking rows matched: {len(cmp)} / {len(rank)}")
for col in ["lactobacillus_worst_case_uM", "bv_worst_case_uM", "lactobacillus_typical_case_uM",
            "bv_typical_case_uM", "selectivity_worst_case", "selectivity_typical_case"]:
    d = np.abs(cmp[f"{col}_file"] - cmp[f"{col}_calc"]).max()
    print(f"  max |delta {col}| = {d:.3e}")

if "min_spare_over_typical_kill" in rank.columns:
    d = np.abs(
        cmp["min_spare_over_typical_kill_file"]
        - cmp["min_spare_over_typical_kill_calc"]
    ).max()
    print(f"  max |delta min_spare_over_typical_kill| = {d:.3e}")
else:
    print("  [note] archived ranking lacks min_spare_over_typical_kill (added in main.py)")

# C3: sort order
is_sorted = rank["selectivity_worst_case"].is_monotonic_decreasing
print(f"\nC3. ranking sorted by selectivity_worst_case descending: {is_sorted}")

used = set(recomputed_species[recomputed_species.role.isin(["spare", "core"])]["role"])
unused_roles = set(panel["role"]) - {"spare", "core"}
print(f"\nRoles contributing to score: {sorted(used)}")
print(f"Roles NOT in score (reported only): {sorted(unused_roles)}")
n_excl = panel[panel.role.isin(unused_roles)].shape[0]
print(f"  genomes excluded from score: {n_excl} / {len(panel)}")

print()
print("=" * 100)
print("Strain-count weighting check (does multi-strain species get extra weight?)")
print("=" * 100)
counts = panel[panel.role.isin(["spare", "core"])].groupby(["role", "species"]).size()
print(counts.to_string())

print()
print("=" * 100)
print("C4/C5. Proxy route and artifact provenance")
print("=" * 100)
proxy_mic = ARCHIVE / "mic_predictions.csv"
proxy_rank = ARCHIVE / "selectivity_ranking.csv"
old_genome_rank = ARCHIVE / "selectivity_ranking_genome.csv"
print(f"proxy mic rows: {len(pd.read_csv(proxy_mic)) if proxy_mic.exists() else 'missing'}")
if proxy_mic.exists():
    pm = pd.read_csv(proxy_mic)
    print(f"  unique peptides: {pm['Sequence'].nunique()}")
    print(f"  roles: {sorted(pm['role'].dropna().unique())}")
    print(f"  real_species: {sorted(pm['real_species'].dropna().unique())}")
    print(f"  proxy_species: {sorted(pm['proxy_species'].dropna().unique())}")
    # Gardnerella double-count risk: multiple rows per peptide with same real_species
    g = pm[pm["real_species"] == "Gardnerella vaginalis"]
    print(f"  Gardnerella rows per peptide (median): {g.groupby('Sequence').size().median()}")
if proxy_rank.exists():
    pr = pd.read_csv(proxy_rank)
    print(f"proxy ranking peptides: {len(pr)}  cols: {list(pr.columns)}")
if old_genome_rank.exists():
    ogr = pd.read_csv(old_genome_rank)
    print(f"legacy genome ranking peptides: {len(ogr)}")
print(f"current candidates: {len(cands)} in data/candidates/current.csv")
print(f"strain-panel ranking peptides: {len(rank)}")
print("Canonical workflow: python main.py screen -> results/current/")
print("Archive CSVs are historical only (see results/archive/README.md).")

if (RES / cal_name).exists() if cal_name else False:
    print()
    print("=" * 100)
    print("Calibration: derived genome feature vs LLAMP native stored feature")
    print("=" * 100)
    cal = pd.read_csv(RES / cal_name)
    summary = cal.groupby(["species", "accession"])["delta_log_uM"].agg(
        n="size", mean="mean", max_abs=lambda s: s.abs().max()
    )
    print(summary.to_string())

print()
print("=" * 100)
print("Absolute potency reality check (A4)")
print("=" * 100)
print(rank[["bv_worst_case_uM", "bv_typical_case_uM", "lactobacillus_worst_case_uM",
            "selectivity_worst_case", "selectivity_typical_case"]].describe().to_string())
BV_POTENCY_THRESHOLD_UM = 32.0
n_weak = int((rank["bv_typical_case_uM"] > BV_POTENCY_THRESHOLD_UM).sum())
print(f"\npeptides with typical BV MIC > {BV_POTENCY_THRESHOLD_UM:g} µM: {n_weak}/{len(rank)}")
print("\nTop 10 by selectivity_worst_case with absolute MICs:")
print(rank.head(10)[["Sequence", "lactobacillus_worst_case_uM", "bv_worst_case_uM",
                     "bv_typical_case_uM", "selectivity_worst_case",
                     "selectivity_typical_case"]].to_string(index=False))

print()
print("=" * 100)
print("A3 metric: lowest_spare_MIC / average_kill_MIC")
print("=" * 100)
alt = rank.copy()
alt["min_spare_over_typical_kill"] = (
    alt["lactobacillus_worst_case_uM"] / alt["bv_typical_case_uM"]
)
alt = alt.sort_values("min_spare_over_typical_kill", ascending=False)
print(alt.head(10)[["Sequence", "lactobacillus_worst_case_uM", "bv_typical_case_uM",
                    "min_spare_over_typical_kill", "selectivity_worst_case"]].to_string(index=False))
overlap = set(alt.head(10)["Sequence"]) & set(rank.head(10)["Sequence"])
print(f"\noverlap of top-10 between new metric and worst_case: {len(overlap)}/10")

if scramble_name and (RES / scramble_name).exists():
    print()
    print("=" * 100)
    print("Scramble + descriptor controls")
    print("=" * 100)
    print(pd.read_csv(RES / scramble_name).to_string(index=False))
    if desc_name and (RES / desc_name).exists():
        print()
        print(pd.read_csv(RES / desc_name).to_string(index=False))
