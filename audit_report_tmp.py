"""TEMPORARY: summarize genome identity audit findings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "audit_genomes_tmp.json").read_text(encoding="utf-8"))

# Known NCBI taxonomy renames: panel name -> accepted current NCBI name
RENAMES = {
    "Prevotella timonensis": "Hoylesella timonensis",
    "Mycoplasma hominis": "Metamycoplasma hominis",
    "Candidatus Lachnocurva vaginae": None,  # MAG names vary
    "TM7-H1": None,
    "Lactobacillus gasseri": "Lactobacillus gasseri",
}

print("=" * 110)
print("GENOME IDENTITY AUDIT")
print("=" * 110)

issues = []
for r in data:
    lid = r["llamp_id"]
    if r["status"] != "checked":
        issues.append((lid, "STATUS", r["status"]))
        continue

    # 1. accession echo
    if r.get("ncbi_accession_returned") and r["ncbi_accession_returned"] != r["accession"]:
        issues.append((lid, "ACCESSION",
                       f"panel {r['accession']} -> NCBI returned {r['ncbi_accession_returned']}"))

    # 2. sequence accession set: local FASTA vs NCBI assembly
    loc = set(a.split()[0] for a in r.get("local_seq_accs", []))
    rem = set(a for a in r.get("ncbi_seq_accs", []) if a)
    if rem:
        if not loc <= rem:
            extra = sorted(loc - rem)[:3]
            issues.append((lid, "SEQ_ACC",
                           f"{len(loc - rem)} local seq accs not in assembly, e.g. {extra}"))
        if len(loc) != len(rem):
            issues.append((lid, "CONTIG_COUNT",
                           f"local {len(loc)} vs NCBI assembly {len(rem)} sequences"))

    # 3. base count
    lb, nb = r.get("local_bases", 0), r.get("ncbi_bases", 0)
    if nb and abs(lb - nb) > max(1000, 0.001 * nb):
        issues.append((lid, "LENGTH", f"local {lb:,} bp vs NCBI {nb:,} bp"))

    # 4. organism name
    panel_sp = r["panel_species"]
    ncbi_org = r.get("ncbi_organism", "")
    expected = RENAMES.get(panel_sp, panel_sp)
    if expected is not None and not ncbi_org.startswith(expected):
        issues.append((lid, "ORGANISM", f"panel '{panel_sp}' vs NCBI '{ncbi_org}'"))

    # 5. strain
    ps, ns = r["panel_strain"].strip(), (r.get("ncbi_strain") or "").strip()
    if ns and ps and ps not in ns and ns not in ps:
        issues.append((lid, "STRAIN", f"panel '{ps}' vs NCBI '{ns}'"))

    # 6. assembly level
    if r.get("ncbi_level") and r["panel_level"] != r["ncbi_level"]:
        issues.append((lid, "LEVEL", f"panel '{r['panel_level']}' vs NCBI '{r['ncbi_level']}'"))

    # 7. MAG detection in headers
    if "MAG" in r.get("local_first_header", "") and r["confidence"] == "standard":
        issues.append((lid, "MAG_CONF",
                       f"header says MAG but confidence='standard': {r['local_first_header'][:80]}"))

print(f"\nChecked {len(data)} genomes, {len(issues)} finding(s)\n")
by_kind = {}
for lid, kind, msg in issues:
    by_kind.setdefault(kind, []).append((lid, msg))
for kind in sorted(by_kind):
    print(f"--- {kind} ({len(by_kind[kind])}) ---")
    for lid, msg in by_kind[kind]:
        print(f"  {lid}\n      {msg}")
    print()

print("=" * 110)
print("FULL TABLE: panel vs NCBI")
print("=" * 110)
print(f"{'llamp_id':<62} {'NCBI organism':<42} {'NCBI strain':<22} {'lvl':<16} {'local bp':>11} {'ncbi bp':>11} {'ctg l/n':>10}")
for r in data:
    print(f"{r['llamp_id'][:60]:<62} {str(r.get('ncbi_organism',''))[:40]:<42} "
          f"{str(r.get('ncbi_strain',''))[:20]:<22} {str(r.get('ncbi_level',''))[:14]:<16} "
          f"{r.get('local_bases',0):>11,} {r.get('ncbi_bases',0):>11,} "
          f"{str(r.get('local_contigs','?')) + '/' + str(len(r.get('ncbi_seq_accs',[]))):>10}")
