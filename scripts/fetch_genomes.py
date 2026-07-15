"""Fetch nucleotide genome FASTAs for the BV / commensal target species.

LLAMP is species-aware through a 340-dim genome k-mer fingerprint. To score the
peptides in ``shortlist.csv`` against the *real* target species (rather than the
in-model proxies) we need one representative nucleotide genome per species.

This script uses the NCBI Datasets v2 REST API via the system ``curl.exe`` (which
uses the Windows certificate store and sidesteps the Python/pip SSL issues seen on
this machine). For each species it:

  1. queries the ``dataset_report`` endpoint to pick the best RefSeq assembly,
  2. downloads that assembly's GENOME_FASTA as a zip,
  3. extracts the ``.fna`` into ``data/Genomic_featrues/<Species_name>.fna``,
  4. records the accession in ``data/Genomic_featrues/genome_manifest.csv``.

If a download fails (e.g. SSL / offline), the species is marked ``MISSING`` in the
manifest and the resolved accession + direct URL are printed so the FASTA can be
dropped into ``data/Genomic_featrues/`` manually. Everything downstream in the
notebook then works unchanged.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "Genomic_featrues")
MANIFEST = os.path.join(OUT_DIR, "genome_manifest.csv")

API = "https://api.ncbi.nlm.nih.gov/datasets/v2"

# (display name, output file stem, NCBI taxon to search, role)
# Genus-only targets use a representative species genome.
SPECIES = [
    ("Lactobacillus jensenii",  "Lactobacillus_jensenii",  "Lactobacillus jensenii",  "spare"),
    ("Lactobacillus crispatus", "Lactobacillus_crispatus", "Lactobacillus crispatus", "spare"),
    ("Lactobacillus iners",     "Lactobacillus_iners",     "Lactobacillus iners",     "spare"),
    ("Gardnerella vaginalis",   "Gardnerella_vaginalis",   "Gardnerella vaginalis",   "kill"),
    ("Prevotella bivia",        "Prevotella_bivia",        "Prevotella bivia",        "kill"),
    ("Fannyhessea vaginae",     "Fannyhessea_vaginae",     "Fannyhessea vaginae",     "kill"),
    ("Mobiluncus spp.",         "Mobiluncus",              "Mobiluncus curtisii",     "kill"),
    ("Megasphaera spp.",        "Megasphaera",             "Megasphaera lornae",      "kill"),
    ("Sneathia spp.",           "Sneathia",                "Sneathia amnii",          "kill"),
]

# Preference order for assembly level (best first).
LEVEL_RANK = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}


def _curl(url, dest=None, accept=None):
    """Run curl.exe. Returns bytes (dest=None) or writes to dest and returns b''."""
    cmd = ["curl.exe", "-sS", "-L", "--fail", "--retry", "3", "--retry-delay", "2"]
    if accept:
        cmd += ["-H", "Accept: " + accept]
    if dest:
        cmd += ["-o", dest]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "curl failed (%d) for %s\n%s"
            % (proc.returncode, url, proc.stderr.decode("utf-8", "replace").strip())
        )
    return proc.stdout


def resolve_accession(taxon):
    """Pick the best RefSeq assembly accession for a taxon, or None."""
    taxon_url = taxon.replace(" ", "%20")
    url = (
        "%s/genome/taxon/%s/dataset_report"
        "?filters.assembly_source=RefSeq&page_size=200"
        % (API, taxon_url)
    )
    data = json.loads(_curl(url, accept="application/json").decode("utf-8"))
    reports = data.get("reports") or []
    if not reports:
        return None, None

    def sort_key(rep):
        info = rep.get("assembly_info", {})
        level = info.get("assembly_level", "")
        is_ref = info.get("refseq_category", "") in ("reference genome", "representative genome")
        acc = rep.get("accession", "")
        is_gcf = acc.startswith("GCF_")
        return (
            0 if is_ref else 1,
            LEVEL_RANK.get(level, 9),
            0 if is_gcf else 1,
        )

    best = sorted(reports, key=sort_key)[0]
    return best.get("accession"), best.get("assembly_info", {}).get("assembly_level")


def download_fasta(accession, out_fna):
    """Download GENOME_FASTA zip for an accession and extract the .fna to out_fna."""
    url = (
        "%s/genome/accession/%s/download?include_annotation_type=GENOME_FASTA"
        % (API, accession)
    )
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "genome.zip")
        _curl(url, dest=zip_path, accept="application/zip")
        with zipfile.ZipFile(zip_path) as zf:
            fnas = [n for n in zf.namelist() if n.lower().endswith((".fna", ".fasta", ".fa"))]
            if not fnas:
                raise RuntimeError("no FASTA inside zip for %s" % accession)
            with zf.open(fnas[0]) as src, open(out_fna, "wb") as dst:
                shutil.copyfileobj(src, dst)


def direct_url(accession):
    return (
        "%s/genome/accession/%s/download?include_annotation_type=GENOME_FASTA"
        % (API, accession)
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, stem, taxon, role in SPECIES:
        out_fna = os.path.join(OUT_DIR, stem + ".fna")
        rel = os.path.relpath(out_fna, ROOT).replace("\\", "/")

        if os.path.exists(out_fna) and os.path.getsize(out_fna) > 0:
            print("[skip] %-26s already present -> %s" % (name, rel))
            rows.append((name, "(pre-existing)", "", role, rel, "OK"))
            continue

        try:
            accession, level = resolve_accession(taxon)
        except Exception as exc:  # noqa: BLE001
            print("[FAIL] %-26s report lookup failed: %s" % (name, exc))
            rows.append((name, "", "", role, rel, "MISSING (lookup failed)"))
            continue

        if not accession:
            print("[FAIL] %-26s no RefSeq assembly found for taxon '%s'" % (name, taxon))
            rows.append((name, "", "", role, rel, "MISSING (no assembly)"))
            continue

        try:
            download_fasta(accession, out_fna)
            size = os.path.getsize(out_fna)
            print("[ok]   %-26s %s (%s, %.1f MB) -> %s"
                  % (name, accession, level or "?", size / 1e6, rel))
            rows.append((name, accession, level or "", role, rel, "OK"))
        except Exception as exc:  # noqa: BLE001
            print("[FAIL] %-26s download failed for %s: %s" % (name, accession, exc))
            print("       manual URL: %s" % direct_url(accession))
            rows.append((name, accession, level or "", role, rel, "MISSING (download failed)"))

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "accession", "assembly_level", "role", "file", "status"])
        w.writerows(rows)

    ok = sum(1 for r in rows if r[5] == "OK")
    print("\nManifest written: %s" % os.path.relpath(MANIFEST, ROOT).replace("\\", "/"))
    print("Genomes available: %d / %d" % (ok, len(rows)))
    missing = [r for r in rows if r[5] != "OK"]
    if missing:
        print("\nMissing (drop the .fna into data/Genomic_featrues/ manually):")
        for r in missing:
            hint = ("accession %s -> %s" % (r[1], direct_url(r[1]))) if r[1] else "resolve accession on NCBI"
            print("  - %-26s %s" % (r[0], hint))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
