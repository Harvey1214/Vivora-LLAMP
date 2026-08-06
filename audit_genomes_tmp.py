"""TEMPORARY audit: verify each panel FASTA matches its declared NCBI accession."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "config" / "species_panel.csv"
API = "https://api.ncbi.nlm.nih.gov/datasets/v2"


def curl_json(url):
    p = subprocess.run(
        ["curl.exe", "-sS", "-L", "--fail", "--max-time", "60", "--retry", "3",
         "-H", "Accept: application/json", url],
        capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:200])
    return json.loads(p.stdout.decode("utf-8"))


def local_stats(path):
    """Return (headers, n_contigs, total_bases) for a FASTA."""
    headers, bases = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                headers.append(line[1:].strip())
            else:
                bases += len(line.strip())
    return headers, len(headers), bases


def main():
    rows = list(csv.DictReader(PANEL.open(newline="", encoding="utf-8")))
    out = []
    for row in rows:
        path = ROOT / row["file"]
        rec = {
            "llamp_id": row["llamp_id"],
            "panel_species": row["species"],
            "panel_strain": row["strain"],
            "accession": row["accession"],
            "source": row["source"],
            "panel_level": row["assembly_level"],
            "role": row["role"],
            "confidence": row["confidence"],
        }
        if not path.exists():
            rec["status"] = "MISSING FILE"
            out.append(rec)
            continue

        headers, n_contigs, bases = local_stats(path)
        rec["local_contigs"] = n_contigs
        rec["local_bases"] = bases
        rec["local_first_header"] = headers[0] if headers else ""
        rec["local_seq_accs"] = [h.split()[0] for h in headers]

        try:
            if row["source"] == "NCBI Nucleotide":
                url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                       f"?db=nuccore&id={row['accession']}&retmode=json")
                d = curl_json(url)
                uid = d["result"]["uids"][0]
                r = d["result"][uid]
                rec["ncbi_organism"] = r.get("organism", "")
                rec["ncbi_strain"] = ""
                rec["ncbi_level"] = "sequence record"
                rec["ncbi_bases"] = int(r.get("slen", 0))
                rec["ncbi_contigs"] = 1
                rec["ncbi_seq_accs"] = [r.get("accessionversion", "")]
            else:
                d = curl_json(f"{API}/genome/accession/{row['accession']}/dataset_report")
                r = d["reports"][0]
                org = r.get("organism", {})
                infra = org.get("infraspecific_names", {}) or {}
                stats = r.get("assembly_stats", {}) or {}
                rec["ncbi_accession_returned"] = r.get("accession")
                rec["ncbi_organism"] = org.get("organism_name", "")
                rec["ncbi_strain"] = infra.get("strain") or infra.get("isolate") or ""
                rec["ncbi_level"] = r.get("assembly_info", {}).get("assembly_level", "")
                rec["ncbi_bases"] = int(stats.get("total_sequence_length", 0) or 0)
                rec["ncbi_contigs"] = int(stats.get("number_of_contigs", 0) or 0)
                rec["ncbi_taxid"] = org.get("tax_id")
                # sequence-level accession list for identity proof
                try:
                    sr = curl_json(
                        f"{API}/genome/accession/{row['accession']}/sequence_reports"
                        "?page_size=1000"
                    )
                    accs = [s.get("refseq_accession") or s.get("genbank_accession")
                            for s in sr.get("reports", [])]
                    rec["ncbi_seq_accs"] = [a for a in accs if a]
                except Exception as e:  # noqa: BLE001
                    rec["ncbi_seq_accs"] = []
                    rec["seq_report_error"] = str(e)[:120]
            rec["status"] = "checked"
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"NCBI ERROR: {str(e)[:150]}"
        out.append(rec)
        print(f"[done] {rec['llamp_id']}", file=sys.stderr)

    (ROOT / "audit_genomes_tmp.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    print("wrote audit_genomes_tmp.json")


if __name__ == "__main__":
    main()
