"""Command-line entry point for the LLAMP strain-level genome screen.

Examples:
    python main.py genomes
    python main.py screen
    python main.py screen --candidates path/to/candidates.csv
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PANEL = ROOT / "config" / "species_panel.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "candidates" / "current.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "current"
DEFAULT_MODEL_WEIGHT = ROOT / "model_weight" / "LLAMP.pth"
DEFAULT_GENOME_FEATURES = ROOT / "data" / "Genomic_features" / "genome_features.pt"

NCBI_DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2"
PANEL_COLUMNS = {
    "llamp_id",
    "species",
    "strain",
    "accession",
    "source",
    "assembly_level",
    "role",
    "confidence",
    "file",
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
META_COLUMNS = ["Novelty", "pct", "Selectivity_Score", "Length"]


def project_path(value):
    """Resolve a CLI path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_panel(panel_path):
    """Read and validate the pinned strain panel."""
    with panel_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Species panel is empty: {panel_path}")
    missing = PANEL_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Species panel is missing columns: {sorted(missing)}")

    ids = [row["llamp_id"] for row in rows]
    files = [row["file"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Species panel contains duplicate llamp_id values")
    if len(files) != len(set(files)):
        raise ValueError("Each panel row must use a distinct genome file")
    return rows


def run_curl(url, destination, accept):
    """Download a URL with curl, using the Windows certificate store when applicable."""
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl is required to download genomes")
    command = [
        executable,
        "-sS",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-H",
        f"Accept: {accept}",
        "-o",
        str(destination),
        url,
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"curl failed for {url}: {error}")


def download_assembly(accession, destination):
    """Download and extract one NCBI Assembly genome FASTA."""
    url = (
        f"{NCBI_DATASETS_API}/genome/accession/{accession}/download"
        "?include_annotation_type=GENOME_FASTA"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "genome.zip"
        run_curl(url, archive, "application/zip")
        with zipfile.ZipFile(archive) as zipped:
            fasta_names = [
                name
                for name in zipped.namelist()
                if name.lower().endswith((".fna", ".fasta", ".fa"))
            ]
            if not fasta_names:
                raise RuntimeError(f"No FASTA found in NCBI archive for {accession}")
            with zipped.open(fasta_names[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def download_nucleotide(accession, destination):
    """Download one NCBI Nucleotide FASTA."""
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    )
    run_curl(url, destination, "text/plain")


def setup_genomes(panel_path):
    """Download genome files missing from the pinned panel."""
    rows = read_panel(panel_path)
    failures = []
    for row in rows:
        destination = project_path(row["file"])
        label = f'{row["species"]} ({row["strain"]})'
        if destination.exists() and destination.stat().st_size:
            print(f"[skip] {label}: {destination.relative_to(ROOT)}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if row["source"] == "NCBI Nucleotide":
                download_nucleotide(row["accession"], destination)
            else:
                download_assembly(row["accession"], destination)
            if not destination.stat().st_size:
                raise RuntimeError("Downloaded file is empty")
            print(f"[ok]   {label}: {row['accession']}")
        except Exception as error:  # noqa: BLE001 - report every failed genome
            destination.unlink(missing_ok=True)
            failures.append((label, str(error)))
            print(f"[fail] {label}: {error}")

    print(f"\nGenomes available: {len(rows) - len(failures)} / {len(rows)}")
    if failures:
        raise RuntimeError(f"{len(failures)} genome download(s) failed")


def load_candidates(pd, candidates_path):
    """Load valid standard-amino-acid peptide sequences."""
    candidates = pd.read_csv(candidates_path)
    if "Sequence" not in candidates.columns:
        raise ValueError(f"Candidate CSV needs a Sequence column: {candidates_path}")

    original_count = len(candidates)
    candidates = candidates.dropna(subset=["Sequence"]).copy()
    candidates["Sequence"] = (
        candidates["Sequence"].astype(str).str.strip().str.upper()
    )
    candidates = candidates[
        candidates["Sequence"].map(
            lambda sequence: bool(sequence) and set(sequence) <= STANDARD_AA
        )
    ].drop_duplicates(subset=["Sequence"])
    candidates = candidates.reset_index(drop=True)
    if candidates.empty:
        raise ValueError("Candidate CSV contains no valid peptide sequences")
    print(
        f"Loaded {len(candidates)} valid unique candidates from {candidates_path} "
        f"({original_count - len(candidates)} rows removed)"
    )
    return candidates


def load_targets(pd, panel_path):
    """Load panel metadata and verify that all FASTAs exist."""
    targets = pd.DataFrame(read_panel(panel_path))
    missing = targets[~targets["file"].map(lambda value: project_path(value).exists())]
    if not missing.empty:
        names = ", ".join(missing["llamp_id"])
        raise FileNotFoundError(
            f"Missing {len(missing)} genome FASTA(s): {names}\n"
            "Run `python main.py genomes` first."
        )
    print(
        f"Loaded {len(targets)} genomes across {targets['species'].nunique()} species "
        f"from {panel_path}"
    )
    return targets


def choose_device(torch, requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def predict_mic(torch, dataset_module, model, device, feature_dictionary,
                sequences, target_ids, batch_size):
    """Predict log10 MIC for parallel sequence and target-ID lists."""
    sequences = list(sequences)
    features = dataset_module.get_features(list(target_ids), feature_dictionary)
    loader = dataset_module.data_loader(
        sequences,
        features,
        torch.zeros(len(sequences)),
        BATCH_SIZE=batch_size,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, genome_features, _ in loader:
            values = model(
                input_ids.to(device),
                attention_mask.to(device),
                genome_features.to(device),
            )
            predictions.append(values.detach().cpu().reshape(-1))
    return torch.cat(predictions).numpy()


def inject_genome_features(np, genome_utils, targets, feature_dictionary):
    """Calculate and register the 340-value feature vector for every assembly."""
    blocks = [(0, 4), (4, 20), (20, 84), (84, 340)]
    for row in targets.itertuples(index=False):
        feature = np.asarray(
            genome_utils.get_genome_feature_from_file(project_path(row.file)),
            dtype=np.float64,
        )
        if feature.shape != (340,):
            raise ValueError(f"{row.llamp_id}: expected 340 features, got {feature.shape}")
        norms = [float(np.linalg.norm(feature[start:end])) for start, end in blocks]
        if any(abs(norm - 1.0) >= 1e-6 for norm in norms):
            raise ValueError(f"{row.llamp_id}: invalid normalized block(s): {norms}")
        feature_dictionary[row.llamp_id] = [feature]


def aggregate_predictions(pd, results, candidates):
    """Aggregate strains by species and calculate candidate selectivity."""
    by_species = (
        results.groupby(["Sequence", "species", "role"], as_index=False)
        .agg(
            strain_count=("accession", "nunique"),
            strain_min_mic_uM=("mic_uM", "min"),
            strain_max_mic_uM=("mic_uM", "max"),
            species_typical_mic_log_uM=("mic_log_uM", "mean"),
        )
    )
    by_species["species_typical_mic_uM"] = (
        10 ** by_species["species_typical_mic_log_uM"]
    )

    spare = by_species[by_species["role"] == "spare"]
    core = by_species[by_species["role"] == "core"]
    if spare.empty or core.empty:
        raise ValueError("Panel must contain both spare and core roles")

    ranking = pd.DataFrame({
        "lactobacillus_worst_case_uM": (
            spare.groupby("Sequence")["strain_min_mic_uM"].min()
        ),
        "bv_worst_case_uM": (
            core.groupby("Sequence")["strain_max_mic_uM"].max()
        ),
        "lactobacillus_typical_case_uM": (
            10 ** spare.groupby("Sequence")["species_typical_mic_log_uM"].mean()
        ),
        "bv_typical_case_uM": (
            10 ** core.groupby("Sequence")["species_typical_mic_log_uM"].mean()
        ),
    })
    ranking["selectivity_worst_case"] = (
        ranking["lactobacillus_worst_case_uM"] / ranking["bv_worst_case_uM"]
    )
    ranking["selectivity_typical_case"] = (
        ranking["lactobacillus_typical_case_uM"] / ranking["bv_typical_case_uM"]
    )
    metadata = [column for column in META_COLUMNS if column in candidates.columns]
    if metadata:
        ranking = ranking.merge(
            candidates[["Sequence"] + metadata], on="Sequence", how="left"
        )
    ranking = ranking.sort_values(
        "selectivity_worst_case", ascending=False
    ).reset_index(drop=True)
    return by_species, ranking, core


def run_controls(pd, torch, dataset_module, model, device, feature_dictionary,
                 native_features, targets, candidates, ranking, core, batch_size):
    """Run scramble, descriptor-correlation, and in-distribution anchor controls."""
    gardnerella = targets[targets["species"] == "Gardnerella vaginalis"]
    if gardnerella.empty:
        raise ValueError("Panel needs Gardnerella vaginalis for the scramble control")

    generator = random.Random(42)

    def scramble_sequence(sequence):
        characters = list(sequence)
        generator.shuffle(characters)
        return "".join(characters)

    scramble = pd.DataFrame({"Sequence": ranking.head(5)["Sequence"]})
    scramble["scrambled"] = scramble["Sequence"].map(scramble_sequence)
    target_id = gardnerella.iloc[0]["llamp_id"]
    repeated_target = [target_id] * len(scramble)
    scramble["orig_mic_uM"] = 10 ** predict_mic(
        torch, dataset_module, model, device, feature_dictionary,
        scramble["Sequence"], repeated_target, batch_size,
    )
    scramble["scrambled_mic_uM"] = 10 ** predict_mic(
        torch, dataset_module, model, device, feature_dictionary,
        scramble["scrambled"], repeated_target, batch_size,
    )
    scramble["pass"] = scramble["orig_mic_uM"] < scramble["scrambled_mic_uM"]

    descriptor_names = [
        "Charge at pH 7", "Charge_pH7", "Charge Density", "Charge_Density",
        "Hydrophobic Moment", "Hydrophobic_Moment", "Gravy Score", "Gravy",
        "Aromaticity", "Aliphatic Index", "Aliphatic_Index",
    ]
    descriptor_names = [
        name for name in descriptor_names if name in candidates.columns
    ]
    correlations = pd.DataFrame(columns=["descriptor", "spearman"])
    if descriptor_names:
        bv_mic = (
            core.groupby("Sequence")["species_typical_mic_log_uM"]
            .mean()
            .rename("bv_mic_log_uM")
        )
        correlation_input = candidates.merge(bv_mic, on="Sequence")
        values = (
            correlation_input[descriptor_names + ["bv_mic_log_uM"]]
            .corr(method="spearman")["bv_mic_log_uM"]
            .drop("bv_mic_log_uM")
            .sort_values()
        )
        correlations = values.rename("spearman").rename_axis("descriptor").reset_index()

    calibration_parts = []
    sequences = candidates["Sequence"].tolist()
    for anchor in targets[targets["role"] == "calibration"].itertuples(index=False):
        if anchor.species not in native_features:
            print(f"[warn] Stored LLAMP feature missing for anchor: {anchor.species}")
            continue
        derived = predict_mic(
            torch, dataset_module, model, device, feature_dictionary,
            sequences, [anchor.llamp_id] * len(sequences), batch_size,
        )
        native = predict_mic(
            torch, dataset_module, model, device, native_features,
            sequences, [anchor.species] * len(sequences), batch_size,
        )
        part = pd.DataFrame({
            "Sequence": sequences,
            "species": anchor.species,
            "accession": anchor.accession,
            "derived_mic_log_uM": derived,
            "native_mic_log_uM": native,
        })
        part["delta_log_uM"] = (
            part["derived_mic_log_uM"] - part["native_mic_log_uM"]
        )
        calibration_parts.append(part)
    calibration = (
        pd.concat(calibration_parts, ignore_index=True)
        if calibration_parts
        else pd.DataFrame()
    )
    return scramble, correlations, calibration


def run_screen(args):
    """Execute the complete genome-conditioned screen."""
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import pandas as pd
    import torch

    from utils import dataset as dataset_module
    from utils import model as model_module
    from utils import utils as genome_utils

    panel_path = project_path(args.panel)
    candidates_path = project_path(args.candidates)
    output_dir = project_path(args.output)
    model_weight = project_path(args.model_weight)
    genome_features_path = project_path(args.genome_features)

    for required_path in (panel_path, candidates_path, model_weight, genome_features_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    candidates = load_candidates(pd, candidates_path)
    targets = load_targets(pd, panel_path)
    device = choose_device(torch, args.device)
    print(f"Using device: {device}")

    model = model_module.LLAMP(hidden_feat=256, pooling="CLS")
    state = torch.load(model_weight, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    feature_dictionary = torch.load(genome_features_path, map_location=device)
    native_features = dict(feature_dictionary)
    inject_genome_features(np, genome_utils, targets, feature_dictionary)

    target_ids = targets["llamp_id"].tolist()
    prediction_grid = candidates[["Sequence"]].merge(
        pd.Series(target_ids, name="llamp_id"), how="cross"
    )
    prediction_grid["mic_log_uM"] = predict_mic(
        torch,
        dataset_module,
        model,
        device,
        feature_dictionary,
        prediction_grid["Sequence"],
        prediction_grid["llamp_id"],
        args.batch_size,
    )
    prediction_grid["mic_uM"] = 10 ** prediction_grid["mic_log_uM"]
    metadata = [
        "llamp_id", "species", "strain", "accession", "source",
        "assembly_level", "role", "confidence",
    ]
    results = prediction_grid.merge(targets[metadata], on="llamp_id", how="left")
    results = results[["Sequence"] + metadata + ["mic_log_uM", "mic_uM"]]

    by_species, ranking, core = aggregate_predictions(pd, results, candidates)
    scramble, correlations, calibration = run_controls(
        pd,
        torch,
        dataset_module,
        model,
        device,
        feature_dictionary,
        native_features,
        targets,
        candidates,
        ranking,
        core,
        args.batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions_by_strain.csv", index=False)
    by_species.to_csv(output_dir / "predictions_by_species.csv", index=False)
    ranking.to_csv(output_dir / "candidate_ranking.csv", index=False)
    scramble.to_csv(output_dir / "scramble_control.csv", index=False)
    correlations.to_csv(output_dir / "descriptor_correlations.csv", index=False)
    if not calibration.empty:
        calibration.to_csv(output_dir / "calibration.csv", index=False)

    columns = [
        "Sequence",
        "lactobacillus_worst_case_uM",
        "bv_worst_case_uM",
        "selectivity_worst_case",
        "selectivity_typical_case",
    ]
    print("\nTop candidates by worst-case selectivity:")
    print(ranking[columns].head(10).to_string(index=False))
    print(f"\nScramble control: {int(scramble['pass'].sum())}/{len(scramble)} pass")
    print(f"Results written to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Set up and run the LLAMP strain-level genome screen."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    genomes = subparsers.add_parser(
        "genomes", help="download genome FASTAs missing from the pinned panel"
    )
    genomes.add_argument("--panel", default=str(DEFAULT_PANEL.relative_to(ROOT)))

    screen = subparsers.add_parser(
        "screen", help="predict MIC and rank candidate peptides"
    )
    screen.add_argument(
        "--candidates", default=str(DEFAULT_CANDIDATES.relative_to(ROOT))
    )
    screen.add_argument("--panel", default=str(DEFAULT_PANEL.relative_to(ROOT)))
    screen.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    screen.add_argument(
        "--model-weight", default=str(DEFAULT_MODEL_WEIGHT.relative_to(ROOT))
    )
    screen.add_argument(
        "--genome-features", default=str(DEFAULT_GENOME_FEATURES.relative_to(ROOT))
    )
    screen.add_argument("--batch-size", type=int, default=32)
    screen.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    screen.add_argument(
        "--online",
        action="store_true",
        help="allow Transformers to download the peptide-tuned ESM-2 model",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "genomes":
        setup_genomes(project_path(args.panel))
    else:
        run_screen(args)


if __name__ == "__main__":
    main()
