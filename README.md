# LLAMP vaginal peptide screen

This project uses the species-aware LLAMP model to predict peptide MICs against
a pinned panel of vaginal commensals, BV-associated organisms, secondary
pathogens, and calibration species.

`main.py` is the only supported entry point. The notebook in
`examples/legacy/` is retained for historical reference and is not part of the
current workflow.

Predictions for organisms outside LLAMP's training distribution are screening
signals for prioritization, not experimentally validated MIC measurements.

## Project layout

```text
main.py                         Main command-line program
config/species_panel.csv        Species, strains, accessions, and analysis roles
data/candidates/current.csv     Default peptide input
data/Genomic_features/          Genome FASTAs and LLAMP's stored genome features
model_weight/LLAMP.pth          LLAMP model weight (download separately)
results/current/                Results from the latest run
results/archive/                Preserved historical result CSVs
examples/legacy/Inference.ipynb Archived exploratory notebook
utils/                          LLAMP model, tokenizer, and genome feature code
```

## Installation

Python 3.9 is recommended.

```bash
conda create -n LLAMP python=3.9.13
conda activate LLAMP
pip install -r requirements.txt
```

Download the LLAMP weight and save it as `model_weight/LLAMP.pth`:

https://drive.google.com/file/d/1hmfL7uRZsHo4pn0o0nqaqPcntxGJIhE7/view

The peptide-tuned ESM-2 files are obtained from:

https://huggingface.co/Daehun/peptide_tuned_ESM-2

The screen runs offline by default, so ESM-2 must already be in the Hugging Face
cache. On the first run, pass `--online` to allow Transformers to download it.

## Quick start

From the repository root:

```bash
# Download only genomes that are not already present
python main.py genomes

# Run the default candidate file against the complete panel
python main.py screen
```

On Windows, if `python` does not select the LLAMP environment:

```powershell
& "C:\Users\<user>\anaconda3\envs\LLAMP\python.exe" "main.py" screen
```

Run `python main.py --help` or `python main.py screen --help` for all options.
Useful examples:

```bash
python main.py screen --candidates my_candidates.csv
python main.py screen --device cpu --batch-size 16
python main.py screen --output results/experiment_01
```

## Candidate input

The candidate CSV must contain a `Sequence` column. Sequences are converted to
uppercase; empty, duplicate, or non-standard amino-acid sequences are removed.
Other columns are preserved when relevant. The default input is:

`data/candidates/current.csv`

Use `--candidates PATH` to screen another file.

## Species panel

All tested organisms are specified explicitly in:

`config/species_panel.csv`

Each row is one genome presented to LLAMP as a distinct target. Important
columns are:

- `species`, `strain`, `accession`: biological and NCBI identity;
- `role`: how the organism is used in analysis;
- `confidence`: interpretation warning for unusual genomes or MAGs;
- `file`: expected local FASTA path.

Roles:

- `spare`: protective Lactobacillus species; high MIC is preferred;
- `core`: BV-associated targets used in the selectivity score;
- `exploratory`: highly out-of-distribution organisms reported separately;
- `secondary`: successor/opportunistic pathogens reported separately;
- `calibration`: in-distribution species used to verify feature extraction.

To change the panel, edit the CSV using exact NCBI accessions, then run
`python main.py genomes`. Multiple strains of one species are aggregated before
the species-level score, so they do not receive extra weight.

## Results

`python main.py screen` writes these files to `results/current/`:

- `predictions_by_strain.csv`: every peptide × genome prediction;
- `predictions_by_species.csv`: strain predictions aggregated by species;
- `candidate_ranking.csv`: final selectivity ranking;
- `scramble_control.csv`: sequence-scrambling plausibility control;
- `descriptor_correlations.csv`: descriptor correlations when available;
- `calibration.csv`: derived-versus-native feature comparison for anchors.

MIC predictions are in µM and `mic_log_uM` is log10(µM).

The worst-case selectivity score is:

```text
lowest MIC among must-spare Lactobacillus strains
-------------------------------------------------
highest MIC among core BV target strains
```

Values above 1 indicate the desired direction, but do not establish biological
selectivity without wet-lab MIC measurements.

## License

Code is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

