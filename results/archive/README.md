# Historical results

These CSVs are preserved outputs from workflows that predate `main.py`. Their
filenames and schemas vary. New runs write the documented schema to
`results/current/` or to the directory supplied with `--output`.

Do not use these files for current triage decisions.

Notable caveats:

- `mic_predictions.csv` / `selectivity_ranking.csv` are the old **proxy** route
  (22 peptides). Shared taxonomic proxies and two Gardnerella proxy rows per
  peptide can bias typical-case means.
- `selectivity_ranking_genome.csv` / `mic_predictions_genome.csv` are the older
  single-genome-per-species screen, not the current strain panel.
- `selectivity_ranking_genome_strain_panel.csv` and the
  `mic_predictions_genome_strains.csv` /
  `mic_predictions_genome_species.csv` pair are closer to the current panel
  schema but may still predate columns such as
  `min_spare_over_typical_kill` and `bv_potency_warning`.
- `Selectivity_Score` in older ranking CSVs is upstream candidate metadata, not
  the LLAMP selectivity index.
