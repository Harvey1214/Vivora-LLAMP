"""Validation runner mirroring the genome-route cells of Inference.ipynb.

Reproduces the notebook logic outside Jupyter (nbconvert is unavailable in this env)
to confirm the genome-based screen runs end-to-end and to generate the output CSVs.
Uses the locally cached ESM-2 weights (no network needed).
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import model as model_mod
from utils import dataset
from utils import utils as _utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LLAMP_model = model_mod.LLAMP(hidden_feat=256, pooling='CLS')
LLAMP_model.load_state_dict(torch.load('model_weight/LLAMP.pth', map_location=device), strict=False)
LLAMP_model.to(device)
LLAMP_model.eval()

genome_feat_dict = torch.load('data/Genomic_featrues/genome_features.pt', map_location=device)

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
shortlist = pd.read_csv('shortlist.csv').dropna(subset=['Sequence']).copy()
shortlist['Sequence'] = shortlist['Sequence'].astype(str).str.strip().str.upper()
shortlist = shortlist[shortlist['Sequence'].apply(lambda s: len(s) > 0 and set(s) <= STANDARD_AA)]
shortlist = shortlist.reset_index(drop=True)


def predict_mic(seq_list, species_list, batch_size=32):
    seq_list = list(seq_list)
    feats = dataset.get_features(list(species_list), genome_feat_dict)
    loader = dataset.data_loader(seq_list, feats, torch.zeros(len(seq_list)), BATCH_SIZE=batch_size)
    LLAMP_model.eval()
    out = []
    with torch.no_grad():
        for input_ids, attention_mask, genome_feat, _ in loader:
            p = LLAMP_model(input_ids.to(device), attention_mask.to(device), genome_feat.to(device))
            out.append(p.detach().cpu().reshape(-1))
    return torch.cat(out).numpy()


real_targets = pd.DataFrame([
    ('Lactobacillus jensenii',  'data/Genomic_featrues/Lactobacillus_jensenii.fna',  'spare'),
    ('Lactobacillus crispatus', 'data/Genomic_featrues/Lactobacillus_crispatus.fna', 'spare'),
    ('Lactobacillus iners',     'data/Genomic_featrues/Lactobacillus_iners.fna',     'spare'),
    ('Gardnerella vaginalis',   'data/Genomic_featrues/Gardnerella_vaginalis.fna',   'kill'),
    ('Prevotella bivia',        'data/Genomic_featrues/Prevotella_bivia.fna',        'kill'),
    ('Fannyhessea vaginae',     'data/Genomic_featrues/Fannyhessea_vaginae.fna',     'kill'),
    ('Mobiluncus spp.',         'data/Genomic_featrues/Mobiluncus.fna',              'kill'),
    ('Megasphaera spp.',        'data/Genomic_featrues/Megasphaera.fna',             'kill'),
    ('Sneathia spp.',           'data/Genomic_featrues/Sneathia.fna',                'kill'),
], columns=['real_species', 'fasta', 'role'])


def _block_norms(v):
    return [float(np.linalg.norm(v[a:b])) for a, b in [(0, 4), (4, 20), (20, 84), (84, 340)]]


for _, r in real_targets.iterrows():
    feat = np.asarray(_utils.get_genome_feature_from_file(r['fasta']), dtype=np.float64)
    assert feat.shape == (340,), f"{r['real_species']}: {feat.shape}"
    assert all(abs(n - 1.0) < 1e-6 for n in _block_norms(feat)), r['real_species']
    genome_feat_dict[r['real_species']] = [feat]

real_species = real_targets['real_species'].tolist()
grid_g = shortlist[['Sequence']].merge(pd.Series(real_species, name='real_species'), how='cross')
grid_g['mic_log_uM'] = predict_mic(grid_g['Sequence'], grid_g['real_species'])
grid_g['mic_uM'] = 10 ** grid_g['mic_log_uM']
results_genome = grid_g.merge(real_targets[['real_species', 'role']], on='real_species', how='left')
results_genome = results_genome[['Sequence', 'real_species', 'role', 'mic_log_uM', 'mic_uM']]

commensal_mic_g = results_genome[results_genome.role == 'spare'].groupby('Sequence')['mic_uM'].min()
bv_mic_g = results_genome[results_genome.role == 'kill'].groupby('Sequence')['mic_uM'].max()
summary_genome = pd.DataFrame({'commensal_mic_uM': commensal_mic_g, 'bv_mic_uM': bv_mic_g})
summary_genome['selectivity_index'] = summary_genome['commensal_mic_uM'] / summary_genome['bv_mic_uM']
summary_genome = summary_genome.merge(shortlist[['Sequence', 'Novelty', 'pct']], on='Sequence', how='left')
summary_genome = summary_genome.sort_values('selectivity_index', ascending=False).reset_index(drop=True)

rng = random.Random(42)
def _scramble(seq):
    chars = list(seq)
    rng.shuffle(chars)
    return ''.join(chars)

bv_genome = 'Gardnerella vaginalis'
top_seqs_g = summary_genome.head(5)['Sequence'].tolist()
scr_g = pd.DataFrame({'Sequence': top_seqs_g})
scr_g['scrambled'] = scr_g['Sequence'].apply(_scramble)
scr_g['orig_MIC_uM'] = 10 ** predict_mic(scr_g['Sequence'], [bv_genome] * len(scr_g))
scr_g['scrambled_MIC_uM'] = 10 ** predict_mic(scr_g['scrambled'], [bv_genome] * len(scr_g))
scr_g['pass'] = scr_g['orig_MIC_uM'] < scr_g['scrambled_MIC_uM']

phys_cols = ['Charge at pH 7', 'Charge Density', 'Hydrophobic Moment',
             'Gravy Score', 'Aromaticity', 'Aliphatic Index']
phys_cols = [c for c in phys_cols if c in shortlist.columns]
bv_log_g = (results_genome[results_genome.role == 'kill'].groupby('Sequence')['mic_log_uM']
            .mean().rename('bv_mic_log_uM'))
corr_df_g = shortlist.merge(bv_log_g, on='Sequence')
corr_genome = (corr_df_g[phys_cols + ['bv_mic_log_uM']].corr(method='spearman')['bv_mic_log_uM']
               .drop('bv_mic_log_uM').sort_values())

os.makedirs('data/M3CAD', exist_ok=True)
results_genome.to_csv('data/M3CAD/mic_predictions_genome.csv', index=False)
summary_genome.to_csv('data/M3CAD/selectivity_ranking_genome.csv', index=False)

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print('=== Top 10 by genome-based selectivity ===')
print(summary_genome.head(10).to_string(index=False))
print('\n=== Scramble control vs real Gardnerella vaginalis ===')
print(scr_g.to_string(index=False))
print('\n=== Spearman descriptor vs genome BV MIC (log uM) ===')
print(corr_genome.to_string())
print('\nScramble pass rate: %d/%d' % (int(scr_g['pass'].sum()), len(scr_g)))
print('Wrote data/M3CAD/mic_predictions_genome.csv and selectivity_ranking_genome.csv')
