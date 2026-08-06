import random
from itertools import product

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import torch

def set_random_seed(random_seed = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    

def compute_metrics(preds, labels):
    R_squre = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    pearson = float(np.corrcoef(labels, preds)[1,0])
    return {
        'R_squre': R_squre,
        'mae' : mae,
        'mse' : mse,
        'rmse' : rmse,
        'Pearson Q' : pearson,
        # 'Kendall tau' : tau,
    }
       
def count_mer(mers, sequence):
    """Count k-mers using the original LLAMP non-overlapping count behavior."""
    return {mer: sequence.count(mer) for mer in mers}

def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError('Cannot normalize an all-zero genome feature block')
    return x / norm


def get_features(sequences):
    """Calculate LLAMP's normalized mono- through tetra-nucleotide features."""
    nucleotides = ('A', 'T', 'G', 'C')
    mers_by_size = [
        [''.join(values) for values in product(nucleotides, repeat=size)]
        for size in range(1, 5)
    ]
    feature_blocks = [np.zeros(len(mers), dtype=np.float64) for mers in mers_by_size]

    for sequence in sequences:
        for index, mers in enumerate(mers_by_size):
            counts = count_mer(mers, sequence)
            feature_blocks[index] += np.fromiter(counts.values(), dtype=np.float64)

    return np.concatenate([normalize(block) for block in feature_blocks])

def _parse_fasta(file_path):
    """Yield (description, sequence) records from a FASTA file.

    Prefers Biopython when available, otherwise falls back to a dependency-free
    reader so the genome route works even without Bio.SeqIO installed. Both paths
    feed identical sequence strings into get_features, keeping new-species vectors
    constructed the same way as the pre-computed training-species features.
    """
    try:
        from Bio import SeqIO
    except ImportError:
        SeqIO = None

    if SeqIO is not None:
        for seq_record in SeqIO.parse(file_path, 'fasta'):
            yield seq_record.description, str(seq_record.seq)
        return

    description = None
    chunks = []
    with open(file_path, 'r') as fh:
        for line in fh:
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith('>'):
                if description is not None:
                    yield description, ''.join(chunks)
                description = line[1:].strip()
                chunks = []
            elif line:
                chunks.append(line.strip())
    if description is not None:
        yield description, ''.join(chunks)


def get_genome_feature_from_file(file_path):
    genome_seqs = []
    descriptions = []
    for description, seq in _parse_fasta(file_path):
        descriptions.append(description)
        genome_seqs.append(seq.upper())

    return get_features(genome_seqs)
