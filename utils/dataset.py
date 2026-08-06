import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from transformers import EsmTokenizer


tokenizer = EsmTokenizer.from_pretrained('Daehun/peptide_tuned_ESM-2')


def sequence_to_input(sequences):
    """Tokenize peptide sequences with one amino acid per ESM token."""
    spaced_sequences = [' '.join(sequence) for sequence in sequences]
    inputs = tokenizer.batch_encode_plus(
        spaced_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
    )
    return inputs['input_ids'], inputs['attention_mask']


def data_loader(sequences, genome_features, labels, BATCH_SIZE,
                NUM_THREADS=0, shuffle=False):
    """Build the inference/training DataLoader expected by LLAMP."""
    input_ids, attention_mask = sequence_to_input(sequences)
    tensors = TensorDataset(input_ids, attention_mask, genome_features, labels)
    return DataLoader(
        dataset=tensors,
        batch_size=BATCH_SIZE,
        num_workers=NUM_THREADS,
        shuffle=shuffle,
    )


def get_features(target_ids, feature_dictionary):
    """Return the first stored genome feature vector for each target ID."""
    encoded = [
        torch.as_tensor(feature_dictionary[target_id][0], dtype=torch.float32)
        for target_id in target_ids
    ]
    return torch.stack(encoded)