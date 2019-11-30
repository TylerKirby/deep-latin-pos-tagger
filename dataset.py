import json

import numpy as np
from torch.utils.data import Dataset


class ProielDataset(Dataset):
    def __init__(self, file, label='pos'):
        self.file = file
        self.label = label
        self.X, self.X_lengths, self.y, self.y_lengths, self.vocab_mapping, self.label_mapping = self.read_json()

    def read_json(self):
        if self.label not in ['pos', 'morph']:
            raise Exception(f'{self.label} is not a valid option')

        with open(self.file) as f:
            data = json.load(f)
        sentences = [s for s in data['sentences']]

        vocabulary = list(set([t['word'].lower() for s in sentences for t in s]))
        vocab_mapping = {v: k+1 for k, v in enumerate(vocabulary)}
        vocab_mapping['pad_token'] = 0

        labels = list(set([t[self.label].lower() for s in sentences for t in s]))
        label_mapping = {v: k+1 for k, v in enumerate(labels)}
        label_mapping['pad_token'] = 0

        X_lengths = [len(s) for s in sentences]
        y_lengths = [len(s) for s in sentences]

        max_token_size = max(X_lengths)

        X = [self.pad_data([vocab_mapping[t['word'].lower()] for t in s], max_token_size) for s in sentences]
        y = [self.pad_data([label_mapping[t[self.label].lower()] for t in s], max_token_size) for s in sentences]

        return X, X_lengths, y, y_lengths, vocab_mapping, label_mapping

    def pad_data(self, s, max_token_size):
        padded = np.zeros((max_token_size,), dtype=np.int64)
        padded[:len(s)] = s
        return padded

    def decode(self, sequence, type='X'):
        if type == 'X':
            return [list(self.vocab_mapping.keys())[list(self.vocab_mapping.values()).index(i)] for i in sequence]
        elif type == 'y':
            return [list(self.label_mapping.keys())[list(self.label_mapping.values()).index(i)] for i in sequence]
        else:
            raise Exception(f'{type} is not a valid option')

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


if __name__ == '__main__':
    ds = ProielDataset('proiel.json')
    print(len(ds))
    print(ds[1])
    print(ds.decode(ds[1][0]))
    print(ds.decode(ds[1][1], type='y'))
    print(ds.X_lengths[1])
    print(ds.y_lengths[1])

        
    
