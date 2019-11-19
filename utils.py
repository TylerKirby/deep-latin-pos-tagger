import json

from torch.utils.data import Dataset


class ProielDataset(Dataset):
    def __init__(self, file, label='pos'):
        self.file = file
        self.label = label
        self.X, self.y, self.vocab_mapping, self.label_mapping = self.read_json()

    def read_json(self):
        if self.label not in ['pos', 'morph']:
            raise Exception(f'{self.label} is not a valid option')

        with open(self.file) as f:
            data = json.load(f)
        sentences = data['sentences']

        vocabulary = list(set([t['word'].lower() for s in sentences for t in s]))
        vocab_mapping = {v: k for k, v in enumerate(vocabulary)}

        labels = list(set([t[self.label].lower() for s in sentences for t in s]))
        label_mapping = {v: k for k, v in enumerate(labels)}

        X = [[vocab_mapping[t['word'].lower()] for t in s] for s in sentences]
        y = [[label_mapping[t[self.label].lower()] for t in s] for s in sentences]

        return X, y, vocab_mapping, label_mapping

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

        
    
