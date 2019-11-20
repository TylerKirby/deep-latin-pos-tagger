import argparse

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import ProielDataset


class StandardRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 embedding_dim=64,
                 hidden_size=64,
                 num_layers=5,
                 dropout=0,
                 bidirectional=False):
        super(StandardRNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)


if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validation_split', default=0.2)
    parser.add_argument('-bs', '--batch_size', default=128)
    parser.add_argument('-e', '--epochs', default=10)
    parser.add_argument('-ms', '--max_sentence_size', default=128)
    args = parser.parse_args()

    VALIDATION_SPLIT = args.validation_split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MAX_SENTENCE_SIZE = args.max_sentence_size

    dataset = ProielDataset('proiel.json', max_sent_size=MAX_SENTENCE_SIZE)
    dataset_size = len(dataset)
    vocab_size = len(dataset.vocab_mapping)
    tag_size = len(dataset.label_mapping)

    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print(f'Training Set Size: {len(train_indices)}')
    print(f'Validation Set Size: {len(val_indices)}')

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    
    model = StandardRNN(vocab_size, tag_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    for e in range(EPOCHS):
        for (sentences, labels) in (train_loader):
            y_hat = model(sentences).view(-1, tag_size)
            labels = labels.view(-1)
            loss = criterion(y_hat, labels)

            print(loss)
            loss.backward()
            optimizer.step()
