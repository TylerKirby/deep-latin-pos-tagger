from comet_ml import Experiment

import argparse
import os

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
from allennlp.modules.augmented_lstm import AugmentedLstm
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from dataset import ProielDataset


# experiment = Experiment(
#     api_key=os.getenv('COMET_API_KEY'),
#     project_name='deep-latin-tagger',
#     workspace='tylerkirby'
# )


class BayesianDropoutLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 embedding_dim=64,
                 hidden_size=64,
                 recurrent_dropout_probability=0):
        super(BayesianDropoutLSTM, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.augmented_lstm = AugmentedLstm(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            recurrent_dropout_probability=recurrent_dropout_probability
        )
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.augmented_lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)


class StandardLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 embedding_dim=64,
                 hidden_size=64,
                 num_layers=5,
                 dropout=0,
                 bidirectional=False):
        super(StandardLSTM, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)


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
    # Set random seed
    np.random.seed(1)

    # Experiment parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validation_split', default=0.2)
    parser.add_argument('-bs', '--batch_size', default=128)
    parser.add_argument('-e', '--epochs', default=10)
    parser.add_argument('-ms', '--max_sentence_size', default=128)
    parser.add_argument('-m', '--model', default='standard_rnn')
    parser.add_argument('-lr', '--learning_rate', default=0.0001)
    args = parser.parse_args()

    VALIDATION_SPLIT = args.validation_split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MAX_SENTENCE_SIZE = args.max_sentence_size
    MODEL = args.model
    LEARNING_RATE = args.learning_rate

    # Load data set and create vocabulary
    dataset = ProielDataset('proiel.json', max_sent_size=MAX_SENTENCE_SIZE)
    dataset_size = len(dataset)
    vocab_size = len(dataset.vocab_mapping)
    tag_size = len(dataset.label_mapping)

    # Split data set into training, validation, and test set
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_set_indices = indices[split:], indices[:split]
    val_split = int(np.floor(0.5 * len(val_set_indices)))
    val_indices, test_indices = val_set_indices[val_split:], val_set_indices[:val_split]
    print(f'Training Set Size: {len(train_indices)}')
    print(f'Validation Set Size: {len(val_indices)}')
    print(f'Test Set Size: {len(test_indices)}')

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    # Initialize model
    if MODEL == 'standard_rnn':
        model = StandardRNN(vocab_size, tag_size)
    elif MODEL == 'standard_lstm':
        model = StandardLSTM(vocab_size, tag_size)
    elif MODEL == 'bayesian_lstm':
        model = BayesianDropoutLSTM(vocab_size, tag_size)
    else:
        raise Exception(f'{MODEL} is not a valid model')

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.NLLLoss()
    
    # Training loop
    for e in tqdm(range(EPOCHS)):
        model.train()
        training_loss = 0
        for (sentences, labels) in (train_loader):
            y_hat = model(sentences).view(-1, tag_size)
            labels = labels.view(-1)
            loss = criterion(y_hat, labels)
            training_loss += loss
            loss.backward()
            optimizer.step()
        training_loss /= len(train_loader)

        model.eval()
        validation_loss = 0
        for (sentences, labels) in (validation_loader):
            y_hat = model(sentences).view(-1, tag_size)
            labels = labels.view(-1)
            loss = criterion(y_hat, labels)
            validation_loss += loss
        validation_loss /= len(validation_loader)

    # Test loop
    test_loss = 0

