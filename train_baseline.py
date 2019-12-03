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
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import ProielDataset

COMET_API_KEY = os.getenv('COMET_API_KEY')
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name='deep-latin-tagger',
    workspace='tylerkirby',
)


class BayesianDropoutLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 X_lengths,
                 embedding_dim=64,
                 hidden_size=64,
                 recurrent_dropout_probability=0):
        super(BayesianDropoutLSTM, self).__init__()
        self.X_lengths = X_lengths
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.augmented_lstm = AugmentedLstm(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            recurrent_dropout_probability=recurrent_dropout_probability
        )
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x, X_lengths):
        # Embed word tokens
        x = self.embedding_layer(x)
        # Pack tokens to hide padded tokens from model
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True, enforce_sorted=False)
        # Pass packed tokens to RNN
        x, _ = self.augmented_lstm(x)
        # Unpack tokens
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=237)
        # Reshape and pass data to fully connected layer
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return x


class StandardLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 X_lengths,
                 embedding_dim=16,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0,
                 bidirectional=False):
        super(StandardLSTM, self).__init__()
        self.X_lengths = X_lengths
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x, X_lengths):
        # Embed word tokens
        x = self.embedding_layer(x)
        # Pack tokens to hide padded tokens from model
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True, enforce_sorted=False)
        # Pass packed tokens to RNN
        x, _ = self.lstm(x)
        # Unpack tokens
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=237)
        # Reshape and pass data to fully connected layer
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return x


class StandardRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 X_lengths,
                 embedding_dim=16,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0,
                 bidirectional=False):
        super(StandardRNN, self).__init__()
        self.X_lengths = X_lengths
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x, X_lengths):
        # Embed word tokens
        x = self.embedding_layer(x)
        # Pack tokens to hide padded tokens from model
        x = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True, enforce_sorted=False)
        # Pass packed tokens to RNN
        x, _ = self.rnn(x)
        # Unpack tokens
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=237)
        # Reshape and pass data to fully connected layer
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Set random seed
    np.random.seed(1)

    # Experiment parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validation_split', default=0.2)
    parser.add_argument('-bs', '--batch_size', default=32)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-m', '--model', default='standard_rnn')
    parser.add_argument('-lr', '--learning_rate', default=0.00001)
    parser.add_argument('-c', '--use_cuda', default=False)
    args = parser.parse_args()

    VALIDATION_SPLIT = args.validation_split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = args.model
    LEARNING_RATE = args.learning_rate
    USE_CUDA = args.use_cuda

    # Set device
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Load data set and create vocabulary
    dataset = ProielDataset('proiel.json')
    dataset_size = len(dataset)
    vocab_size = len(dataset.vocab_mapping)
    tag_size = len(dataset.label_mapping)
    X_lengths = dataset.X_lengths

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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

    # Initialize model
    if MODEL == 'standard_rnn':
        model = StandardRNN(vocab_size, tag_size, X_lengths)
    elif MODEL == 'standard_lstm':
        model = StandardLSTM(vocab_size, tag_size, X_lengths)
    elif MODEL == 'bayesian_lstm':
        model = BayesianDropoutLSTM(vocab_size, tag_size, X_lengths)
    else:
        raise Exception(f'{MODEL} is not a valid model')

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device=device)
    
    # Training loop
    with experiment.train():
        for e in tqdm(range(EPOCHS)):
            model.train()
            training_loss = 0
            for (sentences, labels) in (train_loader):
                X_lengths = [len([i for i in s if i > 0]) for s in sentences]
                y_hat = model(sentences, X_lengths).view(-1, tag_size)
                labels = labels.view(-1)
                loss = criterion(y_hat, labels)
                training_loss += loss
                loss.backward()
                optimizer.step()
            training_loss /= len(train_loader)

            model.eval()
            validation_loss = 0
            for (sentences, labels) in (validation_loader):
                X_lengths = [len([i for i in s if i > 0]) for s in sentences]
                y_hat = model(sentences, X_lengths).view(-1, tag_size)
                labels = labels.view(-1)
                loss = criterion(y_hat, labels)
                validation_loss += loss
            validation_loss /= len(validation_loader)
            experiment.log_metric('train_loss', training_loss.detach().numpy(), step=e)
            experiment.log_metric('val_loss', validation_loss.detach().numpy(), step=e)

    # Test loop
    with experiment.test():
        model.eval()
        test_sentences = []
        test_labels = []
        test_predictions = []
        for i in test_indices:
            sentence, labels = dataset[i]
            X_lengths = [len([i for i in sentence if i > 0])]
            y_hat = model(torch.tensor(sentence).view(1, -1), torch.tensor(X_lengths)).view(-1, tag_size)
            y_hat_classes = torch.argmax(y_hat, dim=1).tolist()
            non_padded_label_length = len([l for l in labels if l > 0])
            test_sentences.append(sentence[:non_padded_label_length])
            test_labels.append(labels[:non_padded_label_length])
            test_predictions.append(y_hat_classes[:non_padded_label_length])
        test_sentences = np.hstack(test_sentences)
        test_labels = np.hstack(test_labels)
        test_predictions = np.hstack(test_predictions)

        print(classification_report(test_labels, test_predictions))

