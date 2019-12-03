from comet_ml import Experiment

import argparse
import os

import numpy as np
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

EXPERIMENT_HASH = experiment.get_key()


class BayesianDropoutLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 tag_size,
                 X_lengths,
                 embedding_dim,
                 hidden_size,
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
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 dropout=0):
        super(StandardLSTM, self).__init__()
        self.X_lengths = X_lengths
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
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
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 dropout=0):
        super(StandardRNN, self).__init__()
        self.X_lengths = X_lengths
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, dropout=dropout)
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
    parser.add_argument('-v', '--validation_split', default=0.2, type=float)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-m', '--model', default='standard_rnn', type=str)
    parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float)
    parser.add_argument('-c', '--use_cuda', default=False, type=bool)
    parser.add_argument('-em', '--embedding_size', default=16, type=int)
    parser.add_argument('-hs', '--hidden_size', default=64, type=int)
    parser.add_argument('-nl', '--number_of_layers', default=2, type=int)
    parser.add_argument('-d', '--dropout', default=0, type=float)
    args = parser.parse_args()

    VALIDATION_SPLIT = args.validation_split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = args.model
    LEARNING_RATE = args.learning_rate
    USE_CUDA = args.use_cuda
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_SIZE = args.hidden_size
    NUMBER_OF_LAYERS = args.number_of_layers
    DROPOUT = args.dropout

    model_parameters = {
        'validation_split': VALIDATION_SPLIT,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'model': MODEL,
        'learning_rate': LEARNING_RATE,
        'embedding_size': EMBEDDING_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'number_of_layers': NUMBER_OF_LAYERS,
        'dropout': DROPOUT
    }

    experiment.log_parameters(model_parameters)

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
        model = StandardRNN(
            vocab_size,
            tag_size,
            X_lengths,
            EMBEDDING_SIZE,
            HIDDEN_SIZE,
            NUMBER_OF_LAYERS,
            DROPOUT
        )
    elif MODEL == 'standard_lstm':
        model = StandardLSTM(
            vocab_size,
            tag_size,
            X_lengths,
            EMBEDDING_SIZE,
            HIDDEN_SIZE,
            NUMBER_OF_LAYERS,
            DROPOUT
        )
    elif MODEL == 'bayesian_lstm':
        model = BayesianDropoutLSTM(
            vocab_size,
            tag_size,
            X_lengths,
            EMBEDDING_SIZE,
            HIDDEN_SIZE,
            DROPOUT
        )
    else:
        raise Exception(f'{MODEL} is not a valid model')

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device=device)
    
    # Training loop
    for e in tqdm(range(EPOCHS)):
        with experiment.train():
            model.train()
            training_loss = 0
            for (sentences, labels) in (train_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                X_lengths = [len([i for i in s if i > 0]) for s in sentences]
                y_hat = model(sentences, X_lengths).view(-1, tag_size)
                labels = labels.view(-1)
                loss = criterion(y_hat, labels)
                training_loss += loss
                loss.backward()
                optimizer.step()
            training_loss /= len(train_loader)
            experiment.log_metric('loss', training_loss.detach().cpu().numpy(), step=e)

        with experiment.validate():
            model.eval()
            validation_loss = 0
            for (sentences, labels) in (validation_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                X_lengths = [len([i for i in s if i > 0]) for s in sentences]
                y_hat = model(sentences, X_lengths).view(-1, tag_size)
                labels = labels.view(-1)
                loss = criterion(y_hat, labels)
                validation_loss += loss
            validation_loss /= len(validation_loader)
            experiment.log_metric('loss', validation_loss.detach().cpu().numpy(), step=e)

    # Test loop
    model.eval()
    test_sentences = []
    test_labels = []
    test_predictions = []
    for i in test_indices:
        sentence, labels = dataset[i]
        sentence = sentence.to(device)
        X_lengths = [len([i for i in sentence if i > 0])]
        y_hat = model(torch.tensor(sentence).view(1, -1), torch.tensor(X_lengths)).view(-1, tag_size)
        y_hat_classes = torch.argmax(y_hat, dim=1).tolist()
        non_padded_label_length = len([l for l in labels if l > 0])
        test_sentences.append(sentence[:non_padded_label_length])
        test_labels.append(labels[:non_padded_label_length])
        test_predictions.append(y_hat_classes[:non_padded_label_length])

    # Create Classification report
    label_mapping = {v: k for k, v in dataset.label_mapping.items()}
    labels = [label_mapping[i] for i in range(tag_size)]

    classification_report = classification_report(np.hstack(test_labels), np.hstack(test_predictions), output_dict=True, labels=range(tag_size),target_names=labels)

    for k, v in classification_report.items():
        if k == 'accuracy':
            experiment.log_metric('accuracy', v)
        else:
            for metric, value in v.items():
                experiment.log_metric(f'{k}_{metric}', value)

    # Print example sentences
    vocab_mapping = {v: k for k,v in dataset.vocab_mapping.items()}
    for i in range(3):
        sentence = [vocab_mapping[t] for t in test_sentences[i]]
        prediction = [label_mapping[t] for t in test_predictions[i]]
        labels = [label_mapping[t] for t in test_labels[i]]
        print(f'Example {i+1}:\nSentence: {sentence}\nPrediction: {prediction}\nLabels: {labels}')

    # Save model
    torch.save(model.state_dict(), f'model_{EXPERIMENT_HASH}.pt')
    print(f'Saved model model_{EXPERIMENT_HASH}.pt')

    experiment.validate()
