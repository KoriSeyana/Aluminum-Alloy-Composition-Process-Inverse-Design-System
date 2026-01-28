import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import random
from sympy.core.random import rng


FILENAME = 'data.xlsx'
SHEET_NAME = 'Sheet6'

PARAMS = {
    'input_dim': 12,
    'output_dim': 3,
    'hidden_layers': [78, 78, 78],
    'learning_rate': 0.003445,
    'epochs': 2000,
    'batch_size': 32,
    'dropout_rate': 0.2,
    'weight_decay': 1e-4,
    'test_size': 0.3,
    'seed': int(time.time())
}


def set_seed():
    print(f"random seed: {PARAMS['seed']}")
    torch.manual_seed(PARAMS['seed'])
    np.random.seed(PARAMS['seed'])
    random.seed(PARAMS['seed'])


def load_and_process_data():
    if not os.path.exists(FILENAME):
        print(f"Error: The file '{FILENAME}' does not exist.")
        exit()

    try:
        df = pd.read_excel(FILENAME, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"An error occurred while reading the Excel file.: {e}")
        exit()

    X = df.iloc[:, :PARAMS['input_dim']].values
    y = df.iloc[:, PARAMS['input_dim']:].values


    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=PARAMS['test_size'], random_state=PARAMS['seed']
    )


    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_x, scaler_y


class ForwardDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate):
        super(ForwardDNN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(p=dropout_rate))

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x