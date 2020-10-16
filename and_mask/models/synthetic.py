import torch
from torch import nn as nn


def get_synthetic_model(args, device):
    assert args.n_hidden_layers >= 1
    hidden_layers = [torch.nn.Linear(2 + args.n_dims, args.n_hidden_units)]
    for _ in range(args.n_hidden_layers - 1):
        hidden_layers.append(torch.nn.Linear(args.n_hidden_units, args.n_hidden_units))
    classification_layer = torch.nn.Linear(args.n_hidden_units, 1)
    layers = []
    for linear_layer in hidden_layers:
        layers.append(linear_layer)
        if args.batch_norm:
            layers.append(nn.BatchNorm1d(args.n_hidden_units))
        layers.append(nn.LeakyReLU())
        if args.dropout_p > 0.0:
            layers.append(nn.Dropout(p=args.dropout_p))
    layers.append(classification_layer)
    model = torch.nn.Sequential(*layers).to(device)
    return model