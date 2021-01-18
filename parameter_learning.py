import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def parameter_learning(model, x_train, y_train, epochs, threshold, patience, learning_rate, metrics,
                       loss_fn, opt_func=torch.optim.Adam, minibatch=False, bs=256):
    """
    Parameter learning for QBAF
    Training stops if maximum number of epochs is reached or if training loss decreases less than threshold
    for a specified number of epochs (patience)
    Batch learning is also possible but slow at the moment
    """
    optimizer = opt_func(model.parameters(), lr=learning_rate)
    model.training_loss = torch.autograd.Variable(torch.tensor(np.inf, dtype=torch.float))
    # threshold = 1e-3
    # patience = 20
    epoch_loss = 0.0
    best_score = None
    count = 0
    if minibatch:
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        for epoch in range(epochs):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_accuracy = 0.0
            for nb, (x_batch, y_batch) in enumerate(train_dl):
                y_pred_train = model(x_batch)
                loss = loss_fn(y_pred_train, y_batch)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_accuracy += metrics(y_pred_train, y_batch).item()
            train_loss = batch_loss / (nb+1)
            accuracy = batch_accuracy / (nb+1)
            # if training loss decreases less than threshold by defined number of epochs stop training
            if best_score is None:
                best_score = train_loss
            elif best_score - train_loss < threshold:
                count += 1
            else:
                best_score = train_loss
                count = 0
            if count >= patience:
                break
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred_train = model(x_train)
            loss = loss_fn(y_pred_train, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            accuracy = metrics(y_pred_train, y_train).item()
            # if training loss decreases less than threshold by defined number of epochs stop training
            if best_score is None:
                best_score = train_loss
            elif best_score - train_loss < threshold:
                count += 1
            else:
                best_score = train_loss
                count = 0
            if count >= patience:
                break

    model.accuracy = accuracy
    model.training_loss = train_loss
