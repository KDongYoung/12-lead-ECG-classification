import numpy as np
import torch

from metric import *


class Fit:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.min_loss = np.inf

        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(dev)

    def loss_batch(self, xb, yb, optimizer=None, lb=None):
        prediction = self.model(xb)

        loss = self.loss_fn(prediction, yb)
        loss = loss

        if optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        _prediction = prediction.data.cpu().numpy()
        _yb = yb.data.cpu().numpy()
        return loss.item(), len(xb), _prediction, _yb

    def fit(self, epochs, dl_train, dl_valid, classes, val_classes=None):
        for epoch in range(1, epochs + 1):
            # Training
            self.train_fn(dl_train, epoch, classes, mode='train')

        # Validation
        self.train_fn(dl_valid, epoch, classes, mode='eval')

        print(f'Training finished')
        return self.min_loss

    def train_fn(self, dl, epoch, classes, mode='train'):
        if mode == 'train':
            self.model.train()
            losses, nums, predictions, ybs = zip(
                *[self.loss_batch(xb, yb, optimizer=self.optimizer) for xb, yb, _ in dl]
            )
        elif mode == 'eval':
            self.model.eval()
            with torch.no_grad():
                losses, nums, predictions, ybs = zip(
                    *[self.loss_batch(xb, yb) for xb, yb, _ in dl]
                )
        loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # Converting torch tensor to numpy
        predictions, ybs = list(predictions), list(ybs)
        y_pred = np.array(predictions[:-1]).reshape([-1, len(classes)])
        y_pred = np.append(y_pred, predictions[-1].reshape(-1, len(classes)), axis=0)  # [None, classes]
        y_label = np.array(ybs[:-1]).reshape([-1, len(classes)])
        y_label = np.append(y_label, ybs[-1].reshape(-1, len(classes)), axis=0)

        metric = Metric(y_true=y_label, y_pred=y_pred, classes=classes)

        if mode == 'train':
            print(f'Epoch[{epoch}]: Training   loss={loss:.5f}')
        elif mode == 'eval':
            print(f'Epoch[{epoch}]: Validation loss={loss:.5f}\n')
            for cls in classes:
                print(f'{cls} / F1 score: {metric.score[cls]["f1_score"]:.3f}')
            print(f'F1 score: {metric.score["ovr"]["cpsc"]:.3f}')
