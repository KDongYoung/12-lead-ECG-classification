from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch


class Loader:
    def __init__(self, dataset, batch_size, shuffle=False, valid_split=.2, seed=None, worker=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.shuffle = shuffle
        self.worker = worker

        if valid_split != 0:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(valid_split * dataset_size))
            if shuffle:
                if seed is None:
                    np.random.seed(4)
                else:
                    np.random.seed(seed)
                np.random.shuffle(indices)
            self.train_indices = indices[split:]
            self.val_indices = indices[:split]

    def fetch(self):
        if self.valid_split != 0:
            train_sampler = SubsetRandomSampler(self.train_indices)
            valid_sampler = SubsetRandomSampler(self.val_indices)

            train_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=self.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=self.worker)
            validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                            batch_size=self.batch_size * 2,
                                                            sampler=valid_sampler,
                                                            num_workers=self.worker)
            return train_loader, validation_loader
        else:
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                 shuffle=self.shuffle,
                                                 num_workers=self.worker)
            return loader


class WrappedDataLoader:
    def __init__(self, dl, x_dim, y_dim):
        self.dl = dl
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.preprocess(*b)

    def preprocess(self, x, y, id_):
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.view(-1, *self.x_dim).to(dev)
        y = y.to(dev)
        return x, y, id_
