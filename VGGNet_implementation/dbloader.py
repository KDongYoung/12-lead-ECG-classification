from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import ConcatDataset
import numpy as np
import torch
from collections import Counter

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

    def gan_noising(self, dataset):
        noise_dataset=[]
        for data in dataset:
            noise=np.random.uniform(-0.5,0.5,len(data[0][0]))
            for i in range(len(data[0])):
                data[0][i]=data[0][i]+noise*(data[0].std(1)[i].numpy())
            noise_dataset.append(data)
        self.dataset=ConcatDataset([dataset,noise_dataset])

    def fetch(self, phase):
        
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
            if phase=="train":
                self.gan_noising(self.dataset)
                weights=self.make_weights_for_balanced_classes(self.dataset)
                loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                        sampler=WeightedRandomSampler(weights,num_samples=self.batch_size*2, replacement=True),
                                                        num_workers=self.worker)
            elif phase=="valid":
                loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     num_workers=self.worker)

            # weights=self.make_weights_for_balanced_classes(self.dataset)
            # loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
            #                                         sampler=WeightedRandomSampler(weights,num_samples=self.batch_size*2, replacement=True),
            #                                         num_workers=self.worker)
            return loader
    
    ###################################################################################
    # get weights (use the number of samples) => balance samples
    def make_weights_for_balanced_classes(self,dataset):  # y값 class에 따라
        counts = Counter()
        classes = []
        for y in dataset:
            clas=[]
            for idx, y_segment in enumerate(y[1]):
                if int(y_segment)==1:
                    counts[idx] += 1 # count each class samples
                    clas.append(idx) 
            classes.append(clas)
            # y = int(y[1]) # class에 접근
            # counts[y] += 1 # count each class samples
            # classes.append(y) 
        n_classes = len(counts)

        weight_per_class = {}
        for y in counts: # the key of counts
            weight_per_class[y] = 1 / (counts[y] * n_classes)

        weights = torch.zeros(len(dataset))
        for i, y in enumerate(classes):
            total_weight=0.0
            for k in y:
                total_weight+=weight_per_class[k]
            weights[i] = total_weight/len(y)

        return weights


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
