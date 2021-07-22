import datetime
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

from segnet import SegNet
from utils import get_device, setup_logger, get_rank
from dataset import COCODataset

class Trainer(object) :
    def __init__(self,
                 data_path, data_type, parallel,
                 batch_size, lr, momentum, weight_decay, num_workers, num_classes,
                 log_dir, step):
        self.device = get_device()
        self.fix_seed()
        self.step = step

        train_dataset = COCODataset(split='train')
        test_dataset = COCODataset(split='val')

        self.iters_per_epoch = len(train_dataset) // batch_size

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        self.model = SegNet(num_classes=num_classes)

        if parallel : self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.logger = setup_logger("CLASSIFICATION", log_dir, get_rank(), filename='{}_log.txt'.format(data_type))
        self.history = self.get_history()

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (image, target) in enumerate(self.train_loader):
            image, target = image.to(self.device), target.to(self.device)

            # forward propagation
            output = self.model(image)
            loss = self.criterion(output, target)

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            self.history['train_iter_loss'].append(loss.item())

            if (batch_idx + 1) % self.step == 0:
                print("EPOCH {} | batch idx {} | Loss : {:0.6f}".format(epoch, batch_idx + 1, loss.item()))

        return running_loss / len(self.train_loader)

    def _test_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                image, target = image.to(self.device), target.to(self.device)

                output = self.model(image)

                loss = self.criterion(output, target)
                running_loss += loss.item()
                self.history['test_iter_loss'].append(loss.item())

                if (batch_idx + 1) % self.step == 0:
                    print("EPOCH {} | batch idx {} | Loss : {:0.6f}".format(epoch, batch_idx + 1, loss.item()))

        return running_loss / len(self.test_loader)

    def fit(self, epochs):
        max_iters = epochs * self.iters_per_epoch
        self.logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        for epoch in tqdm(range(1, epochs + 1)):
            start_time = time()

            print("TRAIN PHASE")
            train_loss = self._train_epoch(epoch)

            print("TEST PHASE")
            test_loss = self._test_epoch(epoch)


            epoch_time = time() - start_time
            eta_string = str(datetime.timedelta(seconds=int(epoch_time)))

            self.logger.info(
                "EPOCH {} | Train Loss : {:0.6f} | Test Loss : {:0.6f} | Time : {}".format(
                    epoch, train_loss, test_loss, str(datetime.timedelta(seconds=int(time() - start_time))), eta_string)
            )

        return self.model, self.history

    def get_history(self):
        history = dict()
        history['train_iter_loss'] = list()
        history['test_iter_loss'] = list()

        return history

    def save_history(self, history):
        train_history_df = pd.DataFrame(history['train_iter_loss'])
        test_history_df = pd.DataFrame(history['test_iter_loss'])

        train_history_df.to_csv(index=False)
        test_history_df.to_csv(index=False)

        import matplotlib.pyplot as plt
        plt.plot(train_history_df)
        plt.show()

    def save_model(self, model):
        torch.save(model.state_dict(), 'model.pth')

    def fix_seed(self):
        torch.manual_seed(4321)
        if self.device == 'cuda' :
            torch.cuda.manual_seed_all(4321)